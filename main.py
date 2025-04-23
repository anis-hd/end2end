# ==============================================================================
# Imports
# ==============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision import transforms # Keep for InterpolationMode potentially
# --- Added for AMP and tqdm ---
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
# --- Added for Logging ---
import logging
# --- Standard imports ---
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import random
import re
import sys
import traceback
import datetime # For timestamp in log

# --- Optional: torch.compile for speedup (PyTorch 2.0+) ---
# try:
#     import torch._dynamo
#     torch._dynamo.config.suppress_errors = True # Optional: Less verbose compile errors
#     TORCH_COMPILE_AVAILABLE = True
# except ImportError:
#     TORCH_COMPILE_AVAILABLE = False
TORCH_COMPILE_AVAILABLE = hasattr(torch, 'compile') # Simpler check for PyTorch 2.0+

# ==============================================================================
# Configuration Parameters (MODIFIED)
# ==============================================================================
class TrainConfig:
    # --- Data Configuration ---
    final_dir = "final/"
    flow_dir = "flow/"
    # *** MODIFIED Image Resolution ***
    resize_height = 360  # Must be divisible by 8
    resize_width = 640   # Must be divisible by 8
    # *** ADDED Data Augmentation Flag ***
    use_augmentation = True

    # --- Model Configuration ---
    restore_ckpt = None
    raft_iters = 12
    dropout = 0.1 # Often set to 0 for fine-tuning RAFT, adjust if needed
    mixed_precision = True

    # --- Training Configuration ---
    gpu = 0
    # *** INCREASED Epochs ***
    epochs = 600
    batch_size = 1 # Keep batch size 1 if using large images and grad accum
    accumulation_steps = 2 # Effective batch size = batch_size * accumulation_steps
    # *** INCREASED Learning Rate ***
    lr = 0.0001      # Increased LR (closer to TF example)
    wdecay = 0.0001
    clip = 1.0
    # *** Gamma for EPE Sequence Loss ***
    loss_gamma = 0.8
    num_workers = 4
    seed = 42

    # --- Checkpointing ---
    checkpoint_dir = './raft_checkpoints_amp_epe' # Changed dir name
    resume = True

    # --- Logging & Visualization ---
    log_file = 'training_log_epe.txt'           # Changed log file name
    vis_dir = './raft_visualizations_amp_epe'   # Changed vis dir name
    visualize_end_of_epoch = True
    visualize_while_training = False # Keep false unless debugging, can be slow
    vis_interval = 200


# ==============================================================================
# Logging Setup (Unchanged)
# ==============================================================================
def setup_logger(log_file_path):
    logger = logging.getLogger('TrainingLog')
    logger.setLevel(logging.INFO)
    # Prevent adding multiple handlers if called again
    if not logger.handlers:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        fh = logging.FileHandler(log_file_path, mode='a') # Append mode
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Optional: Add a handler for console output as well
        # sh = logging.StreamHandler()
        # sh.setLevel(logging.INFO)
        # sh.setFormatter(formatter)
        # logger.addHandler(sh)
    return logger

# ==============================================================================
# Helper Functions (Unchanged, except flow_to_color_np robust check)
# ==============================================================================
def read_flo_file(filename):
    try:
        with open(filename, 'rb') as f:
            magic = np.frombuffer(f.read(4), np.float32, count=1)
            if not np.isclose(magic[0], 202021.25):
                 print(f"Warning: Invalid magic number {magic} in {filename}")
                 return None # Indicate failure
            width = np.frombuffer(f.read(4), np.int32, count=1)[0]
            height = np.frombuffer(f.read(4), np.int32, count=1)[0]
            if width <= 0 or height <= 0 or width*height > 5000*5000: # Basic sanity check
                print(f"Warning: Invalid dimensions ({width}x{height}) in {filename}")
                return None
            data = np.frombuffer(f.read(), np.float32, count=-1) # Read rest of file
            expected_elements = height * width * 2
            if data.size != expected_elements:
                 print(f"Warning: Read {data.size} elements, expected {expected_elements} in {filename}")
                 # Try reshaping anyway if possible, otherwise return None
                 if data.size < expected_elements: return None
                 data = data[:expected_elements] # Truncate if too long

            return data.reshape((height, width, 2))
    except FileNotFoundError: print(f"Error: File not found {filename}"); return None
    except Exception as e: print(f"Error reading {filename}: {e}"); traceback.print_exc(); return None

def flow_to_color_np(flow_uv, clip_flow=None):
    # Input: HxWx2 numpy array
    if not isinstance(flow_uv, np.ndarray) or flow_uv.ndim != 3 or flow_uv.shape[2] != 2:
        print(f"Error: Invalid input shape {flow_uv.shape if isinstance(flow_uv, np.ndarray) else type(flow_uv)} for flow_to_color_np. Expected HxWx2.")
        # Return a black image of a default size or based on some known context if possible
        # For now, let's assume a fallback size or return None/raise error
        return np.zeros((256, 256, 3), dtype=np.uint8) # Example fallback

    h, w, _ = flow_uv.shape
    u, v = flow_uv[:, :, 0], flow_uv[:, :, 1]

    # Handle NaNs or Infs that might come from predictions
    u = np.nan_to_num(u)
    v = np.nan_to_num(v)

    magnitude = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u) # Output range: [-pi, pi]

    # Map angle to hue [0, 1]
    hue = (angle + np.pi) / (2 * np.pi) # Add pi to get [0, 2pi], then divide

    # Normalize magnitude for saturation/value
    if clip_flow is not None:
        magnitude = np.clip(magnitude, 0, clip_flow)

    # Robust max magnitude calculation (avoiding potential NaNs/Infs if not handled before)
    max_mag = np.max(magnitude[np.isfinite(magnitude)]) if np.any(np.isfinite(magnitude)) else 0.0

    saturation = magnitude / max_mag if max_mag > 1e-6 else np.zeros_like(magnitude)
    saturation = np.clip(saturation, 0, 1) # Ensure saturation is [0, 1]

    value = np.ones_like(magnitude) # Value is typically 1 for this visualization

    # HSV to RGB conversion
    hsv = np.stack([hue, saturation, value], axis=-1)
    try:
        from matplotlib.colors import hsv_to_rgb
        rgb = hsv_to_rgb(hsv) * 255.0
    except ImportError:
        print("Warning: matplotlib.colors not found. HSV conversion will be approximate.")
        # Basic manual conversion (less accurate than matplotlib's)
        h_i = (hue * 6).astype(int)
        f = hue * 6 - h_i
        p = value * (1 - saturation)
        q = value * (1 - f * saturation)
        t = value * (1 - (1 - f) * saturation)
        r, g, b = 0, 0, 0
        mask = (h_i == 0); r[mask], g[mask], b[mask] = value[mask], t[mask], p[mask]
        mask = (h_i == 1); r[mask], g[mask], b[mask] = q[mask], value[mask], p[mask]
        mask = (h_i == 2); r[mask], g[mask], b[mask] = p[mask], value[mask], t[mask]
        mask = (h_i == 3); r[mask], g[mask], b[mask] = p[mask], q[mask], value[mask]
        mask = (h_i == 4); r[mask], g[mask], b[mask] = t[mask], p[mask], value[mask]
        mask = (h_i == 5); r[mask], g[mask], b[mask] = value[mask], p[mask], q[mask]
        rgb = np.stack([r, g, b], axis=-1) * 255.0

    return rgb.astype(np.uint8)


def frame_id_from_path(path):
    # More robust regex to handle potential variations
    match = re.search(r'(?:frame_|img_|im_|image_)(\d+)\.(png|jpg|jpeg|flo|tif)', os.path.basename(path), re.IGNORECASE)
    return int(match.group(1)) if match else -1

# ==============================================================================
# RAFT Model Import (Unchanged)
# ==============================================================================
try:
    from core.raft import RAFT
    print("Using RAFT from local 'core' module")
    RAFT_NEEDS_ARGS = True # Assume local RAFT might need args object
except ImportError:
    try:
        # Attempt to import the newer interface first (if available)
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        print("Using RAFT large from torchvision.models.optical_flow")
        RAFT = raft_large
        RAFT_NEEDS_ARGS = False # torchvision models typically don't need the args object
    except ImportError:
         try:
             # Fallback to older torchvision location if necessary
             from torchvision.models.raft import raft_large
             print("Using RAFT large from torchvision.models.raft (older path)")
             RAFT = raft_large
             RAFT_NEEDS_ARGS = False
         except ImportError:
            print("\n" + "="*60 + "\nERROR: Could not import RAFT model from 'core' or 'torchvision'.\n" + "="*60 + "\n")
            print("Please ensure you have a RAFT implementation available.")
            print("1. Clone the official RAFT repo (https://github.com/princeton-vl/RAFT) and place its 'core' directory here.")
            print("2. Or install a recent version of torchvision (`pip install --upgrade torchvision`).")
            sys.exit(1)

# ==============================================================================
# Dataset Definition (MODIFIED for Augmentation and Normalization)
# ==============================================================================
class FlowDataset(Dataset):
    def __init__(self, config: TrainConfig): # Removed transform argument
        self.final_dir = config.final_dir
        self.flow_dir = config.flow_dir
        # self.transform = transform # Removed: normalization is implicit, augmentation done here
        self.resize_shape = (config.resize_height, config.resize_width)
        self.use_augmentation = config.use_augmentation # Store augmentation flag
        self.image_pairs = []
        self.indices_to_skip = set() # Keep track of problematic indices

        print(f"Scanning directories: final='{self.final_dir}', flow='{self.flow_dir}'")
        if not os.path.isdir(self.final_dir):
             print(f"Error: final_dir '{self.final_dir}' not found or not a directory.")
             return
        if not os.path.isdir(self.flow_dir):
             print(f"Error: flow_dir '{self.flow_dir}' not found or not a directory.")
             return

        sequence_folders = sorted([f for f in os.listdir(self.final_dir) if os.path.isdir(os.path.join(self.final_dir, f))])
        if not sequence_folders: print(f"Warning: No sequence subfolders found in {self.final_dir}")

        valid_pairs_count = 0
        missing_flow_dirs = 0
        unreadable_flows = 0
        skipped_frames = 0

        for seq in tqdm(sequence_folders, desc="Scanning Sequences"):
            final_seq_dir = os.path.join(self.final_dir, seq)
            flow_seq_dir = os.path.join(self.flow_dir, seq)

            if not os.path.isdir(flow_seq_dir):
                # print(f"Debug: Skipping sequence '{seq}' - Flow directory not found: {flow_seq_dir}")
                missing_flow_dirs += 1
                continue

            # Find all common image types
            final_frames = sorted(
                glob.glob(os.path.join(final_seq_dir, 'frame_*.png')) +
                glob.glob(os.path.join(final_seq_dir, 'frame_*.jpg')) +
                glob.glob(os.path.join(final_seq_dir, 'frame_*.jpeg')) +
                glob.glob(os.path.join(final_seq_dir, 'frame_*.tif')), # Add other types if needed
                key=lambda x: frame_id_from_path(x)
            )
            flow_maps = sorted(glob.glob(os.path.join(flow_seq_dir, 'frame_*.flo')), key=lambda x: frame_id_from_path(x))

            # Create a lookup for flow maps by frame ID for efficient access
            seq_flow_lookup = {frame_id_from_path(f): f for f in flow_maps if frame_id_from_path(f) != -1}

            for i in range(len(final_frames) - 1):
                img1_path = final_frames[i]
                img2_path = final_frames[i+1]

                frame1_id = frame_id_from_path(img1_path)
                frame2_id = frame_id_from_path(img2_path)

                # Check for consecutive frames and valid ID
                if frame1_id != -1 and frame2_id == frame1_id + 1:
                    # Check if corresponding flow map exists in our lookup
                    if frame1_id in seq_flow_lookup:
                        flow_path = seq_flow_lookup[frame1_id]
                        # Final check: ensure flow file exists and is not empty/corrupt (basic size check)
                        if os.path.exists(flow_path) and os.path.getsize(flow_path) > 12: # 12 bytes = magic + w + h
                            self.image_pairs.append((img1_path, img2_path, flow_path))
                            valid_pairs_count += 1
                        elif os.path.exists(flow_path):
                             # print(f"Debug: Skipping pair for frame {frame1_id} - Flow file too small: {flow_path}")
                             unreadable_flows += 1
                        # else: Flow path derived from lookup, so it should exist if in lookup
                    # else: print(f"Debug: Skipping pair for frame {frame1_id} - Flow map not found in lookup.")
                else:
                    # print(f"Debug: Skipping non-consecutive frames: {os.path.basename(img1_path)} ({frame1_id}) -> {os.path.basename(img2_path)} ({frame2_id})")
                    skipped_frames += 1


        print(f"Found {len(sequence_folders)} potential sequence folders.")
        if missing_flow_dirs > 0: print(f"Warning: Skipped {missing_flow_dirs} sequences due to missing flow dir.")
        if unreadable_flows > 0: print(f"Warning: Skipped {unreadable_flows} pairs due to potentially corrupt/empty flow files.")
        if skipped_frames > 0: print(f"Info: Skipped {skipped_frames} non-consecutive frame pairs.")
        print(f"Found {valid_pairs_count} valid frame pairs with flow maps.")
        if valid_pairs_count == 0 and len(sequence_folders) > 0:
            print("---------------------------------------------------------")
            print("WARNING: No valid training pairs found. Possible issues:")
            print(f"  - Check `final_dir` ({self.final_dir}) and `flow_dir` ({self.flow_dir}) paths.")
            print("  - Ensure subdirectories exist within these paths.")
            print("  - Verify frame naming convention (e.g., 'frame_0001.png', 'frame_0001.flo').")
            print("  - Check if flow files (.flo) correspond correctly to frame pairs.")
            print("  - Ensure flow files are not corrupted or empty.")
            print("---------------------------------------------------------")


    def __len__(self):
        return len(self.image_pairs) - len(self.indices_to_skip) # Return valid length

    def __getitem__(self, idx):
        # Map public index to internal list index, skipping bad ones
        actual_idx = idx
        skipped_count = 0
        while actual_idx in self.indices_to_skip:
            actual_idx += 1
            skipped_count += 1
            if actual_idx >= len(self.image_pairs):
                 raise IndexError(f"Index {idx} is out of bounds after skipping {skipped_count} items.")
        # Adjust index based on how many were skipped *before* the current valid index
        # This requires iterating or maintaining a map, simpler to just find the next valid one.
        # The above loop finds the *next* valid index, need to map idx -> valid_idx.
        # Let's retry the skipping logic for simplicity for now. User should filter dataset ideally.
        # A better approach is to filter self.image_pairs *once* at init or lazily.

        # --- Simplified Skipping (may lead to uneven data access if many skips) ---
        internal_idx = idx
        retries = 0
        max_retries = len(self.image_pairs) # Prevent infinite loop
        while internal_idx in self.indices_to_skip and retries < max_retries:
            internal_idx = (internal_idx + 1) % len(self.image_pairs) # Wrap around for robustness? Or just fail?
            retries += 1
        if retries == max_retries and len(self.image_pairs) > 0:
            raise RuntimeError(f"Could not find a valid item after checking all indices starting from {idx}.")
        elif len(self.image_pairs) == 0:
            raise IndexError("Dataset is empty.")
        # ------------------------------------------------------------------------

        img1_path, img2_path, flow_path = self.image_pairs[internal_idx]

        try:
            # --- Load Data ---
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
            flow_gt = read_flo_file(flow_path)

            if flow_gt is None:
                # print(f"Warning: read_flo_file failed for {flow_path} at index {internal_idx}. Skipping.")
                self.indices_to_skip.add(internal_idx)
                # Retry getting the next item recursively or raise error
                # For simplicity, let's try getting the next index
                return self.__getitem__((idx + 1) % len(self)) # CAUTION: recursive, potential stack overflow if many errors

            # --- Convert to Tensor (Implicitly scales images to [0, 1]) ---
            img1_tensor = TF.to_tensor(img1)
            img2_tensor = TF.to_tensor(img2)
            # Permute flow: HxWx2 (numpy) -> CxHxW (torch)
            flow_gt_tensor = torch.from_numpy(flow_gt.astype(np.float32)).permute(2, 0, 1)

            # --- Resize ---
            target_h, target_w = self.resize_shape
            # Basic check for valid resize dimensions
            if target_h <= 0 or target_w <= 0:
                 raise ValueError(f"Invalid resize dimensions: {self.resize_shape}")
            if target_h % 8 != 0 or target_w % 8 != 0:
                 # Log this warning once during init if possible
                 # print(f"Warning: Resize dimensions ({target_h}x{target_w}) not divisible by 8. RAFT may require this.")
                 pass # Allow for now, but RAFT might fail later

            original_h, original_w = img1_tensor.shape[1:]

            img1_tensor = TF.resize(img1_tensor, [target_h, target_w], antialias=True)
            img2_tensor = TF.resize(img2_tensor, [target_h, target_w], antialias=True)

            # Resize and Scale Flow
            # Add batch dim for resize, then remove
            flow_gt_tensor = flow_gt_tensor.unsqueeze(0)
            # Use antialias=False for flow resizing as recommended by some sources
            flow_gt_resized = TF.resize(flow_gt_tensor, [target_h, target_w],
                                        interpolation=transforms.InterpolationMode.BILINEAR,
                                        antialias=False)

            # Calculate scaling factors (handle potential division by zero)
            scale_w = float(target_w) / float(original_w) if original_w > 0 else 1.0
            scale_h = float(target_h) / float(original_h) if original_h > 0 else 1.0

            # Apply scaling: flow_gt_resized is (1, 2, H, W)
            flow_gt_resized[:, 0, :, :] *= scale_w # Scale u component
            flow_gt_resized[:, 1, :, :] *= scale_h # Scale v component

            flow_gt_tensor = flow_gt_resized.squeeze(0) # Remove batch dim -> (2, H, W)

            # --- Augmentation (Random Horizontal Flip) ---
            if self.use_augmentation and random.random() > 0.5:
                img1_tensor = TF.hflip(img1_tensor)
                img2_tensor = TF.hflip(img2_tensor)
                flow_gt_tensor = TF.hflip(flow_gt_tensor)
                # **Crucially, negate the horizontal flow component (u)**
                flow_gt_tensor[0, :, :] *= -1

            # --- Normalization ---
            # No explicit normalization here if model expects [0, 1] inputs.
            # If model expects [-1, 1], apply normalization *after* augmentation:
            # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # img1_tensor = normalize(img1_tensor)
            # img2_tensor = normalize(img2_tensor)
            # Current setup assumes model works with [0, 1] like TF example.

            return img1_tensor, img2_tensor, flow_gt_tensor

        except Exception as e:
             print(f"\nERROR in dataset __getitem__ for index {idx} (internal {internal_idx}): {e}")
             print(f" Files: {img1_path}, {img2_path}, {flow_path}")
             traceback.print_exc()
             self.indices_to_skip.add(internal_idx)
             # Try getting the next item
             return self.__getitem__((idx + 1) % len(self)) # CAUTION: recursive

# ==============================================================================
# Evaluation Metric: End-Point Error (EPE) (Unchanged - Should work)
# ==============================================================================
def calculate_epe(flow_pred: torch.Tensor, flow_gt: torch.Tensor):
    """Calculates average End-Point Error (EPE). Upsamples pred if needed."""
    # Ensure GT is float32
    flow_gt = flow_gt.float()
    flow_pred = flow_pred.float() # Ensure pred is float

    # Upsample prediction to match GT resolution if necessary
    if flow_pred.shape[-2:] != flow_gt.shape[-2:]:
         # Use antialias=False for flow upsampling? Common practice.
         flow_pred = TF.resize(flow_pred, flow_gt.shape[-2:], # Target size (H, W)
                              interpolation=transforms.InterpolationMode.BILINEAR,
                              antialias=False) # Or True? Test if matters. Default is True now.

    # Calculate EPE per pixel: sqrt((u1-u2)^2 + (v1-v2)^2)
    epe_map = torch.sqrt(torch.sum((flow_pred - flow_gt)**2, dim=1)) # Sum over channel dim (C=2)
    # Average EPE over batch and spatial dimensions
    return epe_map.mean().item()

# ==============================================================================
# Loss Function (MODIFIED to EPE Sequence Loss)
# ==============================================================================
def epe_sequence_loss(flow_preds: list[torch.Tensor], flow_gt: torch.Tensor, gamma: float):
    """
    Calculates the weighted sum of average EPEs for RAFT predictions.
    Resizes GT flow *down* to match prediction size for each level.

    Args:
        flow_preds: List of flow predictions from RAFT (B, 2, H_i, W_i). Usually low-res.
        flow_gt: Ground truth flow (B, 2, H_gt, W_gt). Usually high-res.
        gamma: Weight decay factor.

    Returns:
        total_loss: Weighted EPE loss (scalar tensor).
        final_flow_upsampled: The last flow prediction, upsampled to GT resolution (B, 2, H_gt, W_gt).
    """
    n_predictions = len(flow_preds)
    total_loss = 0.0

    # Ensure GT is on the same device and float type as predictions
    flow_gt_device = flow_gt.to(flow_preds[0].device, dtype=torch.float32) # Use float32 for loss calc

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        flow_pred = flow_preds[i].float() # Ensure prediction is float

        pred_h, pred_w = flow_pred.shape[2:]
        gt_h, gt_w = flow_gt_device.shape[2:]

        # Resize GT *down* to prediction size and scale values
        if pred_h != gt_h or pred_w != gt_w:
            # Add batch dim, resize, remove batch dim
            flow_gt_resized = flow_gt_device.unsqueeze(0)
            flow_gt_resized = TF.resize(flow_gt_resized, [pred_h, pred_w],
                                        interpolation=transforms.InterpolationMode.BILINEAR,
                                        antialias=False) # No AA for downscaling flow typically

            # Calculate scaling factors for downscaling
            scale_w = float(pred_w) / float(gt_w) if gt_w > 0 else 1.0
            scale_h = float(pred_h) / float(gt_h) if gt_h > 0 else 1.0

            flow_gt_resized[:, 0, :, :] *= scale_w # Scale u
            flow_gt_resized[:, 1, :, :] *= scale_h # Scale v
            flow_gt_resized = flow_gt_resized.squeeze(0) # Remove batch dim
        else:
            flow_gt_resized = flow_gt_device # No resize needed

        # Calculate EPE for this level
        epe_map = torch.sqrt(torch.sum((flow_pred - flow_gt_resized)**2, dim=1) + 1e-8) # Add epsilon for numerical stability
        i_epe = epe_map.mean() # Average EPE for this prediction level
        total_loss += i_weight * i_epe

    # Upsample the *final* prediction to the original GT resolution for evaluation/visualization
    final_flow_pred = flow_preds[-1].float()
    gt_h_final, gt_w_final = flow_gt.shape[2:] # Use original GT shape

    if final_flow_pred.shape[-2:] != (gt_h_final, gt_w_final):
         # Add batch dim, resize, remove batch dim
         final_flow_upsampled = final_flow_pred.unsqueeze(0)
         final_flow_upsampled = TF.resize(final_flow_upsampled, [gt_h_final, gt_w_final],
                                           interpolation=transforms.InterpolationMode.BILINEAR,
                                           antialias=False) # No AA for flow? Test this.

         # Calculate scaling factors for upsampling
         final_pred_h, final_pred_w = final_flow_pred.shape[2:]
         scale_w_up = float(gt_w_final) / float(final_pred_w) if final_pred_w > 0 else 1.0
         scale_h_up = float(gt_h_final) / float(final_pred_h) if final_pred_h > 0 else 1.0

         final_flow_upsampled[:, 0, :, :] *= scale_w_up
         final_flow_upsampled[:, 1, :, :] *= scale_h_up
         final_flow_upsampled = final_flow_upsampled.squeeze(0)
    else:
         final_flow_upsampled = final_flow_pred # Already at correct resolution

    return total_loss, final_flow_upsampled.detach() # Detach upsampled flow, loss computation is done


# ==============================================================================
# Visualization Function (MODIFIED for [0, 1] image range)
# ==============================================================================
def visualize_batch(img1, img2, pred_flow, epoch, batch_idx_str, save_dir):
    """Visualizes the first sample in a batch."""
    os.makedirs(save_dir, exist_ok=True)

    # Ensure tensors are detached, on CPU, and converted to numpy
    # Images are expected to be in [0, 1] range now
    try:
        img1_np = img1[0].float().cpu().detach().permute(1, 2, 0).numpy()
        img2_np = img2[0].float().cpu().detach().permute(1, 2, 0).numpy()
        pred_flow_np = pred_flow[0].float().cpu().detach().permute(1, 2, 0).numpy()

        # Clip images just in case, though they should be [0, 1] from ToTensor
        img1_np = img1_np.clip(0, 1)
        img2_np = img2_np.clip(0, 1)

        # Convert predicted flow to color
        pred_flow_color = flow_to_color_np(pred_flow_np)

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Epoch {epoch}, Sample: {batch_idx_str}', fontsize=16)

        axs[0].imshow(img1_np)
        axs[0].set_title('Input Frame 1')
        axs[0].axis('off')

        axs[1].imshow(img2_np)
        axs[1].set_title('Input Frame 2')
        axs[1].axis('off')

        axs[2].imshow(pred_flow_color)
        axs[2].set_title('Predicted Flow')
        axs[2].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

        # Sanitize filename
        safe_batch_idx_str = re.sub(r'[\\/*?:"<>|]', '_', str(batch_idx_str))
        save_path = os.path.join(save_dir, f'epoch_{epoch:04d}_sample_{safe_batch_idx_str}.png')

        plt.savefig(save_path)
        plt.close(fig) # Close the figure to free memory

    except Exception as e:
        print(f"Warning: Could not save/create visualization for epoch {epoch}, sample {batch_idx_str}: {e}")
        traceback.print_exc()
        plt.close('all') # Close any potentially lingering figures

# ==============================================================================
# Training Function (MODIFIED for EPE Loss and other changes)
# ==============================================================================
def train(config: TrainConfig, logger: logging.Logger):
    # --- Device Setup (Unchanged) ---
    use_amp = config.mixed_precision and torch.cuda.is_available() and config.gpu is not None
    if config.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{config.gpu}")
        torch.cuda.set_device(device)
        print(f"Using device: {device} (CUDA)")
        logger.info(f"Using device: {device} (CUDA)")
        if use_amp: print("Automatic Mixed Precision (AMP) Enabled."); logger.info("AMP Enabled.")
        else: print("AMP Disabled. Using FP32."); logger.info("AMP Disabled. Using FP32.")
    else:
        device = torch.device("cpu")
        use_amp = False # Ensure AMP is off if on CPU
        print(f"Using device: {device} (CPU). AMP Disabled."); logger.info(f"Using device: {device} (CPU). AMP Disabled.")

    # --- Reproducibility (Unchanged) ---
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed) # for multi-GPU
            # Potential performance trade-off for full determinism
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
        logger.info(f"Set random seed to {config.seed}")
    else:
        # Ensure benchmark is enabled if no seed is set for potential speedup
        if device.type == 'cuda':
             torch.backends.cudnn.benchmark = True

    # --- Data Loading (MODIFIED: No external transform needed) ---
    # Images will be [0, 1] from ToTensor in Dataset, augmentation handled there.
    # img_transforms = None # No longer needed here
    try:
        train_dataset = FlowDataset(config=config) # Pass config for augmentation flag etc.
        if len(train_dataset) == 0:
             print("\nERROR: Training dataset is empty. Please check configuration and data paths.")
             logger.error("Training dataset initialization resulted in 0 valid samples.")
             return # Exit gracefully

        # Filter out skipped indices permanently for DataLoader if feasible
        # This is complex if __getitem__ can dynamically add skips.
        # A safer approach is robust error handling in the training loop.

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=(device.type == 'cuda'), # Enable pin_memory on GPU
                                  drop_last=True, # Important if using accumulation/fixed steps
                                  persistent_workers=(config.num_workers > 0) # Can speed up epoch start
                                  )
        logger.info(f"DataLoader initialized with {len(train_dataset)} potential samples.")
        print(f"DataLoader uses {config.num_workers} workers, batch size {config.batch_size}, accumulation {config.accumulation_steps}.")
    except Exception as e:
        print(f"\nFATAL ERROR during Dataset/DataLoader initialization: {e}")
        logger.exception("Dataset or DataLoader initialization failed")
        traceback.print_exc()
        return

    # --- Model Setup (MODIFIED: Optional torch.compile) ---
    if RAFT_NEEDS_ARGS:
        # Create a simple args object mimicking the original RAFT repo's structure if needed
        model_args = type('raft_args', (object,), {
            'small': False, # Assuming large model
            'dropout': config.dropout,
            'mixed_precision': use_amp # Pass AMP flag if the model uses it internally
            # Add other args the specific 'core.raft' implementation might need
            })()
        model = RAFT(model_args)
        print("Initialized RAFT from 'core' using args object.")
    else:
        # Try initializing torchvision RAFT with default weights (None means random init)
        try:
            model = RAFT(weights=None) # Random initialization for training from scratch
            print("Initialized RAFT from torchvision (random weights).")
            # If you wanted pretrained weights:
            # weights = Raft_Large_Weights.DEFAULT # Or specific weights like .C_T_V2
            # model = RAFT(weights=weights)
            # print(f"Initialized RAFT from torchvision with weights: {weights}")
        except Exception as e:
             print(f"Error initializing torchvision RAFT: {e}. Check import/torchvision version.")
             logger.error(f"Torchvision RAFT initialization failed: {e}")
             return


    if config.restore_ckpt and os.path.isfile(config.restore_ckpt):
         logger.info(f"Attempting to load *EXTERNAL* pretrained weights from: {config.restore_ckpt}")
         print(f"Loading *EXTERNAL* pretrained weights from: {config.restore_ckpt}")
         # This is for loading weights BEFORE starting training or resuming internal checkpoints
         try:
             checkpoint_data = torch.load(config.restore_ckpt, map_location='cpu') # Load to CPU first
             # Flexible state_dict extraction
             if isinstance(checkpoint_data, dict):
                if 'model_state_dict' in checkpoint_data: state_dict = checkpoint_data['model_state_dict']
                elif 'state_dict' in checkpoint_data: state_dict = checkpoint_data['state_dict']
                elif 'model' in checkpoint_data: state_dict = checkpoint_data['model']
                else: state_dict = checkpoint_data # Assume the dict itself is the state_dict
             else:
                 state_dict = checkpoint_data # Assume the loaded object *is* the state_dict

             # Handle 'module.' prefix if saved from DataParallel/DDP
             if all(k.startswith('module.') for k in state_dict.keys()):
                 print("Removing 'module.' prefix from state_dict keys.")
                 state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

             load_result = model.load_state_dict(state_dict, strict=False) # Use strict=False initially
             print(f"External weights load result: {load_result}")
             logger.info(f"External weights loaded. Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}")
             if load_result.missing_keys and not load_result.unexpected_keys:
                  print("Info: Some weights were missing (expected if fine-tuning parts of the model).")
             elif load_result.unexpected_keys:
                  print("Warning: Some weights in the checkpoint were not found in the model.")

         except Exception as e:
             print(f"ERROR loading external pretrained weights: {e}");
             logger.error(f"Failed loading external pretrained weights from {config.restore_ckpt}: {e}")
             # Decide whether to proceed with random weights or exit
             # proceed = input("Proceed with random weights? (y/n): ")
             # if proceed.lower() != 'y': return
    elif config.restore_ckpt:
        print(f"WARNING: External pretrained checkpoint not found: {config.restore_ckpt}")
        logger.warning(f"External pretrained checkpoint specified but not found: {config.restore_ckpt}")

    model.to(device)

    # --- Optional: Apply torch.compile (after loading weights and moving to device) ---
    if TORCH_COMPILE_AVAILABLE and device.type == 'cuda': # Compile usually best on CUDA
        compile_mode = "reduce-overhead" # Or "default", "max-autotune"
        try:
            print(f"Attempting to compile model with torch.compile (mode: {compile_mode})...")
            # Log before compiling
            logger.info(f"Attempting torch.compile with mode={compile_mode}")
            start_compile = time.time()
            # model = torch.compile(model, mode=compile_mode) # Full graph breaks RAFT sometimes
            # Compile specific sub-modules if full compile fails
            if hasattr(model, 'feature_encoder'): model.feature_encoder = torch.compile(model.feature_encoder, mode=compile_mode)
            if hasattr(model, 'context_encoder'): model.context_encoder = torch.compile(model.context_encoder, mode=compile_mode)
            if hasattr(model, 'update_block'): model.update_block = torch.compile(model.update_block, mode=compile_mode)

            compile_time = time.time() - start_compile
            print(f"Model compilation finished in {compile_time:.2f}s.")
            logger.info(f"torch.compile applied successfully in {compile_time:.2f}s.")
        except Exception as e:
            print(f"WARNING: torch.compile failed: {e}. Continuing without compilation.")
            logger.warning(f"torch.compile failed: {e}. Performance may not be optimal.")
            # Ensure model remains the original uncompiled version
            # (it should be if compilation failed mid-way)
    elif TORCH_COMPILE_AVAILABLE and device.type != 'cuda':
        print("Skipping torch.compile as device is not CUDA.")
        logger.info("torch.compile available but skipped (device is not CUDA).")
    else:
        print("torch.compile not available (requires PyTorch 2.0+).")
        logger.info("torch.compile not available.")

    # --- Optional: Channels Last (Can improve GPU performance) ---
    # if device.type == 'cuda':
    #     print("Converting model and inputs to channels_last memory format.")
    #     logger.info("Using channels_last memory format.")
    #     model = model.to(memory_format=torch.channels_last)
    #     # Input tensors will need conversion in the loop: img1.to(memory_format=torch.channels_last)

    # --- Optimizer, GradScaler (NO SCHEDULER) ---
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wdecay, eps=1e-8)
    # NO SCHEDULER
    scaler = GradScaler(enabled=use_amp)
    logger.info(f"Optimizer: AdamW, Fixed Learning Rate: {config.lr}, Weight Decay: {config.wdecay}")
    print(f"Using fixed learning rate: {config.lr}")

    # --- Checkpoint Loading for Resuming Training State ---
    start_epoch = 0
    checkpoint_dir = config.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    latest_checkpoint_path = None
    if config.resume:
        # Find the checkpoint with the highest epoch number
        checkpoints = glob.glob(os.path.join(checkpoint_dir, 'raft_epoch_*.pth'))
        if checkpoints:
            checkpoints.sort(key=lambda x: int(re.search(r'raft_epoch_(\d+).pth', x).group(1)))
            latest_checkpoint_path = checkpoints[-1]
            print(f"Found latest checkpoint: {latest_checkpoint_path}")
        else:
            print(f"Resume requested, but no checkpoints found in {checkpoint_dir}.")
            logger.info(f"Resume=True, but no checkpoints found in {checkpoint_dir}")

    if latest_checkpoint_path:
        logger.info(f"Attempting to resume training from: {latest_checkpoint_path}")
        print(f"Attempting to resume training from: {latest_checkpoint_path}")
        try:
            checkpoint = torch.load(latest_checkpoint_path, map_location=device) # Load directly to target device

            # --- Load Model State ---
            # Handle potential 'module.' prefix from DDP/DP saving
            model_state = checkpoint['model_state_dict']
            if all(k.startswith('module.') for k in model_state.keys()):
                print("Detected 'module.' prefix in checkpoint model state. Removing.")
                model_state = {k.replace('module.', '', 1): v for k, v in model_state.items()}

            # Load model state (allow non-strict loading if necessary)
            load_result = model.load_state_dict(model_state, strict=True) # Try strict first
            print(f"Resumed Model state loaded. Result: {load_result}")


            # --- Load Optimizer State ---
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Loaded Optimizer state.")
                # --- !!! FORCE FIXED LEARNING RATE !!! ---
                print(f"Applying fixed learning rate from config ({config.lr}) after resuming.")
                logger.info(f"Overriding resumed optimizer LR with fixed config LR: {config.lr}")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config.lr
                # -----------------------------------------
            except Exception as optim_e:
                 print(f"Warning: Could not load optimizer state: {optim_e}. Optimizer reinitialized.")
                 logger.warning(f"Optimizer state loading failed: {optim_e}. Optimizer state may be lost.")
                 # Ensure LR is set even if optim state load failed
                 for param_group in optimizer.param_groups: param_group['lr'] = config.lr


            # --- Load Scaler State (if using AMP) ---
            if use_amp and 'scaler_state_dict' in checkpoint:
                try:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    print("Loaded GradScaler state.")
                except Exception as scaler_e:
                    print(f"Warning: Could not load GradScaler state: {scaler_e}. Scaler reinitialized.")
                    logger.warning(f"GradScaler state loading failed: {scaler_e}. Scaler state may be lost.")
            elif use_amp:
                print("Warning: Resuming AMP training, but no scaler state found in checkpoint.")
                logger.warning("AMP enabled, but scaler_state_dict not found in checkpoint.")

            # --- Load Epoch ---
            # Use .get for resilience if keys are missing
            start_epoch = checkpoint.get('epoch', -1) + 1 # Start from the next epoch
            last_loss = checkpoint.get('loss', 'N/A')
            last_epe = checkpoint.get('epe', 'N/A') # Get EPE if saved previously
            print(f"Resuming from epoch {start_epoch}. Last saved Avg Loss: {last_loss}, Last Avg EPE: {last_epe}")
            logger.info(f"Successfully resumed. Starting epoch {start_epoch}. Last AvgLoss: {last_loss}, Last AvgEPE: {last_epe}")

        except Exception as e:
            print(f"ERROR loading checkpoint '{latest_checkpoint_path}': {e}.")
            logger.exception(f"Checkpoint loading failed from {latest_checkpoint_path}")
            print("Starting training from scratch (or using external pretrained weights if provided).")
            start_epoch = 0
            # Ensure LR is correctly set if checkpoint loading failed
            print(f"Ensuring fixed learning rate is set: {config.lr}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.lr


    # --- Training Loop ---
    print(f"\n--- Starting Training (Epochs {start_epoch + 1} to {config.epochs}) ---")
    logger.info(f"Starting training loop from epoch {start_epoch + 1} up to {config.epochs}")
    total_steps = 0
    effective_batch_size = config.batch_size * config.accumulation_steps
    logger.info(f"Effective batch size: {effective_batch_size} (Batch: {config.batch_size}, Accum: {config.accumulation_steps})")

    # Initialize optimizer grad explicitly before loop
    optimizer.zero_grad(set_to_none=True) # More efficient potentially

    for epoch in range(start_epoch, config.epochs):
        model.train() # Set model to training mode
        epoch_loss_total = 0.0
        epoch_epe_total = 0.0
        steps_in_epoch = 0
        actual_optimizer_steps = 0 # Track actual optimizer steps per epoch
        epoch_start_time = time.time()
        last_batch_data_for_viz = None

        # Use len(train_loader) for tqdm total if dataset size is known
        try:
             loader_len = len(train_loader)
        except TypeError: # Happens if dataset has no __len__ (IterableDataset)
             loader_len = None
        step_iterator = tqdm(enumerate(train_loader), total=loader_len, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False)

        # Get the current fixed LR for display (should not change)
        current_lr = optimizer.param_groups[0]['lr']

        for i, batch_data in step_iterator:
            try:
                img1, img2, flow_gt = batch_data
                # Move data to device
                img1 = img1.to(device, non_blocking=True) # Use non_blocking if workers > 0 and pin_memory=True
                img2 = img2.to(device, non_blocking=True)
                flow_gt = flow_gt.to(device, non_blocking=True)

                # --- Optional: Channels Last for Input ---
                # if device.type == 'cuda':
                #     img1 = img1.to(memory_format=torch.channels_last)
                #     img2 = img2.to(memory_format=torch.channels_last)

                # --- Forward pass with AMP context ---
                with autocast(enabled=use_amp):
                    # Check how the imported RAFT model expects to be called
                    if RAFT_NEEDS_ARGS or hasattr(model, 'iters'): # Heuristic check
                        flow_predictions = model(img1, img2, iters=config.raft_iters)
                    else:
                         # Torchvision RAFT doesn't take iters in forward
                         flow_predictions = model(img1, img2)

                    # Ensure output is a list (torchvision RAFT returns list)
                    if not isinstance(flow_predictions, list):
                         if torch.is_tensor(flow_predictions):
                             flow_predictions = [flow_predictions] # Wrap single tensor output
                         else:
                              logger.error(f"Epoch {epoch+1} Step {i}: Model output type unexpected: {type(flow_predictions)}")
                              raise TypeError(f"Model output expected list or tensor, got {type(flow_predictions)}")

                    # Calculate EPE loss
                    loss, final_flow_upsampled = epe_sequence_loss(
                        flow_predictions, flow_gt, gamma=config.loss_gamma
                    )

                    # Normalize loss for accumulation
                    loss_scaled_for_accum = loss / config.accumulation_steps

                # --- EPE Metric Calculation (on final upsampled flow) ---
                # No need for torch.no_grad here as final_flow_upsampled is detached in loss func
                epe = calculate_epe(final_flow_upsampled, flow_gt) # Use the upsampled flow

                # --- Backward pass ---
                # Scales the loss and calls backward()
                scaler.scale(loss_scaled_for_accum).backward()

                # --- Accumulate Metrics ---
                epoch_loss_total += loss.item() # Accumulate the *original* loss value
                epoch_epe_total += epe
                steps_in_epoch += 1
                total_steps += 1 # Global step counter

                # --- Optimizer Step ---
                if (i + 1) % config.accumulation_steps == 0:
                    actual_optimizer_steps += 1
                    # Unscale gradients before clipping
                    scaler.unscale_(optimizer)

                    # Clip gradients
                    if config.clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip)

                    # Optimizer step (updates weights)
                    scaler.step(optimizer)

                    # Update the scale for next iteration
                    scaler.update()

                    # Zero gradients for the next accumulation cycle
                    optimizer.zero_grad(set_to_none=True)

                # --- Update step progress bar ---
                step_iterator.set_postfix(Loss=f"{loss.item():.4f}", EPE=f"{epe:.3f}", LR=f"{current_lr:.6f}")

                # --- Visualization ---
                if config.visualize_while_training and (total_steps % config.vis_interval == 0):
                    with torch.no_grad():
                         visualize_batch(img1, img2, final_flow_upsampled, epoch + 1, f"step_{total_steps}", config.vis_dir)
                # Store last batch for end-of-epoch viz
                if i == loader_len - 1 if loader_len else False: # Check if it's the last batch
                     last_batch_data_for_viz = (img1.detach(), img2.detach(), final_flow_upsampled.detach())

            # --- Handle potential errors within a batch ---
            except StopIteration: # Raised by tqdm sometimes?
                logger.warning(f"Epoch {epoch+1}: StopIteration encountered in dataloader.")
                break # Exit epoch loop cleanly
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.error(f"Epoch {epoch+1} Step {i}: CUDA Out Of Memory!")
                    print(f"\nERROR: CUDA Out Of Memory at Epoch {epoch+1}, Step {i}. Try reducing batch size or accumulation steps.")
                    # Optional: Clear cache and try to continue? Risky.
                    # torch.cuda.empty_cache()
                    # return # Stop training
                    sys.exit("CUDA OOM Error") # Exit more forcefully
                else:
                    logger.exception(f"Epoch {epoch+1} Step {i}: Runtime error")
                    print(f"\nERROR: Runtime error at step {i} in epoch {epoch+1}: {e}")
                    traceback.print_exc()
                    # Optionally try to skip batch, but could indicate deeper problem
                    optimizer.zero_grad(set_to_none=True) # Clear potentially bad grads
                    continue # Skip to next batch
            except Exception as batch_exception:
                 logger.exception(f"Epoch {epoch+1} Step {i}: Unhandled error in training loop")
                 print(f"\nERROR during training step {i} in epoch {epoch+1}: {batch_exception}")
                 print("Skipping this batch...")
                 traceback.print_exc(limit=1)
                 optimizer.zero_grad(set_to_none=True) # Clear gradients before continuing
                 continue # Skip to next batch

        # --- End of Epoch ---
        epoch_duration = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss_total / steps_in_epoch if steps_in_epoch > 0 else 0
        avg_epoch_epe = epoch_epe_total / steps_in_epoch if steps_in_epoch > 0 else 0

        # --- Log epoch results ---
        log_msg = (f"Epoch {epoch+1}/{config.epochs} | Time: {epoch_duration:.2f}s | "
                   f"Avg Loss: {avg_epoch_loss:.4f} | Avg EPE: {avg_epoch_epe:.3f} | "
                   f"LR: {current_lr:.6f} | Steps: {steps_in_epoch} | Opt Steps: {actual_optimizer_steps}")
        logger.info(log_msg)
        # Update epoch progress bar description
        epoch_iterator = tqdm(range(start_epoch, config.epochs), desc="Epochs", initial=epoch+1, total=config.epochs) # Recreate for update?
        epoch_iterator.set_postfix(AvgLoss=f"{avg_epoch_loss:.4f}", AvgEPE=f"{avg_epoch_epe:.3f}")
        # Print to console as well
        print(log_msg)


        # --- Save Checkpoint ---
        checkpoint_path = os.path.join(checkpoint_dir, f'raft_epoch_{epoch+1:04d}.pth')
        try:
            # Get model state, handling DataParallel/DDP if it were used
            model_state_to_save = model.module.state_dict() if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model.state_dict()

            checkpoint_data = {
                'epoch': epoch, # Save the epoch that just finished
                'model_state_dict': model_state_to_save,
                'optimizer_state_dict': optimizer.state_dict(),
                # No scheduler state to save
                'loss': avg_epoch_loss, # Save average loss for this epoch
                'epe': avg_epoch_epe,   # Save average EPE for this epoch
                'config_dict': vars(config) # Save config for reference
            }
            if use_amp:
                checkpoint_data['scaler_state_dict'] = scaler.state_dict()

            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

            # --- Clean up older checkpoints (optional) ---
            # Keep last N checkpoints, e.g., N=5
            keep_last_n = 5
            all_checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, 'raft_epoch_*.pth')), key=os.path.getmtime)
            if len(all_checkpoints) > keep_last_n:
                 for old_ckpt in all_checkpoints[:-keep_last_n]:
                     try:
                         os.remove(old_ckpt)
                         # logger.info(f"Removed old checkpoint: {old_ckpt}")
                     except OSError as e:
                         logger.warning(f"Could not remove old checkpoint {old_ckpt}: {e}")


        except Exception as e:
            logger.exception(f"ERROR saving checkpoint to {checkpoint_path}")
            print(f"\nERROR saving checkpoint: {e}")

        # --- End-of-Epoch Visualization ---
        if config.visualize_end_of_epoch and last_batch_data_for_viz is not None:
             try:
                 with torch.no_grad():
                    img1_viz, img2_viz, final_flow_viz = last_batch_data_for_viz
                    visualize_batch(img1_viz, img2_viz, final_flow_viz, epoch + 1, "end_of_epoch", config.vis_dir)
             except Exception as viz_e:
                  print(f"Error during end-of-epoch visualization: {viz_e}")
                  logger.error(f"End-of-epoch visualization failed for epoch {epoch+1}: {viz_e}")


    print("\n--- Training finished ---")
    logger.info("Training finished.")

# ==============================================================================
# Main Execution Block (MODIFIED logging setup)
# ==============================================================================
if __name__ == "__main__":
    config = TrainConfig()

    # --- Setup Logger ---
    # Ensure log directory exists before setting up logger
    log_dir = os.path.dirname(config.log_file)
    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
            print(f"Created log directory: {log_dir}")
        except OSError as e:
            print(f"Error creating log directory {log_dir}: {e}. Logging might fail.")
            # Optionally exit or default to current dir
            config.log_file = os.path.basename(config.log_file) # Log in current dir if creation fails
    logger = setup_logger(config.log_file)

    print("--- Configuration ---")
    # Log effective configuration used
    config_dict = {k: getattr(config, k) for k in dir(config) if not k.startswith('__') and not callable(getattr(config, k))}
    logger.info("="*20 + " Training Configuration " + "="*20)
    for key, value in config_dict.items():
         print(f"{key}: {value}")
         logger.info(f"{key}: {value}")
    logger.info("="*58)
    print("---------------------")


    # --- Basic Validation ---
    if not os.path.isdir(config.final_dir):
         logger.error(f"final_dir not found: {config.final_dir}")
         sys.exit(f"ERROR: final_dir ('{config.final_dir}') not found or not a directory.")
    if not os.path.isdir(config.flow_dir):
         logger.error(f"flow_dir not found: {config.flow_dir}")
         sys.exit(f"ERROR: flow_dir ('{config.flow_dir}') not found or not a directory.")

    # Check resize dimensions divisibility by 8 (common RAFT requirement)
    if config.resize_height % 8 != 0:
        logger.warning(f"resize_height ({config.resize_height}) is not divisible by 8. RAFT may require this.")
        print(f"Warning: resize_height ({config.resize_height}) is not divisible by 8.")
    if config.resize_width % 8 != 0:
        logger.warning(f"resize_width ({config.resize_width}) is not divisible by 8. RAFT may require this.")
        print(f"Warning: resize_width ({config.resize_width}) is not divisible by 8.")

    # Check AMP and GPU consistency
    if config.mixed_precision and (config.gpu is None or not torch.cuda.is_available()):
        print("Warning: mixed_precision=True but CUDA unavailable or GPU not selected. Disabling AMP.")
        logger.warning("mixed_precision=True but CUDA unavailable/not selected. Disabling AMP.")
        config.mixed_precision = False # Override config
    elif config.gpu is not None and not torch.cuda.is_available():
         print(f"Warning: GPU {config.gpu} requested, but CUDA not available. Switching to CPU.")
         logger.warning(f"GPU {config.gpu} requested, but CUDA not available. Switching to CPU.")
         config.gpu = None # Override config
         config.mixed_precision = False

    # Start training
    try:
        train(config, logger)
    except KeyboardInterrupt:
         print("\n--- Training interrupted by user (KeyboardInterrupt) ---")
         logger.warning("Training interrupted by user.")
         sys.exit(0) # Clean exit
    except Exception as main_exception:
        print(f"\n--- A critical error occurred during training ---")
        logger.exception("Unhandled exception during training execution block.") # Log the full traceback
        print(f"Error Type: {type(main_exception).__name__}")
        print(f"Error Details: {main_exception}")
        print("\n--- Traceback ---")
        traceback.print_exc()
        print("-----------------")
        sys.exit(1) # Indicate error exit