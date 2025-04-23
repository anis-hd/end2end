# ==============================================================================
# Imports
# ==============================================================================
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import transforms # Keep for InterpolationMode potentially
# --- Added for AMP ---
from torch.cuda.amp import autocast # Use autocast for inference if desired
# --- Added for File I/O and Visualization ---
import cv2
import numpy as np
from tqdm import tqdm
# --- Standard imports ---
import os
import glob
import re
import sys
import traceback
import time
import argparse # Use argparse for flexibility
from pathlib import Path # For easier path manipulation

# ==============================================================================
# RAFT Model Import (Copied from previous script)
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
# Configuration Parameters (Adjusted for Sequence Processing)
# ==============================================================================
class FlowGenerationConfig:
    # --- Paths ---
    input_base_dir = './sequence'      # Base directory containing sequence folders (00061, etc.)
    output_base_dir = './generated_flow' # Base directory to save generated .flo files
    checkpoint_dir = './raft_checkpoints_amp_epe' # Directory where training checkpoints are saved

    # --- Model & Processing ---
    # *** MUST match training ***
    resize_height = 360  # Must be divisible by 8
    resize_width = 640   # Must be divisible by 8
    raft_iters = 12      # Number of refinement iterations in RAFT
    dropout = 0.0        # Typically 0 for inference
    mixed_precision = True # Use AMP for potentially faster inference if GPU supports it

    # --- Device ---
    gpu = 0 # GPU ID to use (e.g., 0). Use None for CPU.

    # --- Frame Matching Pattern ---
    frame_prefix = "im"
    frame_suffix = ".png"

# ==============================================================================
# Helper Functions
# ==============================================================================
def find_latest_checkpoint(checkpoint_dir):
    """Finds the checkpoint with the highest epoch number in the directory."""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'raft_epoch_*.pth'))
    if not checkpoints:
        return None
    # Extract epoch numbers using regex
    epoch_nums = []
    valid_checkpoints = []
    for ckpt in checkpoints:
        match = re.search(r'raft_epoch_(\d+).pth', os.path.basename(ckpt))
        if match:
            epoch_nums.append(int(match.group(1)))
            valid_checkpoints.append(ckpt)
        else:
            print(f"Warning: Found file matching pattern but couldn't extract epoch: {ckpt}")

    if not valid_checkpoints:
        return None

    # Find the checkpoint corresponding to the highest epoch number
    latest_epoch_idx = np.argmax(epoch_nums)
    return valid_checkpoints[latest_epoch_idx]

def preprocess_frame(frame_np, resize_shape, device):
    """Converts OpenCV frame (BGR, HxWxC, uint8) to model input tensor."""
    if frame_np is None:
        return None
    try:
        # 1. BGR to RGB (assuming cv2 reads in BGR)
        rgb_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        # 2. To Tensor (scales to [0, 1]) and permute to CxHxW
        tensor = TF.to_tensor(rgb_frame)
        # 3. Resize
        resized_tensor = TF.resize(tensor, list(resize_shape), antialias=True)
        # 4. Add batch dimension and move to device
        return resized_tensor.unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error preprocessing frame: {e}")
        return None

def upsample_flow(flow_pred: torch.Tensor, target_shape: tuple[int, int]):
    """
    Upsamples predicted flow (B, 2, H, W) to target shape (H_out, W_out)
    and scales the flow values accordingly.
    Assumes flow_pred is the *final* prediction from the model.
    """
    if flow_pred is None:
        return None
    batch_size, _, pred_h, pred_w = flow_pred.shape
    target_h, target_w = target_shape

    if (pred_h, pred_w) == (target_h, target_w):
        return flow_pred # No upsampling needed

    try:
        # Use bilinear interpolation for flow upsampling
        flow_upsampled = TF.resize(flow_pred, [target_h, target_w],
                                   interpolation=transforms.InterpolationMode.BILINEAR,
                                   antialias=False) # Often False for flow

        # Calculate and apply scaling factors
        scale_w = float(target_w) / float(pred_w) if pred_w > 0 else 1.0
        scale_h = float(target_h) / float(pred_h) if pred_h > 0 else 1.0

        flow_upsampled[:, 0, :, :] *= scale_w # Scale u component
        flow_upsampled[:, 1, :, :] *= scale_h # Scale v component

        return flow_upsampled
    except Exception as e:
        print(f"Error upsampling flow: {e}")
        return None

def write_flo_file(filename, flow_uv):
    """Writes a .flo file.

    Args:
        filename (str): Path to save the .flo file.
        flow_uv (np.ndarray): Flow field (H, W, 2) where flow_uv[y, x] = (u, v).
                               Must be float32.
    """
    if flow_uv is None or flow_uv.ndim != 3 or flow_uv.shape[2] != 2:
        print(f"Error: Invalid flow data provided for saving to {filename}.")
        return
    if flow_uv.dtype != np.float32:
        print(f"Warning: Flow data type is {flow_uv.dtype}, converting to float32 for saving {filename}.")
        flow_uv = flow_uv.astype(np.float32)

    height, width, _ = flow_uv.shape
    magic = np.float32(202021.25) # Magic number for .flo format

    try:
        with open(filename, 'wb') as f:
            f.write(magic.tobytes())
            f.write(np.int32(width).tobytes())
            f.write(np.int32(height).tobytes())
            f.write(flow_uv.tobytes()) # Writes data in row-major order (u1, v1, u2, v2, ...)
    except IOError as e:
        print(f"Error writing .flo file {filename}: {e}")
    except Exception as e:
        print(f"Unexpected error writing .flo file {filename}: {e}")

def extract_frame_num(filename, prefix, suffix):
    """Extracts the numeric part of a frame filename."""
    basename = os.path.basename(filename)
    if basename.startswith(prefix) and basename.endswith(suffix):
        try:
            # Remove prefix and suffix, then convert to int
            num_str = basename[len(prefix):-len(suffix)]
            return int(num_str)
        except ValueError:
            return -1 # Indicate failure if conversion fails
    return -1 # Indicate failure if prefix/suffix don't match

# ==============================================================================
# Main Flow Generation Function
# ==============================================================================
def generate_flow_for_sequences(config: FlowGenerationConfig):
    print("--- Starting Flow Generation for Sequences ---")
    input_base_path = Path(config.input_base_dir)
    output_base_path = Path(config.output_base_dir)

    # --- Basic Checks ---
    if not input_base_path.is_dir():
        print(f"ERROR: Input base directory not found: {input_base_path}")
        sys.exit(1)
    if not os.path.isdir(config.checkpoint_dir):
        print(f"ERROR: Checkpoint directory not found: {config.checkpoint_dir}")
        sys.exit(1)
    if config.resize_height % 8 != 0 or config.resize_width % 8 != 0:
        print(f"Warning: Resize dimensions ({config.resize_height}x{config.resize_width}) "
              "are not divisible by 8. RAFT might require this.")

    # Create output base directory if it doesn't exist
    output_base_path.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved under: {output_base_path}")

    # --- Device Setup ---
    use_amp = config.mixed_precision and torch.cuda.is_available() and config.gpu is not None
    if config.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{config.gpu}")
        torch.cuda.set_device(device)
        print(f"Using device: {device} (CUDA)")
        if use_amp: print("Automatic Mixed Precision (AMP) Enabled for inference.")
    else:
        device = torch.device("cpu")
        use_amp = False # AMP only works on CUDA
        print(f"Using device: {device} (CPU). AMP Disabled.")

    # --- Load Model ---
    print("Loading RAFT model...")
    if RAFT_NEEDS_ARGS:
        model_args = type('raft_args', (object,), {
            'small': False, # Assuming large model from training config
            'dropout': config.dropout,
            'mixed_precision': use_amp # Pass AMP flag if model needs it
        })()
        model = RAFT(model_args)
    else:
        # Load torchvision RAFT without pretrained weights (we'll load our checkpoint)
        model = RAFT(weights=None)

    # --- Load Checkpoint ---
    latest_checkpoint_path = find_latest_checkpoint(config.checkpoint_dir)
    if not latest_checkpoint_path:
        print(f"ERROR: No valid checkpoints found in {config.checkpoint_dir}")
        sys.exit(1)

    print(f"Loading checkpoint: {latest_checkpoint_path}")
    try:
        checkpoint = torch.load(latest_checkpoint_path, map_location='cpu') # Load to CPU first
        model_state = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))

        if not model_state:
             raise KeyError("Could not find 'model_state_dict' or 'state_dict' in checkpoint.")

        # Handle 'module.' prefix (if saved from DataParallel/DDP)
        if all(k.startswith('module.') for k in model_state.keys()):
            print("Removing 'module.' prefix from state_dict keys.")
            model_state = {k.replace('module.', '', 1): v for k, v in model_state.items()}

        # Handle '_orig_mod.' prefix (from torch.compile)
        first_key = next(iter(model_state))
        if '_orig_mod.' in first_key:
            print("Removing '_orig_mod.' prefix from state_dict keys (likely from torch.compile).")
            new_model_state = {}
            for k, v in model_state.items():
                new_k = k.replace('._orig_mod', '')
                new_model_state[new_k] = v
            model_state = new_model_state

        load_result = model.load_state_dict(model_state, strict=False) # Use strict=False
        print(f"Checkpoint loaded. Load Result:")
        if load_result.missing_keys: print(f"  Missing Keys: {load_result.missing_keys}")
        if load_result.unexpected_keys: print(f"  Unexpected Keys: {load_result.unexpected_keys}")

    except Exception as e:
        print(f"ERROR loading checkpoint '{latest_checkpoint_path}': {e}")
        traceback.print_exc()
        sys.exit(1)

    model.to(device)
    model.eval() # Set model to evaluation mode

    # --- Process Sequences ---
    print("Scanning for sequences and processing...")
    total_pairs_processed = 0
    total_pairs_skipped = 0
    sequences_processed = 0
    subsequences_processed = 0

    # Find top-level sequence directories (e.g., 00061, 00070)
    # Sort numerically if they are numbers
    try:
        sequence_dirs = sorted([
            d for d in input_base_path.iterdir() if d.is_dir()
            ], key=lambda x: int(x.name) if x.name.isdigit() else float('inf'))
        # Filter out non-numeric names if necessary, or handle them differently
        sequence_dirs = [d for d in sequence_dirs if d.name.isdigit()]
    except ValueError:
        print(f"Warning: Could not sort sequence directories numerically. Using alphabetical order.")
        sequence_dirs = sorted([d for d in input_base_path.iterdir() if d.is_dir()])

    if not sequence_dirs:
        print(f"No sequence directories found in {input_base_path}")
        return

    # Setup overall progress bar
    # Counting total pairs beforehand is slow, so we won't set total initially
    pbar = tqdm(desc="Processing Frame Pairs", unit="pair")

    try:
        for seq_dir in tqdm(sequence_dirs, desc="Sequences", leave=False):
            seq_id = seq_dir.name
            sequences_processed += 1

            # Find sub-sequence directories (e.g., 0001, 1000)
            try:
                 sub_sequence_dirs = sorted([
                    d for d in seq_dir.iterdir() if d.is_dir()
                    ], key=lambda x: int(x.name) if x.name.isdigit() else float('inf'))
                 sub_sequence_dirs = [d for d in sub_sequence_dirs if d.name.isdigit()]
            except ValueError:
                 print(f"Warning: Could not sort sub-sequence dirs in {seq_dir} numerically. Using alphabetical order.")
                 sub_sequence_dirs = sorted([d for d in seq_dir.iterdir() if d.is_dir()])


            for sub_seq_dir in tqdm(sub_sequence_dirs, desc=f"SubSeqs in {seq_id}", leave=False):
                sub_seq_id = sub_seq_dir.name
                subsequences_processed += 1

                # Find image frames in the current sub-sequence directory
                frame_files = sorted(
                    glob.glob(os.path.join(sub_seq_dir, f"{config.frame_prefix}*{config.frame_suffix}")),
                    key=lambda x: extract_frame_num(x, config.frame_prefix, config.frame_suffix)
                )

                # Filter out files where frame number extraction failed
                frame_files = [f for f in frame_files if extract_frame_num(f, config.frame_prefix, config.frame_suffix) != -1]

                if len(frame_files) < 2:
                    # print(f"Skipping {sub_seq_dir}: Found less than 2 valid frames.")
                    continue

                # Process frame pairs
                for i in range(len(frame_files) - 1):
                    frame1_path = frame_files[i]
                    frame2_path = frame_files[i+1]

                    # Optional: Check for frame consecutiveness (e.g., im1 -> im2)
                    # frame1_num = extract_frame_num(frame1_path, config.frame_prefix, config.frame_suffix)
                    # frame2_num = extract_frame_num(frame2_path, config.frame_prefix, config.frame_suffix)
                    # if frame2_num != frame1_num + 1:
                    #     print(f"Warning: Non-consecutive frames detected in {sub_seq_dir}: {os.path.basename(frame1_path)} -> {os.path.basename(frame2_path)}. Skipping pair.")
                    #     total_pairs_skipped += 1
                    #     continue

                    try:
                        # Load frames
                        img1_np = cv2.imread(frame1_path)
                        img2_np = cv2.imread(frame2_path)

                        if img1_np is None or img2_np is None:
                            print(f"Warning: Failed to load frame pair: {frame1_path} or {frame2_path}. Skipping.")
                            total_pairs_skipped += 1
                            continue

                        # Get original frame dimensions for potential flow resizing later (if needed)
                        # Here we save flow matching the *resized* input dimensions for simplicity
                        # If you need flow matching original frame size, you'd need to resize+scale here.
                        # _, frame_h, frame_w, _ = img1_np.shape # If needed

                        # Preprocess frames
                        img1_tensor = preprocess_frame(img1_np, (config.resize_height, config.resize_width), device)
                        img2_tensor = preprocess_frame(img2_np, (config.resize_height, config.resize_width), device)

                        if img1_tensor is None or img2_tensor is None:
                            print(f"Warning: Failed to preprocess frame pair: {frame1_path} or {frame2_path}. Skipping.")
                            total_pairs_skipped += 1
                            continue

                        # Run Inference
                        with torch.no_grad():
                            with autocast(enabled=use_amp):
                                if RAFT_NEEDS_ARGS or hasattr(model, 'iters'):
                                    flow_predictions = model(img1_tensor, img2_tensor, iters=config.raft_iters)
                                else:
                                    flow_predictions = model(img1_tensor, img2_tensor)

                                # Get final flow prediction (usually the last one)
                                if isinstance(flow_predictions, list):
                                    flow_low_res = flow_predictions[-1]
                                elif torch.is_tensor(flow_predictions):
                                    flow_low_res = flow_predictions
                                else:
                                    print(f"\nWarning: Unexpected model output type: {type(flow_predictions)} for pair {frame1_path}. Skipping.")
                                    total_pairs_skipped += 1
                                    continue

                                # Upsample flow to the *resized* input dimensions
                                flow_output_res = upsample_flow(flow_low_res, (config.resize_height, config.resize_width))

                                if flow_output_res is None:
                                    print(f"Warning: Flow upsampling failed for pair {frame1_path}. Skipping.")
                                    total_pairs_skipped += 1
                                    continue


                        # Convert flow to numpy array (H, W, 2) on CPU
                        flow_np = flow_output_res.squeeze(0).permute(1, 2, 0).cpu().numpy()

                        # --- Prepare Output Path ---
                        frame1_basename = os.path.basename(frame1_path)
                        flow_filename = frame1_basename.replace(config.frame_suffix, ".flo")
                        output_sub_dir = output_base_path / seq_id / sub_seq_id
                        output_flow_path = output_sub_dir / flow_filename

                        # Create output directory if it doesn't exist
                        output_sub_dir.mkdir(parents=True, exist_ok=True)

                        # --- Save .flo file ---
                        write_flo_file(str(output_flow_path), flow_np)

                        total_pairs_processed += 1
                        pbar.update(1) # Update overall progress bar

                    except Exception as e:
                        print(f"\nERROR processing pair {frame1_path} -> {frame2_path}: {e}")
                        traceback.print_exc(limit=1)
                        total_pairs_skipped += 1
                        continue # Skip to next pair

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during sequence processing: {e}")
        traceback.print_exc()
    finally:
        pbar.close()
        print("--- Flow Generation Finished ---")
        print(f"Sequences Scanned: {len(sequence_dirs)}")
        print(f"Sub-sequences Scanned: {subsequences_processed}")
        print(f"Frame Pairs Processed Successfully: {total_pairs_processed}")
        print(f"Frame Pairs Skipped (errors/missing): {total_pairs_skipped}")


# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == "__main__":
    # Use argparse to override config defaults from command line if needed
    parser = argparse.ArgumentParser(description="Generate RAFT optical flow for nested image sequences.")
    parser.add_argument('--input_dir', type=str, default=FlowGenerationConfig.input_base_dir, help='Base directory containing sequence folders (e.g., ./sequence).')
    parser.add_argument('--output_dir', type=str, default=FlowGenerationConfig.output_base_dir, help='Base directory to save generated .flo files (e.g., ./generated_flow).')
    parser.add_argument('--ckpt_dir', type=str, default=FlowGenerationConfig.checkpoint_dir, help='Directory containing model checkpoints.')
    parser.add_argument('--height', type=int, default=FlowGenerationConfig.resize_height, help='Resize height for model input (must be divisible by 8).')
    parser.add_argument('--width', type=int, default=FlowGenerationConfig.resize_width, help='Resize width for model input (must be divisible by 8).')
    parser.add_argument('--iters', type=int, default=FlowGenerationConfig.raft_iters, help='Number of RAFT iterations.')
    parser.add_argument('--gpu', type=int, default=FlowGenerationConfig.gpu, help='GPU ID to use (leave blank or -1 for CPU).')
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision (AMP) inference.')
    parser.add_argument('--frame_prefix', type=str, default=FlowGenerationConfig.frame_prefix, help='Prefix for frame filenames (e.g., "im").')
    parser.add_argument('--frame_suffix', type=str, default=FlowGenerationConfig.frame_suffix, help='Suffix for frame filenames (e.g., ".png").')

    args = parser.parse_args()

    # Update config from args
    config = FlowGenerationConfig()
    config.input_base_dir = args.input_dir
    config.output_base_dir = args.output_dir
    config.checkpoint_dir = args.ckpt_dir
    config.resize_height = args.height
    config.resize_width = args.width
    config.raft_iters = args.iters
    config.gpu = args.gpu if args.gpu >= 0 else None
    config.mixed_precision = not args.no_amp
    config.frame_prefix = args.frame_prefix
    config.frame_suffix = args.frame_suffix

    # Run the main generation process
    generate_flow_for_sequences(config)