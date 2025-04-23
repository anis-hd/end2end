# ==============================================================================
# Imports
# ==============================================================================
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import transforms # Keep for InterpolationMode potentially
# --- Added for AMP ---
from torch.cuda.amp import autocast # Use autocast for inference if desired
# --- Added for Video and Visualization ---
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

# ==============================================================================
# RAFT Model Import (Copied from training script)
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
# Configuration Parameters (Adjusted for Inference)
# ==============================================================================
class InferenceConfig:
    # --- Paths ---
    input_video = './input.mp4'
    output_video = './output.mp4'
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

    # --- Visualization ---
    vector_step = 16     # Draw vector every N pixels
    vector_scale = 1.0   # Multiplier for vector length visualization
    vector_color = (0, 255, 0) # Green in BGR format for OpenCV

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
    # 1. BGR to RGB
    rgb_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
    # 2. To Tensor (scales to [0, 1]) and permute to CxHxW
    tensor = TF.to_tensor(rgb_frame)
    # 3. Resize
    resized_tensor = TF.resize(tensor, list(resize_shape), antialias=True)
    # 4. Add batch dimension and move to device
    return resized_tensor.unsqueeze(0).to(device)

def upsample_flow(flow_pred: torch.Tensor, target_shape: tuple[int, int]):
    """
    Upsamples predicted flow (B, 2, H, W) to target shape (H_out, W_out)
    and scales the flow values accordingly.
    Assumes flow_pred is the *final* prediction from the model.
    """
    batch_size, _, pred_h, pred_w = flow_pred.shape
    target_h, target_w = target_shape

    if (pred_h, pred_w) == (target_h, target_w):
        return flow_pred # No upsampling needed

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

def draw_flow_vectors(frame, flow, step=16, scale=1.0, color=(0, 255, 0)):
    """Draws flow vectors on the frame.

    Args:
        frame (np.ndarray): Output frame (BGR format). Modified in place.
        flow (np.ndarray): Flow field (H, W, 2), where flow[y, x] = (u, v).
        step (int): Grid step for drawing vectors.
        scale (float): Multiplier for vector length.
        color (tuple): Color of vectors in BGR.
    """
    h, w = frame.shape[:2]
    for y in range(0, h, step):
        for x in range(0, w, step):
            try:
                fx, fy = flow[y, x] * scale # Get flow components and apply scale
                # Calculate end point, ensuring coordinates are integers
                x2, y2 = int(x + fx), int(y + fy)

                # Draw arrow (line + circle marker for simplicity)
                # Check if start/end points are valid before drawing
                if 0 <= x < w and 0 <= y < h and 0 <= x2 < w and 0 <= y2 < h:
                     # cv2.line(frame, (x, y), (x2, y2), color, 1, cv2.LINE_AA) # Draw line
                     # cv2.circle(frame, (x2, y2), 1, color, -1) # Draw circle at the end point
                     cv2.arrowedLine(frame, (x,y), (x2, y2), color, 1, tipLength=0.3, line_type=cv2.LINE_AA) # More direct arrow

            except IndexError:
                # This might happen if flow map size is slightly different or rounding issues
                # print(f"Warning: Skipping flow vector drawing at ({x},{y}) due to index error.")
                continue
            except Exception as e:
                print(f"Warning: Error drawing vector at ({x},{y}): {e}")
                continue
    return frame # Return modified frame

# ==============================================================================
# Main Inference Function
# ==============================================================================
def run_inference(config: InferenceConfig):
    print("--- Starting Motion Flow Inference ---")

    # --- Basic Checks ---
    if not os.path.exists(config.input_video):
        print(f"ERROR: Input video not found: {config.input_video}")
        sys.exit(1)
    if not os.path.isdir(config.checkpoint_dir):
        print(f"ERROR: Checkpoint directory not found: {config.checkpoint_dir}")
        sys.exit(1)
    if config.resize_height % 8 != 0 or config.resize_width % 8 != 0:
        print(f"Warning: Resize dimensions ({config.resize_height}x{config.resize_width}) "
              "are not divisible by 8. RAFT might require this.")

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

        # --- START MODIFICATION ---

        # 1. Handle 'module.' prefix (if saved from DataParallel/DDP) - Keep this
        if all(k.startswith('module.') for k in model_state.keys()):
            print("Removing 'module.' prefix from state_dict keys.")
            model_state = {k.replace('module.', '', 1): v for k, v in model_state.items()}

        # 2. Handle '_orig_mod.' prefix (from torch.compile)
        # Check if the first key suggests compilation was used
        first_key = next(iter(model_state))
        if '_orig_mod.' in first_key:
            print("Removing '_orig_mod.' prefix from state_dict keys (likely from torch.compile).")
            new_model_state = {}
            for k, v in model_state.items():
                # Remove the '_orig_mod.' part
                new_k = k.replace('._orig_mod', '')
                new_model_state[new_k] = v
            model_state = new_model_state # Use the corrected state dict

        # 3. Load with strict=False to ignore potential extra keys like num_batches_tracked
        load_result = model.load_state_dict(model_state, strict=False)
        print(f"Checkpoint loaded. Load Result:")
        if load_result.missing_keys:
            print(f"  Missing Keys: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"  Unexpected Keys (potentially ok if minor like num_batches_tracked): {load_result.unexpected_keys}")
        # You might want to add a check here: if missing_keys contains essential weights, raise an error.
        # For now, we assume missing keys (if any after fix) are non-critical for inference.

        # --- END MODIFICATION ---

    except Exception as e:
        print(f"ERROR loading checkpoint '{latest_checkpoint_path}': {e}")
        traceback.print_exc()
        sys.exit(1)

    model.to(device)
    model.eval() # Set model to evaluation mode

    # --- Video I/O Setup ---
    print(f"Opening input video: {config.input_video}")
    cap = cv2.VideoCapture(config.input_video)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {config.input_video}")
        sys.exit(1)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Input video properties: {frame_width}x{frame_height} @ {original_fps:.2f} FPS, {frame_count} frames")

    print(f"Setting up output video: {config.output_video}")
    # Define the codec and create VideoWriter object
    # Use 'mp4v' for .mp4 files. Others include 'XVID' for .avi
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Output video will have the *original* dimensions
    writer = cv2.VideoWriter(config.output_video, fourcc, original_fps, (frame_width, frame_height))
    if not writer.isOpened():
        print(f"ERROR: Could not open video writer for: {config.output_video}")
        cap.release()
        sys.exit(1)

    # --- Process Video ---
    print("Processing video frames...")
    prev_frame_tensor = None
    prev_frame_orig = None
    frame_num = 0

    # Use tqdm for progress bar
    pbar = tqdm(total=frame_count, desc="Processing Frames", unit="frame")

    try:
        while cap.isOpened():
            ret, current_frame_orig = cap.read()
            if not ret:
                # End of video or error
                break

            frame_num += 1
            pbar.update(1)

            # Preprocess the current frame for the model
            current_frame_tensor = preprocess_frame(current_frame_orig,
                                                    (config.resize_height, config.resize_width),
                                                    device)

            output_frame = None
            if prev_frame_tensor is not None and prev_frame_orig is not None:
                # We have a pair of frames, calculate flow
                try:
                    with torch.no_grad(): # Essential for inference
                        with autocast(enabled=use_amp): # Use AMP context if enabled
                            # Call model - adapt based on RAFT version
                            if RAFT_NEEDS_ARGS or hasattr(model, 'iters'):
                                flow_predictions = model(prev_frame_tensor, current_frame_tensor, iters=config.raft_iters)
                            else:
                                flow_predictions = model(prev_frame_tensor, current_frame_tensor)

                            # Ensure list and get final prediction (usually the last one)
                            if isinstance(flow_predictions, list):
                                flow_low_res = flow_predictions[-1]
                            elif torch.is_tensor(flow_predictions):
                                flow_low_res = flow_predictions
                            else:
                                print(f"\nWarning: Unexpected model output type: {type(flow_predictions)}. Skipping frame {frame_num-1}.")
                                # Write the previous frame without flow
                                output_frame = prev_frame_orig.copy()


                            if output_frame is None: # Proceed if flow was calculated
                                # Upsample flow to the *resized* input dimensions
                                flow_resized_input = upsample_flow(flow_low_res, (config.resize_height, config.resize_width))

                                # Convert flow to numpy array on CPU (B, 2, H, W) -> (H, W, 2)
                                flow_resized_np = flow_resized_input.squeeze(0).permute(1, 2, 0).cpu().numpy()

                                # --- Resize flow to ORIGINAL frame dimensions and scale vectors ---
                                flow_orig_res_np = cv2.resize(flow_resized_np,
                                                            (frame_width, frame_height),
                                                            interpolation=cv2.INTER_LINEAR) # Linear interp for flow

                                # Scale vectors based on the resize ratio
                                scale_x = float(frame_width) / config.resize_width if config.resize_width > 0 else 1.0
                                scale_y = float(frame_height) / config.resize_height if config.resize_height > 0 else 1.0
                                flow_orig_res_np[:, :, 0] *= scale_x # Scale u
                                flow_orig_res_np[:, :, 1] *= scale_y # Scale v
                                # -------------------------------------------------------------------

                                # Draw vectors on the *previous* original frame
                                annotated_frame = prev_frame_orig.copy() # Don't modify the original directly
                                annotated_frame = draw_flow_vectors(annotated_frame,
                                                                    flow_orig_res_np,
                                                                    step=config.vector_step,
                                                                    scale=config.vector_scale,
                                                                    color=config.vector_color)
                                output_frame = annotated_frame

                except Exception as e:
                    print(f"\nERROR calculating flow or drawing for frame pair ending at {frame_num}: {e}")
                    traceback.print_exc()
                    # Write the previous frame without flow as a fallback
                    output_frame = prev_frame_orig.copy()

            # --- Write the output frame (which is the *previous* frame, possibly annotated) ---
            if output_frame is not None:
                writer.write(output_frame)

            # Update previous frame for the next iteration
            prev_frame_orig = current_frame_orig
            prev_frame_tensor = current_frame_tensor

        # --- Handle the very last frame ---
        # The loop finishes after reading the last frame, but the *second to last* frame
        # (stored in prev_frame_orig) was the last one written with annotations.
        # Write the final frame without annotations.
        if prev_frame_orig is not None:
            print("Writing the final frame (unannotated).")
            writer.write(prev_frame_orig)

    except KeyboardInterrupt:
        print("\nInference interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during video processing: {e}")
        traceback.print_exc()
    finally:
        # --- Cleanup ---
        print("Releasing video resources...")
        pbar.close()
        cap.release()
        writer.release()
        print(f"Output video saved to: {config.output_video}")
        cv2.destroyAllWindows() # Close any OpenCV windows that might have opened

    print("--- Inference finished ---")

# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == "__main__":
    # You can optionally use argparse to override config defaults from command line
    parser = argparse.ArgumentParser(description="Run RAFT optical flow inference on a video and visualize vectors.")
    parser.add_argument('--input', type=str, default=InferenceConfig.input_video, help='Path to input video file.')
    parser.add_argument('--output', type=str, default=InferenceConfig.output_video, help='Path to output video file.')
    parser.add_argument('--ckpt_dir', type=str, default=InferenceConfig.checkpoint_dir, help='Directory containing model checkpoints.')
    parser.add_argument('--height', type=int, default=InferenceConfig.resize_height, help='Resize height for model input (must be divisible by 8).')
    parser.add_argument('--width', type=int, default=InferenceConfig.resize_width, help='Resize width for model input (must be divisible by 8).')
    parser.add_argument('--iters', type=int, default=InferenceConfig.raft_iters, help='Number of RAFT iterations.')
    parser.add_argument('--gpu', type=int, default=InferenceConfig.gpu, help='GPU ID to use (leave blank or -1 for CPU).')
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision (AMP) inference.')
    parser.add_argument('--step', type=int, default=InferenceConfig.vector_step, help='Grid step for drawing flow vectors.')
    parser.add_argument('--scale', type=float, default=InferenceConfig.vector_scale, help='Scale factor for visualizing vector length.')

    args = parser.parse_args()

    # Update config from args
    config = InferenceConfig()
    config.input_video = args.input
    config.output_video = args.output
    config.checkpoint_dir = args.ckpt_dir
    config.resize_height = args.height
    config.resize_width = args.width
    config.raft_iters = args.iters
    config.gpu = args.gpu if args.gpu >= 0 else None
    config.mixed_precision = not args.no_amp
    config.vector_step = args.step
    config.vector_scale = args.scale

    # Run the main inference process
    run_inference(config)