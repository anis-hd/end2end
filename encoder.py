# ==============================================================================
# Imports
# ==============================================================================
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import transforms
import cv2
import numpy as np
from tqdm import tqdm
import os
import glob
import re
import sys
import traceback
import time
import argparse
from pathlib import Path
import struct # For packing data into bitstream

# --- Entropy Coding ---
try:
    import constriction # Correct library name for import
    from constriction.stream import stack
except ImportError:
    print("ERROR: Could not import the 'constriction' library.")
    # --- CORRECTED INSTALLATION INSTRUCTION ---
    print("Please install it using: pip install constriction")
    # ------------------------------------------
    sys.exit(1)

# --- Model Components ---
# Assume these files are in the same directory or Python path
try:
    from modules import * # Import all modules defined in modules.py
    from codec import VideoCodec # Import the main codec class
    from utils import load_checkpoint # Import checkpoint loading utility
except ImportError as e:
    print(f"ERROR: Failed to import necessary modules (codec.py, modules.py, utils.py): {e}")
    print("Ensure these files are accessible in your Python environment.")
    sys.exit(1)

# --- RAFT Model Import (Adapted from flow generation script) ---
try:
    from core.raft import RAFT as RAFT_core
    print("Using RAFT from local 'core' module")
    RAFT_MODEL_CLASS = RAFT_core
    RAFT_NEEDS_ARGS = True
except ImportError:
    try:
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        print("Using RAFT large from torchvision.models.optical_flow")
        RAFT_MODEL_CLASS = raft_large
        RAFT_NEEDS_ARGS = False
    except ImportError:
         try:
             from torchvision.models.raft import raft_large
             print("Using RAFT large from torchvision.models.raft (older path)")
             RAFT_MODEL_CLASS = raft_large
             RAFT_NEEDS_ARGS = False
         except ImportError:
            print("\nERROR: Could not import RAFT model. Ensure 'core' dir or torchvision is available.\n")
            sys.exit(1)

# ==============================================================================
# Configuration Parameters (Hardcoded Checkpoints + Argparse for others)
# ==============================================================================

# --- !!! HARDCODED CHECKPOINT PATHS !!! ---
HARDCODED_CODEC_CHECKPOINT = "/home/anis/Desktop/new/tensorflow2pytorch/codec_checkpoints_pretrained_flow_amp/checkpoint_epoch_0001.pth.tar"
HARDCODED_RAFT_CHECKPOINT_DIR = "/home/anis/Desktop/new/tensorflow2pytorch/raft_checkpoints_amp_epe/"
# ------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Video Encoder using Learned Codec and RAFT Flow (Hardcoded Checkpoints)")

    # --- Paths (Input/Output are still arguments) ---
    parser.add_argument('--input_video', type=str, default='./input.mp4', help='Path to the input video file.')
    parser.add_argument('--output_bitstream', type=str, default='./output.bin', help='Path to save the compressed bitstream.')
    # Removed --raft_checkpoint_dir and --codec_checkpoint arguments

    # --- Model & Processing ---
    parser.add_argument('--resize_height', type=int, default=256, help='Height to resize frames for codec input (must be divisible by 8 for RAFT).')
    parser.add_argument('--resize_width', type=int, default=448, help='Width to resize frames for codec input (must be divisible by 8 for RAFT).')
    parser.add_argument('--raft_iters', type=int, default=12, help='Number of refinement iterations in RAFT.')
    parser.add_argument('--no_raft_amp', action='store_true', help='Disable AMP for RAFT inference.')

    # --- Device ---
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (e.g., 0). Use -1 for CPU.')

    args = parser.parse_args()

    if args.resize_height % 8 != 0 or args.resize_width % 8 != 0:
        print(f"Warning: Resize dimensions ({args.resize_height}x{args.resize_width}) "
              "are not divisible by 8. RAFT might require this.")

    # Add hardcoded paths to the args object for convenience within encode_video
    args.codec_checkpoint = HARDCODED_CODEC_CHECKPOINT
    args.raft_checkpoint_dir = HARDCODED_RAFT_CHECKPOINT_DIR

    return args

# ==============================================================================
# Helper Functions (Remain the same)
# ==============================================================================

def find_latest_checkpoint_raft(checkpoint_dir):
    """Finds the RAFT checkpoint with the highest epoch number."""
    # Check if checkpoint_dir exists before globbing
    if not os.path.isdir(checkpoint_dir):
        print(f"ERROR: RAFT checkpoint directory not found: {checkpoint_dir}")
        return None
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'raft_epoch_*.pth'))
    if not checkpoints: return None
    latest_epoch = -1
    latest_ckpt = None
    for ckpt in checkpoints:
        match = re.search(r'raft_epoch_(\d+).pth', os.path.basename(ckpt))
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_ckpt = ckpt
    return latest_ckpt

def preprocess_frame(frame_np, resize_shape, device):
    """Converts OpenCV frame (BGR, HxWxC, uint8) to model input tensor (B=1, C=3, H, W), RGB, [0,1]."""
    if frame_np is None: return None
    try:
        rgb_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        tensor = TF.to_tensor(rgb_frame) # Scales to [0, 1], CxHxW
        resized_tensor = TF.resize(tensor, list(resize_shape), antialias=True)
        return resized_tensor.unsqueeze(0).to(device) # Add batch dim, move to device
    except Exception as e:
        print(f"Error preprocessing frame: {e}")
        traceback.print_exc()
        return None

def load_raft_model(checkpoint_dir, device, use_amp, raft_iters):
    """Loads the RAFT model and checkpoint."""
    print("Loading RAFT model...")
    if RAFT_NEEDS_ARGS:
        # Create a dummy args object if the RAFT implementation needs it
        model_args = type('raft_args', (object,), {
            'small': False, 'dropout': 0.0, 'mixed_precision': use_amp
        })()
        model = RAFT_MODEL_CLASS(model_args)
    else:
        model = RAFT_MODEL_CLASS(weights=None) # Load structure, not pretrained weights

    latest_checkpoint_path = find_latest_checkpoint_raft(checkpoint_dir)
    if not latest_checkpoint_path:
        # Error message printed within find_latest_checkpoint_raft if dir not found
        if os.path.isdir(checkpoint_dir):
             print(f"ERROR: No valid RAFT checkpoints (raft_epoch_*.pth) found in {checkpoint_dir}")
        return None

    print(f"Loading RAFT checkpoint: {latest_checkpoint_path}")
    try:
        checkpoint = torch.load(latest_checkpoint_path, map_location='cpu')
        # Handle various potential state_dict keys
        model_state = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        if not model_state:
             raise KeyError("Could not find model state dictionary in RAFT checkpoint.")

        # Handle 'module.' prefix
        if all(k.startswith('module.') for k in model_state.keys()):
            model_state = {k.replace('module.', '', 1): v for k, v in model_state.items()}
        # Handle '_orig_mod.' prefix (from torch.compile)
        first_key = next(iter(model_state))
        if '_orig_mod.' in first_key:
             print("Removing '_orig_mod.' prefix from RAFT state_dict keys.")
             model_state = {k.replace('._orig_mod', ''): v for k, v in model_state.items()}


        load_result = model.load_state_dict(model_state, strict=False)
        print(f"RAFT Checkpoint loaded. Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}")

    except Exception as e:
        print(f"ERROR loading RAFT checkpoint '{latest_checkpoint_path}': {e}")
        traceback.print_exc()
        return None

    model.to(device)
    model.eval()
    print("RAFT model loaded successfully.")
    return model

def load_codec_model(checkpoint_path, device):
    """Loads the VideoCodec model and checkpoint."""
    print(f"Loading Video Codec model from: {checkpoint_path}") # Show hardcoded path
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Codec checkpoint not found: {checkpoint_path}")
        return None

    # --- Determine model parameters from checkpoint if possible ---
    print("Loading codec checkpoint to infer parameters (or use defaults)...")
    ckpt_data = torch.load(checkpoint_path, map_location='cpu')

    # Try to load config from checkpoint
    config = ckpt_data.get('config')
    if config:
        print("Loaded configuration from codec checkpoint.")
        m_lc = config.get('motion_latent_channels', 128)
        r_lc = config.get('residual_latent_channels', 192) # Match train.py default
        m_hc = config.get('motion_hyper_channels', 128)
        r_hc = config.get('residual_hyper_channels', 128)
        mcn_bc = config.get('mcn_base_channels', 32)
    else:
        print("WARNING: Codec config not found in checkpoint. Using default parameters.")
        # Use defaults that match the training script's defaults if possible
        m_lc, r_lc, m_hc, r_hc, mcn_bc = 128, 192, 128, 128, 32 # Match train.py defaults

    print(f"Codec Params: MotionLatent={m_lc}, ResidualLatent={r_lc}, MotionHyper={m_hc}, ResidualHyper={r_hc}, MCNBase={mcn_bc}")

    model = VideoCodec(
        motion_latent_channels=m_lc,
        residual_latent_channels=r_lc,
        motion_hyper_channels=m_hc,
        residual_hyper_channels=r_hc,
        mcn_base_channels=mcn_bc
    )

    # Use the utility to load the state dict correctly
    _ , _ = load_checkpoint(checkpoint_path, model, optimizer=None, device=device) # We only need the model loaded

    model.to(device)
    model.eval()
    print("Video Codec model loaded successfully.")
    return model

def encode_frame_data(coder, data_tuple):
    """Encodes indices using ANS coder and returns bytes."""
    indices, cdf_lower, cdf_upper = data_tuple
    # Ensure data is on CPU and potentially flatten
    indices_np = indices.cpu().numpy().flatten().astype(np.int32) # Ensure int32
    cdf_lower_np = cdf_lower.cpu().numpy().flatten().astype(np.float64) # Use float64 for precision
    cdf_upper_np = cdf_upper.cpu().numpy().flatten().astype(np.float64)

    # Ensure cdf_lower < cdf_upper and probabilities are positive
    min_prob = 1e-9 # Minimum probability mass
    cdf_upper_np = np.maximum(cdf_lower_np + min_prob, cdf_upper_np)

    # Check for invalid ranges (CDF must be [0, 1]) - Clamp just in case
    cdf_lower_np = np.clip(cdf_lower_np, 0.0, 1.0)
    cdf_upper_np = np.clip(cdf_upper_np, 0.0, 1.0)


    # Encode symbols in reverse order for ANS
    coder.encode_reverse(indices_np, cdf_lower_np, cdf_upper_np)
    return coder.get_compressed()

def write_uint32(f, value):
    """Writes a 32-bit unsigned integer to the file."""
    f.write(struct.pack('>I', value)) # Use big-endian for consistency

def write_int32_array(f, arr):
    """Writes a list/array of 4 int32 values."""
    if len(arr) != 4: raise ValueError("Shape array must have 4 elements (B, C, H, W)")
    f.write(struct.pack('>iiii', *arr)) # Big-endian signed integers

# ==============================================================================
# Main Encoding Function (Remains the same internally, uses args passed to it)
# ==============================================================================
@torch.no_grad()
def encode_video(args):
    print("--- Starting Video Encoding ---")
    print(f"Using Codec Checkpoint: {args.codec_checkpoint}")
    print(f"Using RAFT Checkpoint Dir: {args.raft_checkpoint_dir}")

    # --- Device Setup ---
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
        print(f"Using device: {device} (CUDA)")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device} (CPU)")
    use_raft_amp = not args.no_raft_amp and device.type == 'cuda'

    # --- Load Models ---
    # Pass the hardcoded paths from the args object
    raft_model = load_raft_model(args.raft_checkpoint_dir, device, use_raft_amp, args.raft_iters)
    codec_model = load_codec_model(args.codec_checkpoint, device)
    if raft_model is None or codec_model is None:
        sys.exit(1)

    # --- Input/Output Setup ---
    input_video_path = Path(args.input_video)
    output_bitstream_path = Path(args.output_bitstream)
    if not input_video_path.is_file():
        print(f"ERROR: Input video not found: {input_video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {input_video_path}")
        sys.exit(1)

    frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Input video: {frame_count_total} frames, {fps:.2f} FPS")
    print(f"Output bitstream: {output_bitstream_path}")
    print(f"Frames will be resized to: {args.resize_height}x{args.resize_width}")

    # --- Encoding Loop ---
    previous_reconstructed_frame_tensor = None
    frame_idx = 0
    total_bytes_written = 0

    try:
        with open(output_bitstream_path, 'wb') as f_out:
            # --- Write Header (Optional but Recommended) ---
            write_uint32(f_out, args.resize_width)
            write_uint32(f_out, args.resize_height)
            total_bytes_written += 8

            pbar = tqdm(total=frame_count_total, desc="Encoding Frames")
            while True:
                ret, current_frame_raw = cap.read()
                if not ret:
                    break # End of video

                current_frame_tensor = preprocess_frame(current_frame_raw,
                                                        (args.resize_height, args.resize_width),
                                                        device)
                if current_frame_tensor is None:
                    print(f"Warning: Skipping frame {frame_idx} due to preprocessing error.")
                    frame_idx += 1
                    pbar.update(1)
                    continue

                # --- Handle First Frame (I-Frame Limitation) ---
                if frame_idx == 0:
                    print("Processing first frame (used as reference, not encoded by this P-codec)")
                    previous_reconstructed_frame_tensor = current_frame_tensor
                    frame_idx += 1
                    pbar.update(1)
                    continue # Skip encoding for the first frame

                # --- P-Frame Encoding ---
                start_time_frame = time.time()

                # 1. Motion Estimation (RAFT)
                with torch.cuda.amp.autocast(enabled=use_raft_amp):
                    flow_predictions = raft_model(previous_reconstructed_frame_tensor,
                                                  current_frame_tensor,
                                                  iters=args.raft_iters)
                    flow_low_res = flow_predictions[-1] if isinstance(flow_predictions, list) else flow_predictions

                # Upsample flow
                _, _, h_low, w_low = flow_low_res.shape
                h_codec, w_codec = args.resize_height, args.resize_width
                flow_codec_res = TF.resize(flow_low_res, [h_codec, w_codec],
                                           interpolation=transforms.InterpolationMode.BILINEAR,
                                           antialias=False)
                scale_w = float(w_codec) / w_low if w_low > 0 else 1.0
                scale_h = float(h_codec) / h_low if h_low > 0 else 1.0
                flow_scaled = torch.zeros_like(flow_codec_res)
                flow_scaled[:, 0, :, :] = flow_codec_res[:, 0, :, :] * scale_w
                flow_scaled[:, 1, :, :] = flow_codec_res[:, 1, :, :] * scale_h
                flow_input = flow_scaled

                # 2. Motion & Residual Compression (VideoCodec)
                compression_data = codec_model.compress_frame(
                    previous_reconstructed_frame_tensor,
                    current_frame_tensor,
                    flow_input
                )

                # 3. Entropy Coding (Constriction ANS)
                coder = stack.AnsCoder()
                res_main_bytes = encode_frame_data(coder, compression_data["residual"]["main"])
                res_hyper_bytes = encode_frame_data(coder, compression_data["residual"]["hyper"])
                mot_main_bytes = encode_frame_data(coder, compression_data["motion"]["main"])
                mot_hyper_bytes = encode_frame_data(coder, compression_data["motion"]["hyper"])

                # 4. Bitstream Writing
                motion_shape = compression_data["motion_latent_shape"]
                residual_shape = compression_data["residual_latent_shape"]
                write_int32_array(f_out, motion_shape)
                write_int32_array(f_out, residual_shape)
                total_bytes_written += 8 * 4
                write_uint32(f_out, len(mot_hyper_bytes)); f_out.write(mot_hyper_bytes)
                total_bytes_written += 4 + len(mot_hyper_bytes)
                write_uint32(f_out, len(mot_main_bytes)); f_out.write(mot_main_bytes)
                total_bytes_written += 4 + len(mot_main_bytes)
                write_uint32(f_out, len(res_hyper_bytes)); f_out.write(res_hyper_bytes)
                total_bytes_written += 4 + len(res_hyper_bytes)
                write_uint32(f_out, len(res_main_bytes)); f_out.write(res_main_bytes)
                total_bytes_written += 4 + len(res_main_bytes)

                # 5. Local Reconstruction
                quantized_motion_indices = compression_data["motion"]["main"][0].float()
                quantized_residual_indices = compression_data["residual"]["main"][0].float()
                flow_reconstructed = codec_model.motion_decoder(quantized_motion_indices)
                warped_frame = codec_model.warping_layer(previous_reconstructed_frame_tensor, flow_reconstructed)
                frame_motion_compensated = codec_model.motion_compensation_net(warped_frame, flow_reconstructed, previous_reconstructed_frame_tensor)
                residual_reconstructed = codec_model.residual_decoder(quantized_residual_indices)
                current_reconstructed_frame = frame_motion_compensated + residual_reconstructed
                current_reconstructed_frame = torch.clamp(current_reconstructed_frame, 0.0, 1.0)
                previous_reconstructed_frame_tensor = current_reconstructed_frame

                # --- Logging & Progress ---
                frame_time = time.time() - start_time_frame
                pbar.set_postfix({
                    "Frame Time": f"{frame_time:.3f}s",
                    "Total MB": f"{total_bytes_written / (1024*1024):.2f}"
                })
                pbar.update(1)
                frame_idx += 1

    except Exception as e:
        print(f"\nERROR during encoding loop at frame index {frame_idx}: {e}")
        traceback.print_exc()
    finally:
        pbar.close()
        cap.release()
        print("--- Encoding Finished ---")
        if total_bytes_written > 0:
             print(f"Output bitstream size: {total_bytes_written / (1024*1024):.3f} MB")
             print(f"Frames encoded (P-frames): {frame_idx - 1}")
             if frame_idx > 1:
                 avg_bytes_per_frame = total_bytes_written / (frame_idx - 1)
                 if fps > 0:
                      bitrate_kbps = (avg_bytes_per_frame * fps * 8) / 1000
                      print(f"Average bytes/P-frame: {avg_bytes_per_frame:.2f}")
                      print(f"Estimated bitrate: {bitrate_kbps:.2f} kbps")
        else:
            print("No data written to bitstream.")


# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == "__main__":
    # Parse arguments (excluding the hardcoded checkpoints)
    args = parse_args()
    # Run the encoding process
    encode_video(args)