import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms # Keep for functional resize
import argparse
import os
import sys # For sys.exit()
import time
from tqdm import tqdm
import math
import numpy as np
import random
import traceback # Added for logging exceptions

# Import local modules
from modules import * # Import necessary classes from modules.py
from codec import VideoCodec
# *** Import the MODIFIED Dataset ***
from dataset import VideoFrameFlowDatasetNested # Changed import name
from utils import compute_psnr, compute_msssim, save_checkpoint, load_checkpoint, find_latest_checkpoint_file

# *** Import plotting library ***
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive saving
import matplotlib.pyplot as plt

# Function to set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Optional: uncomment for full determinism, may impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

# Configuration Class (or use argparse directly)
class TrainConfig:
    # Paths - *** UPDATED DEFAULTS ***
    frame_base_dir = "./sequence"       # Default base directory for frames
    flow_base_dir = "./generated_flow" # Default base directory for pre-computed flows
    checkpoint_dir = "./codec_checkpoints_pretrained_flow" # Changed dir name
    vis_dir = "./codec_visualizations_pretrained_flow"     # Visualization directory
    log_file = "training_log_pretrained_flow.txt"         # Changed log name

    # Model Hyperparameters
    motion_latent_channels=128
    residual_latent_channels=192
    motion_hyper_channels=128
    residual_hyper_channels=128
    mcn_base_channels=32

    # Training Hyperparameters
    epochs = 500
    batch_size = 1 # Adjusted default based on user snippet
    learning_rate = 1e-4
    lambda_rd = 0.01
    clip_max_norm = 1.0
    seed = 42
    num_workers = 4
    print_freq = 50
    save_freq = 1
    resume = None # Path to checkpoint file or 'latest'

    # Frame processing
    # Define target size for frames (flow will be resized to match this)
    resize_height = 256 # Example: Resize frames to this height
    resize_width = 448  # Example: Resize frames to this width

    # Hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# Visualization Function (Simplified)
# ==============================================================================
def visualize_epoch_end(epoch, save_dir, frame1, frame2, frame2_reconstructed):
    """
    Saves a comparison plot of reference, target, and reconstructed frames
    for the first sample of the batch.
    """
    os.makedirs(save_dir, exist_ok=True)

    try:
        # Select first sample and move to CPU, convert to numpy [0, 1] range
        f1 = frame1[0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        f2 = frame2[0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        f2_rec = frame2_reconstructed[0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)

        # --- Create Simpler 1x3 Plot ---
        fig, axs = plt.subplots(1, 3, figsize=(18, 6)) # Adjusted figsize for 1x3
        fig.suptitle(f"Epoch {epoch} Reconstruction Sample", fontsize=16)

        axs[0].imshow(f1)
        axs[0].set_title("Frame 1 (Reference)")
        axs[0].axis("off")

        axs[1].imshow(f2)
        axs[1].set_title("Frame 2 (Target Original)")
        axs[1].axis("off")

        axs[2].imshow(f2_rec)
        axs[2].set_title("Frame 2 Reconstructed")
        axs[2].axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

        save_path = os.path.join(save_dir, f"epoch_{epoch:04d}_reconstruction.png")
        plt.savefig(save_path)
        plt.close(fig) # Close figure to free memory

    except Exception as e:
        print(f"\nWARNING: Failed to generate visualization for epoch {epoch}: {e}")
        traceback.print_exc(limit=1)
        # Ensure figure is closed if an error occurred
        if 'plt' in locals() and 'fig' in locals() and plt.fignum_exists(fig.number):
             plt.close(fig)

# ==============================================================================
# Main Training Function
# ==============================================================================
def main(config: TrainConfig):
    """Main training loop."""
    set_seed(config.seed)
    device = torch.device(config.device)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    # *** Create visualization directory ***
    os.makedirs(config.vis_dir, exist_ok=True)

    # --- Logging ---
    log_path = os.path.join(config.checkpoint_dir, config.log_file)
    print(f"Logging to: {log_path}")
    log_f = open(log_path, 'a')
    def log_message(message):
        print(message)
        log_f.write(message + '\n')
        log_f.flush()

    log_message("--- Training Configuration (Using Pre-computed Flow) ---")
    for key, value in vars(config).items():
         if not key.startswith('__'): log_message(f"{key}: {value}")
    log_message("------------------------------------------------------")

    # --- Dataset and DataLoader ---
    log_message("Setting up dataset using pre-computed flow...")

    # Define image transforms (including resize if specified)
    img_transforms_list = []
    if config.resize_height and config.resize_width:
        log_message(f"Frames will be resized to: {config.resize_height}x{config.resize_width}")
        img_transforms_list.append(transforms.Resize((config.resize_height, config.resize_width)))
    else:
        log_message("Frames will not be resized by transform (using original size).")
    img_transforms_list.append(transforms.ToTensor())
    img_transform = transforms.Compose(img_transforms_list)

    # *** NO Flow Transform passed to Dataset ***
    flow_transform = None

    try:
        train_dataset = VideoFrameFlowDatasetNested(
            frame_base_dir=config.frame_base_dir,
            flow_base_dir=config.flow_base_dir,
            transform=img_transform, # Only image transform needed now
        )
        if len(train_dataset) == 0:
             log_message(f"ERROR: Training dataset is empty. Check paths:\n"
                         f"  Frame Base Dir: {config.frame_base_dir}\n"
                         f"  Flow Base Dir: {config.flow_base_dir}")
             log_f.close() # Close log file before exiting
             return # Exit if dataset is empty

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True if device.type == 'cuda' else False,
            drop_last=True # Drop last batch if it's smaller than batch_size
        )
    except Exception as e:
         log_message(f"FATAL ERROR during Dataset/DataLoader setup: {e}")
         traceback.print_exc(file=log_f)
         log_f.close()
         return # Exit on setup error

    log_message(f"Dataset size: {len(train_dataset)}, Loader size: {len(train_loader)}")

    # --- Model ---
    log_message("Initializing Video Codec model...")
    model = VideoCodec(
        motion_latent_channels=config.motion_latent_channels,
        residual_latent_channels=config.residual_latent_channels,
        motion_hyper_channels=config.motion_hyper_channels,
        residual_hyper_channels=config.residual_hyper_channels,
        mcn_base_channels=config.mcn_base_channels
    ).to(device)

    # Optional: Use DataParallel for multi-GPU training (simpler setup)
    if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        log_message(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)


    # --- Optimizer ---
    log_message("Setting up optimizer...")
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    # Optional: Learning rate scheduler
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)


    # --- Rate-Distortion Loss Function ---
    # Use torch.nn.functional as F if not already imported
    import torch.nn.functional as F
    def calculate_rd_loss(outputs, target_frame, num_pixels, lambda_rd):
        # Distortion Loss (MSE)
        mse_loss = F.mse_loss(outputs['frame2_reconstructed'], target_frame)

        # Total Rate (BPP = bits per pixel)
        total_rate = (outputs['rate_motion'] + outputs['rate_hyper_motion'] +
                      outputs['rate_residual'] + outputs['rate_hyper_residual'])
        # Ensure bpp is calculated correctly even with DataParallel (outputs are gathered)
        # Rate is usually a scalar sum over the batch already
        bpp = total_rate / (target_frame.shape[0] * num_pixels)

        # RD Loss
        rd_loss = mse_loss + lambda_rd * bpp
        return rd_loss, mse_loss, bpp

    # --- Checkpoint Loading ---
    start_epoch = 0
    best_metric = -float('inf') # Use -inf for PSNR/MS-SSIM maximization

    resume_path = config.resume
    if resume_path:
        if resume_path.lower() == 'latest':
             resume_path = find_latest_checkpoint_file(config.checkpoint_dir) # Use utils function
             if resume_path:
                 log_message(f"Found latest checkpoint: {resume_path}")
             else:
                 log_message("Resume 'latest' specified, but no checkpoints found.")
                 resume_path = None # Start from scratch

        if resume_path and os.path.exists(resume_path):
            # Pass model.module if using DataParallel
            model_to_load = model.module if isinstance(model, nn.DataParallel) else model
            best_metric, start_epoch = load_checkpoint(resume_path, model_to_load, optimizer, device)
            start_epoch += 1 # Start from the next epoch
        elif resume_path:
             log_message(f"Warning: Resume path specified but not found: {resume_path}")


    # --- Training Loop ---
    log_message(f"--- Starting Training from Epoch {start_epoch + 1} ---")
    total_batches = len(train_loader)

    for epoch in range(start_epoch, config.epochs):
        model.train() # Set model to training mode
        epoch_loss, epoch_mse, epoch_bpp, epoch_psnr = 0.0, 0.0, 0.0, 0.0
        batch_iter = tqdm(enumerate(train_loader), total=total_batches, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False)
        start_time = time.time()
        # Variable to store data for visualization
        last_batch_data_for_viz = None

        for i, batch_data in batch_iter:
            try:
                frame1_orig_res, frame2_orig_res, flow12_orig_res = batch_data # Names reflect potential size diff
                frame1 = frame1_orig_res.to(device, non_blocking=True)
                frame2 = frame2_orig_res.to(device, non_blocking=True)
                flow12 = flow12_orig_res.to(device, non_blocking=True) # Original flow

                B, C, H_frame, W_frame = frame1.shape # Target shape from frames
                _, _, H_flow, W_flow = flow12.shape   # Original flow shape

                # <<< --- START: Resize and Scale Flow IN TRAINING LOOP --- >>>
                if H_frame != H_flow or W_frame != W_flow:
                    flow12_resized = transforms.functional.resize(
                        flow12,
                        [H_frame, W_frame],
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        antialias=False # Often False for flow resizing
                    )
                    # Calculate scaling factors based on original flow size and target frame size
                    scale_w = float(W_frame) / W_flow if W_flow > 0 else 1.0
                    scale_h = float(H_frame) / H_flow if H_flow > 0 else 1.0

                    # Apply scaling factors to the *resized* flow
                    # Create a new tensor for the scaled flow
                    flow12_scaled = torch.zeros_like(flow12_resized)
                    flow12_scaled[:, 0, :, :] = flow12_resized[:, 0, :, :] * scale_w # Scale u
                    flow12_scaled[:, 1, :, :] = flow12_resized[:, 1, :, :] * scale_h # Scale v

                    flow12 = flow12_scaled # Use the resized and scaled flow from now on
                # <<< --- END: Resize and Scale Flow IN TRAINING LOOP --- >>>

                # Now flow12 has the same spatial dimensions as frame1 and frame2
                num_pixels = H_frame * W_frame

                optimizer.zero_grad()

                # Forward pass - Model receives correctly sized flow12
                outputs = model(frame1, frame2, flow12)

                # Calculate loss
                loss, mse, bpp = calculate_rd_loss(outputs, frame2, num_pixels, config.lambda_rd)

                # Handle potential loss aggregation if using DataParallel
                if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                    loss = loss.mean() # Average loss across GPUs
                    mse = mse.mean()   # Average mse if it was per-GPU
                    bpp = bpp.mean()   # Average bpp if it was per-GPU


                # Backward pass and optimization
                loss.backward()
                if config.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_max_norm)
                optimizer.step()

                # Logging & Metrics
                batch_loss = loss.item()
                batch_mse = mse.item()
                batch_bpp = bpp.item()
                # Detach outputs for metric calculation to save memory
                with torch.no_grad():
                    batch_psnr = compute_psnr(outputs['frame2_reconstructed'], frame2)

                epoch_loss += batch_loss
                epoch_mse += batch_mse
                epoch_bpp += batch_bpp
                epoch_psnr += batch_psnr

                if (i + 1) % config.print_freq == 0 or i == total_batches - 1:
                    current_lr = optimizer.param_groups[0]['lr']
                    batch_iter.set_postfix(
                        Loss=f"{batch_loss:.4f}", MSE=f"{batch_mse:.6f}",
                        BPP=f"{batch_bpp:.4f}", PSNR=f"{batch_psnr:.2f}", LR=f"{current_lr:.1e}"
                    )

                # Store simplified data for visualization (last batch)
                if i == total_batches - 1:
                     last_batch_data_for_viz = {
                         'frame1': frame1.detach(),
                         'frame2': frame2.detach(),
                         # Only need the final reconstructed frame from outputs
                         'frame2_reconstructed': outputs['frame2_reconstructed'].detach()
                     }

            except Exception as e:
                log_message(f"\nERROR during training batch {i} in epoch {epoch+1}: {e}")
                traceback.print_exc(file=log_f)
                log_message("Skipping batch...")
                continue # Skip to next batch

        # --- End of Epoch ---
        # Calculate average metrics for the epoch
        # Handle division by zero if total_batches is 0 (should not happen if dataset check passed)
        avg_loss = epoch_loss / total_batches if total_batches > 0 else 0
        avg_mse = epoch_mse / total_batches if total_batches > 0 else 0
        avg_bpp = epoch_bpp / total_batches if total_batches > 0 else 0
        avg_psnr = epoch_psnr / total_batches if total_batches > 0 else 0
        epoch_time = time.time() - start_time

        # Enhanced Logging
        log_message("-" * 60)
        log_message(f"Epoch {epoch+1}/{config.epochs} Summary | Time: {epoch_time:.2f}s")
        log_message(f"  Avg Loss: {avg_loss:.5f}")
        log_message(f"  Avg MSE:  {avg_mse:.7f}")
        log_message(f"  Avg BPP:  {avg_bpp:.5f}")
        log_message(f"  Avg PSNR: {avg_psnr:.3f} dB")
        log_message("-" * 60)

        # Optional: Validation Step
        # if val_loader:
        #     val_loss, val_psnr = evaluate(model, val_loader, config, device) # Implement evaluate function
        #     log_message(f"  Validation Loss: {val_loss:.5f} | Validation PSNR: {val_psnr:.3f} dB")
        #     current_metric = val_psnr # Use validation metric for best model saving
        # else:
        current_metric = avg_psnr # Use training PSNR if no validation

        # Optional: Adjust learning rate with scheduler
        # if scheduler: scheduler.step(avg_loss) # Example for ReduceLROnPlateau

        # --- Checkpointing ---
        is_best = current_metric > best_metric
        if is_best:
            best_metric = current_metric
            log_message(f"*** New Best PSNR: {best_metric:.4f} ***")

        if (epoch + 1) % config.save_freq == 0 or is_best:
            checkpoint_state = {
                'epoch': epoch,
                # Save state_dict correctly whether using DataParallel or not
                'state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'scheduler': scheduler.state_dict() if scheduler else None,
                'best_metric': best_metric,
                'config': vars(config) # Save config dictionary
            }
            ckpt_filename = f"checkpoint_epoch_{epoch+1:04d}.pth.tar"
            save_checkpoint(checkpoint_state, is_best, config.checkpoint_dir, filename=ckpt_filename)


        # *** Generate Simplified Visualization ***
        if last_batch_data_for_viz:
            log_message(f"Generating visualization for epoch {epoch+1}...")
            visualize_epoch_end( # Call the simplified function
                epoch + 1,
                config.vis_dir,
                last_batch_data_for_viz['frame1'],
                last_batch_data_for_viz['frame2'],
                last_batch_data_for_viz['frame2_reconstructed'] # Pass only the needed tensor
            )

    log_message("--- Training Finished ---")
    log_f.close()


# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == "__main__":
    config = TrainConfig() # Load default config

    # Argument Parsing (Overrides defaults)
    parser = argparse.ArgumentParser(description="Train Learned Video Codec using Pre-computed Flow (Resize in Loop)")
    # --- Arguments ---
    parser.add_argument('--frame_base_dir', type=str, help="Base directory for frame sequences (e.g., ./sequence).")
    parser.add_argument('--flow_base_dir', type=str, help="Base directory for pre-computed flow files (e.g., ./generated_flow).")
    parser.add_argument('--checkpoint_dir', type=str, help="Directory to save checkpoints.")
    parser.add_argument('--vis_dir', type=str, help="Directory to save visualizations.") # Added
    parser.add_argument('--lambda_rd', type=float, help="Rate-distortion trade-off lambda.")
    parser.add_argument('--lr', type=float, dest='learning_rate', help="Learning rate.")
    parser.add_argument('--batch_size', type=int, help="Batch size.")
    parser.add_argument('--epochs', type=int, help="Number of training epochs.")
    parser.add_argument('--resume', type=str, help="Path to checkpoint to resume from, or 'latest'.")
    parser.add_argument('--resize_h', type=int, dest='resize_height', help="Target height for resizing frames.")
    parser.add_argument('--resize_w', type=int, dest='resize_width', help="Target width for resizing frames.")
    parser.add_argument('--workers', type=int, dest='num_workers', help="Number of data loader workers.")
    parser.add_argument('--seed', type=int, help="Random seed.")

    args = parser.parse_args()

    # --- Update config from args ---
    # Use getattr/setattr for cleaner updates or manual checks like below
    if args.frame_base_dir: config.frame_base_dir = args.frame_base_dir
    if args.flow_base_dir: config.flow_base_dir = args.flow_base_dir
    if args.checkpoint_dir: config.checkpoint_dir = args.checkpoint_dir
    if args.vis_dir: config.vis_dir = args.vis_dir # Update vis_dir
    if args.lambda_rd is not None: config.lambda_rd = args.lambda_rd
    if args.learning_rate is not None: config.learning_rate = args.learning_rate
    if args.batch_size is not None: config.batch_size = args.batch_size
    if args.epochs is not None: config.epochs = args.epochs
    if args.resume is not None: config.resume = args.resume
    if args.resize_height is not None: config.resize_height = args.resize_height
    if args.resize_width is not None: config.resize_width = args.resize_width
    if args.num_workers is not None: config.num_workers = args.num_workers
    if args.seed is not None: config.seed = args.seed

    # --- Final Checks ---
    if not os.path.isdir(config.frame_base_dir):
        print(f"ERROR: Frame base directory not found: {config.frame_base_dir}")
        sys.exit(1)
    if not os.path.isdir(config.flow_base_dir):
        print(f"ERROR: Flow base directory not found: {config.flow_base_dir}")
        sys.exit(1)
    if (config.resize_height and not config.resize_width) or (not config.resize_height and config.resize_width):
         print("ERROR: Both --resize_h and --resize_w must be specified if resizing frames.")
         sys.exit(1)

    # --- Start Training ---
    main(config)