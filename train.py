import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms # Keep for functional resize
# *** Import AMP tools ***
from torch.cuda.amp import GradScaler, autocast
import argparse
import os
import sys
import time
from tqdm import tqdm
import math
import numpy as np
import random
import traceback

# Import local modules
from modules import *
from codec import VideoCodec
from dataset import VideoFrameFlowDatasetNested
from utils import compute_psnr, compute_msssim, save_checkpoint, load_checkpoint, find_latest_checkpoint_file

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Function to set random seeds (same)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

# Configuration Class
class TrainConfig:
    # Paths
    frame_base_dir = "./sequence"
    flow_base_dir = "./generated_flow"
    checkpoint_dir = "./codec_checkpoints_pretrained_flow_amp" # Keep _amp suffix
    vis_dir = "./codec_visualizations_pretrained_flow_amp"     # Keep _amp suffix
    log_file = "training_log_pretrained_flow_amp.txt"         # Keep _amp suffix

    # Model Hyperparameters (same)
    motion_latent_channels=128
    residual_latent_channels=192
    motion_hyper_channels=128
    residual_hyper_channels=128
    mcn_base_channels=32

    # Training Hyperparameters
    epochs = 500
    batch_size = 1 # Physical batch size
    # *** REMOVED Gradient Accumulation Steps ***
    # gradient_accumulation_steps = 8
    learning_rate = 1e-4
    lambda_rd = 0.01
    clip_max_norm = 1.0
    seed = 42
    num_workers = 4
    print_freq = 50
    save_freq = 1
    resume = None
    # *** Keep flag for AMP ***
    use_amp = True # Enable/disable AMP

    # Frame processing (same)
    resize_height = 256
    resize_width = 448

    # Hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"


# Visualization Function (same)
def visualize_epoch_end(epoch, save_dir, frame1, frame2, frame2_reconstructed):
    # ... (visualization code remains the same) ...
    os.makedirs(save_dir, exist_ok=True)
    try:
        f1 = frame1[0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        f2 = frame2[0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        f2_rec = frame2_reconstructed[0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Epoch {epoch} Reconstruction Sample", fontsize=16)
        axs[0].imshow(f1); axs[0].set_title("Frame 1 (Reference)"); axs[0].axis("off")
        axs[1].imshow(f2); axs[1].set_title("Frame 2 (Target Original)"); axs[1].axis("off")
        axs[2].imshow(f2_rec); axs[2].set_title("Frame 2 Reconstructed"); axs[2].axis("off")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(save_dir, f"epoch_{epoch:04d}_reconstruction.png")
        plt.savefig(save_path)
        plt.close(fig)
    except Exception as e:
        print(f"\nWARNING: Failed to generate visualization for epoch {epoch}: {e}")
        traceback.print_exc(limit=1)
        if 'plt' in locals() and 'fig' in locals() and plt.fignum_exists(fig.number):
             plt.close(fig)

# Main Training Function
def main(config: TrainConfig):
    set_seed(config.seed)
    # Determine AMP enablement
    amp_enabled = config.use_amp and config.device == "cuda"
    if config.use_amp and config.device != "cuda":
        print("Warning: AMP requested but device is not CUDA. Disabling AMP.")
    elif amp_enabled:
        print("Automatic Mixed Precision (AMP) Enabled.")

    device = torch.device(config.device)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.vis_dir, exist_ok=True)

    # Logging
    log_path = os.path.join(config.checkpoint_dir, config.log_file)
    log_f = open(log_path, 'a')
    def log_message(message): print(message); log_f.write(message + '\n'); log_f.flush()
    log_message("--- Training Configuration ---")
    if amp_enabled: log_message("AMP Enabled")
    # *** REMOVED Logging for Accumulation/Effective Batch Size ***
    for key, value in vars(config).items():
         if not key.startswith('__') and key != 'gradient_accumulation_steps': # Exclude removed config
              log_message(f"{key}: {value}")
    log_message("-----------------------------")

    # Dataset and DataLoader (same setup)
    log_message("Setting up dataset...")
    img_transforms_list = []
    if config.resize_height and config.resize_width:
        log_message(f"Frames resized to: {config.resize_height}x{config.resize_width}")
        img_transforms_list.append(transforms.Resize((config.resize_height, config.resize_width)))
    img_transforms_list.append(transforms.ToTensor())
    img_transform = transforms.Compose(img_transforms_list)
    try:
        train_dataset = VideoFrameFlowDatasetNested( # ... dataset setup ...
             frame_base_dir=config.frame_base_dir,
             flow_base_dir=config.flow_base_dir,
             transform=img_transform
        )
        if len(train_dataset) == 0: log_message("ERROR: Dataset empty."); log_f.close(); return
        train_loader = DataLoader( # ... loader setup ...
             train_dataset,
             batch_size=config.batch_size, # Uses the direct batch size
             shuffle=True, num_workers=config.num_workers,
             pin_memory=(device.type == 'cuda'), drop_last=True
        )
    except Exception as e: # ... error handling ...
         return
    log_message(f"Dataset size: {len(train_dataset)}, Loader size: {len(train_loader)}")

    # Model (same setup)
    log_message("Initializing Video Codec model...")
    model = VideoCodec( # ... params ...
        motion_latent_channels=config.motion_latent_channels,
        residual_latent_channels=config.residual_latent_channels,
        motion_hyper_channels=config.motion_hyper_channels,
        residual_hyper_channels=config.residual_hyper_channels,
        mcn_base_channels=config.mcn_base_channels
    ).to(device)
    if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        log_message(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)

    # Optimizer (same setup)
    log_message("Setting up optimizer...")
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)

    # --- GradScaler for AMP ---
    scaler = GradScaler(enabled=amp_enabled)
    log_message(f"AMP GradScaler Initialized (Enabled: {amp_enabled})")

    # --- Loss Function (same setup) ---
    import torch.nn.functional as F
    def calculate_rd_loss(outputs, target_frame, num_pixels, lambda_rd):
        # ... loss calculation remains the same ...
        mse_loss = F.mse_loss(outputs['frame2_reconstructed'], target_frame)
        total_rate = (outputs['rate_motion'] + outputs['rate_hyper_motion'] +
                      outputs['rate_residual'] + outputs['rate_hyper_residual'])
        bpp = total_rate / (target_frame.shape[0] * num_pixels)
        rd_loss = mse_loss + lambda_rd * bpp
        return rd_loss, mse_loss, bpp

    # --- Checkpoint Loading ---
    start_epoch = 0
    best_metric = -float('inf')
    resume_path = config.resume
    if resume_path: # ... (same checkpoint loading logic, including scaler state) ...
        if resume_path.lower() == 'latest': resume_path = find_latest_checkpoint_file(config.checkpoint_dir)
        if resume_path and os.path.exists(resume_path):
            log_message(f"Resuming from checkpoint: {resume_path}")
            model_to_load = model.module if isinstance(model, nn.DataParallel) else model
            checkpoint = torch.load(resume_path, map_location=device)
            best_metric, start_epoch = load_checkpoint(resume_path, model_to_load, optimizer, device)
            if amp_enabled and 'scaler_state_dict' in checkpoint:
                try: scaler.load_state_dict(checkpoint['scaler_state_dict']); log_message("GradScaler state loaded.")
                except Exception as e: log_message(f"Warn: GradScaler state load failed: {e}")
            elif amp_enabled: log_message("Warn: Resuming AMP, but no scaler state found.")
            start_epoch += 1
        elif resume_path: log_message(f"Warn: Resume path not found: {resume_path}")


    # --- Training Loop ---
    log_message(f"--- Starting Training from Epoch {start_epoch + 1} ---")
    total_batches = len(train_loader)

    # *** REMOVED optimizer.zero_grad() before loop (do it inside) ***

    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_loss, epoch_mse, epoch_bpp, epoch_psnr = 0.0, 0.0, 0.0, 0.0
        batch_iter = tqdm(enumerate(train_loader), total=total_batches, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False)
        start_time = time.time()
        last_batch_data_for_viz = None

        for i, batch_data in batch_iter:
            # *** Standard Batch Processing (No Accumulation) ***
            try:
                frame1_orig_res, frame2_orig_res, flow12_orig_res = batch_data
                frame1 = frame1_orig_res.to(device, non_blocking=True)
                frame2 = frame2_orig_res.to(device, non_blocking=True)
                flow12 = flow12_orig_res.to(device, non_blocking=True)

                B, C, H_frame, W_frame = frame1.shape
                _, _, H_flow, W_flow = flow12.shape

                # Flow Resizing/Scaling (same as before)
                if H_frame != H_flow or W_frame != W_flow:
                    # ... (resize and scale flow12) ...
                    flow12_resized = transforms.functional.resize(flow12, [H_frame, W_frame], interpolation=transforms.InterpolationMode.BILINEAR, antialias=False)
                    scale_w = float(W_frame) / W_flow if W_flow > 0 else 1.0; scale_h = float(H_frame) / H_flow if H_flow > 0 else 1.0
                    flow12_scaled = torch.zeros_like(flow12_resized); flow12_scaled[:, 0, :, :] = flow12_resized[:, 0, :, :] * scale_w; flow12_scaled[:, 1, :, :] = flow12_resized[:, 1, :, :] * scale_h
                    flow12 = flow12_scaled

                num_pixels = H_frame * W_frame

                # *** Zero gradients before forward pass ***
                optimizer.zero_grad()

                # *** AMP: Forward pass within autocast context ***
                with autocast(enabled=amp_enabled):
                    outputs = model(frame1, frame2, flow12)
                    loss, mse, bpp = calculate_rd_loss(outputs, frame2, num_pixels, config.lambda_rd)
                    # Handle potential loss aggregation if using DataParallel
                    if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                        loss = loss.mean(); mse = mse.mean(); bpp = bpp.mean()

                # *** NO Loss Scaling for accumulation ***

                # *** AMP: Scale loss and call backward ***
                scaler.scale(loss).backward()

                # *** Standard Optimizer Step Block ***
                # *** AMP: Unscale gradients before clipping ***
                scaler.unscale_(optimizer)

                # Gradient clipping
                if config.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_max_norm)

                # *** AMP: Optimizer step via scaler ***
                scaler.step(optimizer)

                # *** AMP: Update scaler ***
                scaler.update()
                # *** END Standard Optimizer Step Block ***


                # Logging & Metrics (using the direct batch loss)
                batch_loss = loss.item()
                batch_mse = mse.item()
                batch_bpp = bpp.item()
                with torch.no_grad(): batch_psnr = compute_psnr(outputs['frame2_reconstructed'].float(), frame2.float())

                epoch_loss += batch_loss
                epoch_mse += batch_mse
                epoch_bpp += batch_bpp
                epoch_psnr += batch_psnr

                # Update progress bar periodically
                if (i + 1) % config.print_freq == 0 or i == total_batches - 1:
                    current_lr = optimizer.param_groups[0]['lr']
                    # Postfix shows current batch stats, not running average
                    batch_iter.set_postfix(
                        Loss=f"{batch_loss:.4f}", MSE=f"{batch_mse:.6f}",
                        BPP=f"{batch_bpp:.4f}", PSNR=f"{batch_psnr:.2f}", LR=f"{current_lr:.1e}"
                    )

                # Store last batch data for visualization
                if i == total_batches - 1:
                    last_batch_data_for_viz = {
                        'frame1': frame1.detach(), 'frame2': frame2.detach(),
                        'frame2_reconstructed': outputs['frame2_reconstructed'].detach()
                    }

            # --- OOM Error Handling ---
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    log_message(f"\n***** CUDA Out of Memory during training batch {i} in epoch {epoch+1} *****")
                    log_message("Try reducing batch_size, resize dimensions, model complexity, or use gradient accumulation.")
                    # torch.cuda.empty_cache() # Might help slightly sometimes
                    log_message("Stopping training due to OOM.")
                    log_f.close()
                    sys.exit(1)
                else: # Handle other runtime errors
                    log_message(f"\nERROR during training batch {i} in epoch {epoch+1}: {e}")
                    traceback.print_exc(file=log_f)
                    log_message("Skipping batch...")
                    optimizer.zero_grad() # Clear potentially corrupted grads
                    continue
            except Exception as e: # Catch other potential errors
                log_message(f"\nUNEXPECTED ERROR during training batch {i} in epoch {epoch+1}: {e}")
                traceback.print_exc(file=log_f)
                log_message("Skipping batch...")
                optimizer.zero_grad()
                continue

        # --- End of Epoch ---
        # Calculate average metrics (same logic)
        num_processed_batches = total_batches
        avg_loss = epoch_loss / num_processed_batches if num_processed_batches > 0 else 0
        avg_mse = epoch_mse / num_processed_batches if num_processed_batches > 0 else 0
        avg_bpp = epoch_bpp / num_processed_batches if num_processed_batches > 0 else 0
        avg_psnr = epoch_psnr / num_processed_batches if num_processed_batches > 0 else 0
        epoch_time = time.time() - start_time

        # Enhanced Logging (same)
        log_message("-" * 60)
        log_message(f"Epoch {epoch+1}/{config.epochs} Summary | Time: {epoch_time:.2f}s")
        log_message(f"  Avg Loss: {avg_loss:.5f}")
        log_message(f"  Avg MSE:  {avg_mse:.7f}")
        log_message(f"  Avg BPP:  {avg_bpp:.5f}")
        log_message(f"  Avg PSNR: {avg_psnr:.3f} dB")
        log_message("-" * 60)

        # Checkpointing (same, includes scaler state)
        current_metric = avg_psnr
        is_best = current_metric > best_metric
        if is_best: best_metric = current_metric; log_message(f"*** New Best PSNR: {best_metric:.4f} ***")
        if (epoch + 1) % config.save_freq == 0 or is_best:
            checkpoint_state = {
                'epoch': epoch,
                'state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_metric': best_metric, 'config': vars(config),
                'scaler_state_dict': scaler.state_dict() if amp_enabled else None # Save scaler state
            }
            ckpt_filename = f"checkpoint_epoch_{epoch+1:04d}.pth.tar"
            save_checkpoint(checkpoint_state, is_best, config.checkpoint_dir, filename=ckpt_filename)

        # Visualization (same)
        if last_batch_data_for_viz:
            log_message(f"Generating visualization for epoch {epoch+1}...")
            visualize_epoch_end(epoch + 1, config.vis_dir, **last_batch_data_for_viz)

    log_message("--- Training Finished ---")
    log_f.close()


# Main Execution Block
if __name__ == "__main__":
    config = TrainConfig()
    parser = argparse.ArgumentParser(description="Train Learned Video Codec (AMP Only)")
    # *** REMOVED accum_steps argument ***
    parser.add_argument('--frame_base_dir', type=str, help="Base directory for frames.")
    parser.add_argument('--flow_base_dir', type=str, help="Base directory for flow files.")
    parser.add_argument('--checkpoint_dir', type=str, help="Directory to save checkpoints.")
    parser.add_argument('--vis_dir', type=str, help="Directory to save visualizations.")
    parser.add_argument('--lambda_rd', type=float, help="Rate-distortion lambda.")
    parser.add_argument('--lr', type=float, dest='learning_rate', help="Learning rate.")
    parser.add_argument('--batch_size', type=int, help="Physical batch size.")
    parser.add_argument('--no_amp', action='store_true', help="Disable Automatic Mixed Precision (AMP).")
    parser.add_argument('--epochs', type=int, help="Number of training epochs.")
    parser.add_argument('--resume', type=str, help="Path to checkpoint to resume from, or 'latest'.")
    parser.add_argument('--resize_h', type=int, dest='resize_height', help="Target height for resizing frames.")
    parser.add_argument('--resize_w', type=int, dest='resize_width', help="Target width for resizing frames.")
    parser.add_argument('--workers', type=int, dest='num_workers', help="Number of data loader workers.")
    parser.add_argument('--seed', type=int, help="Random seed.")

    args = parser.parse_args()

    # Update config from args
    if args.frame_base_dir: config.frame_base_dir = args.frame_base_dir
    if args.flow_base_dir: config.flow_base_dir = args.flow_base_dir
    if args.checkpoint_dir: config.checkpoint_dir = args.checkpoint_dir
    if args.vis_dir: config.vis_dir = args.vis_dir
    if args.lambda_rd is not None: config.lambda_rd = args.lambda_rd
    if args.learning_rate is not None: config.learning_rate = args.learning_rate
    if args.batch_size is not None: config.batch_size = args.batch_size
    if args.no_amp: config.use_amp = False
    if args.epochs is not None: config.epochs = args.epochs
    if args.resume is not None: config.resume = args.resume
    if args.resize_height is not None: config.resize_height = args.resize_height
    if args.resize_width is not None: config.resize_width = args.resize_width
    if args.num_workers is not None: config.num_workers = args.num_workers
    if args.seed is not None: config.seed = args.seed

    # --- Final Checks ---
    if not os.path.isdir(config.frame_base_dir): # ... (same checks) ...
        sys.exit(1)
    if not os.path.isdir(config.flow_base_dir):
        sys.exit(1)
    if (config.resize_height and not config.resize_width) or (not config.resize_height and config.resize_width):
         print("ERROR: Both --resize_h and --resize_w must be specified if resizing frames.")
         sys.exit(1)

    # --- Start Training ---
    main(config)