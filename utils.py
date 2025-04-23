import torch
import math
import os
import glob
import re

def compute_psnr(a, b, max_val=1.0):
    """Computes Peak Signal-to-Noise Ratio between two images."""
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    mse = torch.mean((a - b)**2).item()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val) - 10 * math.log10(mse)

def compute_msssim(a, b, data_range=1.0, size_average=True):
    """Computes Multi-Scale Structural Similarity Index (MS-SSIM).
       Requires pytorch-msssim package: pip install pytorch-msssim
    """
    try:
        from pytorch_msssim import ms_ssim
        # Ensure inputs are 4D tensors (B, C, H, W) and on the same device
        if a.dim() != 4: a = a.unsqueeze(0)
        if b.dim() != 4: b = b.unsqueeze(0)
        a = a.to(b.device) # Ensure devices match

        return ms_ssim(a, b, data_range=data_range, size_average=size_average)
    except ImportError:
        print("Warning: pytorch-msssim not installed. MS-SSIM calculation skipped.")
        print("Install using: pip install pytorch-msssim")
        return torch.tensor(0.0, device=a.device) # Return dummy value

def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth.tar", best_filename="model_best.pth.tar"):
    """Saves model and training parameters."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, best_filename)
        torch.save(state['state_dict'], best_filepath) # Save only model state dict for best model
        print(f"Best model saved to {best_filepath}")


def load_checkpoint(checkpoint_path, model, optimizer=None, device=None):
    """Loads model checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint path does not exist: {checkpoint_path}")
        return None, 0 # Return None state and start epoch 0

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle potential DataParallel prefix 'module.'
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    has_module_prefix = False
    for k, v in state_dict.items():
        if k.startswith('module.'):
            has_module_prefix = True
            break
    if has_module_prefix:
        print("Removing 'module.' prefix from checkpoint state_dict...")
        new_state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    else:
        new_state_dict = state_dict

    # --- Handle potential _orig_mod prefix from torch.compile ---
    compiled_prefix = False
    first_key = next(iter(new_state_dict))
    if '_orig_mod.' in first_key:
        compiled_prefix = True

    if compiled_prefix:
        print("Removing '_orig_mod.' prefix (from torch.compile)...")
        compiled_state_dict = {}
        for k, v in new_state_dict.items():
             new_k = k.replace('._orig_mod', '')
             compiled_state_dict[new_k] = v
        new_state_dict = compiled_state_dict
    # --------------------------------------------------------

    # Load into model (allow missing/unexpected keys for flexibility)
    try:
        load_result = model.load_state_dict(new_state_dict, strict=False)
        print("Model state loaded.")
        if load_result.missing_keys:
            print(f"  Missing Keys: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"  Unexpected Keys: {load_result.unexpected_keys}")
    except Exception as e:
         print(f"ERROR loading model state_dict: {e}")
         print("Model weights NOT loaded from checkpoint.")


    start_epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', -float('inf')) # Default to -inf if not found

    if optimizer and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Optimizer state loaded.")
            # Optionally adjust LR after loading optimizer state if needed
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = new_lr
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}. Optimizer state reset.")
    elif optimizer:
         print("Warning: Optimizer state not found in checkpoint.")

    print(f"Resuming from Epoch: {start_epoch + 1}, Best Metric: {best_metric:.4f}")

    return best_metric, start_epoch


def find_latest_checkpoint_file(checkpoint_dir, pattern="checkpoint_epoch_*.pth.tar"):
    """Finds the latest checkpoint file based on epoch number."""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, pattern))
    if not checkpoints:
        return None

    latest_epoch = -1
    latest_ckpt = None
    for ckpt in checkpoints:
        match = re.search(r'epoch_(\d+)', os.path.basename(ckpt))
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_ckpt = ckpt
    return latest_ckpt