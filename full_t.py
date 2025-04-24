# Single file combining all modules for video codec training using CompressAI

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms # Keep for functional resize and dataset transforms
from PIL import Image
import os
import sys
import glob
import numpy as np
from pathlib import Path
import traceback # For detailed error logging
from tqdm import tqdm # For scanning progress and training loop
import math
import re
import argparse
import time
import random
import compressai
from compressai.entropy_models import EntropyBottleneck
from compressai.entropy_models import GaussianConditional
from compressai.ops import quantize_ste # <--- CORRECT IMPORT

compressai_available = True


# Import AMP tools if available
try:
    from torch.cuda.amp import GradScaler, autocast
    amp_available = True
except ImportError:
    print("WARNING: torch.cuda.amp not available. Automatic Mixed Precision (AMP) will be disabled.")
    amp_available = False
    # Define dummy classes if AMP is not available to avoid errors later
    class autocast:
        def __init__(self, enabled=False): self.enabled = enabled
        def __enter__(self): pass
        def __exit__(self, *args): pass
    class GradScaler:
        def __init__(self, enabled=False): self.enabled=enabled
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def step(self, optimizer): optimizer.step()
        def update(self): pass
        def load_state_dict(self, state_dict): pass
        def state_dict(self): return {}

# Import MS-SSIM if available
try:
    from pytorch_msssim import ms_ssim
    msssim_available = True
except ImportError:
    print("Warning: pytorch-msssim not installed. MS-SSIM calculation will be skipped.")
    print("Install using: pip install pytorch-msssim")
    msssim_available = False
    def ms_ssim(a, b, data_range=1.0, size_average=True):
        # print("Warning: pytorch-msssim not available, returning 0.0 for MS-SSIM.") # Reduce noise
        return torch.tensor(0.0, device=a.device)

# Import plotting tools if available
try:
    import matplotlib
    matplotlib.use('Agg') # Use non-interactive backend
    import matplotlib.pyplot as plt
    plotting_available = True
except ImportError:
    print("Warning: Matplotlib not found. Visualization will be disabled.")
    plotting_available = False

# ==============================================================================
# MODULES (Building Blocks)
# ==============================================================================

# --- Helper Modules & Functions ---

def get_activation(name="leaky_relu"):
    """Returns the specified activation function."""
    if name is None or name.lower() == "none":
        return nn.Identity()
    elif name.lower() == "relu":
        return nn.ReLU(inplace=True)
    elif name.lower() == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif name.lower() == "gelu":
        return nn.GELU()
    elif name.lower() == "sigmoid":
        return nn.Sigmoid()
    elif name.lower() == "tanh":
        return nn.Tanh()
    elif name.lower() == "softplus":
        return nn.Softplus()
    else:
        raise ValueError(f"Unknown activation function: {name}")

class ConvNormAct(nn.Sequential):
    """Basic Convolution -> Normalization -> Activation block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding='same', norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU(0.2, inplace=True), bias=False):
        super().__init__()
        effective_padding = padding
        # Note: PyTorch's 'same' padding mode handles calculation for stride=1
        self.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=effective_padding, bias=bias))
        if norm_layer is not None:
            self.add_module("norm", norm_layer(out_channels))
        if act_layer is not None:
            self.add_module("act", act_layer)

class ConvTransposeNormAct(nn.Sequential):
    """Basic Transposed Convolution -> Normalization -> Activation block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,
                 padding=1, output_padding=1, norm_layer=nn.BatchNorm2d,
                 act_layer=nn.LeakyReLU(0.2, inplace=True), bias=False):
        super().__init__()
        self.add_module("conv_transpose", nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, output_padding=output_padding, bias=bias))
        if norm_layer is not None:
            self.add_module("norm", norm_layer(out_channels))
        if act_layer is not None:
            self.add_module("act", act_layer)

class ResidualBlock(nn.Module):
    """Simple Residual Block."""
    def __init__(self, channels, kernel_size=3, norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU(0.2, inplace=True)):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels, kernel_size, stride=1, padding='same', norm_layer=norm_layer, act_layer=act_layer),
            ConvNormAct(channels, channels, kernel_size, stride=1, padding='same', norm_layer=norm_layer, act_layer=None) # No activation before residual add
        )
        self.final_act = act_layer

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        if self.final_act is not None:
            out = self.final_act(out)
        return out

# --- Core Autoencoder Components ---

class Encoder(nn.Module):
    """Generic CNN Encoder with downsampling."""
    def __init__(self, input_channels, base_channels=64, latent_channels=128, num_downsample_layers=3, num_res_blocks=2):
        super().__init__()
        layers = []
        current_channels = input_channels
        layers.append(ConvNormAct(current_channels, base_channels, kernel_size=5, stride=1, padding='same'))
        current_channels = base_channels
        for i in range(num_downsample_layers):
            out_ch = current_channels * 2
            layers.append(ConvNormAct(current_channels, out_ch, kernel_size=3, stride=2, padding=1))
            current_channels = out_ch
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(current_channels))
        # Final convolution to latent space (no norm/act)
        layers.append(nn.Conv2d(current_channels, latent_channels, kernel_size=3, stride=1, padding='same'))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    """Generic CNN Decoder with upsampling (symmetric to Encoder)."""
    def __init__(self, output_channels, base_channels=64, latent_channels=128, num_upsample_layers=3, num_res_blocks=2, final_activation=None):
        super().__init__()
        layers = []
        channels_before_upsample = base_channels * (2**num_upsample_layers)
        layers.append(ConvNormAct(latent_channels, channels_before_upsample, kernel_size=3, stride=1, padding='same'))
        current_channels = channels_before_upsample
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(current_channels))
        for i in range(num_upsample_layers):
            out_ch = current_channels // 2
            layers.append(ConvTransposeNormAct(current_channels, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1))
            current_channels = out_ch
        layers.append(nn.Conv2d(current_channels, output_channels, kernel_size=5, stride=1, padding='same'))
        if final_activation:
            layers.append(get_activation(final_activation))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

# --- Quantization ---
# REMOVED QuantizerSTE - Handled by CompressAI layers

# --- Entropy Models ---
# REMOVED FactorizedEntropyModel and GaussianConditionalEntropyModel
# Replaced by CompressAI layers within Hyperprior

# --- Hyperprior Network (Using CompressAI) ---

class Hyperprior(nn.Module):
    """
    Hyperprior network using CompressAI layers.
    """
    def __init__(self, latent_channels, hyper_latent_channels=128, num_hyper_layers=2, scale_bound=0.11, likelihood_bound=1e-9):
        super().__init__()
        self.latent_channels = latent_channels
        self.hyper_latent_channels = hyper_latent_channels
        self.scale_bound = float(scale_bound)
        self.likelihood_bound = float(likelihood_bound)

        # Hyper-encoder
        hyper_base_channels = max(32, latent_channels // 4)
        self.hyper_encoder = Encoder(
            input_channels=latent_channels,
            base_channels=hyper_base_channels,
            latent_channels=hyper_latent_channels,
            num_downsample_layers=num_hyper_layers,
            num_res_blocks=0
        )

        # Hyper-decoder
        self.hyper_decoder = Decoder(
            output_channels=latent_channels * 2, # Means + Scales_raw for main latents
            base_channels=hyper_base_channels,
            latent_channels=hyper_latent_channels,
            num_upsample_layers=num_hyper_layers,
            num_res_blocks=0,
            final_activation=None
        )

        # Entropy model for hyper-latents (Factorized)
        self.hyper_entropy_bottleneck = EntropyBottleneck(hyper_latent_channels)
        # Entropy model for main latents (Conditional Gaussian)
        # Note: GaussianConditional expects scales, not scales_raw
        self.main_gaussian_conditional = GaussianConditional(scale_table=None, scale_bound=self.scale_bound)
    def _quantize_latents(self, latents):
        """Quantization for main latents during training (STE)"""
        # Use ste_round for quantization compatible with CompressAI's training
        # return ste_round(latents)
        # Alternative: Simple noise approximation often used with GaussianConditional
        if self.training:
             noise = torch.rand_like(latents) - 0.5
             return latents + noise
        else:
             return torch.round(latents)


    def _get_gaussian_params(self, params):
        # ... (keep this method as is) ...
        means, scales_raw = torch.chunk(params, 2, dim=1)
        scales = F.softplus(scales_raw)
        scales = torch.clamp(scales, min=self.scale_bound)
        return means, scales

    def forward(self, latents): # Training forward pass
        # Encode hyper-latents (use abs value as input)
        hyper_latents_cont = self.hyper_encoder(torch.abs(latents))

        # Process hyper-latents through EntropyBottleneck
        # forward returns: y_hat, likelihoods
        # EntropyBottleneck handles its own quantization internally via self.quantize_ste
        quantized_hyper_latents, hyper_likelihoods = self.hyper_entropy_bottleneck(hyper_latents_cont)

        # Decode hyper-latents to get parameters for main latents
        params = self.hyper_decoder(quantized_hyper_latents)
        means, scales = self._get_gaussian_params(params)

        # --- !!! APPLY quantize_ste HERE !!! ---
        # Quantize the centered latents using the STE function
        quantized_latents_centered = quantize_ste(latents - means)
        # Add the mean back - this is the value fed to the decoder
        quantized_latents_for_decoder = quantized_latents_centered + means

        # Process main latents through GaussianConditional
        # We now calculate the likelihood of the *explicitly quantized* latents.
        # GaussianConditional's forward method can still be used, but the input
        # is now the already quantized version.
        # The internal quantization of GaussianConditional might be skipped or redundant
        # when fed an already quantized input, but it correctly calculates likelihoods.
        _, main_likelihoods = self.main_gaussian_conditional(quantized_latents_for_decoder, scales, means=means)
        # ^^^ Note: The first return value (quantized output) might be ignored now,
        #     as we already have quantized_latents_for_decoder. But we need the likelihoods.

        # Clamp likelihoods for numerical stability before log
        hyper_likelihoods = torch.clamp(hyper_likelihoods, min=self.likelihood_bound)
        main_likelihoods = torch.clamp(main_likelihoods, min=self.likelihood_bound)

        # Calculate rates (sum over all elements)
        rate_hyper = -torch.log2(hyper_likelihoods).sum()
        rate_main = -torch.log2(main_likelihoods).sum()

        # Return the *explicitly quantized* main latents for the decoder path
        # and the calculated rates.
        return quantized_latents_for_decoder, rate_main, rate_hyper


    @torch.no_grad()
    def compress(self, latents):
        """Compresses latents using CompressAI layers."""
        # 1. Encode hyper-latents
        hyper_latents_cont = self.hyper_encoder(torch.abs(latents))

        # 2. Compress hyper-latents
        # compress returns: list of strings, shape
        hyper_strings, hyper_shape = self.hyper_entropy_bottleneck.compress(hyper_latents_cont)

        # 3. Get quantized hyper-latents for decoding parameters
        # Use quantize method with mode="round" or "symbol" for inference
        quantized_hyper_latents = self.hyper_entropy_bottleneck.quantize(hyper_latents_cont, mode="round") # Or "symbol"


        # 4. Decode parameters
        params = self.hyper_decoder(quantized_hyper_latents)
        means, scales = self._get_gaussian_params(params)

        # 5. Compress main latents using derived parameters
        # GaussianConditional.compress takes: x, scales, means=None, **kwargs
        main_strings, main_shape = self.main_gaussian_conditional.compress(latents, scales, means=means)

        return {
            "main": (main_strings, main_shape),
            "hyper": (hyper_strings, hyper_shape)
        }

    @torch.no_grad()
    def decompress(self, main_strings, main_shape, hyper_strings, hyper_shape):
        """Decompresses latents using CompressAI layers."""
        # 1. Decompress hyper-latents
        # decompress takes: list of strings, shape
        quantized_hyper_latents = self.hyper_entropy_bottleneck.decompress(hyper_strings, hyper_shape)

        # 2. Decode parameters
        params = self.hyper_decoder(quantized_hyper_latents)
        means, scales = self._get_gaussian_params(params)

        # 3. Decompress main latents using derived parameters
        # decompress takes: strings, shape, scales, means=None, **kwargs
        quantized_main_latents = self.main_gaussian_conditional.decompress(main_strings, main_shape, scales, means=means)

        return quantized_main_latents

# --- Motion Compensation & Warping ---

class WarpingLayer(nn.Module):
    """ Warps an image using optical flow using F.grid_sample. """
    def __init__(self):
        super().__init__()

    def forward(self, x, flow):
        B, C, H, W = x.size()
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype),
                                        torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype),
                                        indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=2)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        flow_permuted = flow.permute(0, 2, 3, 1)
        norm_flow_x = flow_permuted[..., 0] / ((W - 1) / 2) if W > 1 else torch.zeros_like(flow_permuted[..., 0])
        norm_flow_y = flow_permuted[..., 1] / ((H - 1) / 2) if H > 1 else torch.zeros_like(flow_permuted[..., 1])
        norm_flow = torch.stack((norm_flow_x, norm_flow_y), dim=3)
        sampling_grid = grid + norm_flow
        warped_x = F.grid_sample(x, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)
        return warped_x

class MotionCompensationNetwork(nn.Module):
    """ Refines the warped reference frame using a CNN. """
    def __init__(self, input_channels=3 + 2 + 3, output_channels=3, base_channels=32, num_res_blocks=3):
        super().__init__()
        layers = []
        layers.append(ConvNormAct(input_channels, base_channels, kernel_size=5, padding='same'))
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(base_channels))
        layers.append(nn.Conv2d(base_channels, output_channels, kernel_size=5, padding='same'))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, warped_ref, flow, ref_frame):
        mcn_input = torch.cat([warped_ref, flow, ref_frame], dim=1)
        refined_frame = self.network(mcn_input)
        return refined_frame

# ==============================================================================
# UTILS (Helper Functions)
# ==============================================================================
# (compute_psnr, compute_msssim_metric, save_checkpoint, load_checkpoint, find_latest_checkpoint_file remain the same)

def compute_psnr(a, b, max_val=1.0):
    """Computes Peak Signal-to-Noise Ratio between two tensors."""
    if not isinstance(a, torch.Tensor): a = torch.tensor(a)
    if not isinstance(b, torch.Tensor): b = torch.tensor(b)
    a = a.float(); b = b.float()
    mse = torch.mean((a - b)**2)
    if mse == 0: return float('inf')
    psnr = 20 * math.log10(max_val) - 10 * torch.log10(mse)
    return psnr.item()

def compute_msssim_metric(a, b, data_range=1.0, size_average=True):
    """Computes Multi-Scale Structural Similarity Index (MS-SSIM)."""
    if not msssim_available: return 0.0
    if a.dim() != 4: a = a.unsqueeze(0)
    if b.dim() != 4: b = b.unsqueeze(0)
    a = a.to(b.device).float(); b = b.float()
    msssim_val = ms_ssim(a, b, data_range=data_range, size_average=size_average)
    return msssim_val.item()

def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth.tar", best_filename="model_best.pth.tar"):
    """Saves model and training parameters."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, best_filename)
        torch.save(state['state_dict'], best_filepath)
        print(f"Best model state_dict saved to {best_filepath}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scaler=None, device=None):
    """Loads model checkpoint, optionally optimizer and GradScaler state."""
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint path does not exist: {checkpoint_path}")
        return -float('inf'), 0
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        print("Removing 'module.' prefix from checkpoint state_dict...")
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    if '_orig_mod.' in list(state_dict.keys())[0]:
        print("Removing '_orig_mod.' prefix (from torch.compile)...")
        state_dict = {k.replace('._orig_mod', ''): v for k, v in state_dict.items()}
    try:
        load_result = model.load_state_dict(state_dict, strict=False)
        print("Model state loaded.")
        if load_result.missing_keys: print(f"  > Missing Keys: {load_result.missing_keys}")
        if load_result.unexpected_keys: print(f"  > Unexpected Keys: {load_result.unexpected_keys}")
    except Exception as e: print(f"ERROR loading model state_dict: {e}")
    if optimizer and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Optimizer state loaded.")
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor): state[k] = v.to(device)
        except Exception as e: print(f"Warning: Could not load optimizer state: {e}.")
    elif optimizer: print("Warning: Optimizer state not found in checkpoint.")
    if scaler and scaler.enabled and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
         try:
             scaler.load_state_dict(checkpoint['scaler_state_dict'])
             print("GradScaler state loaded.")
         except Exception as e: print(f"Warning: Could not load GradScaler state: {e}.")
    elif scaler and scaler.enabled: print("Warning: AMP enabled, but GradScaler state not found in checkpoint.")
    start_epoch = checkpoint.get('epoch', -1) + 1
    best_metric = checkpoint.get('best_metric', -float('inf'))
    print(f"Resuming training from Epoch: {start_epoch}, Previous Best Metric: {best_metric:.4f}")
    del checkpoint
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return best_metric, start_epoch

def find_latest_checkpoint_file(checkpoint_dir, pattern="checkpoint_epoch_*.pth.tar"):
    """Finds the latest checkpoint file based on epoch number embedded in the filename."""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, pattern))
    if not checkpoints:
        best_model_path = os.path.join(checkpoint_dir, "model_best.pth.tar")
        if os.path.exists(best_model_path):
             print(f"Warning: No epoch checkpoints found matching '{pattern}'. Found 'model_best.pth.tar'. Cannot fully resume.")
             return None
        return None
    latest_epoch = -1; latest_ckpt = None
    epoch_regex = re.compile(r'epoch_(\d+)')
    for ckpt in checkpoints:
        basename = os.path.basename(ckpt)
        match = epoch_regex.search(basename)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch; latest_ckpt = ckpt
    if latest_ckpt: print(f"Found latest checkpoint: {latest_ckpt} (Epoch {latest_epoch})")
    else: print(f"Warning: No checkpoints found matching pattern '{pattern}' with embedded epoch number.")
    return latest_ckpt

# ==============================================================================
# DATASET
# ==============================================================================
# (read_flo_file and VideoFrameFlowDatasetNested remain the same)

def read_flo_file(filename):
    """ Reads a .flo file (Middlebury format). Returns HxWx2 numpy array or None. """
    try:
        with open(filename, 'rb') as f:
            magic = np.frombuffer(f.read(4), np.float32, count=1)
            if not np.isclose(magic[0], 202021.25): return None
            width = np.frombuffer(f.read(4), np.int32, count=1)[0]
            height = np.frombuffer(f.read(4), np.int32, count=1)[0]
            if width <= 0 or height <= 0 or width > 10000 or height > 10000: return None
            data_bytes = f.read()
            expected_bytes = height * width * 2 * 4
            if len(data_bytes) != expected_bytes: return None
            data = np.frombuffer(data_bytes, np.float32, count=height * width * 2)
            flow = data.reshape((height, width, 2))
            return flow
    except FileNotFoundError: return None
    except Exception as e:
        print(f"Error reading flow file {filename}: {e}")
        return None

class VideoFrameFlowDatasetNested(Dataset):
    """Dataset for loading consecutive frames and pre-computed optical flow."""
    def __init__(self, frame_base_dir, flow_base_dir, frame_prefix="im", frame_suffix=".png", transform=None):
        self.frame_base_path = Path(frame_base_dir).resolve()
        self.flow_base_path = Path(flow_base_dir).resolve()
        self.frame_prefix = frame_prefix
        self.frame_suffix = frame_suffix
        self.transform = transform if transform else transforms.ToTensor()
        self.pairs = []
        print(f"Scanning for frame pairs and flow:")
        print(f"  Frame Base: {self.frame_base_path}")
        print(f"  Flow Base:  {self.flow_base_path}")
        all_frames = list(self.frame_base_path.rglob(f"{self.frame_prefix}*{self.frame_suffix}"))
        print(f"Found {len(all_frames)} potential frame files...")
        frames_by_dir = {}
        for f_path in tqdm(all_frames, desc="Grouping Frames", leave=False):
            dir_path = f_path.parent
            if dir_path not in frames_by_dir: frames_by_dir[dir_path] = []
            frames_by_dir[dir_path].append(f_path)
        print(f"Found frames in {len(frames_by_dir)} directories. Creating pairs...")
        skipped_non_consecutive, skipped_missing_flow, skipped_flow_read_error = 0, 0, 0
        for dir_path, frame_list in tqdm(frames_by_dir.items(), desc="Scanning Directories", leave=False):
            try:
                sorted_frames = sorted(frame_list, key=lambda p: int(p.stem[len(self.frame_prefix):]))
            except ValueError: continue
            for i in range(len(sorted_frames) - 1):
                frame1_path, frame2_path = sorted_frames[i], sorted_frames[i+1]
                try:
                    num1, num2 = int(frame1_path.stem[len(self.frame_prefix):]), int(frame2_path.stem[len(self.frame_prefix):])
                    if num2 != num1 + 1: skipped_non_consecutive += 1; continue
                except (ValueError, IndexError): skipped_non_consecutive += 1; continue
                relative_path_from_base = frame1_path.relative_to(self.frame_base_path)
                flow_path = self.flow_base_path / relative_path_from_base.with_suffix(".flo")
                if flow_path.is_file():
                    flow_data_check = read_flo_file(str(flow_path))
                    if flow_data_check is None: skipped_flow_read_error += 1; continue
                    self.pairs.append((str(frame1_path), str(frame2_path), str(flow_path)))
                else: skipped_missing_flow += 1
        print("-" * 30)
        print(f"Finished scanning. Found {len(self.pairs)} valid frame/flow pairs.")
        if skipped_non_consecutive > 0: print(f"Skipped {skipped_non_consecutive} non-consecutive pairs.")
        if skipped_missing_flow > 0: print(f"Skipped {skipped_missing_flow} pairs due to missing flow.")
        if skipped_flow_read_error > 0: print(f"Skipped {skipped_flow_read_error} pairs due to flow read errors.")
        if not self.pairs: print("ERROR: No valid pairs found! Check paths and naming."); sys.exit(1) # Exit if no data
        print("-" * 30)

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        if idx >= len(self.pairs): raise IndexError("Index out of range")
        frame1_path, frame2_path, flow_path = self.pairs[idx]
        try:
            frame1 = Image.open(frame1_path).convert('RGB')
            frame2 = Image.open(frame2_path).convert('RGB')
            flow_np = read_flo_file(flow_path)
            if flow_np is None: raise RuntimeError(f"Failed to load verified flow file: {flow_path}")
            flow_tensor = torch.from_numpy(flow_np.astype(np.float32)).permute(2, 0, 1)
            if self.transform:
                frame1_transformed, frame2_transformed = self.transform(frame1), self.transform(frame2)
            else:
                frame1_transformed, frame2_transformed = transforms.functional.to_tensor(frame1), transforms.functional.to_tensor(frame2)
            return frame1_transformed, frame2_transformed, flow_tensor
        except Exception as e:
            print(f"\nERROR loading data for index {idx}: {e}")
            print(f"Paths: F1={frame1_path}, F2={frame2_path}, Flow={flow_path}")
            raise RuntimeError(f"Failed to load data at index {idx}") from e

# ==============================================================================
# CODEC (Main Model using CompressAI Hyperprior)
# ==============================================================================

class VideoCodec(nn.Module):
    """ Learned Video Codec using CompressAI Hyperprior. """
    def __init__(self, motion_latent_channels=128, residual_latent_channels=192,
                 motion_hyper_channels=128, residual_hyper_channels=128,
                 mcn_base_channels=32, encoder_base_channels=64,
                 encoder_res_blocks=2, encoder_downsample_layers=3,
                 decoder_res_blocks=2, decoder_upsample_layers=3):
        super().__init__()

        # --- Motion Compression Branch ---
        self.motion_encoder = Encoder(input_channels=2, base_channels=encoder_base_channels // 2,
                                      latent_channels=motion_latent_channels, num_downsample_layers=encoder_downsample_layers,
                                      num_res_blocks=encoder_res_blocks)
        self.motion_hyperprior = Hyperprior(latent_channels=motion_latent_channels,
                                            hyper_latent_channels=motion_hyper_channels)
        self.motion_decoder = Decoder(output_channels=2, base_channels=encoder_base_channels // 2,
                                      latent_channels=motion_latent_channels, num_upsample_layers=decoder_upsample_layers,
                                      num_res_blocks=decoder_res_blocks, final_activation=None)

        # --- Motion Compensation Branch ---
        self.warping_layer = WarpingLayer()
        self.motion_compensation_net = MotionCompensationNetwork(input_channels=3 + 2 + 3, output_channels=3,
                                                                 base_channels=mcn_base_channels)

        # --- Residual Compression Branch ---
        self.residual_encoder = Encoder(input_channels=3, base_channels=encoder_base_channels,
                                        latent_channels=residual_latent_channels, num_downsample_layers=encoder_downsample_layers,
                                        num_res_blocks=encoder_res_blocks)
        self.residual_hyperprior = Hyperprior(latent_channels=residual_latent_channels,
                                              hyper_latent_channels=residual_hyper_channels)
        self.residual_decoder = Decoder(output_channels=3, base_channels=encoder_base_channels,
                                        latent_channels=residual_latent_channels, num_upsample_layers=decoder_upsample_layers,
                                        num_res_blocks=decoder_res_blocks, final_activation=None)

    def forward(self, frame1, frame2, flow12): # Training forward pass
        # Motion Compression
        motion_latents = self.motion_encoder(flow12)
        quantized_motion_latents, rate_motion, rate_hyper_motion = self.motion_hyperprior(motion_latents)
        flow_reconstructed = self.motion_decoder(quantized_motion_latents)

        # Motion Compensation
        warped_frame1 = self.warping_layer(frame1, flow_reconstructed)
        frame2_motion_compensated = self.motion_compensation_net(warped_frame1, flow_reconstructed, frame1)

        # Residual Compression
        residual = frame2 - frame2_motion_compensated
        residual_latents = self.residual_encoder(residual)
        quantized_residual_latents, rate_residual, rate_hyper_residual = self.residual_hyperprior(residual_latents)
        residual_reconstructed = self.residual_decoder(quantized_residual_latents)

        # Final Reconstruction
        frame2_reconstructed = frame2_motion_compensated + residual_reconstructed
        frame2_reconstructed = torch.clamp(frame2_reconstructed, 0.0, 1.0)

        return {
            'frame2_reconstructed': frame2_reconstructed,
            'rate_motion': rate_motion, 'rate_hyper_motion': rate_hyper_motion,
            'rate_residual': rate_residual, 'rate_hyper_residual': rate_hyper_residual,
            'flow_reconstructed': flow_reconstructed,
            'frame2_motion_compensated': frame2_motion_compensated,
            'residual_reconstructed': residual_reconstructed,
        }

    @torch.no_grad()
    def compress_frame(self, frame1, frame2, flow12):
        """ Compresses a P-frame, returns compressed strings and shapes. """
        # --- Motion Compression ---
        motion_latents = self.motion_encoder(flow12)
        # compress returns: {"main": (strings, shape), "hyper": (strings, shape)}
        motion_compress_output = self.motion_hyperprior.compress(motion_latents)

        # --- Need reconstructed flow to calculate residual ---
        # Simulate decompression locally
        quantized_motion_latents = self.motion_hyperprior.decompress(
            motion_compress_output["main"][0], motion_compress_output["main"][1],
            motion_compress_output["hyper"][0], motion_compress_output["hyper"][1]
        )
        flow_reconstructed = self.motion_decoder(quantized_motion_latents)

        # --- Motion Compensation & Residual Calculation ---
        warped_frame1 = self.warping_layer(frame1, flow_reconstructed)
        frame2_motion_compensated = self.motion_compensation_net(warped_frame1, flow_reconstructed, frame1)
        residual = frame2 - frame2_motion_compensated

        # --- Residual Compression ---
        residual_latents = self.residual_encoder(residual)
        residual_compress_output = self.residual_hyperprior.compress(residual_latents)

        # Return compressed data (strings and shapes)
        return {
            "motion": motion_compress_output,
            "residual": residual_compress_output
        }

    @torch.no_grad()
    def decompress_frame(self, frame1, compressed_frame_data):
        """ Decompresses a P-frame using loaded strings/shapes. """
        # --- Motion Decompression ---
        motion_data = compressed_frame_data["motion"]
        quantized_motion_latents = self.motion_hyperprior.decompress(
            motion_data["main"][0], motion_data["main"][1],
            motion_data["hyper"][0], motion_data["hyper"][1]
        )
        flow_reconstructed = self.motion_decoder(quantized_motion_latents)

        # --- Motion Compensation ---
        warped_frame1 = self.warping_layer(frame1, flow_reconstructed)
        frame2_motion_compensated = self.motion_compensation_net(warped_frame1, flow_reconstructed, frame1)

        # --- Residual Decompression ---
        residual_data = compressed_frame_data["residual"]
        quantized_residual_latents = self.residual_hyperprior.decompress(
            residual_data["main"][0], residual_data["main"][1],
            residual_data["hyper"][0], residual_data["hyper"][1]
        )
        residual_reconstructed = self.residual_decoder(quantized_residual_latents)

        # --- Final Reconstruction ---
        frame2_reconstructed = frame2_motion_compensated + residual_reconstructed
        frame2_reconstructed = torch.clamp(frame2_reconstructed, 0.0, 1.0)

        return frame2_reconstructed

# ==============================================================================
# TRAINING SCRIPT
# ==============================================================================

# --- Configuration ---
class TrainConfig:
    # Paths (Using the ones you requested)
    frame_base_dir = "./sequence"
    flow_base_dir = "./generated_flow"
    checkpoint_dir = "./codec_checkpoints_pretrained_flow_amp_compressai" # Updated suffix
    vis_dir = "./codec_visualizations_pretrained_flow_amp_compressai"     # Updated suffix
    log_file = "training_log_pretrained_flow_amp_compressai.txt"         # Updated suffix

    # Model Hyperparameters
    motion_latent_channels: int = 128
    residual_latent_channels: int = 192
    motion_hyper_channels: int = 128
    residual_hyper_channels: int = 128
    mcn_base_channels: int = 32
    encoder_base_channels: int = 64
    encoder_res_blocks: int = 2
    encoder_downsample_layers: int = 3
    decoder_res_blocks: int = 2
    decoder_upsample_layers: int = 3

    # Training Hyperparameters
    epochs: int = 500
    batch_size: int = 4
    learning_rate: float = 1e-4
    lambda_rd: float = 0.01
    clip_max_norm: float = 1.0
    seed: int = 42
    num_workers: int = 4
    print_freq: int = 50
    save_freq: int = 1
    resume: str | None = None
    use_amp: bool = True

    # Frame processing
    resize_height: int | None = 256
    resize_width: int | None = 448

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# --- Utility Functions ---
def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def visualize_epoch_end(epoch, save_dir, frame1, frame2, frame2_reconstructed):
    """Saves a comparison image of original and reconstructed frames."""
    if not plotting_available: return
    if frame1 is None or frame2 is None or frame2_reconstructed is None: return
    os.makedirs(save_dir, exist_ok=True)
    try:
        # Convert to float32 BEFORE converting to numpy
        f1 = frame1[0].detach().cpu().permute(1, 2, 0).float().numpy().clip(0, 1)
        f2 = frame2[0].detach().cpu().permute(1, 2, 0).float().numpy().clip(0, 1)
        f2_rec = frame2_reconstructed[0].detach().cpu().permute(1, 2, 0).float().numpy().clip(0, 1)

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Epoch {epoch} Reconstruction Sample", fontsize=16)
        axs[0].imshow(f1); axs[0].set_title("Frame 1 (Reference)"); axs[0].axis("off")
        axs[1].imshow(f2); axs[1].set_title("Frame 2 (Target)"); axs[1].axis("off")
        axs[2].imshow(f2_rec); axs[2].set_title("Frame 2 Reconstructed"); axs[2].axis("off")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(save_dir, f"epoch_{epoch:04d}_reconstruction.png")
        plt.savefig(save_path)
        plt.close(fig)
    except Exception as e:
        print(f"\nWARNING: Failed to generate visualization for epoch {epoch}: {e}")
        if 'plt' in locals() and 'fig' in locals() and hasattr(fig, 'number') and plt.fignum_exists(fig.number): plt.close(fig)

# --- Loss Function ---
def calculate_rd_loss(outputs, target_frame, num_pixels, lambda_rd):
    """ Calculates the Rate-Distortion loss. """
    mse_loss = F.mse_loss(outputs['frame2_reconstructed'], target_frame)
    # Rates from CompressAI Hyperprior are already sums over spatial dims
    total_rate_bits = (outputs['rate_motion'] + outputs['rate_hyper_motion'] +
                       outputs['rate_residual'] + outputs['rate_hyper_residual'])
    batch_size = target_frame.shape[0]
    if batch_size == 0 or num_pixels == 0: bpp_total = torch.tensor(0.0, device=target_frame.device)
    else: bpp_total = total_rate_bits / (batch_size * num_pixels)
    rd_loss = mse_loss + lambda_rd * bpp_total
    return rd_loss, mse_loss, bpp_total

# --- Main Training Function ---
def main(config: TrainConfig):
    set_seed(config.seed)

    # --- Setup ---
    amp_enabled = config.use_amp and config.device == "cuda" and amp_available
    if config.use_amp and not amp_available: print("Warning: AMP requested but unavailable. Disabling AMP.")
    elif config.use_amp and config.device != "cuda": print("Warning: AMP requested but not on CUDA. Disabling AMP."); amp_enabled = False
    elif amp_enabled: print("Automatic Mixed Precision (AMP) Enabled.")
    else: print("AMP Disabled.")

    device = torch.device(config.device)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.vis_dir, exist_ok=True)

    # Logging
    log_path = os.path.join(config.checkpoint_dir, config.log_file)
    log_f = open(log_path, 'a')
    def log_message(message):
        print(message); log_f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n"); log_f.flush()

    log_message("="*60); log_message("Starting Training Session (CompressAI Backend)"); log_message("="*60)
    log_message("--- Training Configuration ---")
    for key, value in vars(config).items():
         if not key.startswith('__'): log_message(f"{key}: {value}")
    log_message(f"AMP Enabled: {amp_enabled}"); log_message("-----------------------------")

    # --- Dataset and DataLoader ---
    log_message("Setting up dataset...")
    img_transforms_list = []
    if config.resize_height is not None and config.resize_width is not None:
        log_message(f"Frames resized to: {config.resize_height}x{config.resize_width}")
        img_transforms_list.append(transforms.Resize((config.resize_height, config.resize_width), antialias=True))
    else: log_message("Frames not resized.")
    img_transforms_list.append(transforms.ToTensor())
    img_transform = transforms.Compose(img_transforms_list)

    try:
        train_dataset = VideoFrameFlowDatasetNested(config.frame_base_dir, config.flow_base_dir, transform=img_transform)
        if len(train_dataset) == 0: log_message("ERROR: Dataset empty!"); log_f.close(); return
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                  num_workers=config.num_workers, pin_memory=(device.type == 'cuda'), drop_last=True)
        log_message(f"Dataset size: {len(train_dataset)} pairs. DataLoader: {len(train_loader)} batches/epoch.")
    except Exception as e:
         log_message(f"FATAL ERROR during Dataset/DataLoader setup: {e}"); traceback.print_exc(file=log_f); log_f.close(); return

    # --- Model ---
    log_message("Initializing Video Codec model...")
    model = VideoCodec(
        motion_latent_channels=config.motion_latent_channels, residual_latent_channels=config.residual_latent_channels,
        motion_hyper_channels=config.motion_hyper_channels, residual_hyper_channels=config.residual_hyper_channels,
        mcn_base_channels=config.mcn_base_channels, encoder_base_channels=config.encoder_base_channels,
        encoder_res_blocks=config.encoder_res_blocks, encoder_downsample_layers=config.encoder_downsample_layers,
        decoder_res_blocks=config.decoder_res_blocks, decoder_upsample_layers=config.decoder_upsample_layers
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_message(f"Model initialized with {num_params:,} trainable parameters.")
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        log_message(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)

    # --- Optimizer ---
    log_message("Setting up optimizer (AdamW)...")
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)

    # --- GradScaler ---
    scaler = GradScaler(enabled=amp_enabled)
    log_message(f"AMP GradScaler Initialized (Enabled: {amp_enabled})")

    # --- Checkpoint Loading ---
    start_epoch = 0
    best_metric = -float('inf')
    resume_path = config.resume
    if resume_path:
        if resume_path.lower() == 'latest':
            log_message("Attempting to resume from latest checkpoint...")
            resume_path = find_latest_checkpoint_file(config.checkpoint_dir)
            if resume_path: log_message(f"Found latest: {resume_path}")
            else: log_message("No 'latest' found. Starting fresh."); resume_path = None
        if resume_path and os.path.exists(resume_path):
            log_message(f"Resuming from: {resume_path}")
            model_to_load = model.module if isinstance(model, nn.DataParallel) else model
            best_metric, start_epoch = load_checkpoint(resume_path, model_to_load, optimizer, scaler, device)
            log_message(f"Resumed from epoch {start_epoch}. Prev best: {best_metric:.4f}")
        elif resume_path: log_message(f"Warn: Resume path not found: {resume_path}. Starting fresh.")
    else: log_message("No checkpoint specified. Starting fresh.")

    # --- Training Loop ---
    log_message(f"--- Starting Training from Epoch {start_epoch + 1} ---")
    total_batches = len(train_loader)

    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_loss, epoch_mse, epoch_bpp, epoch_psnr, epoch_msssim = 0.0, 0.0, 0.0, 0.0, 0.0
        processed_batches_count = 0
        batch_iter = tqdm(enumerate(train_loader), total=total_batches, desc=f"Epoch {epoch+1}/{config.epochs}", leave=True)
        start_time = time.time()
        last_batch_data_for_viz = None

        for i, batch_data in batch_iter:
            try:
                frame1_batch, frame2_batch, flow12_orig_res_batch = batch_data
                frame1 = frame1_batch.to(device, non_blocking=True)
                frame2 = frame2_batch.to(device, non_blocking=True)
                flow12_orig = flow12_orig_res_batch.to(device, non_blocking=True)

                B, C_f, H_frame, W_frame = frame1.shape
                if B == 0: continue
                B_f, C_flow, H_flow, W_flow = flow12_orig.shape
                if B != B_f: log_message("Warn: Batch size mismatch frames/flow. Skipping."); continue

                # Resize/Scale Flow
                if H_frame != H_flow or W_frame != W_flow:
                    flow12_resized = transforms.functional.resize(flow12_orig, [H_frame, W_frame], interpolation=transforms.InterpolationMode.BILINEAR, antialias=False)
                    scale_w = float(W_frame) / W_flow if W_flow > 0 else 1.0; scale_h = float(H_frame) / H_flow if H_flow > 0 else 1.0
                    flow12_scaled = torch.zeros_like(flow12_resized)
                    flow12_scaled[:, 0, :, :] = flow12_resized[:, 0, :, :] * scale_w
                    flow12_scaled[:, 1, :, :] = flow12_resized[:, 1, :, :] * scale_h
                    flow12 = flow12_scaled
                else: flow12 = flow12_orig

                num_pixels = H_frame * W_frame
                if num_pixels == 0: continue

                # Forward/Backward Pass
                optimizer.zero_grad()
                with autocast(enabled=amp_enabled):
                    outputs = model(frame1, frame2, flow12)
                    loss, mse, bpp = calculate_rd_loss(outputs, frame2, num_pixels, config.lambda_rd)
                    if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                        loss, mse, bpp = loss.mean(), mse.mean(), bpp.mean()

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if config.clip_max_norm > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_max_norm)
                scaler.step(optimizer)
                scaler.update()

                # Logging & Metrics
                batch_loss, batch_mse, batch_bpp = loss.item(), mse.item(), bpp.item()
                with torch.no_grad():
                    rec_frame, tgt_frame = outputs['frame2_reconstructed'].float(), frame2.float()
                    batch_psnr = compute_psnr(rec_frame, tgt_frame)
                    batch_msssim = compute_msssim_metric(rec_frame, tgt_frame)

                epoch_loss += batch_loss; epoch_mse += batch_mse; epoch_bpp += batch_bpp
                epoch_psnr += batch_psnr; epoch_msssim += batch_msssim
                processed_batches_count += 1

                if (i + 1) % config.print_freq == 0 or i == total_batches - 1:
                    lr = optimizer.param_groups[0]['lr']
                    batch_iter.set_postfix(Loss=f"{batch_loss:.4f}", MSE=f"{batch_mse:.6f}", BPP=f"{batch_bpp:.4f}",
                                           PSNR=f"{batch_psnr:.2f}", MSSSIM=f"{batch_msssim:.3f}", LR=f"{lr:.1e}")
                if i == total_batches - 1:
                    last_batch_data_for_viz = {'frame1': frame1.detach().cpu(), 'frame2': frame2.detach().cpu(),
                                               'frame2_reconstructed': outputs['frame2_reconstructed'].detach().cpu()}

            # Error Handling
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    log_message(f"\n ***** CUDA OOM (Epoch {epoch+1}, Batch {i}) *****"); log_message(" Skipping batch.")
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    optimizer.zero_grad(); continue
                else: log_message(f"\nERROR (Batch {i}, Epoch {epoch+1}): {e}"); traceback.print_exc(file=log_f); log_message(" Skipping batch."); optimizer.zero_grad(); continue
            except Exception as e: log_message(f"\nUNEXPECTED ERROR (Batch {i}, Epoch {epoch+1}): {e}"); traceback.print_exc(file=log_f); log_message(" Skipping batch."); optimizer.zero_grad(); continue

        # End of Epoch
        epoch_time = time.time() - start_time
        if processed_batches_count > 0:
            avg_loss, avg_mse, avg_bpp = epoch_loss/processed_batches_count, epoch_mse/processed_batches_count, epoch_bpp/processed_batches_count
            avg_psnr, avg_msssim = epoch_psnr/processed_batches_count, epoch_msssim/processed_batches_count
        else: log_message(f"WARN: Epoch {epoch+1} had 0 successful batches."); avg_loss, avg_mse, avg_bpp, avg_psnr, avg_msssim = 0,0,0,0,0

        log_message("-" * 60)
        log_message(f"Epoch {epoch+1}/{config.epochs} | Time: {epoch_time:.2f}s | Avg Loss: {avg_loss:.5f} | Avg MSE: {avg_mse:.7f} | Avg BPP: {avg_bpp:.5f} | Avg PSNR: {avg_psnr:.3f} dB | Avg MS-SSIM: {avg_msssim:.4f}")
        log_message("-" * 60)

        # Checkpointing
        current_metric = avg_psnr
        is_best = current_metric > best_metric
        if is_best and processed_batches_count > 0:
            best_metric = current_metric; log_message(f"*** New Best PSNR: {best_metric:.4f} ***")
        if processed_batches_count > 0 and ((epoch + 1) % config.save_freq == 0 or is_best):
            ckpt_state = {'epoch': epoch, 'state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                          'optimizer': optimizer.state_dict(), 'best_metric': best_metric, 'config': vars(config),
                          'scaler_state_dict': scaler.state_dict() if amp_enabled else None}
            ckpt_filename = f"checkpoint_epoch_{epoch+1:04d}.pth.tar"
            save_checkpoint(ckpt_state, is_best, config.checkpoint_dir, filename=ckpt_filename)

        # Visualization
        if last_batch_data_for_viz and plotting_available:
            visualize_epoch_end(epoch + 1, config.vis_dir, **last_batch_data_for_viz)
        last_batch_data_for_viz = None

    # End of Training
    log_message("="*60); log_message("--- Training Finished ---"); log_message(f"Best PSNR: {best_metric:.4f}"); log_message("="*60); log_f.close()


# --- Main Execution Block ---
if __name__ == "__main__":
    config = TrainConfig() # Load default config

    # Argument Parsing (remains the same, arguments update the config object)
    parser = argparse.ArgumentParser(description="Train Learned Video Codec (P-frame model w/ CompressAI)")
    parser.add_argument('--frame_base_dir', type=str, help=f"Frame sequences dir. Default: {config.frame_base_dir}")
    parser.add_argument('--flow_base_dir', type=str, help=f"Flow files dir. Default: {config.flow_base_dir}")
    parser.add_argument('--checkpoint_dir', type=str, help=f"Checkpoint dir. Default: {config.checkpoint_dir}")
    parser.add_argument('--vis_dir', type=str, help=f"Visualization dir. Default: {config.vis_dir}")
    parser.add_argument('--lambda_rd', type=float, help=f"R-D lambda. Default: {config.lambda_rd}")
    parser.add_argument('--lr', type=float, dest='learning_rate', help=f"Learning rate. Default: {config.learning_rate}")
    parser.add_argument('--batch_size', type=int, help=f"Batch size. Default: {config.batch_size}")
    parser.add_argument('--epochs', type=int, help=f"Epochs. Default: {config.epochs}")
    parser.add_argument('--resume', type=str, help="Checkpoint to resume from, or 'latest'. Default: None")
    parser.add_argument('--clip_max_norm', type=float, help=f"Gradient clipping max norm. Default: {config.clip_max_norm}")
    parser.add_argument('--seed', type=int, help=f"Random seed. Default: {config.seed}")
    parser.add_argument('--workers', type=int, dest='num_workers', help=f"Dataloader workers. Default: {config.num_workers}")
    parser.add_argument('--print_freq', type=int, help=f"Print frequency (batches). Default: {config.print_freq}")
    parser.add_argument('--save_freq', type=int, help=f"Save frequency (epochs). Default: {config.save_freq}")
    parser.add_argument('--no_amp', action='store_true', help="Disable AMP.")
    parser.add_argument('--resize_h', type=int, dest='resize_height', help="Resize height. Default: None")
    parser.add_argument('--resize_w', type=int, dest='resize_width', help="Resize width. Default: None")
    parser.add_argument('--motion_latent_ch', type=int, dest='motion_latent_channels', help=f"Motion latent channels. Default: {config.motion_latent_channels}")
    parser.add_argument('--residual_latent_ch', type=int, dest='residual_latent_channels', help=f"Residual latent channels. Default: {config.residual_latent_channels}")
    args = parser.parse_args()

    # Update Config from Args
    for key, value in vars(args).items():
        if value is not None: # Only update if arg was provided
            if key == 'no_amp': setattr(config, 'use_amp', not value)
            elif key == 'resize_height' or key == 'resize_width': continue # Handle resize pair below
            else: setattr(config, key, value)
    # Handle resize pair logic
    if args.resize_height is not None and args.resize_width is not None:
        config.resize_height, config.resize_width = args.resize_height, args.resize_width
    elif args.resize_height is not None or args.resize_width is not None:
         print("Warning: Both --resize_h and --resize_w must be provided. Resizing disabled.")
         config.resize_height, config.resize_width = None, None

    # Final Checks
    if not os.path.isdir(config.frame_base_dir): print(f"ERROR: Frame dir not found: '{config.frame_base_dir}'"); sys.exit(1)
    if not os.path.isdir(config.flow_base_dir): print(f"ERROR: Flow dir not found: '{config.flow_base_dir}'"); sys.exit(1)

    # Start Training
    main(config)