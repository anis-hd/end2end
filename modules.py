import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions import Categorical
import math
from torch.autograd import Function
import constriction

# ==============================================================================
# Helper Modules & Functions
# ==============================================================================

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
        self.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, bias=bias))
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

# ==============================================================================
# Core Autoencoder Components
# ==============================================================================

class Encoder(nn.Module):
    """Generic CNN Encoder with downsampling."""
    def __init__(self, input_channels, base_channels=64, latent_channels=128, num_downsample_layers=3, num_res_blocks=2):
        """
        Args:
            input_channels (int): Number of channels in the input tensor.
            base_channels (int): Number of channels after the first convolution.
            latent_channels (int): Number of channels in the output latent tensor.
            num_downsample_layers (int): Number of downsampling stages.
            num_res_blocks (int): Number of residual blocks after downsampling.
        """
        super().__init__()
        layers = []
        current_channels = input_channels

        # Initial convolution
        layers.append(ConvNormAct(current_channels, base_channels, kernel_size=5, stride=1))
        current_channels = base_channels

        # Downsampling layers
        for i in range(num_downsample_layers):
            out_ch = current_channels * 2
            layers.append(ConvNormAct(current_channels, out_ch, kernel_size=3, stride=2, padding=1))
            current_channels = out_ch

        # Residual blocks
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(current_channels))

        # Final convolution to latent space
        # No norm or activation here, typically done before quantization or in hyperprior
        layers.append(nn.Conv2d(current_channels, latent_channels, kernel_size=3, stride=1, padding='same'))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    """Generic CNN Decoder with upsampling (symmetric to Encoder)."""
    def __init__(self, output_channels, base_channels=64, latent_channels=128, num_upsample_layers=3, num_res_blocks=2, final_activation=None):
        """
        Args:
            output_channels (int): Number of channels in the reconstructed output.
            base_channels (int): Number of channels targeted before the final convolution (must match Encoder).
            latent_channels (int): Number of channels in the input latent tensor.
            num_upsample_layers (int): Number of upsampling stages (must match num_downsample_layers).
            num_res_blocks (int): Number of residual blocks before upsampling.
            final_activation (str | None): Name of the final activation function (e.g., 'sigmoid', 'tanh', None).
        """
        super().__init__()
        layers = []
        # Initial convolution from latent space
        # Usually apply norm/act here to start the decoding process
        current_channels = latent_channels * (2**num_upsample_layers) # Channels before upsampling starts
        layers.append(ConvNormAct(latent_channels, current_channels, kernel_size=3, stride=1))

        # Residual blocks
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(current_channels))

        # Upsampling layers
        for i in range(num_upsample_layers):
            out_ch = current_channels // 2
            layers.append(ConvTransposeNormAct(current_channels, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1))
            current_channels = out_ch
            # # Alternative: PixelShuffle
            # layers.append(nn.Conv2d(current_channels, current_channels * 4, kernel_size=3, padding=1))
            # layers.append(nn.PixelShuffle(2))
            # layers.append(nn.BatchNorm2d(current_channels)) # Norm and Act after PixelShuffle
            # layers.append(nn.LeakyReLU(0.2, inplace=True))


        # Final convolution to output channels
        layers.append(nn.Conv2d(current_channels, output_channels, kernel_size=5, stride=1, padding='same'))

        # Final activation
        if final_activation:
            layers.append(get_activation(final_activation))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

# ==============================================================================
# Quantization
# ==============================================================================
# *** Change inheritance from nn.Module to torch.autograd.Function ***
class QuantizerSTE(Function):
    """
    Scalar Quantizer with Straight-Through Estimator (STE).
    Uses additive uniform noise during training and rounding during inference.
    Defined as a custom autograd Function.
    """
    # *** Remove __init__ and super().__init__() - not needed for autograd.Function ***
    # def __init__(self):
    #     super().__init__()

    @staticmethod
    def forward(ctx, inputs):
        """
        Forward pass: Adds noise in training, rounds in inference.
        Returns indices (for entropy coding), dequantized values (for decoder),
        and STE tensor (for backward pass).
        """
        if ctx.needs_input_grad[0]: # Check if gradient calculation is needed
            # Training or requires grad: Add uniform noise for STE
            noise = torch.rand_like(inputs) - 0.5
            quantized = inputs + noise
            # STE: gradient passes through as if it was identity
            # Detach quantized from graph regarding noise addition, attach to inputs grad path
            quantized_ste = inputs + (quantized - inputs).detach()
            # Apply rounding (or floor/ceil depending on convention, round is common)
            indices = torch.round(quantized_ste) # These are the values sent to the entropy coder
            quantized_final = indices # Output for decoder path in training uses rounded noisy values
        else:
            # Inference or no grad needed: use simple rounding
            indices = torch.round(inputs)
            quantized_ste = indices # Gradient path not needed, just pass rounded values
            quantized_final = indices

        # Use indices as the quantized representation (for entropy coding)
        # Use quantized_final as the input to the decoder (dequantized)
        # Use quantized_ste for the backward pass during training
        ctx.save_for_backward(inputs) # Not strictly needed for STE noise but good practice

        # Return indices for entropy coding and dequantized values for decoder
        # Also return STE version for backward pass during training
        # Make sure returned tensors don't carry unnecessary grad history from noise
        return indices.detach(), quantized_final.detach(), quantized_ste # Ensure indices/final don't keep grad path

    @staticmethod
    def backward(ctx, grad_indices, grad_quantized_final, grad_ste):
        """
        Backward pass: STE passes the gradient from the STE output (`grad_ste`)
        straight through to the original inputs.
        Gradients for `indices` and `quantized_final` are ignored.
        """
        # Check if input needs gradient
        if not ctx.needs_input_grad[0]:
             return None # Return None if input doesn't require grad

        # Pass gradient from grad_ste directly to the input
        # Gradients for indices and quantized_final are ignored as they are detached
        return grad_ste # Pass gradient straight through


class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantizer implementation using Exponential Moving Averages (EMA) for codebook updates.
    Not used in the final example but kept for reference if VQ is desired later.
    """
    # ... (Implementation would go here - complex, involves finding nearest neighbors, commitment loss, EMA updates)
    # For simplicity, we stick to Scalar Quantization with STE.
    pass

# ==============================================================================
# Entropy Models
# ==============================================================================

# --- Entropy Models (Modified for Constriction) ---

class GaussianConditionalEntropyModel(nn.Module):
    """
    Entropy model using a conditional Gaussian distribution.
    MODIFIED to work with external entropy coders like Constriction.
    Provides necessary distributions/parameters for encoding/decoding.
    """
    def __init__(self, scale_bound=0.11, likelihood_bound=1e-9, cdf_bound=1e-6):
        super().__init__()
        self.scale_bound = float(scale_bound)
        self.likelihood_bound = float(likelihood_bound) # For rate calculation stability
        self.cdf_bound = float(cdf_bound) # For entropy coder stability (avoid 0 or 1 CDF values)
        self.quantizer = QuantizerSTE.apply # Keep for training gradient

    def _get_gaussian_params(self, means, scales_raw):
        """ Calculates clamped means and scales """
        # Clamp scales for stability during distribution creation AND for entropy coding
        scales = F.softplus(scales_raw) # Use softplus for positivity
        scales = torch.clamp(scales, min=self.scale_bound)
        # Means typically don't need clamping unless there are specific range issues
        return means, scales

    def _get_cdf_endpoints(self, inputs_quantized, means, scales):
        """ Calculates CDF values needed for arithmetic coding. """
        # Inputs here are the *quantized integer indices*
        # We need CDF(x + 0.5) and CDF(x - 0.5)
        dist = Normal(means, scales)
        upper = inputs_quantized + 0.5
        lower = inputs_quantized - 0.5
        cdf_upper = dist.cdf(upper)
        cdf_lower = dist.cdf(lower)

        # Clamp CDF values slightly away from 0 and 1 for numerical stability in entropy coders
        cdf_upper = torch.clamp(cdf_upper, self.cdf_bound, 1.0 - self.cdf_bound)
        cdf_lower = torch.clamp(cdf_lower, self.cdf_bound, 1.0 - self.cdf_bound)

        # Ensure lower < upper after clamping (important!)
        # If cdf_lower >= cdf_upper, set upper slightly higher
        mask = cdf_lower >= cdf_upper
        cdf_upper = torch.where(mask, cdf_lower + self.cdf_bound * 10, cdf_upper) # Add small gap

        return cdf_lower, cdf_upper

    def forward(self, latents, means_pred, scales_pred_raw):
        """
        Forward pass during TRAINING. Quantizes and calculates rate estimate.

        Args:
            latents (torch.Tensor): Continuous latent variables from the encoder.
            means_pred (torch.Tensor): Predicted means from the hyper-decoder.
            scales_pred_raw (torch.Tensor): Predicted raw scales from the hyper-decoder.

        Returns:
            tuple:
                - quantized_latents_decoder (torch.Tensor): Latents ready for the primary decoder (from STE).
                - rate (torch.Tensor): Estimated bitrate (scalar tensor, sum over batch and dims).
        """
        # 1. Get stable means and scales
        means, scales = self._get_gaussian_params(means_pred, scales_pred_raw)

        # 2. Quantize using STE for training backward pass
        indices, quantized_for_decoder, _ = self.quantizer(latents)

        # 3. Estimate likelihood using CDFs (noise approximation) on *continuous* latents
        dist = Normal(means, scales)
        upper = latents + 0.5
        lower = latents - 0.5
        cdf_upper = dist.cdf(upper)
        cdf_lower = dist.cdf(lower)
        likelihoods = torch.clamp(cdf_upper - cdf_lower, min=self.likelihood_bound)

        # 4. Calculate rate estimate
        rate = -torch.log2(likelihoods).sum()

        return quantized_for_decoder, rate


    @torch.no_grad() # No gradients needed for compression/decompression logic
    def compress(self, latents, means_pred, scales_pred_raw):
        """
        Prepares data for entropy encoding.

        Args:
            latents (torch.Tensor): Continuous latent variables.
            means_pred (torch.Tensor): Predicted means.
            scales_pred_raw (torch.Tensor): Predicted raw scales.

        Returns:
            tuple:
                - indices (torch.Tensor): Quantized integer indices.
                - cdf_lower (torch.Tensor): Lower CDF endpoints for entropy coder.
                - cdf_upper (torch.Tensor): Upper CDF endpoints for entropy coder.
        """
        means, scales = self._get_gaussian_params(means_pred, scales_pred_raw)
        # Quantize for real (rounding)
        indices = torch.round(latents).short() # Use short for less memory if range allows
        # Get CDFs based on the *integer indices*
        cdf_lower, cdf_upper = self._get_cdf_endpoints(indices.float(), means, scales) # Need float for dist.cdf
        return indices, cdf_lower, cdf_upper

    @torch.no_grad()
    def decompress(self, means_pred, scales_pred_raw, coder_backend):
        """
        Decodes indices from the bitstream using the provided entropy coder backend.

        Args:
            means_pred (torch.Tensor): Predicted means.
            scales_pred_raw (torch.Tensor): Predicted raw scales.
            coder_backend (constriction.stream.base.EntropyCoderBase): An initialized
                                     Constriction entropy decoder (e.g., ANS or Range Coder).

        Returns:
            torch.Tensor: Dequantized latent variables (float tensor).
        """
        if constriction is None:
            raise RuntimeError("Constriction library not found. Cannot decompress.")

        means, scales = self._get_gaussian_params(means_pred, scales_pred_raw)

        # Need shape information to decode the correct number of symbols
        B, C, H, W = means.shape
        flat_shape = (B * C * H * W,) # Total number of symbols

        # Flatten parameters for symbol-by-symbol decoding
        flat_means = means.reshape(flat_shape)
        flat_scales = scales.reshape(flat_shape)

        # Pre-allocate tensor for decoded symbols
        decoded_indices_flat = torch.empty_like(flat_means, dtype=torch.short) # Decode as short integers

        # Decode symbol by symbol (can be slow, batch decoding might be possible with some backends)
        # print(f"Decoding {flat_shape[0]} symbols...") # Debug
        for i in range(flat_shape[0]):
            # Get distribution parameters for the current symbol
            m, s = flat_means[i], flat_scales[i]
            # Define the Quantized Gaussian distribution for the coder
            # Need integer support range. Assume a reasonable range based on typical latent values.
            # This needs careful consideration! Let's assume range [-32, 32] for now. Adapt as needed.
            min_symbol = -32
            max_symbol = 32
            symbols = torch.arange(min_symbol, max_symbol + 1, device=means.device) # Possible integer values
            cdf_lower, cdf_upper = self._get_cdf_endpoints(symbols.float(), m.expand_as(symbols), s.expand_as(symbols))
            # Ensure PMF sums close to 1 (optional check)
            # pmf = cdf_upper - cdf_lower
            # print(f"Symbol {i} PMF sum: {pmf.sum()}")
            # Decode one symbol using the calculated CDFs
            decoded_symbol = coder_backend.decode_symbol(cdf_lower.cpu().numpy(), cdf_upper.cpu().numpy()) # Coder expects numpy CPU arrays
            decoded_indices_flat[i] = decoded_symbol + min_symbol # Adjust index based on assumed min_symbol

        # Reshape back to original latent shape
        decoded_latents = decoded_indices_flat.reshape(B, C, H, W)

        # Return as float tensor (dequantized)
        return decoded_latents.float()


class FactorizedEntropyModel(nn.Module):
    """
    Entropy model for factorized priors (e.g., hyper-latents).
    MODIFIED for Constriction. Assumes N(0, 1) prior for simplicity.
    """
    def __init__(self, cdf_bound=1e-6):
        super().__init__()
        self.quantizer = QuantizerSTE.apply # For training
        self.cdf_bound = cdf_bound
        # Precompute N(0,1) distribution on the correct device (once)
        self.register_buffer("prior_mean", torch.tensor(0.0))
        self.register_buffer("prior_scale", torch.tensor(1.0))
        # self.prior = Normal(torch.tensor(0.0), torch.tensor(1.0)) # This won't auto-move device


    def _get_cdf_endpoints(self, inputs_quantized):
        """ Calculates CDF values assuming a N(0,1) prior. """
        # Ensure prior parameters are on the same device as input
        prior = Normal(self.prior_mean.to(inputs_quantized.device),
                       self.prior_scale.to(inputs_quantized.device))
        upper = inputs_quantized + 0.5
        lower = inputs_quantized - 0.5
        cdf_upper = prior.cdf(upper)
        cdf_lower = prior.cdf(lower)
        cdf_upper = torch.clamp(cdf_upper, self.cdf_bound, 1.0 - self.cdf_bound)
        cdf_lower = torch.clamp(cdf_lower, self.cdf_bound, 1.0 - self.cdf_bound)
        # Ensure lower < upper
        mask = cdf_lower >= cdf_upper
        cdf_upper = torch.where(mask, cdf_lower + self.cdf_bound * 10, cdf_upper)
        return cdf_lower, cdf_upper

    def forward(self, latents):
        """
        Forward pass during TRAINING. Quantizes and calculates rate estimate.
        """
        indices, quantized_for_decoder, _ = self.quantizer(latents)
        # Estimate likelihood using CDFs (noise approx) on *continuous* latents N(0,1)
        prior = Normal(self.prior_mean.to(latents.device), self.prior_scale.to(latents.device))
        upper = latents + 0.5; lower = latents - 0.5
        likelihoods = torch.clamp(prior.cdf(upper) - prior.cdf(lower), min=1e-9) # Use likelihood_bound here
        rate = -torch.log2(likelihoods).sum()
        return quantized_for_decoder, rate

    @torch.no_grad()
    def compress(self, latents):
        """ Prepares data for entropy encoding. """
        indices = torch.round(latents).short()
        cdf_lower, cdf_upper = self._get_cdf_endpoints(indices.float())
        return indices, cdf_lower, cdf_upper

    @torch.no_grad()
    def decompress(self, shape, coder_backend):
        """ Decodes indices assuming N(0,1) prior. """
        if constriction is None: raise RuntimeError("Constriction library not found.")
        B, C, H, W = shape
        flat_shape = (B * C * H * W,)
        device = self.prior_mean.device # Use device where prior is registered

        decoded_indices_flat = torch.empty(flat_shape, dtype=torch.short, device=device)

        # Define N(0,1) distribution once
        # Assume same quantization range as before for symbol decoding
        min_symbol = -32; max_symbol = 32
        symbols = torch.arange(min_symbol, max_symbol + 1, device=device)
        cdf_lower, cdf_upper = self._get_cdf_endpoints(symbols.float())
        cdf_lower_np = cdf_lower.cpu().numpy(); cdf_upper_np = cdf_upper.cpu().numpy()

        for i in range(flat_shape[0]):
            decoded_symbol = coder_backend.decode_symbol(cdf_lower_np, cdf_upper_np)
            decoded_indices_flat[i] = decoded_symbol + min_symbol

        decoded_latents = decoded_indices_flat.reshape(B, C, H, W)
        return decoded_latents.float()


# ==============================================================================
# Hyperprior Network
# ==============================================================================


class Hyperprior(nn.Module):
    """
    Hyperprior network comprising a hyper-encoder and hyper-decoder.
    Encodes main latents to hyper-latents and decodes hyper-latents
    to parameters (mean, scale) for the main latent entropy model.
    Includes quantization and entropy model for the hyper-latents themselves.
    """
    def __init__(self, latent_channels, hyper_latent_channels=128, num_hyper_layers=2):
        """
        Args:
            latent_channels (int): Number of channels in the main latent space.
            hyper_latent_channels (int): Number of channels in the hyper-latent space.
            num_hyper_layers (int): Number of down/up sampling layers in the hyper network.
        """
        super().__init__()
        self.latent_channels = latent_channels
        self.hyper_latent_channels = hyper_latent_channels

        # --- CORRECTED INITIALIZATION ---
        # Hyper-encoder: Latent -> Hyper-latent
        self.hyper_encoder = Encoder(
            input_channels=latent_channels,
            base_channels=max(32, latent_channels // 2), # Ensure base_channels is reasonable
            latent_channels=hyper_latent_channels,
            num_downsample_layers=num_hyper_layers,
            num_res_blocks=0 # Simpler hyper network
        )

        # Hyper-decoder: Hyper-latent -> Parameters (mean, scale)
        # Calculate the base channels before upsampling starts in the decoder
        # It should match the output channels of the hyper_encoder's downsampling part
        hyper_encoder_output_channels = max(32, latent_channels // 2) * (2**num_hyper_layers)

        self.hyper_decoder = Decoder(
            output_channels=latent_channels * 2, # Means + Scales for main latents
            # Use the calculated channel count from the encoder stage
            # This assumes the decoder structure mirrors the encoder symmetrically
            base_channels=hyper_encoder_output_channels, # Base channels before upsampling
            latent_channels=hyper_latent_channels, # Input channels from hyper-latent space
            num_upsample_layers=num_hyper_layers,
            num_res_blocks=0, # Simpler hyper network
            final_activation=None
        )
        # --- END CORRECTION ---


        # Entropy model for the hyper-latents (factorized prior)
        self.hyper_latent_entropy_model = FactorizedEntropyModel()

        # Main latent entropy model (conditional Gaussian)
        self.main_latent_entropy_model = GaussianConditionalEntropyModel()

    def forward(self, latents): # Training forward pass
        # ... (forward implementation using self.hyper_encoder, etc.) ...
        hyper_latents = self.hyper_encoder(torch.abs(latents))
        quantized_hyper_latents, rate_hyper = self.hyper_latent_entropy_model(hyper_latents)
        params = self.hyper_decoder(quantized_hyper_latents)
        means, scales_raw = torch.chunk(params, 2, dim=1)
        quantized_latents_decoder, rate_main = self.main_latent_entropy_model(latents, means, scales_raw)
        return quantized_latents_decoder, rate_main, rate_hyper


    @torch.no_grad()
    def compress(self, latents):
        # ... (compress implementation - should be correct if using updated entropy models) ...
        hyper_latents = self.hyper_encoder(torch.abs(latents))
        hyper_indices, hyper_cdf_lower, hyper_cdf_upper = self.hyper_latent_entropy_model.compress(hyper_latents)
        params = self.hyper_decoder(hyper_indices.float())
        means, scales_raw = torch.chunk(params, 2, dim=1)
        main_indices, main_cdf_lower, main_cdf_upper = self.main_latent_entropy_model.compress(latents, means, scales_raw)
        return {
            "main": (main_indices, main_cdf_lower, main_cdf_upper),
            "hyper": (hyper_indices, hyper_cdf_lower, hyper_cdf_upper)
        }


    @torch.no_grad()
    def decompress(self, hyper_indices_shape, coder_hyper, coder_main):
        # ... (decompress implementation - should be correct if using updated entropy models) ...
        quantized_hyper_latents = self.hyper_latent_entropy_model.decompress(hyper_indices_shape, coder_hyper)
        params = self.hyper_decoder(quantized_hyper_latents)
        means, scales_raw = torch.chunk(params, 2, dim=1)
        quantized_main_latents = self.main_latent_entropy_model.decompress(means, scales_raw, coder_main)
        return quantized_main_latents
# ==============================================================================
# Motion Compensation & Warping
# ==============================================================================

class WarpingLayer(nn.Module):
    """ Warps an image using optical flow using F.grid_sample. """
    def __init__(self):
        super().__init__()

    def forward(self, x, flow):
        """
        Args:
            x (torch.Tensor): Image to warp (B, C, H, W), typically reference frame.
            flow (torch.Tensor): Optical flow (B, 2, H, W), where flow[:, 0, :, :] is horizontal (u)
                                 and flow[:, 1, :, :] is vertical (v).
                                 Flow indicates where each pixel *came from*.

        Returns:
            torch.Tensor: Warped image (B, C, H, W).
        """
        B, C, H, W = x.size()

        # Create base grid coordinates [-1, 1]
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype),
                                        torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype),
                                        indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0).repeat(B, 1, 1, 1) # (B, H, W, 2)

        # Normalize flow to [-1, 1] range relative to grid dimensions
        # Flow (u, v) needs to be scaled: u maps to grid_x range (width), v maps to grid_y range (height)
        # Normalized offset = flow / (size / 2)
        norm_flow_x = flow[:, 0, :, :] / ((W - 1) / 2) # (B, H, W)
        norm_flow_y = flow[:, 1, :, :] / ((H - 1) / 2) # (B, H, W)
        norm_flow = torch.stack((norm_flow_x, norm_flow_y), dim=3) # (B, H, W, 2)

        # Calculate the sampling grid: where each output pixel should sample from the input
        # If flow(x, y) = (u, v), the pixel at (x, y) came from (x+u, y+v).
        # grid_sample wants the normalized coordinate to sample *from*.
        # So, sampling_grid(x, y) = grid(x, y) + normalized_flow(x, y)
        sampling_grid = grid + norm_flow # (B, H, W, 2)

        # Warp using grid_sample
        # Ensure sampling grid is within [-1, 1] or use padding_mode
        # 'border' replicates edge pixels, 'zeros' pads with zeros
        # 'reflection' reflects the image
        warped_x = F.grid_sample(x, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)

        return warped_x


class MotionCompensationNetwork(nn.Module):
    """
    Refines the warped reference frame using a CNN.
    Takes warped frame, flow, and optionally the original reference frame as input.
    """
    def __init__(self, input_channels=3 + 2 + 3, output_channels=3, base_channels=32, num_res_blocks=3):
        """
        Args:
            input_channels (int): Channels of the concatenated input (e.g., warped_ref, flow, ref). Default assumes 3+2+3=8.
            output_channels (int): Channels of the output (typically 3 for RGB).
            base_channels (int): Base number of channels in the CNN.
            num_res_blocks (int): Number of residual blocks.
        """
        super().__init__()
        layers = []
        layers.append(ConvNormAct(input_channels, base_channels, kernel_size=5))

        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(base_channels))

        # Output convolution - could output refined frame directly or a refinement map
        # Let's output the refined frame directly
        layers.append(nn.Conv2d(base_channels, output_channels, kernel_size=5, padding='same'))
        # Maybe add a final activation? Tanh or Sigmoid if output range is known?
        # Or leave linear if it predicts the frame directly without range constraints.
        # Adding Sigmoid assuming input frames are [0, 1]
        layers.append(nn.Sigmoid())


        self.network = nn.Sequential(*layers)

    def forward(self, warped_ref, flow, ref_frame=None):
        """
        Args:
            warped_ref (torch.Tensor): The reference frame warped by the reconstructed flow (B, C, H, W).
            flow (torch.Tensor): The reconstructed flow used for warping (B, 2, H, W).
            ref_frame (torch.Tensor, optional): The original reference frame (B, C, H, W).

        Returns:
            torch.Tensor: The refined motion-compensated frame (B, C, H, W).
        """
        # Concatenate inputs along the channel dimension
        if ref_frame is not None:
             mcn_input = torch.cat([warped_ref, flow, ref_frame], dim=1)
        else:
             # Alternative: Only use warped frame and flow
             mcn_input = torch.cat([warped_ref, flow], dim=1)
             # Adjust input_channels in __init__ if changing this default behaviour


        refined_frame = self.network(mcn_input)
        return refined_frame