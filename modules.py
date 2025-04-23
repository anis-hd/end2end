import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions import Categorical
import math
from torch.autograd import Function


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

class GaussianConditionalEntropyModel(nn.Module):
    """
    Entropy model based on a conditional Gaussian distribution.
    Assumes latent variables follow N(mean, scale^2), where mean and scale
    are predicted by a hyperprior network.
    Calculates the rate (estimated bits) based on the PDF/CDF.
    Uses STE quantization noise approximation for rate calculation during training.
    """
    def __init__(self, scale_bound=0.11, likelihood_bound=1e-9):
        """
        Args:
            scale_bound (float): Lower bound for the scale parameter for numerical stability.
            likelihood_bound (float): Lower bound for likelihoods for numerical stability.
        """
        super().__init__()
        self.scale_bound = float(scale_bound)
        self.likelihood_bound = float(likelihood_bound)
        # Use QuantizerSTE internally for rate estimation during training
        self.quantizer = QuantizerSTE.apply

    def _likelihood(self, inputs, means, scales):
        """Calculates the likelihood of inputs under the Gaussian model."""
        # Ensure scales are bounded for stability
        scales = torch.clamp(scales, min=self.scale_bound)

        # Create Gaussian distribution object
        dist = Normal(means, scales)

        # Use STE quantization noise approximation during training
        # The 'inputs' here should be the *continuous* latents *before* hard quantization
        if self.training:
             # Approximate probability mass function P(x) by integrating PDF p(x) over [x-0.5, x+0.5]
             # P(x) = CDF(x + 0.5) - CDF(x - 0.5)
             upper = inputs + 0.5
             lower = inputs - 0.5
             cdf_upper = dist.cdf(upper)
             cdf_lower = dist.cdf(lower)
             likelihood_ = cdf_upper - cdf_lower
        else:
             # During inference, we use the actual quantized values (indices)
             # Need to calculate the probability mass for the specific integer index
             # 'inputs' in this case should be the integer indices
             upper = inputs + 0.5
             lower = inputs - 0.5
             cdf_upper = dist.cdf(upper)
             cdf_lower = dist.cdf(lower)
             likelihood_ = cdf_upper - cdf_lower

        # Bound likelihoods away from zero for numerical stability with log
        likelihood_ = torch.clamp(likelihood_, min=self.likelihood_bound)
        return likelihood_

    def forward(self, latents, means, scales):
        """
        Calculates rate and provides quantized values for the decoder.

        Args:
            latents (torch.Tensor): Continuous latent variables from the encoder.
            means (torch.Tensor): Predicted means from the hyper-decoder.
            scales (torch.Tensor): Predicted scales from the hyper-decoder.

        Returns:
            tuple:
                - quantized_latents (torch.Tensor): Latents quantized (rounded) for the decoder.
                - rate (torch.Tensor): Estimated bitrate (scalar tensor, sum over batch and dims).
                - likelihoods (torch.Tensor): Likelihood values (for debugging/analysis).
        """
        # 1. Quantize the latents using STE
        #    We need the indices for potential entropy coding, the dequantized value for the decoder,
        #    and the STE value for the gradient path.
        indices, quantized_for_decoder, latents_ste = self.quantizer(latents) # Pass continuous latents

        # 2. Calculate likelihoods
        #    During training, use the continuous 'latents' with the noise approx CDF calculation.
        #    During inference, use the rounded 'indices' with the CDF calculation.
        if self.training:
             likelihoods = self._likelihood(latents, means, scales) # Use continuous latents here
        else:
             likelihoods = self._likelihood(indices, means, scales) # Use indices here

        # 3. Calculate rate (-log2 likelihood)
        rate = -torch.log2(likelihoods).sum() # Sum over all elements to get total bits for batch

        # Normalize rate to bits per pixel (bpp) of the original image?
        # Often done by dividing by batch size and number of pixels in original image
        # rate = rate / (latents.shape[0] * num_pixels_in_original_image)
        # For now, return total bits for the batch. Normalization happens in loss function.

        # Return the values needed: quantized for decoder, rate, and likelihoods
        # Note: We use 'quantized_for_decoder' which comes from rounding the noisy latent in training
        # or rounding the original latent in inference.
        return quantized_for_decoder, rate, likelihoods


class FactorizedEntropyModel(nn.Module):
    """
    Entropy model for factorized priors (e.g., for hyper-latents).
    Models each latent variable independently.
    Here, we implement a simple version learning the range and assuming uniform dist.
    A more complex version would learn parameters for a Gaussian/Laplace.
    """
    def __init__(self, num_latents):
        super().__init__()
        # Could learn parameters here, e.g., for a Gaussian
        # self.mean = nn.Parameter(torch.zeros(num_latents))
        # self.log_scale = nn.Parameter(torch.zeros(num_latents))
        self.quantizer = QuantizerSTE.apply

        # Simple approach: Assume uniform within a learned range (or fixed range)
        # For simplicity, we won't learn parameters here, just calculate rate based on rounding.
        # We assume the values will be quantized integers, and the decoder uses these integers.
        # The rate estimation depends heavily on the assumed distribution.
        # A common simple assumption is a standard Gaussian N(0,1) prior,
        # which often works reasonably well if hyper-latents are normalized.

    def _likelihood_uniform_approx(self, inputs):
        # Crude approximation: Assume integers are roughly uniform over some range.
        # Probability = 1 / (number of possible integer values)
        # This isn't very accurate but avoids learning extra parameters.
        # A better approach uses Gaussian/Laplace CDFs as in GaussianConditional.
        # Let's use the Gaussian N(0,1) assumption as it's common.
        dist = Normal(torch.zeros_like(inputs), torch.ones_like(inputs))
        # Use STE noise approx for continuous inputs during training
        if self.training:
            upper = inputs + 0.5
            lower = inputs - 0.5
            cdf_upper = dist.cdf(upper)
            cdf_lower = dist.cdf(lower)
            likelihood_ = cdf_upper - cdf_lower
        else: # Inference: use integer indices
            upper = inputs + 0.5
            lower = inputs - 0.5
            cdf_upper = dist.cdf(upper)
            cdf_lower = dist.cdf(lower)
            likelihood_ = cdf_upper - cdf_lower

        likelihood_ = torch.clamp(likelihood_, min=1e-9) # Bound likelihoods
        return likelihood_


    def forward(self, latents):
        """
        Quantizes hyper-latents and estimates their rate assuming a factorized prior.

        Args:
            latents (torch.Tensor): Continuous hyper-latent variables.

        Returns:
            tuple:
                - quantized_latents (torch.Tensor): Quantized (rounded) hyper-latents.
                - rate (torch.Tensor): Estimated bitrate (scalar tensor).
        """
        indices, quantized_for_decoder, latents_ste = self.quantizer(latents)

        if self.training:
             likelihoods = self._likelihood_uniform_approx(latents) # Use continuous
        else:
             likelihoods = self._likelihood_uniform_approx(indices) # Use indices

        rate = -torch.log2(likelihoods).sum()

        return quantized_for_decoder, rate


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

        # Hyper-encoder: Latent -> Hyper-latent (often involves spatial downsampling)
        # Use abs() before hyper-encoder as sign is often handled separately or implicitly
        self.hyper_encoder = Encoder(
            input_channels=latent_channels,
            base_channels=latent_channels // 2, # Adjust base channels if needed
            latent_channels=hyper_latent_channels,
            num_downsample_layers=num_hyper_layers,
            num_res_blocks=0 # Typically simpler than main encoder
        )

        # Hyper-decoder: Hyper-latent -> Parameters (mean, scale) for main latent model
        # Output channels = 2 * latent_channels (one set for mean, one for scale)
        self.hyper_decoder = Decoder(
            output_channels=latent_channels * 2,
            base_channels=latent_channels // 2, # Match hyper-encoder intermediate channels
            latent_channels=hyper_latent_channels,
            num_upsample_layers=num_hyper_layers,
            num_res_blocks=0,
            final_activation=None # Parameters are handled separately
        )

        # Entropy model for the hyper-latents (factorized prior)
        self.hyper_latent_entropy_model = FactorizedEntropyModel(num_latents=hyper_latent_channels)

        # Main latent entropy model (conditional Gaussian)
        self.main_latent_entropy_model = GaussianConditionalEntropyModel()

    def forward(self, latents):
        """
        Forward pass through the hyperprior network.

        Args:
            latents (torch.Tensor): Main latent variables from the primary encoder.

        Returns:
            tuple:
                - quantized_latents_decoder (torch.Tensor): Main latents ready for the primary decoder.
                - rate_main (torch.Tensor): Rate associated with the main latents.
                - rate_hyper (torch.Tensor): Rate associated with the hyper-latents.
        """
        # 1. Encode main latents to hyper-latents
        #    Often take abs() as input, assuming symmetry or handling sign differently
        hyper_latents = self.hyper_encoder(torch.abs(latents))

        # 2. Quantize hyper-latents and estimate their rate
        quantized_hyper_latents, rate_hyper = self.hyper_latent_entropy_model(hyper_latents)

        # 3. Decode quantized hyper-latents to get parameters for main latent model
        params = self.hyper_decoder(quantized_hyper_latents)
        # Split params into mean and scale
        # Ensure shapes match the main latents
        means, scales_raw = torch.chunk(params, 2, dim=1) # Split along channel dimension

        # Apply activation to scales to ensure positivity and stability
        # Softplus or Exp + clamp are common. Let's use softplus here.
        scales = F.softplus(scales_raw) # Ensures scales > 0
        # Alternative: scales = torch.exp(scales_raw)
        # scales = torch.clamp(scales, min=self.main_latent_entropy_model.scale_bound) # Clamp if using exp

        # 4. Quantize main latents using the predicted parameters and estimate their rate
        quantized_latents_decoder, rate_main, _ = self.main_latent_entropy_model(latents, means, scales)

        return quantized_latents_decoder, rate_main, rate_hyper

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