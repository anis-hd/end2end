import torch
import torch.nn as nn
from modules import (
    Encoder, Decoder, Hyperprior, MotionCompensationNetwork, WarpingLayer
)

class VideoCodec(nn.Module):
    """
    End-to-End Learned Video Codec using Optical Flow.

    Combines Motion Autoencoder, Residual Autoencoder, Hyperpriors,
    and Motion Compensation Network.
    """
    def __init__(self, motion_latent_channels=128, residual_latent_channels=128,
                 motion_hyper_channels=128, residual_hyper_channels=128,
                 mcn_base_channels=32):
        """
        Args:
            motion_latent_channels (int): Channels in the motion latent space.
            residual_latent_channels (int): Channels in the residual latent space.
            motion_hyper_channels (int): Channels in the motion hyper-latent space.
            residual_hyper_channels (int): Channels in the residual hyper-latent space.
            mcn_base_channels (int): Base channels for the Motion Compensation Network.
        """
        super().__init__()

        # --- Motion Path ---
        # Optical flow has 2 channels (u, v)
        self.motion_encoder = Encoder(input_channels=2, latent_channels=motion_latent_channels)
        self.motion_hyperprior = Hyperprior(latent_channels=motion_latent_channels,
                                            hyper_latent_channels=motion_hyper_channels)
        self.motion_decoder = Decoder(output_channels=2, latent_channels=motion_latent_channels,
                                      final_activation=None) # Flow can be positive/negative

        # --- Motion Compensation ---
        self.warping_layer = WarpingLayer()
        # MCN input: warped_ref (3), flow (2), ref (3) -> 8 channels
        self.motion_compensation_net = MotionCompensationNetwork(
            input_channels=3 + 2 + 3,
            output_channels=3,
            base_channels=mcn_base_channels
        )

        # --- Residual Path ---
        # Residual has 3 channels (RGB difference)
        self.residual_encoder = Encoder(input_channels=3, latent_channels=residual_latent_channels)
        self.residual_hyperprior = Hyperprior(latent_channels=residual_latent_channels,
                                              hyper_latent_channels=residual_hyper_channels)
        self.residual_decoder = Decoder(output_channels=3, latent_channels=residual_latent_channels,
                                        final_activation=None) # Residuals centered around 0

    def forward(self, frame1, frame2, flow12):
        """
        Encodes and decodes a frame pair using motion and residual information.

        Args:
            frame1 (torch.Tensor): Reference frame (B, 3, H, W), range [0, 1].
            frame2 (torch.Tensor): Target frame (B, 3, H, W), range [0, 1].
            flow12 (torch.Tensor): Optical flow from frame1 to frame2 (B, 2, H, W).

        Returns:
            dict: Contains reconstructed frame, rates, and intermediate results.
                - 'frame2_reconstructed' (torch.Tensor): Final reconstructed target frame.
                - 'rate_motion' (torch.Tensor): Rate for motion latents.
                - 'rate_hyper_motion' (torch.Tensor): Rate for motion hyper-latents.
                - 'rate_residual' (torch.Tensor): Rate for residual latents.
                - 'rate_hyper_residual' (torch.Tensor): Rate for residual hyper-latents.
                - 'flow_reconstructed' (torch.Tensor): Reconstructed optical flow.
                - 'frame2_motion_compensated' (torch.Tensor): Motion compensated frame (output of MCN).
                - 'residual_reconstructed' (torch.Tensor): Reconstructed residual.
        """
        # --- Motion Compression ---
        motion_latents = self.motion_encoder(flow12)
        quantized_motion_latents, rate_motion, rate_hyper_motion = self.motion_hyperprior(motion_latents)
        flow_reconstructed = self.motion_decoder(quantized_motion_latents)

        # --- Motion Compensation ---
        warped_frame1 = self.warping_layer(frame1, flow_reconstructed)
        frame2_motion_compensated = self.motion_compensation_net(warped_frame1, flow_reconstructed, frame1)

        # --- Residual Calculation & Compression ---
        residual = frame2 - frame2_motion_compensated
        residual_latents = self.residual_encoder(residual)
        quantized_residual_latents, rate_residual, rate_hyper_residual = self.residual_hyperprior(residual_latents)
        residual_reconstructed = self.residual_decoder(quantized_residual_latents)

        # --- Final Reconstruction ---
        frame2_reconstructed = frame2_motion_compensated + residual_reconstructed
        # Clip final frame to valid range (e.g., [0, 1])
        frame2_reconstructed = torch.clamp(frame2_reconstructed, 0.0, 1.0)

        return {
            'frame2_reconstructed': frame2_reconstructed,
            'rate_motion': rate_motion,
            'rate_hyper_motion': rate_hyper_motion,
            'rate_residual': rate_residual,
            'rate_hyper_residual': rate_hyper_residual,
            'flow_reconstructed': flow_reconstructed,
            'frame2_motion_compensated': frame2_motion_compensated,
            'residual_reconstructed': residual_reconstructed,
        }

    def compress(self, frame1, frame2, flow12):
        """ Encodes frames and returns quantized indices (simulating bitstream generation). """
        # --- Motion Path ---
        motion_latents = self.motion_encoder(flow12)
        # Need access to internal hyperprior steps to get indices
        motion_hyper_latents = self.motion_hyperprior.hyper_encoder(torch.abs(motion_latents))
        quantized_motion_hyper_indices, _ = self.motion_hyperprior.hyper_latent_entropy_model(motion_hyper_latents) # Use forward to get indices

        params_motion = self.motion_hyperprior.hyper_decoder(quantized_motion_hyper_indices) # Decode from indices
        means_motion, scales_raw_motion = torch.chunk(params_motion, 2, dim=1)
        scales_motion = F.softplus(scales_raw_motion)

        # Get main motion indices using the conditional model (needs modification to output indices)
        # For now, let's get them from the forward pass simulation (less efficient but demonstrates principle)
        _, _, latents_ste_motion = self.motion_hyperprior.main_latent_entropy_model.quantizer(motion_latents)
        quantized_motion_indices = torch.round(latents_ste_motion) # Inferred indices

        # --- Residual Path ---
        # Need reconstructed flow to calculate residual
        quantized_motion_latents_dec = self.motion_hyperprior.main_latent_entropy_model(motion_latents, means_motion, scales_motion)[0]
        flow_reconstructed = self.motion_decoder(quantized_motion_latents_dec)
        warped_frame1 = self.warping_layer(frame1, flow_reconstructed)
        frame2_motion_compensated = self.motion_compensation_net(warped_frame1, flow_reconstructed, frame1)
        residual = frame2 - frame2_motion_compensated

        residual_latents = self.residual_encoder(residual)
        residual_hyper_latents = self.residual_hyperprior.hyper_encoder(torch.abs(residual_latents))
        quantized_residual_hyper_indices, _ = self.residual_hyperprior.hyper_latent_entropy_model(residual_hyper_latents)

        params_residual = self.residual_hyperprior.hyper_decoder(quantized_residual_hyper_indices)
        means_residual, scales_raw_residual = torch.chunk(params_residual, 2, dim=1)
        scales_residual = F.softplus(scales_raw_residual)

        _, _, latents_ste_residual = self.residual_hyperprior.main_latent_entropy_model.quantizer(residual_latents)
        quantized_residual_indices = torch.round(latents_ste_residual)

        # Actual implementation would use an Arithmetic Coder here with the distributions
        # to generate real bitstreams from these indices and distributions.
        return {
            'motion_indices': quantized_motion_indices,
            'motion_hyper_indices': quantized_motion_hyper_indices,
            'residual_indices': quantized_residual_indices,
            'residual_hyper_indices': quantized_residual_hyper_indices
        }

    def decompress(self, frame1, compressed_data):
        """ Decodes from quantized indices (simulating bitstream reading). """
        # Retrieve indices (in practice, decode from bitstream using entropy decoder)
        quantized_motion_indices = compressed_data['motion_indices']
        quantized_motion_hyper_indices = compressed_data['motion_hyper_indices']
        quantized_residual_indices = compressed_data['residual_indices']
        quantized_residual_hyper_indices = compressed_data['residual_hyper_indices']

        # --- Motion Path ---
        # Use hyper indices to get parameters for main motion latents
        params_motion = self.motion_hyperprior.hyper_decoder(quantized_motion_hyper_indices.float()) # Need float
        # Decode main motion latents (these are already dequantized indices)
        flow_reconstructed = self.motion_decoder(quantized_motion_indices.float())

        # --- Motion Compensation ---
        warped_frame1 = self.warping_layer(frame1, flow_reconstructed)
        frame2_motion_compensated = self.motion_compensation_net(warped_frame1, flow_reconstructed, frame1)

        # --- Residual Path ---
        # Decode main residual latents
        residual_reconstructed = self.residual_decoder(quantized_residual_indices.float())

        # --- Final Reconstruction ---
        frame2_reconstructed = frame2_motion_compensated + residual_reconstructed
        frame2_reconstructed = torch.clamp(frame2_reconstructed, 0.0, 1.0)

        return frame2_reconstructed