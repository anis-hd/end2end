import torch
import torch.nn as nn
try: import constriction
except ImportError: constriction = None
from modules import (
    Encoder, Decoder, Hyperprior, MotionCompensationNetwork, WarpingLayer
)


class VideoCodec(nn.Module):
    # ... (__init__ remains the same) ...
    def __init__(self, motion_latent_channels=128, residual_latent_channels=128,
                 motion_hyper_channels=128, residual_hyper_channels=128,
                 mcn_base_channels=32):
        super().__init__()
        self.motion_encoder = Encoder(input_channels=2, latent_channels=motion_latent_channels)
        self.motion_hyperprior = Hyperprior(latent_channels=motion_latent_channels, hyper_latent_channels=motion_hyper_channels)
        self.motion_decoder = Decoder(output_channels=2, latent_channels=motion_latent_channels, final_activation=None)
        self.warping_layer = WarpingLayer()
        self.motion_compensation_net = MotionCompensationNetwork(input_channels=3 + 2 + 3, output_channels=3, base_channels=mcn_base_channels)
        self.residual_encoder = Encoder(input_channels=3, latent_channels=residual_latent_channels)
        self.residual_hyperprior = Hyperprior(latent_channels=residual_latent_channels, hyper_latent_channels=residual_hyper_channels)
        self.residual_decoder = Decoder(output_channels=3, latent_channels=residual_latent_channels, final_activation=None)


    def forward(self, frame1, frame2, flow12): # Training forward pass (same)
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
        frame2_reconstructed = torch.clamp(frame2_reconstructed, 0.0, 1.0)
        # Return dict including rates for loss calculation
        return {
            'frame2_reconstructed': frame2_reconstructed,
            'rate_motion': rate_motion, 'rate_hyper_motion': rate_hyper_motion,
            'rate_residual': rate_residual, 'rate_hyper_residual': rate_hyper_residual,
            'flow_reconstructed': flow_reconstructed, # Keep intermediates if needed
            'frame2_motion_compensated': frame2_motion_compensated,
            'residual_reconstructed': residual_reconstructed,
        }


    @torch.no_grad()
    def compress_frame(self, frame1, frame2, flow12):
        """ Compresses a P-frame, returns data needed for bitstream writing. """
        if constriction is None: raise RuntimeError("Constriction library not found.")

        # --- Motion Compression ---
        motion_latents = self.motion_encoder(flow12)
        motion_compression_data = self.motion_hyperprior.compress(motion_latents)
        # motion_compression_data = { "main": (indices, cdf_l, cdf_u), "hyper": (indices, cdf_l, cdf_u) }

        # --- Need reconstructed flow to proceed ---
        # We need to simulate decompression locally to get residual
        # Option 1: Use the STE quantized values (faster, might mismatch decoder slightly)
        # quantized_motion_latents_ste, _, _ = self.motion_hyperprior.main_latent_entropy_model.quantizer(motion_latents)
        # Option 2: Full decompression simulation (matches decoder exactly but slower)
        # This requires implementing a dummy decoder or modifying decompress
        # Let's use the indices obtained from compress for consistency with true decoding
        quantized_motion_latents_rec = motion_compression_data["main"][0].float() # Use indices
        flow_reconstructed = self.motion_decoder(quantized_motion_latents_rec)

        # --- Motion Compensation & Residual Calculation ---
        warped_frame1 = self.warping_layer(frame1, flow_reconstructed)
        frame2_motion_compensated = self.motion_compensation_net(warped_frame1, flow_reconstructed, frame1)
        residual = frame2 - frame2_motion_compensated

        # --- Residual Compression ---
        residual_latents = self.residual_encoder(residual)
        residual_compression_data = self.residual_hyperprior.compress(residual_latents)

        # --- Return all necessary data ---
        return {
            "motion": motion_compression_data,
            "residual": residual_compression_data,
            "motion_latent_shape": list(motion_latents.shape), # Save shape info B,C,H,W
            "residual_latent_shape": list(residual_latents.shape)
        }

    @torch.no_grad()
    def decompress_frame(self, frame1, compressed_frame_data):
        """ Decompresses a P-frame using loaded indices/distributions from bitstream. """
        if constriction is None: raise RuntimeError("Constriction library not found.")

        # --- Retrieve data for this frame ---
        # In a real system, the entropy decoder reads from the bitstream.
        # Here, we simulate by directly using the data structure.
        motion_coder_backend = compressed_frame_data["motion_coder"] # Assume coder is passed in
        residual_coder_backend = compressed_frame_data["residual_coder"]
        motion_hyper_shape = compressed_frame_data["motion"]["hyper_shape"] # Need hyper shape
        residual_hyper_shape = compressed_frame_data["residual"]["hyper_shape"]

        # --- Motion Decompression ---
        # Need to pass hyper_shape, coder for hyper, coder for main
        quantized_motion_latents = self.motion_hyperprior.decompress(
            hyper_indices_shape=motion_hyper_shape,
            coder_hyper=motion_coder_backend["hyper"], # Pass appropriate coder instance
            coder_main=motion_coder_backend["main"]
        )
        flow_reconstructed = self.motion_decoder(quantized_motion_latents)

        # --- Motion Compensation ---
        warped_frame1 = self.warping_layer(frame1, flow_reconstructed)
        frame2_motion_compensated = self.motion_compensation_net(warped_frame1, flow_reconstructed, frame1)

        # --- Residual Decompression ---
        quantized_residual_latents = self.residual_hyperprior.decompress(
            hyper_indices_shape=residual_hyper_shape,
            coder_hyper=residual_coder_backend["hyper"],
            coder_main=residual_coder_backend["main"]
        )
        residual_reconstructed = self.residual_decoder(quantized_residual_latents)

        # --- Final Reconstruction ---
        frame2_reconstructed = frame2_motion_compensated + residual_reconstructed
        frame2_reconstructed = torch.clamp(frame2_reconstructed, 0.0, 1.0)

        return frame2_reconstructed