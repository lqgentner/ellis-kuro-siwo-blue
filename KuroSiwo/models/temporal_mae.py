import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange

from vit_pytorch.vit import Transformer


class TemporalMAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio=0.75,
        decoder_depth=1,
        decoder_heads=8,
        decoder_dim_head=64,
        num_timepoints=3,  # Number of temporal images
    ):
        super().__init__()
        
        # Handle both single masking ratio and per-timepoint masking ratios
        if isinstance(masking_ratio, (list, tuple)):
            assert len(masking_ratio) == num_timepoints, \
                f"Number of masking ratios ({len(masking_ratio)}) must match num_timepoints ({num_timepoints})"
            for i, ratio in enumerate(masking_ratio):
                assert 0 < ratio < 1, f"Masking ratio at index {i} must be between 0 and 1, got {ratio}"
            self.masking_ratios = list(masking_ratio)
        else:
            assert 0 < masking_ratio < 1, "masking ratio must be kept between 0 and 1"
            self.masking_ratios = [masking_ratio] * num_timepoints
        
        self.num_timepoints = num_timepoints
        
        # Extract some hyperparameters and functions from encoder
        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.num_patches_per_timepoint = num_patches
        
        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])
        
        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]
        
        # Temporal embedding for each timepoint
        self.temporal_embedding = nn.Parameter(torch.randn(num_timepoints, encoder_dim))
        
        # Decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = (
            nn.Linear(encoder_dim, decoder_dim)
            if encoder_dim != decoder_dim
            else nn.Identity()
        )
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_dim=decoder_dim * 4,
        )
        # Decoder position embedding for all patches across all timepoints
        self.decoder_pos_emb = nn.Embedding(num_patches * num_timepoints, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, flood_img, pre_event_1, pre_event_2):
        device = flood_img.device
        batch = flood_img.shape[0]
        
        # Stack temporal images: [batch, num_timepoints, channels, height, width]
        temporal_imgs = torch.stack([flood_img, pre_event_1, pre_event_2], dim=1)
        
        # Process each temporal image to get patches
        all_patches = []
        all_tokens = []
        all_masked_indices = []
        all_unmasked_indices = []
        
        for t in range(self.num_timepoints):
            img = temporal_imgs[:, t]  # [batch, channels, height, width]
            patches = self.to_patch(img)  # [batch, num_patches, patch_dim]
            tokens = self.patch_to_emb(patches)  # [batch, num_patches, encoder_dim]
            
            # Add positional embeddings
            if self.encoder.pool == "cls":
                tokens += self.encoder.pos_embedding[:, 1 : (self.num_patches_per_timepoint + 1)]
            elif self.encoder.pool == "mean":
                tokens += self.encoder.pos_embedding.to(device, dtype=tokens.dtype)
            
            # Add temporal embedding
            tokens += self.temporal_embedding[t].unsqueeze(0).unsqueeze(0)
            
            all_patches.append(patches)
            all_tokens.append(tokens)
            
            # Apply per-timepoint masking
            num_masked_t = int(self.masking_ratios[t] * self.num_patches_per_timepoint)
            rand_indices_t = torch.rand(batch, self.num_patches_per_timepoint, device=device).argsort(dim=-1)
            
            # Adjust indices to account for timepoint offset
            offset = t * self.num_patches_per_timepoint
            masked_indices_t = rand_indices_t[:, :num_masked_t] + offset
            unmasked_indices_t = rand_indices_t[:, num_masked_t:] + offset
            
            all_masked_indices.append(masked_indices_t)
            all_unmasked_indices.append(unmasked_indices_t)
        
        # Concatenate all patches and tokens across time
        all_patches = torch.cat(all_patches, dim=1)  # [batch, num_patches * num_timepoints, patch_dim]
        all_tokens = torch.cat(all_tokens, dim=1)  # [batch, num_patches * num_timepoints, encoder_dim]
        
        # Concatenate masked and unmasked indices
        masked_indices = torch.cat(all_masked_indices, dim=1)  # [batch, total_masked]
        unmasked_indices = torch.cat(all_unmasked_indices, dim=1)  # [batch, total_unmasked]
        
        # Get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device=device)[:, None]
        unmasked_tokens = all_tokens[batch_range, unmasked_indices]
        
        # Get the patches to be masked for the final reconstruction loss
        masked_patches = all_patches[batch_range, masked_indices]
        
        # Encode with vision transformer
        encoded_tokens = self.encoder.transformer(unmasked_tokens)
        
        # Project encoder to decoder dimensions
        decoder_tokens = self.enc_to_dec(encoded_tokens)
        
        # Apply decoder position embedding to unmasked tokens
        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)
        
        # Create mask tokens with positional embeddings
        num_masked = masked_indices.shape[1]
        mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)
        
        # Prepare full sequence for decoder
        total_patches = self.num_patches_per_timepoint * self.num_timepoints
        full_decoder_tokens = torch.zeros(
            batch, total_patches, self.decoder_dim, device=device
        )
        full_decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        full_decoder_tokens[batch_range, masked_indices] = mask_tokens
        
        # Decode
        decoded_tokens = self.decoder(full_decoder_tokens)
        
        # Extract mask tokens and predict pixel values
        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)
        
        # Calculate reconstruction loss
        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        
        return recon_loss