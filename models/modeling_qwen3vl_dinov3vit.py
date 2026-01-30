# coding=utf-8
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3VL-DINOv3 mixed model implementation"""

from typing import List, Optional, Tuple, Union, Any, Callable
from transformers.processing_utils import Unpack
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import TransformersKwargs
from transformers.utils.generic import check_model_inputs


# from dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
# from dinov3_vit.modeling_dinov3_vit import DINOv3ViTLayer


from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTLayer

# from qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLVisionConfig
# from qwen3_vl.modeling_qwen3_vl import (
#     Qwen3VLForConditionalGeneration,
#     Qwen3VLModel,
#     Qwen3VLPreTrainedModel,
#     Qwen3VLVisionModel,
#     Qwen3VLVisionPatchEmbed,
#     Qwen3VLVisionPatchMerger,
#     Qwen3VLVisionRotaryEmbedding,
# )

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
    Qwen3VLVisionModel,
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionPatchMerger,
    Qwen3VLVisionRotaryEmbedding,
    Qwen3VLCausalLMOutputWithPast
)

try:
    from ..flex.models.flex_encoder_mask import FlexSceneEncoder
except ImportError:
    from model.flex.models.flex_encoder_mask import FlexSceneEncoder

# from .configuration_qwen3vl_dinov3vit import Qwen3VLDINOv3ViTConfig
try:
    from .configuration_qwen3vl_dinov3vit import Qwen3VLDINOv3ViTConfig
except (ImportError, ValueError):
    from configuration_qwen3vl_dinov3vit import Qwen3VLDINOv3ViTConfig

from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

class Qwen3VLModelFlex(Qwen3VLModel):
    def get_image_features(self, pixel_values, image_grid_thw):
        # 1. Using Vision Model (Flex Encoder)
        # Out shape: [Total_Images * Scene Token, Hidden_Dim]
        image_embeds, deepstack_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        
        # 2. Collecting Batch Information
        # image_grid_thw shape: (Num_Images, 3)
        num_images = image_grid_thw.shape[0]
        total_tokens = image_embeds.shape[0]
        
        # 3. Calculating total_tokens
        # total_tokens = (Batch * Scene Token)
        if total_tokens % num_images == 0:
            fixed_token_num = total_tokens // num_images
            split_sizes = [fixed_token_num] * num_images
        else:
            # fallback
            import warnings
            warnings.warn(f"Warning: Total tokens {total_tokens} not divisible by num images {num_images}")
            split_sizes = [total_tokens // num_images] * num_images
            split_sizes[-1] += total_tokens % num_images

        # 4. Per Image Feature
        image_embeds = torch.split(image_embeds, split_sizes)
        
        # 5. Disabling Deepstack 
        deepstack_embeds = None
        
        return image_embeds, deepstack_embeds

class DINOv3ViTLayerAdapter(GradientCheckpointingLayer):
    """
    Adapter class to adapt DINOv3ViTLayer to Qwen3VL's interface.
    
    Key functionalities:
    1. Format conversion: (seq_len, hidden_size) ↔ (batch_size, seq_len, hidden_size)
    2. cu_seqlens handling: Convert flattened sequences to batch format
    3. Position embedding adaptation: Add prefix tokens (CLS token), use strategy B (repeat first patch)
    4. Interface compatibility: Same interface as Qwen3VLVisionBlock
    """

    def __init__(self, dinov3_config: DINOv3ViTConfig, qwen3vlvison_config: Qwen3VLVisionConfig):
        super().__init__()
        # Store the config for later use
        self.dinov3_config = dinov3_config
        
        # Ensure the attention implementation is properly set
        if not hasattr(dinov3_config, '_attn_implementation'):
            dinov3_config._attn_implementation = 'flash_attention_2'  # Default attention implementation
            dinov3_config.attn_implementation = 'flash_attention_2'
        
        # 1. Create DINOv3ViTLayer (single layer)
        self.dinov3_layer = DINOv3ViTLayer(dinov3_config)

        # 2. Configure prefix tokens (CLS token, no register tokens)
        self.num_register_tokens = dinov3_config.num_register_tokens  # 0
        self.num_prefix_tokens = 1 + self.num_register_tokens  # 1 (only CLS)

        # 3. CLS token parameter (in DINOv3 dimension)
        if self.num_prefix_tokens > 0:
            self.cls_token = nn.Parameter(
                torch.randn(1, 1, dinov3_config.hidden_size) * 0.02
            )
        else:
            self.cls_token = None

        self.gradient_checkpointing = False


    def _flatten_to_batch(
        self,
        hidden_states: torch.Tensor,  # (seq_len, hidden_size)
        cu_seqlens: torch.Tensor,  # [0, seq1, seq1+seq2, ...]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Convert flattened sequences to batch format.

        Input:
        - hidden_states: (total_seq_len, hidden_size)
        - cu_seqlens: [0, seq1, seq1+seq2, ..., total_seq_len]

        Output:
        - batch_states: (batch_size, max_seq_len, hidden_size)
        - attention_mask: (batch_size, max_seq_len) or None
        """
        device = hidden_states.device
        dtype = hidden_states.dtype

        # 1. Split sequences according to cu_seqlens
        batch_size = len(cu_seqlens) - 1
        sequences = []
        seq_lens = []

        for i in range(batch_size):
            start_idx = cu_seqlens[i].item()
            end_idx = cu_seqlens[i + 1].item()
            seq = hidden_states[start_idx:end_idx]  # (seq_len_i, hidden_size)
            sequences.append(seq)
            seq_lens.append(end_idx - start_idx)

        # 2. Pad to max length
        max_seq_len = max(seq_lens) if seq_lens else 0

        if max_seq_len == 0:
            # Empty batch
            return torch.empty(0, 0, hidden_states.shape[-1], device=device, dtype=dtype), None

        batch_states = torch.zeros(
            batch_size, max_seq_len, hidden_states.shape[-1], device=device, dtype=dtype
        )

        attention_mask = torch.zeros(batch_size, max_seq_len, device=device, dtype=torch.bool)

        for i, (seq, seq_len) in enumerate(zip(sequences, seq_lens)):
            batch_states[i, :seq_len] = seq
            attention_mask[i, :seq_len] = True

        return batch_states, attention_mask


    def _batch_to_flatten(
        self,
        batch_states: torch.Tensor,  # (batch_size, max_seq_len, hidden_size)
        cu_seqlens: torch.Tensor,  # [0, seq1, seq1+seq2, ...]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:  # (total_seq_len, hidden_size)
        """
        Convert batch format back to flattened format.
        """
        device = batch_states.device
        dtype = batch_states.dtype

        batch_size = len(cu_seqlens) - 1
        sequences = []

        for i in range(batch_size):
            start_idx = cu_seqlens[i].item()
            end_idx = cu_seqlens[i + 1].item()
            seq_len = end_idx - start_idx

            seq = batch_states[i, :seq_len]  # (seq_len_i, hidden_size)
            sequences.append(seq)

        # Concatenate all sequences
        flatten_states = torch.cat(sequences, dim=0)  # (total_seq_len, hidden_size)
        # print(f"debug xy flatten_states.shape={flatten_states.shape}")
        return flatten_states

    def _add_prefix_tokens(
        self,
        batch_states: torch.Tensor,  # (batch_size, seq_len, hidden_size)
    ) -> torch.Tensor:  # (batch_size, seq_len + num_prefix_tokens, hidden_size)
        """
        Add prefix tokens (CLS token) to the beginning of sequences.

        Strategy:
        - Add CLS token to the beginning of each sequence
        - If num_register_tokens=0, only add CLS token
        """
        if self.num_prefix_tokens == 0:
            return batch_states

        batch_size = batch_states.shape[0]
        device = batch_states.device
        dtype = batch_states.dtype

        # Expand CLS token (in DINOv3 dimension)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, dinov3_hidden_size)

        # If register tokens exist, also add them
        if self.num_register_tokens > 0:
            # Create register tokens with the correct hidden size from DINOv3 config
            register_tokens = torch.zeros(
                batch_size, 
                self.num_register_tokens, 
                self.dinov3_config.hidden_size,  # Use stored DINOv3 config
                device=device, 
                dtype=dtype
            )
            prefix_tokens = torch.cat([cls_tokens, register_tokens], dim=1)
        else:
            prefix_tokens = cls_tokens

        # Concatenate prefix tokens and projected sequences
        batch_states_with_prefix = torch.cat([prefix_tokens, batch_states], dim=1)

        return batch_states_with_prefix

    def _remove_prefix_tokens(
        self,
        batch_states: torch.Tensor,  # (batch_size, seq_len + num_prefix_tokens, hidden_size)
    ) -> torch.Tensor:  # (batch_size, seq_len, hidden_size)
        """
        Remove prefix tokens, keep only patch tokens.
        """
        if self.num_prefix_tokens == 0:
            return batch_states

        return batch_states[:, self.num_prefix_tokens :, :]

    def _adapt_position_embeddings_for_dinov3(
        self,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # (cos, sin)
        num_patches_per_seq: List[int],  # Number of patches per sequence (excluding prefix tokens)
        batch_states_with_prefix: torch.Tensor,  # (batch_size, seq_len + num_prefix_tokens, hidden_size)
        attention_mask: Optional[torch.Tensor] = None,  # (batch_size, max_seq_len) where True means valid token
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adapt position embeddings to DINOv3 format, ensuring padding tokens get unit rotation (no RoPE effect).

        Input:
        - position_embeddings: (cos, sin), each is (total_patches, head_dim)
        - num_patches_per_seq: List of number of patches per sequence (excluding prefix tokens)
        - batch_states_with_prefix: (batch_size, max_seq_len, hidden_size) where max_seq_len includes prefix tokens
        - attention_mask: (batch_size, max_seq_len) where True means valid token, False means padding

        Output:
        - adapted_position_embeddings: (cos, sin),
          each is (max_seq_len - num_prefix_tokens, head_dim)
          Note: apply_rotary_pos_emb expects (num_patches, head_dim) where num_patches excludes prefix tokens
          and is the same for all batch items (due to padding).
          Padding positions will have unit rotation (cos=1, sin=0) to effectively disable RoPE for them.

        Strategy: 
        1. For batched inputs, pad position embeddings to max_seq_len - num_prefix_tokens
        2. Use unit rotation (cos=1, sin=0) for padding positions to ensure RoPE keeps query/key unchanged
           This way: q_rotated = q * 1 + rotate_half(q) * 0 = q (no rotation applied)
        """
        cos, sin = position_embeddings
        device = cos.device
        dtype = cos.dtype
        head_dim = cos.shape[-1]
        max_seq_len = batch_states_with_prefix.shape[1]  # includes prefix tokens
        max_patches = max_seq_len - self.num_prefix_tokens  # excludes prefix tokens

        # Build position embeddings for each sequence and pad to max_patches
        adapted_cos_list = []
        adapted_sin_list = []

        patch_idx = 0
        for i, seq_patches in enumerate(num_patches_per_seq):
            # Extract current sequence's position embeddings
            seq_cos = cos[patch_idx : patch_idx + seq_patches]  # (seq_patches, head_dim)
            seq_sin = sin[patch_idx : patch_idx + seq_patches]

            # Pad to max_patches if needed
            if seq_patches < max_patches:
                pad_len = max_patches - seq_patches
                # Use unit rotation (cos=1, sin=0) for padding to disable RoPE for padding tokens
                # This ensures that when apply_rotary_pos_emb applies RoPE to padding tokens,
                # it effectively does nothing: q_rotated = q * 1 + rotate_half(q) * 0 = q
                # This keeps the query/key unchanged while attention_mask will mask them out
                pad_cos = torch.ones(pad_len, head_dim, device=device, dtype=dtype)
                pad_sin = torch.zeros(pad_len, head_dim, device=device, dtype=dtype)
                seq_cos = torch.cat([seq_cos, pad_cos], dim=0)
                seq_sin = torch.cat([seq_sin, pad_sin], dim=0)

            adapted_cos_list.append(seq_cos)
            adapted_sin_list.append(seq_sin)

            patch_idx += seq_patches

        # Stack to create batch dimension, then take the first one (all should be same after padding)
        # Actually, we need to return a single (max_patches, head_dim) tensor that works for all batch items
        # Since all sequences are padded to max_patches, we can just use the first one
        # However, we should verify that all sequences have the same padding pattern
        # For now, we'll use the first one, but ideally we should check attention_mask to ensure consistency
        if not adapted_cos_list:
            raise ValueError("Empty batch: num_patches_per_seq is empty, cannot adapt position embeddings")
        adapted_cos = adapted_cos_list[0]  # (max_patches, head_dim)
        adapted_sin = adapted_sin_list[0]

        # Additional safety check: if attention_mask is provided, we can verify padding positions
        # and ensure they have correct position embeddings
        if attention_mask is not None:
            # attention_mask shape: (batch_size, max_seq_len) where max_seq_len includes prefix tokens
            # We need to check the patch tokens part (excluding prefix tokens)
            # For each batch item, find where the actual patches end (after prefix tokens)
            for i, seq_patches in enumerate(num_patches_per_seq):
                # The actual valid patches are from index 0 to seq_patches (after prefix tokens)
                # So padding starts from index seq_patches
                if seq_patches < max_patches:
                    # Verify that padding positions in attention_mask are False
                    # attention_mask[i, self.num_prefix_tokens + seq_patches : self.num_prefix_tokens + max_patches]
                    # should all be False
                    patch_mask = attention_mask[i, self.num_prefix_tokens : self.num_prefix_tokens + max_patches]
                    # The first seq_patches should be True, the rest should be False
                    expected_true = patch_mask[:seq_patches].all().item() if seq_patches > 0 else True
                    expected_false = (~patch_mask[seq_patches:]).all().item() if seq_patches < max_patches else True
                    if not (expected_true and expected_false):
                        import warnings
                        warnings.warn(
                            f"Attention mask verification failed for batch item {i}: "
                            f"seq_patches={seq_patches}, max_patches={max_patches}, "
                            f"expected_true={expected_true}, expected_false={expected_false}"
                        )

        return adapted_cos, adapted_sin

    def forward(
        self,
        hidden_states: torch.Tensor,  # (seq_len, hidden_size)
        cu_seqlens: torch.Tensor,  # [0, seq1, seq1+seq2, ...]
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (cos, sin)
        **kwargs,
    ) -> torch.Tensor:  # (seq_len, hidden_size)
        """
        Complete forward propagation flow with proper gradient handling.
        """
        # Ensure input requires gradients
        if not hidden_states.requires_grad:
            hidden_states = hidden_states.requires_grad_(True)
        
        # Step 1: Convert flattened sequences to batch format
        batch_states, attention_mask = self._flatten_to_batch(hidden_states, cu_seqlens)

        if batch_states.shape[0] == 0:
            # Empty batch, return directly
            return hidden_states

        # Step 2: Calculate number of patches per sequence (for position embedding adaptation)
        batch_size = len(cu_seqlens) - 1
        num_patches_per_seq = [
            cu_seqlens[i + 1].item() - cu_seqlens[i].item() for i in range(batch_size)
        ]

        # Step 3: Add prefix tokens (CLS token)
        batch_states_with_prefix = self._add_prefix_tokens(batch_states)

        # Step 4: Prepare attention mask for DINOv3 layer
        if attention_mask is not None:
            # Add True for prefix tokens (they should participate in attention)
            prefix_mask = torch.ones(
                batch_states_with_prefix.shape[0], self.num_prefix_tokens, 
                device=attention_mask.device, dtype=attention_mask.dtype
            )
            attention_mask_with_prefix = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            attention_mask_with_prefix = None

        # Step 5: Adapt position embeddings (use zero vectors for padding to disable RoPE)
        if position_embeddings is not None:
            adapted_position_embeddings = self._adapt_position_embeddings_for_dinov3(
                position_embeddings, num_patches_per_seq, batch_states_with_prefix, attention_mask_with_prefix
            )
        else:
            adapted_position_embeddings = None

        target_dtype = next(self.dinov3_layer.parameters()).dtype
        batch_states_with_prefix = batch_states_with_prefix.to(target_dtype)
        
        if adapted_position_embeddings is not None:
            adapted_position_embeddings = (
                adapted_position_embeddings[0].to(target_dtype),
                adapted_position_embeddings[1].to(target_dtype),
            )

        # Step 6: Call DINOv3ViTLayer with optional gradient checkpointing
        def create_custom_forward():
            def custom_forward(h_states, attn_mask, pos_emb):
                output = self.dinov3_layer(
                    hidden_states=h_states,
                    attention_mask=attn_mask,
                    position_embeddings=pos_emb,
                )
                # Ensure the output maintains gradients
                if isinstance(output, tuple):
                    return output[0] if len(output) > 0 else h_states
                else:
                    return output
            return custom_forward
        
        # Only use gradient checkpointing if in training mode and enabled
        if self.training and self.gradient_checkpointing:
            try:
                batch_output_with_prefix = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(),
                    batch_states_with_prefix,
                    attention_mask_with_prefix,
                    adapted_position_embeddings,
                    use_reentrant=False,
                )
            except RuntimeError as e:
                if "none of output has requires_grad" in str(e):
                    # Fallback to non-checkpointed execution
                    batch_output_with_prefix = self.dinov3_layer(
                        hidden_states=batch_states_with_prefix,
                        # attention_mask=attention_mask_with_prefix,
                        attention_mask=None,
                        position_embeddings=adapted_position_embeddings,
                    )
                else:
                    raise e
        else:
            batch_output_with_prefix = self.dinov3_layer(
                hidden_states=batch_states_with_prefix,
                # attention_mask=attention_mask_with_prefix,
                attention_mask=None,
                position_embeddings=adapted_position_embeddings,
            )

        # Step 7: Remove prefix tokens
        batch_output = self._remove_prefix_tokens(batch_output_with_prefix)

        # Step 8: Convert back to flattened format
        flatten_output = self._batch_to_flatten(batch_output, cu_seqlens, attention_mask)

        return flatten_output


class Qwen3VLVisionModelWithDINOv3(Qwen3VLPreTrainedModel):
    """
    Vision Model using DINOv3ViTLayer to replace Qwen3VLVisionBlock.

    Kept components:
    - patch_embed (Qwen3VL's 3D patch embedding)
    - pos_embed (Qwen3VL's position embedding)  
    - rotary_pos_emb (Qwen3VL's RoPE)
    - merger (Qwen3VL's patch merger)
    - deepstack_merger_list (DeepStack mechanism)

    Replaced components:
    - blocks: Use DINOv3ViTLayerAdapter
    """

    config: Qwen3VLVisionConfig
    _no_split_modules = ["DINOv3ViTLayerAdapter"]

    def __init__(self, vision_config: Qwen3VLVisionConfig, dinov3_config: DINOv3ViTConfig, flex_config: Optional[dict] = None, *inputs, **kwargs):
        super().__init__(vision_config, *inputs, **kwargs)
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.patch_size = vision_config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        # Keep original components
        self.patch_embed = Qwen3VLVisionPatchEmbed(config=vision_config)

        self.pos_embed = nn.Embedding(vision_config.num_position_embeddings, vision_config.hidden_size)
        self.num_grid_per_side = int(vision_config.num_position_embeddings**0.5)

        head_dim = vision_config.hidden_size // vision_config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        # Make sure DINOv3 config has proper attention implementation
        if not hasattr(dinov3_config, '_attn_implementation'):
            dinov3_config._attn_implementation = getattr(dinov3_config, '_attn_implementation', 'flash_attention_2')
            
        # Replace blocks with DINOv3ViTLayerAdapter
        self.blocks = nn.ModuleList(
            [DINOv3ViTLayerAdapter(dinov3_config, vision_config) for _ in range(vision_config.depth)]
        )
        
        # Keep other components unchanged
        self.merger = Qwen3VLVisionPatchMerger(config=vision_config, use_postshuffle_norm=False)

        self.deepstack_visual_indexes = vision_config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(config=vision_config, use_postshuffle_norm=True)
                for _ in range(len(vision_config.deepstack_visual_indexes))
            ]
        )

        # Add input and output projection layers
        self.input_projection = nn.Linear(
            vision_config.hidden_size,
            dinov3_config.hidden_size,
            bias=False
        )
        
        self.output_projection = nn.Linear(
            dinov3_config.hidden_size,
            vision_config.hidden_size,
            bias=False
        )

        if flex_config is None:
            print("Warning: flex_config is None, using default initialization.")
            flex_config = {}

        flex_config.setdefault("backbone_dim", dinov3_config.hidden_size)

        self.flex_config = flex_config

        self.flex_encoder = FlexSceneEncoder(flex_config)

        self.gradient_checkpointing = False

    def check_parameters_require_grad(self):
        """Check if all model parameters have requires_grad=True"""
        for name, param in self.named_parameters():
            if not param.requires_grad:
                print(f"Parameter {name} does not require gradients!")
            else:
                print(f"Parameter {name} requires gradients ✓")


    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Generate rotary position embeddings (same as original Qwen3VLVisionModel)."""
        merge_size = self.spatial_merge_size

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)  # block row indices
            block_cols = torch.arange(merged_w, device=device)  # block col indices
            intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
            intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        """Interpolate position embeddings (same as original Qwen3VLVisionModel)."""
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward method with proper gradient handling for DDP.
        """
        #TODO: Written T, C to config
        T = self.flex_config.get('num_timesteps', 2)
        C = self.flex_config.get('num_cameras', 6)
        total_images = grid_thw.shape[0]
        assert total_images % (T * C) == 0, \
            f"Error: Grid size ({total_images}) is not divisible by T*C ({T}*{C}={T*C}). Check your input batching."
        B = total_images // (T * C)

        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # for flex
        tokens_per_img = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).long()

        cu_seqlens = tokens_per_img.cumsum(
                dim=0,
                dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
            )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # Project to DINOv3 dimension before processing - ensure gradients flow
        hidden_states = self.input_projection(hidden_states)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            # Enable gradient checkpointing for each block if specified
            if self.gradient_checkpointing and self.training:
                # Use activation checkpointing for memory efficiency
                def create_custom_forward(block):
                    def custom_forward(*inputs):
                        return block(*inputs)
                    return custom_forward
                
                try:
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(blk),
                        hidden_states,
                        cu_seqlens,
                        position_embeddings,
                        use_reentrant=False,
                    )
                except RuntimeError as e:
                    if "none of output has requires_grad" in str(e):
                        # Fallback to non-checkpointed execution
                        layer_outputs = blk(
                            hidden_states,
                            cu_seqlens=cu_seqlens,
                            position_embeddings=position_embeddings,
                            **kwargs,
                        )
                    else:
                        raise e
            else:
                layer_outputs = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            
            hidden_states = layer_outputs # dino (N, D_dino)
            
            if layer_num in self.deepstack_visual_indexes:
                # Project to Qwen3VL dimension for deepstack features
                deepstack_feature_input = self.output_projection(hidden_states)
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    deepstack_feature_input
                )
                deepstack_feature_lists.append(deepstack_feature)

        # print("Flattened hidden_states shape:", hidden_states.shape) # (10400, 384)
        split_tensors = torch.split(hidden_states, tokens_per_img.tolist(), dim=0)
        mask_list = [torch.zeros(length, dtype=torch.bool, device=hidden_states.device) for length in tokens_per_img.tolist()]
        padded_states = pad_sequence(split_tensors, batch_first=True, padding_value=0.0)
        padded_mask = pad_sequence(mask_list, batch_first=True, padding_value=True)

        img_feats = padded_states.view(B, T, C, -1, padded_states.size(-1))
        img_masks = padded_mask.view(B, T, C, -1)

        # print("img_feats shape:", img_feats.shape) 
        hidden_states = self.flex_encoder(img_feats, key_padding_mask=img_masks).flatten(0, 1)
        # print("Flex hidden_states shape:", hidden_states.shape)

        self._flex_batch_size = B

        return hidden_states, deepstack_feature_lists

    def set_gradient_checkpointing(self, value: bool = True):
        """
        Set gradient checkpointing for the vision model.
        """
        self.gradient_checkpointing = value
        for block in self.blocks:
            block.gradient_checkpointing = value


class Qwen3VLForConditionalGenerationWithDINOv3(Qwen3VLPreTrainedModel, GenerationMixin):
    """
    Complete Qwen3VL-DINOv3 mixed model.

    Replaced components:
    - visual: Use Qwen3VLVisionModelWithDINOv3

    Kept components:
    - language_model: Qwen3VLTextModel (unchanged)
    - lm_head: Language model output head (unchanged)
    """

    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = ["lm_head.weight"]
    accepts_loss_kwargs = False
    config: Qwen3VLDINOv3ViTConfig

    def __init__(self, config: Qwen3VLDINOv3ViTConfig):
        super().__init__(config)

        # Create base model with replaced visual component
        base_model = Qwen3VLModelFlex(config)
        base_model.visual = Qwen3VLVisionModelWithDINOv3(
            vision_config=config.vision_config, dinov3_config=config.dinov3_config, flex_config=config.flex_config,
        )

        self.model = base_model
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, "Qwen3VLCausalLMOutputWithPast"]:
        """
        Forward method (same as Qwen3VLForConditionalGeneration).
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        actual_vocab_size = self.lm_head.out_features
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=actual_vocab_size)
            # loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)


        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )

        # Qwen3VL position_ids are prepareed with rope_deltas in forward
        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size, num_images_sample)`)
            video_nums (`torch.LongTensor` of shape `(batch_size, num_videos_sample)`)
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        if inputs_embeds is not None:
            vision_start_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(vision_start_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            image_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            video_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
        else:
            vision_start_mask = input_ids == vision_start_token_id
            image_mask = input_ids == image_token_id
            video_mask = input_ids == video_token_id

        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
            )

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=list(video_nums), repeat_times=expand_size
                    )
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

