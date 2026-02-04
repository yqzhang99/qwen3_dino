"""Qwen3-DINOv3 mixed model implementation"""

from typing import List, Optional, Tuple, Union, Any, Callable
from transformers.processing_utils import Unpack
import math

from collections.abc import Callable
from typing import Optional

import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub, use_kernel_func_from_hub, use_kernelized_func
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import check_model_inputs, maybe_autocast
from .configuration_qwen3 import Qwen3Config

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

# ==============================================================================
# 1. DinoV3_Adapter Definition
# ==============================================================================

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

# ==============================================================================
# 2. Vision Model Definition 
# ==============================================================================

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

# ==============================================================================
# 3. Language Model Definition 
# ==============================================================================

@use_kernel_forward_from_hub("RMSNorm")
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: Qwen3Config | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@use_kernel_func_from_hub("rotary_pos_emb")
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


@use_kernelized_func(apply_rotary_pos_emb)
class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class Qwen3PreTrainedModel(PreTrainedModel):
    config: Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3DecoderLayer,
        "attentions": Qwen3Attention,
    }


@auto_docstring
class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


@auto_docstring
class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Qwen3ForSequenceClassification(GenericForSequenceClassification, Qwen3PreTrainedModel):
    pass


class Qwen3ForTokenClassification(GenericForTokenClassification, Qwen3PreTrainedModel):
    pass


class Qwen3ForQuestionAnswering(GenericForQuestionAnswering, Qwen3PreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`

class Qwen3VLDINOv3ForConditionalGeneration(PreTrainedModel, GenerationMixin):
    config_class = Qwen3VLDINOv3Config
    _supports_flash_attn_2 = True
    _no_split_modules = ["Qwen3DecoderLayer", "DINOv3ViTLayerAdapter"]
    _tied_weights_keys = ["model.lm_head.weight"]

    def __init__(self, config: Qwen3VLDINOv3Config):
        super().__init__(config)

        self.visual = Qwen3VLVisionModelWithDINOv3(
                vision_config=config.vision_config,
                dinov3_config=config.dinov3_config,
                flex_config=config.flex_config
            )

        self.model = Qwen3ForCausalLM(config.text_config)

        self.vocab_size = config.text_config.vocab_size
        self.padding_idx = config.text_config.pad_token_id

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

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
        
        return image_embeds, deepstack_embedsel_values, image_grid_thw):

    def get_video_features(self, pixel_values_videos, video_grid_thw):
        # 复用图像的逻辑
        return self.get_image_features(pixel_values_videos, video_grid_thw)
    
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask


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
    ):

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        
        if pixel_values is not None:
            target_dtype = self.get_input_embeddings().weight.dtype
            image_embeds, _ = self.get_image_features(pixel_values.to(target_dtype), image_grid_thw)

            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask.unsqueeze(-1), image_embeds)

        if pixel_values_videos is not None:
            target_dtype = self.get_input_embeddings().weight.dtype
            video_embeds, _ = self.get_video_features(pixel_values_videos.to(target_dtype), video_grid_thw)
            
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            
            _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds, video_features=video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask.unsqueeze(-1), video_embeds)

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
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


__all__ = [
    "Qwen3ForCausalLM",
    "Qwen3ForQuestionAnswering",
    "Qwen3PreTrainedModel",
    "Qwen3Model",
    "Qwen3ForSequenceClassification",
    "Qwen3ForTokenClassification",
]