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

"""Qwen3-DINOv3 mixed model implementation"""

from typing import List, Optional, Tuple, Union, Any
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub, use_kernel_func_from_hub, use_kernelized_func
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.generic import check_model_inputs, maybe_autocast

from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast

from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTLayer

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionPatchMerger,
    Qwen3VLVisionRotaryEmbedding,
    Qwen3VLPreTrainedModel,
)

# Import FlexSceneEncoder
try:
    from ..flex.models.flex_encoder_mask import FlexSceneEncoder
except ImportError:
    try:
        from model.flex.models.flex_encoder_mask import FlexSceneEncoder
    except ImportError:
        FlexSceneEncoder = None

# Import config
try:
    from .configuration_qwen3_dinov3vit import Qwen3DINOv3Config
except (ImportError, ValueError):
    try:
        from configuration_qwen3_dinov3vit import Qwen3DINOv3Config
    except ImportError:
        Qwen3DINOv3Config = None


# ==============================================================================
# 1. DinoV3 Adapter Definition
# ==============================================================================

class DINOv3ViTLayerAdapter(GradientCheckpointingLayer):
    """
    Adapter class to adapt DINOv3ViTLayer to work with Qwen3's interface.
    
    Key functionalities:
    1. Format conversion: (seq_len, hidden_size) ↔ (batch_size, seq_len, hidden_size)
    2. cu_seqlens handling: Convert flattened sequences to batch format
    3. Position embedding adaptation: Add prefix tokens (CLS token)
    4. Interface compatibility: Same interface as Qwen3VLVisionBlock
    """

    def __init__(self, dinov3_config: DINOv3ViTConfig, qwen3vl_vision_config: Qwen3VLVisionConfig):
        super().__init__()
        self.dinov3_config = dinov3_config
        
        # Ensure the attention implementation is properly set
        if not hasattr(dinov3_config, '_attn_implementation'):
            dinov3_config._attn_implementation = 'flash_attention_2'
            dinov3_config.attn_implementation = 'flash_attention_2'
        
        # Create DINOv3ViTLayer (single layer)
        self.dinov3_layer = DINOv3ViTLayer(dinov3_config)

        # Configure prefix tokens (CLS token, no register tokens)
        self.num_register_tokens = getattr(dinov3_config, 'num_register_tokens', 0)
        self.num_prefix_tokens = 1 + self.num_register_tokens  # 1 (only CLS)

        # CLS token parameter (in DINOv3 dimension)
        if self.num_prefix_tokens > 0:
            self.cls_token = nn.Parameter(
                torch.randn(1, 1, dinov3_config.hidden_size) * 0.02
            )
        else:
            self.cls_token = None

        self.gradient_checkpointing = False

    def _flatten_to_batch(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Convert flattened sequences to batch format."""
        device = hidden_states.device
        dtype = hidden_states.dtype

        batch_size = len(cu_seqlens) - 1
        sequences = []
        seq_lens = []

        for i in range(batch_size):
            start_idx = cu_seqlens[i].item()
            end_idx = cu_seqlens[i + 1].item()
            seq = hidden_states[start_idx:end_idx]
            sequences.append(seq)
            seq_lens.append(end_idx - start_idx)

        max_seq_len = max(seq_lens) if seq_lens else 0

        if max_seq_len == 0:
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
        batch_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convert batch format back to flattened format."""
        batch_size = len(cu_seqlens) - 1
        sequences = []

        for i in range(batch_size):
            start_idx = cu_seqlens[i].item()
            end_idx = cu_seqlens[i + 1].item()
            seq_len = end_idx - start_idx
            seq = batch_states[i, :seq_len]
            sequences.append(seq)

        flatten_states = torch.cat(sequences, dim=0)
        return flatten_states

    def _add_prefix_tokens(
        self,
        batch_states: torch.Tensor,
    ) -> torch.Tensor:
        """Add prefix tokens (CLS token) to the beginning of sequences."""
        if self.num_prefix_tokens == 0:
            return batch_states

        batch_size = batch_states.shape[0]
        device = batch_states.device
        dtype = batch_states.dtype

        cls_tokens = self.cls_token.expand(batch_size, -1, -1).to(dtype=dtype)

        if self.num_register_tokens > 0:
            register_tokens = torch.zeros(
                batch_size, 
                self.num_register_tokens, 
                self.dinov3_config.hidden_size,
                device=device, 
                dtype=dtype
            )
            prefix_tokens = torch.cat([cls_tokens, register_tokens], dim=1)
        else:
            prefix_tokens = cls_tokens

        batch_states_with_prefix = torch.cat([prefix_tokens, batch_states], dim=1)
        return batch_states_with_prefix

    def _remove_prefix_tokens(
        self,
        batch_states: torch.Tensor,
    ) -> torch.Tensor:
        """Remove prefix tokens, keep only patch tokens."""
        if self.num_prefix_tokens == 0:
            return batch_states
        return batch_states[:, self.num_prefix_tokens:, :]

    def _adapt_position_embeddings_for_dinov3(
        self,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        num_patches_per_seq: List[int],
        batch_states_with_prefix: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adapt position embeddings to DINOv3 format."""
        cos, sin = position_embeddings
        device = cos.device
        dtype = cos.dtype
        head_dim = cos.shape[-1]
        max_seq_len = batch_states_with_prefix.shape[1]
        max_patches = max_seq_len - self.num_prefix_tokens

        adapted_cos_list = []
        adapted_sin_list = []

        patch_idx = 0
        for i, seq_patches in enumerate(num_patches_per_seq):
            seq_cos = cos[patch_idx : patch_idx + seq_patches]
            seq_sin = sin[patch_idx : patch_idx + seq_patches]

            if seq_patches < max_patches:
                pad_len = max_patches - seq_patches
                pad_cos = torch.ones(pad_len, head_dim, device=device, dtype=dtype)
                pad_sin = torch.zeros(pad_len, head_dim, device=device, dtype=dtype)
                seq_cos = torch.cat([seq_cos, pad_cos], dim=0)
                seq_sin = torch.cat([seq_sin, pad_sin], dim=0)

            adapted_cos_list.append(seq_cos)
            adapted_sin_list.append(seq_sin)
            patch_idx += seq_patches

        if not adapted_cos_list:
            raise ValueError("Empty batch: num_patches_per_seq is empty")
            
        adapted_cos = adapted_cos_list[0]
        adapted_sin = adapted_sin_list[0]

        return adapted_cos, adapted_sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with proper gradient handling."""
        if not hidden_states.requires_grad:
            hidden_states = hidden_states.requires_grad_(True)
        
        # Step 1: Convert flattened sequences to batch format
        batch_states, attention_mask = self._flatten_to_batch(hidden_states, cu_seqlens)

        if batch_states.shape[0] == 0:
            return hidden_states

        # Step 2: Calculate number of patches per sequence
        batch_size = len(cu_seqlens) - 1
        num_patches_per_seq = [
            cu_seqlens[i + 1].item() - cu_seqlens[i].item() for i in range(batch_size)
        ]

        # Step 3: Add prefix tokens (CLS token)
        batch_states_with_prefix = self._add_prefix_tokens(batch_states)

        # Step 4: Prepare attention mask for DINOv3 layer
        if attention_mask is not None:
            prefix_mask = torch.ones(
                batch_states_with_prefix.shape[0], self.num_prefix_tokens, 
                device=attention_mask.device, dtype=attention_mask.dtype
            )
            attention_mask_with_prefix = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            attention_mask_with_prefix = None

        # Step 5: Adapt position embeddings
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

        # Step 6: Call DINOv3ViTLayer
        if self.training and self.gradient_checkpointing:
            def create_custom_forward():
                def custom_forward(h_states, attn_mask, pos_emb):
                    output = self.dinov3_layer(
                        hidden_states=h_states,
                        attention_mask=attn_mask,
                        position_embeddings=pos_emb,
                    )
                    if isinstance(output, tuple):
                        return output[0] if len(output) > 0 else h_states
                    return output
                return custom_forward
            
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
                    batch_output_with_prefix = self.dinov3_layer(
                        hidden_states=batch_states_with_prefix,
                        attention_mask=None,
                        position_embeddings=adapted_position_embeddings,
                    )
                else:
                    raise e
        else:
            batch_output_with_prefix = self.dinov3_layer(
                hidden_states=batch_states_with_prefix,
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

class Qwen3DINOv3VisionModel(Qwen3VLPreTrainedModel):
    """
    Vision Model using DINOv3ViTLayer.

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

    def __init__(
        self, 
        vision_config: Qwen3VLVisionConfig, 
        dinov3_config: DINOv3ViTConfig, 
        flex_config: Optional[dict] = None, 
        *inputs, 
        **kwargs
    ):
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

        # FlexSceneEncoder config
        if flex_config is None:
            print("Warning: flex_config is None, using default initialization.")
            flex_config = {}

        flex_config.setdefault("backbone_dim", dinov3_config.hidden_size)
        self.flex_config = flex_config

        if FlexSceneEncoder is not None:
            self.flex_encoder = FlexSceneEncoder(flex_config)
        else:
            self.flex_encoder = None
            print("Warning: FlexSceneEncoder not available")

        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Generate rotary position embeddings."""
        merge_size = self.spatial_merge_size

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

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

        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Interpolate position embeddings."""
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

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

        # Since we use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

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
        hidden_states: torch.Tensor, 
        grid_thw: torch.Tensor, 
        **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass with FlexEncoder support."""
        T = self.flex_config.get('num_timesteps', 2)
        C = self.flex_config.get('num_cameras', 6)
        total_images = grid_thw.shape[0]
        
        if total_images % (T * C) == 0:
            B = total_images // (T * C)
        else:
            # Fallback for non-flex mode
            B = total_images
            T = 1
            C = 1

        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        tokens_per_img = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).long()

        cu_seqlens = tokens_per_img.cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # Project to DINOv3 dimension
        hidden_states = self.input_projection(hidden_states)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
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
            
            hidden_states = layer_outputs
            
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature_input = self.output_projection(hidden_states)
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    deepstack_feature_input
                )
                deepstack_feature_lists.append(deepstack_feature)

        # Apply FlexEncoder if available and in flex mode
        if self.flex_encoder is not None and total_images == B * T * C and T > 1:
            split_tensors = torch.split(hidden_states, tokens_per_img.tolist(), dim=0)
            mask_list = [torch.zeros(length, dtype=torch.bool, device=hidden_states.device) for length in tokens_per_img.tolist()]
            padded_states = pad_sequence(split_tensors, batch_first=True, padding_value=0.0)
            padded_mask = pad_sequence(mask_list, batch_first=True, padding_value=True)

            img_feats = padded_states.view(B, T, C, -1, padded_states.size(-1))
            img_masks = padded_mask.view(B, T, C, -1)

            hidden_states = self.flex_encoder(img_feats, key_padding_mask=img_masks).flatten(0, 1)
            self._flex_batch_size = B
        else:
            # Without FlexEncoder, project back to vision config dimension
            hidden_states = self.output_projection(hidden_states)
            hidden_states = self.merger(hidden_states)
            self._flex_batch_size = None

        return hidden_states, deepstack_feature_lists

    def set_gradient_checkpointing(self, value: bool = True):
        """Set gradient checkpointing for the vision model."""
        self.gradient_checkpointing = value
        for block in self.blocks:
            block.gradient_checkpointing = value


# ==============================================================================
# 3. Language Model Components (from Qwen3)
# ==============================================================================

@use_kernel_forward_from_hub("RMSNorm")
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
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
    inv_freq: torch.Tensor

    def __init__(self, config, device=None):
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
    def compute_default_rope_parameters(config=None, device=None, seq_len=None):
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        attention_factor = 1.0
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
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
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for GQA."""
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
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
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

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
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
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else "full_attention"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
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

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ==============================================================================
# 4. Base Model Classes
# ==============================================================================

class Qwen3DINOv3PreTrainedModel(PreTrainedModel):
    """Base class for Qwen3-DINOv3 models."""
    config_class = Qwen3DINOv3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer", "DINOv3ViTLayerAdapter"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,          
        deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,    
        **kwargs,
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

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping.get(decoder_layer.attention_type, causal_mask_mapping["full_attention"]),
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            
            if deepstack_visual_embeds is not None and visual_pos_masks is not None:
                if layer_idx < len(deepstack_visual_embeds):
                    hidden_states = self._deepstack_process(
                        hidden_states,
                        visual_pos_masks,
                        deepstack_visual_embeds[layer_idx],
                    )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    def _deepstack_process(
        self, 
        hidden_states: torch.Tensor, 
        visual_pos_masks: torch.Tensor, 
        visual_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        将 deepstack 视觉特征加到 hidden_states 的视觉位置上。
        
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            visual_pos_masks: (batch, seq_len) - True 表示视觉 token 位置
            visual_embeds: (num_visual_tokens, hidden_dim) - deepstack 视觉特征
        
        Returns:
            更新后的 hidden_states
        """
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        
        # Clone 防止 in-place 操作影响梯度
        hidden_states = hidden_states.clone()
        
        # 在视觉位置上加上 deepstack 特征
        hidden_states[visual_pos_masks, :] = hidden_states[visual_pos_masks, :] + visual_embeds
        
        return hidden_states

class Qwen3DINOv3TextModel(Qwen3DINOv3PreTrainedModel):
    """
    基于 Qwen3Model 修改的文本模型，增加了 DeepStack (视觉特征层间注入) 支持。
    结构上对齐 Qwen3VLTextModel，但复用 Qwen3 的组件。
    """
    def __init__(self, config: Qwen3DINOv3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # 复用标准的 Qwen3DecoderLayer
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # 复用标准的 Qwen3RMSNorm
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 复用标准的 Qwen3RotaryEmbedding
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        
        self.gradient_checkpointing = False
        self.has_sliding_layers = hasattr(config, "layer_types") and "sliding_attention" in config.layer_types
        
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _deepstack_process(
        self, 
        hidden_states: torch.Tensor, 
        visual_pos_masks: torch.Tensor, 
        visual_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        DeepStack 核心逻辑：将视觉特征注入到 hidden_states 中
        """
        # 确保设备和类型一致
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        
        # 1. 提取视觉位置的 hidden states
        # 2. 加上视觉特征 (Residual Connection)
        # 3. 填回原位置
        # 注意：这里使用 clone + advanced indexing 来避免 inplace 操作导致的梯度问题
        current_states = hidden_states[visual_pos_masks, :]
        injected_states = current_states + visual_embeds
        hidden_states[visual_pos_masks, :] = injected_states
        
        return hidden_states

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # --- 新增参数 ---
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
        # ----------------
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

        # 掩码处理逻辑 (与 Qwen3Model 保持一致)
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        
        # 计算 RoPE (Qwen3VL 通常传入的是 3D position_ids，这里直接透传给 rotary_emb)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Decoder Layer 循环
        for layer_idx, decoder_layer in enumerate(self.layers):
            # 1. 正常前向传播
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping.get(decoder_layer.attention_type, causal_mask_mapping["full_attention"]),
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            
            # 2. [DeepStack 注入点]
            # 如果提供了 DeepStack 特征，且当前层在注入范围内，则进行注入
            if deepstack_visual_embeds is not None and visual_pos_masks is not None:
                if layer_idx < len(deepstack_visual_embeds):
                    hidden_states = self._deepstack_process(
                        hidden_states,
                        visual_pos_masks,
                        deepstack_visual_embeds[layer_idx],
                    )

        hidden_states = self.norm(hidden_states)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

# ==============================================================================
# 5. Main Model Class
# ==============================================================================

class Qwen3DINOv3Model(Qwen3DINOv3PreTrainedModel, GenerationMixin):
    """
    Qwen3-DINOv3 model for conditional generation (vision-language tasks).
    
    This model combines:
    - Qwen3 text backbone
    - DINOv3 vision encoder
    - FlexSceneEncoder for multi-camera/timestep processing
    """
    
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config):
        super().__init__(config)
        
        # Vision encoder
        if hasattr(config, 'vision_config') and hasattr(config, 'dinov3_config'):
            self.visual = Qwen3DINOv3VisionModel(
                vision_config=config.vision_config,
                dinov3_config=config.dinov3_config,
                flex_config=getattr(config, 'flex_config', None),
            )
        else:
            self.visual = None
            
        # Text model
        self.language_model = Qwen3DINOv3TextModel(config)
        
        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Token IDs
        self.image_token_id = getattr(config, 'image_token_id', None)
        self.video_token_id = getattr(config, 'video_token_id', None)
        self.vision_start_token_id = getattr(config, 'vision_start_token_id', None)
        
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_image_features(
        self, 
        pixel_values: torch.FloatTensor, 
        image_grid_thw: torch.LongTensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Extract image features using the vision encoder."""
        if self.visual is None:
            raise ValueError("Vision encoder not initialized")
        return self.visual(pixel_values, grid_thw=image_grid_thw)

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass with optional vision inputs.
        """
        # 1. Prepare Inputs Embeddings (if not provided)
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Initialize visual variables
        image_mask = None
        visual_pos_masks = None
        deepstack_visual_embeds = None

        # 2. Process Vision Inputs
        if pixel_values is not None and image_grid_thw is not None and self.visual is not None:
            # pixel_values 需要转为 visual model 的 dtype
            pixel_values = pixel_values.type(self.visual.dtype)
            
            # Forward pass through Vision Encoder (DINOv3 + Flex)
            # Returns: (all_image_features, list_of_deepstack_features)
            image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            
            # Qwen3VL logic: ensure image_embeds is a single concatenated tensor
            # (Note: self.visual output usually is already concat, but good to ensure dtype/device)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

            # 3. Create Placeholder Mask & Scatter (The Merge Step)
            # This identifies where image_token_id is and replaces embeddings with visual features
            image_mask = self.get_placeholder_mask(
                input_ids, 
                inputs_embeds=inputs_embeds, 
                image_features=image_embeds
            )
            
            # 🌟 CORE FUSION: Overwrite text embeddings with image embeddings at mask locations
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            
            # 4. Handle DeepStack (Optional, consistent with Qwen3VL)
            # DeepStack features need to be injected into early LLM layers
            if image_mask is not None:
                # Reduce mask dimension: (Batch, Seq, Hidden) -> (Batch, Seq)
                image_mask = image_mask[..., 0] 
                visual_pos_masks = image_mask
                deepstack_visual_embeds = deepstack_image_embeds

        # 5. Handle Position IDs (mRoPE) - CRITICAL for Qwen3-VL
        # Qwen3 uses 3D position IDs calculated based on image grid sizes
        if position_ids is None:
            # Handle attention mask format (convert to int if needed)
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                # Flatten standard 4D mask if necessary for processing
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE Deltas (Standard Qwen3VL Logic)
            # We assume self.rope_deltas is initialized in __init__ as None
            if not hasattr(self, 'rope_deltas'): 
                self.rope_deltas = None
                
            # Logic to determine if we need to recalculate position IDs (Pre-fill stage)
            is_prefill = (input_ids is not None and input_ids.shape[1] != 1) or \
                         (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            
            if is_prefill or self.rope_deltas is None:
                # This function MUST be implemented (copied from Qwen3VLModel)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            else:
                # Decoding stage: use cached deltas
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    # Fix shapes for broadcasting
                    if hasattr(delta, 'shape') and len(delta.shape) > 0:
                         delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                # Expand to 3D (T, H, W) format
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # Set input_ids to None because we are passing inputs_embeds
        input_ids = None 

        # Forward through language model
        outputs = self.language_model(
            input_ids=input_ids, # None
            attention_mask=attention_mask,
            position_ids=position_ids, # Passed the 3D position IDs
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, # Fused embeddings
            use_cache=use_cache,
            cache_position=cache_position,
            # DeepStack Arguments
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return outputs

class Qwen3VLForConditionalGenerationWithDINOv3(Qwen3DINOv3PreTrainedModel, GenerationMixin):

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        
        self.model = Qwen3DINOv3Model(config)
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.post_init()

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.language_model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @property
    def visual(self):
        return self.model.visual

    @property
    def language_model(self):
        return self.model.language_model

    def get_image_features(self, pixel_values, image_grid_thw):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:

        outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                **kwargs
            )

        hidden_states = outputs.last_hidden_state
        
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,  
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
        image_grid_thw=None,
        **kwargs,
    ):
        """Prepare inputs for generation."""
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs,
        )

        # Only include pixel values on first iteration
        if cache_position is not None and cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["image_grid_thw"] = None
        else:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_grid_thw"] = image_grid_thw

        return model_inputs


__all__ = [
    "DINOv3ViTLayerAdapter",
    "Qwen3DINOv3VisionModel",
    "Qwen3DINOv3TextModel",
    "Qwen3DINOv3PreTrainedModel",
    "Qwen3VLForConditionalGenerationWithDINOv3",
]
