# coding=utf-8
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.

"""Qwen3VL-DINOv3 mixed model implementation"""

from typing import List, Optional, Tuple, Union, Any
from collections.abc import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import Unpack, TransformersKwargs, can_return_tuple, check_model_inputs
from transformers.cache_utils import Cache, DynamicCache

# ----------------------------------------------------------------------
# 1. Imports from Standard Qwen3VL and DINOv3
# ----------------------------------------------------------------------
# 尝试从 transformers 导入，如果失败尝试本地导入（兼容你的环境）
try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLPreTrainedModel,
        Qwen3VLTextModel,  # [Critical] 直接复用官方文本模型以支持 mRoPE
        Qwen3VLVisionPatchEmbed,
        Qwen3VLVisionPatchMerger,
        Qwen3VLVisionRotaryEmbedding,
        Qwen3VLCausalLMOutputWithPast
    )
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
except ImportError:
    # Fallback to local file if transformers version is old or custom
    try:
        from modeling_qwen3_vl import (
            Qwen3VLPreTrainedModel,
            Qwen3VLTextModel,
            Qwen3VLVisionPatchEmbed,
            Qwen3VLVisionPatchMerger,
            Qwen3VLVisionRotaryEmbedding,
            Qwen3VLCausalLMOutputWithPast
        )
        from configuration_qwen3_vl import Qwen3VLVisionConfig
    except ImportError:
        raise ImportError("Could not import Qwen3VL components. Ensure transformers>=4.49 or local files exist.")

from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTLayer

# ----------------------------------------------------------------------
# 2. Imports for Custom Config and Flex
# ----------------------------------------------------------------------
# Import FlexSceneEncoder
try:
    from ..flex.models.flex_encoder_mask import FlexSceneEncoder
except ImportError:
    try:
        from model.flex.models.flex_encoder_mask import FlexSceneEncoder
    except ImportError:
        FlexSceneEncoder = None

# Import Config (Matches your uploaded file name)
try:
    from .configuration_qwen3vl_dinov3vit import Qwen3VLDINOv3ViTConfig
except (ImportError, ValueError):
    try:
        from configuration_qwen3vl_dinov3vit import Qwen3VLDINOv3ViTConfig
    except ImportError:
        # Fallback definition if file is missing during simple syntax check
        Qwen3VLDINOv3ViTConfig = None


# ==============================================================================
# 3. DinoV3 Adapter
# ==============================================================================

class DINOv3ViTLayerAdapter(GradientCheckpointingLayer):
    """
    Adapter class to adapt DINOv3ViTLayer to work with Qwen3's interface.
    """

    def __init__(self, dinov3_config: DINOv3ViTConfig, qwen3vl_vision_config: Qwen3VLVisionConfig):
        super().__init__()
        self.dinov3_config = dinov3_config
        
        if not hasattr(dinov3_config, '_attn_implementation'):
            dinov3_config._attn_implementation = 'flash_attention_2'
            dinov3_config.attn_implementation = 'flash_attention_2'
        
        self.dinov3_layer = DINOv3ViTLayer(dinov3_config)

        self.num_register_tokens = getattr(dinov3_config, 'num_register_tokens', 0)
        self.num_prefix_tokens = 1 + self.num_register_tokens

        if self.num_prefix_tokens > 0:
            self.cls_token = nn.Parameter(
                torch.randn(1, 1, dinov3_config.hidden_size) * 0.02
            )
        else:
            self.cls_token = None

        self.gradient_checkpointing = False

    def _flatten_to_batch(self, hidden_states, cu_seqlens):
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

        batch_states = torch.zeros(batch_size, max_seq_len, hidden_states.shape[-1], device=device, dtype=dtype)
        attention_mask = torch.zeros(batch_size, max_seq_len, device=device, dtype=torch.bool)

        for i, (seq, seq_len) in enumerate(zip(sequences, seq_lens)):
            batch_states[i, :seq_len] = seq
            attention_mask[i, :seq_len] = True

        return batch_states, attention_mask

    def _batch_to_flatten(self, batch_states, cu_seqlens, attention_mask=None):
        batch_size = len(cu_seqlens) - 1
        sequences = []
        for i in range(batch_size):
            start_idx = cu_seqlens[i].item()
            end_idx = cu_seqlens[i + 1].item()
            seq_len = end_idx - start_idx
            seq = batch_states[i, :seq_len]
            sequences.append(seq)
        return torch.cat(sequences, dim=0)

    def _add_prefix_tokens(self, batch_states):
        if self.num_prefix_tokens == 0:
            return batch_states
        batch_size = batch_states.shape[0]
        device = batch_states.device
        dtype = batch_states.dtype
        cls_tokens = self.cls_token.expand(batch_size, -1, -1).to(dtype=dtype)
        if self.num_register_tokens > 0:
            register_tokens = torch.zeros(batch_size, self.num_register_tokens, self.dinov3_config.hidden_size, device=device, dtype=dtype)
            prefix_tokens = torch.cat([cls_tokens, register_tokens], dim=1)
        else:
            prefix_tokens = cls_tokens
        return torch.cat([prefix_tokens, batch_states], dim=1)

    def _remove_prefix_tokens(self, batch_states):
        if self.num_prefix_tokens == 0:
            return batch_states
        return batch_states[:, self.num_prefix_tokens:, :]

    def _adapt_position_embeddings_for_dinov3(self, position_embeddings, num_patches_per_seq, batch_states_with_prefix, attention_mask=None):
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
            raise ValueError("Empty batch")
        return adapted_cos_list[0], adapted_sin_list[0]

    def forward(self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
        if not hidden_states.requires_grad:
            hidden_states = hidden_states.requires_grad_(True)
        
        batch_states, attention_mask = self._flatten_to_batch(hidden_states, cu_seqlens)
        if batch_states.shape[0] == 0:
            return hidden_states

        batch_size = len(cu_seqlens) - 1
        num_patches_per_seq = [cu_seqlens[i + 1].item() - cu_seqlens[i].item() for i in range(batch_size)]
        
        batch_states_with_prefix = self._add_prefix_tokens(batch_states)
        
        attention_mask_with_prefix = None
        if attention_mask is not None:
            prefix_mask = torch.ones(batch_states_with_prefix.shape[0], self.num_prefix_tokens, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask_with_prefix = torch.cat([prefix_mask, attention_mask], dim=1)

        adapted_position_embeddings = None
        if position_embeddings is not None:
            adapted_position_embeddings = self._adapt_position_embeddings_for_dinov3(
                position_embeddings, num_patches_per_seq, batch_states_with_prefix, attention_mask_with_prefix
            )

        target_dtype = next(self.dinov3_layer.parameters()).dtype
        batch_states_with_prefix = batch_states_with_prefix.to(target_dtype)
        if adapted_position_embeddings:
            adapted_position_embeddings = (adapted_position_embeddings[0].to(target_dtype), adapted_position_embeddings[1].to(target_dtype))

        if self.training and self.gradient_checkpointing:
            def create_custom_forward():
                def custom_forward(h_states, attn_mask, pos_emb):
                    output = self.dinov3_layer(hidden_states=h_states, attention_mask=attn_mask, position_embeddings=pos_emb)
                    if isinstance(output, tuple): return output[0] if len(output) > 0 else h_states
                    return output
                return custom_forward
            try:
                batch_output_with_prefix = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(), batch_states_with_prefix, attention_mask_with_prefix, adapted_position_embeddings, use_reentrant=False,
                )
            except RuntimeError as e:
                if "none of output has requires_grad" in str(e):
                    batch_output_with_prefix = self.dinov3_layer(hidden_states=batch_states_with_prefix, attention_mask=None, position_embeddings=adapted_position_embeddings)
                else: raise e
        else:
            batch_output_with_prefix = self.dinov3_layer(hidden_states=batch_states_with_prefix, attention_mask=None, position_embeddings=adapted_position_embeddings)

        batch_output = self._remove_prefix_tokens(batch_output_with_prefix)
        return self._batch_to_flatten(batch_output, cu_seqlens, attention_mask)


# ==============================================================================
# 4. Vision Model
# ==============================================================================

class Qwen3DINOv3VisionModel(Qwen3VLPreTrainedModel):
    config: Qwen3VLVisionConfig
    _no_split_modules = ["DINOv3ViTLayerAdapter"]

    def __init__(self, vision_config: Qwen3VLVisionConfig, dinov3_config: DINOv3ViTConfig, flex_config: Optional[dict] = None, *inputs, **kwargs):
        super().__init__(vision_config, *inputs, **kwargs)
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.patch_size = vision_config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3VLVisionPatchEmbed(config=vision_config)
        self.pos_embed = nn.Embedding(vision_config.num_position_embeddings, vision_config.hidden_size)
        self.num_grid_per_side = int(vision_config.num_position_embeddings**0.5)
        
        head_dim = vision_config.hidden_size // vision_config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        if not hasattr(dinov3_config, '_attn_implementation'):
            dinov3_config._attn_implementation = getattr(dinov3_config, '_attn_implementation', 'flash_attention_2')
            
        self.blocks = nn.ModuleList([DINOv3ViTLayerAdapter(dinov3_config, vision_config) for _ in range(vision_config.depth)])
        self.merger = Qwen3VLVisionPatchMerger(config=vision_config, use_postshuffle_norm=False)

        self.deepstack_visual_indexes = vision_config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList([
            Qwen3VLVisionPatchMerger(config=vision_config, use_postshuffle_norm=True) for _ in range(len(vision_config.deepstack_visual_indexes))
        ])

        self.input_projection = nn.Linear(vision_config.hidden_size, dinov3_config.hidden_size, bias=False)
        self.output_projection = nn.Linear(dinov3_config.hidden_size, vision_config.hidden_size, bias=False)

        if flex_config is None: flex_config = {}
        flex_config.setdefault("backbone_dim", dinov3_config.hidden_size)
        self.flex_config = flex_config
        self.flex_encoder = FlexSceneEncoder(flex_config) if FlexSceneEncoder is not None else None
        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
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
            if num_frames > 1: coords = coords.repeat(num_frames, 1)
            pos_ids[offset : offset + coords.shape[0]] = coords
            offset += coords.shape[0]

        return freq_table[pos_ids].flatten(1)

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
            h_idxs_floor, w_idxs_floor = h_idxs.int(), w_idxs.int()
            h_idxs_ceil, w_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1), (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            dh, dw = h_idxs - h_idxs_floor, w_idxs - w_idxs_floor
            base_h, base_h_ceil = h_idxs_floor * self.num_grid_per_side, h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(), (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(), (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(), ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(), (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device)
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])
        
        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1).permute(0, 1, 3, 2, 4, 5).flatten(0, 4)
            patch_pos_embeds_permute.append(pos_embed)
        return torch.cat(patch_pos_embeds_permute)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        T = self.flex_config.get('num_timesteps', 2)
        C = self.flex_config.get('num_cameras', 6)
        total_images = grid_thw.shape[0]
        
        if total_images % (T * C) == 0:
            B = total_images // (T * C)
        else:
            B, T, C = total_images, 1, 1

        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states + self.fast_pos_embed_interpolate(grid_thw)
        
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        tokens_per_img = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).long()
        cu_seqlens = F.pad(tokens_per_img.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0)

        # Cast input to project layer dtype
        hidden_states = self.input_projection(hidden_states.to(self.input_projection.weight.dtype))

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(block):
                    def custom_forward(*inputs): return block(*inputs)
                    return custom_forward
                try:
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(blk), hidden_states, cu_seqlens, position_embeddings, use_reentrant=False,
                    )
                except RuntimeError:
                    layer_outputs = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings, **kwargs)
            else:
                layer_outputs = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings, **kwargs)
            
            hidden_states = layer_outputs
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature_input = self.output_projection(hidden_states)
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](deepstack_feature_input)
                deepstack_feature_lists.append(deepstack_feature)

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
            hidden_states = self.output_projection(hidden_states)
            hidden_states = self.merger(hidden_states)
            self._flex_batch_size = None

        return hidden_states, deepstack_feature_lists

    def set_gradient_checkpointing(self, value: bool = True):
        self.gradient_checkpointing = value
        for block in self.blocks: block.gradient_checkpointing = value


# ==============================================================================
# 5. Qwen3DINOv3Model (The Intermediate Layer)
# ==============================================================================

class Qwen3DINOv3Model(Qwen3VLPreTrainedModel, GenerationMixin):
    """
    Qwen3-DINOv3 model for conditional generation (vision-language tasks).
    Functions as a drop-in replacement for Qwen3VLModel but with DINOv3 visual encoder.
    """
    
    config_class = Qwen3VLDINOv3ViTConfig # [Compat Fix] Use the user's config class
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config: Qwen3VLDINOv3ViTConfig):
        super().__init__(config)
        
        # 1. Vision Encoder
        if hasattr(config, 'vision_config') and hasattr(config, 'dinov3_config'):
            self.visual = Qwen3DINOv3VisionModel(
                vision_config=config.vision_config, 
                dinov3_config=config.dinov3_config, 
                flex_config=getattr(config, 'flex_config', None)
            )
        else:
            self.visual = None
            
        # 2. Text Model [Compat Fix]
        # Use Qwen3VLTextModel directly to ensure mRoPE and DeepStack support
        self.language_model = Qwen3VLTextModel(config.text_config)
        
        # Token IDs
        self.image_token_id = getattr(config, 'image_token_id', 151655)
        self.video_token_id = getattr(config, 'video_token_id', 151656)
        self.vision_start_token_id = getattr(config, 'vision_start_token_id', 151652)
        
        self.rope_deltas = None
        self.post_init()

    def get_input_embeddings(self): return self.language_model.embed_tokens
    def set_input_embeddings(self, value): self.language_model.embed_tokens = value
    def get_image_features(self, pixel_values, image_grid_thw):
        if self.visual is None: raise ValueError("Vision encoder not initialized")
        return self.visual(pixel_values, grid_thw=image_grid_thw)

    # [Compat Fix] Ported logic from Qwen3VLModel for 3D RoPE
    def get_rope_index(self, input_ids, image_grid_thw, attention_mask=None, video_grid_thw=None):
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.image_token_id
        video_token_id = self.video_token_id
        vision_start_token_id = self.vision_start_token_id
        mrope_position_deltas = []
        
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            if attention_mask is None: attention_mask = torch.ones_like(input_ids)
            position_ids = torch.ones(3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device)
            image_index, video_index = 0, 0
            
            for i, seq_input_ids in enumerate(input_ids):
                seq_input_ids = seq_input_ids[attention_mask[i] == 1]
                vision_start_indices = torch.argwhere(seq_input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = seq_input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = seq_input_ids.tolist()
                llm_pos_ids_list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0: ed_image = input_tokens.index(image_token_id, st)
                    else: ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0: ed_video = input_tokens.index(video_token_id, st)
                    else: ed_video = len(input_tokens) + 1
                    
                    if ed_image < ed_video:
                        t, h, w = image_grid_thw[image_index]
                        image_index += 1; remain_images -= 1; ed = ed_image
                    else:
                        t, h, w = video_grid_thw[video_index]
                        video_index += 1; remain_videos -= 1; ed = ed_video
                        
                    llm_grid_t, llm_grid_h, llm_grid_w = t.item(), h.item() // spatial_merge_size, w.item() // spatial_merge_size
                    text_len = ed - st
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                    
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
                mrope_position_deltas.append(llm_positions.max() + 1 - len(seq_input_ids))
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
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
                mrope_position_deltas = torch.zeros([input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype)
            return position_ids, mrope_position_deltas

    def get_placeholder_mask(self, input_ids, inputs_embeds, image_features=None, video_features=None):
        if input_ids is None:
            special_image_mask = (inputs_embeds == self.get_input_embeddings()(torch.tensor(self.image_token_id, dtype=torch.long, device=inputs_embeds.device))).all(-1)
            special_video_mask = (inputs_embeds == self.get_input_embeddings()(torch.tensor(self.video_token_id, dtype=torch.long, device=inputs_embeds.device))).all(-1)
        else:
            special_image_mask = input_ids == self.image_token_id
            special_video_mask = input_ids == self.video_token_id
        
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        return special_image_mask, special_video_mask

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
        video_grid_thw: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        
        if inputs_embeds is None: inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None
        deepstack_image_embeds = []
        deepstack_video_embeds = []

        if pixel_values is not None and image_grid_thw is not None and self.visual is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None and video_grid_thw is not None and self.visual is not None:
             pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
             video_embeds, deepstack_video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
             video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
             _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds, video_features=video_embeds)
             inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None or video_mask is not None:
            if image_mask is not None: image_mask = image_mask[..., 0]
            if video_mask is not None: video_mask = video_mask[..., 0]
            
            if image_mask is not None and video_mask is not None:
                visual_pos_masks = image_mask | video_mask
                deepstack_visual_embeds = []
                # Mix deepstack features
                for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                    embed_joint = torch.zeros(visual_pos_masks.sum(), img_embed.shape[-1], device=img_embed.device, dtype=img_embed.dtype)
                    embed_joint[image_mask[visual_pos_masks]] = img_embed
                    embed_joint[video_mask[visual_pos_masks]] = vid_embed
                    deepstack_visual_embeds.append(embed_joint)
            elif image_mask is not None:
                visual_pos_masks = image_mask
                deepstack_visual_embeds = deepstack_image_embeds
            else:
                visual_pos_masks = video_mask
                deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            attention_mask_tensor = attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                if attention_mask_tensor.dtype.is_floating_point: attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            if not hasattr(self, 'rope_deltas'): self.rope_deltas = None
            is_prefill = (input_ids is not None and input_ids.shape[1] != 1) or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            
            if is_prefill or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw, attention_mask=attention_mask_tensor, video_grid_thw=video_grid_thw)
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device).view(1, -1).expand(batch_size, -1)
                if hasattr(delta, 'shape') and len(delta.shape) > 0: delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta).unsqueeze(0).expand(3, -1, -1)

        input_ids = None 
        # [Compat Fix] Delegate directly to Qwen3VLTextModel which supports deepstack args
        outputs = self.language_model(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
            cache_position=cache_position, visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds, **kwargs,
        )
        return outputs

# ==============================================================================
# 6. Wrapper Class (Matches save script usage)
# ==============================================================================

class Qwen3VLForConditionalGenerationWithDINOv3(Qwen3VLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    config_class = Qwen3VLDINOv3ViTConfig # [Compat Fix]
    
    def __init__(self, config: Qwen3VLDINOv3ViTConfig):
        super().__init__(config)
        self.model = Qwen3DINOv3Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self): return self.model.language_model.embed_tokens
    def set_input_embeddings(self, value): self.model.language_model.embed_tokens = value
    def get_output_embeddings(self): return self.lm_head
    def set_output_embeddings(self, new_embeddings): self.lm_head = new_embeddings
    
    @property
    def visual(self): return self.model.visual
    @property
    def language_model(self): return self.model.language_model

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
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
            cache_position=cache_position, pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kwargs
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # config.vocab_size usually refers to text vocab
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs)

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values,
            rope_deltas=self.model.rope_deltas,  
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, position_ids=None, use_cache=True, pixel_values=None, image_grid_thw=None, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask,
            inputs_embeds=inputs_embeds, cache_position=cache_position, position_ids=position_ids, use_cache=use_cache, **kwargs
        )
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
    "Qwen3VLForConditionalGenerationWithDINOv3",
]