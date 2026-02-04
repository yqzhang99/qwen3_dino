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

"""Qwen3VL-DINOv3 mixed model configuration"""

from typing import Optional, Union, Dict, Any

from transformers.configuration_utils import PretrainedConfig

from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLVisionConfig


class Qwen3VLDINOv3ViTConfig(Qwen3VLConfig):
    """
    Extension of Qwen3VLConfig with DINOv3-related configuration.
    
    This configuration class stores the configuration for a Qwen3VL-DINOv3 mixed model,
    where DINOv3ViTLayer replaces Qwen3VLVisionBlock in the vision encoder.
    """

    model_type = "qwen3_vl_svit"

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        dinov3_config: Optional[Union[DINOv3ViTConfig, dict]] = None,
        use_dinov3_backbone: bool = True,
        flex_config: Optional[Dict[str, Any]] = None,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        tie_word_embeddings=False,
        **kwargs,
    ):
        super().__init__(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_token_id,
            vision_end_token_id=vision_end_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # Handle vision_config and text_config if they're existing objects (not dict, not None)
        # Qwen3VLConfig.__init__ only sets these if they're dict or None
        if vision_config is not None and not isinstance(vision_config, dict):
            self.vision_config = vision_config
        if text_config is not None and not isinstance(text_config, dict):
            self.text_config = text_config

        # Handle DINOv3 configuration
        if dinov3_config is None:
            # Use default DINOv3 config (aligned to Qwen3VL)
            self.dinov3_config = self._create_default_dinov3_config()
        elif isinstance(dinov3_config, dict):
            self.dinov3_config = DINOv3ViTConfig(**dinov3_config)
        else:
            self.dinov3_config = dinov3_config

        if flex_config is None:
            self.flex_config = {
                "num_cameras": 6,
                "num_timesteps": 2,
                "num_scene_tokens": 900,
                "hidden_dim": 768,
                "num_layers": 8,
                "dropout": 0.1,
            }
        else:
            self.flex_config = flex_config

        if self.dinov3_config:
            self.flex_config["backbone_dim"] = self.dinov3_config.hidden_size

        if hasattr(self, "hidden_size"):
             self.flex_config["llm_dim"] = self.hidden_size
        elif self.text_config and hasattr(self.text_config, "hidden_size"):
             self.flex_config["llm_dim"] = self.text_config.hidden_size

        self.use_dinov3_backbone = use_dinov3_backbone

        # Validate configuration compatibility
        self._validate_config_compatibility()

    def _create_default_dinov3_config(self) -> DINOv3ViTConfig:
        """
        Create default DINOv3 configuration aligned to Qwen3VL.
        
        Returns:
            DINOv3ViTConfig: Default DINOv3 configuration with parameters aligned to Qwen3VL.
        """
        # Inherit _attn_implementation from vision_config if available, otherwise use "eager"
        # This ensures DINOv3 uses the same attention implementation as Qwen3VL vision model
        # 
        # _attn_implementation options and their impacts:
        # - "eager": PyTorch native (default, most compatible, lower performance)
        # - "sdpa": Scaled Dot-Product Attention (better performance, requires PyTorch 2.0+)
        # - "flash_attention_2": Flash Attention v2 (best performance, requires flash-attn library and Ampere+ GPU)
        # - "flex_attention": Flex Attention (if supported, check transformers version)
        #
        # By inheriting from vision_config, we maintain consistency with the Qwen3VL model's
        # attention implementation, which is typically set during model loading via PreTrainedModel.__init__()
        # 
        # Note: getattr returns None if the attribute exists but is None, so we need to check for None
        # and fallback to "eager" to avoid KeyError in DINOv3ViTAttention.forward
        attn_impl = getattr(self.vision_config, "_attn_implementation", "eager")
        attn_impl = attn_impl if attn_impl is not None else "eager"
        
        return DINOv3ViTConfig(
            # Must match Qwen3VL
            hidden_size=self.vision_config.hidden_size,  # 1152
            num_attention_heads=self.vision_config.num_heads,  # 16
            patch_size=self.vision_config.patch_size,  # 16
            num_channels=3,  # RGB images

            # Recommended configuration (confirmed)
            intermediate_size=4608,  # 4x relationship (recommended) or 4304 (aligned to Qwen3VL)
            hidden_act="gelu_pytorch_tanh",  # Aligned to Qwen3VL (confirmed)
            layer_norm_eps=1e-6,  # Aligned to Qwen3VL

            # Register tokens (confirmed: not used)
            num_register_tokens=0,  # Not used, keep consistent with Qwen3VL

            # Layer Scale and Drop Path (confirmed: keep)
            layerscale_value=1.0,  # Keep Layer Scale, initial value 1.0
            drop_path_rate=0.0,  # Not used initially, can be adjusted to 0.1 later

            # RoPE configuration (although we use Qwen3VL's position encoding, keep consistent)
            rope_theta=10000.0,  # Aligned to Qwen3VL's visual RoPE

            # Attention implementation: inherit from vision_config, fallback to "eager"
            # This parameter controls the attention backend used in DINOv3ViTAttention
            # - "eager": Default, works everywhere but slower
            # - "sdpa": Better performance on modern GPUs (PyTorch 2.0+)
            # - "flash_attention_2": Best performance on compatible NVIDIA GPUs (requires flash-attn)
            # Inheriting from vision_config ensures consistency with Qwen3VL's attention implementation
            attn_implementation=attn_impl,

            # Other configuration
            num_hidden_layers=1,  # Single layer, create 27 layers through ModuleList
            attention_dropout=0.0,  # Aligned to Qwen3VL
            initializer_range=0.02,  # Standard initialization
            use_gated_mlp=False,  # Not use gated MLP
        )

    def _validate_config_compatibility(self):
        """
        Validate configuration compatibility between Qwen3VL and DINOv3.
        
        Raises:
            AssertionError: If configurations are incompatible.
        """
        # assert (
        #     self.dinov3_config.hidden_size == self.vision_config.hidden_size
        # ), f"DINOv3 hidden_size ({self.dinov3_config.hidden_size}) must match Qwen3VL vision_config.hidden_size ({self.vision_config.hidden_size})"

        # assert (
        #     self.dinov3_config.num_attention_heads == self.vision_config.num_heads
        # ), f"DINOv3 num_attention_heads ({self.dinov3_config.num_attention_heads}) must match Qwen3VL vision_config.num_heads ({self.vision_config.num_heads})"

        assert (
            self.dinov3_config.patch_size == self.vision_config.patch_size
        ), f"DINOv3 patch_size ({self.dinov3_config.patch_size}) must match Qwen3VL vision_config.patch_size ({self.vision_config.patch_size})"

    @classmethod
    def from_configs(cls, 
                    qwen3vl_config: Optional[Qwen3VLConfig] = None,
                    dinov3_config: Optional[DINOv3ViTConfig] = None,
                    flex_config: Optional[Dict[str, Any]] = None,
                    **kwargs) -> "Qwen3VLDINOv3ViTConfig":
        """
        Create Qwen3VLDINOv3ViTConfig from existing Qwen3VLConfig and optionally DINOv3ViTConfig.
        
        Args:
            qwen3vl_config: Existing Qwen3VLConfig instance (optional if vision_config provided in kwargs).
            dinov3_config: Existing DINOv3ViTConfig instance (optional).
            **kwargs: Additional arguments to override.
        
        Returns:
            Qwen3VLDINOv3ViTConfig: New configuration instance.
        """
        # Extract parameters from Qwen3VL config if provided
        if qwen3vl_config is not None:
            qwen3vl_kwargs = {
                "text_config": qwen3vl_config.text_config,
                "vision_config": qwen3vl_config.vision_config,
                "image_token_id": qwen3vl_config.image_token_id,
                "video_token_id": qwen3vl_config.video_token_id,
                "vision_start_token_id": qwen3vl_config.vision_start_token_id,
                "vision_end_token_id": qwen3vl_config.vision_end_token_id,
                "tie_word_embeddings": qwen3vl_config.tie_word_embeddings,
            }
        else:
            # If no Qwen3VL config provided, rely on kwargs for essential parameters
            qwen3vl_kwargs = {}
            
        # Add DINOv3 config if provided
        if dinov3_config is not None:
            if not hasattr(dinov3_config, '_attn_implementation') or dinov3_config._attn_implementation is None:
                dinov3_config._attn_implementation = 'eager'  # flash_attention_2? sdpa?
            if not hasattr(dinov3_config, 'attn_implementation') or dinov3_config.attn_implementation is None:
                dinov3_config.attn_implementation = 'eager' # flash_attention_2? sdpa?
            kwargs["dinov3_config"] = dinov3_config
        
        if flex_config is not None:
            kwargs["flex_config"] = flex_config

        # Merge all parameters
        final_kwargs = {**qwen3vl_kwargs, **kwargs}
        
        return cls(**final_kwargs)
    @classmethod
    def from_qwen3vl_config(cls, qwen3vl_config: Qwen3VLConfig, **kwargs) -> "Qwen3VLDINOv3ViTConfig":
        """
        Create Qwen3VLDINOv3ViTConfig from existing Qwen3VLConfig.
        
        Args:
            qwen3vl_config: Existing Qwen3VLConfig instance.
            **kwargs: Additional arguments to override.
        
        Returns:
            Qwen3VLDINOv3ViTConfig: New configuration instance.
        """
        return cls.from_configs(qwen3vl_config=qwen3vl_config, **kwargs)


__all__ = ["Qwen3VLDINOv3ViTConfig"]

