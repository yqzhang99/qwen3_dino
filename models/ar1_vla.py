"""
ar1 -vla 
"""

import math
from typing import Any, Callable, Optional, Union
from ray import method
from transformers import (
    AutoModel, 
    AutoConfig,
)

import copy
import logging
from typing import Any

import einops
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.generation import GenerationMixin

from transformers import Qwen3VLTextModel

from model.utils.token_utils import (
    StopAfterEOS,
    replace_padding_after_eos,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers import (
    AutoProcessor,
    PretrainedConfig,
    PreTrainedModel,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen2Tokenizer
)
from transformers.modeling_outputs import ModelOutput

from model.diffusion import CondOTProbPath
from model.diffusion.solver import ode_solver
from extras.action_space import DxyActionSpace,UnicycleAccelCurvatureActionSpace
from dataclasses import dataclass
from overwatch import initialize_overwatch
from .qwen3vl_dinov3vit import Qwen3VLForConditionalGenerationWithDINOv3, Qwen3VLDINOv3ViTConfig

from transformers import (
    AutoConfig,
    AutoModel,
)

AutoConfig.register("qwen3_vl_svit", Qwen3VLDINOv3ViTConfig)
AutoModel.register(Qwen3VLDINOv3ViTConfig, Qwen3VLForConditionalGenerationWithDINOv3)

overwatch = initialize_overwatch(__name__)
def integration_loss_t2(pred: torch.Tensor, real: torch.Tensor, method = "sum") -> torch.Tensor:
    B, N, D = pred.shape
    # 蕴涵了一步假设，x0 + v = x1 , 所以，如果能让fm越接近一步推理，这种设计的效果应该会越好。 
    pred_traj = torch.cumsum(pred, dim=1)
    real_traj = torch.cumsum(real, dim=1)

    if method == "sum":
        return  (pred_traj - real_traj).norm(dim =2).sum(dim=1).mean()
    elif method == "mean":
        return  (pred_traj - real_traj).norm(dim =2).mean()
    elif method == "mean_split":
        # return  (pred_traj - real_traj)**2.mean()
        return 0.5 * F.mse_loss(pred_traj, real_traj) + 0.5 * F.mse_loss(pred_traj.norm(dim = 1), real_traj.norm(dim = 1))

def total_loss_dynamic_scaled(v_pred, v_real, L0=1.0, k=10.0, method='sum'):
    """
    v_pred, v_real: (B,N,D)
    L0: diff_loss 阈值，用于控制初期压制强度
    k: sigmoid陡度
    method: 'linear' 或 'sigmoid'
    """
    traj_loss = integration_loss_t2(v_pred, v_real) # 主loss
    if method == "sum":
        diff_loss = (v_pred - v_real).norm(dim=2).sum(dim=1).mean()  # 辅助loss， 尽可能让v贴近
        return diff_loss, traj_loss
    elif method == "mean":
        diff_loss = (v_pred - v_real).norm(dim = 2).mean()
        return diff_loss, traj_loss
    elif method == "mean_split": # 使用均值且同时对 dx, dy 分别处理 
        # diff_loss = (v_pred - v_real).mean() # dx, dy 需要分别加上各自的累加loss 
        diff_loss = 0.5 * F.mse_loss(v_pred, v_real) + 0.5 * F.mse_loss(v_pred.norm(dim = 1),v_real.norm(dim = 1))
        return diff_loss, traj_loss



def weighted_mse(v_pred, v_real, L0 = 1.0, method = "mean"):
    """
    v_pred, v_real: (B,N,D)
    L0: diff_loss 阈值，用于控制初期压制强度
    k: sigmoid陡度
    method: 'linear' 或 'sigmoid'
    """
    
    diff = (v_pred - v_real)  # (B,N,T)
    B,N,T = diff.shape
    
    weighted_array = torch.arange(N,0,-1)
    weighted_array = weighted_array.to(diff.device)
    diff = diff * weighted_array[None,:,None]  # (B,N)

    diff_norm =  (v_pred.norm(dim = 2) - v_real.norm(dim = 2))

    diff_norm = diff_norm * weighted_array[None,:]



    if method == "mean":
        # return 0.5 * ((diff**2).mean(dim = (1,2))).mean() + 0.5 * ((diff_norm**2).mean(dim = 1)).sqrt().mean()
        return 0.5 * (diff**2).mean() + 0.5 * (diff_norm**2).mean()
    elif method == "sum":
        return (diff.sum(dim = 1)**2).mean()




@dataclass
class AR1MOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


# Action Proj
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """Normalize the input tensor."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """Normalize the input tensor."""
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class MLPEncoder(nn.Module):
    """Basic MLP encoder."""

    def __init__(self, num_input_feats: int, num_enc_layers: int, hidden_size: int, outdim: int):
        super().__init__()
        assert 1 <= num_enc_layers, f"{num_enc_layers=} must be >= 1"

        enc_layers = [
            nn.Linear(num_input_feats, hidden_size),
            nn.SiLU(),
        ]
        for layeri in range(num_enc_layers):
            if layeri < num_enc_layers - 1:
                enc_layers.extend(
                    [
                        RMSNorm(hidden_size, eps=1e-5),
                        nn.Linear(hidden_size, hidden_size),
                        nn.SiLU(),
                    ]
                )
            else:
                enc_layers.extend(
                    [
                        RMSNorm(hidden_size, eps=1e-5),
                        nn.Linear(hidden_size, outdim),
                    ]
                )

        self.trunk = nn.Sequential(*enc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C) -> (B, outdim)"""
        return self.trunk(x)


class FourierEncoderV2(nn.Module):
    """Improved Fourier feature encoder with logarithmically-spaced frequencies."""

    def __init__(self, dim: int, max_freq: float = 100.0):
        """Initialize the Fourier encoder V2.

        Args:
            dim: Output dimension of the encoder. Must be even as it's split into
                sine and cosine components.
            max_freq: Maximum frequency for the logarithmic frequency spacing.
                Defaults to 100.0.
        """
        super().__init__()
        half = dim // 2  # 
        freqs = torch.logspace(0, math.log10(max_freq), steps=half) # 对数间隔频率 [0,..., max_freq]
        self.out_dim = dim
        self.register_buffer("freqs", freqs[None, :])  # (1, half)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Fourier encoder V2.

        Args:
            x: Input tensor of arbitrary shape (..., ).

        Returns:
            Fourier-encoded features of shape (..., dim).
            
        sqrt(2) [cos(2pif), sin(2pif)]
        """
        arg = x[..., None] * self.freqs * 2 * torch.pi  # (*, half_dim) [[2pif0,...,2pifn/2]..]
        return torch.cat([torch.sin(arg), torch.cos(arg)], -1) * math.sqrt(2) # sqrt(2) ([sin2pif0... , cos2pif0])


class PerWaypointActionInProjV2(torch.nn.Module):
    """Improved per-waypoint action input projection module.
    逐点action 投影, 将action + timestamp 投影到高维表示

    It uses FourierEncoderV2 with logarithmically-spaced frequencies and includes layer normalization. Projects
    action sequences with timestep information into a higher-dimensional representation.
    """

    def __init__(
        self,
        in_dims: list[int],
        out_dim: int,
        num_enc_layers: int = 4,
        hidden_size: int = 1024,
        max_freq: float = 100.0,
        num_fourier_feats: int = 20,
    ):
        """Initialize the per-waypoint action projection module V2.

        Args:
            in_dims: List of input dimensions. The last element specifies the number
                of action dimensions to encode separately.  最后一个维度指定要单独编码的动作维度数量
            out_dim: Output dimension of the projection.
            num_enc_layers: Number of layers in the MLP encoder. Defaults to 4.
            hidden_size: Hidden dimension size of the MLP encoder. Defaults to 1024.
            max_freq: Maximum frequency for the Fourier encoding. Defaults to 100.0.
            num_fourier_feats: Number of Fourier features for encoding. Defaults to 20.
        """
        super().__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        sinus = []
        for _ in range(in_dims[-1]): # 对需要单独编码的动作每个进行编码
            sinus.append(FourierEncoderV2(dim=num_fourier_feats, max_freq=max_freq)) # 每个动作特征维度
        self.sinus = nn.ModuleList(sinus)
        self.timestep_fourier_encoder = FourierEncoderV2(dim=num_fourier_feats, max_freq=max_freq) # 时间戳的编码
        num_input_feats = sum(s.out_dim for s in self.sinus) + self.timestep_fourier_encoder.out_dim # 总特征输出维度求和
        self.encoder = MLPEncoder(
            num_input_feats=num_input_feats,
            num_enc_layers=num_enc_layers,
            hidden_size=hidden_size,
            outdim=out_dim,
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass of the per-waypoint action projection V2.

        Args:
            x: Action tensor of shape (batch_size, num_waypoints, action_dim).
            timesteps: Timestep tensor of shape (batch_size, ...). The last dimension
                is used for encoding.

        Returns:
            Normalized projected action features of shape
            (batch_size, num_waypoints, out_dim).
        """
        B, T, _ = x.shape # 每个样本 ，就1条轨迹， 省略，直接给点数？
        action_feats = torch.cat([s(x[:, :, i]) for i, s in enumerate(self.sinus)], dim=-1) # 每层动作特征编码拼接 B,T, sum of feat, B,N -> B,N -> B,N,1,sum(feat)
        timestep_feats = self.timestep_fourier_encoder(timesteps[..., -1]) # B,N -> B -> B,1, feat
        # print("ttf:",timestep_feats)
        timestep_feats = timestep_feats.repeat(B, T, 1) # TODO:需要检查下之前的实现
        # print("tfs:",timestep_feats.shape)
        x = torch.cat((action_feats, timestep_feats), dim=-1)
        return self.norm(self.encoder(x.flatten(0, 1)).reshape(B, T, -1))


# 轨迹相关词的分/logic --> -inf , 强制 llm 对这些 action token 词不感兴趣-->-inf
class ExpertLogitsProcessor(LogitsProcessor):
    """Masks out the logits for discrete trajectory tokens."""

    def __init__(self, traj_token_offset: int, traj_vocab_size: int):
        """Initialize the ExpertLogitsProcessor.

        Args:
            traj_token_offset: The offset of the trajectory tokens.
            traj_vocab_size: The vocabulary size of the trajectory tokens.
        """
        super().__init__()
        self.traj_token_offset = traj_token_offset
        self.traj_vocab_size = traj_vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Call the ExpertLogitsProcessor to mask out the logits for discrete trajectory tokens.

        The discrete trajectory tokens are not used for the expert model thus masking them out for
        better CoC generation.

        Args:
            input_ids: The input IDs.
            scores: The scores.

        Returns:
            torch.FloatTensor: The modified scores tensor with trajectory tokens masked out (set to -inf).
        """
        # Directly assign -inf to the trajectory token positions in the scores tensor
        scores[:, self.traj_token_offset : self.traj_token_offset + self.traj_vocab_size] = float('-inf')
        return scores

class CacheMapper(nn.Module):
    """Map VLM past_key_values (per-layer k/v) to expert past_key_values using a
    learned gating matrix and optional per-head projections.

    The mapper expects past_key_values as a tuple of length vlm_layers where
    each element is a (k, v) pair with shape (batch, n_heads, seq_len, head_dim).
    It returns a tuple of length expert_layers with mapped (k, v) tensors.
    """

    def __init__(self, vlm_layers: int, expert_layers: int, vlm_head_dim: int, expert_head_dim: int):
        super().__init__()
        self.vlm_layers = vlm_layers
        self.expert_layers = expert_layers
        # gating logits: will be softmaxed over vlm_layers for each expert layer
        self.gating_logits = nn.Parameter(torch.zeros(expert_layers, vlm_layers))

        # If head dims differ, project last dimension after weighted-sum
        if vlm_head_dim != expert_head_dim:
            self.k_proj = nn.Linear(vlm_head_dim, expert_head_dim, bias=False)
            self.v_proj = nn.Linear(vlm_head_dim, expert_head_dim, bias=False)
        else:
            self.k_proj = None
            self.v_proj = None

    def forward(self, past_key_values: tuple) -> tuple:
        # stack across layer dim -> (vlm_layers, batch, n_heads, seq_len, head_dim)
        ks = torch.stack([pv[0] for pv in past_key_values], dim=0)
        vs = torch.stack([pv[1] for pv in past_key_values], dim=0)

        # gating weights: (expert_layers, vlm_layers)
        gating = F.softmax(self.gating_logits, dim=-1)

        mapped = []
        for e in range(self.expert_layers):
            w = gating[e].view(self.vlm_layers, 1, 1, 1, 1).to(ks.dtype)
            # weighted sum over vlm layers
            k_e = (ks * w).sum(dim=0)
            v_e = (vs * w).sum(dim=0)

            if self.k_proj is not None:
                b, h, s, d = k_e.shape
                k_e = self.k_proj(k_e.reshape(-1, d)).reshape(b, h, s, -1)
                v_e = self.v_proj(v_e.reshape(-1, d)).reshape(b, h, s, -1)

            mapped.append((k_e, v_e))

        return tuple(mapped)



class AlpamayoR1Config(PretrainedConfig):
    model_type = "alpamayo_r1"

    def __init__(self,     
        mode:str = "AR",   
        diffusion_cfg: dict[str, Any] | None = None,
        action_space_cfg: dict[str, Any] | None = None,
        action_in_proj_cfg: dict[str, Any] | None = None,
        action_out_proj_cfg: dict[str, Any] | None = None,
        expert_cfg: dict[str, Any] | None = None,
        keep_same_dtype: bool = True,
        expert_non_causal_attention: bool = True,
        enable_cache_mapping: bool = False,
        vlm_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct",
        vlm_backend: str = "qwenvl3",
        traj_tokenizer_cfg: dict[str, Any] | None = None,
        hist_traj_tokenizer_cfg: dict[str, Any] | None = None,
        traj_vocab_size: int = 768,
        tokens_per_history_traj: int = 20,
        tokens_per_future_traj: int = 60,
        model_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        add_special_tokens: bool = False,
        traj_future_start_token_id: int = -1,
        flow_match_cfg: dict[str,Any] | None = None,
        pad_token_id: int | None = None,
        # diff_loss_weight: float = 1,
        freeze_vlm_in_fm:bool = True,
        # cumsum_loss_weight: float = 1,    
        **kwargs: Any):
        super().__init__(**kwargs)
        self.mode = mode
        self.diffusion_cfg = diffusion_cfg # diffusion cfg 
        self.action_space_cfg = action_space_cfg # action space cfg 
        self.action_in_proj_cfg = action_in_proj_cfg # action in projection cfg
        self.action_out_proj_cfg = action_out_proj_cfg # action out projection cfg
        self.expert_cfg = expert_cfg # expert cfg
        self.keep_same_dtype = keep_same_dtype # keep same dtype
        self.expert_non_causal_attention = expert_non_causal_attention # expert non causal attention
        self.flow_match_cfg = flow_match_cfg
        # whether to enable learned gating-based cache mapping from VLM to expert
        self.enable_cache_mapping = enable_cache_mapping
        
        # VLM cfg
        self.vlm_name_or_path = vlm_name_or_path
        self.vlm_backend = vlm_backend.lower()
        self.model_dtype = model_dtype
        self.attn_implementation = attn_implementation

        self.traj_tokenizer_cfg = traj_tokenizer_cfg
        self.hist_traj_tokenizer_cfg = hist_traj_tokenizer_cfg

        self.traj_vocab_size = traj_vocab_size
        self.tokens_per_history_traj = tokens_per_history_traj
        self.tokens_per_future_traj = tokens_per_future_traj
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.add_special_tokens = add_special_tokens
        self.vocab_size = kwargs.get("vocab_size", None)  # to be updated later
        self.traj_future_start_token_id = traj_future_start_token_id

        self.pad_token_id = pad_token_id
        self.freeze_vlm_in_fm = freeze_vlm_in_fm

        # loss cfg
        # self.diff_loss_weight = diff_loss_weight
        # self.cumsum_loss_weight = cumsum_loss_weight
     
     
    
class AlpamayoR1Model(PreTrainedModel,GenerationMixin):

    config_class: type[AlpamayoR1Config] = AlpamayoR1Config
    base_model_prefix:str = "alpamayo_r1"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    accepts_loss_kwargs = False
    
    # _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {}

    def __init__(self, config: AlpamayoR1Config, pretrained_modules = None,original_vocab_size = None):
        super().__init__(config)
        overwatch.info(f"AlpamayoR1Config:{config}")
        # print("pretrained modules:", pretrained_modules.keys())
        # 构建vla模型，基于配置构建实现，需要,TODO: 注意后续模型的加载，需要从训练后的模型权重中进行加载。
        if pretrained_modules is not None:
            self.vlm = pretrained_modules["vlm"] 
            self.original_vocab_size = original_vocab_size
        else:
            vlm_config = AutoConfig.from_pretrained(
                "/mnt/data/models/qwen3VL-DINOv3-Flex-2B-Instruct-200",
                dtype = config.model_dtype,
                attn_implementation = config.attn_implementation
            )
            self.original_vocab_size = vlm_config.text_config.vocab_size
            vlm_config.text_config.vocab_size = config.vocab_size % 64 == 0 and config.vocab_size or (config.vocab_size + 64 - config.vocab_size % 64) 
            vlm_config.vocab_size = config.vocab_size
            self.vlm = Qwen3VLForConditionalGenerationWithDINOv3(vlm_config)

        # 基于配置构建fm expert 
        if config.mode == "FM":
            # 手动启用vlm的梯度检查点 
            self.vlm.supports_gradient_checkpointing = False 
            
            expert_config = copy.deepcopy(self.vlm.config.text_config)
            if config.expert_cfg is not None:
                expert_config.update(config.expert_cfg)
            #TODO: 这种方式构建的，不包含预训练参数。
            self.expert = AutoModel.from_config(expert_config)
            del self.expert.embed_tokens

            # build cache mapper: map vlm layers -> expert layers (optionally)
            # get vlm layer/attention dims from vlm config
            try:
                vlm_text_cfg = self.vlm.config.text_config
            except Exception:
                vlm_text_cfg = self.vlm.config
            self.vlm_hidden_layers = getattr(vlm_text_cfg, "num_hidden_layers", None)
            self.vlm_hidden_size = getattr(vlm_text_cfg, "hidden_size", getattr(vlm_text_cfg, "n_embd", None))
            self.vlm_num_heads = getattr(vlm_text_cfg, "num_attention_heads", None)

            # expert dims
            expert_hidden_layers = getattr(expert_config, "num_hidden_layers", None)
            expert_hidden_size = getattr(expert_config, "hidden_size", None)
            expert_num_heads = getattr(expert_config, "num_attention_heads", None)

            if self.vlm_hidden_layers is None:
                raise ValueError("Unable to determine VLM number of hidden layers for cache mapping")

            if self.vlm_num_heads is None or expert_num_heads is None:
                # fallback: assume same head count -> compute head dim from hidden sizes
                vlm_head_dim = self.vlm_hidden_size // (getattr(vlm_text_cfg, "num_attention_heads", 1) or 1)
                expert_head_dim = expert_hidden_size // (expert_num_heads or 1)
            else:
                vlm_head_dim = self.vlm_hidden_size // self.vlm_num_heads
                expert_head_dim = expert_hidden_size // expert_num_heads

            if self.config.enable_cache_mapping:
                self.cache_mapper = CacheMapper(
                    vlm_layers=self.vlm_hidden_layers,
                    expert_layers=expert_hidden_layers,
                    vlm_head_dim=vlm_head_dim,
                    expert_head_dim=expert_head_dim,
                )
            else:
                self.cache_mapper = None
            
            #TODO: 手动配置 action配置
            # self.action_space = UnicycleAccelCurvatureActionSpace(**config.action_space_cfg)
            self.action_space = DxyActionSpace(**config.action_space_cfg)

            self.x_dims=self.action_space.get_action_space_dims()
            # 类型问题，需要处理一下,
            
            self.flow_match = CondOTProbPath(**config.flow_match_cfg,dtype = self.expert.dtype)
            self.ode_solver = ode_solver.ODESolver
            
            # Todo: action proj
            self.action_in_proj = PerWaypointActionInProjV2(**config.action_in_proj_cfg,in_dims=self.x_dims, # [n,2]
                out_dim=expert_config.hidden_size)
            
            # TODO： action_out_proj
            self.action_out_proj = torch.nn.Linear(in_features=expert_config.hidden_size,out_features=self.x_dims[-1])
            
            expert_dtype = self.expert.dtype
            if self.config.keep_same_dtype:
                self.flow_match = self.flow_match.to(dtype=expert_dtype)
                self.action_in_proj = self.action_in_proj.to(dtype=expert_dtype)
                self.action_out_proj = self.action_out_proj.to(dtype=expert_dtype)
                # also cast mapper if exists
                if getattr(self, "cache_mapper", None) is not None:
                    self.cache_mapper = self.cache_mapper.to(dtype=expert_dtype)
                
            self.traj_future_start_token_id = self.config.traj_future_start_token_id # TODO: vlm生成时候的轨迹开始token

            self.stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id = self.traj_future_start_token_id)])
            self.logits_processor = LogitsProcessorList(
                [
                    ExpertLogitsProcessor(traj_token_offset=self.config.traj_future_start_token_id,
                                          traj_vocab_size=self.config.traj_vocab_size)  # 轨迹开始token,轨迹数量
                ]
            )
            
            
            
        self.post_init()
        


    """
    模型需要的输入:
    input_ids: 分词后的token id
    attention_mask: 注意力mask
    position_ids: 位置编码
    past_key_values: 上一时刻的key和value
    inputs_embeds: 输入的embedding
    labels: 目标标签
    pixel_values: 图像的像素值
    pixel_values_videos: 视频的像素值
    image_grid_thw: 图像的网格
    video_grid_thw: 视频的网格
    cache_position: 缓存的位置
    logits_to_keep: 保留的logits
    **kwargs: 其他参数
    his_traj: 需要基于历史轨迹计算acc / kappa 
    cur_traj: 计算 acc / kappa 作为fm 标签 
    
    # 数据集处理时候的输入应该包括:
    input_ids
    his_traj
    fut_traj
    """    
            
    def forward_ar(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
        ):
        """AR 训练直接出token，不使用expert head, tokenizer 是新增的， 模型本身需要预处理一下，修改词表size """
        # print("input ids:",str(input_ids.cpu().numpy().tolist()))
        # print("labels  : ",str(kwargs["labels"].cpu().numpy().tolist()))
        output =  self.vlm(input_ids, attention_mask=attention_mask, position_ids=position_ids,
                        past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                        pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
                        image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                        cache_position=cache_position, **kwargs)
        # print(output)
        return output

    # 先处理数据
    def forward_fm(self,
                   history_xyzyaw: torch.FloatTensor = None,
                   future_xyzyaw: torch.FloatTensor = None,
                   t0_state: torch.FloatTensor = None,
                   **kwargs
                   ):
        device = kwargs.get('input_ids').device
        input_ids = kwargs.get('input_ids')
        # Extract batch information from input_ids
        b_star = input_ids.shape[0]
        # TODO： 考虑实际使用场景，目前假设输入数据格式完全一致，因此不需要额外做pading， 各个样本长度必须是对齐的。
        
        # Check if traj_future_start_token_id exists and truncate input_ids accordingly
        traj_future_start_mask = input_ids == self.traj_future_start_token_id
        has_traj_future_start = traj_future_start_mask.any(dim=1)
        
        # Find truncation positions using vectorized operations
        # Get positions of traj_future_start_token_id for all samples
        traj_start_positions = traj_future_start_mask.int().argmax(dim=1)  # [B]
        
        # Create indices for next token check
        batch_indices = torch.arange(b_star, device=device)
        next_token_positions = traj_start_positions + 1
        
        # Check if next token is 220 (vectorized)
        valid_next_pos = next_token_positions < input_ids.shape[1]
        is_next_token_220 = torch.zeros(b_star, dtype=torch.bool, device=device)
        is_next_token_220[valid_next_pos] = input_ids[batch_indices[valid_next_pos], next_token_positions[valid_next_pos]] == 220
        
        # Calculate truncation positions
        truncate_positions = torch.where(
            has_traj_future_start,
            torch.where(is_next_token_220, traj_start_positions + 1, traj_start_positions),
            torch.full((b_star,), input_ids.shape[1] - 1, device=device)
        ) + 1  # +1 to include the position
        
        max_truncate_pos = truncate_positions.max().item()
        
        # Create truncated tensors with left padding
        batch_size = input_ids.shape[0]
        truncated_input_ids = torch.full(
            (batch_size, max_truncate_pos),
            self.config.pad_token_id,
            dtype=input_ids.dtype,
            device=device
        )
        truncated_attention_mask = torch.zeros(
            (batch_size, max_truncate_pos),
            dtype=kwargs.get('attention_mask').dtype if 'attention_mask' in kwargs else torch.long,
            device=device
        )
        
        # Vectorized left padding using advanced indexing
        for i in range(batch_size):
            trunc_len = truncate_positions[i].item()
            pad_len = max_truncate_pos - trunc_len
            truncated_input_ids[i, pad_len:] = input_ids[i, :trunc_len]
            truncated_attention_mask[i, pad_len:] = 1
        
        # Handle labels if present
        truncated_labels = None
        if 'labels' in kwargs and kwargs['labels'] is not None:
            labels = kwargs['labels']
            truncated_labels = torch.full(
                (batch_size, max_truncate_pos),
                -100,  # Standard ignore index for loss computation
                dtype=labels.dtype,
                device=device
            )
            for i in range(batch_size):
                trunc_len = truncate_positions[i].item()
                pad_len = max_truncate_pos - trunc_len
                truncated_labels[i, pad_len:] = labels[i, :trunc_len]
        
        # Update kwargs with truncated and padded data
        truncated_kwargs = kwargs.copy()
        truncated_kwargs['input_ids'] = truncated_input_ids
        truncated_kwargs['attention_mask'] = truncated_attention_mask
        if truncated_labels is not None:
            truncated_kwargs['labels'] = truncated_labels
        # print(">>>>>> vlm_outputs:",truncated_kwargs)
        # Get VLM forward outputs for prompt encoding 
        vlm_outputs = self.vlm(use_cache = True,**truncated_kwargs)
        rope_deltas = vlm_outputs.rope_deltas.clone()
        # overwatch.info(f"rope_deltas:{rope_deltas}")
        
        prompt_cache = vlm_outputs.past_key_values
        # print(">>>>>> pc:",prompt_cache)
        # 对prompt cache 做门控映射（如果启用），  keep original prompt_cache object intact
        # Convert Cache -> legacy tuple, map, then wrap back to same Cache type so expert can accept it
        # print("vlm_outputs:",vlm_outputs)
        # print("prompt_cache:",prompt_cache)
        legacy_cache = prompt_cache.to_legacy_cache() if hasattr(prompt_cache, "to_legacy_cache") else tuple(prompt_cache)
        if getattr(self, "cache_mapper", None) is not None:
            mapped_tuple = self.cache_mapper(legacy_cache)
        else:
            mapped_tuple = legacy_cache
        try:
            mapped_prompt_past = type(prompt_cache).from_legacy_cache(mapped_tuple)
        except Exception:
            mapped_prompt_past = DynamicCache.from_legacy_cache(mapped_tuple)
        
        # Prepare position IDs for diffusion tokens
        n_diffusion_tokens = self.action_space.get_action_space_dims()[0]  # 60个token
        position_ids = torch.arange(n_diffusion_tokens, device=device)  # [0...59]
        position_ids = einops.repeat(position_ids, "l -> b l", b=b_star).clone()  # B, n_diffusion_tokens
        delta = rope_deltas + truncate_positions[:, None]  # 加上偏移
        position_ids += delta.to(position_ids.device)  # B, n_diffusion_tokens
        
        # Prepare attention mask for expert model
        # attention_mask = torch.zeros(
        #     (b_star, prompt_cache.get_seq_length() + n_diffusion_tokens),
        #     dtype=torch.float32,
        #     device=device
        # )
        # for i in range(b_star):
        #     attention_mask[i, truncate_positions[i]:] = torch.finfo(attention_mask.dtype).min

        ######################################## FM ####################################################################
        # Flow matching training: sample noisy actions and predict denoising direction
        # 从一个norm -> 另一个norm 
        x1=self.action_space.traj_to_action(
            traj_history_xyzyaw=history_xyzyaw, 
            traj_future_xyzyaw=future_xyzyaw, 
            t0_states=t0_state # TODO: 对 dxy 空间无效， 
        )
        xt, dx_t, t,x0 = self.flow_match.sample(x1)   
        
        # print("xxxxx,",xt.shape, t.shape)
        # Project noisy actions to token embeddings
        future_token_embeds = self.action_in_proj(xt, t)
        # print("future_token_embeds:",future_token_embeds.shape)
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(b_star, n_diffusion_tokens, -1)
        
        # Use precomputed mapped_prompt_past (Cache) and run expert with cached prefill
        expert_out_base = self.expert(
            inputs_embeds=future_token_embeds,
            position_ids=position_ids,
            past_key_values=mapped_prompt_past,
            # attention_mask=attention_mask,
            use_cache=True,
            is_causal=not self.config.expert_non_causal_attention
        )

        # Extract predictions from expert output
        last_hidden = expert_out_base.last_hidden_state[:, -n_diffusion_tokens:]
        pred_v = self.action_out_proj(last_hidden).view(
            -1, *self.x_dims
        )
        # Compute flow matching loss
        loss = F.mse_loss(pred_v, dx_t)  #
        # a_loss, x_loss = total_loss_dynamic_scaled(pred_v,dx_t,method = "mean_split") 
        
        # 加权 mes_loss
        weighted_mse_loss = weighted_mse(pred_v,dx_t,method = "mean")
        # if not self.config.freeze_vlm_in_fm:
            # loss += vlm_outputs['loss']
        # print("a loss:",a_loss.item(),"x loss:",x_loss.item())
        return AR1MOutput(
            loss= weighted_mse_loss,
            # loss = loss,
            # loss = 1 * a_loss + 0.6 * x_loss     # 一阶损失由向量场保证，
            # loss = loss  # 加权权重，会要求前面点误差尽可能小，后面点适当放宽，但实际上并不能对应积分后的轨迹。
        )
    
    def forward(self, **kwargs):
        if self.config.mode == "FM":
            return self.forward_fm(**kwargs)
        else:
            return self.forward_ar(**kwargs)

    def get_output_embeddings(self) -> torch.nn.Module:
        """Get the output embeddings of the model."""
        return self.vlm.get_output_embeddings()

    def get_input_embeddings(self) -> torch.nn.Module:
        """Get the input embeddings of the model."""
        return self.vlm.language_model.embed_tokens

    def tie_weights(self) -> None:
        """Delegate weight tying to the nested VLM model."""
        if hasattr(self.vlm, "tie_weights"):
            self.vlm.tie_weights()

    @classmethod
    def from_pretrained_submodules(
        cls,
        config: AlpamayoR1Config,
    ) -> "AlpamayoR1Model":
        """Load submodules with pretrained submodules and initialize the model."""
        pretrained_modules = {}

        # Load VLM
        # hard coded load svit version qwen3-vl-2B-svit
        # vlm = Qwen3VLForConditionalGeneration.from_pretrained(
        #     config.vlm_name_or_path,
        #     dtype=config.model_dtype,
        #     attn_implementation=config.attn_implementation,
        # )

        vlm = Qwen3VLForConditionalGenerationWithDINOv3.from_pretrained("/mnt/data/models/qwen3VL-DINOv3-Flex-2B-Instruct-200",dtype = config.model_dtype,attn_implementation = config.attn_implementation)


        overwatch.info(f">>> orin token embedding:{vlm.model.language_model.embed_tokens}")

        original_vocab_size = vlm.config.text_config.vocab_size
        # vlm.resize_token_embeddings(config.vocab_size)
        vlm.resize_token_embeddings(
            new_num_tokens=config.vocab_size,
            pad_to_multiple_of=64  # or 32, 64, 128 depending on your model
        )

        vlm.config.text_config.vocab_size = config.vocab_size
        vlm.config.vocab_size = config.vocab_size
        pretrained_modules["vlm"] = vlm


        return cls(
            config,
            pretrained_modules=pretrained_modules,
            original_vocab_size=original_vocab_size,
        )

    # # 生成方法，生成AR轨迹和FM轨迹
    # def generate():
    #     pass 
    # TODO: ar版本不需要额外改动。
    def generate(self,*args,**kwargs):
        if self.config.mode == "AR":
            kwargs.pop("future_xyzyaw")
            kwargs.pop("history_xyzyaw")
            return self.vlm.generate(*args,**kwargs)

        elif self.config.mode == "FM":

            kwargs.pop("future_xyzyaw") # TODO: 临时，

            return  self.sample_trajectories_from_data_with_vlm_rollout(**kwargs)
        else:
            raise ValueError("未实现的模型推理策略")

    def sample_trajectories_from_data_with_vlm_rollout(
        self,
        history_xyzyaw:torch.FloatTensor = None,
        t0_state = None,
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 1,
        num_traj_sets: int = 1,
        diffusion_kwargs: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # generate:
        # B, n_traj_group, _, _ = history_xyzyaw.shape
        B,_,_ = history_xyzyaw.shape
        # 预处理器处理后的token id
        device = kwargs.get('input_ids').device
        # device = input_ids.device 
        
        # 生成配置 
        max_generation_length = kwargs.get(
            "max_generation_length", self.config.tokens_per_future_traj
        )
        
        generation_config = self.vlm.generation_config
        generation_config.top_p = top_p
        generation_config.temperature = temperature
        generation_config.do_sample = True
        generation_config.num_return_sequences = num_traj_samples
        generation_config.max_new_tokens = max_generation_length
        generation_config.output_logits = False
        generation_config.return_dict_in_generate = True
        generation_config.top_k = top_k
        generation_config.pad_token_id = self.config.pad_token_id
        generation_config.use_cache = True  # TODO: 自定义的 这个 变成 False了，原因未知..... 
        
        # TODO：
        vlm_outputs = self.vlm.generate(
            generation_config=generation_config,
            stopping_criteria=self.stopping_criteria,
            logits_processor=self.logits_processor,
            **kwargs
        )
        
        # TODO: 需要理解这玩意到底干吗的？
        vlm_outputs.rope_deltas = self.vlm.model.rope_deltas 
        
        # 
        vlm_outputs.sequences = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=self.traj_future_start_token_id,
            pad_token_id = self.config.pad_token_id
        )
        # print("vlm_output:",vlm_outputs.sequences)
        prompt_cache = vlm_outputs.past_key_values 
        prefill_seq_len = prompt_cache.get_seq_length()
        
        if getattr(self, "cache_mapper", None) is not None:
            mapped_tuple = self.cache_mapper(prompt_cache.to_legacy_cache())
            try:
                prompt_cache = type(prompt_cache).from_legacy_cache(mapped_tuple)
            except Exception:
                prompt_cache = DynamicCache.from_legacy_cache(mapped_tuple)
        # print("vlm_outputs.sequences:",vlm_outputs.sequences, print(vlm_outputs.sequences.shape))
        # 
        b_star = vlm_outputs.sequences.shape[0]
        traj_future_start_mask = vlm_outputs.sequences == self.traj_future_start_token_id
        
        # 
        has_traj_future_start = traj_future_start_mask.any(dim = 1)
        
        for i in range(b_star):
            if not has_traj_future_start[i]:
                overwatch.warning(f"Trajectory {i} does not have a future start token.")
                
        
        # 
        traj_future_start_positions = traj_future_start_mask.int().argmax(dim = 1)
        last_token_positions = torch.full(
            (b_star,), vlm_outputs.sequences.shape[1] - 1, device = device
        )
        
        # 
        valid_token_pos_id = torch.where(
            has_traj_future_start, traj_future_start_positions, last_token_positions
        )
        
        offset = valid_token_pos_id + 1 # 开始生成的 第一个 token 位置 
        
        # modify the position ids to remove padding tokens
        n_diffusion_tokens = self.action_space.get_action_space_dims()[0]  # 60个token
        position_ids = torch.arange(n_diffusion_tokens, device=device) # [0...59]
        position_ids = einops.repeat(position_ids, "l -> 3 b l", b=b_star).clone() # 3，B，a
        delta = vlm_outputs.rope_deltas + offset[:, None] # TODO:加上偏移？
        position_ids += delta.to(position_ids.device) # 3 b l 

        

        # modify the attention_masks to remove padding tokens, 表征 action - seq + action 的注意力关系
        # attention_mask = torch.zeros(
        #     (b_star, 1, n_diffusion_tokens, prompt_cache.get_seq_length() + n_diffusion_tokens),
        #     dtype=torch.float32,
        #     device=device,
        # )
        
        # for i in range(b_star):
        #     attention_mask[i,:, :, offset[i] : -n_diffusion_tokens] = torch.finfo(
        #         attention_mask.dtype
        #     ).min


            

        # 2) Define denoising step that consumes noisy action and timestep
        def step_fn(
            x: torch.Tensor,
            t: torch.Tensor,
        ) -> torch.Tensor:
            # x: (B*, *action_dim)
            # t: broadcastable to x leading dims
            b_star = x.shape[0]
            # Project noisy action to expert token embeddings for the n future tokens
            # Expect shape (b*, n_token_per_traj, hidden_size)
            future_token_embeds = self.action_in_proj(x, t) # 做词嵌入，每个量单独编码后拼接，再MLP
            if future_token_embeds.dim() == 2:
                future_token_embeds = future_token_embeds.view(b_star, n_diffusion_tokens, -1)

            expert_out_base = self.expert(
                inputs_embeds=future_token_embeds,
                position_ids=position_ids,
                past_key_values=prompt_cache,
                # attention_mask=attention_mask if not self.config.expert_non_causal_attention else None,
                use_cache=True,
                is_causal=not self.config.expert_non_causal_attention
            )
            # crop the prompt cache to remove the newly added tokens
            prompt_cache.crop(prefill_seq_len)
            last_hidden = expert_out_base.last_hidden_state  # (b*, Tf, hidden_size)
            last_hidden = last_hidden[:, -n_diffusion_tokens:]
            v_pred = self.action_out_proj(last_hidden).view(
                -1, *self.x_dims
            )  # (b*, Tf, C_action) -> noise/vector field
            # print(">>> v v_pred:",t, v_pred.mean(dim = -2))
            return v_pred

        # 3) Diffusion sampling in action space with multiple samples per input
        total_batch = B
        if diffusion_kwargs is None:
            diffusion_kwargs = {}

        sampled_action = self.flow_match.solver(
            batch_size=total_batch,
            step_fn=step_fn,
            device=device,
            return_all_steps=False,
            **diffusion_kwargs,
        )

        pred_xyzyaw = self.action_space.action_to_traj(
            sampled_action, history_xyzyaw,t0_state
        )
        return pred_xyzyaw #N,B,T,4
    
    # def map_past_key_values(self, past_key_values: Union[Cache, tuple]) -> Cache:
    #     """Map a Cache or legacy tuple to an expert Cache (wrapped as Cache instance).

    #     Accepts either a `Cache` instance (e.g. DynamicCache/EncoderDecoderCache) or a legacy
    #     tuple of per-layer (k, v) tensors. Returns a `Cache` instance of the same type as input
    #     when possible, otherwise a `DynamicCache`.
    #     """
    #     # Convert input to legacy tuple
    #     if isinstance(past_key_values, Cache) or hasattr(past_key_values, "to_legacy_cache"):
    #         legacy = past_key_values.to_legacy_cache()
    #         orig_cache = past_key_values
    #     else:
    #         legacy = past_key_values
    #         orig_cache = None

    #     # Map
    #     if getattr(self, "cache_mapper", None) is None:
    #         mapped_tuple = legacy
    #     else:
    #         mapped_tuple = self.cache_mapper(legacy)

    #     # Wrap back
    #     if orig_cache is not None:
    #         try:
    #             return type(orig_cache).from_legacy_cache(mapped_tuple)
    #         except Exception:
    #             return DynamicCache.from_legacy_cache(mapped_tuple)
    #     else:
    #         return DynamicCache.from_legacy_cache(mapped_tuple)