# coding=utf-8
# Copyright 2026 The OpenPangu Team and The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...cache_utils import Cache
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import ModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, is_torch_npu_available, logging
from ..qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLTextAttention,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextModel,
)
from .configuration_openpangu_vl import OpenPanguVLConfig, OpenPanguVLVisionConfig


logger = logging.get_logger(__name__)


if is_torch_npu_available():
    import torch_npu
else:
    torch_npu = None


class OpenPanguVLRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, use_fused_rms_norm: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.use_fused_rms_norm = use_fused_rms_norm
        self._warned_fallback = False

    def _forward_fallback(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_fused_rms_norm:
            if torch_npu is None:
                if not self._warned_fallback:
                    logger.warning_once(
                        "`use_fused_rms_norm=True` but `torch_npu` is unavailable, falling back to standard RMSNorm."
                    )
                    self._warned_fallback = True
                return self._forward_fallback(hidden_states)

            hidden_states, _ = torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)
            return hidden_states

        return self._forward_fallback(hidden_states)


class OpenPanguVLVisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class OpenPanguVLVisionPatchEmbed(nn.Module):
    def __init__(self, config: OpenPanguVLVisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class OpenPanguVLVisionMLP(nn.Module):
    def __init__(self, config: OpenPanguVLVisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
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
    return attn_output.transpose(1, 2).contiguous(), attn_weights


class OpenPanguVLVisionAttention(nn.Module):
    def __init__(self, config: OpenPanguVLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qkv = nn.Linear(
            self.dim, self.num_key_value_heads * (self.num_key_value_groups + 2) * self.head_dim, bias=True
        )
        self.proj = nn.Linear(self.num_heads * self.head_dim, self.dim, bias=True)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        qkv_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, self.num_key_value_heads, self.num_key_value_groups + 2, self.head_dim)
            .permute(2, 0, 1, 3)
        )
        qkv_states = qkv_states.unbind(0)
        query_states = torch.cat(qkv_states[: self.num_key_value_groups], dim=1)
        key_states = qkv_states[self.num_key_value_groups]
        value_states = qkv_states[self.num_key_value_groups + 1]

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if self.config._attn_implementation == "flash_attention_2":
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]
            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
        attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


class OpenPanguVLVisionPatchMerger(nn.Module):
    def __init__(self, config: OpenPanguVLVisionConfig) -> None:
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        norm_in_dim = config.hidden_size
        if config.norm_type in {"RMSNorm", "FusedRMSNorm"}:
            self.norm = OpenPanguVLRMSNorm(
                norm_in_dim, eps=config.epsilon, use_fused_rms_norm=config.use_fused_rms_norm
            )
        else:
            self.norm = nn.LayerNorm(norm_in_dim, eps=config.epsilon)
        self.use_gatedmerger = config.use_gatedmerger
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_gate = nn.Linear(self.hidden_size, self.hidden_size * 2) if self.use_gatedmerger else None
        self.act_fn = nn.GELU()
        self.gate_act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        merge_unit = self.spatial_merge_size**2
        x = self.norm(x)
        if x.ndim == 2:
            seq_len = x.shape[0]
            x = x.view(seq_len // merge_unit, self.hidden_size)
        else:
            batch_size, seq_len, _ = x.shape
            x = x.view(batch_size, seq_len // merge_unit, self.hidden_size)
        x = self.act_fn(self.linear_fc1(x))
        if self.use_gatedmerger:
            gatex = self.linear_gate(x)
            x, gate = torch.chunk(gatex, 2, dim=-1)
            x = x * self.gate_act(gate)
        return x


class OpenPanguVLVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        if config.norm_type in {"RMSNorm", "FusedRMSNorm"}:
            self.norm1 = OpenPanguVLRMSNorm(
                config.hidden_size, eps=config.epsilon, use_fused_rms_norm=config.use_fused_rms_norm
            )
            self.norm2 = OpenPanguVLRMSNorm(
                config.hidden_size, eps=config.epsilon, use_fused_rms_norm=config.use_fused_rms_norm
            )
        else:
            self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.epsilon)
            self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.epsilon)
        self.attn = OpenPanguVLVisionAttention(config=config)
        self.mlp = OpenPanguVLVisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class OpenPanguVLPreTrainedModel(PreTrainedModel):
    config: OpenPanguVLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "OpenPanguVLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3VLTextDecoderLayer,
        "attentions": Qwen3VLTextAttention,
    }


class OpenPanguVLVisionModel(OpenPanguVLPreTrainedModel):
    config: OpenPanguVLVisionConfig
    _no_split_modules = ["OpenPanguVLVisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = OpenPanguVLVisionPatchEmbed(config=config)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = OpenPanguVLVisionRotaryEmbedding(head_dim)

        self.blocks = nn.ModuleList([OpenPanguVLVisionBlock(config) for _ in range(config.depth)])
        self.merger = OpenPanguVLVisionPatchMerger(config=config)

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
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]
        return embeddings.flatten(1)
    def get_window_index(self, grid_thw):
        window_index = []
        cu_window_seqlens = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = grid_h // self.spatial_merge_size, grid_w // self.spatial_merge_size
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w, device=grid_thw.device).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_new = index_padded.reshape(-1)
            index_new = index_new[index_new != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()

        return torch.cat(window_index, dim=0), cu_window_seqlens

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        reverse_indices = torch.argsort(window_index)

        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index].reshape(seq_len, -1)

        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index].reshape(seq_len, -1)
        position_embeddings = (rotary_pos_emb.cos(), rotary_pos_emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(self.blocks):
            cu_seqlens_now = cu_seqlens if layer_num in self.fullatt_block_indexes else cu_window_seqlens
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings, **kwargs)

        hidden_states = self.merger(hidden_states)
        hidden_states = hidden_states[reverse_indices]
        return hidden_states


@dataclass
class OpenPanguVLModelOutputWithPast(ModelOutput):
    r"""
    Base class for OpenPanguVL model outputs, with hidden states and attentions.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


class OpenPanguVLModel(Qwen3VLModel):
    config: OpenPanguVLConfig
    _checkpoint_conversion_mapping = {}
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "OpenPanguVLVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = OpenPanguVLVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3VLTextModel._from_config(config.text_config)
        self.mhc_visual_expand_linear = config.vision_config.mhc_visual_expand_linear
        self.mhc_num_stream = config.vision_config.mhc_num_stream

        visual_in_dim = config.vision_config.out_hidden_size
        expanded_dim = visual_in_dim * self.mhc_num_stream if self.mhc_visual_expand_linear else visual_in_dim
        text_hidden_size = config.text_config.hidden_size
        projector_bias = bool(config.vision_config.projector.get("bias", True))

        # Projection block required by OpenPangu-VL before MHC expansion.
        self.pre_llm_visual_projection = nn.Sequential(
            nn.Linear(visual_in_dim, visual_in_dim, bias=projector_bias),
            nn.GELU(),
        )
        self.pre_llm_visual_expand = (
            nn.Linear(visual_in_dim, expanded_dim, bias=projector_bias) if self.mhc_visual_expand_linear else None
        )
        self.pre_llm_visual_align = (
            nn.Linear(expanded_dim, text_hidden_size, bias=projector_bias) if expanded_dim != text_hidden_size else None
        )

    def _project_visual_before_llm(self, visual_embeds: torch.Tensor) -> torch.Tensor:
        visual_embeds = self.pre_llm_visual_projection(visual_embeds)
        if self.pre_llm_visual_expand is not None:
            visual_embeds = self.pre_llm_visual_expand(visual_embeds)
        if self.pre_llm_visual_align is not None:
            visual_embeds = self.pre_llm_visual_align(visual_embeds)
        return visual_embeds

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        image_embeds = tuple(self._project_visual_before_llm(emb) for emb in image_embeds)
        return image_embeds, []

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    def forward(
        self,
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
        **kwargs,
    ) -> Union[tuple, OpenPanguVLModelOutputWithPast]:
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            **kwargs,
        )
        if isinstance(outputs, tuple):
            return outputs

        return OpenPanguVLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
            rope_deltas=outputs.rope_deltas,
        )


@dataclass
class OpenPanguVLCausalLMOutputWithPast(ModelOutput):
    r"""
    Base class for OpenPanguVL causal language model outputs.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


@auto_docstring
class OpenPanguVLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    config: OpenPanguVLConfig
    _checkpoint_conversion_mapping = {}

    def __init__(self, config):
        super().__init__(config)
        self.model = OpenPanguVLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

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
        **kwargs,
    ) -> Union[tuple, OpenPanguVLCausalLMOutputWithPast]:
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        if isinstance(outputs, tuple):
            return outputs

        return OpenPanguVLCausalLMOutputWithPast(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
            rope_deltas=outputs.rope_deltas,
        )


__all__ = [
    "OpenPanguVLConfig",
    "OpenPanguVLForConditionalGeneration",
    "OpenPanguVLModel",
    "OpenPanguVLVisionConfig",
]
