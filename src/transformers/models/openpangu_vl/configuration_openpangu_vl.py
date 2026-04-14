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

from ...configuration_utils import PretrainedConfig
from ..qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLTextConfig


class OpenPanguVLVisionConfig(PretrainedConfig):
    model_type = "openpangu_vl"
    base_config_key = "vision_config"

    def __init__(
        self,
        # aligned user-facing names
        num_layers=26,
        attention_hidden_size=1280,
        activation="gelu",
        mlp_hidden_size=3840,
        attention_head_num=16,
        num_key_value_heads=None,
        in_channels=3,
        patch_size=14,
        token_down_size=2,
        temporal_patch_size=2,
        window_attention_size=112,
        last_hidden_size=3584,
        num_position_embeddings=2304,
        deepstack_visual_indexes=[8, 16, 24],
        fullattn_block_index=[5, 12, 19, 25],
        rope_sections=[4, 6, 6],
        use_gatedmerger=True,
        mhc_visual_expand_linear=True,
        mhc_num_stream=1,
        mhc_visual_expand_liner=True,
        projector=None,
        norm_type="RMSNorm",
        use_fused_rms_norm=False,
        epsilon=1e-6,
        type="hiev3",
        min_pixels=50176,
        max_pixels=1806336,
        min_video_total_pixels=6422528,
        max_video_total_pixels=6422528,
        # HF/internal aliases
        depth=None,
        hidden_size=None,
        hidden_act=None,
        intermediate_size=None,
        num_heads=None,
        spatial_merge_size=None,
        window_size=None,
        out_hidden_size=None,
        fullatt_block_indexes=None,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Resolve aliases with priority to explicit HF/internal kwargs if provided.
        depth = num_layers if depth is None else depth
        hidden_size = attention_hidden_size if hidden_size is None else hidden_size
        hidden_act = activation if hidden_act is None else hidden_act
        intermediate_size = mlp_hidden_size if intermediate_size is None else intermediate_size
        num_heads = attention_head_num if num_heads is None else num_heads
        if num_key_value_heads is None:
            num_key_value_heads = max(1, num_heads // 4)
        spatial_merge_size = token_down_size if spatial_merge_size is None else spatial_merge_size
        window_size = window_attention_size if window_size is None else window_size
        out_hidden_size = last_hidden_size if out_hidden_size is None else out_hidden_size
        fullatt_block_indexes = fullattn_block_index if fullatt_block_indexes is None else fullatt_block_indexes

        if projector is None:
            projector = {"bias": True}

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.num_key_value_heads = num_key_value_heads
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.window_size = window_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.deepstack_visual_indexes = deepstack_visual_indexes
        self.fullatt_block_indexes = fullatt_block_indexes
        self.rope_sections = rope_sections
        self.use_gatedmerger = use_gatedmerger
        self.mhc_visual_expand_linear = mhc_visual_expand_linear and mhc_visual_expand_liner
        self.mhc_num_stream = mhc_num_stream
        self.projector = projector
        self.norm_type = norm_type
        self.use_fused_rms_norm = use_fused_rms_norm or norm_type == "FusedRMSNorm"
        self.epsilon = epsilon
        self.type = type
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.min_video_total_pixels = min_video_total_pixels
        self.max_video_total_pixels = max_video_total_pixels

        # Keep the original user-facing field names for easy round-trip/debugging.
        self.num_layers = depth
        self.attention_hidden_size = hidden_size
        self.activation = hidden_act
        self.mlp_hidden_size = intermediate_size
        self.attention_head_num = num_heads
        self.num_key_value_head_num = num_key_value_heads
        self.token_down_size = spatial_merge_size
        self.window_attention_size = window_size
        self.last_hidden_size = out_hidden_size
        self.fullattn_block_index = fullatt_block_indexes
        self.use_gated_merger = use_gatedmerger
        self.mhc_visual_expand_liner = self.mhc_visual_expand_linear
        self.initializer_range = initializer_range


class OpenPanguVLTextConfig(Qwen3VLTextConfig):
    model_type = "openpangu_vl_text"
    base_config_key = "text_config"


class OpenPanguVLConfig(Qwen3VLConfig):
    model_type = "openpangu_vl"
    sub_configs = {"vision_config": OpenPanguVLVisionConfig, "text_config": OpenPanguVLTextConfig}


__all__ = ["OpenPanguVLConfig", "OpenPanguVLTextConfig", "OpenPanguVLVisionConfig"]
