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
"""Video processor for OpenPangu-VL."""

from typing import Optional, Union

import torch
from torchvision.transforms.v2 import functional as F

from ...image_processing_utils import BatchFeature
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, ChannelDimension, PILImageResampling, SizeDict, get_image_size
from ...processing_utils import Unpack, VideosKwargs
from ...utils import TensorType
from ...utils.import_utils import requires
from ...video_processing_utils import BaseVideoProcessor
from ...video_utils import group_videos_by_shape, reorder_videos
from ..qwen2_vl.image_processing_qwen2_vl import smart_resize


class OpenPanguVLVideoProcessorKwargs(VideosKwargs):
    min_pixels: Optional[int]
    max_pixels: Optional[int]
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]
    factor_size: Optional[int]
    dynamic_video_pixels: Optional[bool]
    min_video_total_pixels: Optional[int]
    max_video_total_pixels: Optional[int]
    min_frame_pixels: Optional[int]
    max_frame_pixels: Optional[int]
    random_frame_pixels: Optional[bool]


@requires(backends=("torchvision",))
class OpenPanguVLVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    size = {"height": 448, "width": 448}
    do_resize = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    do_convert_rgb = True
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    factor_size = 56
    min_pixels = 100352
    max_pixels = 200704
    dynamic_video_pixels = True
    min_video_total_pixels = 6422528
    max_video_total_pixels = 6422528
    min_frame_pixels = 100352
    max_frame_pixels = 200704
    random_frame_pixels = False
    valid_kwargs = OpenPanguVLVideoProcessorKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]
    dtype = torch.bfloat16

    def __init__(self, **kwargs: Unpack[OpenPanguVLVideoProcessorKwargs]):
        super().__init__(**kwargs)

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_resize: bool,
        size: SizeDict,
        interpolation,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean,
        image_std,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        num_frames = sum(video.shape[0] for video in videos)
        if not self.dynamic_video_pixels:
            min_pixels = self.min_frame_pixels
            max_pixels = self.max_frame_pixels
        else:
            min_pixels = max(
                min(self.min_video_total_pixels // max(num_frames, 1), self.max_frame_pixels), self.min_frame_pixels
            )
            max_pixels = max(
                min(self.max_video_total_pixels // max(num_frames, 1), self.max_frame_pixels), self.min_frame_pixels
            )

        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            height, width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)
            resized_height, resized_width = height, width
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=self.factor_size,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                stacked_videos = F.resize(stacked_videos, size=(resized_height, resized_width), interpolation=interpolation)
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_video_grid_thw = {}
        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)

            if do_rescale:
                stacked_videos = stacked_videos * rescale_factor
            if do_normalize:
                stacked_videos = F.normalize(stacked_videos.to(dtype=torch.float32), image_mean, image_std)

            batch_size, grid_t_full, channel = stacked_videos.shape[:3]
            if grid_t_full % temporal_patch_size != 0:
                repeats = stacked_videos[:, -1:].repeat(1, temporal_patch_size - grid_t_full % temporal_patch_size, 1, 1, 1)
                stacked_videos = torch.cat([stacked_videos, repeats], dim=1)

            grid_t = stacked_videos.shape[1] // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            stacked_videos = stacked_videos.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            stacked_videos = stacked_videos.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            processed_videos_grouped[shape] = stacked_videos.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )
            processed_video_grid_thw[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_video_grid_thw = reorder_videos(processed_video_grid_thw, grouped_videos_index)
        pixel_values_videos = torch.cat(processed_videos, dim=0).to(self.dtype)
        video_grid_thw = torch.tensor(processed_video_grid_thw)
        return BatchFeature(
            data={"pixel_values_videos": pixel_values_videos, "video_grid_thw": video_grid_thw},
            tensor_type=return_tensors,
        )


__all__ = ["OpenPanguVLVideoProcessor"]
