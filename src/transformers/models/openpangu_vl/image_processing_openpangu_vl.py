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
"""Image processor for OpenPangu-VL."""

from typing import Optional, Union

import torch
from torchvision.transforms.v2 import functional as F

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import BaseImageProcessorFast, DefaultFastImageProcessorKwargs, group_images_by_shape, reorder_images
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring
from ..qwen2_vl.image_processing_qwen2_vl import smart_resize


class OpenPanguVLFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    min_pixels: Optional[int]
    max_pixels: Optional[int]
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]
    factor_size: Optional[int]
    anyres: Optional[bool]


def rescale_and_normalize(
    images: torch.Tensor,
    do_rescale: bool,
    rescale_factor: float,
    do_normalize: bool,
    image_mean: Union[float, list[float]],
    image_std: Union[float, list[float]],
) -> torch.Tensor:
    if do_rescale:
        images = images * rescale_factor
    if do_normalize:
        images = F.normalize(images.to(dtype=torch.float32), image_mean, image_std)
    return images.to(OpenPanguVLImageProcessorFast.dtype)


@auto_docstring
class OpenPanguVLImageProcessorFast(BaseImageProcessorFast):
    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 50176, "longest_edge": 1806336}
    do_rescale = True
    do_normalize = True
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    factor_size = 56
    anyres = True
    min_pixels = 50176
    max_pixels = 1806336
    valid_kwargs = OpenPanguVLFastImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]
    dtype = torch.bfloat16

    def __init__(self, **kwargs: Unpack[OpenPanguVLFastImageProcessorKwargs]):
        size = kwargs.pop("size", None)
        min_pixels = kwargs.pop("min_pixels", None)
        max_pixels = kwargs.pop("max_pixels", None)
        size = self.size if size is None else size
        if min_pixels is not None:
            size["shortest_edge"] = min_pixels
        if max_pixels is not None:
            size["longest_edge"] = max_pixels
        super().__init__(size=size, min_pixels=min_pixels, max_pixels=max_pixels, **kwargs)

    def _further_process_kwargs(
        self,
        size: Optional[SizeDict] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        **kwargs,
    ) -> dict:
        if min_pixels is not None and max_pixels is not None:
            size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        elif size is None:
            size = {**self.size}
        return super()._further_process_kwargs(size=size, min_pixels=min_pixels, max_pixels=max_pixels, **kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[OpenPanguVLFastImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, None, **kwargs)

    def _preprocess(
        self,
        images: list[torch.Tensor],
        do_resize: bool,
        size: SizeDict,
        interpolation,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean,
        image_std,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        factor_size: int = 56,
        **kwargs,
    ):
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            height, width = stacked_images.shape[-2:]
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=factor_size,
                    min_pixels=size["shortest_edge"],
                    max_pixels=size["longest_edge"],
                )
                stacked_images = self.resize(
                    image=stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    interpolation=interpolation,
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        processed_grids = {}
        for shape, stacked_images in grouped_images.items():
            resized_height, resized_width = stacked_images.shape[-2:]
            patches = rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            if patches.ndim == 4:
                patches = patches.unsqueeze(1)
            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(1, temporal_patch_size - patches.shape[1] % temporal_patch_size, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)

            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
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
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )
            processed_images_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_grids = reorder_images(processed_grids, grouped_images_index)

        pixel_values = torch.cat(processed_images, dim=0).to(self.dtype)
        image_grid_thw = torch.tensor(processed_grids)
        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
            tensor_type=return_tensors,
        )


__all__ = ["OpenPanguVLImageProcessorFast", "rescale_and_normalize"]
