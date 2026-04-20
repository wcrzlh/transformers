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
"""Processor class for OpenPangu-VL."""

from typing import Optional, Union

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...video_utils import VideoInput


class OpenPanguVLVideosProcessorKwargs(VideosKwargs, total=False):
    fps: Union[list[float], float]


class OpenPanguVLImagesKwargs(ImagesKwargs):
    min_pixels: Optional[int]
    max_pixels: Optional[int]
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]


class OpenPanguVLProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: OpenPanguVLImagesKwargs
    videos_kwargs: OpenPanguVLVideosProcessorKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        }
    }


class OpenPanguVLProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer", "video_processor"]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = ("OpenPanguTokenizer", None)

    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        self.image_token = getattr(tokenizer, "image_token", "[unused19]")
        self.video_token = getattr(tokenizer, "video_token", "[unused32]")
        self.vision_start_token = getattr(tokenizer, "vision_start_token", "[unused18]")
        self.vision_end_token = getattr(tokenizer, "vision_end_token", "[unused20]")
        self.image_token_id = (
            tokenizer.image_token_id if getattr(tokenizer, "image_token_id", None) else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id if getattr(tokenizer, "video_token_id", None) else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        self.vision_start_token_id = (
            tokenizer.vision_start_token_id
            if getattr(tokenizer, "vision_start_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_start_token)
        )
        self.vision_end_token_id = (
            tokenizer.vision_end_token_id
            if getattr(tokenizer, "vision_end_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_end_token)
        )
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        videos: Optional[VideoInput] = None,
        **kwargs: Unpack[OpenPanguVLProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            OpenPanguVLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = {}
        video_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        if videos is not None:
            fps = output_kwargs["videos_kwargs"].get("fps", 2.0)
            video_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = video_inputs["video_grid_thw"]
            if isinstance(fps, (int, float)):
                second_per_grid_ts = [self.video_processor.temporal_patch_size / fps] * len(video_grid_thw)
            else:
                second_per_grid_ts = [self.video_processor.temporal_patch_size / value for value in fps]
            video_inputs["second_per_grid_ts"] = second_per_grid_ts

        if not isinstance(text, list):
            text = [text]
        text = text.copy()

        if images is not None:
            self._process_vision_placeholders(
                text=text,
                vision_token=self.image_token,
                grid_thw=image_grid_thw,
                merge_size=self.image_processor.merge_size,
                vision_start_token=self.vision_start_token,
                vision_end_token=self.vision_end_token,
            )

        if videos is not None:
            self._process_vision_placeholders(
                text=text,
                vision_token=self.video_token,
                grid_thw=video_grid_thw,
                merge_size=self.video_processor.merge_size,
                vision_start_token=self.vision_start_token,
                vision_end_token=self.vision_end_token,
            )

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs, **video_inputs}, tensor_type=return_tensors)

    @staticmethod
    def _process_vision_placeholders(
        text,
        vision_token: str,
        grid_thw,
        merge_size: int,
        vision_start_token: str,
        vision_end_token: str,
    ) -> None:
        index = 0
        for i in range(len(text)):
            while vision_token in text[i]:
                grid_t, grid_h, grid_w = (int(value) for value in grid_thw[index])
                seq_length_per_time = (grid_h * grid_w) // (merge_size**2)
                placeholder_string = (
                    vision_start_token + ("<|vision_placeholder|>" * seq_length_per_time) + vision_end_token
                )
                if grid_t > 1:
                    placeholder_string *= grid_t
                placeholder_string = placeholder_string.removeprefix(vision_start_token)
                placeholder_string = placeholder_string.removesuffix(vision_end_token)
                text[i] = text[i].replace(vision_token, placeholder_string, 1)
                index += 1
            text[i] = text[i].replace("<|vision_placeholder|>", vision_token)


__all__ = ["OpenPanguVLProcessor", "OpenPanguVLProcessorKwargs"]
