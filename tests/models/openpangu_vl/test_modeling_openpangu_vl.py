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
"""Testing suite for the PyTorch OpenPangu-VL model."""

import copy
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from transformers import OpenPanguVLConfig, OpenPanguVLForConditionalGeneration, OpenPanguVLModel, is_torch_available
from transformers.testing_utils import require_torch, torch_device
from transformers.models.openpangu_vl.modeling_openpangu_vl import OpenPanguVLVisionModel

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import floats_tensor, ids_tensor


if is_torch_available():
    import torch


class OpenPanguVLVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        num_channels=3,
        ignore_index=-100,
        image_size=28,
        text_config={
            "bos_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 2,
            "hidden_act": "silu",
            "head_dim": 8,
            "hidden_size": 32,
            "vocab_size": 99,
            "intermediate_size": 37,
            "max_position_embeddings": 512,
            "model_type": "openpangu_vl",
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "rope_theta": 10000,
            "tie_word_embeddings": True,
            "rope_scaling": {"rope_type": "default", "mrope_section": [16, 8, 8], "mrope_interleaved": True},
        },
        vision_config={
            "num_layers": 0,
            "activation": "gelu",
            "mlp_hidden_size": 32,
            "last_hidden_size": 32,
            "attention_hidden_size": 32,
            "attention_head_num": 4,
            "num_key_value_heads": 2,
            "patch_size": 14,
            "token_down_size": 1,
            "temporal_patch_size": 2,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [],
            "fullattn_block_index": [],
            "window_attention_size": 14,
        },
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
        tie_word_embeddings=True,
        is_training=True,
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.is_training = is_training
        self.vision_config = vision_config
        self.text_config = text_config
        self.vocab_size = text_config["vocab_size"]
        self.pad_token_id = text_config["pad_token_id"]
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.num_image_tokens = 32
        self.seq_length = seq_length + self.num_image_tokens
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.tie_word_embeddings = tie_word_embeddings

    def get_config(self):
        return OpenPanguVLConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=self.vision_end_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        patch_size = config.vision_config.patch_size
        temporal_patch_size = config.vision_config.temporal_patch_size
        pixel_values = floats_tensor(
            [
                self.batch_size * (self.image_size**2) // (patch_size**2),
                self.num_channels * (patch_size**2) * temporal_patch_size,
            ]
        )
        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        input_ids[:, -1] = self.pad_token_id
        input_ids[input_ids == self.video_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[input_ids == self.vision_start_token_id] = self.pad_token_id
        input_ids[:, self.num_image_tokens] = self.image_token_id
        input_ids[:, self.num_image_tokens - 1] = self.vision_start_token_id
        inputs_dict = {
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor([[1, 1, 1]] * self.batch_size, device=torch_device),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class OpenPanguVLModelTest(unittest.TestCase):
    all_model_classes = (
        (OpenPanguVLModel, OpenPanguVLForConditionalGeneration) if is_torch_available() else ()
    )
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = OpenPanguVLVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=OpenPanguVLConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_mismatching_num_image_tokens(self):
        config = self.model_tester.get_config()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            mask_model = model.model if hasattr(model, "model") else model
            input_ids = torch.tensor([[1, self.model_tester.vision_start_token_id, self.model_tester.image_token_id, 2]])
            inputs_embeds = model.get_input_embeddings()(input_ids.to(torch_device))
            image_features = torch.randn(2, config.text_config.hidden_size, device=torch_device)

            with self.assertRaises(ValueError):
                _ = mask_model.get_placeholder_mask(
                    input_ids.to(torch_device),
                    inputs_embeds=inputs_embeds,
                    image_features=image_features,
                )

            matching_image_features = torch.randn(1, config.text_config.hidden_size, device=torch_device)
            image_mask, _ = mask_model.get_placeholder_mask(
                input_ids.to(torch_device),
                inputs_embeds=inputs_embeds,
                image_features=matching_image_features,
            )
            self.assertEqual(image_mask[..., 0].sum().item(), 1)

    def _get_regular_multimodal_inputs(self):
        config = copy.deepcopy(self.model_tester.get_config())
        patch_size = config.vision_config.patch_size
        temporal_patch_size = config.vision_config.temporal_patch_size
        pixel_values = floats_tensor([4, 3 * (patch_size**2) * temporal_patch_size]).to(torch_device, dtype=torch.float32)
        image_grid_thw = torch.tensor([[1, 2, 2]], device=torch_device)
        input_ids = torch.tensor(
            [
                [
                    config.text_config.bos_token_id,
                    17,
                    21,
                    self.model_tester.vision_start_token_id,
                    self.model_tester.image_token_id,
                    self.model_tester.image_token_id,
                    self.model_tester.image_token_id,
                    self.model_tester.image_token_id,
                    self.model_tester.vision_end_token_id,
                    33,
                    config.text_config.eos_token_id,
                    config.text_config.pad_token_id,
                ]
            ],
            device=torch_device,
        )
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]], device=torch_device)
        return config, pixel_values, image_grid_thw, input_ids, attention_mask

    def test_model_forward_with_regular_image_inputs(self):
        config, pixel_values, image_grid_thw, input_ids, attention_mask = self._get_regular_multimodal_inputs()
        model = OpenPanguVLModel(config).to(torch_device).float()
        model.eval()

        image_embeds = torch.randn(4, config.text_config.hidden_size, device=torch_device)
        mock_outputs = SimpleNamespace(
            last_hidden_state=torch.randn(1, input_ids.shape[1], config.text_config.hidden_size, device=torch_device),
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            rope_deltas=None,
        )

        with patch.object(model, "get_image_features", return_value=((image_embeds,), [])) as mocked_get_image_features:
            with patch.object(model.language_model, "forward", return_value=mock_outputs):
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                    )

        mocked_get_image_features.assert_called_once()
        self.assertEqual(outputs.__class__.__name__, "OpenPanguVLModelOutputWithPast")
        self.assertEqual(outputs.last_hidden_state.shape, (1, input_ids.shape[1], config.text_config.hidden_size))

    def test_text_model_forward_with_regular_inputs(self):
        config, _, _, input_ids, attention_mask = self._get_regular_multimodal_inputs()
        model = OpenPanguVLModel(config).to(torch_device).float()
        text_model = model.language_model
        text_model.eval()

        with ExitStack() as stack:
            for layer in text_model.layers:
                stack.enter_context(patch.object(layer, "forward", side_effect=lambda hidden_states, **kwargs: hidden_states))

            with torch.no_grad():
                outputs = text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

        self.assertEqual(outputs.last_hidden_state.shape, (1, input_ids.shape[1], config.text_config.hidden_size))

    def test_vision_model_forward_with_regular_inputs(self):
        config = copy.deepcopy(self.model_tester.get_config())
        config.vision_config.depth = 0
        config.vision_config.num_layers = 0

        vision_model = OpenPanguVLVisionModel._from_config(config.vision_config).to(torch_device).float()
        vision_model.eval()

        patch_size = config.vision_config.patch_size
        temporal_patch_size = config.vision_config.temporal_patch_size
        pixel_values = floats_tensor([4, 3 * (patch_size**2) * temporal_patch_size]).to(torch_device, dtype=torch.float32)
        image_grid_thw = torch.tensor([[1, 2, 2]], device=torch_device)

        with patch.object(vision_model.merger, "forward", side_effect=lambda hidden_states: hidden_states):
            with torch.no_grad():
                outputs = vision_model(
                    hidden_states=pixel_values,
                    grid_thw=image_grid_thw,
                )

        self.assertEqual(outputs.shape, (4, config.vision_config.hidden_size))

    def test_conditional_generation_forward_with_regular_image_inputs(self):
        config, pixel_values, image_grid_thw, input_ids, attention_mask = self._get_regular_multimodal_inputs()
        model = OpenPanguVLForConditionalGeneration(config).to(torch_device).float()
        model.eval()

        mock_outputs = SimpleNamespace(
            loss=None,
            logits=torch.randn(1, input_ids.shape[1], config.text_config.vocab_size, device=torch_device),
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            rope_deltas=None,
        )

        with patch(
            "transformers.models.openpangu_vl.modeling_openpangu_vl.Qwen3VLForConditionalGeneration.forward",
            return_value=mock_outputs,
        ):
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )

        self.assertEqual(outputs.__class__.__name__, "OpenPanguVLCausalLMOutputWithPast")
        self.assertEqual(outputs.logits.shape, (1, input_ids.shape[1], config.text_config.vocab_size))
