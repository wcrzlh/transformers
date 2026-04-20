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

import os
import shutil
import tempfile
import unittest

import numpy as np
import sentencepiece as spm

from transformers import AutoProcessor, OpenPanguTokenizer
from transformers.testing_utils import require_sentencepiece, require_torch, require_torchvision
from transformers.utils import is_torch_available


if is_torch_available():
    import torch

if is_torch_available():
    from transformers import OpenPanguVLImageProcessorFast, OpenPanguVLProcessor, OpenPanguVLVideoProcessor


@require_torch
class OpenPanguVLProcessorPlaceholderTest(unittest.TestCase):
    def test_process_vision_placeholders_repeats_per_frame(self):
        text = ["prefix [unused32] suffix"]
        grid_thw = torch.tensor([[2, 4, 6]])

        OpenPanguVLProcessor._process_vision_placeholders(
            text=text,
            vision_token="[unused32]",
            grid_thw=grid_thw,
            merge_size=2,
            vision_start_token="[unused18]",
            vision_end_token="[unused20]",
        )

        self.assertEqual(text[0].count("[unused32]"), 12)


@require_sentencepiece
@require_torch
@require_torchvision
class OpenPanguVLProcessorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        cls._save_processor_assets(cls.tmpdirname)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    @classmethod
    def _save_processor_assets(cls, tmpdirname):
        corpus_file = os.path.join(tmpdirname, "corpus.txt")
        with open(corpus_file, "w", encoding="utf-8") as handle:
            handle.write("hello world\n")
            handle.write("vision language model\n")
            handle.write("[unused18] [unused19] [unused20] [unused32] [unused10]\n")

        model_prefix = os.path.join(tmpdirname, "tokenizer")
        spm.SentencePieceTrainer.train(
            input=corpus_file,
            model_prefix=model_prefix,
            vocab_size=64,
            model_type="bpe",
            bos_id=1,
            eos_id=-1,
            pad_id=-1,
            unk_id=0,
            user_defined_symbols=["[unused10]", "[unused18]", "[unused19]", "[unused20]", "[unused32]"],
        )

        tokenizer = OpenPanguTokenizer(vocab_file=model_prefix + ".model")
        tokenizer._auto_class = None
        image_processor = OpenPanguVLImageProcessorFast()
        video_processor = OpenPanguVLVideoProcessor()
        processor = OpenPanguVLProcessor(
            tokenizer=tokenizer,
            image_processor=image_processor,
            video_processor=video_processor,
        )
        processor._auto_class = None
        processor.save_pretrained(tmpdirname)

    def test_processor_injects_expected_number_of_image_placeholders(self):
        processor = OpenPanguVLProcessor.from_pretrained(self.tmpdirname)
        image = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)

        inputs = processor(text="describe [unused19]", images=image, return_tensors="pt")

        expected_image_tokens = int(torch.prod(inputs["image_grid_thw"][0]).item() // (processor.image_processor.merge_size**2))
        observed_image_tokens = int((inputs["input_ids"] == processor.image_token_id).sum().item())

        self.assertEqual(observed_image_tokens, expected_image_tokens)
        self.assertSetEqual(
            set(inputs.keys()),
            {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"},
        )

    def test_save_load_pretrained_default(self):
        processor = AutoProcessor.from_pretrained(self.tmpdirname)

        self.assertIsInstance(processor, OpenPanguVLProcessor)
        self.assertIsInstance(processor.tokenizer, OpenPanguTokenizer)
        self.assertIsInstance(processor.image_processor, OpenPanguVLImageProcessorFast)
        self.assertIsInstance(processor.video_processor, OpenPanguVLVideoProcessor)
