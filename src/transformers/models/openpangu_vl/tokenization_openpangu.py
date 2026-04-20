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
"""Tokenization classes for OpenPangu."""

import os
import re
from shutil import copyfile
from typing import Any, Optional

import sentencepiece as spm

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}


def _convert_bool(value):
    if isinstance(value, str):
        lower = value.lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
    return value


class OpenPanguTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    _auto_class = "AutoTokenizer"

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="[unused10]",
        pad_token=None,
        sp_model_kwargs: Optional[dict[str, Any]] = None,
        add_bos_token=False,
        add_eos_token=False,
        decode_with_prefix_space=False,
        clean_up_tokenization_spaces=False,
        padded_size=151552,
        divided_by=16,
        image_token="[unused19]",
        video_token="[unused32]",
        vision_start_token="[unused18]",
        vision_end_token="[unused20]",
        **kwargs,
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)
        self.vocab_file = vocab_file
        self.add_bos_token = _convert_bool(add_bos_token)
        self.add_eos_token = _convert_bool(add_eos_token)
        self.decode_with_prefix_space = decode_with_prefix_space
        self.padded_size = padded_size
        self.divided_by = divided_by
        self.image_token = image_token
        self.video_token = video_token
        self.vision_start_token = vision_start_token
        self.vision_end_token = vision_end_token
        self._no_prefix_space_tokens = None

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

        self.image_token_id = self.convert_tokens_to_ids(self.image_token)
        self.video_token_id = self.convert_tokens_to_ids(self.video_token)
        self.vision_start_token_id = self.convert_tokens_to_ids(self.vision_start_token)
        self.vision_end_token_id = self.convert_tokens_to_ids(self.vision_end_token)

    @property
    def no_prefix_space_tokens(self):
        if self._no_prefix_space_tokens is None:
            vocab = self.convert_ids_to_tokens(list(range(self.vocab_size)))
            self._no_prefix_space_tokens = {i for i, token in enumerate(vocab) if not token.startswith("▁")}
        return self._no_prefix_space_tokens

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size()

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.sp_model.bos_id()

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text, **kwargs):
        if re.fullmatch(r"\(\d{3},\d{3}\)", text):
            return [token for char in text for token in self.sp_model.encode(char, out_type=str)]
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        return self.sp_model.IdToPiece(index)

    def _maybe_add_prefix_space(self, tokens, decoded):
        if tokens and tokens[0] not in self.no_prefix_space_tokens:
            return " " + decoded
        return decoded

    def convert_tokens_to_string(self, tokens):
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            if token in self.all_special_tokens:
                if current_sub_tokens:
                    out_string += self.sp_model.decode(current_sub_tokens)
                    current_sub_tokens = []
                out_string += token
            else:
                current_sub_tokens.append(token)

        if current_sub_tokens:
            out_string += self.sp_model.decode(current_sub_tokens)
        if self.clean_up_tokenization_spaces:
            out_string = self.clean_up_tokenization(out_string)
        out_string = self._maybe_add_prefix_space(tokens=tokens, decoded=out_string)
        return out_string[1:] if out_string.startswith(" ") else out_string

    def decode(self, token_ids, spaces_between_special_tokens=False, **kwargs):
        return super().decode(
            token_ids=token_ids,
            spaces_between_special_tokens=spaces_between_special_tokens,
            **kwargs,
        )

    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None):
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return ("",)

        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as file:
                file.write(self.sp_model.serialized_model_proto())

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_ids = [self.bos_token_id] if self.add_bos_token else []
        output = bos_token_ids + token_ids_0
        if token_ids_1 is not None:
            output = output + token_ids_1
        if self.add_eos_token:
            output = output + [self.eos_token_id]
        return output


__all__ = ["OpenPanguTokenizer"]
