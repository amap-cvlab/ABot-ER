# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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
from itertools import chain
from typing import Any, Dict, List

from .processor_utils import DatasetProcessor


@dataclass
class PretrainDatasetProcessor(DatasetProcessor):
    def preprocess_dataset(self, examples: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        # build grouped texts with format `X1 X2 X3 ...` if packing is enabled
        eos_token = "<|end_of_text|>" if self.data_args.template == "llama3" else self.tokenizer.eos_token
        text_examples = [messages[0]["content"] + eos_token for messages in examples["prompt"]]

        if not self.data_args.packing:
            if self.data_args.template == "gemma":
                text_examples = [self.tokenizer.bos_token + example for example in text_examples]

            result = self.tokenizer(
                text_examples, add_special_tokens=False, max_length=self.data_args.cutoff_len, truncation=True
            )

            input_ids = result["input_ids"]
            valid_mask = [False] * len(input_ids)
            for i in range(len(input_ids)):
                if examples["video"][i] and examples["video_frames"][i]:
                    continue
                valid_mask[i] = self.template.plugin.validate_input(input_ids[i], examples, i)

            valid_images = [image for i, image in enumerate(examples["image"]) if valid_mask[i]]
            valid_videos = [video for i, video in enumerate(examples["video"]) if valid_mask[i]]
            valid_video_frames = [video_frames for i, video_frames in enumerate(examples["video_frames"]) if valid_mask[i]]
            valid_audios = [audio for i, audio in enumerate(examples["audio"]) if valid_mask[i]]
            valid_multimodals = {
                "image": valid_images,
                "video": valid_videos,
                "video_frames": valid_video_frames,
                "audio": valid_audios,
            }

            for k, v in result.items():
                result[k] = [v[i] for i in range(len(v)) if valid_mask[i]]

            result.update(valid_multimodals)
        else:
            tokenized_examples = self.tokenizer(text_examples, add_special_tokens=False)
            concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
            total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
            block_size = self.data_args.cutoff_len
            total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            if self.data_args.template == "gemma":
                for i in range(len(result["input_ids"])):
                    result["input_ids"][i][0] = self.tokenizer.bos_token_id

        return result

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
