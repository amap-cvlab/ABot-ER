# Copyright 2024 the LlamaFactory team.
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

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...extras.logging import get_logger
from ..utils import Role
from .processor_utils import DatasetProcessor, infer_seqlen


logger = get_logger(__name__)


@dataclass
class UnsupervisedDatasetProcessor(DatasetProcessor):
    def _encode_unsupervised_example(
        self,
        prompt: Sequence[Dict[str, str]],
        response: Sequence[Dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
    ) -> Tuple[List[int], List[int]]:
        if len(response) == 1:
            messages = prompt + response
        else:
            messages = prompt + [{"role": Role.ASSISTANT.value, "content": ""}]

        input_ids, labels = self.template.encode_oneturn(self.tokenizer, messages, system, tools)
        if self.template.efficient_eos:
            labels += [self.tokenizer.eos_token_id]

        source_len, target_len = infer_seqlen(len(input_ids), len(labels), self.data_args.cutoff_len)
        input_ids = input_ids[:source_len]
        labels = labels[:target_len]
        return input_ids, labels

    def preprocess_dataset(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        # build inputs with format `<bos> X` and labels with format `Y <eos>`
        model_inputs = defaultdict(list)
        for i in range(len(examples["prompt"])):
            messages = examples["prompt"][i] + examples["response"][i]
            if len(examples["prompt"][i]) % 2 != 1:
                logger.warning(f"Dropped invalid example: {messages} due to invalid format.")
                continue

            if examples["video"][i] and examples["video_frames"][i]:
                logger.warning(f"Dropped invalid example: {messages} due to both video and video_frames are provided.")
                continue

            input_ids, labels = self._encode_unsupervised_example(
                prompt=examples["prompt"][i],
                response=examples["response"][i],
                system=examples["system"][i],
                tools=examples["tools"][i],
            )

            if not self.template.plugin.validate_input(input_ids, examples, i):
                logger.warning(f"Dropped invalid example: {messages} due to invalid multimodal token numbers.")
                continue

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["image"].append(examples["image"][i])
            model_inputs["video"].append(examples["video"][i])
            model_inputs["video_frames"].append(examples["video_frames"][i])
            model_inputs["audio"].append(examples["audio"][i])

        return model_inputs

    def print_data_example(self, example: Dict[str, List[int]]) -> None:
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(self.tokenizer.decode(example["labels"], skip_special_tokens=False)))
