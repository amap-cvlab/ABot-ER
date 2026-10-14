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

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from .processor_utils import DatasetProcessor, infer_seqlen


logger = get_logger(__name__)


@dataclass
class PairwiseDatasetProcessor(DatasetProcessor):
    def _encode_pairwise_example(
        self,
        prompt: Sequence[Dict[str, str]],
        response: Sequence[Dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        chosen_messages = prompt + [response[0]]
        rejected_messages = prompt + [response[1]]
        prompt_ids, chosen_ids = self.template.encode_oneturn(self.tokenizer, chosen_messages, system, tools)
        _, rejected_ids = self.template.encode_oneturn(self.tokenizer, rejected_messages, system, tools)

        if self.template.efficient_eos:
            chosen_ids += [self.tokenizer.eos_token_id]
            rejected_ids += [self.tokenizer.eos_token_id]

        source_len, target_len = infer_seqlen(
            len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), self.data_args.cutoff_len
        )  # consider the response is more important
        prompt_ids = prompt_ids[:source_len]
        chosen_ids = chosen_ids[:target_len]
        rejected_ids = rejected_ids[:target_len]

        chosen_input_ids = prompt_ids + chosen_ids
        chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids
        rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids

        return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels

    def preprocess_dataset(self, examples: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
        model_inputs = defaultdict(list)

        for i in range(len(examples["prompt"])):
            messages = examples["prompt"][i] + examples["response"][i]
            if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) < 2:
                logger.warning(f"Dropped invalid example: {messages} due to invalid format.")
                continue

            if examples["video"][i] and examples["video_frames"][i]:
                logger.warning(f"Dropped invalid example: {messages} due to both video and video_frames are provided.")
                continue

            chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = self._encode_pairwise_example(
                prompt=examples["prompt"][i],
                response=examples["response"][i],
                system=examples["system"][i],
                tools=examples["tools"][i],
            )

            if not self.template.plugin.validate_input(chosen_input_ids, examples, i):
                logger.warning(f"Dropped invalid example: {messages} due to invalid multimodal token numbers.")
                continue

            if not self.template.plugin.validate_input(rejected_input_ids, examples, i):
                logger.warning(f"Dropped invalid example: {messages} due to invalid multimodal token numbers.")
                continue

            model_inputs["chosen_input_ids"].append(chosen_input_ids)
            model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
            model_inputs["chosen_labels"].append(chosen_labels)
            model_inputs["rejected_input_ids"].append(rejected_input_ids)
            model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
            model_inputs["rejected_labels"].append(rejected_labels)
            model_inputs["image"].append(examples["image"][i])
            model_inputs["video"].append(examples["video"][i])
            model_inputs["video_frames"].append(examples["video_frames"][i])
            model_inputs["audio"].append(examples["audio"][i])

        return model_inputs

    def print_data_example(self, example: Dict[str, List[int]]) -> None:
        valid_chosen_labels = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_labels"]))
        valid_rejected_labels = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_labels"]))
        print("chosen_input_ids:\n{}".format(example["chosen_input_ids"]))
        print("chosen_inputs:\n{}".format(self.tokenizer.decode(example["chosen_input_ids"], skip_special_tokens=False)))
        print("chosen_label_ids:\n{}".format(example["chosen_labels"]))
        print("chosen_labels:\n{}".format(self.tokenizer.decode(valid_chosen_labels, skip_special_tokens=False)))
        print("rejected_input_ids:\n{}".format(example["rejected_input_ids"]))
        print(
            "rejected_inputs:\n{}".format(self.tokenizer.decode(example["rejected_input_ids"], skip_special_tokens=False))
        )
        print("rejected_label_ids:\n{}".format(example["rejected_labels"]))
        print("rejected_labels:\n{}".format(self.tokenizer.decode(valid_rejected_labels, skip_special_tokens=False)))
