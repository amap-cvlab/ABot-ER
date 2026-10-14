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
class FeedbackDatasetProcessor(DatasetProcessor):
    def _encode_feedback_example(
        self,
        prompt: Sequence[Dict[str, str]],
        response: Sequence[Dict[str, str]],
        kl_response: Sequence[Dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
    ) -> Tuple[List[int], List[int], List[int], List[int], bool]:
        if response[0]["content"]:  # desired example
            kto_tag = True
            messages = prompt + [response[0]]
        else:  # undesired example
            kto_tag = False
            messages = prompt + [response[1]]

        if kl_response[0]["content"]:
            kl_messages = prompt + [kl_response[0]]
        else:
            kl_messages = prompt + [kl_response[1]]

        prompt_ids, response_ids = self.template.encode_oneturn(self.tokenizer, messages, system, tools)
        kl_prompt_ids, kl_response_ids = self.template.encode_oneturn(self.tokenizer, kl_messages, system, tools)

        if self.template.efficient_eos:
            response_ids += [self.tokenizer.eos_token_id]
            kl_response_ids += [self.tokenizer.eos_token_id]

        source_len, target_len = infer_seqlen(len(prompt_ids), len(response_ids), self.data_args.cutoff_len)
        prompt_ids = prompt_ids[:source_len]
        response_ids = response_ids[:target_len]
        kl_source_len, kl_target_len = infer_seqlen(
            len(kl_prompt_ids), len(kl_response_ids), self.data_args.cutoff_len
        )
        kl_prompt_ids = kl_prompt_ids[:kl_source_len]
        kl_response_ids = kl_response_ids[:kl_target_len]

        input_ids = prompt_ids + response_ids
        labels = [IGNORE_INDEX] * source_len + response_ids
        kl_input_ids = kl_prompt_ids + kl_response_ids
        kl_labels = [IGNORE_INDEX] * kl_source_len + kl_response_ids

        return input_ids, labels, kl_input_ids, kl_labels, kto_tag

    def preprocess_dataset(self, examples: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        # create unrelated input-output pairs for estimating the KL term by flipping the matched pairs
        kl_response = examples["response"][::-1]
        model_inputs = defaultdict(list)
        for i in range(len(examples["prompt"])):
            messages = examples["prompt"][i] + examples["response"][i]
            if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) < 2:
                logger.warning(f"Dropped invalid example: {messages} due to invalid format.")
                continue

            if examples["video"][i] and examples["video_frames"][i]:
                logger.warning(f"Dropped invalid example: {messages} due to both video and video_frames are provided.")
                continue

            input_ids, labels, kl_input_ids, kl_labels, kto_tag = self._encode_feedback_example(
                prompt=examples["prompt"][i],
                response=examples["response"][i],
                kl_response=kl_response[i],
                system=examples["system"][i],
                tools=examples["tools"][i],
            )

            if not self.template.plugin.validate_input(input_ids, examples, i):
                logger.warning(f"Dropped invalid example: {messages} due to invalid multimodal token numbers.")
                continue

            if not self.template.plugin.validate_input(kl_input_ids, examples, i):
                logger.warning(f"Dropped invalid example: {messages} due to invalid multimodal token numbers.")
                continue

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["kl_input_ids"].append(kl_input_ids)
            model_inputs["kl_attention_mask"].append([1] * len(kl_input_ids))
            model_inputs["kl_labels"].append(kl_labels)
            model_inputs["kto_tags"].append(kto_tag)
            model_inputs["image"].append(examples["image"][i])
            model_inputs["video"].append(examples["video"][i])
            model_inputs["video_frames"].append(examples["video_frames"][i])
            model_inputs["audio"].append(examples["audio"][i])

        desirable_num = sum([1 for tag in model_inputs["kto_tags"] if tag])
        undesirable_num = len(model_inputs["kto_tags"]) - desirable_num
        if desirable_num == 0 or undesirable_num == 0:
            logger.warning("Your dataset only has one preference type.")

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")
