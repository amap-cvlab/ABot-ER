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
from .processor_utils import DatasetProcessor, greedy_knapsack, infer_seqlen


logger = get_logger(__name__)


@dataclass
class SupervisedDatasetProcessor(DatasetProcessor):
    def _encode_supervised_example(
        self,
        prompt: Sequence[Dict[str, str]],
        response: Sequence[Dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        loss_masks: Optional[Sequence[Sequence[int]]] = None,
    ) -> Tuple[List[int], List[int]]:
        messages = prompt + response
        input_ids, labels = [], []

        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
        total_length = 1 if self.template.efficient_eos else 0

        # per-turn mask wins over `mask_history`; empty/None falls back to legacy logic.
        # each loss_masks entry is [u_or_-1, a_mask]: u=-1 → fall back to train_on_prompt;
        # u=1 → train user; u=0 → mask user; a controls assistant.
        use_per_turn = bool(loss_masks)
        if use_per_turn and len(loss_masks) != len(encoded_pairs):
            logger.warning_once(
                f"`loss_masks` length {len(loss_masks)} != encoded turns {len(encoded_pairs)}; "
                "falling back to default mask for this sample."
            )
            use_per_turn = False
        if use_per_turn and self.data_args.mask_history:
            logger.warning_once(
                "`history_masks` (per-turn) overrides `mask_history=True`; ignoring `mask_history`."
            )

        reverse_for_mask_history = self.data_args.mask_history and not use_per_turn
        if reverse_for_mask_history:
            encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= self.data_args.cutoff_len:
                break

            source_len, target_len = infer_seqlen(
                len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
            )
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            if use_per_turn:
                u_override, a_train = int(loss_masks[turn_idx][0]), bool(loss_masks[turn_idx][1])
            else:
                u_override, a_train = -1, True
                if reverse_for_mask_history and turn_idx != 0:  # train on the last turn only
                    a_train = False

            if u_override == 1:
                source_label = source_ids
            elif u_override == 0:
                source_label = self.template.source_mask(self.tokenizer, source_ids, turn_idx)
            elif self.data_args.train_on_prompt:  # u_override == -1 → fall back
                source_label = source_ids
            else:
                source_label = self.template.source_mask(self.tokenizer, source_ids, turn_idx)

            target_label = target_ids if a_train else [IGNORE_INDEX] * target_len

            if reverse_for_mask_history:
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label

        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]

        return input_ids, labels

    def preprocess_dataset(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = defaultdict(list)
        for i in range(len(examples["prompt"])):
            messages = examples["prompt"][i] + examples["response"][i]
            if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
                logger.warning(f"Dropped invalid example: {messages} due to invalid format.")
                continue

            if examples["video"][i] and examples["video_frames"][i]:
                logger.warning(f"Dropped invalid example: {messages} due to both video and video_frames are provided.")
                continue

            input_ids, labels = self._encode_supervised_example(
                prompt=examples["prompt"][i],
                response=examples["response"][i],
                system=examples["system"][i],
                tools=examples["tools"][i],
                loss_masks=examples["loss_masks"][i] if "loss_masks" in examples else None,
            )

            if not self.template.plugin.validate_input(input_ids, examples, i):
                logger.warning(f"Dropped invalid example: {messages} due to invalid multimodal token numbers.")
                continue

            if all(label == IGNORE_INDEX for label in labels):
                logger.warning(f"Dropped invalid example: {messages} due to no trainable labels after cutoff.")
                continue

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["image"].append(examples["image"][i])
            model_inputs["video"].append(examples["video"][i])
            model_inputs["video_frames"].append(examples["video_frames"][i])
            model_inputs["audio"].append(examples["audio"][i])
            model_inputs["sample_weight"].append(float(examples.get("sample_weight", [1.0] * len(examples["prompt"]))[i]))

        return model_inputs

    def print_data_example(self, example: Dict[str, List[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(self.tokenizer.decode(valid_labels, skip_special_tokens=False)))


@dataclass
class PackedSupervisedDatasetProcessor(SupervisedDatasetProcessor):
    def _get_sample_length(
        self,
        examples: Dict[str, List[Any]],
        input_ids: List[int],
        labels: List[int],
        sample_idx: int,
    ) -> int:
        # for non-multimodal SFT, nothing to do.
        if self.template.processor is None:
            return len(input_ids)

        assert self.data_args.tokenized_path is not None, "MLLM SFT packing only support in data preprocess phase, please packing MLLM data when tokenized_path is not None."
        # get the seq_len after expansion of multi-modal sample placeholder token.

        mm_input = {
            "input_ids": input_ids.copy(),
            "attention_mask": [1] * len(input_ids),
            "labels": labels.copy(),
        }
        images = examples["image"][sample_idx]
        audios = examples["audio"][sample_idx]
        videos = examples["video"][sample_idx]
        video_frames = examples["video_frames"][sample_idx]
        assert not (videos and video_frames), "Only one of video and video_frames can be provided in one sample."
        videos = videos or video_frames

        processed_token_ids, _ = self.template.plugin.process_single_mm_input(**mm_input, images=images, videos=videos, audios=audios)

        return len(processed_token_ids['input_ids'])

    def preprocess_dataset(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        valid_num = 0
        batch_input_ids, batch_labels = [], []
        batch_medias = {k: [] for k in ["image", "video", "audio", "video_frames"] if k in examples}
        lengths = []
        length2indexes = defaultdict(list)
        for i in range(len(examples["prompt"])):
            messages = examples["prompt"][i] + examples["response"][i]
            if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
                logger.warning(f"Dropped invalid example: {messages} due to invalid format.")
                continue

            if examples["video"][i] and examples["video_frames"][i]:
                logger.warning(f"Dropped invalid example: {messages} due to both video and video_frames are provided.")
                continue

            input_ids, labels = self._encode_supervised_example(
                prompt=examples["prompt"][i],
                response=examples["response"][i],
                system=examples["system"][i],
                tools=examples["tools"][i],
                loss_masks=examples["loss_masks"][i] if "loss_masks" in examples else None,
            )

            if not self.template.plugin.validate_input(input_ids, examples, i):
                logger.warning(f"Dropped invalid example: {messages} due to invalid multimodal token numbers.")
                continue

            if all(label == IGNORE_INDEX for label in labels):
                logger.warning(f"Dropped invalid example: {messages} due to no trainable labels after cutoff.")
                continue

            length = self._get_sample_length(examples, input_ids, labels, i)
            if length > self.data_args.cutoff_len:
                logger.warning(f"Dropped lengthy example with length {length} > {self.data_args.cutoff_len}.")
            else:
                lengths.append(length)
                length2indexes[length].append(valid_num)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                for k in batch_medias:
                    batch_medias[k].append(examples[k][i])
                valid_num += 1

        model_inputs = defaultdict(list)
        index2lengths = lengths.copy()
        knapsacks = greedy_knapsack(lengths, self.data_args.cutoff_len)
        for knapsack in knapsacks:
            packed_input_ids, packed_attention_masks, packed_labels = [], [], []
            packed_length = 0
            mm_expand_length = 0
            packed_mm_features = {k: [] for k in batch_medias}
            for i, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                packed_input_ids += batch_input_ids[index]
                packed_labels += batch_labels[index]
                packed_length += index2lengths[index]
                mm_expand_length += index2lengths[index] - len(batch_input_ids[index])
                for k in batch_medias:
                    packed_mm_features[k] += batch_medias[k][index]
                if self.data_args.neat_packing:
                    packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
                else:
                    packed_attention_masks += [1] * len(batch_input_ids[index])

            if packed_length < self.data_args.cutoff_len + 1:
                pad_length = self.data_args.cutoff_len - packed_length + 1
                packed_input_ids += [self.tokenizer.pad_token_id] * pad_length
                packed_labels += [IGNORE_INDEX] * pad_length
                if self.data_args.neat_packing:
                    packed_attention_masks += [0] * pad_length
                else:
                    packed_attention_masks += [1] * pad_length  # more efficient flash_attn

            if len(packed_input_ids) + mm_expand_length != self.data_args.cutoff_len + 1:
                raise ValueError("The length of packed example should be identical to the cutoff length.")

            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["labels"].append(packed_labels)
            for k in batch_medias:
                model_inputs[k].append(packed_mm_features[k])

        return model_inputs

@dataclass
class CPPaddedPackedSupervisedDatasetProcessor(PackedSupervisedDatasetProcessor):
    def _get_padded_length(self, length: int) -> int:
        return (length + self.data_args.sequence_length_pad_multiple - 1) // self.data_args.sequence_length_pad_multiple * self.data_args.sequence_length_pad_multiple

    def preprocess_dataset(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        valid_num = 0
        batch_input_ids, batch_labels = [], []
        batch_medias = {k: [] for k in ["image", "video", "audio", "video_frames"] if k in examples}
        lengths = []
        length2indexes = defaultdict(list)
        index2lengths = []
        for i in range(len(examples["prompt"])):
            messages = examples["prompt"][i] + examples["response"][i]
            if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
                logger.warning(f"Dropped invalid example: {messages} due to invalid format.")
                continue

            if examples["video"][i] and examples["video_frames"][i]:
                logger.warning(f"Dropped invalid example: {messages} due to both video and video_frames are provided.")
                continue

            input_ids, labels = self._encode_supervised_example(
                prompt=examples["prompt"][i],
                response=examples["response"][i],
                system=examples["system"][i],
                tools=examples["tools"][i],
                loss_masks=examples["loss_masks"][i] if "loss_masks" in examples else None,
            )

            if not self.template.plugin.validate_input(input_ids, examples, i):
                logger.warning(f"Dropped invalid example: {messages} due to invalid multimodal token numbers.")
                continue

            if all(label == IGNORE_INDEX for label in labels):
                logger.warning(f"Dropped invalid example: {messages} due to no trainable labels after cutoff.")
                continue

            length = self._get_sample_length(examples, input_ids, labels, i)
            padded_length = self._get_padded_length(length)
            if padded_length > self.data_args.cutoff_len:
                logger.warning(f"Dropped lengthy example with length {padded_length} > {self.data_args.cutoff_len}.")
            else:
                index2lengths.append(length)
                lengths.append(padded_length)
                length2indexes[padded_length].append(valid_num)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                for k in batch_medias:
                    batch_medias[k].append(examples[k][i])
                valid_num += 1

        model_inputs = defaultdict(list)
        knapsacks = greedy_knapsack(lengths, self.data_args.cutoff_len)
        for knapsack in knapsacks:
            packed_input_ids, packed_attention_masks, packed_labels = [], [], []
            packed_length = 0
            mm_expand_length = 0
            packed_mm_features = {k: [] for k in batch_medias}
            for i, padded_length in enumerate(knapsack):
                index = length2indexes[padded_length].pop()
                length = index2lengths[index]

                input_ids = batch_input_ids[index]
                labels = batch_labels[index]

                # NOTE: padding each sub-sequence to `sequence_length_pad_multiple`.
                if padded_length > length:
                    pad_length = padded_length - length
                    input_ids += [self.tokenizer.pad_token_id] * pad_length
                    labels += [IGNORE_INDEX] * pad_length

                packed_input_ids += input_ids
                packed_labels += labels
                packed_length += padded_length
                mm_expand_length += padded_length - len(input_ids)
                for k in batch_medias:
                    packed_mm_features[k] += batch_medias[k][index]
                # padding token's attention_mask doesn't need to be set to 0 since its label has been set to IGNORE.
                if self.data_args.neat_packing:
                    packed_attention_masks += [i + 1] * len(input_ids)  # start from 1
                else:
                    packed_attention_masks += [1] * len(input_ids)

            if packed_length < self.data_args.cutoff_len + 1:
                pad_length = self.data_args.cutoff_len - packed_length + 1
                packed_input_ids += [self.tokenizer.pad_token_id] * pad_length
                packed_labels += [IGNORE_INDEX] * pad_length
                if self.data_args.neat_packing:
                    packed_attention_masks += [0] * pad_length
                else:
                    packed_attention_masks += [1] * pad_length  # more efficient flash_attn

            if len(packed_input_ids) + mm_expand_length != self.data_args.cutoff_len + 1:
                raise ValueError("The length of packed example should be identical to the cutoff length.")

            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["labels"].append(packed_labels)
            for k in batch_medias:
                model_inputs[k].append(packed_mm_features[k])

        return model_inputs
