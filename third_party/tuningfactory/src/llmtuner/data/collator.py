# Copyright 2024 OpenAccess AI Collective and the LlamaFactory team.
#
# This code is inspired by the OpenAccess AI Collective's axolotl library.
# https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/monkeypatch/utils.py
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

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Sequence

import torch
from transformers import DataCollatorForSeq2Seq

from ..extras.constants import IGNORE_INDEX


if TYPE_CHECKING:
    from .template import Template


def prepare_4d_attention_mask(attention_mask_with_indices: "torch.Tensor", dtype: "torch.dtype") -> "torch.Tensor":
    r"""
    Expands the attention mask with indices from (batch_size, seq_len) to (batch_size, 1, seq_len, seq_len),
    while handles packed sequences and transforms the mask to lower triangular form to prevent future peeking.

    e.g.
    ```
    [[1, 1, 2, 2, 2, 0]]
    ```
    ->
    ```
    [
        [
            [
                [o, x, x, x, x, x],
                [o, o, x, x, x, x],
                [x, x, o, x, x, x],
                [x, x, o, o, x, x],
                [x, x, o, o, o, x],
                [x, x, x, x, x, x],
            ]
        ]
    ]
    ```
    where `o` equals to `0.0`, `x` equals to `min_dtype`.
    """
    bsz, seq_len = attention_mask_with_indices.size()
    min_dtype = torch.finfo(dtype).min
    expanded_mask = attention_mask_with_indices[:, None, None, :].expand(bsz, 1, seq_len, seq_len)
    # Create a binary mask from the original mask where zeros remain zeros and all other values are set to one
    padding_mask = torch.where(expanded_mask != 0, 1, 0)
    # Create a block-diagonal mask.
    attention_mask_4d = torch.eq(expanded_mask, expanded_mask.transpose(-1, -2)).int() * padding_mask
    # Use the lower triangular mask to zero out the upper triangular part
    attention_mask_4d *= torch.tril(torch.ones((seq_len, seq_len), dtype=torch.long))
    # Invert the attention mask.
    attention_mask_4d = torch.where(attention_mask_4d != 0, torch.tensor(0, dtype=dtype), min_dtype)
    return attention_mask_4d


@dataclass
class MultiModalDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    template: Optional["Template"] = None
    is_turbo: bool = False
    predict_mode: bool = False

    def _pre_process_multi_modal(self, features: Sequence[Dict[str, Any]]):
        """
        append empty image at end to avoid zero3 hangs
        """
        # TODO: support mock audio
        if self.template.name in ["qwen2-audio"]:
            return features
        images, videos = [], []
        for feature in features:
            images.extend(feature.get("image", None) or [])
            videos.extend(feature.get("video", None) or [])
            video_frames = feature.get("video_frames", None) or []
        if len(images) + len(videos) + len(video_frames) > 0:
            return features
        features[0] = self.template.plugin.append_mock_media(features[0])
        return features

    def _process_multi_modal(self, feature: Dict[str, Any]):
        """
        get multi-modal features and process token ids
        """
        images = feature.pop("image", None)
        audios = feature.pop("audio", None)
        videos = feature.pop("video", None)
        video_frames = feature.pop("video_frames", None)
        assert not (videos and video_frames), "Only one of video and video_frames can be provided in one sample."
        videos = videos or video_frames
        not_process_feature = {}
        if "sample_weight" in feature:
            not_process_feature["sample_weight"] = feature.pop("sample_weight")
        if self.predict_mode:
            not_process_feature["labels"] = feature.pop("labels", None)
        # Streaming JSONL can expose sparse metadata columns only after the
        # sample used to infer ``remove_columns``.  Never forward those
        # dataset-only fields (for example ``split``) into the strict plugin
        # API.  The plugin accepts token tensors only; media are passed above.
        model_feature = {
            key: feature[key]
            for key in ("input_ids", "labels", "attention_mask")
            if key in feature
        }
        feature, mm_features = self.template.plugin.process_single_mm_input(
            images=images, audios=audios, videos=videos, **model_feature
        )
        return {**feature, **not_process_feature}, mm_features

    def _process_turbo(self, feature: Dict[str, Any]):
        labels = feature.pop("labels")[1:] + [IGNORE_INDEX]
        return {**feature, "labels": labels}

    def __call__(self, features: Sequence[Dict[str, Any]], return_tensors=None):
        for f in features:
            if "labels" in f:
                continue
            # for pretrain mode
            f["labels"] = copy.deepcopy(f["input_ids"])
            if self.tokenizer.pad_token_id is not None:
                f["labels"] = [label if label != self.tokenizer.pad_token_id else -100 for label in f["labels"]]

        mm_features = {}
        if self.template is not None and self.template.processor is not None:
            features = self._pre_process_multi_modal(features)
            text_features, mm_features = [], []
            for feature in features:
                text_feature, mm_feature = self._process_multi_modal(feature)
                text_features.append(text_feature)
                mm_features.append(mm_feature)
            features = text_features
        else:  # no multimodal features
            for feature in features:
                [feature.pop(k, None) for k in ("image", "audio", "video", "video_frames")]

        if self.is_turbo:
            features = [self._process_turbo(feature) for feature in features]
        features = super().__call__(features)

        if mm_features:
            mm_features = self.template.plugin.mm_collate_fn(mm_features, collated_text_features=features, model=self.model)

        return {**features, **mm_features}


@dataclass
class SFTDataCollatorWith4DAttentionMask(MultiModalDataCollatorForSeq2Seq):
    r"""
    Data collator for 4d attention mask.
    """

    block_diag_attn: bool = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    compute_dtype: "torch.dtype" = torch.float32
    require_position_ids: bool = False

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        if not self.require_position_ids:
            features = [{k: v for k, v in d.items() if k != "position_ids"} for d in features]
        features = super().__call__(features)
        if self.block_diag_attn and self.attn_implementation != "flash_attention_2":
            features["attention_mask"] = prepare_4d_attention_mask(features["attention_mask"], self.compute_dtype)

        return {**features}


@dataclass
class SFTDataCollatorWithSequenceParallel(MultiModalDataCollatorForSeq2Seq):
    r"""
    Data collator for sequence parallel.
    """
    sequence_parallel_size: int = 1
    sequence_parallel_mode: str = "ulysses"
    rank: int = 0

    def process_sp_dataset(self, seq_ids: "torch.Tensor") -> "torch.Tensor":
        if self.sequence_parallel_mode == "ulysses":
            split_size = seq_ids.shape[1] // self.sequence_parallel_size
            chunks = torch.split(seq_ids, split_size, dim=1)
            local_values = chunks[self.rank]
            return local_values
        elif self.sequence_parallel_mode == "zigzag-ring":
            split_size = seq_ids.shape[1] // (2 * self.sequence_parallel_size)
            chunks = torch.split(seq_ids, split_size, dim=1)
            local_values = torch.cat((chunks[self.rank], chunks[2 * self.sequence_parallel_size - self.rank - 1]), dim=1)
            return local_values
        else:
            raise NotImplementedError(f"{self.sequence_parallel_mode} is not implemented.")

    def sp_pad_sequence(self, features: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
        # add position_ids
        bsz, seq_len = features["input_ids"].shape
        features["position_ids"] = torch.arange(seq_len).expand(bsz, -1)

        # shift and pad labels
        pad_token_id = self.label_pad_token_id
        pad_sequence = torch.full((features["labels"].shape[0], 1), pad_token_id, dtype=features["labels"].dtype)
        features["labels"] = torch.cat([features["labels"][:, 1:], pad_sequence], dim=1)
        return features

    def sp_split(self, features: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
        for k, v in features.items():
            if k.endswith("attention_mask"):
                continue
            features[k] = self.process_sp_dataset(v)
        return features

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        features = super().__call__(features)
        features = self.sp_pad_sequence(features)
        return self.sp_split(features)


@dataclass
class DistillDataCollatorWith4DAttentionMask(MultiModalDataCollatorForSeq2Seq):
    r"""
    Data collator for Distillation
    """
    teacher_template: str = None

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        student_features , teacher_features = [], []
        for key in ("student", "teacher"):
            for feature in features:
                target_feature = {}
                target_feature.update({k.replace("{}_".format(key), ""): v for k, v in feature.items() if k.startswith(key)})
                target_feature.update({k: v for k, v in feature.items() if not k.startswith(("teacher", "student"))})

                if key == "student":
                    student_features.append(target_feature)
                elif key == "teacher":
                    teacher_features.append(target_feature)
        student_features = super().__call__(student_features)
        super_template = self.template
        self.template = self.teacher_template
        teacher_features = super().__call__(teacher_features)
        self.template = super_template
        return {
            'student_features': student_features,
            'teacher_features': teacher_features
        }


@dataclass
class PairwiseDataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        for key in ("chosen", "rejected"):
            for feature in features:
                target_feature = {
                    "input_ids": feature["{}_input_ids".format(key)],
                    "attention_mask": feature["{}_attention_mask".format(key)],
                    "labels": feature["{}_labels".format(key)],
                }
                target_feature.update({k: v for k, v in feature.items() if not k.startswith(("chosen", "rejected"))})
                concatenated_features.append(target_feature)

        return super().__call__(concatenated_features)


@dataclass
class KTODataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
    r"""
    Data collator for KTO data.
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        target_features = []
        kl_features = []
        kto_tags = []
        for feature in features:
            target_feature ={
                "input_ids": feature.pop("input_ids"),
                "attention_mask": feature.pop("attention_mask"),
                "labels": feature.pop("labels"),
            }
            kl_feature = {
                "input_ids": feature.pop("kl_input_ids"),
                "attention_mask": feature.pop("kl_attention_mask"),
                "labels": feature.pop("kl_labels"),
            }
            kto_tags.append(feature.pop("kto_tags"))

            target_feature.update(feature)
            kl_feature.update(feature)

            target_features.append(target_feature)
            kl_features.append(kl_feature)

        batch = super().__call__(target_features)
        kl_batch = super().__call__(kl_features)
        batch["kl_input_ids"] = kl_batch["input_ids"]
        batch["kl_attention_mask"] = kl_batch["attention_mask"]
        batch["kl_labels"] = kl_batch["labels"]

        batch["kto_tags"] = torch.tensor(kto_tags)
        return batch
