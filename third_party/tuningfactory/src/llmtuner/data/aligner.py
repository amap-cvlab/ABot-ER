import os
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Union
import json

from ..extras.constants import AUDIO_PLACEHOLDER, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER
from ..extras.logging import get_logger
from .ailake_utils import split_ailake_bytes_data
from .utils import Role, get_sample_data


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments

    from ..hparams import DataArguments
    from .parser import DatasetAttr
    from .template import Template


logger = get_logger(__name__)


def convert_multimodal(multimodals: List[Any], multimodal_folder: str, dataset_attr: "DatasetAttr") -> List[Any]:
    r"""
    Optionally concatenates image path to dataset dir when loading from local disk.
    """
    if multimodals is None:
        return []
    if dataset_attr.dataset_name.startswith("ailake"):
        multimodals = split_ailake_bytes_data(multimodals)

    if not isinstance(multimodals, list):
        multimodals = [multimodals]
    elif len(multimodals) == 0:
        return []
    else:
        multimodals = multimodals[:]

    def convert_path(obj: Any) -> Any:
        if multimodal_folder is not None and isinstance(obj, str) and not obj.startswith(("http://", "https://")):
            return os.path.join(multimodal_folder, obj)
        else:
            return obj

    if dataset_attr.load_from in ["script", "file"]:
        for i in range(len(multimodals)):
            if isinstance(multimodals[i], list):
                multimodals[i] = [convert_path(o) for o in multimodals[i]]
            else:
                multimodals[i] = convert_path(multimodals[i])

    return multimodals


def convert_placeholder_token(example: Dict[str, Any], multimodel_token_maps: dict[str, str]):
    def replace_token(text: str):
        if multimodel_token_maps["image"] is not None and "image" in example:
            text = text.replace(IMAGE_PLACEHOLDER, multimodel_token_maps["image"])
        if multimodel_token_maps["video"] is not None and ("video" in example or "video_frames" in example):
            text = text.replace(VIDEO_PLACEHOLDER, multimodel_token_maps["video"])
        if multimodel_token_maps["audio"] is not None and "audio" in example:
            text = text.replace(AUDIO_PLACEHOLDER, multimodel_token_maps["audio"])
        return text

    def replace_content_list(content_list: List[Dict[str, str]]):
        return [{"role": content["role"], "content": replace_token(content["content"])} for content in content_list]

    example["prompt"] = replace_content_list(example["prompt"])
    example["response"] = replace_content_list(example["response"])
    example["system"] = replace_token(example["system"])
    example["tools"] = replace_token(example["tools"])
    return example


@dataclass
class AlpacaDatasetConverter:
    dataset_attr: "DatasetAttr"
    data_args: "DataArguments"
    multimodel_token_maps: dict[str, str]

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        r"""
        Converts alpaca format dataset to the standard format.
        """
        image_folder = self.dataset_attr.image_folder or self.data_args.image_folder
        video_folder = self.dataset_attr.video_folder or self.data_args.video_folder
        audio_folder = self.dataset_attr.audio_folder or self.data_args.audio_folder
        convert_images = partial(
            convert_multimodal, dataset_attr=self.dataset_attr, multimodal_folder=image_folder
        )
        convert_videos = partial(
            convert_multimodal, dataset_attr=self.dataset_attr, multimodal_folder=video_folder
        )
        convert_audios = partial(
            convert_multimodal, dataset_attr=self.dataset_attr, multimodal_folder=audio_folder
        )

        prompt = []
        if self.dataset_attr.history and isinstance(example[self.dataset_attr.history], list):
            for old_prompt, old_response in example[self.dataset_attr.history]:
                prompt.append({"role": Role.USER.value, "content": old_prompt})
                prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

        content = []
        if self.dataset_attr.prompt and example[self.dataset_attr.prompt]:
            content.append(example[self.dataset_attr.prompt])

        if self.dataset_attr.query and example[self.dataset_attr.query]:
            content.append(example[self.dataset_attr.query])

        prompt.append({"role": Role.USER.value, "content": "\n".join(content)})  # "prompt\nquery"

        if self.dataset_attr.kto_tag and isinstance(example[self.dataset_attr.kto_tag], bool):  # kto example
            response = [{"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.response]}]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], str)
            and isinstance(example[self.dataset_attr.rejected], str)
        ):  # pairwise example
            response = [
                {"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.chosen]},
                {"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.rejected]},
            ]
        elif self.dataset_attr.response and isinstance(
            example[self.dataset_attr.response], (int, str)
        ):  # normal example
            response = [{"role": Role.ASSISTANT.value, "content": str(example[self.dataset_attr.response])}]
        else:  # unsupervised
            response = []

        loss_masks = self._build_loss_masks(example, num_history_turns=len(prompt) // 2)

        output = {
            "prompt": prompt,
            "response": response,
            "system": example[self.dataset_attr.system] if self.dataset_attr.system else "",
            "tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "image": convert_images(example.get(self.dataset_attr.image)) if self.dataset_attr.image else [],
            "video": convert_videos(example.get(self.dataset_attr.video)) if self.dataset_attr.video else [],
            "video_frames": convert_videos(example.get(self.dataset_attr.video_frames)) if self.dataset_attr.video_frames else [],
            "audio": convert_audios(example.get(self.dataset_attr.audio)) if self.dataset_attr.audio else [],
            "loss_masks": loss_masks,
            "sample_weight": float(example.get(self.dataset_attr.sample_weight, 1.0))
            if self.dataset_attr.sample_weight else 1.0,
        }
        return convert_placeholder_token(output, self.multimodel_token_maps)

    def _build_loss_masks(self, example: dict[str, Any], num_history_turns: int) -> List[List[int]]:
        r"""
        Build per-turn loss masks from `history_masks` column.

        Raw shape is one entry per history turn (length must equal `num_history_turns`).
        Each entry is either:
          - a scalar `0/1` → only controls assistant; user follows `train_on_prompt`,
          - a `[u_mask, a_mask]` 2-list → explicitly overrides both, takes precedence over `train_on_prompt`.

        Returns a normalized list of `[u_or_-1, a_mask]` pairs of length `num_history_turns + 1`.
        `u_or_-1 == -1` is a sentinel meaning "fall back to `train_on_prompt`".
        The final instruction+output turn is appended as `[-1, 1]` (always trained, user follows `train_on_prompt`).
        Returns [] (sentinel) when:
          - the column is not configured,
          - the raw value is missing/None,
          - the length does not match `num_history_turns`,
          - any entry is malformed.
        Empty list signals downstream "use the default mask logic".
        """
        col = self.dataset_attr.history_masks
        if not col:
            return []
        raw = example.get(col)
        if raw is None:
            return []
        if not isinstance(raw, list) or len(raw) != num_history_turns:
            logger.warning_once(
                f"`history_masks` length mismatch: got {len(raw) if hasattr(raw, '__len__') else type(raw).__name__}, "
                f"expected {num_history_turns}. Falling back to default mask for affected sample(s)."
            )
            return []

        per_turn: List[List[int]] = []
        for entry in raw:
            if isinstance(entry, bool) or isinstance(entry, int):
                # scalar: assistant mask only; user follows train_on_prompt
                per_turn.append([-1, int(entry)])
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                u_mask, a_mask = int(entry[0]), int(entry[1])
                per_turn.append([u_mask, a_mask])
            else:
                logger.warning_once(
                    "`history_masks` entries must be int (assistant only) or [user_mask, assistant_mask]; "
                    "malformed entry encountered, falling back to default mask for affected sample(s)."
                )
                return []

        per_turn.append([-1, 1])  # final (instruction+output) turn: always trained, user follows train_on_prompt
        return per_turn


@dataclass
class SharegptDatasetConverter:
    dataset_attr: "DatasetAttr"
    data_args: "DataArguments"
    multimodel_token_maps: dict[str, str]

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        r"""
        Converts sharegpt format dataset to the standard format.
        """
        image_folder = self.dataset_attr.image_folder or self.data_args.image_folder
        video_folder = self.dataset_attr.video_folder or self.data_args.video_folder
        audio_folder = self.dataset_attr.audio_folder or self.data_args.audio_folder
        convert_images = partial(
            convert_multimodal, dataset_attr=self.dataset_attr, multimodal_folder=image_folder
        )
        convert_videos = partial(
            convert_multimodal, dataset_attr=self.dataset_attr, multimodal_folder=video_folder
        )
        convert_audios = partial(
            convert_multimodal, dataset_attr=self.dataset_attr, multimodal_folder=audio_folder
        )
        tag_mapping = {
            self.dataset_attr.user_tag: Role.USER.value,
            self.dataset_attr.assistant_tag: Role.ASSISTANT.value,
            self.dataset_attr.observation_tag: Role.OBSERVATION.value,
            self.dataset_attr.function_tag: Role.FUNCTION.value,
            self.dataset_attr.system_tag: Role.SYSTEM.value,
        }
        odd_tags = (self.dataset_attr.user_tag, self.dataset_attr.observation_tag)
        even_tags = (self.dataset_attr.assistant_tag, self.dataset_attr.function_tag)
        accept_tags = (odd_tags, even_tags)

        messages = example[self.dataset_attr.messages]
        if isinstance(messages, str):
            messages = json.loads(messages)

        if (
            self.dataset_attr.system_tag
            and len(messages) != 0
            and messages[0][self.dataset_attr.role_tag] == self.dataset_attr.system_tag
        ):
            system = messages[0][self.dataset_attr.content_tag]
            messages = messages[1:]
        else:
            system = example[self.dataset_attr.system] if self.dataset_attr.system else ""

        aligned_messages = []
        broken_data = False
        for turn_idx, message in enumerate(messages):
            if message[self.dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
                logger.warning("Invalid role tag in {}.".format(messages))
                broken_data = True
                break

            aligned_messages.append(
                {
                    "role": tag_mapping[message[self.dataset_attr.role_tag]],
                    "content": message[self.dataset_attr.content_tag],
                }
            )

        if (not self.dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
            self.dataset_attr.ranking and len(aligned_messages) % 2 == 0
        ):
            logger.warning("Invalid message count in {}.".format(messages))
            broken_data = True

        if broken_data:
            logger.warning("Skipping this abnormal example.")
            prompt, response = [], []
        elif self.dataset_attr.kto_tag and isinstance(example[self.dataset_attr.kto_tag], bool):  # kto example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], dict)
            and isinstance(example[self.dataset_attr.rejected], dict)
        ):  # pairwise example
            chosen = example[self.dataset_attr.chosen]
            rejected = example[self.dataset_attr.rejected]
            if (
                chosen[self.dataset_attr.role_tag] not in accept_tags[-1]
                or rejected[self.dataset_attr.role_tag] not in accept_tags[-1]
            ):
                logger.warning("Invalid role tag in {}.".format([chosen, rejected]))
                broken_data = True

            prompt = aligned_messages
            response = [
                {
                    "role": tag_mapping[chosen[self.dataset_attr.role_tag]],
                    "content": chosen[self.dataset_attr.content_tag],
                },
                {
                    "role": tag_mapping[rejected[self.dataset_attr.role_tag]],
                    "content": rejected[self.dataset_attr.content_tag],
                },
            ]
        else:  # normal example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]

        output = {
            "prompt": prompt,
            "response": response,
            "system": system,
            "tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "image": convert_images(example.get(self.dataset_attr.image)) if self.dataset_attr.image else [],
            "video": convert_videos(example.get(self.dataset_attr.video)) if self.dataset_attr.video else [],
            "video_frames": convert_videos(example.get(self.dataset_attr.video_frames)) if self.dataset_attr.video_frames else [],
            "audio": convert_audios(example.get(self.dataset_attr.audio)) if self.dataset_attr.audio else [],
            "sample_weight": float(example.get(self.dataset_attr.sample_weight, 1.0))
            if self.dataset_attr.sample_weight else 1.0,
        }
        return convert_placeholder_token(output, self.multimodel_token_maps)


DATASET_CONVERTERS = {
    "alpaca": AlpacaDatasetConverter,
    "sharegpt": SharegptDatasetConverter,
}


def get_dataset_converter(name: str, dataset_attr: "DatasetAttr", data_args: "DataArguments", multimodel_token_maps: dict[str, str]):
    r"""Get a dataset converter."""
    if name not in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} not found.")

    return DATASET_CONVERTERS[name](dataset_attr, data_args, multimodel_token_maps)


def align_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    template: "Template",
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Aligned dataset:
        prompt: [{"role": "user", "content": "..."}] * (2T - 1)
        response: [{"role": "assistant", "content": "..."}] * N (N > 1 for ranking dataset)
        system: "..."
        tools: "..."
    """
    multimodel_token_maps = {
        "image": template.image_token,
        "video": template.video_token,
        "audio": template.audio_token,
    }
    dataset_converter = get_dataset_converter(dataset_attr.formatting, dataset_attr, data_args, multimodel_token_maps)

    sample_data = get_sample_data(dataset=dataset, use_subprocess=True)
    column_names = set(sample_data.keys())
    feature_keys = {
        "prompt", "response", "system", "tools", "image", "video", "video_frames", "audio",
        "loss_masks", "sample_weight",
    }
    kwargs = {}
    if not data_args.streaming:
        local_process_index = int(os.getenv("LOCAL_PROCESS_RANK", training_args.local_process_index))
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (local_process_index != 0),
            desc="Converting format of dataset",
        )

    dataset = dataset.map(
        dataset_converter,
        batched=False,
        remove_columns=sorted(column_names - feature_keys),
        **kwargs,
    )
    return dataset
