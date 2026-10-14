import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from ..extras.constants import DATA_CONFIG
from ..extras.misc import use_modelscope


if TYPE_CHECKING:
    from ..hparams import DataArguments


@dataclass
class DatasetAttr:
    r"""
    Dataset attributes.
    """

    # basic configs
    load_from: Literal["hf_hub", "ms_hub", "script", "file"]
    dataset_name: str
    formatting: Literal["alpaca", "sharegpt"] = "alpaca"
    ranking: bool = False
    # extra configs
    subset: Optional[str] = None
    folder: Optional[str] = None
    num_samples: Optional[int] = None
    # per-dataset multimodal path prefixes; override global DataArguments.{image,video,audio}_folder
    image_folder: Optional[str] = None
    video_folder: Optional[str] = None
    audio_folder: Optional[str] = None
    # common columns
    system: Optional[str] = None
    tools: Optional[str] = None
    image: Optional[str] = None
    video: Optional[str] = None
    video_frames: Optional[str] = None
    audio: Optional[str] = None
    sample_weight: Optional[str] = None
    # rlhf columns
    chosen: Optional[str] = None
    rejected: Optional[str] = None
    kto_tag: Optional[str] = None
    # alpaca columns
    prompt: Optional[str] = "instruction"
    query: Optional[str] = "input"
    response: Optional[str] = "output"
    history: Optional[str] = None
    history_masks: Optional[str] = None
    # sharegpt columns
    messages: Optional[str] = "conversations"
    # sharegpt tags
    role_tag: Optional[str] = "from"
    content_tag: Optional[str] = "value"
    user_tag: Optional[str] = "human"
    assistant_tag: Optional[str] = "gpt"
    observation_tag: Optional[str] = "observation"
    function_tag: Optional[str] = "function_call"
    system_tag: Optional[str] = "system"

    def __repr__(self) -> str:
        return self.dataset_name

    def set_attr(self, key: str, obj: Dict[str, Any], default: Optional[Any] = None) -> None:
        setattr(self, key, obj.get(key, default))


def get_dataset_list(data_args: "DataArguments", get_eval=False) -> List["DatasetAttr"]:
    if get_eval:
        dataset_name, file_name = data_args.eval_dataset_name, data_args.eval_file_name
    else:
        dataset_name, file_name = data_args.dataset_name, data_args.file_name

    if dataset_name or file_name:
        def get_dataset_attr(dataset_name, file_name):
            if dataset_name is None and file_name is None:
                return None
            return DatasetAttr(
                load_from="hf_hub" if dataset_name else "file",
                formatting="sharegpt" if data_args.messages is not None else "alpaca",
                dataset_name=dataset_name or file_name,
                **data_args.get_data_attrs()
            )
        dataset_attr = get_dataset_attr(dataset_name, file_name)
        return [dataset_attr] if dataset_attr else []

    dataset_str = data_args.eval_dataset if get_eval else data_args.dataset
    dataset_names = [ds.strip() for ds in dataset_str.split(",")] if dataset_str is not None else []
    try:
        if data_args.dataset_info_json is not None:
            dataset_info = data_args.dataset_info_json
        else:
            with open(os.path.join(data_args.dataset_dir, DATA_CONFIG), "r") as f:
                dataset_info = json.load(f)
    except Exception as err:
        if data_args.dataset is not None:
            raise ValueError(
                "Cannot open {} due to {}.".format(os.path.join(data_args.dataset_dir, DATA_CONFIG), str(err))
            )
        dataset_info = None

    dataset_list: List[DatasetAttr] = []
    for name in dataset_names:
        if name not in dataset_info:
            raise ValueError("Undefined dataset {} in {}.".format(name, DATA_CONFIG))

        has_hf_url = "hf_hub_url" in dataset_info[name]
        has_ms_url = "ms_hub_url" in dataset_info[name]

        if has_hf_url or has_ms_url:
            if (use_modelscope() and has_ms_url) or (not has_hf_url):
                dataset_attr = DatasetAttr("ms_hub", dataset_name=dataset_info[name]["ms_hub_url"])
            else:
                dataset_attr = DatasetAttr("hf_hub", dataset_name=dataset_info[name]["hf_hub_url"])
        elif "script_url" in dataset_info[name]:
            dataset_attr = DatasetAttr("script", dataset_name=dataset_info[name]["script_url"])
        else:
            dataset_attr = DatasetAttr("file", dataset_name=dataset_info[name]["file_name"])

        dataset_attr.set_attr("subset", dataset_info[name])
        dataset_attr.set_attr("folder", dataset_info[name])
        dataset_attr.set_attr("image_folder", dataset_info[name])
        dataset_attr.set_attr("video_folder", dataset_info[name])
        dataset_attr.set_attr("audio_folder", dataset_info[name])
        dataset_attr.set_attr("ranking", dataset_info[name], default=False)
        dataset_attr.set_attr("formatting", dataset_info[name], default="alpaca")

        if "columns" in dataset_info[name]:
            column_names = [
                "system",
                "tools",
                "image",
                "video",
                "audio",
                "video_frames",
                "sample_weight",
                "chosen",
                "rejected",
                "kto_tag",
            ]
            if dataset_attr.formatting == "alpaca":
                column_names.extend(["prompt", "query", "response", "history", "history_masks"])
            else:
                column_names.extend(["messages"])

            for column_name in column_names:
                dataset_attr.set_attr(column_name, dataset_info[name]["columns"])

        if dataset_attr.formatting == "sharegpt" and "tags" in dataset_info[name]:
            tag_names = (
                "role_tag",
                "content_tag",
                "user_tag",
                "assistant_tag",
                "observation_tag",
                "function_tag",
                "system_tag",
            )
            for tag in tag_names:
                dataset_attr.set_attr(tag, dataset_info[name]["tags"])

        dataset_list.append(dataset_attr)

    return dataset_list
