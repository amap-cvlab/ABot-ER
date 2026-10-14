import json
from dataclasses import dataclass, field, fields
from typing import Literal, Optional
from ..extras.logging import get_logger
from ..extras.misc import set_odps_table_reader_mode

logger = get_logger(__name__)


@dataclass
class DataAttrArguments:
    prompt: Optional[str] = field(
        default=None, metadata={"help": "Which column to use as prompt in training and eval."}
    )
    query: Optional[str] = field(default=None, metadata={"help": "Which column to use as query in training and eval."})
    response: Optional[str] = field(
        default=None, metadata={"help": "Which column to use as response in training and eval."}
    )
    history: Optional[str] = field(
        default=None, metadata={"help": "Which column to use as history in training and eval."}
    )
    history_masks: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which column to use as per-turn loss masks for `history` (alpaca format). "
                    "Length must equal `len(history)`. Each entry is either: "
                    "(a) a scalar 0/1 → controls assistant only; user follows `train_on_prompt`; "
                    "(b) a `[user_mask, assistant_mask]` 2-list → explicitly overrides both, "
                    "takes precedence over `train_on_prompt` for that turn. "
                    "Mixed forms in one list are allowed, e.g. `[[1,1], 0, 1]`. "
                    "1 = train, 0 = mask. The final (instruction+output) turn is always trained "
                    "with user side following `train_on_prompt`. Takes precedence over `mask_history`."
        },
    )

    system: Optional[str] = field(
        default=None,
        metadata={"help": "Which column to use as system in training and eval. Conflicts with `--system_prompt`."},
    )
    # rlhf columns
    chosen: Optional[str] = field(
        default=None, metadata={"help": "Which column to use as chosen in DPO/RM stage."}
    )
    rejected: Optional[str] = field(
        default=None, metadata={"help": "Which column to use as rejected in DPO/RM stage."}
    )
    kto_tag: Optional[str] = field(default=None, metadata={"help": "Which column to use as KTO tag."})
    ranking: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use ranking loss or not. True for rm and dpo stage."}
    )

    image: Optional[str] = field(default=None, metadata={"help": "Which column to use as image in training and eval."})
    video: Optional[str] = field(default=None, metadata={"help": "Which column to use as video in training and eval."})
    video_frames: Optional[str] = field(default=None, metadata={"help": "Which column to use as video_frames in training and eval."})
    audio: Optional[str] = field(default=None, metadata={"help": "Which column to use as audio in training and eval."})
    messages: Optional[str] = field(
        default=None, metadata={"help": "Which column to use as messages in training and eval (sharedgpt format)."}
    )
    tools: Optional[str] = field(
        default=None, metadata={"help": "Which column to use as tools in training and eval (sharedgpt format)."}
    )
    sample_weight: Optional[str] = field(
        default=None,
        metadata={"help": "Which column contains a positive per-example SFT loss weight."},
    )

    # extra configs
    subset: Optional[str] = field(
        default=None, metadata={"help": "Which OpenLM/HF dataset subset to use for training and evaluation."}
    )
    folder: Optional[str] = field(
        default=None, metadata={"help": "Which OpenLM/HF dataset folder to use for training and evaluation."}
    )

    # sharegpt tags
    role_tag: Optional[str] = field(default="from", metadata={"help": "role tag for sharegpt format default: `from`"})
    content_tag: Optional[str] = field(default="value", metadata={"help": "content tag for sharegpt format default: `value`"})
    user_tag: Optional[str] = field(default="human", metadata={"help": "user tag for sharegpt format default: `human`"})
    assistant_tag: Optional[str] = field(default="gpt", metadata={"help": "assistant tag for sharegpt format default: `gpt`"})
    observation_tag: Optional[str] = field(
        default="observation", metadata={"help": "observation tag for sharegpt format default: `observation`"}
    )
    function_tag: Optional[str] = field(
        default="function_call", metadata={"help": "function tag for sharegpt format default: `function_call`"}
    )
    system_tag: Optional[str] = field(
        default="system", metadata={"help": "system tag for sharegpt format default: `system`"}
    )

    def get_data_attrs(self):
        return {k.name: getattr(self, k.name) for k in fields(DataAttrArguments)}


@dataclass
class DataArguments(DataAttrArguments):
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """

    template: Optional[str] = field(
        default=None,
        metadata={"help": "Which template to use for constructing prompts in training and inference."},
    )
    teacher_template: Optional[str] = field(
        default=None,
        metadata={"help": "Which template to use for teachers constructing prompts in training and inference."},
    )
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The name of provided dataset(s) to use. Use commas to separate multiple datasets."},
    )
    eval_dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of provided dataset(s) to use for evaluation. Use commas to separate multiple datasets."
        },
    )
    eval_dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of provided dataset(s) to use for evaluation. Use commas to separate multiple datasets."
        },
    )
    dataset_dir: Optional[str] = field(
        default="data",
        metadata={"help": "Path to the folder containing the datasets."},
    )
    split: Optional[str] = field(
        default="train",
        metadata={"help": "Which dataset split to use for training and evaluation."},
    )
    cutoff_len: Optional[int] = field(
        default=2048,
        metadata={"help": "The cutoff length of the model inputs after tokenization."},
    )
    train_on_prompt: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to disable the mask on the prompt or not."},
    )
    mask_history: bool = field(
        default=False,
        metadata={"help": "Whether or not to mask the history and train on the last turn only."},
    )
    streaming: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable dataset streaming."},
    )
    buffer_size: Optional[int] = field(
        default=1024,
        metadata={"help": "Size of the buffer to randomly sample examples from in dataset streaming."},
    )
    mix_strategy: Optional[Literal["concat", "interleave_under", "interleave_over"]] = field(
        default="concat",
        metadata={"help": "Strategy to use in dataset mixing (concat/interleave) (undersampling/oversampling)."},
    )
    interleave_probs: Optional[str] = field(
        default=None,
        metadata={"help": "Probabilities to sample data from datasets. Use commas to separate multiple datasets."},
    )
    overwrite_cache: Optional[bool] = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of examples for each dataset."},
    )
    eval_num_beams: Optional[int] = field(
        default=None,
        metadata={"help": "Number of beams to use for evaluation. This argument will be passed to `model.generate`"},
    )
    ignore_pad_token_for_loss: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether or not to ignore the tokens corresponding to padded labels in the loss computation."
        },
    )
    val_size: Optional[float] = field(
        default=0,
        metadata={"help": "Size of the development set, should be an integer or a float in range `[0,1)`."},
    )
    packing: Optional[bool] = field(
        default=None,
        metadata={"help": "Enable sequences packing in training. Will automatically enable in pre-training."},
    )
    neat_packing: bool = field(
        default=False,
        metadata={"help": "Enable sequence packing without cross-attention."},
    )
    sequence_length_pad_multiple: int = field(
        default=None,
        metadata={"help": "Valid only when CP > 1 and packing=True. The multiple to pad each sequence to. With CP enabled, this should be set to a multiple of 2*CP."},
    )
    image_folder: Optional[str] = field(default=None, metadata={"help": "Path to the folder containing the images."})
    video_folder: Optional[str] = field(default=None, metadata={"help": "Path to the folder containing the videos."})
    audio_folder: Optional[str] = field(default=None, metadata={"help": "Path to the folder containing the audios."})
    drop_no_image: Optional[bool] = field(default=False, metadata={"help": "Drop samples with no image."})
    drop_no_multimodal: Optional[bool] = field(default=False, metadata={"help": "Drop samples with no multimodal input."})
    tables: Optional[str] = field(default=None, metadata={"help": "dummy argument for odps auth"})

    # ------- 数据加载层在线修复（src/llmtuner/data/fixers.py） -------
    apply_data_fixes: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable in-loader data fix pipeline (see src/llmtuner/data/fixers.py)."},
    )
    data_fixes: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated fixer names to apply in order; empty/None falls back to "
                    "fixers.DEFAULT_FIXER_SEQUENCE."
        },
    )
    max_images_per_sample: Optional[int] = field(
        default=10,
        metadata={"help": "Per-sample image cap used by the `cap_images` fixer."},
    )

    dataset_info_json: Optional[str] = field(
        default=None, metadata={"help": "The JSON string of config for multiple datasets info."}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of openlm-dataset for train. Conflicts with `--file_name`"}
    )
    subset: Optional[str] = field(
        default=None, metadata={"help": "The subset name of openlm-dataset for train. work with `--dataset_name`"}
    )
    file_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of file path or odps table name for train. Conflicts with `--dataset_name`"},
    )
    eval_dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of openlm-dataset. Conflicts with `--eval_file_name`"}
    )
    eval_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of file path or odps table name for eval. Conflicts with `--eval_dataset_name`"},
    )
    dynamic_probs: Optional[bool] = field(
        default=False, metadata={"help": "Dynamic probabilities to sample data from datasets"}
    )
    train_dataset_describe: Optional[str] = field(
        default=None,
        metadata={"help": "train dataset describe info, be used for dataio failover check"},
    )
    enable_thinking: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to enable thinking mode for reasoning models."},
    )
    tokenized_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to save or load the tokenized datasets. "
                "If tokenized_path not exists, it will save the tokenized datasets. "
                "If tokenized_path exists, it will load the tokenized datasets."
            )
        },
    )
    shuffle_for_sequence_parallel: Optional[bool] = field(
        default=True,
        metadata={"help": "Shuffle the dataset for sequence parallel training."},
    )

    def __post_init__(self):
        logger.warning("`image_folder` used as `video_folder` will be deprecated in the future. Use `video_folder` instead if set video.")

        if self.streaming and self.val_size > 1e-6 and self.val_size < 1:
            raise ValueError("Streaming mode should have an integer val size.")

        if self.video is not None and self.video_folder is None and self.image_folder is not None:
            self.video_folder = self.image_folder

        if self.drop_no_image and not self.drop_no_multimodal:
            self.drop_no_multimodal = True
            logger.warning("`drop_no_image` will be Deprecated in the future. Use `drop_no_multimodal` instead.")

        if self.streaming and self.max_samples is not None:
            raise ValueError("`max_samples` is incompatible with `streaming`.")

        if self.mask_history and self.train_on_prompt:
            raise ValueError("`mask_history` is incompatible with `train_on_prompt`.")

        if self.dataset_info_json is not None:
            try:
                self.dataset_info_json = json.loads(self.dataset_info_json)
            except json.JSONDecodeError:
                raise ValueError(f"`dataset_info_json` {self.dataset_info_json} is not a valid JSON string.")

        if self.interleave_probs is not None:
            self.interleave_probs = [float(prob.strip()) for prob in self.interleave_probs.split(",")]

        assert not (self.dataset_name and self.file_name), "only support one dataset from openlm or file"
        assert not (
            self.eval_dataset_name and self.eval_file_name
        ), "only support one eval dataset from openlm or file"
        assert not (
            self.messages and self.prompt
        ), "messages config for sharedgpt format datasets, prompt config for alpaca format datasets, cannot be both"

        if (
            self.file_name
            and self.file_name.startswith("odps://")
            and (self.history or self.messages or self.image or self.video or self.audio or self.streaming)
        ):
            set_odps_table_reader_mode()
