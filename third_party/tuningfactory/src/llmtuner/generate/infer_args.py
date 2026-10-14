import copy
import json
import os
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import torch
from transformers import HfArgumentParser

from ..data import get_ailake_columns_name, get_odps_columns_name, get_odps_columns_name_in_tunnel
from ..extras.logging import get_logger
from ..extras.misc import set_odps_table_reader_mode
from ..hparams import FinetuningArguments, GeneratingArguments, ModelArguments
from .utils import is_external_cluster


logger = get_logger(__name__)


def get_column_index(column, column_name_list):
    if column is not None:
        column = int(column) if column.isdigit() else column_name_list.index(column)
    return column


@dataclass
class InferArguments:
    r"""
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """
    template: str = field(
        metadata={"help": "Which template to use for constructing prompts in training and inference."},
    )
    inputs: str = field(default=None, metadata={"help": "Input dataset for inference."})
    batch_size: int = field(default=1, metadata={"help": "Batch size for inference."})
    seed: Optional[int] = field(default=1, metadata={"help": "Seed for randomness."})
    compute_dtype: Optional[Literal[torch.float32, torch.bfloat16, torch.float16]] = field(
        default=None, metadata={"help": "set model dtype, otherwise use config's torch_dtype"}
    )
    infer_mode: Optional[Literal["default", "vllm-async", "vllm", "vllm-multi-node"]] = field(
        default="vllm-async",
    )
    input_columns: Optional[str] = field(
        default="",
    )
    # ailake数据中用于表示非多模态二进制数据的列，仅用于ailake，无需配置，仅参数传递用
    # TODO: 后续dataloader需细分为ailake dataloader和odps dataloader，到时候将这段逻辑移过去
    ailake_non_multimodel_data_columns: Optional[str] = field(
        default=None,
    )
    prompt_column: Optional[str] = field(
        default="0",
    )
    system_column: Optional[str] = field(
        default=None,
    )
    response_column: Optional[str] = field(
        default=None,
        metadata={"help": "Used for RM model, to calculate assistant score"},
    )
    history_column: Optional[str] = field(
        default=None,
    )
    query_column: Optional[str] = field(
        default=None,
    )
    image_column: Optional[str] = field(
        default=None,
    )
    image: Optional[str] = field(
        default=None,
    )
    image_folder: Optional[str] = field(default=None, metadata={"help": "Path to the folder containing the images."})
    video: Optional[str] = field(
        default=None,
    )
    video_folder: Optional[str] = field(default=None, metadata={"help": "Path to the folder containing the videos."})
    audio: Optional[str] = field(
        default=None,
    )
    audio_folder: Optional[str] = field(default=None, metadata={"help": "Path to the folder containing the audios."})
    mix_multimodel: Optional[bool] = field(
        default=False,
        metadata={"help": "use image, video, audio in the meantime"},
    )
    tables: Optional[str] = field(
        default=None,
    )
    outputs: str = field(default=None, metadata={"help": "Output path for inference."})
    cutoff_len: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum length of the formatted prompt after tokenization."},
    )
    # for openlm-eval
    load_from: Optional[Literal["file", "openlm"]] = field(
        default="file",
        metadata={"help": "Load dataset from openlm or file(odps/oss).", "choices": ["file", "openlm"]},
    )
    dataset_split: Optional[str] = field(
        default="test",
        metadata={
            "help": "OpenLM Dataset split for inference. normally 'validation','test', 'train'. split with , for multiple datasets.",
        },
    )
    field_mapping: Optional[str] = field(default=None, metadata={"help": "Field mapping for openlm-dataset."})

    ######vLLM config######
    pipeline_parallel_size: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of pipeline stages for vLLM",
        },
    )
    tensor_parallel_size: Optional[int] = field(
        default=torch.cuda.device_count(),
        metadata={
            "help": "Number of tensor parallel replicas for vLLM",
        },
    )
    enable_expert_parallel: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use expert parallelism instead of tensor parallelism for MoE layers."
        },
    )
    gpu_memory_utilization: Optional[float] = field(
        default=0.8,
        metadata={
            "help": "Fraction of GPU memory to use for the vLLM execution. default set 0.8",
        },
    )
    max_model_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum length of a sequence (including prompt and output). If None, will be derived from the model. default set None",
        },
    )
    max_seq_len_to_capture: Optional[int] = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length covered by CUDA graph",
        },
    )
    vllm_max_num_seqs: int = field(
        default=256,
        metadata={
            "help": "vllm's max_num_seqs, default 256",
        },
    )
    enable_prefix_caching: Optional[bool] = field(
        default=False,
        metadata={"help": "vllm's enable_prefix_caching, default False"},
    )
    num_scheduler_steps: Optional[int] = field(
        default=1,
        metadata={
            "help": "vllm's num_scheduler_steps, default 1",
        },
    )
    disable_custom_all_reduce: Optional[bool] = field(
        default=False,
        metadata={"help": "vllm's disable_custom_all_reduce, default False"},
    )
    max_lora_rank: Optional[int] = field(
        default=16,
        metadata={
            "help": "vllm's max_lora_rank, default 16",
        },
    )
    limit_mm_per_prompt: Optional[int] = field(
        default=1,
        metadata={
            "help": "For each multimodal plugin, limit how many image to allow for each prompt. Expects a comma-separated list of items, e.g.: image=16 allows a maximum of 16 images per prompt. Defaults to 1 for each modality.",
        },
    )
    limit_image_per_prompt: Optional[int] = field(
        default=1,
        metadata={
            "help": "For each multimodal plugin, limit how many image to allow for each prompt. Expects a comma-separated list of items, e.g.: image=16 allows a maximum of 16 images per prompt. Defaults to 1 for each modality.",
        },
    )
    presence_penalty: Optional[float] = field(
        default=0.0,
        metadata={"help": "[Used in vllm mode only] Float that penalizes new tokens based on whether they appear in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens."},
    )
    optimization_level: Optional[int] = field(
        default=1,
        metadata={"help": "The optimization level. These levels trade startup time cost for performance"}
    )
    quantization: Optional[str] = field(
        default=None,
        metadata={"help": "vllm's quantization, default None"},
    )
    ######vLLM config END######
    output_scores: Optional[bool] = field(
        default=False,
        metadata={"help": "get the probabilities of generated tokens"},
    )
    output_audio: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to generate audio output"},
    )
    enable_thinking: bool = field(
        default=True,
        metadata={"help": "Whether or not to enable thinking mode for reasoning models."},
    )
    write_buffer_size: int = field(
        default=20,
        metadata={
            "help": "EIP writer write buffer, default 20",
        },
    )
    def __post_init__(self):
        logger.warning("`image_folder` used as `video_folder` will be deprecated in the future. Use `video_folder` instead if set video.")

        assert self.inputs or self.tables, "Need either a dataset name or a table name."
        assert self.outputs, "Need an output path."

        # TODO:后期下掉limit_image_per_prompt
        self.limit_mm_per_prompt = max(self.limit_mm_per_prompt, self.limit_image_per_prompt)

        # 统一训练和推理的图像列key名称
        if self.image and self.image_column is None:
            self.image_column = self.image

        if not self.mix_multimodel:
            assert (self.image_column is None and self.video is None) \
                or (self.image_column is None and self.audio is None) \
                or (self.video is None and self.audio is None), \
                "either image or video or audio"

        if self.video is not None and self.video_folder is None and self.image_folder is not None:
            self.video_folder = self.image_folder

        self.input_columns = self.input_columns.split(',') if self.input_columns else []
        if self.tables:
            if self.tables.startswith("odps"):
                _table_name = self.tables.split(',')[0]
                if is_external_cluster():
                    column_name_list = get_odps_columns_name_in_tunnel(_table_name)
                else:
                    column_name_list = get_odps_columns_name(_table_name)
            elif self.tables.startswith("ailake"):
                column_name_list = get_ailake_columns_name(self.tables)
            # in odps, input_columns is numbers, so need to int
            self.input_columns = [get_column_index(_tmp_col, column_name_list) for _tmp_col in self.input_columns]
            self.prompt_column = get_column_index(self.prompt_column, column_name_list)
            self.image_column = get_column_index(self.image_column, column_name_list)
            self.video = get_column_index(self.video, column_name_list)
            self.audio = get_column_index(self.audio, column_name_list)
            self.system_column = get_column_index(self.system_column, column_name_list)
            self.response_column = get_column_index(self.response_column, column_name_list)
            self.history_column = get_column_index(self.history_column, column_name_list)
            self.query_column = get_column_index(self.query_column, column_name_list)


            # Redefine the index based on the selected input column
            if self.input_columns:
                self.prompt_column = self.input_columns.index(self.prompt_column)
                self.image_column = self.input_columns.index(self.image_column) if self.image_column else None
                self.video = self.input_columns.index(self.video) if self.video else None
                self.audio = self.input_columns.index(self.audio) if self.audio else None
                self.system_column = self.input_columns.index(self.system_column) if self.system_column else None
                self.response_column = self.input_columns.index(self.response_column) if self.response_column else None
                self.history_column = self.input_columns.index(self.history_column) if self.history_column else None
                self.query_column = self.input_columns.index(self.query_column) if self.query_column else None
            else:
                # The format of each odps table is the same, so pick first one
                input_columns_cnt = len(column_name_list)
                self.input_columns = list(range(input_columns_cnt))

            if self.tables.startswith("ailake"):
                self.ailake_non_multimodel_data_columns = []
                for index, column_index in enumerate(self.input_columns):
                    if index not in (self.image_column, self.video, self.audio):
                        self.ailake_non_multimodel_data_columns.append(index)

        if self.load_from == "openlm":
            self.field_mapping = json.loads(self.field_mapping)
            if isinstance(self.field_mapping, list):
                self.field_mapping = self.field_mapping[0]
            logger.info(f"field_mapping: {self.field_mapping}")


def _verify_args(
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    infer_args: "InferArguments",
):
    if infer_args.tables and (
        infer_args.history_column or infer_args.video or infer_args.audio or infer_args.image_column
    ):
        set_odps_table_reader_mode()

    if infer_args.infer_mode in ('vllm-multi-node', 'vllm-async'):
        logger.info(f"vllm-multi-node or vllm-async set batch_size = 1")
        infer_args.batch_size = 1
        if infer_args.video or infer_args.audio or infer_args.image_column:
            os.environ['VLLM_ENGINE_ITERATION_TIMEOUT_S'] = "600"

    if infer_args.output_scores:
            assert generating_args.num_return_sequences == 1, 'Now output_scores only work with num_return_sequences==1'


def get_generate_args() -> Tuple["ModelArguments", "FinetuningArguments", "GeneratingArguments", "InferArguments"]:
    parser = HfArgumentParser((ModelArguments, FinetuningArguments, GeneratingArguments, InferArguments))
    (
        model_args,
        finetuning_args,
        generating_args,
        infer_args,
    ) = parser.parse_args_into_dataclasses()
    _verify_args(model_args, finetuning_args, generating_args, infer_args)
    return model_args, finetuning_args, generating_args, infer_args
@dataclass
class ChatInferArguments(InferArguments):

    timeout: int = field(default=24 * 60, metadata={"help": "maximum lifespan in minute unit of a chat app. "})

    def __post_init__(self):
        pass
