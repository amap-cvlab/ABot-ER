import contextlib
import gc
import os
import types
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
from .optional_mos import (
    get_last_ckpt_by_uri,
    get_mos_version_uri_in_mdl_task,
    get_version_info,
    repo_download,
)
from .optional_mos import MosCkptNotFoundError, MosVersionNotFoundError, OpenlmHubClientError
from transformers import InfNanRemoveLogitsProcessor, LogitsProcessorList
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import (
    is_torch_bf16_gpu_available,
    is_torch_cuda_available,
    is_torch_mps_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)

from ..extras.constants import NEBULA_TRANSFORMERS_OUTPUT_DIR
from .logging import get_logger


_is_fp16_available = is_torch_npu_available() or is_torch_cuda_available()
try:
    _is_bf16_available = is_torch_bf16_gpu_available()
except Exception:
    _is_bf16_available = False


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments

    from llmtuner.hparams import ModelArguments

    from ..generate.infer_args import InferArguments
    from ..hparams.data_args import DataArguments
    from ..hparams.finetuning_args import FinetuningArguments


logger = get_logger(__name__)


class AverageMeter:
    r"""
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by 2
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def get_current_device() -> torch.device:
    r"""
    Gets the current available device.
    """
    if is_torch_xpu_available():
        device = "xpu:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif is_torch_npu_available():
        device = "npu:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif is_torch_mps_available():
        device = "mps:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif is_torch_cuda_available():
        device = "cuda:{}".format(os.environ.get("LOCAL_RANK", "0"))
    else:
        device = "cpu"

    return torch.device(device)


def get_device_count() -> int:
    return torch.cuda.device_count()


def get_logits_processor() -> "LogitsProcessorList":
    r"""
    Gets logits processor that removes NaN and Inf logits.
    """
    logits_processor = LogitsProcessorList()
    logits_processor.append(InfNanRemoveLogitsProcessor())
    return logits_processor


def infer_optim_dtype(model_dtype: torch.dtype) -> torch.dtype:
    r"""
    Infers the optimal dtype according to the model_dtype and device compatibility.
    """
    if _is_bf16_available and model_dtype == torch.bfloat16:
        return torch.bfloat16
    elif _is_fp16_available:
        return torch.float16
    else:
        return torch.float32


def torch_gc() -> None:
    r"""
    Collects GPU memory.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def try_download_model_from_ms(model_args: "ModelArguments") -> None:
    if not use_modelscope() or os.path.exists(model_args.model_name_or_path):
        return

    try:
        from modelscope import snapshot_download

        revision = "master" if model_args.model_revision == "main" else model_args.model_revision
        model_args.model_name_or_path = snapshot_download(
            model_args.model_name_or_path, revision=revision, cache_dir=model_args.cache_dir
        )
    except ImportError:
        raise ImportError("Please install modelscope via `pip install modelscope -U`")


def use_modelscope() -> bool:
    return bool(int(os.environ.get("USE_MODELSCOPE_HUB", "0")))


def down_model(uri, used_for=None):
    return repo_download(
        uri,
        ignore_patterns=["global_step*/*_states.pt", "*iter_0000001/dist_optimizer/*"],
        repo_meta={"usage": used_for},
        use_subprocess=True,
    )


def get_resume_checkpoint(training_args: "Seq2SeqTrainingArguments", task_mos_uri: Optional[str] = None) -> Optional[str]:
    resume_from_checkpoint = training_args.resume_from_checkpoint
    if isinstance(resume_from_checkpoint, str) and os.path.exists(resume_from_checkpoint):
        return resume_from_checkpoint

    if isinstance(resume_from_checkpoint, str):
        try:
            _ = get_version_info(resume_from_checkpoint)
        except MosVersionNotFoundError as e:
            raise ValueError(f"resume_from_checkpoint: {resume_from_checkpoint} is not a local_path and not found in MOS: {e}")
        mos_uri = resume_from_checkpoint
        local_state_path = repo_download(mos_uri, use_subprocess=True)
        logger.info(f"{mos_uri} finished downloading to {local_state_path}")
        return local_state_path

    if task_mos_uri is None:
        if (
            (resume_from_checkpoint is None or resume_from_checkpoint is True)
            and not getattr(training_args, "overwrite_output_dir", False)
            and os.path.exists(training_args.output_dir)
        ):
            resume_from_checkpoint = get_last_checkpoint(training_args.output_dir)
        return resume_from_checkpoint

    try:
        last_ckpt = get_last_ckpt_by_uri(task_mos_uri)
    except MosCkptNotFoundError:
        last_ckpt = None

    if last_ckpt is None:
        logger.warning(f"Latest checkpoint not exist for mos uri: {task_mos_uri}. New training will be started.")
        return None

    mos_uri = last_ckpt.uri if last_ckpt is not None else resume_from_checkpoint
    if getattr(training_args, "overwrite_output_dir", False):
        logger.warning(f"MOS not support `overwrite_output_dir`, will resume from {mos_uri}.")

    local_state_path = repo_download(mos_uri, use_subprocess=True)
    logger.info(f"{mos_uri} finished downloading to {local_state_path}")
    return local_state_path


def process_mos_info_for_training_args(training_args: "Seq2SeqTrainingArguments"):
    if not training_args.do_train:
        return
    fresh_output = os.environ.get("TP_MOS_FRESH_OUTPUT") == "1"
    task_mos_uri = os.environ.get("TP_MOS_OUTPUT_URI")
    if task_mos_uri:
        logger.info(f"Use explicit task-planning MOS output URI: {task_mos_uri}")
    else:
        try:
            task_mos_uri = get_mos_version_uri_in_mdl_task()
        except OpenlmHubClientError:
            task_mos_uri = None

    if task_mos_uri is not None:
        training_args.output_dir = NEBULA_TRANSFORMERS_OUTPUT_DIR
        logger.info(f"Set output_dir to {NEBULA_TRANSFORMERS_OUTPUT_DIR} since checkpoint will be saved to MOS: {task_mos_uri}.")

    if fresh_output:
        if task_mos_uri is None:
            raise ValueError("TP_MOS_FRESH_OUTPUT=1 requires an explicit TP_MOS_OUTPUT_URI")
        if training_args.resume_from_checkpoint not in (None, False):
            raise ValueError(
                "TP_MOS_FRESH_OUTPUT=1 is incompatible with an explicit resume_from_checkpoint"
            )
        training_args.resume_from_checkpoint = None
        logger.warning(
            f"[mos-output] fresh=true; skip auto-resume lookup for new URI: {task_mos_uri}"
        )
    else:
        training_args.resume_from_checkpoint = get_resume_checkpoint(
            training_args, task_mos_uri=task_mos_uri
        )

    if training_args.resume_from_checkpoint is None:
        logger.warning("No valid checkpoint found in output directory or Mos uri. New training will be started.")
    else:
        logger.warning(
            f"Resume training from checkpoint: {training_args.resume_from_checkpoint}. "
            "Make sure the checkpoint is compatible with current training arguments!"
        )

def patch_training_args(training_args: "Seq2SeqTrainingArguments"):
    @contextlib.contextmanager
    def _main_process_first(self, local=True, desc="work"):
        import torch.distributed as dist
        from transformers.utils import is_sagemaker_mp_enabled, is_torch_available, is_torch_xla_available

        if is_torch_xla_available():
            import torch_xla.core.xla_model as xm
        if is_sagemaker_mp_enabled():
            import smdistributed.modelparallel.torch as smp

        if is_torch_available() and self.world_size > 1:
            main_process_desc = "main local process" if local else "main process"
            if self.distributed_state is not None:
                local_rank = int(os.getenv("LOCAL_PROCESS_RANK", os.getenv("LOCAL_RANK", "0")))
                is_main_process = local_rank == 0 if local else self.distributed_state.is_main_process
            elif is_sagemaker_mp_enabled():
                is_main_process = smp.rank() == 0
            logger.warning(f"rank{dist.get_rank()} {is_main_process}: {main_process_desc} performing {desc}")
            try:
                if not is_main_process:
                    # tell all replicas to wait
                    logger.debug(f"{self.process_index}: waiting for the {main_process_desc} to perform {desc}")

                    if is_torch_xla_available():
                        xm.rendezvous(desc)
                    else:
                        dist.barrier()
                yield
            finally:
                if is_main_process:
                    # the wait is over
                    logger.debug(f"{self.process_index}: {main_process_desc} completed {desc}, releasing all replicas")
                    if is_torch_xla_available():
                        xm.rendezvous(desc)
                    else:
                        dist.barrier()
        else:
            yield

    training_args.main_process_first = types.MethodType(_main_process_first, training_args)


def patch_model_save_pretrained():
    """
    Patch PreTrainedModel.save_pretrained to use 5GB max_shard_size by default.
    This overrides the default 50GB in transformers>=5.0.0. 
    """
    from ..extras.packages import is_transformers_version_greater_than

    # Only patch when transformers version is greater than 5.0.0
    if not is_transformers_version_greater_than("5.0.0"):
        return

    from transformers import PreTrainedModel

    original_save_pretrained = PreTrainedModel.save_pretrained

    def patched_save_pretrained(self, *args, **kwargs):
        if "max_shard_size" not in kwargs or kwargs["max_shard_size"] is None:
            from ..extras.constants import MAX_SHARD_SIZE
            kwargs["max_shard_size"] = MAX_SHARD_SIZE
        return original_save_pretrained(self, *args, **kwargs)

    PreTrainedModel.save_pretrained = patched_save_pretrained
    logger.info("Patched PreTrainedModel.save_pretrained to use 5GB max_shard_size by default")


def get_dataset_type(path: str):
    if path.startswith("/data/oss_bucket"):
        type = "oss"
    elif path.startswith("odps://"):
        type = "odps"
    elif os.path.isfile(path):
        type = "local_file"
    else:
        type = "unknown"
    return type

def log_meta_to_openlm(
    data_args: "DataArguments" = None,
    finetuning_args: "FinetuningArguments" = None,
    infer_args: "InferArguments" = None,
):
    if os.environ.get("ABOT_ENABLE_MOS") != "1":
        return
    try:
        from openlm_hub import log_meta, sync_meta, DatasetMeta, DataLineageMeta, ModelArtifactMeta

        # only for parse single dataset
        # TODO for parse multi dataset
        # data_args for train job
        if data_args:
            dataset_metas = []
            if data_args.file_name or data_args.dataset_name:
                if data_args.dataset_name:
                    dataset_metas.append(
                        DatasetMeta(
                            name=data_args.dataset_name,
                            path=data_args.dataset_name,
                            usage="train",
                            info={"subset": data_args.subset},
                            type="openlm",
                        )
                    )
                else:
                    dataset_metas.append(
                        DatasetMeta(
                            name=data_args.file_name,
                            path=data_args.file_name,
                            usage="train",
                            type=get_dataset_type(data_args.file_name),
                        )
                    )
            if data_args.eval_dataset_name or data_args.eval_file_name:
                if data_args.eval_dataset_name:
                    dataset_metas.append(
                        DatasetMeta(
                            name=data_args.eval_dataset_name,
                            path=data_args.eval_dataset_name,
                            usage="eval",
                            type="openlm",
                        )
                    )
                else:
                    dataset_metas.append(
                        DatasetMeta(
                            name=data_args.eval_file_name,
                            path=data_args.eval_file_name,
                            usage="eval",
                            type=get_dataset_type(data_args.eval_file_name),
                        )
                    )
            log_meta(
                meta=DataLineageMeta(
                    datasets=dataset_metas,
                    model_artifact=ModelArtifactMeta(
                        type=finetuning_args.finetuning_type, stage=finetuning_args.stage
                    ),
                    extra_metas={
                        "use_turbo": finetuning_args.use_turbo,
                        "task_mode": "train",
                    },
                )
            )
        # infer_args for offline infer
        if infer_args:
            input_data = []
            output_data = []
            if infer_args.inputs:
                if infer_args.tables:
                    input_data.append(
                        DatasetMeta(
                            name=infer_args.inputs,
                            path=infer_args.inputs,
                            usage="input",
                            info={
                                "prompt_column": infer_args.prompt_column,
                                "image_column": infer_args.image_column,
                                "system_column": infer_args.system_column,
                            },
                            type="odps",
                        )
                    )
                else:
                    if infer_args.load_from == "file":
                        input_data.append(
                            DatasetMeta(
                                name=infer_args.inputs,
                                path=infer_args.inputs,
                                usage="input",
                                info={
                                    "prompt_column": infer_args.prompt_column,
                                    "image_column": infer_args.image_column,
                                    "system_column": infer_args.system_column,
                                },
                                type=get_dataset_type(infer_args.inputs),
                            )
                        )
                    elif infer_args.load_from == "openlm":
                        input_data.append(
                            DatasetMeta(
                                name=infer_args.inputs,
                                path=infer_args.inputs,
                                usage="input",
                                info={
                                    "prompt_column": infer_args.prompt_column,
                                    "image_column": infer_args.image_column,
                                    "system_column": infer_args.system_column,
                                },
                                type="openlm",
                            )
                        )
            if infer_args.outputs:
                if infer_args.tables:
                    output_data.append(
                        DatasetMeta(
                            name=infer_args.outputs,
                            path=infer_args.outputs,
                            usage="output",
                            type="odps",
                        )
                    )
                elif infer_args.load_from == 'file':
                    output_data.append(
                        DatasetMeta(
                            name=infer_args.outputs,
                            path=infer_args.outputs,
                            usage="output",
                            type=get_dataset_type(infer_args.outputs),
                        )
                    )
                elif infer_args.load_from == "openlm":
                    output_data.append(
                        DatasetMeta(
                            name=infer_args.outputs,
                            path=infer_args.outputs,
                            usage="output",
                            type="openlm",
                        )
                    )
            log_meta(
                meta=DataLineageMeta(
                    datasets=input_data,
                    extra_metas={
                        "use_turbo": finetuning_args.use_turbo,
                        "task_mode": "offline_infer",
                        "infer_mode": infer_args.infer_mode
                    },
                    output_datasets=output_data
                )
            )
        sync_meta()
    except Exception:
        logger.exception("log openlm-hub lineage failed,ignore this error.")


def is_float_tensor(tensor):
    if not isinstance(tensor, torch.Tensor):
        return False
    return tensor.dtype in (torch.float, torch.float16, torch.bfloat16)


def recursive_converter(converter, value):
    if isinstance(value, list):
        new_value = []
        for v in value:
            new_value += [recursive_converter(converter, v)]
        return new_value
    else:
        return converter(value)


def to_device(data: Dict, device: torch.device):
    def cast_tensor(v):
        if isinstance(v, torch.Tensor):
            return v.to(device=device)
        else:
            return v

    new_data = {}
    for k, v in data.items():
        new_data[k] = recursive_converter(cast_tensor, v)
    return new_data


def set_odps_table_reader_mode():
    logger.info("set odps table reader to TunnelReader")
    import common_io
    common_io.table.TableReader = common_io.table_tunnel.TableTunnelReader
