import inspect
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch
from megatron.core import mpu
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from openlm_hub import repo_download
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, HfArgumentParser
from transformers_turbo.models import AutoModel as AutoTurboModel
from transformers_turbo.models.converter.dist_converter import DistConverter
from transformers_turbo.models.converter.model_converter import ModelConverter
from transformers_turbo.models.converter.post_converter import convert_checkpoint_to_hf
from transformers_turbo.models.converter.template import get_template
from transformers_turbo.training_args import DistributingParallelArguments
from transformers_turbo.utils import get_logger


if TYPE_CHECKING:
    from transformers_turbo.models.converter.template import Template

logger = get_logger(__name__)


@dataclass
class ConvertArguments:
    model_name_or_path: str
    adapter_name_or_path: Optional[str] = field(default=None)
    export_dir: str = field(default="./output")
    bf16: bool = field(default=False)
    fp16: bool = field(default=False)
    convert_model_max_length: Optional[int] = field(default=None)
    low_mem: bool = field(default=True)

    def __post_init__(self):
        self.model_name_or_path = repo_download(
            self.model_name_or_path, ignore_patterns=["*dist_optimizer*", "*/optimizer.pt"]
        )
        if self.adapter_name_or_path is not None:
            self.adapter_name_or_path = repo_download(
                self.adapter_name_or_path, ignore_patterns=["*dist_optimizer*", "*/optimizer.pt"]
            )


def convert_hf_to_turbo(convert_args: ConvertArguments, dist_args: DistributingParallelArguments):
    dist_args.pipeline_model_parallel_size = dist_args.pipeline_model_parallel_size or 1
    dist_args.tensor_model_parallel_size = dist_args.tensor_model_parallel_size or 1
    dist_args.expert_model_parallel_size = dist_args.expert_model_parallel_size or 1
    hf_config = AutoConfig.from_pretrained(convert_args.model_name_or_path, trust_remote_code=True)
    template: "Template" = get_template(hf_config.model_type)
    turbo_config = template.convert_hf_to_turbo_config(
        hf_config,
        **dist_args.get_config_dict()
    )
    template.set_turbo_config_for_ops(turbo_config)
    mpu.set_tensor_model_parallel_world_size(dist_args.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(dist_args.pipeline_model_parallel_size)
    mpu.set_expert_model_parallel_world_size(dist_args.expert_model_parallel_size)
    if dist_args.virtual_pipeline_model_parallel_size is not None:
        mpu.set_virtual_pipeline_model_parallel_world_size(dist_args.virtual_pipeline_model_parallel_size)

    model_converter = ModelConverter(turbo_config=turbo_config, verbose=True)

    for dist_converter in tqdm(
        DistConverter.dist_converter_iter(turbo_config=turbo_config),
        total=dist_args.tensor_model_parallel_size
        * dist_args.pipeline_model_parallel_size
        * dist_args.expert_model_parallel_size,
        desc="Converting",
    ):
        mpu.set_tensor_model_parallel_rank(dist_converter.tensor_model_parallel_rank)
        mpu.set_pipeline_model_parallel_rank(dist_converter.pipeline_model_parallel_rank)
        mpu.set_expert_model_parallel_rank(dist_converter.expert_model_parallel_rank)
        model_parallel_cuda_manual_seed(42)
        turbo_config.use_cpu_initialization = True
        turbo_config.perform_initialization = False
        turbo_model = AutoTurboModel.from_config(config=turbo_config)
        turbo_state_dict = {}
        for i in range(len(turbo_model.models)):
            key = "model"
            dist_converter = DistConverter(
                turbo_config=turbo_config,
                tensor_model_parallel_rank=dist_converter.tensor_model_parallel_rank,
                pipeline_model_parallel_rank=dist_converter.pipeline_model_parallel_rank,
                expert_model_parallel_rank=dist_converter.expert_model_parallel_rank,
                virtual_pipeline_model_parallel_rank=i
            )
            if dist_args.virtual_pipeline_model_parallel_size is not None:
                key = f"model{i}"
                mpu.set_virtual_pipeline_model_parallel_rank(i)
            turbo_state_dict[key] = model_converter.get_turbo_state_dict(
                dist_converter, model_converter.hf_state_dict_iter(convert_args.model_name_or_path, dist_converter)
            )

        missing_keys, unexpected_keys = turbo_model.load_state_dict(turbo_state_dict, strict=False)
        if missing_keys:  # something about fp8 ignored for now
            missing_keys = [key for key in missing_keys if not key.endswith("._extra_state")]
        assert unexpected_keys is None or len(unexpected_keys) == 0, f"unexpected_keys: {unexpected_keys}"
        assert missing_keys is None or len(missing_keys) == 0, f"missing_keys: {missing_keys}"
        logger.info(
            f"Saving model tp_rank: {dist_converter.tensor_model_parallel_rank} "
            f"pp_rank: {dist_converter.pipeline_model_parallel_rank} "
            f"ep_rank: {dist_converter.expert_model_parallel_rank} to {convert_args.export_dir}"
        )
        turbo_config.use_cpu_initialization = False
        turbo_model.save_pretrained(convert_args.export_dir)
        del turbo_model
        template.release()

    tokenizer = AutoTokenizer.from_pretrained(convert_args.model_name_or_path, trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(convert_args.model_name_or_path, trust_remote_code=True)
    except Exception as e:
        logger.info(f"Processor was not found: {e}.")
        processor = None
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    if processor is not None:
        setattr(processor, "tokenizer", tokenizer)
    else:
        processor = tokenizer
    processor.save_pretrained(convert_args.export_dir)


def add_turbo_state_dicts_to_hf(state_dicts, dist_reverter: "DistConverter", template: "Template", hf_state_dict):
    tp_rank, pp_rank, ep_rank, vp_rank = (
        dist_reverter.tensor_model_parallel_rank,
        dist_reverter.pipeline_model_parallel_rank,
        dist_reverter.expert_model_parallel_rank,
        dist_reverter.virtual_pipeline_model_parallel_rank,
    )
    for turbo_name in state_dicts[0].keys():
        if turbo_name.endswith("._extra_state"):
            continue
        weights = [state_dict[turbo_name] if turbo_name in state_dict else None for state_dict in state_dicts]
        turbo_named_weights = dist_reverter(turbo_name, weights)
        converted_state_dict = {}
        if turbo_named_weights is not None:
            for turbo_name, turbo_weight in turbo_named_weights.items():
                converted = template.add_turbo_weight(turbo_name, turbo_weight)
                assert (
                    len(set(converted_state_dict.keys()).intersection(converted.keys())) == 0
                ), f"converted_state_dict: {converted_state_dict.keys()} converted: {converted.keys()}"
                converted_state_dict.update(converted)
        if converted_state_dict is not None and len(converted_state_dict) > 0:
            for hf_name, hf_weight in converted_state_dict.items():
                if hf_name in hf_state_dict:
                    if not hf_weight.equal(hf_state_dict[hf_name]):
                        raise ValueError(
                            f"weight of hf_name:{hf_name} turbo_name:{turbo_name} in "
                            f"tp_rank, pp_rank, ep_rank, vp_rank:{tp_rank} {pp_rank} {ep_rank} {vp_rank} "
                            f"diff max:{torch.abs(hf_weight - hf_state_dict[hf_name]).max()}"
                        )
                hf_state_dict[hf_name] = hf_weight
                logger.info(f"turbo_name: {turbo_name} -> hf_name: {hf_name}")
        else:
            logger.info(f"turbo_name: {turbo_name} added but not converted")


def convert_turbo_to_hf(convert_args: ConvertArguments, dist_args: DistributingParallelArguments):
    torch_dtype = None
    if convert_args.bf16:
        torch_dtype = torch.bfloat16
    elif convert_args.fp16:
        torch_dtype = torch.float16
    
    kwargs = {}
    if "low_mem" in inspect.signature(convert_checkpoint_to_hf).parameters:
        kwargs["low_mem"] = convert_args.low_mem
    convert_checkpoint_to_hf(
        convert_args.model_name_or_path,
        convert_args.export_dir,
        adapter_name_or_path=convert_args.adapter_name_or_path,
        torch_dtype=torch_dtype,
        **kwargs,
    )
    config = AutoConfig.from_pretrained(convert_args.export_dir, trust_remote_code=True)
    if convert_args.convert_model_max_length is not None:
        config.model_max_length = convert_args.convert_model_max_length
        config.max_position_embeddings = convert_args.convert_model_max_length
    config.save_pretrained(convert_args.export_dir)
    logger.info(f"\n ==============HF config===========: \n {config}")


def main():
    convert_args, dist_args = HfArgumentParser(
        [ConvertArguments, DistributingParallelArguments]
    ).parse_args_into_dataclasses()

    turbo_config_path = os.path.join(convert_args.model_name_or_path, "turbo_config.json")
    if convert_args.adapter_name_or_path is not None:
        turbo_config_path = os.path.join(convert_args.adapter_name_or_path, "turbo_config.json")
    from_turbo = os.path.exists(turbo_config_path)

    if not from_turbo:
        convert_hf_to_turbo(convert_args, dist_args)
    else:
        convert_turbo_to_hf(convert_args, dist_args)

    from openlm_hub import get_mos_version_uri_in_mdl_task, push_to_mos
    from openlm_hub.utils import gen_mos_uri, parse_mos_uri
    try:
        mos_uri = get_mos_version_uri_in_mdl_task()
        project, model, version, ckpt_id = parse_mos_uri(mos_uri)
        if not ckpt_id:
            ckpt_id = "converted"
        else:
            mos_uri = gen_mos_uri(project, model, version)
        push_to_mos(convert_args.export_dir, checkpoint_id=ckpt_id, mos_version_uri=mos_uri)
    except Exception as e:
        logger.warning(f"Cannot push to mos due to {e}")


if __name__ == "__main__":
    main()
