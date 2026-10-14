from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional

from typing_extensions import Self

from ..extras.logging import get_logger


logger = get_logger(__name__)


@dataclass
class ProcessorArguments:
    r"""
    Arguments pertaining to the image processor.
    """

    image_resolution: int = field(
        default=8192 * 8192,
        metadata={"help": "The maximum number of pixels of image inputs."},
    )
    video_resolution: int = field(
        default=2048 * 2048,
        metadata={"help": "The maximum number of pixels of video inputs."},
    )
    min_resolution: int = field(
        default=4 * 28 * 28,
        metadata={"help": "The minimum number of pixels of image/video inputs."},
    )
    video_fps: float = field(
        default=2.0,
        metadata={"help": "The frames to sample per second for video inputs."},
    )
    video_maxlen: int = field(
        default=128,
        metadata={"help": "The maximum number of sampled frames for video inputs."},
    )
    # TODO:video_longest_edge and video_shortest_edge are exclusively for Qwen3-VL. 
    # A*B*C, BC do not change, A is visual tokens cnts
    video_longest_edge: int = field(
        default=16384*32*32,
        metadata={"help": "video_longest_edge represents the maximum total number of pixels across all frames in a video — for a video of shape T×H×W, the product T×H×W must not exceed size video_longest_edge "},
    )
    video_shortest_edge: int = field(
        default=256*32*32,
        metadata={"help": "video_shortest_edge sets the minimum total pixel budget for the video."},
    )
    video_resize_chunk_frames: int = field(
        default=64,
        metadata={"help": "The frames of the chunk to resize the video."}
    )


@dataclass
class ModelArguments(ProcessorArguments):
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    teacher_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the teacher model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the adapter weight or identifier from huggingface.co/models."},
    )
    teacher_adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the teacher adapter weight or identifier from huggingface.co/models."},
    )
    adapter_folder: Optional[str] = field(
        default=None,
        metadata={"help": "The folder containing the adapter weights to load."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn."},
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to use one of the fast tokenizer (backed by the tokenizers library)."},
    )
    resize_vocab: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to resize the tokenizer vocab and the embedding layers."},
    )
    split_special_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not the special tokens should be split during the tokenization process."},
    )
    model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    quantization_method: Literal["bitsandbytes", "hqq", "eetq"] = field(
        default="bitsandbytes",
        metadata={"help": "Quantization method to use for on-the-fly quantization."},
    )
    quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the model."},
    )
    quantization_type: Optional[Literal["fp4", "nf4"]] = field(
        default="nf4",
        metadata={"help": "Quantization data type to use in int4 training."},
    )
    double_quantization: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to use double quantization in int4 training."},
    )
    quantization_device_map: Optional[Literal["auto"]] = field(
        default=None,
        metadata={"help": "Device map used to infer the 4-bit quantized model, needs bitsandbytes>=0.43.0."},
    )
    rope_scaling: Optional[Literal["linear", "dynamic", "yarn"]] = field(
        default=None,
        metadata={"help": "Which scaling strategy should be adopted for the RoPE embeddings."},
    )
    flash_attn: Literal["auto", "disabled", "sdpa", "fa2", "True", "False"] = field(
        default="fa2",
        metadata={"help": "Enable FlashAttention for faster training and inference."},
    )
    mixture_of_depths: Optional[Literal["convert", "load"]] = field(
        default=None,
        metadata={"help": "Convert the model to mixture-of-depths (MoD) or load the MoD model."},
    )
    use_unsloth: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to use unsloth's optimization for the LoRA training."},
    )
    enable_liger_kernel: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable liger kernel for faster training."},
    )
    moe_aux_loss_coef: Optional[float] = field(
        default=None,
        metadata={"help": "Coefficient of the auxiliary router loss in mixture-of-experts model."},
    )
    disable_gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to disable gradient checkpointing."},
    )
    upcast_layernorm: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to upcast the layernorm weights in fp32."},
    )
    upcast_lmhead_output: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to upcast the output of lm_head in fp32."},
    )
    hf_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Hugging Face Hub."},
    )
    ms_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with ModelScope Hub."},
    )
    export_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory to save the exported model."},
    )
    export_size: Optional[int] = field(
        default=1,
        metadata={"help": "The file shard size (in GB) of the exported model."},
    )
    export_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the exported model."},
    )
    export_quantization_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the dataset or dataset name to use in quantizing the exported model."},
    )
    export_quantization_nsamples: Optional[int] = field(
        default=128,
        metadata={"help": "The number of samples used for quantization."},
    )
    export_quantization_maxlen: Optional[int] = field(
        default=1024,
        metadata={"help": "The maximum length of the model inputs used for quantization."},
    )
    export_legacy_format: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to save the `.bin` files instead of `.safetensors`."},
    )
    export_hub_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the repository if push the model to the Hugging Face hub."},
    )
    print_param_status: Optional[bool] = field(
        default=False,
        metadata={"help": "For debugging purposes, print the status of the parameters in the model."},
    )
    device_map: Optional[str] = field(
        default=None,
        metadata={"help": "transformer's from_pretrained device map"}
    )
    image_processor: Optional[bool] = field(
        default=False,
        metadata={"help": "Deprecated!!! No need to config, will auto detect from ckpt."}
    )
    audio_processor: Optional[bool] = field(
        default=False,
        metadata={"help": "Deprecated!!! No need to config, will auto detect from ckpt."}
    )
    add_tokens_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the JSON file containing the tokens to add."}
    )
    enable_dhta: bool = field(
        default=False,
        metadata={"help": "Enable deterministic hierarchical task adaptation for Qwen3.5."},
    )
    dhta_qa_token: str = field(
        default="[2D_QA]",
        metadata={"help": "Atomic routing token used by the 2D-QA branch."},
    )
    dhta_task_plan_token: str = field(
        default="[TASK_PLAN]",
        metadata={"help": "Atomic routing token used by the Task Plan branch."},
    )
    dhta_layers: str = field(
        default="19,23,27,31",
        metadata={"help": "Comma-separated Qwen3.5 full-attention decoder layers receiving DHTA residuals."},
    )
    dhta_rank: int = field(
        default=16,
        metadata={"help": "Rank of each task-specific DHTA residual branch."},
    )
    dhta_alpha: float = field(
        default=16.0,
        metadata={"help": "Scaling numerator for DHTA residuals (scale = alpha / rank)."},
    )
    dhta_require_route_token: bool = field(
        default=True,
        metadata={"help": "Fail closed when a training or prefill sample lacks exactly one DHTA route token."},
    )
    sequence_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Number of GPUs to process one data sequence. Values greater than 1 means enabling sequence parallelism."
        }
    )
    sequence_parallel_mode: Literal["zigzag-ring", "ulysses"] = field(
        default="ulysses",
        metadata={"help": "Specific mode of sequence parallel implementation."},
    )

    def __post_init__(self):
        self.compute_dtype = None
        self.model_max_length = None
        self.block_diag_attn: bool = False

        if self.model_name_or_path is None:
            raise ValueError("Please provide `model_name_or_path`.")

        if self.split_special_tokens and self.use_fast_tokenizer:
            raise ValueError("`split_special_tokens` is only supported for slow tokenizers.")

        if self.adapter_name_or_path is not None:  # support merging multiple lora weights
            self.adapter_name_or_path = [path.strip() for path in self.adapter_name_or_path.split(",")]

        assert self.quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."
        assert self.export_quantization_bit in [None, 8, 4, 3, 2], "We only accept 2/3/4/8-bit quantization."

        if self.export_quantization_bit is not None and self.export_quantization_dataset is None:
            raise ValueError("Quantization dataset is necessary for exporting.")

        if self.flash_attn == "True":
            self.flash_attn = "fa2"
        elif self.flash_attn == "False":
            self.flash_attn = "auto"

        if self.image_processor or self.audio_processor:
            logger.warning("image_processor or audio_processor is Deprecated, don't need to set it.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def copyfrom(cls, old_arg: Self, **kwargs) -> Self:
        arg_dict = old_arg.to_dict()
        arg_dict.update(**kwargs)
        new_arg = cls(**arg_dict)
        new_arg.compute_dtype = old_arg.compute_dtype
        new_arg.device_map = old_arg.device_map
        new_arg.model_max_length = old_arg.model_max_length
        new_arg.block_diag_attn = old_arg.block_diag_attn
        return new_arg
