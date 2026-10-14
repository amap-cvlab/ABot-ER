import abc

from typing import List, Optional, Tuple, Type, TypeVar, Union
from loguru import logger

from embodied_eval.utils import (
    simple_parse_args_string
)

T = TypeVar("T", bound="BaseAPIModel")

AVAILABLE_MODELS = {
    "internvl3": "InternVL3",
    "iflybot_vlm": "IFlyBotVLM",
    "iflybotvlm": "embodied_eval.models.iflybot_vlm.IFlyBotVLM",
    "internvl2_5": "InternVL2_5",
    "llava_onevision": "Llava_OneVision",
    "qwen2_vl": "Qwen2_VL",
    "qwen2_5_vl": "Qwen2_5_VL",
    "qwen3_vl": "Qwen3_VL",
    "qwen3_5": "Qwen3_5",
    "aboter": "ABotER",
    "rxbrain": "RxBrain",
    "vila": "VILA",
    "openai_compatible": "OpenAICompatible",
    "openai_async_compatible": "OpenAIAsyncCompatible",
    "robobrain": "RoboBrain",
    "robobrain2": "RoboBrain2",
    "magma": "Magma",
    "vebrain": "VeBrain",
    "cosmos_reason1": "CosmosReason1",
    "cosmos3": "Cosmos3Edge",
    "spacer": "SpaceR",
    "roboannotatorx": "RoboAnnotatorX",
    "embodiedgpt": "EmbodiedGPT",
    "embodied_r1": "EmbodiedR1",
    "robopoint": "RoboPoint",
    "spatialvlm": "SpatialVLM",
    "wall_oss": "Wall_OSS",
    "pelican_vl": "PelicanVL",
    "nuoyin": "Nuoyin",
    "internvl3_5": "InternVL3_5",
    "mimo_embodied": "MimoEmbodied",
    "embodied_brain": "EmbodiedBrain",
    "cambrian": "Cambrian",
    "step3_vl": "Step3_VL",
    "rynnbrain": "RynnBrain",
    "wall_brain_preview": "WallBrainPreview",
    "thinker_vl": "Thinker_VL",
    "embodied_vlm": "EmbodiedVLM",
    "embodiedvlm": "embodied_eval.models.embodied_vlm.EmbodiedVLM"
}

class BaseAPIModel(abc.ABC):
    def __init__(self) -> None:
        # set rank and world size to a single process, by default.
        self._rank = 0
        self._world_size = 1
        self.task_dict = {}

    @property
    def rank(self):
        # used in the case of parallelism. Hardcoded to
        # ensure no errors arise using API models which do
        # not support multi-device parallelism nor expect it.
        return self._rank

    @property
    def world_size(self):
        # used in the case of parallelism. Hardcoded to
        # ensure no errors arise using API models which do
        # not support multi-device parallelism nor expect it.
        return self._world_size
    
    @abc.abstractmethod
    def respond(self, context: str, visuals: Union[List[str], List[Tuple[str, int]]], **gen_kwargs) -> str:
        """
        Generate a response based on the context and visuals.

        Parameters:
        - context: The input text context.
        - visuals: List of visual inputs, either as file paths or tuples of (file_path, index).
        - gen_kwargs: Additional generation parameters.

        Returns:
        - Generated response string.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    @classmethod
    def create_from_arg_string(cls: Type[T], arg_string: str, additional_config: Optional[dict] = None) -> T:
        """
        Creates an instance of the LMM class using the given argument string and additional config.

        Parameters:
        - arg_string: A string containing arguments in the format key1=value1,key2=value2.
        - additional_config: Optional dictionary containing additional configuration parameters.

        Returns:
        - Instance of the LMM class.
        """
        additional_config = {} if additional_config is None else additional_config
        args = simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)
    

def get_model(model_name):
    """
    model_name --> model_name.py --> model_class
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not found in available models.")

    model_class = AVAILABLE_MODELS[model_name]
    if "." not in model_class:
        model_class = f"embodied_eval.models.{model_name}.{model_class}"

    try:
        model_module, model_class = model_class.rsplit(".", 1)
        module = __import__(model_module, fromlist=[model_class])
        return getattr(module, model_class) # module.model_class
    except Exception as e:
        logger.error(f"Failed to import {model_class} from {model_name}: {e}")
        raise
