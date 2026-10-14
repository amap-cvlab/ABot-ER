from typing import TYPE_CHECKING, Optional, Union

from transformers import Trainer

from ...extras.callbacks import PushToMosCallback, SaveProcessorCallback
from ...extras.packages import is_transformers_version_greater_than
from ..mixin import OpenlmHubUtils


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin


class CustomTrainer(Trainer):
    def __init__(self, data_args, processor: Optional["ProcessorMixin"], **kwargs) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        self.openlm_hub_utils = OpenlmHubUtils()

        if processor is not None:
            self.model_accepts_loss_kwargs = False

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if self.openlm_hub_utils.mos_uri is not None:
            self.add_callback(PushToMosCallback(self, self.args, self.model.config, data_args.cutoff_len))
