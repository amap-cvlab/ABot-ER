import gc
from enum import Enum, unique
from typing import TYPE_CHECKING, Dict, Sequence, Set, Union

import torch
from datasets import DatasetDict

from ..extras.logging import get_logger


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from llmtuner.hparams import DataArguments


logger = get_logger(__name__)


SLOTS = Sequence[Union[str, Set[str], Dict[str, str]]]


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"

    def __str__(self):
        return self.value


def split_dataset(
    dataset: Union["Dataset", "IterableDataset"], data_args: "DataArguments", seed: int
) -> "DatasetDict":
    r"""
    Splits the dataset and returns a dataset dict containing train set and validation set.

    Supports both map dataset and iterable dataset.
    """
    if data_args.streaming:
        dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)
        val_set = dataset.take(int(data_args.val_size))
        train_set = dataset.skip(int(data_args.val_size))
        return DatasetDict({"train": train_set, "validation": val_set})
    else:
        val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
        dataset = dataset.train_test_split(test_size=val_size, seed=seed)
        return DatasetDict({"train": dataset["train"], "validation": dataset["test"]})

def trans_func(x):return x

def get_sample_data(dataset, use_subprocess):
    if use_subprocess:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, collate_fn=trans_func)
        sample_data = next(iter(dataloader))[0]
        del dataloader
        gc.collect()
        return sample_data
    else:
        return next(iter(dataset))

