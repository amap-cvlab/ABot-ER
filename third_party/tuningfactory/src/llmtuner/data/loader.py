import inspect
import ast
import os
import sys
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Sequence, Union

from datasets import (
    DatasetDict,
    Features,
    IterableDataset,
    Value,
    concatenate_datasets,
    interleave_datasets,
    load_dataset,
    load_from_disk,
)
# 别名导入：HF datasets.Sequence 与 typing.Sequence 同名，
# 直接导入会 shadow 模块顶部 `from typing import ... Sequence`，
# 让本文件 line 185 的类型注解 `Sequence["DatasetAttr"]` 走到 HF Sequence
# （未实现 `__class_getitem__`），抛 `TypeError: 'type' object is not subscriptable`。
from datasets import Sequence as HFSequence

from ..extras.constants import FILEEXT2TYPE
from ..extras.file_system import get_storage_options, has_tokenized_data
from ..extras.logging import get_logger
from .aligner import align_dataset
from .fixers import build_pipeline as _build_fixer_pipeline, make_map_fn as _make_fixer_map_fn
from .parser import get_dataset_list
from .preprocess import get_dataset_processor
from .template import get_template_and_fix_tokenizer
from .utils import get_sample_data, split_dataset


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments
    from transformers.tokenization_utils import PreTrainedTokenizer

    from ..hparams import DataArguments, ModelArguments
    from .parser import DatasetAttr
    from .template import Template


logger = get_logger(__name__)


def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    template: "Template",
):
    logger.info("Loading dataset {}...".format(dataset_attr))

    data_path, data_name, data_dir, data_files = None, None, None, None
    if dataset_attr.load_from in ["hf_hub", "ms_hub"]:
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "script":
        data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "file":
        data_files = []
        local_path: str = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            # Keep sharded physical mixtures reproducible across hosts and
            # filesystems whose readdir order differs.
            for file_name in sorted(os.listdir(local_path)):
                data_files.append(os.path.join(local_path, file_name))
                if data_path is None:
                    data_path = FILEEXT2TYPE.get(file_name.split(".")[-1], None)
                elif data_path != FILEEXT2TYPE.get(file_name.split(".")[-1], None):
                    raise ValueError("File types should be identical.")
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
            data_path = FILEEXT2TYPE.get(local_path.split(".")[-1], None)
        elif os.path.isfile(dataset_attr.dataset_name):  # is file
            data_files.append(dataset_attr.dataset_name)
            data_path = FILEEXT2TYPE.get(dataset_attr.dataset_name.split(".")[-1], None)
        elif dataset_attr.dataset_name.startswith("odps://"):
            data_path = "odps"
            data_files += dataset_attr.dataset_name.split(",")
        elif dataset_attr.dataset_name.startswith("ailake://"):
            data_path = "ailake"
            data_files += dataset_attr.dataset_name.split(",")
        else:
            raise ValueError("File not found.")

        if data_path is None:
            raise ValueError("File extension must be txt, csv, json or jsonl.")

    else:
        raise NotImplementedError

    if "trust_remote_code" in inspect.signature(load_dataset).parameters:  # for datasets==2.16.0
        kwargs = {"trust_remote_code": True}
    else:
        kwargs = {}

    if not data_args.streaming:
        kwargs["num_proc"] = data_args.preprocessing_num_workers

    if data_path in ["ailake", "odps"]:
        kwargs['num_workers'] = training_args.dataloader_num_workers if data_args.streaming else data_args.preprocessing_num_workers
        logger.info(f"Loading {data_path} dataset, num_workers is {kwargs['num_workers']}")

    dataset = load_dataset(
        path=data_path,
        name=data_name,
        data_dir=data_dir,
        data_files=data_files,
        split=data_args.split,
        cache_dir=model_args.cache_dir,
        token=model_args.hf_hub_token,
        streaming=data_args.streaming,
        **kwargs,
    )
    if data_args.streaming and not isinstance(dataset, IterableDataset):
        # ensure num_shards >= world_size (for correct rank sharding via IterableDatasetShard),
        # and is a multiple of dataloader_num_workers (so each worker gets whole shards).
        dnw = max(training_args.dataloader_num_workers, 1)
        ws = max(getattr(training_args, "world_size", 1), 1)
        num_shards = max(dnw, ((ws + dnw - 1) // dnw) * dnw)
        dataset = dataset.to_iterable_dataset(num_shards=num_shards)

    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    # ---- 在线数据修复（raw sharegpt schema 上）----
    # 必须在 align_dataset 之前；align 之后字段会被重命名为 prompt/response/image，
    # 且 <image> 占位会被 template 替换，此时再修语义太晚。
    # 见 src/llmtuner/data/fixers.py 注释。
    fixer_pipeline = _build_fixer_pipeline(dataset_attr, data_args)
    if fixer_pipeline is not None:
        dataset = dataset.map(_make_fixer_map_fn(fixer_pipeline), batched=False)

    dataset = align_dataset(dataset, dataset_attr, data_args, training_args, template)

    # fixer pipeline 通过把 messages 置 [] 标记丢弃；
    # SharegptDatasetConverter 会走 broken_data 分支输出 prompt=[]，这里实际过滤掉。
    if fixer_pipeline is not None:
        dataset = dataset.filter(lambda ex: len(ex.get("prompt") or []) > 0)

    return dataset


def merge_dataset(
    all_datasets: List[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    if len(all_datasets) == 1:
        return all_datasets[0]
    elif data_args.mix_strategy == "concat":
        if data_args.streaming:
            logger.warning("The samples between different datasets will not be mixed in streaming mode.")
            # streaming + concat 时 HF 用首批样本推断每个 dataset 的 features，
            # 各数据集的 image / video / video_frames / audio 字段会被推断为
            #   - List(Value('null'))   —— 字段恒为空（如纯文本 NavQA）
            #   - List(Value('string')) —— 字段有值
            # _check_if_features_can_be_aligned 不接受 List(null) ↔ List(string) 自动对齐
            # （只允许 null ↔ 任意类型），concat 直接 raise。
            # 在这里显式把 align_dataset 输出的 8 个固定字段统一 cast 成同一 Features，
            # 让所有数据集（纯文本 / 图片 / 视频 / 多模态混合）都能拼到一起。
            unified_features = Features({
                "prompt": [{"role": Value("string"), "content": Value("string")}],
                "response": [{"role": Value("string"), "content": Value("string")}],
                "system": Value("string"),
                "tools": Value("string"),
                "image": HFSequence(Value("string")),
                "video": HFSequence(Value("string")),
                "video_frames": HFSequence(Value("string")),
                "audio": HFSequence(Value("string")),
            })
            all_datasets = [d.cast(unified_features) for d in all_datasets]
        return concatenate_datasets(all_datasets)
    elif data_args.mix_strategy.startswith("interleave"):
        if not data_args.streaming:
            logger.warning("We recommend using `mix_strategy=concat` in non-streaming mode.")
        return interleave_datasets(
            datasets=all_datasets,
            probabilities=data_args.interleave_probs,
            seed=training_args.seed,
            stopping_strategy="first_exhausted" if data_args.mix_strategy.endswith("under") else "all_exhausted",
        )
    else:
        raise ValueError("Unknown mixing strategy.")


def dict_datasets(all_datasets):
    if len(all_datasets) == 0:
        return None
    if len(all_datasets) == 1:
        return all_datasets[0]
    return {f"dataset_{i}": all_datasets[i] for i in range(len(all_datasets))}


def _get_merged_dataset(
    dataset_attrs: Optional[Sequence["DatasetAttr"]],
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    as_dict: bool = False,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""
    Gets the merged datasets in the standard format.
    """
    if dataset_attrs is None or len(dataset_attrs) == 0:
        return None

    datasets = []
    for dataset_attr in dataset_attrs:
        if (stage == "rm" and dataset_attr.ranking is False) or (stage != "rm" and dataset_attr.ranking is True):
            raise ValueError("The dataset is not applicable in the current training stage.")

        datasets.append(_load_single_dataset(dataset_attr, model_args, data_args, training_args, template))
    if as_dict:
        return dict_datasets(datasets)
    return merge_dataset(datasets, data_args, training_args)


def _get_preprocessed_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    is_eval: bool = False,
    get_predict: bool = False,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    if dataset is None:
        return None
    predict_with_generate = False
    if training_args.predict_with_generate:
        if training_args.do_predict:
            # 单独走predict会触发
            predict_with_generate = True
        if training_args.do_train and training_args.do_eval and get_predict:
            # 边train边eval时,predict数据集会触发
            predict_with_generate = True

    dataset_processor = get_dataset_processor(data_args, stage, template, tokenizer, predict_with_generate)
    # 在用户配置--image后，输入数据会加入image列，经过preprocess_func处理后image列需要保留
    reserve_column_name = set()
    if data_args.image:
        reserve_column_name.add('image')
    if data_args.video:
        reserve_column_name.add('video')
    if data_args.audio:
        reserve_column_name.add('audio')

    sample_data = get_sample_data(dataset=dataset, use_subprocess=True)
    column_names = sorted(sample_data.keys() - reserve_column_name)
    kwargs = {}
    if not data_args.streaming:
        local_process_index = int(os.getenv("LOCAL_PROCESS_RANK", training_args.local_process_index))
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (local_process_index != 0),
            desc="Running tokenizer on dataset",
        )

    dataset = dataset.map(dataset_processor.preprocess_dataset, batched=True, remove_columns=column_names, **kwargs)

    if training_args.should_log:
        try:
            if predict_with_generate:
                print("generate example:")
            elif is_eval:
                print("eval example:")
            else:
                print("training example:")
            sample_data = get_sample_data(dataset=dataset, use_subprocess=True)
            dataset_processor.print_data_example(sample_data)
        except StopIteration:
            raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")

    if stage == "ppo":
        dataset = dataset.remove_columns(["labels"])

    return dataset


def _get_sequence_parallel_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    model_args: "ModelArguments",
    training_args: "Seq2SeqTrainingArguments",
    tokenizer: "PreTrainedTokenizer",
    is_eval: bool = False,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    if data_args.shuffle_for_sequence_parallel:
        dataset = dataset.shuffle(seed=training_args.seed)
    return dataset


def _get_dataset_dict(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    predict_with_generate: bool = False,
) -> "DatasetDict":
    if template is None:
        template = get_template_and_fix_tokenizer(
            tokenizer, data_args.template, processor_args=model_args, enable_thinking=data_args.enable_thinking
        )
    if data_args.train_on_prompt and template.efficient_eos:
        raise ValueError("Current template does not support `train_on_prompt`.")
    if data_args.tokenized_path is not None:
        tokenized_paths = data_args.tokenized_path.split(",")
        if has_tokenized_data(tokenized_paths[0]):
            logger.warning("Loading dataset from disk will ignore other data arguments.")
            dataset_dict = defaultdict(list)
            for tokenized_path in tokenized_paths:
                load_dict = load_from_disk(tokenized_path, storage_options=get_storage_options(tokenized_path))
                for k, v in load_dict.items():
                    dataset_dict[k].append(v)
            dataset_dict = {k: concatenate_datasets(v) if len(v) > 1 else v[0] for k, v in dataset_dict.items()}
            return dataset_dict
        elif tokenized_paths[0].startswith("odps://") or tokenized_paths[0].startswith("ailake://"):
            if tokenized_paths[0].startswith("odps://"):
                data_path = "odps"
            else:
                data_path = "ailake"

            if "trust_remote_code" in inspect.signature(load_dataset).parameters:  # for datasets==2.16.0
                kwargs = {"trust_remote_code": True}
            else:
                kwargs = {}

            if not data_args.streaming:
                kwargs["num_proc"] = data_args.preprocessing_num_workers

            if data_path in ["ailake", "odps"]:
                kwargs['num_workers'] = training_args.dataloader_num_workers if data_args.streaming else data_args.preprocessing_num_workers
                logger.info(f"Loading {data_path} dataset, num_workers is {kwargs['num_workers']}")

            dataset_dict = load_dataset(
                path=data_path,
                name=None,
                data_dir=None,
                data_files=tokenized_paths,
                cache_dir=model_args.cache_dir,
                token=model_args.hf_hub_token,
                streaming=data_args.streaming,
                **kwargs,
            )

            def parse_postprocess_example(example, cutoff_len=None):
                input_ids = ast.literal_eval(example['input_ids'])
                labels = ast.literal_eval(example['labels'])
                loss_mask = ast.literal_eval(example['loss_mask'])
                attn_mask = ast.literal_eval(example['attn_mask'])

                new_labels = [label if mask else -100 for label, mask in zip(labels, loss_mask)]
                if cutoff_len is not None:
                    input_ids = input_ids[:cutoff_len]
                    new_labels = new_labels[:cutoff_len]
                    attn_mask = attn_mask[:cutoff_len]
                return {
                    "input_ids": input_ids,
                    "labels": new_labels,
                    "attention_mask": attn_mask,
                }

            kwargs = {}
            if not data_args.streaming:
                kwargs = dict(
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=(not data_args.overwrite_cache),
                    desc="Running parse_postprocess_example on dataset",
                )
            fn_kwargs = dict(
                cutoff_len=data_args.cutoff_len,
            )
            dataset_dict = dataset_dict.map(parse_postprocess_example, batched=False,
                                  remove_columns=['idx', 'loss_mask', 'attn_mask'], fn_kwargs=fn_kwargs, **kwargs)
            return dataset_dict
        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

        if training_args.world_size > 1:
            raise ValueError("set world_size > 1 when use preprocess datasets; if more Disk or CPU are required, increase gpu num in cluster.json file")

    # Load and preprocess dataset
    with training_args.main_process_first(desc="load dataset"):
        dataset_attrs = get_dataset_list(data_args, get_eval=False)
        eval_attrs = get_dataset_list(data_args, get_eval=True)
        dataset = _get_merged_dataset(dataset_attrs, model_args, data_args, training_args, stage, template=template)
        eval_dataset = _get_merged_dataset(
            eval_attrs, model_args, data_args, training_args, stage, template=template, as_dict=True
        )
        if predict_with_generate:
            predict_dataset = _get_merged_dataset(
                eval_attrs, model_args, data_args, training_args, stage, template=template, as_dict=True
            )
        else:
            predict_dataset = None

    with training_args.main_process_first(desc="pre-process dataset"):
        dataset = _get_preprocessed_dataset(
            dataset, data_args, training_args, stage, template, tokenizer, is_eval=False, get_predict=False
        )
        if isinstance(eval_dataset, dict):
            for k, v in eval_dataset.items():
                eval_dataset[k] = _get_preprocessed_dataset(
                    v, data_args, training_args, stage, template, tokenizer, is_eval=True, get_predict=False
                )
        else:
            eval_dataset = _get_preprocessed_dataset(
                eval_dataset, data_args, training_args, stage, template, tokenizer, is_eval=True, get_predict=False
            )
        if isinstance(predict_dataset, dict):
            for k, v in predict_dataset.items():
                predict_dataset[k] = _get_preprocessed_dataset(
                    v, data_args, training_args, stage, template, tokenizer, is_eval=False, get_predict=True
                )
        else:
            predict_dataset = _get_preprocessed_dataset(
                predict_dataset, data_args, training_args, stage, template, tokenizer, is_eval=False, get_predict=True
            )

        if model_args.sequence_parallel_size > 1:
            dataset = _get_sequence_parallel_dataset(
                dataset, data_args, model_args, training_args, tokenizer, is_eval=False
            )
            if eval_dataset is not None:
                eval_dataset = _get_sequence_parallel_dataset(
                    eval_dataset, data_args, model_args, training_args, tokenizer, is_eval=True
                )

        if data_args.val_size > 1e-6:
            assert eval_dataset is None, "The validation dataset is not supported when `val_size` is set."
            dataset_dict = split_dataset(dataset, data_args, seed=training_args.seed)
        else:
            dataset_dict = {}
            if dataset is not None:
                if data_args.streaming:
                    dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)

                dataset_dict["train"] = dataset

            if eval_dataset is not None:
                if isinstance(eval_dataset, dict):
                    for k, v in eval_dataset.items():
                        dataset_dict[f"validation_{k}"] = v
                else:
                    if data_args.streaming:
                        eval_dataset = eval_dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)
                    dataset_dict["validation"] = eval_dataset

            if predict_dataset is not None:
                if isinstance(predict_dataset, dict):
                    for k, v in predict_dataset.items():
                        dataset_dict[f"test_{k}"] = v
                else:
                    if data_args.streaming:
                        predict_dataset = predict_dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)
                    dataset_dict["test"] = predict_dataset

            dataset_dict = DatasetDict(dataset_dict)
        if data_args.tokenized_path is not None:
            dataset_dict.save_to_disk(
                data_args.tokenized_path,
                storage_options=get_storage_options(data_args.tokenized_path),
                num_proc=data_args.preprocessing_num_workers,
            )
            logger.info("Saved tokenized dataset to {}".format(data_args.tokenized_path))
            if training_args.world_size > 1:
                logger.warning("because world_size > 1, the program exit with error code -1. ignore it.")
                sys.exit(-1)
            sys.exit(0)
        return dataset_dict


def get_dataset(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    predict_with_generate: bool = False,
) -> Dict[str, Union["Dataset", "IterableDataset", "DatasetDict"]]:
    dataset_dict = _get_dataset_dict(
        tokenizer, model_args, data_args, training_args, stage, template, predict_with_generate
    )
    dataset_module = {}
    dataset_module["eval_dataset"] = {
        k.replace("validation_", ""): v for k, v in dataset_dict.items() if k.startswith("validation_")
    }
    dataset_module["predict_dataset"] = {
        k.replace("test_", ""): v for k, v in dataset_dict.items() if k.startswith("test_")
    }
    if len(dataset_module["eval_dataset"]) == 0:
        dataset_module.pop("eval_dataset")
    if len(dataset_module["predict_dataset"]) == 0:
        dataset_module.pop("predict_dataset")

    if "train" in dataset_dict:
        dataset_module["train_dataset"] = dataset_dict["train"]
    if "validation" in dataset_dict:
        dataset_module["eval_dataset"] = dataset_dict["validation"]
    if "test" in dataset_dict:
        dataset_module["predict_dataset"] = dataset_dict["test"]
    return dataset_module
