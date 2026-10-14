import abc
import os
import copy
import collections
import glob

import random
import math

from datasets import Dataset, load_dataset, load_from_disk, DownloadMode, DatasetDict
from collections.abc import Callable
from typing import Dict, List, Mapping, Optional, Union
from dataclasses import asdict, dataclass, field
from loguru import logger as eval_logger
from tqdm import tqdm

import yaml

from embodied_eval.common.instance import Instance
from embodied_eval.common.metric import aggregation_for_metric
from embodied_eval.utils import load_yaml_config, load_json, pattern_match, create_iterator

# ======================= Task Config =======================
@dataclass
class TaskConfig(dict):
    # task naming/registry
    task: str = None
    # HF dataset options: which dataset to use, and what splits for what purpose
    dataset_path: str = None
    dataset_name: str = None
    load_from_disk: bool = False
    eval_split: str = None
    dataset_kwargs: dict = None
    # Formatting
    doc_to_visual: Union[Callable, str] = None
    doc_to_text: Union[Callable, str] = None
    doc_to_target: Union[Callable, str] = None
    doc_to_choice: Union[Callable, str, dict, list] = None
    process_results: Union[Callable, str] = None
    # Inference
    output_type: str = "generate_until"
    generation_kwargs: dict = None
    repeats: int = 1
    # Score
    metric_kwargs: dict = None

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        return setattr(self, item, value)

    def to_dict(self):
        cfg_dict = asdict(self)
        # remove values that are `None`
        for k, v in list(cfg_dict.items()):
            if v is None:
                cfg_dict.pop(k)
            elif isinstance(v, Callable):
                cfg_dict[k] = str(v)
        return cfg_dict

# ======================= Task Object =======================
class Task(abc.ABC):
    """
    A task represents an entire benchmark including its dataset, problems,
    answers, and evaluation methods.
    """

    def __init__(
            self,
            config: Optional[dict] = None,
            model_name: Optional[str] = None,
    ) -> None:
        self._instances = None
        self._config = TaskConfig(**config)
        if self.config is None:
            raise ValueError("Must pass a config to Task")

        self.dataset_path = getattr(self.config, 'dataset_path', None)
        self.dataset_name = getattr(self.config, 'dataset_name', None)
        self.load_from_disk = getattr(self.config, 'load_from_disk', False)
        self.dataset_kwargs = getattr(self.config, 'dataset_kwargs', None)
        self.output_type = getattr(self.config, 'output_type')

        self.prepare_dataset()
        self.task_docs = self.task_docs()
        self.features = list(self.task_docs.features.keys())

        # TODO MC

    @property
    def config(self):
        """Returns the TaskConfig associated with this class."""
        return self._config

    @property
    def instances(self):
        """After calling `task.build_all_requests()`, tasks
        maintain a list of the dataset instances which will be evaluated.
        """
        return self._instances

    @property
    def eval_docs(self) -> Dataset:
        if self.config.eval_split is not None:
            return self.dataset[self.config.eval_split]
        else:
            return self.dataset

    def dump_config(self) -> dict:
        """Returns a dictionary representing the task's config.

        :returns: str
            The fewshot context.
        """
        return self.config.to_dict()

    def task_docs(self) -> Dataset:
        if self.config.eval_split is not None:
            return self.dataset[self.config.eval_split]
        else:
            return self.dataset
    
    def doc_to_visual(self, doc: dict):
        if callable(self.config.doc_to_visual):
            if self.dataset_kwargs is not None:
                return (self.config.doc_to_visual(doc, self.config.dataset_kwargs))
            else:
                return (self.config.doc_to_visual(doc))
    
    def doc_to_text(self, doc: dict):
        if callable(self.config.doc_to_text):
            if self.dataset_kwargs is not None:
                return (self.config.doc_to_text(doc, self.config.dataset_kwargs))
            else:
                return (self.config.doc_to_text(doc))

    def _sample_doc_iterator(self, sample_doc_ids):
        for raw_doc_id in sample_doc_ids:
            doc_id = int(raw_doc_id)
            yield doc_id, self.eval_docs[doc_id]

    def doc_iterator(
        self,
        *,
        rank: int = 0,
        limit: Union[int, None] = None,
        world_size: int = 1,
        sample_doc_ids: Union[list[int], None] = None,
    ):
        if sample_doc_ids is not None:
            raw_iterator = self._sample_doc_iterator(sample_doc_ids)
            limit = None
        else:
            raw_iterator = enumerate(self.eval_docs)
            limit = int(limit) if limit is not None else None

        return create_iterator(
            raw_iterator,
            rank=int(rank),
            limit=limit,
            world_size=int(world_size),
        )

    def prepare_dataset(self):
        if not self.load_from_disk:
            self.dataset = load_dataset(
                path=self.dataset_path,
                name=self.dataset_name,
                download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
            )
        else:
            if self.dataset_path.endswith(".json") or self.dataset_path.endswith(".jsonl"):
                if '*' in self.dataset_path:
                    self.dataset = []
                    for file_path in glob.glob(self.dataset_path):
                        data = load_json(file_path)
                        if isinstance(data, dict):
                            self.dataset.append(data)
                        else:
                            self.dataset.extend(data)
                else:
                    self.dataset = load_json(self.dataset_path)
                    
                self.dataset = Dataset.from_list(self.dataset)
            elif self.dataset_path.endswith(".yaml"):
                self.dataset = []
                with open (self.dataset_path, "r") as f:
                    yaml_data = yaml.safe_load(f)
                    datasets = yaml_data.get("datasets", [])
                    # datasts:
                    #    - json_path: xxx1.json
                    #      sampling_strategy: first/end/random:1000
                    for dataset in datasets:
                        json_path = dataset.get("json_path")
                        sampling_strategy = dataset.get("sampling_strategy")
                        sampling_number = None

                        cur_dataset = load_json(json_path)
                        if ":" in sampling_strategy:
                            sampling_strategy, sampling_number = sampling_strategy.split(":")
                            if "%" in sampling_number:
                                sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_dataset) / 100)
                            else:
                                sampling_number = int(sampling_number)
                        if sampling_strategy == "first" and sampling_number is not None:
                            cur_dataset = cur_dataset[:sampling_number]
                        elif sampling_strategy == "end" and sampling_number is not None:
                            cur_dataset = cur_dataset[-sampling_number:]
                        elif sampling_strategy == "random" and sampling_number is not None:
                            random.shuffle(cur_dataset)
                            cur_dataset = cur_dataset[:sampling_number]

                        self.dataset.extend(cur_dataset)
                self.dataset = Dataset.from_list(self.dataset)
            else:  # dir
                if os.path.exists(os.path.join(self.dataset_path, "dataset_info.json")) or \
                   os.path.exists(os.path.join(self.dataset_path, "dataset_dict.json")):
                    self.dataset = load_from_disk(self.dataset_path)
                else:
                    self.dataset = load_dataset(
                        path=self.dataset_path,
                        name=self.dataset_name,
                        trust_remote_code=True
                    )
            
            # TODO Debug
            # if isinstance(self.dataset, dict):  
            #     self.dataset = self.dataset["val"]
            #     self.dataset = self.dataset.select(range(2))
            #     self.dataset = DatasetDict({"val": self.dataset})
            # else:
            #     self.dataset = self.dataset.select(range(2))
            

    def build_all_requests(
        self,
        limit: Union[int, None] = None,
        rank: int = 0,
        world_size: int = 1,
        sample_doc_ids: Union[list[int], None] = None,
    ):
        """Build a set of Instances for a task, and store them in task.instances"""
        eval_logger.info(f"Building contexts for {self.config.task} on rank {rank}...")
        if sample_doc_ids is not None:
            limit = None

        instances = []

        doc_id_docs = self.doc_iterator(
            rank=rank,
            limit=limit,
            world_size=world_size,
            sample_doc_ids=sample_doc_ids,
        )
        total_docs = len(sample_doc_ids) if sample_doc_ids is not None else len(self.eval_docs)
        count_limit = None if sample_doc_ids is not None else limit
        doc_iterator_for_counting = create_iterator(
            range(total_docs),
            rank=rank,
            limit=count_limit,
            world_size=world_size,
        )

        num_docs = sum(1 for _ in doc_iterator_for_counting)

        for doc_id, doc in tqdm(
            doc_id_docs,
            total=num_docs,
        ):
            ctx = self.doc_to_text(doc)
            per_task_metadata = {"task": self.config["task"], "doc_id": doc_id, "repeats": self.config.repeats, "split": self.config["eval_split"]}
            inst = self.construct_requests(ctx=ctx, doc_id=doc_id, metadata=per_task_metadata)
            if not isinstance(inst, list):
                inst = [inst]

            instances.append(inst)

        sliced_instances = instances[:limit] if limit is not None else instances
        flattened_instances = [instance for instance_group in sliced_instances for instance in instance_group]
        self._instances = flattened_instances


    def construct_requests(self, ctx, doc_id, **kwargs) -> Union[List[Instance], Instance]:
        split = kwargs.get("metadata").get("split")
        # TODO MC
        if self.output_type == "generate_until":
            arguments = (ctx, copy.deepcopy(self.config.generation_kwargs), self.doc_to_visual, doc_id,
                         self.config.task, split)

        return Instance(request_type=self.output_type, arguments=arguments, idx=0, **kwargs)

    def apply_filters(self) -> Optional[List[Instance]]:
        """Iterates over FilterEnsembles and applies them to instances"""
        if hasattr(self, "_filters"):
            for f in self._filters:
                f.apply(self._instances, self.task_docs)
        else:
            eval_logger.warning("No filter defined, passing through instances")
            return self._instances

    def process_results(self, doc, results,):
        if self.output_type == "generate_until":
            if isinstance(results, list) and isinstance(results[0], list):
                results = [res.strip() for res in results[0]]
            else:
                results = [res.strip() for res in results]

        kwargs = self.config.dataset_kwargs
        if callable(self.config.process_results):
            return self.config.process_results(doc, results, kwargs)
    
    def aggregate_results(self, value,):
        if callable(self.config.metric_kwargs["aggregation"]):
            return self.config.metric_kwargs["aggregation"](value)

# ======================= Task Output =======================
class TaskOutput:
    def __init__(
        self,
        task=None,
        task_name=None,
        task_config=None,
    ):
        self.task = task
        self.task_name = task_name
        self.task_config = task_config
        self.logged_samples = []

    @classmethod
    def from_taskdict(cls, task_name: str, task):
        task_config = dict(task.dump_config())
        return cls(
            task=task,
            task_name=task_name,
            task_config=task_config
        )
    
    def calculate_aggregate_metric(self) -> None:
        agg_result = self.task.aggregate_results(self.logged_samples)
        return agg_result
        # self.agg_metrics[metric] = aggregation_for_metric(value, aggregation)


# ======================= Task Manager =======================

class TaskManager:
    def __init__(
            self,
    ) -> None:
        self. _task_index = self.initialize_tasks(include_defaults=True)
        self._all_tasks = sorted(list(self._task_index.keys()))
        # TODO 评估单一task

    @property
    def all_tasks(self):
        return self._all_tasks

    @property
    def task_index(self):
        return self._task_index

    def initialize_tasks(
            self,
            include_defaults: bool = True,
    ):
        if include_defaults:
            task_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
        # TODO 额外的task dir
        tasks = self._get_tasks(task_dir=task_dir)
        return tasks

    def _get_tasks(self, task_dir: str):
        ignore_dirs = [
            "__pycache__",
        ]
        tasks = collections.defaultdict()
        for root, dirs, file_list in os.walk(task_dir):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for f in file_list:
                if f.endswith(".yaml"):
                    yaml_path = os.path.join(root, f)
                    # IMPORTANT: During task discovery we only need the `task` field and
                    # the YAML path. Do NOT import `!function ...` targets here, otherwise
                    # a missing `process.py` in any unrelated task will crash evaluation.
                    # The actual task config (with functions imported) is loaded lazily
                    # in `_get_config()` when the task is explicitly requested.
                    config = load_yaml_config(yaml_path, mode="ignore")

                    # Some YAML files under the tasks tree are not task configs
                    # (e.g., env files, README scaffolding). Skip files that
                    # don't define the required `task` field to avoid KeyError.
                    if not config or not isinstance(config, dict) or "task" not in config:
                        eval_logger.debug(f"Skipping yaml without 'task' field: {yaml_path}")
                        continue

                    tasks[config["task"]] = {
                        "type": "task",
                        "yaml_path": yaml_path,
                    }
        return tasks

    def _get_config(self, name):
        if name not in self.task_index:
            raise ValueError("Task {} not found".format(name))
        yaml_path = self.task_index[name]["yaml_path"]
        return load_yaml_config(yaml_path)

    def _load_individual_task(
            self,
            name: Optional[str] = None,
    ):
        def _load_task(config, task):
            task_object = Task(config=config, model_name=name)
            return {task: task_object}

        if isinstance(name, str):
            task_config = self._get_config(name)
            return _load_task(task_config, task=name)

    def load_tasks(self, task_list):
        if isinstance(task_list, str):
            task_list = [task_list]
        all_loaded_tasks = dict(collections.ChainMap(*map(self._load_individual_task, task_list)))
        return all_loaded_tasks

    def match_tasks(self, task_list):
        return pattern_match(task_list, self.all_tasks)

def get_task_dict(
    task_name_list: Union[str, List[Union[str, Dict, Task]]],
    task_manager: Optional[TaskManager] = None,
):
    if isinstance(task_name_list, str):
        task_name_list = [task_name_list]
    string_task_name_list = [task for task in task_name_list if isinstance(task, str)]

    if len(string_task_name_list) > 0:
        if task_manager is None:
            task_manager = TaskManager()

        task_name_from_string_dict = task_manager.load_tasks(
            string_task_name_list,
        )

    final_task_dict = {
        **task_name_from_string_dict,
    }

    return final_task_dict

def get_task_list(task_dict: dict):
    outputs = []
    for task_name, task_obj in task_dict.items():
        task_output = TaskOutput.from_taskdict(task_name, task_obj)
        outputs.append(task_output)

    return outputs
