import collections
import datetime
import functools
import fnmatch
import inspect
import yaml
import os
import pytz
import json

import importlib.util
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)
from itertools import islice
from loguru import logger as eval_logger


def load_json(json_path):
    if json_path.endswith('.json'):
        with open(json_path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                return [json.loads(line) for line in f if line.strip()]
    elif json_path.endswith('.jsonl'):
        data = []
        with open(json_path, "r") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    else:
        return None

def ignore_constructor(loader, node):
    return node

def import_function(loader, node):
    function_name = loader.construct_scalar(node)
    yaml_path = os.path.dirname(loader.name)

    *module_name, function_name = function_name.split(".")
    if isinstance(module_name, list):
        module_name = ".".join(module_name)
    module_path = os.path.normpath(os.path.join(yaml_path, "{}.py".format(module_name)))

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    function = getattr(module, function_name)
    return function

def load_yaml_config(yaml_path=None, mode="func"):
    constructor_fn = import_function if mode == "func" else ignore_constructor
    yaml.add_constructor("!function", constructor_fn)

    with open(yaml_path, "rb") as file:
        yaml_config = yaml.full_load(file)
    # TODO Group/Include
    return yaml_config


class Collator:
    """
    A utility class for reordering and batching elements of an array.

    This class provides methods to:
    - Sort an array using a provided sorting function.
    - Group elements based on a grouping function.
    - Generate batches from the sorted and grouped data.
    """

    def __init__(
            self,
            arr: List,
            sort_fn: Callable,
            group_fn: Callable = lambda x: x[1],
            grouping: bool = False,
    ) -> None:
        """
        Initializes the Collator class.

        Parameters:
        - arr (List): The list to reorder and group.
        - sort_fn (Callable): The sorting function.
        - group_fn (Callable, optional): The grouping function. Defaults to grouping by the second element.
        - grouping (bool, optional): Whether to group the elements. Defaults to False.
        """
        self.grouping = grouping
        self.sort_fn = sort_fn
        self.group_fn = lambda x: group_fn(x[1])  # Grouping by the second element of each item
        self.reorder_indices: List = []
        self.size = len(arr)
        self.arr_with_indices: Iterable[Any] = tuple(enumerate(arr))  # Enumerate with original indices
        if self.grouping:
            self.group_by_index()

    def __len__(self):
        """Returns the size of the array."""
        return self.size

    def get_original(self, newarr: List) -> List:
        """
        Restores the original order of elements from a reordered list.

        Parameters:
        - newarr (List): The reordered array.

        Returns:
        - List: The array with elements restored to their original order.
        """
        restored = [None] * self.size
        covered = [False] * self.size

        for idx, value in zip(self.reorder_indices, newarr):
            restored[idx] = value
            covered[idx] = True

        assert all(covered), "Not all elements were restored properly."

        return restored

    def group_by_index(self) -> None:
        """Groups elements based on the group function."""
        self.arr_with_indices = self.group(self.arr_with_indices, fn=self.group_fn, values=False)

    @staticmethod
    def group(arr: Iterable, fn: Callable, values: bool = False) -> Iterable:
        """
        Groups elements of an iterable based on a provided function.

        Parameters:
        - arr (Iterable): The iterable to be grouped.
        - fn (Callable): The function used for grouping.
        - values (bool, optional): If True, returns only the grouped values. Defaults to False.

        Returns:
        - Iterable: An iterable of grouped elements.
        """
        grouped = collections.defaultdict(list)
        for item in arr:
            try:
                hashable_group = tuple(
                    (
                        key,
                        tuple(value) if isinstance(value, collections.abc.Iterable) else value,
                    )
                    for key, value in sorted(fn(item).items())
                )
                grouped[hashable_group].append(item)
            except TypeError:
                grouped[fn(item)].append(item)

        return grouped.values() if values else grouped

    def get_batched(self, n: int = 1, batch_fn: Optional[Callable] = None) -> Iterator:
        """
        Generates and yields batches from the reordered array.

        Parameters:
        - n (int, optional): The size of each batch. Defaults to 1.
        - batch_fn (Optional[Callable], optional): A function to determine batch sizes. Defaults to None.

        Yields:
        - Iterator: An iterator over batches of reordered elements.
        """
        if self.grouping:
            for key, values in self.arr_with_indices.items():  # type: ignore
                reordered_values = self._reorder(values)
                batch = self.get_chunks(reordered_values, n=n, fn=batch_fn)
                yield from batch
        else:
            reordered_values = self._reorder(self.arr_with_indices)  # type: ignore
            batch = self.get_chunks(reordered_values, n=n, fn=batch_fn)
            yield from batch

    def _reorder(self, arr: Union[List, Tuple[Tuple[int, Any], ...]]) -> List:
        """
        Reorders elements in the array based on the sorting function.

        Parameters:
        - arr (List or Tuple): The array to reorder.

        Yields:
        - Iterator: The reordered elements one by one.
        """
        sorted_arr = sorted(arr, key=lambda x: self.sort_fn(x[1]))
        self.reorder_indices.extend([x[0] for x in sorted_arr])
        yield from [x[1] for x in sorted_arr]

    @staticmethod
    def get_chunks(iterable: Iterable, n: int = 0, fn: Optional[Callable] = None) -> Iterator:
        """
        Divides an iterable into chunks of a specified size or based on a function.

        Parameters:
        - iterable (Iterable): The input iterable to divide into chunks.
        - n (int, optional): The size of each chunk. Default is 0.
        - fn (Optional[Callable], optional): A function that returns the size of a chunk based on index. Default is None.

        Yields:
        - Iterator: An iterator yielding chunks of the iterable.

        Example:
        '''
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for chunk in get_chunks(data, 3):
            print(chunk)
        '''

        Output:
        [1, 2, 3]
        [4, 5, 6]
        [7, 8, 9]
        [10]
        """
        chunk = []
        for i, item in enumerate(iterable):
            chunk.append(item)
            if len(chunk) == (fn(i, iterable) if fn else n):
                yield chunk
                chunk = []

        if chunk:
            yield chunk

def pattern_match(patterns, source_list):
    if type(patterns) == str:
        patterns = [patterns]

    task_names = set()
    for pattern in patterns:
        try:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        except Exception as e:
            eval_logger.error(f"Error matching pattern {pattern}: {e}")
    return sorted(list(task_names))


def get_datetime_str(timezone="Asia/Singapore"):
    """
    Gets the current datetime in UTC+8 timezone as a string.
    """
    # Default: UTC+8 timezone
    tz = pytz.timezone(timezone)
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    local_time = utc_now.astimezone(tz)
    return local_time.strftime("%Y%m%d_%H%M%S")

def handle_arg_string(arg):
    if arg.lower() == "true":
        return True
    elif arg.lower() == "false":
        return False
    elif arg.isnumeric():
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg

def simple_parse_args_string(args_string):
    """
    Parses something like
        args1=val1,arg2=val2
    Into a dictionary
    """
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = [arg for arg in args_string.split(",") if arg]
    args_dict = {k: handle_arg_string(v) for k, v in [arg.split("=") for arg in arg_list]}
    return args_dict

def create_iterator(raw_iterator, rank, world_size, limit=None):
    return islice(raw_iterator, rank, limit, world_size)

def positional_deprecated(fn):
    """
    A decorator to nudge users into passing only keyword args (`kwargs`) to the
    wrapped function, `fn`.
    """

    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        if len(args) != 1 if inspect.ismethod(fn) else 0:
            print(f"WARNING: using {fn.__name__} with positional arguments is " "deprecated and will be disallowed in a future version of " "lmms-evaluation-harness!")
        return fn(*args, **kwargs)

    return _wrapper