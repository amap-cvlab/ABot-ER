import os
from typing import Optional

import fsspec


def get_storage_options(path: str) -> Optional[dict]:
    if path.startswith("oss://"):
        res = {}
        if os.getenv("OSS_ACCESS_KEY_ID", None):
            res["key"] = os.getenv("OSS_ACCESS_KEY_ID")
        if os.getenv("OSS_ACCESS_KEY_SECRET", None):
            res["secret"] = os.getenv("OSS_ACCESS_KEY_SECRET")
        if os.getenv("OSS_ENDPOINT", None):
            res["endpoint"] = os.getenv("OSS_ENDPOINT")
        return res
    elif path.startswith("dfs://"):
        return {"singleton": False}  # avoid hang when using multi-processor
    else:
        return None


def get_fs(path: str) -> fsspec.AbstractFileSystem:
    storage_options = get_storage_options(path)
    if path.startswith("oss://"):
        return fsspec.filesystem("oss", **storage_options)
    elif path.startswith("dfs://"):
        return fsspec.filesystem("dfs", **storage_options)
    else:
        return fsspec.filesystem("file")


def has_tokenized_data(path: str) -> bool:
    fs = get_fs(path)
    return fs.isdir(path) and len(fs.listdir(path)) > 0
