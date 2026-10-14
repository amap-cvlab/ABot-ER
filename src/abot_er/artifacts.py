"""Download immutable HF artifacts with a checked, explicit file allowlist."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import re

from .reproduction import read_json, sha256, write_json


def validate_index(index):
    if index.get("schema_version") != 1:
        raise ValueError("Unsupported artifact index schema")
    if index.get("status") != "published":
        raise ValueError("Artifact is not published; a placeholder is not a download URL")
    if index.get("repo_type") not in {"model", "dataset"}:
        raise ValueError("repo_type must be model or dataset")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", index.get("repo_id", "")):
        raise ValueError("Explicit HF namespace/repository is required")
    if not re.fullmatch(r"[a-f0-9]{40}", index.get("revision", "")):
        raise ValueError("Pin a full immutable HF commit SHA; branches/tags are not accepted")
    files = index.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("A non-empty file allowlist is required")
    seen = set()
    for item in files:
        name = item.get("path", "")
        path = PurePosixPath(name)
        if (not name or path.is_absolute() or str(path) != name or ".." in path.parts
                or "\\" in name or any(part.startswith(".") for part in path.parts)
                or name == "abot_download_receipt.json" or any(ord(c) < 32 for c in name)):
            raise ValueError(f"Unsafe artifact path: {name!r}")
        if name in seen:
            raise ValueError(f"Duplicate artifact path: {name}")
        seen.add(name)
        if not re.fullmatch(r"[a-f0-9]{64}", item.get("sha256", "")):
            raise ValueError(f"Missing file SHA256: {name}")
        if type(item.get("size_bytes")) is not int or item["size_bytes"] < 0:
            raise ValueError(f"Invalid file size: {name}")
    return files


def checked_path(root, name):
    path = root / name
    # Reject symlink traversal in existing parent directories or final file.
    current = root
    for part in PurePosixPath(name).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlink in artifact destination: {current}")
    if not path.resolve().is_relative_to(root):
        raise ValueError("Artifact escaped destination")
    return path


def file_matches(path, item):
    return path.is_file() and path.stat().st_size == item["size_bytes"] and sha256(path) == item["sha256"]


def download(args):
    index = read_json(args.index)
    files = validate_index(index)
    root = Path(args.output).expanduser().resolve()
    paths = [(item, checked_path(root, item["path"])) for item in files]
    summary = {"repo_id": index["repo_id"], "repo_type": index["repo_type"],
               "revision": index["revision"], "file_count": len(files),
               "total_bytes": sum(item["size_bytes"] for item in files), "output": str(root)}
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return summary
    # Refuse overwriting user files. Correct already-downloaded files are reusable.
    for item, path in paths:
        if path.exists() and not file_matches(path, item):
            raise FileExistsError(f"Existing file differs from manifest: {path}; use a new output directory")
    from huggingface_hub import hf_hub_download
    root.mkdir(parents=True, exist_ok=True)
    for item, path in paths:
        if not file_matches(path, item):
            hf_hub_download(repo_id=index["repo_id"], repo_type=index["repo_type"],
                            revision=index["revision"], filename=item["path"], local_dir=str(root))
        checked_path(root, item["path"])
        if not file_matches(path, item):
            raise ValueError(f"Downloaded content failed size/SHA256 verification: {path}")
    summary["status"] = "verified"
    summary["artifact_index_sha256"] = sha256(Path(args.index))
    write_json(root / "abot_download_receipt.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def add_arguments(parser):
    parser.add_argument("--index", required=True, help="Published, commit-pinned artifact file index")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
