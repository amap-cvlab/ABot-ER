"""Verify downloaded ModelScope publication bytes before local path relocation.

No network, resampling, label construction or writes to downloaded source data.
The checked-in pointer anchors the checksum list; the list anchors every shard.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from .reproduction import ROOT, read_json, sha256, write_json

POINTER = ROOT / "data/modelscope_release.json"


def verify_publication(root, stage, pointer=POINTER):
    root = Path(root).resolve()
    pinned = read_json(pointer)
    expected = pinned["stages"][stage]
    for name, field in (("SHA256SUMS.txt", "checksums_sha256"),
                        ("RELEASE_MANIFEST.json", "release_manifest_sha256")):
        if sha256(root / name) != pinned[field]:
            raise ValueError(f"Publication identity mismatch: {name}; use the pinned revision")
    release = read_json(root / "RELEASE_MANIFEST.json")
    if release["release_version"] != pinned["release_version"]:
        raise ValueError("Publication version mismatch")
    if any(release["stages"][stage][key] != value for key, value in expected.items()):
        raise ValueError("Publication count mismatch")
    checksums = {}
    for line in (root / "SHA256SUMS.txt").read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValueError("Malformed publication checksum line")
        digest, name = match.groups()
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or name in checksums:
            raise ValueError("Unsafe or duplicate publication checksum path")
        checksums[name] = digest
    prefix = f"data/{stage}/"
    names = sorted(name for name in checksums if name.startswith(prefix) and name.endswith(".jsonl"))
    if len(names) != expected["shards"] or any(PurePosixPath(name).parent.as_posix() != prefix.rstrip("/") for name in names):
        raise ValueError("Publication shard set mismatch")
    actual = sorted(p.name for p in (root / prefix).glob("*.jsonl"))
    if actual != [PurePosixPath(name).name for name in names]:
        raise ValueError("Missing or extra JSONL shards")
    ordered, rows, shards = hashlib.sha256(), 0, []
    for name in names:
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Publication shard escapes download root")
        digest, count = hashlib.sha256(), 0
        with path.open("rb") as stream:
            for raw in stream:
                record = json.loads(raw)
                if record.get("task_family") not in ("2d_qa", "task_plan"):
                    raise ValueError("Missing/unknown task routing metadata")
                digest.update(raw)
                ordered.update(raw)
                count += 1
        if digest.hexdigest() != checksums[name]:
            raise ValueError(f"Publication shard checksum mismatch: {path.name}")
        rows += count
        shards.append({"file": path.name, "rows": count, "bytes": path.stat().st_size,
                       "sha256": digest.hexdigest()})
    if rows != expected["physical_records"]:
        raise ValueError("Publication record count mismatch")
    return {"schema_version": "1", "release_id": f"ABot-ER-{pinned['release_version']}-{stage}",
            "dataset_key": f"abot_er_public_{stage}", "physical_records": rows,
            "ordered_sha256": ordered.hexdigest(), "hash_method": "shard_concat_raw_v1",
            "shards": shards, "publication": {key: pinned[key] for key in
                ("repository", "revision", "checksums_sha256")},
            "publication_stage": stage, "reproduction_scope": "public_export_not_historical_bytes"}


def run(args):
    output = Path(args.output).resolve()
    source = Path(args.data).resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError("Use a new manifest path outside the immutable download directory")
    manifest = verify_publication(source, args.stage)
    write_json(output, manifest)
    print(json.dumps({"manifest": str(output), "physical_records": manifest["physical_records"],
                      "ordered_sha256": manifest["ordered_sha256"]}, indent=2))
