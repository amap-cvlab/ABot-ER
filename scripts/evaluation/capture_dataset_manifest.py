"""Capture explicitly provided local eval data identities; never copy answers/media.

Usage: python scripts/evaluation/capture_dataset_manifest.py --output NEW.json
This maintenance tool uses the readable reference YAML paths. External users
instead fill the path manifest for their own downloaded benchmark packages.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/embodied_eval_snapshot"))
os.environ["NOTEBOOK_CONTAINER"] = "1"
os.environ["OPENAI_API_KEY"] = "local-judge-unused"
os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:9/v1"
os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9/v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    from abot_er.evaluation import DEFAULT_TASKS
    from embodied_eval.tasks import TaskManager
    manager = TaskManager()
    result = {"scope": "eval-only; metadata identity, not full media content hash", "tasks": {}}
    for name in DEFAULT_TASKS:
        task = manager._load_individual_task(name)[name]
        config = task.config
        path = Path(config.dataset_path)
        meta_files = [path] if path.is_file() else sorted(path.rglob("dataset_info.json")) + sorted(path.rglob("state.json"))
        if not meta_files and path.is_dir():
            # Raw HF exports (e.g. W2P) use Parquet without save_to_disk metadata.
            # Hash annotation containers, never recurse into separate media roots.
            meta_files = sorted(p for p in path.rglob("*") if p.is_file()
                                and not p.name.startswith(".")
                                and p.suffix in {".parquet", ".json", ".jsonl", ".arrow"})
        digest = hashlib.sha256()
        for f in meta_files:
            digest.update(str(f.relative_to(path) if path.is_dir() else f.name).encode())
            with f.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
        if not meta_files:
            raise ValueError(f"No dataset metadata files found: {path}")
        entry = {"dataset_path": str(path.resolve()), "expected_count": len(task.eval_docs),
                 "revision": "local-metadata-sha256:" + digest.hexdigest()}
        for field in ("video_dir", "image_root"):
            value = (task.dataset_kwargs or {}).get(field)
            if value:
                entry[field] = str(Path(value).resolve())
        result["tasks"][name] = entry
        print(f"{name}: {entry['expected_count']}", flush=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(f"manifest={target}")


if __name__ == "__main__":
    main()
