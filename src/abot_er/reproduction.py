"""Small, auditable reproduction utilities; no model import during data checks."""
from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from functools import lru_cache
from itertools import zip_longest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def round_robin_digest(directory, names):
    """Historical pretrain builder: stripped lines, 8-byte big-endian length prefix."""
    digest = hashlib.sha256()
    with ExitStack() as stack:
        files = [stack.enter_context((Path(directory) / name).open("rb")) for name in sorted(names)]
        for batch in zip_longest(*files):
            for raw in batch:
                if raw is not None:
                    line = raw.rstrip(b"\r\n")
                    digest.update(len(line).to_bytes(8, "big"))
                    digest.update(line)
    return digest.hexdigest()


def verify_release(directory, manifest):
    directory = Path(directory).resolve()
    meta = read_json(manifest)
    expected = meta["shards"]
    if not expected or len({x["file"] for x in expected}) != len(expected):
        raise ValueError("Empty/duplicate shard manifest")
    names = [item["file"] for item in expected]
    if any(Path(name).name != name or "\\" in name for name in names):
        raise ValueError("Manifest shard names must be basenames")
    actual = sorted(x.name for x in directory.glob("*.jsonl"))
    if actual != sorted(names):
        raise ValueError("Release shard names do not match manifest")
    total = 0
    ordered = hashlib.sha256()
    for item in sorted(expected, key=lambda x: x["file"]):
        p = directory / item["file"]
        if p.is_symlink() or not p.resolve().is_relative_to(directory):
            raise ValueError("Release shard escapes its directory")
        h = hashlib.sha256()
        rows = 0
        with p.open("rb") as stream:
            for line in stream:
                if not line.strip():
                    raise ValueError(f"Blank record in {p.name}")
                json.loads(line)
                h.update(line)
                ordered.update(line)
                rows += 1
        if rows != item["rows"] or h.hexdigest() != item["sha256"]:
            raise ValueError(f"Shard integrity mismatch: {p.name}")
        if p.stat().st_size != item["bytes"]:
            raise ValueError(f"Shard byte-size mismatch: {p.name}")
        total += rows
    method = meta.get("hash_method", "shard_concat_raw_v1")
    if method == "round_robin_length_prefix_v1":
        digest = round_robin_digest(directory, names)
    elif method == "shard_concat_raw_v1":
        digest = ordered.hexdigest()
    else:
        raise ValueError(f"Unsupported hash method: {method}")
    if total != meta["physical_records"] or digest != meta["ordered_sha256"]:
        raise ValueError("Ordered release identity mismatch")
    return {"release_id": meta["release_id"], "physical_records": total,
            "ordered_sha256": digest, "shards": len(expected)}


def relocate_media(value, mapping, check_exists=True):
    if isinstance(value, list):
        return [relocate_media(x, mapping, check_exists) for x in value]
    if not isinstance(value, str):
        raise ValueError("Media reference must be a path string or list")
    matches = [(old, new) for old, new in mapping.items()
               if value == old or value.startswith(old.rstrip("/") + "/")]
    if not matches:
        raise ValueError(f"No media-root mapping for: {value}")
    old, new = max(matches, key=lambda x: len(x[0]))
    base = Path(new).resolve()
    dest = (base / value[len(old):].lstrip("/")).resolve()
    if dest != base and base not in dest.parents:
        raise ValueError("Media path escapes its configured root")
    if check_exists and not dest.is_file():
        raise FileNotFoundError(dest)
    return str(dest)


def prepare_release(source, manifest, output, mapping_file):
    """Preserve row/shard order and all labels; only replace media path roots."""
    verified = verify_release(source, manifest)
    meta = read_json(manifest)
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(Path(source).resolve()):
        raise FileExistsError("Use a new output directory; releases are immutable")
    mapping = read_json(mapping_file)
    if not isinstance(mapping, dict) or not mapping or not all(
        isinstance(k, str) and isinstance(v, str) and k and v for k, v in mapping.items()
    ):
        raise ValueError("Media mapping must be a nonempty object of old root to local root")
    output.mkdir(parents=True)
    shard_dir = output / "shards"
    shard_dir.mkdir()
    @lru_cache(maxsize=500_000)
    def relocate_path(value):
        # Releases/media are immutable. Check each distinct reference on first
        # use; repeated frame/image exposure need not repeat costly NAS stats.
        return relocate_media(value, mapping)

    def relocate(value):
        return [relocate(item) for item in value] if isinstance(value, list) else relocate_path(value)

    prepared = []
    ordered = hashlib.sha256()
    for item in sorted(meta["shards"], key=lambda x: x["file"]):
        dest = shard_dir / item["file"]
        with (Path(source) / item["file"]).open(encoding="utf-8") as source_file, dest.open("wb") as target:
            for line in source_file:
                record = json.loads(line)
                for field in ("images", "videos", "audios", "video_frames"):
                    if record.get(field):
                        record[field] = relocate(record[field])
                encoded = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                target.write(encoded)
                ordered.update(encoded)
        prepared.append({"file": dest.name, "rows": item["rows"],
                         "bytes": dest.stat().st_size, "sha256": sha256(dest)})
        if len(prepared) % 16 == 0 or len(prepared) == len(meta["shards"]):
            print(f"prepare-data: {len(prepared)}/{len(meta['shards'])} shards verified/relocated", flush=True)
    result = {**meta, "source_ordered_sha256": verified["ordered_sha256"],
              "ordered_sha256": ordered.hexdigest(), "shards": prepared,
              "hash_method": "shard_concat_raw_v1",
              "hash_definition": "SHA256 of raw JSONL lines in lexicographic shard filename order",
              "transformation": "media-root relocation only; row order/labels unchanged"}
    write_json(output / "manifest.json", result)
    write_json(output / "dataset_info.json", {meta["dataset_key"]: {
        "file_name": "shards", "formatting": "sharegpt", "num_samples": meta["physical_records"],
        "columns": {"messages": "conversations", "image": "images", "video": "videos"},
        "image_folder": "/", "video_folder": "/"}})
    return result


def rank_capacity(meta, world_size, accumulation, micro_batch=1):
    if world_size < 1 or accumulation < 1 or micro_batch < 1:
        raise ValueError("Batch parameters must be positive")
    rows = [x["rows"] for x in sorted(meta["shards"], key=lambda x: x["file"])]
    if len(rows) % world_size:
        raise ValueError("Shard count must be divisible by world size")
    capacities = [sum(rows[r::world_size]) // (accumulation * micro_batch)
                  for r in range(world_size)]
    return capacities


def render_recipe(recipe, model, data, output, world_size, public_data=False):
    import yaml
    spec = yaml.safe_load(Path(recipe).read_text())
    cfg = dict(spec["training"])
    if cfg.get("enable_dhta") is not True:
        raise ValueError("Released training recipes require task-conditioned residuals")
    model = Path(model).resolve()
    data = Path(data).resolve()
    model_config = read_json(model / "config.json")
    from .checkpoint_contract import normalize_checkpoint_config
    normalize_checkpoint_config(model_config)
    text_config = model_config.get("text_config", model_config)
    if text_config.get("num_hidden_layers") != spec["model"]["decoder_layers"]:
        raise ValueError("Backbone decoder layer count does not match recipe")
    if model_config.get("model_type") not in spec["model"]["model_types"]:
        raise ValueError("Backbone model_type does not match recipe")
    if spec["phase"] == "posttrain":
        if model_config.get("dhta_enabled") is not True:
            raise ValueError("Posttraining requires an existing task-conditioned checkpoint, not an unadapted base")
        for key, value in {"dhta_layers": [19, 23, 27, 31], "dhta_rank": 16}.items():
            if model_config.get(key) != value:
                raise ValueError(f"Parent checkpoint {key} mismatch")
        token_ids = [model_config.get(key) for key in ("dhta_qa_token_id", "dhta_task_plan_token_id")]
        if any(type(x) is not int or x < 0 for x in token_ids) or token_ids[0] == token_ids[1]:
            raise ValueError("Parent checkpoint has invalid DHTA route token IDs")
    elif model_config.get("dhta_enabled"):
        raise ValueError("Pretraining requires the original base, not an existing task-conditioned checkpoint")
    manifest = read_json(data / "manifest.json")
    if public_data:
        pinned = read_json(ROOT / "data/modelscope_release.json")
        stage = "stage_a_pretrain" if spec["phase"] == "pretrain" else "stage_b_posttrain"
        identity = {key: pinned[key] for key in ("repository", "revision", "checksums_sha256")}
        if (manifest.get("publication") != identity or manifest.get("publication_stage") != stage
                or manifest["physical_records"] != pinned["stages"][stage]["physical_records"]
                or manifest["release_id"] != f"ABot-ER-{pinned['release_version']}-{stage}"):
            raise ValueError("Wrong public dataset release; run verify-public-data and prepare-data")
    else:
        if manifest["release_id"] != spec["release_id"]:
            raise ValueError("Wrong dataset release; public exports require --public-data")
        if manifest.get("source_ordered_sha256", manifest["ordered_sha256"]) != spec["source_ordered_sha256"]:
            raise ValueError("Wrong source dataset hash")
    batch = spec["global_batch"]
    micro = cfg["per_device_train_batch_size"]
    if world_size <= 0 or batch % (world_size * micro):
        raise ValueError("GPU count must divide the frozen global batch")
    cfg["gradient_accumulation_steps"] = batch // (world_size * micro)
    capacities = rank_capacity(manifest, world_size, cfg["gradient_accumulation_steps"], micro)
    if min(capacities) <= cfg["max_steps"]:
        raise ValueError("Insufficient per-rank capacity or no tail reserve")
    cfg.update(model_name_or_path=str(model), dataset_dir=str(data), dataset=manifest["dataset_key"],
               output_dir=str(Path(output).resolve()), report_to="none", run_name=spec["name"],
               add_tokens_file=str(ROOT / "configs/runtime/dhta_task_tokens.json"),
               deepspeed=str(ROOT / "configs/runtime/ds_zero3_no_overlap.json"))
    return cfg, {"name": spec["name"], "phase": spec["phase"], "world_size": world_size,
                 "data_scope": "modelscope_public_export" if public_data else "historical_training_bytes",
                 "publication": manifest.get("publication") if public_data else None,
                 "global_batch": batch, "min_rank_capacity": min(capacities),
                 "tail_steps": min(capacities) - cfg["max_steps"],
                 "data_manifest_sha256": sha256(data / "manifest.json"),
                 "model_config_sha256": sha256(model / "config.json"),
                 "model_metadata_sha256": {name: sha256(model / name) for name in
                     ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
                      "processor_config.json", "model.safetensors.index.json") if (model / name).is_file()},
                 "dhta_implementation_sha256": sha256(ROOT / "third_party/tuningfactory/src/llmtuner/model/model_utils/dhta.py"),
                 "recipe_sha256": sha256(Path(recipe)), "training": cfg}
