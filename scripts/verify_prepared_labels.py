"""Verify all prepared records preserve supervision/order, allowing media relocation only.

This is release transport verification, not data construction or a leakage audit.
"""
import argparse
from itertools import zip_longest
from functools import lru_cache
import json
from pathlib import Path

from abot_er.reproduction import read_json, relocate_media, verify_release

MEDIA_FIELDS = ("images", "videos", "audios", "video_frames")


def verify(source, source_manifest, prepared, roots):
    source, prepared = Path(source), Path(prepared)
    original = verify_release(source, source_manifest)
    relocated = verify_release(prepared / "shards", prepared / "manifest.json")
    metadata = read_json(source_manifest)
    mapping = read_json(roots)
    identity_mapping = all(Path(old) == Path(new) for old, new in mapping.items())
    @lru_cache(maxsize=500_000)
    def relocate_path(value):
        return relocate_media(value, mapping, check_exists=False)

    def relocate(value):
        return [relocate(item) for item in value] if isinstance(value, list) else relocate_path(value)
    if read_json(prepared / "manifest.json").get("source_ordered_sha256") != original["ordered_sha256"]:
        raise ValueError("Prepared release does not name the verified source")
    rows = 0
    for shard in sorted(metadata["shards"], key=lambda x: x["file"]):
        with (source / shard["file"]).open() as a, (prepared / "shards" / shard["file"]).open() as b:
            for index, (left, right) in enumerate(zip_longest(a, b), 1):
                if left is None or right is None:
                    raise ValueError("Prepared/source row counts differ")
                expected, actual = json.loads(left), json.loads(right)
                for field in MEDIA_FIELDS:
                    if expected.get(field) and not (identity_mapping and expected[field] == actual.get(field)):
                        expected[field] = relocate(expected[field])
                if expected != actual:
                    raise ValueError(f"Non-relocation change at {shard['file']}:{index}")
                rows += 1
    return {"status": "pass", "rows_compared": rows,
            "source_ordered_sha256": original["ordered_sha256"],
            "prepared_ordered_sha256": relocated["ordered_sha256"],
            "scope": "all fields/all records; only declared media path relocation allowed"}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--source-manifest", required=True)
    p.add_argument("--prepared", required=True)
    p.add_argument("--media-roots", required=True)
    args = p.parse_args()
    print(json.dumps(verify(args.source, args.source_manifest, args.prepared, args.media_roots), indent=2))
