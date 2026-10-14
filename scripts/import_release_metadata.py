"""Import reviewed, non-media release metadata from existing audit artifacts."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--pretrain-shards", required=True)
parser.add_argument("--posttrain-report", required=True)
parser.add_argument("--eval-contract", required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
pre = json.loads(Path(args.pretrain_shards).read_text())
post = json.loads(Path(args.posttrain_report).read_text())["release"]
for name, key, count, sha, shards in (
    ("pretrain_vabxml8", "joint_tp_2dqa_r4_clean_dhta_opensource_vabenchxml8_20260921", 462390,
     "549ecedd25bd843a8175bc8a2f60313d1ff8ea014f42786478fdfa851ba0c956", pre),
    ("posttrain_final_joint", "joint_tp_2dqa_dhta_final_joint_v1_20260910", 104960,
     "5b7d276eed62e059aacab0c9d26d63bee0c08ccf45ee028d5e00a27ec72f1d46", post["shards"]),
):
    assert sum(x["rows"] for x in shards) == count
    obj = {"schema_version": "1", "release_id": key, "dataset_key": key,
           "physical_records": count, "ordered_sha256": sha, "shards": shards,
           "public_download": None, "redistribution_review": "pending",
           "hash_method": "round_robin_length_prefix_v1" if name == "pretrain_vabxml8" else "shard_concat_raw_v1",
           "hash_definition": (
               "Sorted shards read round-robin; strip CR/LF; SHA256 of 8-byte big-endian length followed by each line"
               if name == "pretrain_vabxml8" else
               "SHA256 of raw JSONL lines in lexicographic shard filename order")}
    (root / "data/manifests" / f"{name}.json").write_text(json.dumps(obj, indent=2) + "\n")
contract = json.loads(Path(args.eval_contract).read_text())
for entry in contract.get("sources", {}).values():
    path = entry.get("path", "")
    for repo in ("embodied-arena/", "EvalTaskPlanning/"):
        if repo in path:
            entry["path"] = repo + path.split(repo, 1)[1]
            break
contract["status"] = "historical_snapshot_not_current_evaluator_certification"
(root / "evaluation/contracts/historical_20260910.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")
