# Evaluation

The repository preserves the original nine-benchmark EQA and three-benchmark
Task Planning implementations. Start with [running instructions](RUNNING.md),
[contracts](contracts/README.md), and the [adapter boundary](adapters/README.md).

## Dataset coverage

The 13 task configurations cover nine EQA benchmarks. Expected sample counts:
VSI 5,130; ERQA 400; W2P 100; EmbSpatial 3,640; Cosmos 100/100/100/100/110;
VABench 300; RoboVQA 1,921; OpenEQA 1,636; UniEQA 5,278.

Validate these counts and your media paths before inference. Unit tests do not
replace GPU inference, judge validation or simulator execution.

## Preserved behavior

VABench's raw YAML uses 0–1000; the actual ABotER runtime overrides it to 0–1.
The [recorded override](../configs/eval/vabench_runtime_contract.json) is applied
explicitly by `abot-er evaluate-eqa`. W2P uses its separate tuple-list contract.
Cosmos is the equal-weight mean of five subsets. EQA/TP means remain separate.

TP retains the official ET interactive-agent loop. Local changes only stage
paths safely, isolate AI2-THOR caches without rewriting HOME, and retain raw
artifacts by default; see [patches](../third_party/task_planning_snapshot/PATCHES.md).
The original source repositories, benchmark datasets and leaderboard CSVs are
not modified by these adapters.

Record actual runtime configuration, source hashes, data revision, generation,
media sampling, tokenizer and immutable judge weights. Report parsing failures,
judge failures, execution errors and valid zero scores separately. Full weights
require a distribution checksum manifest; config hashes alone are insufficient.
