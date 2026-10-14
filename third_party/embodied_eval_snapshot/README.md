# Evaluation source snapshot

Scope: ABotER/Qwen adapters, task-conditioned residuals, frozen-template
compatibility, nine-benchmark task processors/YAMLs, pointing utilities,
registry/task core, sequential evaluator and local judge. Portable orchestration
is in `src/abot_er/evaluation.py`; use `abot-er evaluate-eqa`. Other baseline
adapters are outside this snapshot.

`SOURCE_MANIFEST.json` records current file hashes and, for modified files,
original hashes. Dataset locations are placeholders: supply local datasets and
media explicitly through the portable runner. Prompts and scoring algorithms
are unchanged by path sanitization.

## Important runtime-contract difference

The source `vabench_aboter.yaml` contains a legacy 0–1000 prompt and coordinate
decoder. The actual ABotER wrapper defaults to `--vabench_coord_mode=0_1` and
overrides **both** at runtime. The observed override is recorded in
`configs/eval/vabench_runtime_contract.json`, including the wrapper SHA. Never
use the raw YAML as the aligned evaluation entry point. No source scorer was
rewritten to conceal this difference.

VABench divides point hits by eight (including missing points in the
denominator); W2P divides by actual parsed point count. Repeated coordinates are
accepted by the current implementation. Normalized coordinates use width/height
and Python round; coordinates exactly 1 can map outside the image. Synthetic
tests preserve these behaviors; they do not change the official protocol.

No benchmark media, test answers, model outputs or credentials are bundled.
See [third-party notices](../../THIRD_PARTY_NOTICES.md) for licensing information.
