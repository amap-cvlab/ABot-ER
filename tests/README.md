# Test plan

`test_publication_compatibility.py` covers both residual schemas, all released
layer layouts, mixed-schema rejection, route exclusivity, the TP tensor-key
contract, disabled legacy service factories, and pinned public-data integrity
using synthetic fixtures. It does not load production weights or run a GPU.

`test_residual_cpu.py` checks the training/evaluation residual implementations
with real CPU tensors in both parameter namespaces, mixed task routes, gradients
and strict state-dict reload. With `safetensors` installed it also tests planning
weight merging and prevents double application. These small fixtures do not
replace a full-size GPU loading or distributed training test.

Required before release: Task-Conditioned Adaptive Training Strategy state/token validation; dense/MoE hook dimensions; VABench XML8 and W2P tuple contracts; media-tag alignment; split/identity leakage checks; deterministic release hashes; all-rank shard capacity; judge-failure accounting; exported checkpoint inference parity.

`test_reproduction.py` checks integrity, relocation and shard arithmetic using synthetic fixtures. `legacy/test_dhta.py` and `legacy/test_dhta_qwen35_attention_smoke.py` are migrated framework tests. `scripts/check_preview.py` only verifies documentation links and JSON syntax. Passing these tests does not by itself establish distributed-training or official benchmark equivalence.

`test_point_scorer_snapshot.py` runs unchanged upstream VABench/W2P processors
on synthetic images/masks. It requires Pillow, NumPy, pandas and loguru. Run with
`python -m unittest discover -s tests -p 'test_point_scorer_snapshot.py' -v`.
The test covers source hashes, XML8 denominator, missing/repeated/out-of-range
points, W2P variable denominator, and the runtime0–1 prompt override.
