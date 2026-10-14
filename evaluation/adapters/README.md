# Benchmark adapters

Embodied QA uses the pinned adapters under
`third_party/embodied_eval_snapshot`; task planning uses
`third_party/task_planning_snapshot`. The public entry points relocate dataset
paths while retaining benchmark prompts, parsers and scorers.

See [evaluation commands](../RUNNING.md) for dataset setup, model loading and
simulator requirements. Simulator dependencies are separate from training.
