# Architecture review

## Scope

Build one portable research repository for embodied QA and task planning. Separate the reusable Python package, experiment recipes, immutable data metadata, evaluation contracts and infrastructure adapters. Large media and model weights live in versioned artifact storage, not Git.

## Dependencies

```text
source manifests → data preparation → audited release → training recipe
backbone adapter + task-conditioned configuration + release → trainer → exported checkpoint
checkpoint + frozen evaluation contract → benchmark adapter → result manifest
```

Public training/inference code must not import an internal cluster client. Cluster launchers should call the public entry points. Benchmark adapters must use the benchmark's original scorer or a pinned fork whose changes are documented. The public data interface is the sealed release plus metadata; internal data-construction pipelines are outside the agreed public scope.

## Module ownership

| Module | Responsibility | Excluded responsibility |
| --- | --- | --- |
| models | Backbone-specific hooks, Task-Conditioned Adaptive Training Strategy construction, state validation | Dataset-specific sampling |
| data | Media readers, schema transforms, mixture, split/audit checks | Selecting test answers for supervision |
| training | Optimizer, scheduler, distributed trainer, export | Hard-coded storage credentials |
| inference | Processor, route injection, generation | Silently changing output coordinates |
| evaluation | Frozen prompt, parser, scorer and aggregation | Training data construction |

## Multi-backbone design

Qwen3.5 dense, Qwen3.6 MoE and Qwen3.8 adapters have independent compatibility records. Read layer indices, projection dimensions, token IDs and processor settings from verified configuration. Do not propagate the 9B layer list to other sizes without checking their architecture. Store implementation commit and supported Transformers version with each exported model.

## Lessons captured

- VABench-P XML and W2P tuple-list remain separate, versioned protocols.
- A Task-Conditioned Adaptive Training Strategy checkpoint requires its route tokens and residual weights; a generic loader is insufficient.
- Record planned versus actually consumed records; validate distributed shard capacity after all filtering.
- Distinguish parser/judge failure from a model's incorrect answer.
- Keep raw scores, coverage and judge revision; do not compare incomplete means to full-coverage means.
- An audit can establish zero observed overlap only under its stated checks and coverage.

## Initial implementation boundary

The first migration retains the real training framework under third_party/tuningfactory rather than reimplementing its numerical paths. The portable package owns data integrity, relocation, recipe validation and HF download verification. EQA orchestration and TP workspace/model/server launchers are implemented against frozen source snapshots. Validate training, inference and simulator execution in the target environment; utility tests alone do not establish end-to-end equivalence.
