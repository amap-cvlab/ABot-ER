# Local operational patches

`SOURCE_MANIFEST.json` records original and current hashes for modified files.
Public upstream baseline commits are in `UPSTREAM.md`.

Operational changes:

1. `scripts/run_full_eval_local.py`: replace a per-shard `HOME` override and
   recursive deletion with an explicit `ABOT_AI2THOR_BASE_DIR` cache. Validate
   shard names and release links. Retain caches for diagnosis.
2. `EmbodiedBench/.../env/thor_env.py`: route AI2-THOR's `base_dir` property to
   that private cache without changing `HOME`. Reject incompatible cache APIs.
3. `scripts/run_tp_full_eval_pipeline.py`: artifact pruning is opt-in rather
   than automatic. Reproduction wrappers never request pruning.
4. Remove machine-specific path comments, a personal Conda prefix and an
   obsolete proxy example; runtime algorithms are unchanged.
5. Accept the published `task_conditioned-qv-v1` schema as well as the
   historical compatibility schema. Check tensor namespaces and route tokens;
   fold only the planning residual into a separate serving copy, then remove
   adaptive config fields to prevent applying the residual twice.

No prompt, action schema, simulator episode, success rule or scorer was changed.
Passing path/cache unit tests is not proof of simulator-level equivalence;
ALFRED/Habitat/ET runtime smoke tests remain necessary before final release.

External simulator binaries and benchmark datasets are not embedded. The
portable CLI stages independent code and links only explicitly declared assets.
Task-Conditioned Adaptive Training Strategy Task-Planning merging is performed on a verified real copy of a checkpoint,
never its original files or hardlinks. Merged copies are TP-serving-only.

## Released-model scope

The source-model preparation entry requires task-conditioned residuals. It
rejects unconditioned source checkpoints; the generated TP-only merged serving
copy remains supported through its verified server manifest.
