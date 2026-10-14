# Inference

`abot_er.evaluation` runs embodied QA evaluation; `abot_er.task_planning`
prepares and serves the planning route. See the
[evaluation guide](../../../evaluation/RUNNING.md).

The evaluated model must contain the Task-Conditioned Adaptive Training Strategy
configuration and residual weights. Loaders validate configuration aliases,
weight coverage and route IDs rather than silently evaluating the base model.
The independent judge has a separate model-loading contract.
