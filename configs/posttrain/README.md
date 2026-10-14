# Posttraining recipe

`q35_9b_dhta_final_joint.yaml` defines one joint Stage B training run from the
selected task-conditioned Stage A checkpoint. It consumes the frozen
post-training release; optimizer and scheduler state are reset at the boundary.
See the [training guide](../../docs/reproduction.md) for checkpoint preparation
and launch commands. The filename retains the implementation compatibility identifier.
