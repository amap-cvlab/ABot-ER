# Training

The released Qwen3.5-9B recipe is implemented by `abot_er.reproduction` and the
training framework in `third_party/tuningfactory`. See the
[training guide](../../../docs/reproduction.md) for commands and requirements.

Stage A initializes task-conditioned residuals on the original base model;
Stage B continues from the selected task-conditioned Stage A checkpoint.
Released recipes require the residuals to be enabled. Checkpoint export must
retain the configuration, residual weights, tokenizer additions and route IDs.
