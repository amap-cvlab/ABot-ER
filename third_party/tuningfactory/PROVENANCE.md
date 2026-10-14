# Training framework provenance

This framework derives from LLaMA-Factory/TuningFactory. Original copyright
headers are retained. Adaptations include task-conditioned residuals, multimodal
handling, sealed streaming recipes and opt-in legacy backend hooks. The released
recipe does not enable sample-loss weighting.

The portable SFT entry point uses local model/data paths and explicit output
directories. Organization-specific launch scripts, service endpoints and
operational records are not distributed. The mathematical implementation,
routing and training losses are preserved.

See [third-party notices](../../THIRD_PARTY_NOTICES.md) for attribution and
license-packaging requirements. Validate installation, checkpoint loading and
GPU execution in the target environment before a full run.

Release maintenance (2026-10-14): the local loaders accept the published
`task_conditioned-qv-v1` namespace without renaming checkpoint tensors. The
legacy WebUI/API entry points are disabled, and the optional value-head loader
is safetensors-only. An Apache-2.0 license text from upstream v0.9.4 is included;
this does not claim that this mixed snapshot is v0.9.4 or is vulnerability-free.
The exact source revision and a validated GPU dependency lock remain pending.
