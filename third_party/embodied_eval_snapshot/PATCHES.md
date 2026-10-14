# Model-loading compatibility

The adapter accepts the published task-conditioned configuration and tensor
namespace, as well as historical compatibility fields. It checks layer/rank,
route tokens and residual weight-loading errors and uses the local dense/MoE
implementation rather than executing model-repository Python code.

The evaluated-model loader requires task-conditioned residuals; it does not
provide an unconditioned-model fallback. The independent judge is unaffected.

`SOURCE_MANIFEST.json` records the current files and their preceding identities.
No benchmark prompt, parser, judge prompt, metric, or scoring algorithm is
changed by this model-loading adaptation.
