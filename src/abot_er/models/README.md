# Models and Task-Conditioned Adaptive Training Strategy

The real Qwen/Task-Conditioned Adaptive Training Strategy implementation is retained in [the framework snapshot](../../../third_party/tuningfactory/PROVENANCE.md). A clean public inference API and per-backbone compatibility registry are pending. The verified numerical file is `third_party/tuningfactory/src/llmtuner/model/model_utils/dhta.py`; do not replace it with a new approximation.

Each model configuration must include base revision, Task-Conditioned Adaptive Training Strategy enable flag, exact layer indices and indexing convention, rank, Q/V hook names and dimensions, route token IDs and initialization behavior. Validate existing residual weights rather than silently initializing missing ones.

Task-Conditioned Adaptive Training Strategy EQA inputs use one `[2D_QA]`; planning uses one `[TASK_PLAN]`. Injection belongs in one shared input preparation path. All released ABot-ER models use this routing. Verify this on tokenized prefill, not only prompt strings.
