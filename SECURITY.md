# Security and supported execution paths

The supported entry points are the documented `abot-er` CLI commands. The
vendored training framework's old Gradio WebUI and generic/publishing API
entry points are disabled. They are not supported public services. The separate
Task-Planning serving command binds to `127.0.0.1`; do not expose it without an
independently reviewed authenticated gateway.

Model downloads must use safetensors. The optional legacy value-head loader no
longer falls back to pickle/PyTorch serialization. Simulator assets can still
contain legacy pickle files: obtain those only from trusted, versioned upstream
distributions and verify their checksums before use. No claim is made that all
upstream deserialization paths have been eliminated.

Historical environment exports are provenance, not installation instructions.
Do not deploy their old PyTorch, Gradio or Transformers versions. A complete
validated GPU/simulator environment lock is still pending; changing a version
constraint alone does not establish that a vulnerability or ABI issue is fixed.

The ModelScope publication uses `task_conditioned_*` fields. The local adapters
also accept the historical `dhta_*` schema, validate routing and tensor keys,
and do not execute the model repository's remote Python when loading through
the documented evaluator. Incompatible or conflicting schemas must fail closed.

Security-relevant changes here do not constitute a penetration test or a
successful distributed training/inference validation. Do not include secrets,
private paths or sensitive model inputs in public issue reports.
