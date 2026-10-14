# Contributing

Use a focused branch and describe the change, evidence and affected protocol/data/model versions. Keep existing experiment records immutable. Any output-contract change must increment its contract version and retain prior scoring behavior for reproducibility.

Do not commit weights, raw media, secrets or machine-specific runtime paths. Add migration/parity evidence when enabling an implementation marked pending. Use `python3 scripts/check_preview.py` for the current documentation preview checks; this does not validate training correctness.
