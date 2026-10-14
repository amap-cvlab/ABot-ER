# Experiment configurations

`pretrain/q35_9b_dhta.yaml` and `posttrain/q35_9b_dhta_final_joint.yaml` contain parameters extracted from the sealed 9B runs. They are recipe documents consumed by `abot-er train`, not raw YAML for direct use with the legacy trainer. Portable GPU regression is pending.

Every recipe must bind a model config, data release hash, runtime version, initialization checkpoint, optimizer/scheduler, batch and media settings. All released ABot-ER models use the Task-Conditioned Adaptive Training Strategy. Per-size layer indices must match the saved checkpoint configuration.

Portable differences: local checkpoint paths replace MOS; report_to is disabled; loss plotting is disabled; run output is a new explicit directory. These changes do not establish numerical parity by themselves. The actual Task-Conditioned Adaptive Training Strategy math file is unchanged from the source snapshot.
