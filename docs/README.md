# Documentation

[Project overview — English](../README.md) · [项目概览 — 简体中文](../README.zh-CN.md)

## Getting started

- [Installation and two-stage training](reproduction.md)
- [Datasets, mixtures, and schemas](../data/README.md)
- [ModelScope model and data downloads](../data/artifacts/README.md)
- [Evaluation setup and commands](../evaluation/RUNNING.md)
- [Benchmark contracts and reporting](../evaluation/README.md)

## Reference

- [Architecture and module boundaries](architecture.md)
- [Training stages and checkpoint selection](reproduction.md#2-获取并验证模型和数据)
- [Pre-training configuration](../configs/pretrain/q35_9b_dhta.yaml)
- [Post-training configuration](../configs/posttrain/q35_9b_dhta_final_joint.yaml)
- [Third-party notices](../THIRD_PARTY_NOTICES.md)

## Maintainers

- [Release checklist](release.md)
- [Training-framework provenance](../third_party/tuningfactory/PROVENANCE.md)
- [Evaluation-source provenance](../third_party/embodied_eval_snapshot/README.md)
- [Task-planning portability patches](../third_party/task_planning_snapshot/PATCHES.md)

## Documentation organization

The root README is the user-facing entry point. Installation details, protocol
contracts, historical release identities, and acceptance evidence belong in the
linked guides rather than in the project introduction. Unpublished artifacts
must remain explicitly marked as unavailable; documentation must not imply that
a CPU smoke test establishes full GPU training or benchmark reproduction.
The English and Simplified Chinese READMEs cover the same content. Keep their
commands, numerical settings, download availability, and validation scope in sync.
