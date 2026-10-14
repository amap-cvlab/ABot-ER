# Third-party notices

## Training framework

`third_party/tuningfactory` derives from LLaMA-Factory/TuningFactory and contains
upstream model code with original copyright headers retained. See its
[provenance](third_party/tuningfactory/PROVENANCE.md) for adaptations.

## Evaluation integrations

`third_party/embodied_eval_snapshot` contains the scoped EQA integration.
`third_party/task_planning_snapshot` contains the ET-Plan and EmbodiedBench
integration; public upstream references are in its `UPSTREAM.md` and local
changes are in `PATCHES.md`. Each `SOURCE_MANIFEST.json` records current hashes
and original hashes for modified files. Path sanitization does not change
benchmark prompts or scoring algorithms.

Original copyright headers are retained. The training framework's Apache-2.0
license text is included in `third_party/tuningfactory/LICENSE`; it was checked
against the official LLaMA-Factory v0.9.4 LICENSE (SHA-256
`50e6751797c50dedd75ef1b8a0d9e42f5f8472e9fbce91f34718e9f97b0c780a`).
This identifies the license text, **not** the version of all vendored source.
The exact training-framework upstream revision remains unrecorded.

### ALFRED attribution

The ALFRED subdirectory's [MIT license](third_party/task_planning_snapshot/EmbodiedBench/embodiedbench/envs/eb_alfred/LICENSE)
is included unchanged from the [pinned EmbodiedBench revision](https://github.com/EmbodiedBench/EmbodiedBench/blob/9be4e980e9cd6bcb38373cd4aab7c32724bdd401/embodiedbench/envs/eb_alfred/LICENSE).
It carries the original ALFRED copyright notice and is not treated as a blanket
license for all EmbodiedBench code or its external assets.

This repository does not blanket-relicense ET-Plan, EmbodiedBench, VirtualHome,
ALFRED, Habitat or their assets.

## Separate artifacts

Model weights, simulator binaries and media datasets are not bundled; their
licenses apply separately. Architecture references are citations, not permission
to redistribute third-party assets. Before redistributing a component, record
its public upstream URL, revision, license, modifications and required notices.
