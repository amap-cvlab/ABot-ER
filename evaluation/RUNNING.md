# Running the frozen evaluation adapters

Status: implementation and data-loading checks are available; full GPU and
simulator end-to-end acceptance is still in progress. Do not treat synthetic
smoke scores as benchmark results. All commands run from a source checkout.

## EQA: nine benchmarks, thirteen task configurations

The ModelScope release uses `task_conditioned_*` configuration and tensor keys.
The adapters accept that schema and the historical `dhta_*` schema, preserve the
matching tensor namespace, validate residual loading, and inject `[2D_QA]` for
EQA or `[TASK_PLAN]` for planning. Unknown/conflicting schemas fail closed.
Do not rename fields or tokens manually. Remote model Python is not executed by
the local EQA loader. Complete GPU acceptance of the publication adapter is pending.

Use an evaluation environment with the frozen Transformers/Qwen3.5 runtime,
PyTorch, Accelerate, Datasets, Decord, Pillow, SciPy, scikit-learn, OpenAI client,
Loguru and tqdm. The exact tested GPU environment will be published after the
current environment probe; do not infer it from the login-node package list.

Copy `configs/eval/datasets.example.json` to a local manifest and replace every
path and revision. Its task counts are the verified formal coverage. Paths may
point at local HF `save_to_disk` exports, Parquet/JSON datasets and their media
roots, as required by the original task loader. Benchmark data is evaluation-only.

```bash
abot-er evaluate-eqa --model /models/abot-er --datasets /data/eqa-manifest.json \
  --output /runs/eqa-preflight --no-judge --check-data

abot-er evaluate-eqa --model /models/abot-er --datasets /data/eqa-manifest.json \
  --output /runs/eqa-smoke --tasks vabench_aboter,where2place-point_aboter \
  --limit 2 --no-judge --batch-size 1

abot-er evaluate-eqa --model /models/abot-er --datasets /data/eqa-manifest.json \
  --output /runs/eqa-full --judge-model /models/frozen-judge --batch-size 4
```

Use `--dry-run` to inspect the resolved contract without loading a GPU model.
`--check-data` loads all task datasets, checks counts, and renders **the first
sample per task**, not every video. `--limit` changes generation length for a
cheap smoke test and marks its manifest `smoke`; omit it for formal evaluation.

The runner imports the frozen renderer/parser/scorer, preserves per-task
generation settings, uses 32 uniform video frames, and records source hashes,
dataset revision/count, checkpoint metadata and judge identity. It rejects
prompt/scorer overrides through the data manifest. Task-Conditioned Adaptive Training Strategy EQA routing is injected
by the original adapter. Released ABot-ER models require the matching task route. Full weight hashes
must come from the checkpoint's distribution manifest, not metadata hashes.

VABench's **runtime** 0–1 override is applied explicitly; the raw upstream YAML
uses 0–1000 and must not be run on its own. W2P retains tuple-list scoring. Cosmos
is the unweighted mean of five subtask scores, not a pooled sample mean. Keep
judge failures visible; `--no-judge` runs are not complete nine-benchmark scores.

After evaluation, `artifact_audit.json` checks sample coverage/uniqueness and
counts empty or missing responses. A smoke run with an empty generation fails
even if the upstream generator caught the exception and exited normally. Full
runs keep original scores and disclose empty responses rather than silently
repairing or excluding them. Artifact completeness is not proof of answer
format compliance, judge correctness or benchmark accuracy.

## Task Planning: official interactive agent and simulators

Install the pinned VirtualHome, AI2-THOR and Habitat assets/environments described
by the benchmark upstreams, plus Xvfb and software OpenGL where required. Their
validated installation lock/container is a pending release artifact. ET's
direct-plan experimental path is not used by this runner.

Fill `configs/eval/tp_assets.example.json` with your own absolute paths and
immutable asset-manifest revision. Prepare a new private workspace:

```bash
abot-er prepare-tp --assets /data/tp-assets.json --output /runs/tp-runtime
abot-er prepare-tp-model --model /models/abot-er --output /models/abot-er-tp-copy
```

The second command copies and hashes the checkpoint before folding the fixed
Task-Planning residual into Q/V projections. It never modifies original weights.
Do not use the TP-only merged copy for EQA or training. The generated
`abot_tp_server_manifest.json` states whether routing is enabled and whether a
historical chat-template profile applies. If that profile exists, the serving
command **must** use its recorded `chat_template` and generation defaults; do
not substitute the checkpoint's newer template silently.

Inspect, then start the vLLM server using this TP-only copy. The launcher verifies
the prepared weight hashes and applies the original TP thinking-on policy and
historical template/generation profile. EQA separately retains thinking-off.
GPU runtime validation is still pending; this is not yet a tested install lock.

```bash
abot-er serve-tp --server-manifest /models/abot-er-tp-copy/abot_tp_server_manifest.json --dry-run
abot-er serve-tp --server-manifest /models/abot-er-tp-copy/abot_tp_server_manifest.json
```

The default is one node, tensor parallel 8, BF16, max length 8,192, localhost
port 8000. It respects scheduler GPU masks. After the server is healthy:

```bash
abot-er evaluate-tp --runtime /runs/tp-runtime/runtime.json \
  --server-manifest /models/abot-er-tp-copy/abot_tp_server_manifest.json \
  --job-name smoke-et --suite et --max-jobs 1 --workers 1 --dry-run

abot-er evaluate-tp --runtime /runs/tp-runtime/runtime.json \
  --server-manifest /models/abot-er-tp-copy/abot_tp_server_manifest.json \
  --job-name full-tp --api-base http://127.0.0.1:8000/v1
```

Run a separate smoke for `--suite alfred` and `--suite habitat` before `all`.
`--max-jobs` truncates shard coverage and is never a formal result. Formal
coverage must be ET 2,023 / ALFRED 300 / Habitat 300, with no unresolved execution
failures. Completed failed episodes are genuine scores; infrastructure errors
are a separate retry/coverage issue. Raw artifacts are retained by default.

Neither CLI updates the internal leaderboard CSVs or publishes results. Compare
full-precision results only under identical coverage and frozen contracts;
report EQA and Task Planning means separately.
