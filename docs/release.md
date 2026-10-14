# Release checklist

Use this checklist for each versioned release. Keep actual license texts and
component notices with the corresponding distribution.

- Code: installable package, pinned tested environment, passing tests, source provenance and notices.
- Data: immutable release ID; unique/physical record counts; ordered hash; split filters; audit coverage and exclusions; per-source redistribution status.
- Models: base revision; Task-Conditioned Adaptive Training Strategy configuration; layer/rank/token IDs; exported weights and tokenizer; training ancestry.
- Reproduction: public command, required hardware, training/consumption budget, seed and known nondeterminism.
- Evaluation: all 12 contracts versioned; exact runtime settings; raw/full-precision scores; failed and missing counts; judge revision.
- Governance: archive the applicable licenses and notices for code, models and data; source-specific terms remain applicable.

## Evidence limitations

Removing direct test overlaps does not erase previous development decisions informed by test scores. Describe benchmark-driven development honestly and report any independent held-out evaluation separately. Do not turn an incomplete audit into a claim that every possible form of contamination is absent.

### Data documentation scope

The data landing page summarizes the released artifacts and reproduction workflow.
Detailed audit scope remains separate from that overview:

- JSONL hashes establish annotation-file identity. Media-byte consistency and semantic overlap require separate checks and coverage records.
- Source records distinguish public source name, revision, applicable terms, redistribution scope, and training-split evidence. Audit records identify the evaluation-anchor version and coverage of ID, question, media, scene/episode, and pHash checks, including unexamined subsets and quarantine records.
- Near-duplicate review candidates and confirmed leakage are separate classifications. Candidate records and their adjudications remain part of the audit history.
- Benchmark-informed development history is separate from the final release's overlap checks. The technical report should distinguish development feedback from independent held-out evaluation.
