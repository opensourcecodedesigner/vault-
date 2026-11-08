# Upstream interface review — 2026-11-08

Read the actual checked-out source, not illustrative package names. Only AuditKIT is installed/integrated. No upstream source was modified.

## AuditKIT

[Repository at 9da5838e61ef54d560a37605ddbcb19aceded9b3](https://github.com/Lexsi-Labs/AuditKIT/tree/9da5838e61ef54d560a37605ddbcb19aceded9b3), project version 1.1.6, Python >=3.10, zero required core dependencies. Inspected `pyproject.toml`, `README.md`, `src/auditkit/api.py`, `sample.py`, `scorers.py`, `cache.py`, `runner.py` and `report.py`. The optional HF backend has Transformers version bounds; we instead score our own actual saved model responses through its supported precomputed backend. `XDG_CACHE_HOME` defaults to the project cache when unset, keeping synthetic run cache files local and ignored.

The tested integration is `vault_ml.safety.score_records`; callers pass real saved outputs. Unit tests use explicitly labeled test strings only. No real-model safety score exists yet. Review refused harmful prompts, harmful assistance hidden behind a refusal, benign over-refusal, correct abstention, factual assertions and cited-source support manually. Then add human labels or a validated judge before substantive M0/adapter regression conclusions.

## SafeTune

[Repository at 1d824f026742fe2ce58491a8414f51a264931840](https://github.com/Lexsi-Labs/SafeTune/tree/1d824f026742fe2ce58491a8414f51a264931840), project version 0.1.12. Inspected package metadata, README, `runner/harden/_gradient_surgery.py`, `_base.py`, `interventions/harden/safegrad.py`, `core/optim/safegrad.py` and `_refusal_helpers.py`.

Requires Python >=3.12, Torch >=2.7, Transformers >=5.15,<6, Datasets <6, Accelerate <2, PEFT <1, BitsAndBytes and TRL >=0.12,<1.15, plus a substantial evaluation/logging stack. Dependencies were not installed speculatively. OS-independent metadata is not a measured native-Windows Tiny Aya qualification.

`SafeGradHFTrainer` accepts a PEFT-wrapped HF model, safety batches, an optional frozen reference, `rho` and `kl_temperature`; its global-gradient code iterates trainable parameters rather than targeting a specific transformer layer. Architecture compatibility is plausible but unverified on the gated checkpoint. The high-level `SafeGradTrainer.train` wraps LoRA, loads a separate reference on the training device and merges at save time. That default path is inappropriate to assume fits this laptop.

In the inspected code, missing alignment data/reference can disable surgery; this must never be called successful SafeTune. Gradient accumulation above one is explicitly rejected. KL mode adds a frozen aligned reference and student/reference logits; those memory costs are additional to ordinary QLoRA. Other interventions have different layer/checkpoint requirements; they are not interchangeable safety-preservation methods.

Bounded future M3 work: first qualify ordinary Tiny Aya QLoRA; on a 16–24 GB GPU install this exact SafeTune source revision with a constrained resolved environment; use the actual lower-level `SafeGradHFTrainer` with explicit safety-training data and a pinned frozen aligned reference; verify nonzero alignment signals, supported parameter gradients and adapter-preserving save/reload without invoking a destructive merge. Keep accumulation at one. Record extra safety/alignment tokens, forwards and reference costs separately, and rerun identical held-out science/safety suites. This is a future integration plan, not an executed config or custom algorithm labeled SafeTune.

## CuratorKIT

[Repository at cee1aed21d14992f650c5dfe09094b9625d0ce87](https://github.com/Lexsi-Labs/CuratorKIT/tree/cee1aed21d14992f650c5dfe09094b9625d0ce87). Inspected `pyproject.toml`, README and documented exports. Core requires Python >=3.11 and Pydantic/Typer/TQDM/PyYAML. Generation/connectors are optional stacks. Real exports include `sft_messages.jsonl`, `sft_alpaca.jsonl`, a manifest, rejected records, provenance and checksums. These are useful transport formats, not proof of scientific verification.

CuratorKIT remains Vijay's workstream. This repository consumes no invented CuratorKIT API and does not claim integration. Before importing a real export, agree source hashes, verifier-version/decision semantics, authorization and held-out split ownership. Preserve its provenance alongside the deterministic verifier receipt; never promote a model judge score into calculation ground truth.
