# VAULT model workstream handoff

## Current result

Implemented the native Windows workstream and verified the parts that do not require gated Tiny Aya files. **Official-model execution remains blocked by a credential/session mismatch.** On resumption from `9f0ba2e8eb01638393056ea5a2960952160fb68a`, the user reports completed authentication, accepted terms and successful `whoami` in their terminal. The agent's exact project `hf.exe auth whoami` still reports not logged in; no local credential is visible and the pinned config download returned `GatedRepoError` outside the sandbox. Non-secret credential-path output from the successful terminal has been requested. No conclusion about the user's account approval is inferred from the missing local credential.

No authentic M0 response, Tiny Aya training run, Tiny Aya adapter, completed M1/M2/M3 model or real-model safety result exists. No scientific reliability improvement or safety-preservation conclusion is claimed.

## Inventory

- `vault_ml.config`: immutable pinned model/generation and validated training settings.
- `vault_ml.model_loader`: official Cohere2 loader, explicit NF4 GPU placement, optional unquantized inference offload, gated/cache diagnostics, admitted local-adapter hash/revision checks.
- `vault_ml.infer`: reusable `InferenceEngine` and structured request/CLI; no public server.
- `vault_ml.contracts`: proposed evidence and response schema, validation, missing-evidence warnings and structured source-ID allowlist.
- `vault_ml.resources`: CUDA allocator/available memory, sampled process RSS, system available RAM, CPU/wall time.
- `vault_ml.evaluation` / `fixtures`: ten deterministic held-out scientific scenarios, raw-response reports, fixture hashes and narrow structured-output scores.
- `vault_ml.training_data` / `train`: isolated synthetic training generator, bounded assistant-masked QLoRA, actual loss logging and changed/finite/saved/reloaded-adapter verification. Production checkpoint unexecuted.
- `vault_ml.safety`: eight held-out behavior cases and real AuditKIT precomputed scoring integration. Lexical refusal screens require human interpretation; no external judge is called.
- `vault_ml.comparison`: reads completed M0/M1/M2 artifacts, rejects mismatched suites/configs/adapters/token budgets and refuses a safety-preservation claim.
- `scripts/run_pipeline.py`: priority-ordered synthetic execution, separate processes, immutable command logs and nonzero exit on the first failure.
- `experiments/training/cloud-16gb.json`: unexecuted 16–24 GB Linux GPU fallback; exact base revision retained.

## Actual verification and metrics

Final test discovery: 21 fast tests passed, two opt-in GPU tests skipped. Separate GPU invocation: both tests passed. They use miniature random Cohere2 models in temporary directories and exercise ordinary LoRA and NF4 QLoRA forward/backward, parameter changes and exact adapter reload. They are not substitute VAULT base models. Full command output, including early failures and fixes, is retained under `experiments/environment/`.

Native NF4 primitive diagnostic: finite loss `0.24945829808712006`, forward/backward passed. That scalar is unrelated to scientific performance or Tiny Aya training. Production baseline accuracy, training losses, model VRAM/RAM peaks, training runtime, adapter identifiers and safety metrics are **unavailable**, because gated files could not be loaded. Real AuditKIT unit-test calls passed on labeled test strings; no test string is presented as a model response.

`pip check` passes. Torch remains 2.11.0+cu128, CUDA 12.8 available, RTX 4050 Laptop 6 GB, driver 621.01, Python 3.12.10. Main additions: Transformers 5.21.1, Accelerate 1.15.0, Hub 2.4.0, PEFT 0.21.2, BitsAndBytes 0.50.2, Safetensors 0.8.0, psutil 7.2.2, AuditKIT 1.1.6 at `9da5838e61ef54d560a37605ddbcb19aceded9b3`. Complete exact versions: `requirements-ml-lock.txt`; original installed report: `experiments/environment/final-freeze.txt`.

Official base revision: `af89d219b53ed9b13b8a4645f9c8028973510324`. Weights/tokenizer/config files have not been downloaded. The public metadata and model card were inspected; the public default chat template was format-tested. Cache and all weights/secrets/venv are excluded from Git. New resumption access failures are recorded in `experiments/access/20261108T165643_626514Z/access.json` and `experiments/access/20261108T165708_712540Z/access.json`; both report absent local credentials and gated-access denial, not model inference results.

## Continue after local authentication

In PowerShell:

```powershell
cd D:\VAULT
.\.venv\Scripts\hf.exe auth login
.\.venv\Scripts\python.exe -m vault_ml.infer --request examples/request.json --output experiments/baseline/first-authenticated-response.json
.\.venv\Scripts\python.exe scripts/run_pipeline.py
```

Only after the smoke adapter succeeds, run `scripts/run_pipeline.py --compare --steps 20`. Each M1/M2 trial then processes 5,120 input tokens with the same rank/settings/evaluation suites. Report supervised-token differences and human-review limitations. Do not treat these tiny experiments as statistically meaningful. On a real OOM, allow one reduced-sequence trial before using [the cloud fallback](CLOUD_TRAINING.md). No cloud job has run or been purchased.

## Team handoff

Pawan: use `InferenceEngine(ModelConfig()).infer(InferenceRequest.from_dict(payload))` in a serialized local worker after model access works. Validate caller/source authorization and resolve adapters through an allowlist. See [the proposed integration contract](INTEGRATION_CONTRACT.md). No production backend module exists here and no public endpoint was exposed.

Vijay: provide source version IDs/hashes, explicit units/measurands/assumptions and trusted deterministic verifier receipts. Retain raw records unchanged. The local training generator's approval boolean is synthetic; it does not admit arbitrary external data. Agree production authority, source hashes and split ownership before adding an external import. CuratorKIT export inspection and SafeTune's bounded integration plan are in [the upstream review](UPSTREAM_REVIEW.md).

SafeTune remains incomplete: the inspected SafeGrad path needs explicit alignment data and, for KL mode, a frozen reference with additional memory and validated gradients. No replacement custom algorithm is called SafeTune. M3 is disabled until qualification.

## Git and artifacts

Starting state: only `.venv`, no Git repository, no initial commit or teammate code. The new local Git milestone includes source, docs, synthetic fixtures, dependency reports and command logs. No remote was configured or pushed. Exact final commit is returned in the chat; use `git rev-parse HEAD`, `git log --oneline` and `git status --short` to verify.

Scientific fixture export: `experiments/baseline/20261108T163941_736282Z/fixtures.json` (`model_run: false`). Auth-denied smoke log: `experiments/environment/model-smoke-attempt.log`. Negative offline pipeline run: `experiments/pipeline/20261108T164645_172331Z/`. These are discovery/validation records, not successful model experiments. Full chronology: [experiment ledger](EXPERIMENT_LEDGER.md).
