# Experiment ledger

All artifacts are local and use synthetic inputs. Existing records are never overwritten.
Timestamped run directories use UTC. User-facing working date: 2026-11-08 Asia/Calcutta.

## Discovery — 2026-11-08

- Starting repository: absent (`git status` and `git rev-parse HEAD` both failed: not a repository).
- Only `.venv` existed. No starter kit, teammate modules, contracts, role instructions or earlier test results found.
- Initialized an empty Git repository. Starting commit: none. No files overwritten.
- Python 3.12.10; torch/torchaudio 2.11.0+cu128; torchvision 0.26.0+cu128; CUDA 12.8 available.
- NVIDIA RTX 4050 Laptop, 6,438,780,928 bytes VRAM; driver 621.01. Approximately 1,135 MiB used by desktop.
- Disk: D: 43,791,400,960 bytes free, C: 91,422,773,248 bytes free at discovery.
- Initial `pip check`: no broken requirements. Starting freeze and NVIDIA output: `experiments/environment/`.
- Official model metadata resolved revision `af89d219b53ed9b13b8a4645f9c8028973510324`.
- Public README downloaded. Config access without a token failed HTTP 401 (gated repository).
- User reports approved model access and will authenticate locally. No inference/training result exists yet.
- Official BitsAndBytes documentation lists native Windows x86-64 CUDA 12.8 wheels and NF4 support.
- Dependency installation is constrained by the complete initial freeze, protecting the working CUDA stack.

## Implementation and verification — 2026-11-08

- Installed 29 justified packages/dependencies into the existing venv. Pip's install report confirms Torch was not modified. Final `pip check`: no broken requirements.
- Exact installed versions and full install output are retained in `experiments/environment/`; pinned reproducible requirements use the inspected AuditKIT Git SHA.
- Native NF4 CUDA primitive: actual forward/backward passed; scalar diagnostic loss `0.24945829808712006`. This is a tiny linear-layer diagnostic, **not** a Tiny Aya training loss.
- Added pinned model loader, greedy inference CLI, placement-aware input handling, resource sampling, strict evidence contract, citation allowlist, immutable experiment output and fail-fast pipeline orchestration.
- Public Hub metadata also exposes the default chat template. Tested that exact template through Transformers with a toy formatting-only vocabulary: system/developer preamble handling and assistant generation marker passed. Actual production tokenizer files remain unavailable without login.
- Added 10 held-out scientific cases, eight safety cases and a separate 16-row synthetic training generator. Scientific ground truth and assumptions are explicit. All model explanations/safety behaviors require manual review.
- Scientific fixture SHA256: `690ff7fe71bac8b54992f5ca0046dcbdc33570d56a8207a8c96fe8c1bd858a46`.
- Safety fixture SHA256: `8e6564805f17a9a6fdbb57a0ad9c947c509bde468e35f72c41cdd7ebf7144bac`.
- Fixtures-only artifact: `experiments/baseline/20261108T163941_736282Z/fixtures.json`; explicitly `model_run: false`.
- Added bounded QLoRA training, loss/resource logging, real adapter change/hash/reload checks, matched M1/M2 processed-token budgets and strict completed-artifact comparison checks. These production-model paths await gated access.
- First fast test run failed on two actual API differences: AuditKIT custom scorers require an explicit direction, and Hub HTTP exceptions require a response object. A subsequent test exposed AuditKIT's default cache location outside the workspace; the integration now sets the supported cache environment variable to the local project cache when unset. Original failure logs are retained; no earlier failed run is marked passed.
- Final fast discovery: **23 tests discovered; 21 passed, 2 GPU tests skipped by design** in `unit-tests-verified.log`.
- Separate opt-in GPU run: **2/2 passed**, ordinary LoRA and NF4 QLoRA on a miniature **random** Cohere2 architecture. Both performed an actual backward/optimizer update, detected changed tensors, saved an adapter and verified exact reloaded tensors. Temporary test models/adapters were removed with their test directory. This establishes API/kernel compatibility only, not official checkpoint feasibility.
- `gpu-compatibility-final.log`: two tests completed in 14.606 seconds. `unit-tests-verified.log`: final discovery completed in 9.974 seconds.
- Official loader smoke attempt with network access returned sanitized gated-access denial, exit code **1**, recorded in `model-smoke-attempt.log`. No inference artifact or model response was fabricated.
- User confirmed they cannot authenticate at present; a subsequent credential-presence check remained false. Download permission is granted, but the gated repository still requires their local credential. User was shown the model page/token-login route (the chat demo is a separate Space).
- Offline pipeline failure-path run `experiments/pipeline/20261108T164645_172331Z`: diagnostics succeeded; M0 inference failed due uncached model files; pipeline stopped with exit code **1** before evaluation/training. This is a negative reliability test, not a completed model pipeline.
- SafeTune/CuratorKIT source reviewed at exact commits in `UPSTREAM_REVIEW.md`; neither is installed or claimed as integrated. AuditKIT's real precomputed scoring path passed local integration unit tests. No real-model safety audit exists.

## Completion boundary

Working local checks: dependency imports, CUDA/NF4 primitive, contracts, official-template formatting, deterministic fixtures, output immutability, evaluation scoring, comparison guards, real AuditKIT API, miniature architecture LoRA/QLoRA save/reload and CLI/pipeline failure handling.

Blocked: official config/tokenizer/weight download, authentic Tiny Aya M0 responses, full-size training feasibility, real Tiny Aya adapters, completed M1/M2 experiments and comparative safety results. No model accuracy, loss trajectory, adapter ID or safety-preservation metric exists for Tiny Aya. Cloud configuration is executable code/configuration only and has not run.

The Git milestone covers this implemented and tested workstream. Starting commit was absent; obtain the final exact commit with `git rev-parse HEAD` (also reported in the chat handoff). No remote push is performed.

## Authenticated-run resumption — 2026-11-08, 22:26–22:28 IST

- Requested starting commit `9f0ba2e8eb01638393056ea5a2960952160fb68a` matched HEAD; working tree was clean before modifications. Read handoff/ledger and reused the existing infrastructure unchanged.
- Torch remains `2.11.0+cu128`; CUDA available; initial free VRAM `5,326,766,080` bytes of `6,438,780,928` total.
- User reports completing Hub authentication and accepting terms. The process credential-presence check returned false. Checked only credential presence/locations, never token contents or values.
- First normal-sandbox pinned-config access encountered DNS denial: `experiments/access/20261108T165628_395116Z/access.json`. This is a sandbox network failure, not evidence about account approval.
- Retried outside the network-restricted sandbox: `experiments/access/20261108T165643_626514Z/access.json` records `GatedRepoError` and absent local credential.
- After the user reported successful `whoami`, repeated the pinned-config check: `experiments/access/20261108T165708_712540Z/access.json` again records absent local credential and `GatedRepoError`.
- Direct invocation of the exact project `hf.exe auth whoami` also reports not logged in. Default `HF_HOME` is `C:\Users\cosmic\.cache\huggingface`; neither `token` nor `stored_tokens` exists there, and no process `HF_TOKEN`, legacy token file, user-level token environment value or project-cache token was found.
- Asked for non-secret credential-path/environment-presence output from the user's successful terminal to resolve this session mismatch. No access-control bypass, alternate model, baseline run or adapter result was attempted or claimed.
