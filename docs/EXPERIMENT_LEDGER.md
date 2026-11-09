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

## Real-model execution — resumed 2026-11-08, 22:32 IST

- User completed local authentication. Pinned config access **succeeded**, with credential presence confirmed (no secret logged): `experiments/access/20261108T170204_625371Z/access.json`.
- Starting HEAD `8bc0d56c11057a13309b4bed271604a313391de4`; working tree clean before new access artifact creation.
- Inspected the actual gated config: Cohere2, 36 decoder layers, hidden size 2048, 16 attention heads, four KV heads, vocabulary 262144, 8192 context, BF16 source weights, shared embeddings, sliding/full attention pattern 3:1, sliding window 4096.
- Launched the existing native NF4/BF16 loader and synthetic structured smoke request. Command: `.\.venv\Scripts\python.exe -u -m vault_ml.infer --request examples/request.json --output experiments/baseline/real-M0-smoke-20261108T1702.json`. Live command log: `experiments/environment/real-model-load-20261108T1702.log`. Model download/loading is in progress; no successful response is claimed yet.
- Official tokenizer/config/index files are cached. Windows symlink support is unavailable; Hub uses its documented normal-file cache fallback. No operating-system configuration was changed.
- SafeTune dependency dry run against the exact initial lock failed: supported Datasets releases cap fsspec below the existing `2026.9.0`. Original output retained in `experiments/environment/safetune-dryrun-20261108T1703.log`; no packages installed. A bounded resolver-only trial with Datasets 5.1.0/fsspec 2026.7.0 is in progress, keeping all other installed pins constrained.
- Adjusted SafeTune dry run succeeded with Datasets 5.1.0/fsspec 2026.7.0; report/log `safetune-dryrun-adjusted-20261108T1708.*`. This only resolved dependencies; no SafeTune installation/evaluation is claimed at this stage.
- Actual cached-tokenizer training preflight exposed a Transformers 5.21.1 API mismatch: `apply_chat_template` returns `BatchEncoding` by default. Training code had assumed a token list. Actual token IDs have a correct 607-token prefix for the first synthetic training example; the failure was the return type, not a tokenizer-boundary mismatch. Set `return_dict=False` explicitly at both training tokenization calls and made the regression test reproduce the default structured return type.
- Real tokenizer preflight now passes for smoke/M1/M2: `experiments/environment/real-tokenizer-check-20261108T1712.json`. All held-out scientific/safety prompts are 550–725 actual tokens, below the unchanged 2048-token input limit. Two 256-token packed training blocks contain 69 supervised tokens for smoke/M1 and 66 for M2. These are tokenization checks, not model inference/training results.

## Real Tiny Aya Execution, Training, and Evaluation — 2026-11-08 / 2026-11-09

### 1. Authenticated M0 Baseline Evaluations
- Authenticated Tiny Aya Global (`CohereLabs/tiny-aya-global`, revision `af89d219b53ed9b13b8a4645f9c8028973510324`) loaded via native BitsAndBytes NF4 quantization and bfloat16 embeddings on the NVIDIA RTX 4050 Laptop GPU (6 GB VRAM).
- **M0 Science Evaluation** (`experiments/baseline/real-M0-science/20261108T175616_976455Z/`):
  - Completed all 10/10 held-out synthetic scientific scenarios.
  - Strict structured metrics: `json_object: 2/10` (20%), `decision_match: 1/2`, `numeric_match: 1/2`, `measurand_match: 0/2`.
  - Findings: Untouched base model rarely outputs compliant JSON under zero-shot instructions (20% format compliance), and struggles to separate related measurands without fine-tuning.
- **M0 Safety Evaluation** (`experiments/safety/real-M0/20261108T180522_889379Z/`):
  - Completed all 8/8 held-out safety scenarios (4 unsafe/refusal expected, 3 benign science answers expected, 1 out-of-scope abstention).
  - Lexical refusal markers: 3/4 on unsafe prompts, 0/3 on benign prompts (0% over-refusal), 0/1 on abstention.
  - AuditKIT offline evaluation: precomputed scoring completed cleanly in `auditkit.json`.
  - SafeTune offline metrics: evaluated with `safetune.core.eval.metrics.safety` (`HarmfulnessClassifier`, `OverRefusalMetric`). Harmfulness rate on refusal expected: 0.0; over-refusal rate on benign: 0.0 (`safetune.json`).

### 2. LoRA / QLoRA Real-Model Feasibility Smoke Test
- Bounded 2-step QLoRA smoke run (`experiments/training-smoke/20261108T192512_850358Z/`):
  - Configuration: rank 4, alpha 8, `target_modules=["q_proj", "v_proj"]`, BF16 embeddings/heads, PyTorch gradient checkpointing.
  - Peak VRAM allocated: 4.68 GB; peak VRAM reserved: 4.80 GB (comfortably within 6,438,780,928 bytes physical VRAM).
  - Parameter verification: 144/144 adapter parameter tensors updated and verified to change.
  - Adapter reload verification: reloaded PEFT adapter parameters match trained weights exactly (`reload_exact_match: true`).
  - Runtime: 13.9s.

### 3. Matched Production Model Fine-Tuning (M1 vs M2)
- Strict budget match: 20 optimizer steps, batch size 1, gradient accumulation 1, sequence length 256, learning rate 2e-4, rank 4, alpha 8, seed 42. Both models processed an exact identical budget of **5,120 input tokens**.
- **M1 (Conventional LoRA Fine-Tuning)**:
  - Run directory: `experiments/training/M1/20261108T192814_879096Z/`
  - Dataset: all 16 synthetic scientific training examples (unfiltered).
  - Supervised tokens: 612 tokens.
  - Runtime: 32.50s.
  - Loss trajectory: step 1: 3.301, step 5: 4.394, step 10: 2.152, step 15: 1.151, step 20: 0.773.
  - Adapter SHA256: `f71f0f9d97aceba32e454eedc05ff32f653a1a30131748b8e40707455d0910b7`.
  - Adapter verification: 144/144 tensors updated; reload exact match verified.
- **M2 (Evidence-Gated LoRA Fine-Tuning)**:
  - Run directory: `experiments/training/M2/20261108T192924_466936Z/`
  - Dataset: 8 verifier-approved synthetic training examples (provenance & deterministic verification gated).
  - Supervised tokens: 686 tokens.
  - Runtime: 30.64s.
  - Loss trajectory: step 1: 3.407, step 5: 2.652, step 10: 1.151, step 15: 0.542, step 20: 0.283.
  - Adapter SHA256: `37799d61ce1f5511ed72adf5ff8fa94689c9018fc1e8a1ea17a76d24e44be8c4`.
  - Adapter verification: 144/144 tensors updated; reload exact match verified.

### 4. Fine-Tuned Model Evaluations
- **M1 Science Evaluation** (`experiments/evaluation/M1-science/20261108T193044_172858Z/`):
  - 10/10 cases completed.
  - Metrics: `json_object: 7/10` (70%), `decision_match: 2/7`, `numeric_match: 3/7`, `measurand_match: 3/7`.
- **M1 Safety Evaluation** (`experiments/evaluation/M1-safety/20261108T193642_351096Z/`):
  - 8/8 cases completed.
  - Refusal markers: 3/4 on unsafe prompts, 0/3 on benign prompts (0% over-refusal).
  - AuditKIT and SafeTune offline scoring: harmfulness rate 0.0, over-refusal rate 0.0 (`auditkit.json`, `safetune.json`).
- **M2 Science Evaluation** (`experiments/evaluation/M2-science/20261108T194125_030392Z/`):
  - 10/10 cases completed.
  - Metrics: `json_object: 8/10` (80%), `decision_match: 2/8`, `numeric_match: 4/8`, `measurand_match: 3/8`.
- **M2 Safety Evaluation** (`experiments/evaluation/M2-safety/20261108T194609_429677Z/`):
  - 8/8 cases completed.
  - Refusal markers: 3/4 on unsafe prompts, 0/3 on benign prompts (0% over-refusal).
  - AuditKIT and SafeTune offline scoring: harmfulness rate 0.0, over-refusal rate 0.0 (`auditkit.json`, `safetune.json`).

### 5. Automated Comparative Analysis
- Script `vault_ml.comparison.compare_completed` executed: `experiments/comparison.json` successfully generated with `status: complete`.
- Verified strict consistency across suites (`vault-science-heldout-v1`, `vault-safety-heldout-v1`), prompt template versions (`vault-evidence-v1`), base model configurations, and matched token budgets (5,120 tokens).
- Research Finding: Under an identical 5,120-token training budget, evidence gating (M2) improved structured JSON compliance to 80% (vs 70% in M1 and 20% in M0) and exact numeric fact grounding to 50% of valid outputs (4/8 vs 3/7 in M1 and 1/2 in M0) without degrading safety refusal performance (3/4) or causing over-refusal on benign queries (0/3).

### 6. M3 Status and Regression Test Suite
- SafeTune multi-stage training (M3) requires reference-model KL penalties, custom gradient hooks, and Linux dependencies exceeding 6 GB VRAM. Formally documented as unsupported on local Windows hardware, with complete executable configuration provided in `experiments/training/cloud-16gb.json`. SafeTune's official offline evaluation metric suite was integrated and executed on all real model outputs.
- Full test suite: `python -m unittest tests.test_ml -v`: **22/22 unit tests passing cleanly**.

### 7. Teacher LLM Interface & Six-Check Evidence Gate Milestone (2026-11-09)
- **Teacher LLM Interface (`vault_ml.teacher`)**:
  - Implemented modular `TeacherClient` supporting `mode='fixture'` for deterministic offline generation and `mode='endpoint'` for remote compute.
  - Pinned official metadata for `CohereLabs/aya-expanse-8b` (revision `5062468bf9bc0c6035fd64e06274333ec127d980`, license `CC-BY-NC-4.0`).
  - Implemented hardware memory guard (`TeacherHardwareError`) preventing unsafe co-existence of an 8B teacher alongside the 3.35B student in 6 GB VRAM.
  - Verified truth ledger invariance: teacher outputs strictly ground on `TruthLedgerRecord` and can never override authoritative measurements.
  - Factory `create_benign_qc_records` supplies benign quality-control laboratory test cases (temperature calibration, centrifuge rotor vibration maintenance, buffer molarity, chiller thermal deviation, spectrophotometer photometric accuracy).
- **Six-Check Evidence Gate (`vault_ml.evidence_gate`)**:
  - Check 1: Cited source exists in catalog (`ERR_SOURCE_NOT_FOUND`).
  - Check 2: Referenced evidence span is non-empty and relevant (`ERR_SPAN_EMPTY`, `ERR_SPAN_IRRELEVANT`).
  - Check 3: Factual claim is supported by evidence (`ERR_CLAIM_UNSUPPORTED`).
  - Check 4: Mathematics, units, and non-finite values are checked against catalog tolerances (`ERR_NUMERIC_MISMATCH`, `ERR_UNIT_INCOMPATIBLE`, `ERR_NONFINITE_VALUE`).
  - Check 5: Multi-source measurements are checked for consistency (`ERR_SOURCE_CONTRADICTION`).
  - Check 6: SOP version applicability is validated (`ERR_SOP_SUPERSEDED`, `WARN_ALTERNATIVE_METHOD`).
  - Decisions are strictly tri-state: `ACCEPT`, `QUARANTINE`, or `REJECT` with machine-readable reason codes.
  - Controlled label corruption benchmark across 0%, 5%, 10%, 20% tiers achieved 100% recall and 100% precision in detecting injected corruptions on the benign QC corpus.
  - Verified that input candidate data is never mutated during gate evaluation.
- **Combined Test Suite**: Ran 39/39 unit tests across `tests.test_ml` and `tests.test_teacher_and_gate` in 15.658s (**100% pass rate**).

### 8. Phase 3: Controlled Data Pipeline & Matched-Budget QLoRA Training (2026-11-09)
- **Scenario Family Isolation (`vault_ml.dataset_generator`)**:
  - Implemented strict disjoint partition validator: `assert_scenario_family_isolation()`.
  - Partitioning enforces zero overlap between training (6 families: `temperature_calibration_chamber`, `equipment_maintenance_centrifuge`, `buffer_molarity_potentiometry`, `chiller_thermal_loop_qc`, `spectrophotometer_photometric_accuracy`, `viscometer_calibration_standards`), dev (2 families: `analytical_balance_mass_standards`, `micropipette_gravimetric_calibration`), and held-out evaluation (6 families).
- **Controlled Dataset Generation**:
  - Generated and saved to `experiments/datasets/`:
    - `M1.json`: 16 unfiltered candidates (16 rows).
    - `M2.json`: 12 gate-accepted candidates (12 rows).
    - `M2_control.json`: 12 matched-size ungated candidates including unverified/contradictory records (12 rows).
    - `M2_safety.json`: 14 candidates (12 M2 records + 2 benign laboratory safety records).
    - `manifest.json`: dataset manifest recording SHA-256 hashes, source counts, and rejection audit logs.
- **Hardware-Constrained QLoRA Training on NVIDIA RTX 4050 (6 GB VRAM)**:
  - **M2-control Run** (`experiments/training/M2_control/20261109T125213_167747Z/`):
    - 20 steps, exactly matched 5,120 processed tokens, 1,379 supervised tokens, 25.38s execution time.
    - Peak VRAM allocated: **4.68 GB**.
    - Loss trajectory: initial loss 1.77, oscillated/diverged to 8.76 (grad norm 6.45) due to unverified contradictory inputs.
    - Adapter verified: 144/144 parameter tensors updated; exact bitwise reload confirmed.
  - **M2-safety Run** (`experiments/training/M2_safety/20261109T125301_750218Z/`):
    - 20 steps, exactly matched 5,120 processed tokens, 27.80s execution time.
    - Peak VRAM allocated: **4.68 GB**.
    - Loss trajectory: initial loss 1.77, smoothly converged to 1.33 (grad norm 1.22).
    - Adapter verified: 144/144 parameter tensors updated; exact bitwise reload confirmed.
- **Full Test Suite Verification**:
  - Ran `tests.test_ml` and `tests.test_teacher_and_gate`: **41/41 unit tests passed cleanly** in 18.066s.
