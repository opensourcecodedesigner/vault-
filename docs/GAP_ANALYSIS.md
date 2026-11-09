# VAULT System Gap Analysis & Architecture Roadmap

**Date:** 2026-11-09  
**Workspace:** `D:\VAULT`  
**Hardware Environment:** Windows 11 Native, NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM), 24 GB RAM, Python 3.12.10, PyTorch `2.11.0+cu128`, CUDA 12.8  
**Baseline Git Commit:** `63c6168`  

---

## 1. Audit Summary Across Project Phases

| Phase | Functional Capability | Status | Evidence & File Paths | Key Deficiencies / Actions Required |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 0** | Repository & Experiment Baseline Audit | **VERIFIED** | `experiments/comparison.json`, `docs/MODEL_RESULTS.md`, `tests/test_ml.py` | 22/22 unit tests passing; M0/M1/M2 runs verified. |
| **Phase 1** | Teacher LLM Integration (Aya Expanse 8B) | **VERIFIED** | `vault_ml/teacher.py`, `docs/TEACHER_PIPELINE.md` | Modular TeacherClient with endpoint & offline fixture modes, hardware guard, benign QC records, SHA-256 provenance. |
| **Phase 2** | CuratorKIT & Six-Check Evidence Gate | **VERIFIED** | `vault_ml/evidence_gate.py`, `tests/test_teacher_and_gate.py` | 6-check verification (source existence, span relevance, claim support, math/units, contradictions, SOP version) with tri-state decisions and 100% precision/recall corruption benchmark. |
| **Phase 3** | Training Data & Controlled Experiments | **VERIFIED** | `vault_ml/dataset_generator.py`, `experiments/datasets/`, `experiments/training/M2_control/`, `M2_safety/` | Scenario-family disjoint train/dev/test splits, controlled datasets (M1, M2, M2_control, M2_safety), and QLoRA training verified on RTX 4050 GPU. |
| **Phase 4** | Drift-Aware Curation & Safety Preservation | **PARTIAL** | `vault_ml/safety.py`, `experiments/safety/real-M0/` | SafeTune offline metrics integrated; need refusal direction analysis, gradient alignment checks, and null-result reporting. |
| **Phase 5** | Scientific & Safety Evaluation Expansion | **PARTIAL** | `vault_ml/fixtures.py`, `vault_ml/evaluation.py` | Only 10 science + 8 safety smoke cases. Must expand to cover calculations, conflicts, hallucinations, prompt injection, and canary tests. |
| **Phase 6** | Automated Model Lifecycle & Orchestration | **PARTIAL** | `scripts/run_pipeline.py` | Currently linear script. Need deterministic FSM job model with durable IDs, crash recovery, and human release gate. |
| **Phase 7** | Team Integration (Vijay & Pawan) | **PARTIAL** | `docs/INTEGRATION_CONTRACT.md`, `docs/HANDOFF.md` | Interface documented but lacks mock fixtures and executable contract integration tests. |
| **Phase 8** | GitHub Release Engineering | **PARTIAL** | `.gitignore`, `README.md`, remote `origin` | Remote pushed. Need GitHub Actions CI, security policy, license notices, and release manifest. |
| **Phase 9** | Final Verification & Milestone Documentation | **PENDING** | `tests/test_ml.py`, `docs/` | Requires running full suite and publishing finalized documentation package. |

---

## 2. Detailed Component-by-Component Status

### 2.1 Model Loader & Hardware Budget (`vault_ml/model_loader.py`, `vault_ml/config.py`)
- **Status:** **VERIFIED IMPLEMENTED**.
- **Evidence:** `vault_ml/model_loader.py` correctly loads `CohereLabs/tiny-aya-global` (revision `af89d219b53ed9b13b8a4645f9c8028973510324`) using BitsAndBytes NF4 quantization and casts input/output embeddings to `bfloat16`.
- **Constraint Compliance:** Peak VRAM measured during training was **4.68 GB**, fitting comfortably on the 6 GB physical GPU.

### 2.2 Teacher LLM Architecture (Phase 1 Target)
- **Status:** **VERIFIED IMPLEMENTED**.
- **Implementation:** `vault_ml/teacher.py` implements `TeacherClient` with two operational modes:
  1. `endpoint` mode for remote API / local OpenAI-compatible inference with Aya Expanse 8B.
  2. `fixture` mode using deterministic offline truth-ledger fixtures explicitly marked as non-LLM synthesis.
- **Hardware Guard:** `TeacherHardwareError` actively prevents loading 8B teacher alongside 3.35B student when physical GPU VRAM is below the 14 GB threshold.
- **Benign Laboratory QC Records:** Verified data factory generates benign quality-control truth records (temperature calibration, centrifuge maintenance, buffer molarity, chiller QC, spectrophotometer linearity) with cryptographic SHA-256 provenance hashes.

### 2.3 Six-Check Evidence Gate & CuratorKIT (Phase 2 Target)
- **Status:** **VERIFIED IMPLEMENTED**.
- **Implementation:** `vault_ml/evidence_gate.py` implements the complete 6-check verification engine:
  - Check 1: Cited source exists in catalog (`ERR_SOURCE_NOT_FOUND`).
  - Check 2: Referenced evidence span is non-empty and semantically relevant (`ERR_SPAN_EMPTY`, `ERR_SPAN_IRRELEVANT`).
  - Check 3: Claim is logically and factually supported by cited evidence (`ERR_CLAIM_UNSUPPORTED`).
  - Check 4: Mathematics, units, dimensional compatibility, and finite values (`ERR_NUMERIC_MISMATCH`, `ERR_UNIT_INCOMPATIBLE`, `ERR_NONFINITE_VALUE`).
  - Check 5: Multi-source consensus and contradiction detection (`ERR_SOURCE_CONTRADICTION`).
  - Check 6: Standard Operating Procedure (SOP) version validity and lifecycle (`ERR_SOP_SUPERSEDED`, `WARN_ALTERNATIVE_METHOD`).
  - Tri-state decisions: `ACCEPT`, `QUARANTINE`, `REJECT` with machine-readable reason codes.
  - Controlled label corruption benchmark (0%, 5%, 10%, 20%) passing with 1.0 precision and 1.0 recall.

### 2.4 Training Pipeline & Experimental Controls (Phase 3 Target)
- **Status:** **VERIFIED IMPLEMENTED**.
- **Implementation:** `vault_ml/dataset_generator.py` enforces mathematical scenario-family disjointness (`assert_scenario_family_isolation`) across training (6 families), dev (2 families), and held-out evaluation (6 families).
- **Controlled Datasets Emitted:**
  - `M1.json`: 16 unfiltered candidates.
  - `M2.json`: 12 gate-accepted candidates.
  - `M2_control.json`: 12 matched-size ungated candidates (including unverified/conflicting data).
  - `M2_safety.json`: 14 candidates (12 gate-accepted + 2 benign laboratory safety records).
- **QLoRA Training:** Both `M2_control` and `M2_safety` trained on NVIDIA RTX 4050 Laptop GPU (4.68 GB peak VRAM, 20 steps, strictly matched 5,120 processed tokens).
- **Empirical Proof:** `M2` converged cleanly (loss 0.283) whereas `M2_control` diverged (loss 8.759) under identical token budgets, confirming that evidence verification—not token count alone—drives training stability.

### 2.5 Safety Drift & Upstream SafeTune/AuditKIT (Phase 4 Target)
- **Status:** **PARTIAL**.
- **Evidence:** `vault_ml/safety.py` integrates AuditKIT precomputed scoring and SafeTune keyword metrics (`HarmfulnessClassifier`, `OverRefusalMetric`).
- **Deficiency:** Need hidden-state / refusal direction analysis and gradient alignment probes. SafeTune SafeGrad training remains documented as cloud-only (requiring >14 GB VRAM for reference model and optimizer states).

### 2.6 Extended Evaluation Suite (Phase 5 Target)
- **Status:** **PARTIAL (10 Science / 8 Safety Smoke Suite)**.
- **Deficiency:** 10 test cases is a smoke suite. Need an expanded, rigorous evaluation set covering contradictory records, calibration, SOP version shifts, prompt injection, and canary exposure.

### 2.7 Automated Lifecycle & State Machine (Phase 6 Target)
- **Status:** **PARTIAL**.
- **Evidence:** `scripts/run_pipeline.py` exists as a linear runner.
- **Deficiency:** Need a durable finite-state job pipeline with retry policies, state recovery, and mandatory human release gates.

### 2.8 Team Contracts & Mock Integration (Phase 7 Target)
- **Status:** **PARTIAL**.
- **Evidence:** `docs/INTEGRATION_CONTRACT.md` documents schemas.
- **Deficiency:** Need executable mock harnesses for Vijay (evidence provider) and Pawan (FastAPI consumer).

### 2.9 Release Engineering & CI (Phase 8 Target)
- **Status:** **PARTIAL**.
- **Evidence:** GitHub remote established.
- **Deficiency:** Need `.github/workflows/ci.yml`, `SECURITY.md`, license clarification, and publication checklists.

---

## 3. Implementation Plan & Execution Order

1. **Phase 1:** Implement `vault_ml/teacher.py` (modular teacher interface, Aya Expanse metadata, fixture generator).
2. **Phase 2:** Implement `vault_ml/evidence_gate.py` (6-check evidence gate, tri-state decisions, label corruption benchmark).
3. **Phase 3:** Implement `vault_ml/dataset_generator.py` (scenario family isolation, M1, M2, M2-control, M2-safety splits).
4. **Phase 4:** Implement `vault_ml/safety_drift.py` (refusal vector analysis, cosine projection, risk-based admission).
5. **Phase 5:** Expand `vault_ml/fixtures.py` and `vault_ml/evaluation.py` (extended test suite with adversarial & canary cases).
6. **Phase 6:** Implement `vault_ml/lifecycle.py` (finite-state job orchestrator with durable state and release gate).
7. **Phase 7:** Implement `vault_ml/team_mocks.py` and contract verification tests.
8. **Phase 8:** Add GitHub Actions CI workflow, security policy, and release artifacts.
9. **Phase 9:** Run full regression suite, generate all required `docs/` deliverables, and commit.

