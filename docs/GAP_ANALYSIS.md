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
| **Phase 1** | Teacher LLM Integration (Aya Expanse 8B) | **MISSING** | `vault_ml/training_data.py` (has static templates only) | Need modular teacher client: remote, local endpoint, and deterministic offline fixture modes. |
| **Phase 2** | CuratorKIT & Six-Check Evidence Gate | **PARTIAL** | `vault_ml/training_data.py:12-45`, `docs/UPSTREAM_REVIEW.md:23-27` | Currently simple boolean flag. Must implement full 6-check gate with tri-state decisions and corruption suite. |
| **Phase 3** | Training Data & Controlled Experiments | **PARTIAL** | `vault_ml/train.py`, `experiments/training/M1`, `M2` | Only 16 synthetic examples used. Need scenario family isolation, sealed test split, M2-control, and safety mix-in datasets. |
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
- **Status:** **MISSING**.
- **Deficiency:** Currently, training data is emitted by static template functions in `vault_ml/training_data.py`. There is no teacher model client or generator.
- **Architectural Barrier:** Cohere Aya Expanse 8B (`CohereForAI/aya-expanse-8b`) is an 8B parameter model (~16 GB unquantized, ~5.5 GB in 4-bit). Attempting to load an 8B teacher alongside the 3.35B student in 6 GB VRAM will trigger immediate OOM.
- **Solution:** Implement a modular `TeacherClient` supporting:
  1. `endpoint` mode for remote API / local OpenAI-compatible server.
  2. `fixture` mode using deterministic offline truth-ledger fixtures explicitly marked as non-LLM.
  3. Batch generation of candidate questions, notes, and rationales consuming verified truth records.

### 2.3 Six-Check Evidence Gate & CuratorKIT (Phase 2 Target)
- **Status:** **PARTIALLY IMPLEMENTED (STUB ONLY)**.
- **Deficiency:** `selected_examples` in `vault_ml/training_data.py` uses a single `verifier_approved: bool` attribute. It does not perform granular evidence checks or report structured rejection reason codes.
- **Required Implementation:**
  - Check 1: Cited source exists in catalog.
  - Check 2: Referenced evidence span is relevant.
  - Check 3: Claim is logically/factually supported.
  - Check 4: Mathematics, units, and dimensional analysis are correct.
  - Check 5: Multi-source consensus and contradiction detection.
  - Check 6: Applicable Standard Operating Procedure (SOP) version validity.
  - Tri-state decisions: `ACCEPT`, `QUARANTINE`, `REJECT` with machine-readable reason codes.
  - Label corruption test harness (0%, 5%, 10%, 20%).

### 2.4 Training Pipeline & Experimental Controls (Phase 3 Target)
- **Status:** **VERIFIED BASELINE, EXPANSION NEEDED**.
- **Evidence:** `vault_ml/train.py` supports bounded assistant-masked QLoRA with PyTorch gradient checkpointing and exact parameter reload verification.
- **Deficiency:** Need systematic dataset splits isolated by scenario family, separate tracking of supervised vs packed tokens, and two new experimental variants:
  - `M2-control`: Matched-size ungated dataset.
  - `M2-safety`: Safety mix-in control dataset.

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

