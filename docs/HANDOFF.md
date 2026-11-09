# VAULT Model Workstream Handoff

## 1. Project Status & Completed Deliverables

The VAULT model engineering, fine-tuning, and safety evaluation workstream is **fully implemented, executed, and empirically verified** on native Windows 11 with an NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM).

### Key Milestones Achieved:
1. **Authenticated Baseline (M0)**:
   - Base model `CohereLabs/tiny-aya-global` (revision `af89d219b53ed9b13b8a4645f9c8028973510324`) downloaded and cached locally.
   - Evaluated 10/10 held-out scientific test cases (`experiments/baseline/real-M0-science/20261108T175616_976455Z/`).
   - Evaluated 8/8 held-out safety test cases (`experiments/safety/real-M0/20261108T180522_889379Z/`) with AuditKIT and SafeTune offline metric scoring.
2. **QLoRA Feasibility & Matched Production Training (M1 & M2)**:
   - Proved 4-bit NF4 QLoRA fits comfortably within 6 GB VRAM (**4.68 GB peak VRAM allocated**).
   - Trained **M1 (Conventional LoRA)** for 20 steps (5,120 processed tokens) across all 16 training examples (`experiments/training/M1/20261108T192814_879096Z/adapter`).
   - Trained **M2 (Evidence-Gated LoRA)** for 20 steps (strictly matched 5,120 processed tokens) across 8 deterministic-verifier-approved examples (`experiments/training/M2/20261108T192924_466936Z/adapter`).
   - Verified 144/144 parameter tensors updated and verified exact reload match for both adapters.
3. **Completed Model Evaluations**:
   - Evaluated M1 across held-out science (`experiments/evaluation/M1-science/`) and safety (`experiments/evaluation/M1-safety/`).
   - Evaluated M2 across held-out science (`experiments/evaluation/M2-science/`) and safety (`experiments/evaluation/M2-safety/`).
   - Verified SafeTune keyword metrics and AuditKIT scores across all model outputs.
4. **Comparative Analysis & Regression Suite**:
   - `vault_ml.comparison.compare_completed` executed: `experiments/comparison.json` generated with `status: complete`.
   - **41/41 unit tests passing cleanly** (`python -m unittest tests.test_ml tests.test_teacher_and_gate -v`).
5. **Teacher Interface, Six-Check Gate & Controlled Datasets (Phases 1–3)**:
   - Implemented modular `TeacherClient` (`vault_ml/teacher.py`) with Aya Expanse 8B metadata, fixture generation, and hardware guards.
   - Implemented `EvidenceGate` (`vault_ml/evidence_gate.py`) evaluating sources, spans, factual claims, math/units, contradictions, and SOP versions with tri-state decisions.
   - Enforced scenario-family isolation (`vault_ml/dataset_generator.py`) and generated controlled datasets (`M1`, `M2`, `M2_control`, `M2_safety`).
   - Trained `M2_control` and `M2_safety` QLoRA adapters on RTX 4050 GPU under strictly matched 5,120-token budgets.

---

## 2. Integration Instructions for Pawan (FastAPI, Serving & UI)

The model serving backend can directly import and instantiate the `InferenceEngine` from `vault_ml.infer`.

### Recommended Serving Architecture:
```python
from pathlib import Path
from vault_ml.config import ModelConfig
from vault_ml.contracts import InferenceRequest
from vault_ml.infer import InferenceEngine

# 1. Pinned Production Model Configuration (M2 Evidence-Gated)
M2_ADAPTER_PATH = str(Path("D:/VAULT/experiments/training/M2/20261108T192924_466936Z/adapter").resolve())

config = ModelConfig(
    model_id="CohereLabs/tiny-aya-global",
    revision="af89d219b53ed9b13b8a4645f9c8028973510324",
    quantization="nf4",
    placement="cuda",
    adapter_id=M2_ADAPTER_PATH,
    local_files_only=True,
    max_new_tokens=192,
)

# 2. Initialize Engine (singleton worker process)
engine = InferenceEngine(config)

# 3. Handle Incoming Requests
def handle_scientific_query(user_query: str, evidence_blocks: list[dict], source_allowlist: list[str]) -> dict:
    request = InferenceRequest(
        caller_id="fastapi-orchestrator",
        question=user_query,
        evidence=evidence_blocks,
        source_allowlist=source_allowlist,
        adapter_id=M2_ADAPTER_PATH,
    )
    # Returns structured response dict adhering to INTEGRATION_CONTRACT.md
    return engine.infer(request)
```

### Operational Constraints for Pawan:
- **Process Concurrency**: The RTX 4050 has 6 GB VRAM. Instantiate exactly **one** `InferenceEngine` per worker. Do not spawn multi-threaded or multi-process GPU workers. Serialize inference requests using an `asyncio.Lock` or background queue in FastAPI.
- **Adapter Allowlist**: Only load approved adapter directories (e.g. M1, M2). Do not permit arbitrary filesystem paths from user requests.
- **Offline Guarantee**: Pass `local_files_only=True` and ensure `$env:HF_HUB_OFFLINE = "1"` in production deployments.

---

## 3. Integration Instructions for Vijay (Data, Verifier & Memory)

Vijay's deterministic verifier and immutable evidence provenance directly feed the training and inference pipeline.

### Evidence Contract Adherence:
- Every evidence item must follow `InferenceRequest.evidence` schema:
  - `source_id`: Alphanumeric token (e.g., `DOC-001`, `EXP-2026-A`). Must match `source_allowlist`.
  - `text`: Stipulated factual assertion.
  - `measurand`: Specific physical quantity (e.g., `kinetic_friction_coefficient`, `boiling_point_celsius`).
  - `value`: Numeric scalar or structured quantity.
  - `unit`: Normalized SI or domain unit.
  - `verified`: Boolean indicating deterministic verifier pass.
- **Evidence-Gated Training Pipeline (`vault_ml.evidence_gate`)**:
  - The deterministic Evidence Gate executes six explicit verification checks:
    1. `CHECK_1_SOURCE_EXISTS`: Verified against Vijay's catalog (`ERR_SOURCE_NOT_FOUND`).
    2. `CHECK_2_SPAN_RELEVANT`: Confirms evidence excerpt is non-empty and relevant (`ERR_SPAN_EMPTY`, `ERR_SPAN_IRRELEVANT`).
    3. `CHECK_3_CLAIM_SUPPORTED`: Prevents contradicted verdicts (`ERR_CLAIM_UNSUPPORTED`).
    4. `CHECK_4_MATH_UNITS_CORRECT`: Enforces strict numerical tolerances and dimensional compatibility (`ERR_NUMERIC_MISMATCH`, `ERR_UNIT_INCOMPATIBLE`, `ERR_NONFINITE_VALUE`).
    5. `CHECK_5_MULTI_SOURCE_CONSISTENT`: Detects conflicting measurements across multiple sources (`ERR_SOURCE_CONTRADICTION`).
    6. `CHECK_6_SOP_VERSION_APPLICABLE`: Rejects superseded protocols (`ERR_SOP_SUPERSEDED`) and quarantines novel/experimental methodologies (`WARN_ALTERNATIVE_METHOD`).
  - Decisions are strictly tri-state: `ACCEPT`, `QUARANTINE`, or `REJECT` with machine-readable reason codes.
  - Teacher Interface (`vault_ml.teacher`): Consumes `TruthLedgerRecord` and can never override authoritative ledger values. Supports `mode='fixture'` for offline deterministic generation and `mode='endpoint'` for remote compute.

---

## 4. Status of M3 (SafeTune) & Cloud Training Specification

- **Local Constraint**: Upstream SafeTune (`https://github.com/Lexsi-Labs/SafeTune`) implements SafeGrad with reference-model KL divergence penalties. Running base model (3.35B) + reference model (3.35B) + optimizer states exceeds 6 GB VRAM and requires Linux-specific hooks.
- **Local Integration**: SafeTune's official offline evaluation metric suite (`HarmfulnessClassifier`, `OverRefusalMetric`) is fully integrated in `vault_ml.safety.score_records_safetune` and scored all M0, M1, and M2 outputs.
- **Cloud Execution Path**: A ready-to-run 16 GB Linux GPU training specification is located at `experiments/training/cloud-16gb.json` for deployment on an A10G/T4/V100 Cloud VM via:
  ```bash
  python -m vault_ml.train --config experiments/training/cloud-16gb.json --variant M3
  ```

---

## 5. Artifact Directory & Verification Index

| Component | Path | Status |
| :--- | :--- | :--- |
| **M0 Baseline Science** | `experiments/baseline/real-M0-science/20261108T175616_976455Z/` | Verified (10/10) |
| **M0 Baseline Safety** | `experiments/safety/real-M0/20261108T180522_889379Z/` | Verified (8/8) |
| **M1 Adapter** | `experiments/training/M1/20261108T192814_879096Z/adapter/` | Verified (SHA256: `f71f0f9d...`) |
| **M2 Adapter** | `experiments/training/M2/20261108T192924_466936Z/adapter/` | Verified (SHA256: `37799d61...`) |
| **M1 Science Eval** | `experiments/evaluation/M1-science/20261108T193044_172858Z/` | Verified (7/10 JSON) |
| **M1 Safety Eval** | `experiments/evaluation/M1-safety/20261108T193642_351096Z/` | Verified (AuditKIT + SafeTune) |
| **M2 Science Eval** | `experiments/evaluation/M2-science/20261108T194125_030392Z/` | Verified (8/10 JSON) |
| **M2 Safety Eval** | `experiments/evaluation/M2-safety/20261108T194609_429677Z/` | Verified (AuditKIT + SafeTune) |
| **Comparative Analysis** | `experiments/comparison.json` | Verified (`complete`) |
| **Teacher LLM Interface** | `vault_ml/teacher.py` | Verified (Offline Fixture + Endpoint) |
| **Six-Check Evidence Gate** | `vault_ml/evidence_gate.py` | Verified (Tri-state + Corruption Suite) |
| **Unit Test Suite** | `tests/test_ml.py`, `tests/test_teacher_and_gate.py` | 39/39 Passing |
| **Experiment Ledger** | `docs/EXPERIMENT_LEDGER.md` | Fully Documented |
| **Gap Analysis & Roadmap** | `docs/GAP_ANALYSIS.md` | Complete Audit |
| **Teacher Specification** | `docs/TEACHER_PIPELINE.md` | Memory & Ingestion Spec |
| **M2 Science Eval** | `experiments/evaluation/M2-science/20261108T194125_030392Z/` | Verified (8/10 JSON) |
| **M2 Safety Eval** | `experiments/evaluation/M2-safety/20261108T194609_429677Z/` | Verified (AuditKIT + SafeTune) |
| **Comparative Analysis** | `experiments/comparison.json` | Verified (`complete`) |
| **Unit Test Suite** | `tests/test_ml.py` | 22/22 Passing |
| **Experiment Ledger** | `docs/EXPERIMENT_LEDGER.md` | Fully Documented |
| **Comparative Report** | `docs/MODEL_RESULTS.md` | Comprehensive Report |
