# VAULT Teacher LLM Integration Specification

**Component:** `vault_ml.teacher`  
**Planned Teacher Model:** `CohereLabs/aya-expanse-8b`  
**Model Revision:** `5062468bf9bc0c6035fd64e06274333ec127d980`  
**License:** Creative Commons Attribution-NonCommercial 4.0 International (`CC-BY-NC-4.0`)  
**Pipeline Role:** High-capacity multilingual teacher for generating realistic scientific questions, laboratory context, and structured CoT explanations from verified truth-ledger records.

---

## 1. Hardware Memory Budget & Concurrent Co-existence Guard

The local development and testing environment is an **NVIDIA GeForce RTX 4050 Laptop GPU with 6,438,780,928 bytes (~6.0 GB) of physical VRAM**.

### Memory Feasibility Analysis:
- **Tiny Aya Global (3.35B Student):** Occupies **2.56 GB** in NF4 4-bit quantization, with peak allocated training VRAM of **4.68 GB** when allocating LoRA adapter weights, bfloat16 embeddings, and gradient checkpointing activation buffers.
- **Aya Expanse 8B (8B Teacher):**
  - Full Precision / BF16: $8 \times 10^9 \times 2 \text{ bytes} \approx 16.0 \text{ GB}$ VRAM.
  - 4-bit Quantization (NF4): $\approx 4.8 \text{ GB}$ base weights alone, plus $>2.0 \text{ GB}$ for attention KV caches and activations at 2048 sequence length.
- **Concurrent Co-existence Result:** Attempting to host both the 3.35B student model and the 8B teacher model concurrently on a 6 GB VRAM GPU will cause immediate CUDA Out-Of-Memory (`CUDA out of memory`).

### Hardware Guard:
`TeacherClient.verify_local_hardware_budget()` programmatically inspects physical VRAM. If local GPU VRAM is $<14 \text{ GB}$, it blocks attempts to instantiate the 8B teacher locally, directing the pipeline to either:
1. `mode="endpoint"`: Calling an external or separate permitted compute endpoint (e.g., vLLM or Ollama serving Aya Expanse 8B on an A10G/V100/H100 instance).
2. `mode="fixture"`: Deterministic offline generation consuming authoritative truth-ledger records, explicitly flagged with `generator: "deterministic_fixture_non_llm"`.

---

## 2. Ingestion from Truth Ledger (Vijay's Workstream)

The teacher pipeline strictly consumes verified records from the scientific Truth Ledger:

```python
@dataclass(frozen=True)
class TruthLedgerRecord:
    ledger_id: str
    scenario_family: str
    experiment_id: str
    measurand: str
    verified_value: Optional[float]
    unit: Optional[str]
    decision: str  # 'pass', 'flagged', 'inconclusive'
    evidence_text: str
    source_id: str
    sop_version: str
    assumptions: List[str]
```

**Cardinal Rule:** The teacher LLM *never* invents the scientific verdict, numerical measurement, or SOP status. It only formats naturalistic laboratory notes, context framing, and structured rationales that adhere to the verified verdict.

---

## 3. Teacher Output Contract & Provenance

Every generated candidate produces a `TeacherOutput` record:
```json
{
  "record_id": "TCH-LEDGER-001",
  "ledger_id": "LEDGER-001",
  "scenario_family": "biochemistry",
  "candidate_question": "Based on the verification of EXP-101, what is the confirmed michaelis constant km um under SOP-BIO-v2?",
  "lab_notes": "Laboratory protocol SOP-BIO-v2: Experiment EXP-101. Observed excerpt: \"Michaelis constant Km determined to be 12.4 uM by Lineweaver-Burk fit.\"",
  "explanation": "According to verified document [DOC-001], the michaelis constant km um was measured at 12.4 uM. The deterministic verification passed against SOP SOP-BIO-v2.",
  "evidence_spans": ["Michaelis constant Km determined to be 12.4 uM by Lineweaver-Burk fit."],
  "source_ids": ["DOC-001"],
  "target_response": {
    "decision": "pass",
    "measurand": "michaelis_constant_km_um",
    "correct_value": 12.4,
    "unit": "uM",
    "explanation": "According to verified document [DOC-001]...",
    "citations": ["DOC-001"]
  },
  "metadata": {
    "teacher_model_id": "CohereLabs/aya-expanse-8b",
    "teacher_revision": "5062468bf9bc0c6035fd64e06274333ec127d980",
    "teacher_license": "CC-BY-NC-4.0",
    "prompt_version": "vault-teacher-v1",
    "mode": "fixture",
    "generator": "deterministic_fixture_non_llm"
  },
  "content_hash": "8414063575a51378..."
}
```

---

## 4. Operational Commands

### Generate Dataset in Offline Fixture Mode:
```python
from vault_ml.teacher import TeacherClient, batch_generate_teacher_dataset
client = TeacherClient(mode="fixture")
candidates = batch_generate_teacher_dataset(ledger_records, client, Path("experiments/teacher/candidates.json"))
```

### Connect to Remote Aya Expanse Endpoint:
```python
client = TeacherClient(
    mode="endpoint",
    endpoint_url="https://aya-expanse-worker.internal.net/v1/completions",
    api_key="...",
    max_retries=3
)
```

