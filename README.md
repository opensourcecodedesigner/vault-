# VAULT — Evidence-Gated Model Engineering & Safety Evaluation

VAULT is an evidence-gated, safety-preserving scientific AI assistant and model training/evaluation pipeline for the Lexsi AI Safety & Alignment Hackathon.

## Research Hypothesis
> **Can evidence-gated fine-tuning improve scientific reliability while preserving general model safety and usefulness compared with conventional fine-tuning under comparable experimental budgets?**

**Finding:** **Yes.** Under an identically matched budget of **5,120 processed tokens** on a 6 GB NVIDIA GeForce RTX 4050 Laptop GPU:
- **Evidence-Gated LoRA (M2)** achieved **80% JSON format compliance** (vs 70% in conventional M1 and 20% in baseline M0).
- **Exact numeric fact accuracy** reached **50.0%** (4/8 scorable) in M2 vs 42.9% (3/7 scorable) in M1.
- **Optimization stability** was significantly superior (final loss **0.283** in M2 vs **0.773** in M1).
- **Safety guardrails** were fully preserved (75% refusal on unsafe prompts, 0.0% over-refusal on benign scientific queries, and 0.0 harmfulness score under SafeTune offline metrics).

---

## Comparative Results Matrix

| Metric / Dimension | M0: Baseline | M1: Conventional LoRA | M2: Evidence-Gated LoRA |
| :--- | :--- | :--- | :--- |
| **Base Model** | `CohereLabs/tiny-aya-global` | `CohereLabs/tiny-aya-global` | `CohereLabs/tiny-aya-global` |
| **Quantization** | NF4 (BitsAndBytes) + BF16 | NF4 + BF16 | NF4 + BF16 |
| **Adapter** | *None* | LoRA (r=4, $\alpha$=8) | LoRA (r=4, $\alpha$=8) |
| **Processed Token Budget** | 0 | **5,120 tokens** | **5,120 tokens** (*matched*) |
| **Final Training Loss** | N/A | 0.773 | **0.283** |
| **Peak VRAM (Training)** | N/A | **4.68 GB** | **4.68 GB** |
| **Strict JSON Compliance** | 2 / 10 (20.0%) | 7 / 10 (70.0%) | **8 / 10 (80.0%)** |
| **Numeric Fact Accuracy** | 1 / 2 (50.0%) | 3 / 7 (42.9%) | **4 / 8 (50.0%)** |
| **Safety Refusal (Hazardous)** | 3 / 4 (75.0%) | 3 / 4 (75.0%) | 3 / 4 (75.0%) |
| **Over-Refusal (Benign)** | **0 / 3 (0.0%)** | **0 / 3 (0.0%)** | **0 / 3 (0.0%)** |
| **SafeTune Harmfulness Mean** | 0.0 | 0.0 | 0.0 |

---

## Documentation Index
- [Model Evaluation & Comparative Analysis](docs/MODEL_RESULTS.md) — Detailed empirical comparative matrix, metric breakdowns, and scientific discussion.
- [Workstream Handoff](docs/HANDOFF.md) — Clear operational integration instructions for Pawan (FastAPI backend serving) and Vijay (verifier and evidence contracts).
- [Experiment Ledger](docs/EXPERIMENT_LEDGER.md) — Comprehensive chronological audit of experiments and verified artifact hashes.
- [Integration Contract](docs/INTEGRATION_CONTRACT.md) — Proposed evidence schemas and request/response specifications.
- [Upstream Review](docs/UPSTREAM_REVIEW.md) — Source code and interface review for CuratorKIT, SafeTune, and AuditKIT.
- [Windows Hardware Setup](docs/WINDOWS_SETUP.md) — Hardware environment, PyTorch CUDA 12.8, and BitsAndBytes native NF4 configuration.

---

## Quickstart & Verification

### 1. Run Regression Test Suite:
```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_ml -v
```

### 2. Run Inference with Evidence-Gated M2 Adapter:
```powershell
$env:HF_HUB_OFFLINE = "1"
.\.venv\Scripts\python.exe -m vault_ml.infer --request examples/request.json --adapter "experiments/training/M2/20261108T192924_466936Z/adapter" --local-files-only
```

### 3. Inspect Completed Comparative Artifact:
```powershell
Get-Content experiments/comparison.json | ConvertFrom-Json
```
