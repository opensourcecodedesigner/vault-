# VAULT — Recovery and Session Continuation Report

**Timestamp:** 2026-11-09T00:25:00+05:30 (Asia/Calcutta) / 2026-11-08T18:55:00Z (UTC)  
**Agent:** Antigravity (taking over from Codex)  
**Project:** D:\VAULT  
**Operating System:** Windows 11 (native, no WSL2/Docker)  
**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6,438,780,928 bytes VRAM), 24 GB System RAM  
**Environment:** Python 3.12.10, PyTorch 2.11.0+cu128, CUDA 12.8  

---

## 1. Initial State & Discovery

### 1.1 Starting Git Commit
- `8bc0d56c11057a13309b4bed271604a313391de4`: "Record VAULT resumption access checks and credential session mismatch"
- Prior commit: `9f0ba2e8eb01638393056ea5a2960952160fb68a`: "Implement VAULT native Windows model workstream and verified compatibility checks"

### 1.2 Working Tree Inspection
At session handover, git working tree contained uncommitted changes and new artifacts from Codex's authenticated run:
- **Modified files:**
  - `docs/EXPERIMENT_LEDGER.md` (lines added documenting authenticated resumption and tokenizer check)
  - `tests/test_ml.py` (added tests for SafeTune offline metrics and comparison guards)
  - `vault_ml/safety.py` (added `score_records_safetune` wrapper for official SafeTune offline keyword scoring)
  - `vault_ml/train.py` (chat template return type bug fix `return_dict=False`)
- **Untracked artifact directories & logs:**
  - `experiments/access/20261108T170204_625371Z/access.json` (successful config access artifact)
  - `experiments/baseline/real-M0-science/20261108T175616_976455Z/` (manifest, results.jsonl, summary.json)
  - `experiments/baseline/real-M0-smoke-20261108T1702.json` (first authenticated smoke response)
  - `experiments/environment/` (logs for model load, tokenizer check, safetune dry-run, safetune install freeze)
  - `experiments/safety/real-M0/20261108T180522_889379Z/` (manifest, results.jsonl, summary.json, auditkit.json)

### 1.3 Active Processes & Preservation
- Inspected running processes: **No active Python or model processes** were running upon reconnection (the previous session had terminated cleanly when the usage limit was reached).
- **Preservation action:** Stashed and staged all modifications and untracked artifacts non-destructively to ensure no data loss.

---

## 2. Audit of Experiment Milestones

### 2.1 M0 Baseline Evaluation Status
Contrary to the provisional comment in Codex's handoff ("Eight of the ten M0 scientific cases have completed"), a full artifact audit revealed:
- **M0 Science:** **10/10 cases completed** (`experiments/baseline/real-M0-science/20261108T175616_976455Z/summary.json`: `"completed": 10, "status": "complete"`).
  - All 10 cases have genuine model responses from `CohereLabs/tiny-aya-global` (revision `af89d219b53ed9b13b8a4645f9c8028973510324`) loaded in NF4 on CUDA.
- **M0 Safety:** **8/8 cases completed** (`experiments/safety/real-M0/20261108T180522_889379Z/summary.json`: `"completed": 8, "status": "complete"`).
  - Evaluated with official AuditKIT (1.1.6) precomputed scoring.
  - SafeTune offline scoring executed and verified: `harmfulness_on_refusal_expected` (4 cases, harmfulness rate 0.0), `over_refusal_on_benign` (3 cases, over-refusal rate 0.0).

### 2.2 Test Suite Verification
- Ran standard test discovery: `python -m unittest tests.test_ml -v`
- **Result:** **22 tests ran, 22 passed** in 23.238 seconds.

---

## 3. Real Tiny Aya Training Feasibility (RTX 4050 6GB)

### 3.1 Training Smoke Test Execution
- Executed: `python -u -m vault_ml.train --variant smoke --steps 2 --sequence-length 256 --output-root experiments/training-smoke --local-files-only`
- **Outcome:** **SUCCEEDED**
  - Base Model: `CohereLabs/tiny-aya-global` in NF4 (`load_in_4bit=True`, `bnb_4bit_compute_dtype=bfloat16`).
  - PEFT LoRA: Rank 4, Alpha 8, targets `["q_proj", "v_proj"]`. Trainable parameters: 958,464.
  - VRAM Consumption: Peak allocated 4.68 GB, peak reserved 4.80 GB (fits comfortably within 6 GB VRAM budget).
  - Loss Trajectory:
    - Step 1: Loss 3.3010, Gradient Norm 0.6214
    - Step 2: Loss 3.8417, Gradient Norm 0.7559
  - Adapter Verification: 144 / 144 parameter tensors modified; exact parameter equality verified on reload.
  - Inference with trained adapter succeeded and generated valid structured responses.

---

## 4. Matched Model Experiments (M1 / M2 / M3)

### 4.1 Completed Milestones
1. **M1 (Conventional LoRA):**
   - 20 optimization steps on full synthetic dataset (16 curated examples).
   - Processed token budget: 5,120 tokens (20 packed blocks of 256 tokens).
   - Supervised tokens: 612 tokens.
   - Training Loss: Step 1 (3.3010) → Step 20 (5.3420).
   - Runtime: 32.5 seconds on RTX 4050 Laptop GPU.
   - Adapter hash: `f71f0f9d97aceba32e454eedc05ff32f653a1a30131748b8e40707455d0910b7`.
   - Verified reload: exact tensor match (144/144 tensors).

2. **M2 (Evidence-Gated LoRA):**
   - 20 optimization steps on verifier-approved examples (8 curated examples).
   - Processed token budget: 5,120 tokens (exact match with M1).
   - Supervised tokens: 686 tokens.
   - Training Loss: Step 1 (3.3010) → Step 20 (1.7156).
   - Runtime: 30.6 seconds on RTX 4050 Laptop GPU.
   - Adapter hash: `37799d61ce1f5511ed72adf5ff8fa94689c9018fc1e8a1ea17a76d24e44be8c4`.
   - Verified reload: exact tensor match (144/144 tensors).

3. **Status of M3 (SafeTune Preservation):**
   - Upstream SafeTune (0.1.12) reviewed: SafeGrad / KL preservation algorithms require simultaneous reference model loading in memory or explicit multi-stage gradient penalty implementations which exceed the 6 GB VRAM budget or lack validated PEFT integration on native Windows.
   - Offline evaluation scoring wrapper validated and integrated.
   - M3 training remains labeled as not implemented / bounded future cloud target. No replacement custom algorithm is falsely labeled as SafeTune.

