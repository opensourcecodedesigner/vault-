# Native Windows setup and reproduction

Run PowerShell from `D:\VAULT`. Reuse `.venv`; no WSL, Docker, system packages or alternate base model.

## Verified environment

Python 3.12.10, Windows 11 build 26300, RTX 4050 Laptop 6 GB (compute capability 8.9), NVIDIA driver 621.01. PyTorch and torchaudio 2.11.0+cu128, torchvision 0.26.0+cu128. Torch CUDA runtime 12.8. System RAM 25,439,199,232 bytes (~23.69 GiB). Available RAM was ~9.1 GiB at the dependency check, so do not assume all physical RAM is free.

Installed model dependencies: Transformers 5.21.1, Accelerate 1.15.0, Hugging Face Hub 2.4.0, Safetensors 0.8.0, PEFT 0.21.2, BitsAndBytes 0.50.2 (official Windows wheel), psutil 7.2.2, AuditKIT 1.1.6 from commit `9da5838e61ef54d560a37605ddbcb19aceded9b3`.

`experiments/environment/starting-freeze.txt` preserves the initial environment. `final-freeze.txt` is the literal installed report, including AuditKIT's local source path. `requirements-ml-lock.txt` replaces that machine-local path with its exact upstream Git SHA and adds the CUDA wheel index for reproduction. It is a pinned environment report, not a cross-platform hashed lock. The pip install report includes downloaded wheel URLs and hashes. No PyTorch downgrade occurred.

To restore dependencies into the existing working venv:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt -c experiments/environment/starting-freeze.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m vault_ml.diagnostics
```

For exact fresh-machine Windows reproduction with the same CUDA wheel availability, use `pip install -r requirements-ml-lock.txt`; do not apply the Windows lock to Linux. Keep all installs in a virtual environment. No installation command above starts a server.

## Model access and cache

Official model: [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global). Accept its gated noncommercial terms with your own account, then authenticate **locally**:

```powershell
.\.venv\Scripts\hf.exe auth login
.\.venv\Scripts\hf.exe auth whoami
```

Never paste tokens into chat, source files, command arguments, experiment JSON or Git. The loader uses the standard Hub credential store or an existing `HF_TOKEN` environment variable. It never prints the token. Model weights go into `D:\VAULT\.cache\huggingface\hub`, ignored by Git. Public model metadata resolved revision `af89d219b53ed9b13b8a4645f9c8028973510324`; config/tokenizer/weights were inaccessible without authentication. An approved account still needs a local credential with read permission for this gated model.

## Commands

Inference (downloads pinned files on first authorized access):

```powershell
.\.venv\Scripts\python.exe -m vault_ml.infer --prompt "Explain a concentration mismatch."
.\.venv\Scripts\python.exe -m vault_ml.infer --request examples/request.json --output experiments/baseline/my-smoke.json
```

`--output` is opt-in and uses exclusive creation. Choose a new filename on every run. Persist only synthetic or explicitly permitted requests. Failure returns exit code 1 and produces no model response. The authentication-denied path was executed and returned 1. Model source errors are sanitized to avoid credential leakage.

Default placement is explicit CUDA NF4 with double quantization and BF16 compute. Full precision cannot fit this GPU. Unquantized `--placement auto --quantization none` is an **unexecuted** inference-offload option, capped at 75% of free VRAM and 12 GiB CPU memory; inspect currently available RAM before use. `--placement cpu --quantization none` is also unexecuted and can require substantial RAM. Auto offload is never used for training. Inputs follow the embedding execution hook rather than assuming `model.device` represents a dispatched model.

Scientific baseline and safety suite:

```powershell
.\.venv\Scripts\python.exe -m vault_ml.evaluation
.\.venv\Scripts\python.exe -m vault_ml.safety
```

Tiny Aya training feasibility (not yet executed successfully):

```powershell
.\.venv\Scripts\python.exe -m vault_ml.train --variant smoke --steps 2 --sequence-length 256
```

Inspect `experiments/training/<UTC-run>/result.json` only if status is complete. A failure writes `failure.json` and exits 1. It never silently falls back to a different model. If an authenticated run fails with OOM, make at most one reduced-sequence trial (`--sequence-length 128`); then use the cloud configuration. Do not repeatedly tune arbitrary loader flags.

The complete synthetic sequence, in separate processes to release VRAM:

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py
.\.venv\Scripts\python.exe scripts/run_pipeline.py --compare --steps 20
```

## Tests

Fast tests do not download a model:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

GPU tests explicitly create a tiny **random** Cohere2 in a temporary directory. They are compatibility tests, not a substitute base model or a VAULT experiment:

```powershell
$env:VAULT_RUN_GPU_TESTS = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_gpu_integration.py -v
Remove-Item Env:VAULT_RUN_GPU_TESTS
```

The native NF4 primitive was tested independently in diagnostics. See the ledger for actual test counts and retained failure logs. The upstream warning about Windows distributed-log redirects is unrelated to this single-GPU training path. Triton is not installed; no Triton/FlashAttention kernel or performance claim is required.
