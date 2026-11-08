# Executable Linux/cloud fallback — not run

Use a private Linux machine with a 16–24 GB NVIDIA GPU, Python 3.12 and enough disk for the ~6.7 GB official checkpoint plus dependencies/artifacts. These are target resources, not a measured guarantee. No VM, job or paid resource has been created. No scientific data has been uploaded. Copy only this code and synthetic datasets; exclude `.venv`, caches, credentials and any private records.

In that machine's own virtual environment, install the platform-appropriate supported CUDA PyTorch build using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install -r requirements-ml.txt
python -m pip check
hf auth login
python -m vault_ml.diagnostics
python -m vault_ml.infer --prompt 'Explain a concentration mismatch.'
python -m vault_ml.train --config experiments/training/cloud-16gb.json
```

The JSON pins the official base revision, NF4/BF16 settings, rank 4, alpha 8, batch 1, accumulation 1, sequence 256, seed 42, AdamW 2e-4 and 20 steps. It trains synthetic M1 only. There is no alternate model and no SafeTune label. Retain the cloud environment's own `pip freeze`, GPU/driver information, exact code commit and run artifacts. Do not apply the Windows transitive lock to Linux.

For matched synthetic M1/M2 feasibility and identical held-out evaluations, after an ordinary smoke run succeeds:

```bash
python scripts/run_pipeline.py --compare --steps 20
```

The comparison uses equal processed-token budgets and records differing supervised-token counts. It remains a tiny feasibility experiment. Both model safety and scientific explanation quality require manual review; M3 needs the separately documented SafeTune qualification. This configuration has not been executed on Linux or the official Tiny Aya checkpoint.
