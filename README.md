# VAULT model workstream

Native Windows Tiny Aya inference, synthetic scientific/safety evaluation and bounded QLoRA training.
This is Rohith's isolated workstream. No backend, verifier or immutable scientific-memory implementation is fabricated here.

**Current status:** dependencies and local compatibility checks work. Official Tiny Aya execution is blocked by missing Hugging Face authentication. No M0 response, Tiny Aya adapter, M1/M2 comparison or safety-preservation result has been produced.

Start with [the handoff](docs/HANDOFF.md), [Windows setup](docs/WINDOWS_SETUP.md), [experiment ledger](docs/EXPERIMENT_LEDGER.md) and [proposed team contract](docs/INTEGRATION_CONTRACT.md).

```powershell
cd D:\VAULT
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\hf.exe auth login
.\.venv\Scripts\python.exe -m vault_ml.infer --prompt "Explain a concentration mismatch."
```

After login, `scripts/run_pipeline.py` runs the synthetic workstream in priority order and stops on failure. Use `--compare` to include bounded M1/M2 trials. Each command records its exit status and log. No public server or external data upload is started.
