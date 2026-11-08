"""Run the authorized synthetic model workstream, stopping on the first failed priority.

Run from repository root using the existing venv. Does not install packages, start
servers, upload data, or substitute a model. Each subprocess has separate GPU lifetime.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vault_ml.artifacts import now, run_directory, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="Also run matched tiny M1/M2 experiments, then science/safety suites.")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    directory = run_directory("experiments/pipeline")
    records = []

    def execute(label, module, *arguments):
        command = [sys.executable, "-u", "-m", module, *map(str, arguments)]
        print(f"Starting {label}; log: {directory / (label + '.log')}", flush=True)
        env = {**os.environ, "PYTHONUTF8": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
        with (directory / (label + ".log")).open("x", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        records.append({"label": label, "command": command, "exit_code": completed.returncode, "completed_at": now()})
        write_json(directory / (label + "-status.json"), records[-1])
        if completed.returncode:
            raise RuntimeError(f"{label} failed with exit {completed.returncode}; see local log.")

    try:
        execute("diagnostics", "vault_ml.diagnostics", "--output", directory / "diagnostics.json")
        execute("M0-inference", "vault_ml.infer", "--prompt", "Explain a concentration mismatch.", "--output", directory / "M0-inference.json")
        execute("M0-science", "vault_ml.evaluation", "--output-root", directory / "M0-science")
        execute("training-smoke", "vault_ml.train", "--steps", 2, "--output-root", directory / "training-smoke")
        execute("M0-safety", "vault_ml.safety", "--output-root", directory / "M0-safety")
        if args.compare:
            science_directories = {"M0": next((directory / "M0-science").iterdir())}
            safety_directories = {"M0": next((directory / "M0-safety").iterdir())}
            training_manifests = {}
            for variant in ["M1", "M2"]:
                root = directory / variant
                execute(variant + "-train", "vault_ml.train", "--variant", variant, "--steps", args.steps, "--output-root", root)
                manifests = list(root.glob("*/result.json"))
                if len(manifests) != 1:
                    raise RuntimeError("Expected exactly one successful adapter manifest.")
                adapter = json.loads(manifests[0].read_text(encoding="utf-8"))["adapter_id"]
                execute(variant + "-science", "vault_ml.evaluation", "--adapter", adapter, "--output-root", directory / (variant + "-science"))
                execute(variant + "-safety", "vault_ml.safety", "--adapter", adapter, "--output-root", directory / (variant + "-safety"))
                science_directories[variant] = next((directory / (variant + "-science")).iterdir())
                safety_directories[variant] = next((directory / (variant + "-safety")).iterdir())
                training_manifests[variant] = manifests[0]
            from vault_ml.comparison import compare_completed
            compare_completed(science_directories, safety_directories, training_manifests, directory / "comparison.json")
        write_json(directory / "result.json", {"status": "complete", "commands": records, "M3": "not implemented/validated"})
        print(directory)
        return 0
    except Exception as exc:
        write_json(directory / "failure.json", {"status": "failed", "commands": records, "error": str(exc)})
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
