"""Compare completed records only; never synthesize missing checkpoints or human labels."""
import json
from pathlib import Path

from .artifacts import write_json


def compare_completed(science_directories, safety_directories, training_manifests, output):
    if set(science_directories) != {"M0", "M1", "M2"} or set(safety_directories) != {"M0", "M1", "M2"}:
        raise ValueError("Require completed M0/M1/M2 science and safety reports.")
    result = {"status": "complete", "science": {}, "safety": {}, "training": {},
              "M3": "not implemented/validated", "safety_preservation_claim": False,
              "limitation": "Descriptive smoke comparison only; human safety and explanation review pending."}
    for variant in ["M0", "M1", "M2"]:
        for family, directories in [("science", science_directories), ("safety", safety_directories)]:
            directory = Path(directories[variant])
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            if summary["status"] != "complete":
                raise ValueError("Cannot compare incomplete evaluation.")
            result[family][variant] = {"manifest": manifest, "summary": summary, "directory": str(directory)}
    for family in ["science", "safety"]:
        hashes = {v["manifest"]["suite_hash"] for v in result[family].values()}
        prompt_versions = {v["manifest"]["prompt_template_version"] for v in result[family].values()}
        configs = [{k: v for k, v in x["manifest"]["model"].items() if k != "adapter_id"} for x in result[family].values()]
        if len(hashes) != 1 or len(prompt_versions) != 1 or any(config != configs[0] for config in configs[1:]):
            raise ValueError("Held-out suite or execution configuration differs between checkpoints.")
        if result[family]["M0"]["manifest"]["model"]["adapter_id"] is not None:
            raise ValueError("M0 must have no adapter.")
    for variant in ["M1", "M2"]:
        record = json.loads(Path(training_manifests[variant]).read_text(encoding="utf-8"))
        if record["status"] != "complete" or not record["adapter_verification"]["reload_exact_match"]:
            raise ValueError("Training or adapter verification incomplete.")
        for family in ["science", "safety"]:
            if result[family][variant]["manifest"]["model"]["adapter_id"] != record["adapter_id"]:
                raise ValueError("Evaluation does not use the recorded trained adapter.")
            if any(record["model"][key] != result[family][variant]["manifest"]["model"][key]
                   for key in ["model_id", "revision", "quantization"]):
                raise ValueError("Training and evaluation base models differ.")
        result["training"][variant] = record
    first, second = result["training"]["M1"], result["training"]["M2"]
    fields = ["steps", "sequence_length", "batch_size", "gradient_accumulation", "learning_rate", "rank", "alpha", "seed"]
    if first["processed_token_budget"] != second["processed_token_budget"] or any(first["training"][k] != second["training"][k] for k in fields):
        raise ValueError("Training budgets/hyperparameters differ; do not call this a matched comparison.")
    result["matched_processed_tokens"] = first["processed_token_budget"]
    write_json(output, result)
    return result
