import argparse
import json
import math
import sys

from .artifacts import fingerprint, now, run_directory, write_json
from .config import PROMPT_VERSION
from .contracts import InferenceRequest, messages_for
from .fixtures import SCIENCE_VERSION, science_cases
from .infer import InferenceEngine, config_from_args, model_arguments


def structured_metrics(raw, ground_truth):
    """Strict output-contract scores; these do not certify scientific reasoning."""
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        value = None
    if not isinstance(value, dict):
        return {"json_object": False, "decision_match": None, "numeric_match": None, "measurand_match": None}
    expected = ground_truth["correct_value"]
    number = value.get("correct_value")
    numeric = ((number is None and "correct_value" in value) if expected is None else
               isinstance(number, (int, float)) and not isinstance(number, bool) and math.isfinite(number)
               and math.isclose(number, expected, rel_tol=1e-6, abs_tol=1e-9)
               and value.get("unit") == ground_truth["unit"])
    return {"json_object": True, "decision_match": value.get("decision") == ground_truth["decision"],
            "numeric_match": numeric, "measurand_match": "measurand" in value and value["measurand"] == ground_truth["measurand"]}


def evaluate_science(engine, root):
    directory = run_directory(root)
    cases = science_cases()
    write_json(directory / "manifest.json", {"created_at": now(), "suite": SCIENCE_VERSION,
        "suite_hash": fingerprint(cases), "model": engine.config.to_dict(), "prompt_template_version": PROMPT_VERSION,
        "status": "started", "scoring": "Strict JSON/decision/numeric/unit/measurand checks; explanations require manual review."})
    records = []
    with (directory / "results.jsonl").open("x", encoding="utf-8") as stream:
        for case in cases:
            request = InferenceRequest.from_dict({**case["request"], "adapter_id": engine.config.adapter_id})
            raw, metadata = engine.generate_messages(messages_for(request))
            record = {"test_id": case["test_id"], "fixture_hash": fingerprint(case),
                      "suite": SCIENCE_VERSION, "category": case["category"],
                      "model_id": engine.config.model_id, "revision": engine.config.revision,
                      "adapter_id": engine.config.adapter_id, "prompt_template_version": PROMPT_VERSION,
                      "request": case["request"], "generation_metadata": metadata, "raw_response": raw,
                      "ground_truth": case["ground_truth"], "metrics": structured_metrics(raw, case["ground_truth"]),
                      "manual_review_required": True,
                      "ambiguities": ["Strict formatting scores do not verify explanation validity or safety.",
                                      "Synthetic stipulated facts have no real-world evidentiary authority."]}
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            print(f"completed {case['test_id']}", file=sys.stderr, flush=True)
    summary = {"completed": len(records), "suite_hash": fingerprint(cases), "status": "complete",
               "metrics": {name: {"true": sum(r["metrics"][name] is True for r in records),
                                    "scorable": sum(r["metrics"][name] is not None for r in records)}
                           for name in records[0]["metrics"]},
               "limitation": "10 synthetic smoke cases; no statistical improvement or scientific reliability claim."}
    write_json(directory / "summary.json", summary)
    return directory


def main():
    parser = argparse.ArgumentParser()
    model_arguments(parser)
    parser.add_argument("--output-root", default="experiments/baseline")
    parser.add_argument("--fixtures-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.fixtures_only:
            directory = run_directory(args.output_root)
            write_json(directory / "fixtures.json", {"suite": SCIENCE_VERSION, "fixtures": science_cases(),
                                                      "suite_hash": fingerprint(science_cases()), "model_run": False})
        else:
            directory = evaluate_science(InferenceEngine(config_from_args(args)), args.output_root)
        print(directory)
        return 0
    except Exception as exc:
        print(f"Evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
