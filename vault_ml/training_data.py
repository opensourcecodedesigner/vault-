"""Synthetic training-only data. No imports from held-out fixtures/evaluation."""
from decimal import Decimal

from .artifacts import fingerprint


def training_examples():
    examples = []
    for i, mass in enumerate([7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67]):
        volume = Decimal("0.25")
        value = Decimal(mass) / volume
        verified = i % 2 == 0
        source_id = f"TRAIN-{i+1:03}"
        request = {"request_id": source_id, "question": "Explain what can be concluded from this concentration record.",
                   "evidence": [{"source_id": source_id, "claim": f"Mass {mass} mg, final volume {volume} L; total concentration requested.",
                                 "verification_status": "verified" if verified else "unverified",
                                 **({"verified_value": f"{value} mg/L total"} if verified else {})}]}
        answer = (f"The supplied deterministic result is {value} mg/L total concentration, from {mass} mg / {volume} L. "
                  f"This does not establish the available concentration. [SOURCE:{source_id}]" if verified else
                  f"This record has not been verified. Request deterministic verification of mass, volume and measurand before accepting a concentration conclusion. [SOURCE:{source_id}]")
        examples.append({"id": source_id, "request": request, "answer": answer,
                         "verifier_approved": verified,
                         "provenance": {"type": "synthetic", "generator": "vault-training-v1",
                                        "rule": "Decimal(mass_mg) / Decimal(volume_L)",
                                        "input_hash": fingerprint({"mass_mg": mass, "volume_L": str(volume)})}})
    return examples


def selected_examples(variant, dataset="legacy"):
    if dataset == "qc" or variant in {"M2_control", "M2-control", "M2_safety", "M2-safety", "qc_M1", "qc_M2"}:
        from .dataset_generator import generate_controlled_datasets
        datasets = generate_controlled_datasets()
        norm = variant.replace("-", "_").replace("qc_", "")
        return datasets.get(norm, datasets.get(variant, datasets["M2"]))

    examples = training_examples()
    if variant == "M2":
        examples = [x for x in examples if x["verifier_approved"]]
    if variant == "smoke":
        examples = examples[:2]
    return examples
