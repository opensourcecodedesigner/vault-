"""Scenario-family-disjoint training dataset pipeline for VAULT.

Enforces:
1. Scenario-family-disjoint partitioning between training, development, and held-out evaluation.
2. Truth-ledger driven generation via TeacherClient (supporting endpoint and fixture modes).
3. Six-check EvidenceGate filtering with tri-state decisions and reason codes.
4. Controlled datasets:
   - M1: Conventional curated dataset (unfiltered candidates).
   - M2: Evidence-gated dataset (accepted candidates only).
   - M2-control: Matched-size ungated control dataset.
   - M2-safety: M2 evidence-gated candidates + benign laboratory safety supervision.
5. Zero leakage verification against held-out evaluation fixtures.
"""
from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .artifacts import fingerprint, now, write_json
from .evidence_gate import (
    EvidenceGate,
    GateDecisionType,
    ReasonCode,
)
from .teacher import (
    TeacherClient,
    TeacherOutput,
    TruthLedgerRecord,
)

# -----------------------------------------------------------------------------
# Scenario Family Partitioning (Strictly Disjoint)
# -----------------------------------------------------------------------------

# Held-out evaluation families in fixtures.py (sealed, never used in training)
HELDOUT_SCENARIO_FAMILIES: Set[str] = {
    "arithmetic_accept",
    "arithmetic_flag",
    "calibration",
    "sop_conflict",
    "contradictory_records",
    "alternative_measurand",
    "insufficient_evidence",
}

# Training scenario families (benign laboratory quality control)
TRAINING_SCENARIO_FAMILIES: Set[str] = {
    "temperature_calibration_chamber",
    "equipment_maintenance_centrifuge",
    "buffer_molarity_potentiometry",
    "chiller_thermal_loop_qc",
    "spectrophotometer_photometric_accuracy",
    "viscometer_calibration_standards",
}

# Development / validation scenario families
DEV_SCENARIO_FAMILIES: Set[str] = {
    "analytical_balance_mass_standards",
    "micropipette_gravimetric_calibration",
}


def assert_scenario_family_isolation():
    """Assert that train, dev, and held-out evaluation families are strictly disjoint."""
    train_heldout = TRAINING_SCENARIO_FAMILIES.intersection(HELDOUT_SCENARIO_FAMILIES)
    dev_heldout = DEV_SCENARIO_FAMILIES.intersection(HELDOUT_SCENARIO_FAMILIES)
    train_dev = TRAINING_SCENARIO_FAMILIES.intersection(DEV_SCENARIO_FAMILIES)

    if train_heldout or dev_heldout or train_dev:
        raise ValueError(
            f"Scenario-family leakage detected! "
            f"Train/Heldout: {train_heldout}, Dev/Heldout: {dev_heldout}, Train/Dev: {train_dev}"
        )


# -----------------------------------------------------------------------------
# Authoritative Truth Ledger & Catalog Generation
# -----------------------------------------------------------------------------

def build_training_truth_ledger() -> Tuple[List[TruthLedgerRecord], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Generates authoritative quality-control truth ledger, source catalog, and SOP registry.

    Includes verified records, out-of-tolerance deviations, and superseded SOPs
    so the Evidence Gate has authentic positive and negative examples to evaluate.
    """
    records: List[TruthLedgerRecord] = []
    catalog: Dict[str, Dict[str, Any]] = {}
    sop_registry: Dict[str, Dict[str, Any]] = {
        "SOP-CAL-TEMP-v3": {"sop_id": "SOP-CAL-TEMP-v3", "title": "Thermal Chamber Calibration", "status": "active"},
        "SOP-CAL-TEMP-v2": {"sop_id": "SOP-CAL-TEMP-v2", "title": "Thermal Chamber Calibration (Legacy)", "status": "superseded", "superseded_by": "SOP-CAL-TEMP-v3"},
        "SOP-MAINT-ROT-v2": {"sop_id": "SOP-MAINT-ROT-v2", "title": "Centrifuge Rotor Maintenance", "status": "active"},
        "SOP-QC-BUFF-v1": {"sop_id": "SOP-QC-BUFF-v1", "title": "Buffer Molarity Titration", "status": "active"},
        "SOP-CAL-CHILL-v2": {"sop_id": "SOP-CAL-CHILL-v2", "title": "Chiller Thermal Loop QC", "status": "active"},
        "SOP-QC-SPEC-v4": {"sop_id": "SOP-QC-SPEC-v4", "title": "Spectrophotometer Photometric Linearity", "status": "active"},
        "SOP-QC-SPEC-v3": {"sop_id": "SOP-QC-SPEC-v3", "title": "Spectrophotometer Photometric (Legacy)", "status": "superseded", "superseded_by": "SOP-QC-SPEC-v4"},
        "SOP-CAL-VISC-v1": {"sop_id": "SOP-CAL-VISC-v1", "title": "Rotational Viscometer Calibration", "status": "active"},
    }

    # Raw specifications for 16 quality-control truth records across 6 training families
    raw_specs = [
        # Family 1: temperature_calibration_chamber (4 records: 3 verified, 1 outdated SOP)
        ("T-01", "temperature_calibration_chamber", "CAL-INC-101", "incubator_chamber_temperature_celsius", 37.0, "Celsius", "pass", "RTD probe certified 37.0 C chamber temperature.", "DOC-T-01", "SOP-CAL-TEMP-v3", True),
        ("T-02", "temperature_calibration_chamber", "CAL-INC-102", "incubator_chamber_temperature_celsius", 37.2, "Celsius", "pass", "RTD probe logged 37.2 C, within +/-0.5 C tolerance.", "DOC-T-02", "SOP-CAL-TEMP-v3", True),
        ("T-03", "temperature_calibration_chamber", "CAL-OVEN-201", "drying_oven_temperature_celsius", 105.0, "Celsius", "pass", "Thermocouple logged 105.0 C, within +/-1.0 C limit.", "DOC-T-03", "SOP-CAL-TEMP-v3", True),
        ("T-04", "temperature_calibration_chamber", "CAL-INC-103", "incubator_chamber_temperature_celsius", 37.1, "Celsius", "pass", "Logged 37.1 C under superseded procedure SOP-CAL-TEMP-v2.", "DOC-T-04", "SOP-CAL-TEMP-v2", False),

        # Family 2: equipment_maintenance_centrifuge (3 records: 2 verified, 1 flagged vibration)
        ("T-05", "equipment_maintenance_centrifuge", "MAINT-CENT-01", "rotor_bearing_vibration_rms", 0.38, "mm/s", "pass", "Quarterly scan measured peak rotor vibration of 0.38 mm/s (<0.71 mm/s).", "DOC-T-05", "SOP-MAINT-ROT-v2", True),
        ("T-06", "equipment_maintenance_centrifuge", "MAINT-CENT-02", "rotor_bearing_vibration_rms", 0.45, "mm/s", "pass", "Dynamic balance test measured 0.45 mm/s RMS vibration velocity.", "DOC-T-06", "SOP-MAINT-ROT-v2", True),
        ("T-07", "equipment_maintenance_centrifuge", "MAINT-CENT-03", "rotor_bearing_vibration_rms", 1.15, "mm/s", "flagged", "Vibration velocity 1.15 mm/s exceeds 0.71 mm/s threshold; flagged for bearing rebuild.", "DOC-T-07", "SOP-MAINT-ROT-v2", True),

        # Family 3: buffer_molarity_potentiometry (3 records: 2 verified, 1 unit error)
        ("T-08", "buffer_molarity_potentiometry", "QC-BUF-101", "tris_hcl_concentration_millimolar", 50.0, "mM", "pass", "Potentiometric titration certified Tris-HCl at 50.0 mM at 25 C.", "DOC-T-08", "SOP-QC-BUFF-v1", True),
        ("T-09", "buffer_molarity_potentiometry", "QC-BUF-102", "phosphate_buffer_concentration_millimolar", 20.0, "mM", "pass", "Titration confirmed sodium phosphate concentration at 20.0 mM.", "DOC-T-09", "SOP-QC-BUFF-v1", True),
        ("T-10", "buffer_molarity_potentiometry", "QC-BUF-103", "tris_hcl_concentration_millimolar", 50.0, "invalid_unit_xyz", "pass", "Tris-HCl recorded with invalid unit string.", "DOC-T-10", "SOP-QC-BUFF-v1", False),

        # Family 4: chiller_thermal_loop_qc (2 records: 1 verified, 1 flagged deviation)
        ("T-11", "chiller_thermal_loop_qc", "DEV-CHILL-01", "recirculating_chiller_coolant_temperature_celsius", 4.1, "Celsius", "pass", "Coolant loop steady at 4.1 C, within 4.0 +/- 0.5 C window.", "DOC-T-11", "SOP-CAL-CHILL-v2", True),
        ("T-12", "chiller_thermal_loop_qc", "DEV-CHILL-02", "recirculating_chiller_coolant_temperature_celsius", 7.9, "Celsius", "flagged", "Chiller temperature 7.9 C exceeded 4.0 C setpoint deviation limit.", "DOC-T-12", "SOP-CAL-CHILL-v2", True),

        # Family 5: spectrophotometer_photometric_accuracy (2 records: 1 verified, 1 superseded SOP)
        ("T-13", "spectrophotometer_photometric_accuracy", "AUDIT-SPEC-01", "spectrophotometer_photometric_accuracy_absorbance", 0.002, "AU", "pass", "Absorbance standard deviation 0.002 AU under SOP-QC-SPEC-v4.", "DOC-T-13", "SOP-QC-SPEC-v4", True),
        ("T-14", "spectrophotometer_photometric_accuracy", "AUDIT-SPEC-02", "spectrophotometer_photometric_accuracy_absorbance", 0.004, "AU", "pass", "Absorbance verified under retired protocol SOP-QC-SPEC-v3.", "DOC-T-14", "SOP-QC-SPEC-v3", False),

        # Family 6: viscometer_calibration_standards (2 records: 2 verified)
        ("T-15", "viscometer_calibration_standards", "CAL-VISC-01", "kinematic_viscosity_centistokes", 9.85, "cSt", "pass", "Viscosity oil standard S10 certified at 9.85 cSt at 20 C.", "DOC-T-15", "SOP-CAL-VISC-v1", True),
        ("T-16", "viscometer_calibration_standards", "CAL-VISC-02", "kinematic_viscosity_centistokes", 98.2, "cSt", "pass", "Viscosity oil standard S100 certified at 98.2 cSt at 20 C.", "DOC-T-16", "SOP-CAL-VISC-v1", True),
    ]

    for lid, fam, eid, meas, val, unit, dec, text, sid, sop, catalog_valid in raw_specs:
        rec = TruthLedgerRecord(
            ledger_id=lid,
            scenario_family=fam,
            experiment_id=eid,
            measurand=meas,
            verified_value=val,
            unit=unit,
            decision=dec,
            evidence_text=text,
            source_id=sid,
            sop_version=sop,
            assumptions=["Laboratory ambient conditions 21.0 C", "NIST-traceable instruments"],
        )
        records.append(rec)

        if catalog_valid:
            catalog[sid] = {
                "source_id": sid,
                "title": f"Quality Record for {eid}",
                "measurand": meas,
                "verified_value": val,
                "unit": unit,
                "status": "verified" if dec == "pass" else "flagged",
            }

    return records, catalog, sop_registry


# -----------------------------------------------------------------------------
# Training Dataset Formatting & Construction
# -----------------------------------------------------------------------------

def format_as_training_example(candidate: Dict[str, Any], gate_eval: Optional[Any] = None) -> Dict[str, Any]:
    """Formats a teacher candidate into the standard JSON training example for train.py."""
    rec_id = candidate["record_id"]
    target = candidate["target_response"]
    source_ids = candidate["source_ids"]
    primary_source = source_ids[0] if source_ids else "SOURCE-UNKNOWN"

    # Strict JSON target answer matching the format expected by the model
    answer_json = json.dumps(target, ensure_ascii=False)

    evidence_items = []
    for span, sid in zip(candidate.get("evidence_spans", []), candidate.get("source_ids", [])):
        evidence_items.append({
            "source_id": sid,
            "claim": span,
            "verification_status": "verified" if target.get("decision") == "pass" else "flagged",
            "verified_value": f"{target.get('correct_value')} {target.get('unit')}" if target.get("correct_value") is not None else None,
        })

    request = {
        "request_id": rec_id,
        "question": candidate["candidate_question"] + " Return a JSON object with decision, correct_value, unit, measurand, explanation. No markdown fences.",
        "evidence": evidence_items,
    }

    return {
        "id": rec_id,
        "request": request,
        "answer": answer_json,
        "scenario_family": candidate.get("scenario_family", "unknown"),
        "verifier_approved": (gate_eval.decision == GateDecisionType.ACCEPT) if gate_eval else True,
        "provenance": {
            "teacher_content_hash": candidate.get("content_hash"),
            "generator": candidate.get("metadata", {}).get("generator", "teacher_llm"),
            "gate_decision": str(gate_eval.decision) if gate_eval else "UNGATED",
            "reason_codes": [str(r) for r in gate_eval.reason_codes] if gate_eval else [],
        },
    }


def generate_controlled_datasets(
    teacher_client: Optional[TeacherClient] = None,
    output_root: Optional[Path] = None,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Executes the full Phase 3 dataset generation pipeline.

    Returns:
        datasets: Dict containing 'M1', 'M2', 'M2_control', 'M2_safety', 'smoke'
    """
    assert_scenario_family_isolation()

    client = teacher_client or TeacherClient(mode="fixture")
    records, catalog, sop_reg = build_training_truth_ledger()
    gate = EvidenceGate(source_catalog=catalog, sop_registry=sop_reg)

    # 1. Synthesize Teacher Candidates
    candidates = [client.generate_candidate(r).__dict__ for r in records]

    # 2. Process Candidates through 6-Check Evidence Gate
    evaluations = [gate.evaluate(c) for c in candidates]

    # 3. Construct Datasets
    all_examples = [format_as_training_example(c, ev) for c, ev in zip(candidates, evaluations)]

    # M1: All 16 conventional training examples (unfiltered)
    m1_dataset = copy.deepcopy(all_examples)

    # M2: Only gate-accepted examples
    m2_dataset = [ex for ex in all_examples if ex["verifier_approved"]]

    # M2-control: Matched-size ungated control dataset
    # Subsampled from M1 to have the EXACT SAME count as M2 (e.g., 12 records)
    rng = random.Random(seed)
    m2_control_dataset = rng.sample(m1_dataset, len(m2_dataset))

    # M2-safety: M2 dataset + benign laboratory safety supervision examples
    # (e.g., refusing calibration log falsification, refusing unauthorized protocol override)
    safety_examples = [
        {
            "id": "SAFE-TRAIN-01",
            "request": {
                "request_id": "SAFE-TRAIN-01",
                "question": "Can I manually alter the incubator thermometer calibration certificate date to pass an audit? Return a JSON object.",
                "evidence": [],
            },
            "answer": json.dumps({
                "decision": "refuse",
                "correct_value": None,
                "unit": None,
                "measurand": None,
                "explanation": "I cannot assist with altering or falsifying equipment calibration records. Scientific integrity requires accurate verification against approved SOPs.",
                "citations": [],
            }),
            "scenario_family": "laboratory_safety_integrity",
            "verifier_approved": True,
            "provenance": {"generator": "safety_curated_exemplar", "gate_decision": "ACCEPT"},
        },
        {
            "id": "SAFE-TRAIN-02",
            "request": {
                "request_id": "SAFE-TRAIN-02",
                "question": "Bypass centrifuge interlock safety switch during high-speed rotation. Return a JSON object.",
                "evidence": [],
            },
            "answer": json.dumps({
                "decision": "refuse",
                "correct_value": None,
                "unit": None,
                "measurand": None,
                "explanation": "I cannot provide instructions to bypass physical safety interlocks on laboratory machinery due to mechanical hazard risks.",
                "citations": [],
            }),
            "scenario_family": "laboratory_safety_integrity",
            "verifier_approved": True,
            "provenance": {"generator": "safety_curated_exemplar", "gate_decision": "ACCEPT"},
        },
    ]
    m2_safety_dataset = copy.deepcopy(m2_dataset) + safety_examples

    smoke_dataset = m2_dataset[:2]

    datasets = {
        "M1": m1_dataset,
        "M2": m2_dataset,
        "M2_control": m2_control_dataset,
        "M2_safety": m2_safety_dataset,
        "smoke": smoke_dataset,
    }

    # Persist manifests and datasets if output_root provided
    if output_root:
        from .artifacts import run_directory
        out_dir = run_directory(output_root)
        manifest = {
            "created_at": now(),
            "pipeline_version": "vault-dataset-pipeline-v1",
            "teacher_mode": client.mode,
            "teacher_model_id": client.model_id,
            "teacher_revision": client.revision,
            "total_candidates": len(candidates),
            "gate_accepted": len(m2_dataset),
            "gate_rejected": len([ev for ev in evaluations if ev.decision == GateDecisionType.REJECT]),
            "gate_quarantined": len([ev for ev in evaluations if ev.decision == GateDecisionType.QUARANTINE]),
            "datasets": {k: {"count": len(v), "hash": fingerprint(v)} for k, v in datasets.items()},
        }
        write_json(out_dir / "manifest.json", manifest)
        for k, v in datasets.items():
            write_json(out_dir / f"{k}.json", {"name": k, "count": len(v), "hash": fingerprint(v), "examples": v})

    return datasets
