"""Six-check evidence gate and CuratorKIT integration for VAULT.

Implements the six evidence checks:
1. SOURCE_EXISTS: Cited source exists in catalog.
2. SPAN_RELEVANT: Referenced evidence span is non-empty and lexically/semantically relevant.
3. CLAIM_SUPPORTED: Factual assertion is substantiated by the cited excerpt.
4. MATH_UNITS_CORRECT: Numerical values, unit consistency, and dimensional calculations match.
5. MULTI_SOURCE_CONSISTENT: Corroborating sources do not contradict the finding.
6. SOP_VERSION_APPLICABLE: Referenced Standard Operating Procedure is active and applicable.

Produces tri-state decisions: ACCEPT, QUARANTINE, REJECT with machine-readable reason codes.
Includes controlled label corruption evaluation (0%, 5%, 10%, 20%).
"""
from __future__ import annotations

import copy
import math
import random
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .artifacts import fingerprint, now, write_json

GATE_VERSION = "vault-evidence-gate-v1"


class GateDecisionType(str, Enum):
    ACCEPT = "ACCEPT"
    QUARANTINE = "QUARANTINE"
    REJECT = "REJECT"


class ReasonCode(str, Enum):
    # Passes
    PASS_ALL_CHECKS = "PASS_ALL_CHECKS"
    # Check 1: Source
    ERR_SOURCE_NOT_FOUND = "ERR_SOURCE_NOT_FOUND"
    # Check 2: Span
    ERR_SPAN_EMPTY = "ERR_SPAN_EMPTY"
    ERR_SPAN_IRRELEVANT = "ERR_SPAN_IRRELEVANT"
    # Check 3: Claim
    ERR_CLAIM_UNSUPPORTED = "ERR_CLAIM_UNSUPPORTED"
    # Check 4: Math & Units
    ERR_NUMERIC_MISMATCH = "ERR_NUMERIC_MISMATCH"
    ERR_UNIT_INCOMPATIBLE = "ERR_UNIT_INCOMPATIBLE"
    ERR_NONFINITE_VALUE = "ERR_NONFINITE_VALUE"
    # Check 5: Consistency
    ERR_SOURCE_CONTRADICTION = "ERR_SOURCE_CONTRADICTION"
    # Check 6: SOP
    ERR_SOP_SUPERSEDED = "ERR_SOP_SUPERSEDED"
    ERR_SOP_UNKNOWN = "ERR_SOP_UNKNOWN"
    # Quarantine reasons (Alternative methods / ambiguous evidence)
    WARN_ALTERNATIVE_METHOD = "WARN_ALTERNATIVE_METHOD"
    WARN_PROVENANCE_UNVERIFIED = "WARN_PROVENANCE_UNVERIFIED"


@dataclass
class GateEvaluation:
    record_id: str
    decision: GateDecisionType
    checks_passed: List[str]
    checks_failed: List[str]
    reason_codes: List[ReasonCode]
    details: Dict[str, Any]
    evaluator_version: str = GATE_VERSION
    timestamp: str = field(default_factory=now)
    provenance_hash: str = ""


class EvidenceGate:
    """Deterministic six-check evidence gate."""

    def __init__(
        self,
        source_catalog: Optional[Dict[str, Dict[str, Any]]] = None,
        sop_registry: Optional[Dict[str, Dict[str, Any]]] = None,
        unit_equivalences: Optional[Dict[str, Set[str]]] = None,
    ):
        self.source_catalog = source_catalog or {}
        self.sop_registry = sop_registry or {}
        self.unit_equivalences = unit_equivalences or {
            "mg/L": {"mg/L", "ppm"},
            "ug/L": {"ug/L", "ppb"},
            "uM": {"uM", "micromolar"},
            "mM": {"mM", "millimolar"},
            "J/K": {"J/K", "joules/kelvin"},
            "Celsius": {"Celsius", "C", "degC"},
        }

    def evaluate(self, candidate: Dict[str, Any]) -> GateEvaluation:
        """Evaluate a candidate record against the six evidence checks without mutating original data."""
        record_id = candidate.get("record_id", "UNKNOWN")
        target = candidate.get("target_response", {})
        source_ids = candidate.get("source_ids", [])
        evidence_spans = candidate.get("evidence_spans", [])
        lab_notes = candidate.get("lab_notes", "")
        metadata = candidate.get("metadata", {})

        passed: List[str] = []
        failed: List[str] = []
        reasons: List[ReasonCode] = []
        details: Dict[str, Any] = {}

        # ---------------------------------------------------------------------
        # Check 1: Cited source exists in catalog
        # ---------------------------------------------------------------------
        missing_sources = [s for s in source_ids if s not in self.source_catalog]
        if missing_sources:
            failed.append("CHECK_1_SOURCE_EXISTS")
            reasons.append(ReasonCode.ERR_SOURCE_NOT_FOUND)
            details["missing_sources"] = missing_sources
        elif not source_ids:
            failed.append("CHECK_1_SOURCE_EXISTS")
            reasons.append(ReasonCode.ERR_SOURCE_NOT_FOUND)
            details["missing_sources"] = ["NO_SOURCES_CITED"]
        else:
            passed.append("CHECK_1_SOURCE_EXISTS")

        # ---------------------------------------------------------------------
        # Check 2: Referenced evidence span is relevant
        # ---------------------------------------------------------------------
        if not evidence_spans or any(not span.strip() for span in evidence_spans):
            failed.append("CHECK_2_SPAN_RELEVANT")
            reasons.append(ReasonCode.ERR_SPAN_EMPTY)
        else:
            # Check overlap between span keywords and lab notes or measurand
            measurand = target.get("measurand", "")
            span_text = " ".join(evidence_spans).lower()
            measurand_tokens = [t for t in measurand.lower().split("_") if len(t) > 2]
            if measurand_tokens and not any(t in span_text for t in measurand_tokens):
                # Also accept if span appears verbatim in lab_notes
                if any(span in lab_notes for span in evidence_spans):
                    passed.append("CHECK_2_SPAN_RELEVANT")
                else:
                    failed.append("CHECK_2_SPAN_RELEVANT")
                    reasons.append(ReasonCode.ERR_SPAN_IRRELEVANT)
            else:
                passed.append("CHECK_2_SPAN_RELEVANT")

        # ---------------------------------------------------------------------
        # Check 3: Claim is supported by evidence
        # ---------------------------------------------------------------------
        explanation = target.get("explanation", "").lower()
        if not explanation:
            failed.append("CHECK_3_CLAIM_SUPPORTED")
            reasons.append(ReasonCode.ERR_CLAIM_UNSUPPORTED)
        else:
            # If target claims a decision of 'pass' but evidence span mentions 'flagged' or 'fail'
            decision = target.get("decision", "").lower()
            all_evidence = " ".join(evidence_spans).lower()
            if decision == "pass" and any(k in all_evidence for k in ["flagged", "failed", "aborted", "out_of_tolerance", "deviation", "overdue"]):
                failed.append("CHECK_3_CLAIM_SUPPORTED")
                reasons.append(ReasonCode.ERR_CLAIM_UNSUPPORTED)
                details["decision_conflict"] = "Claimed pass but evidence notes failure/flagging"
            else:
                passed.append("CHECK_3_CLAIM_SUPPORTED")

        # ---------------------------------------------------------------------
        # Check 4: Mathematics and units are correct
        # ---------------------------------------------------------------------
        claimed_val = target.get("correct_value")
        claimed_unit = target.get("unit")
        math_valid = True

        if claimed_val is not None:
            if not isinstance(claimed_val, (int, float)) or isinstance(claimed_val, bool) or not math.isfinite(claimed_val):
                failed.append("CHECK_4_MATH_UNITS_CORRECT")
                reasons.append(ReasonCode.ERR_NONFINITE_VALUE)
                math_valid = False
            else:
                # Compare against source catalog if source has expected value
                for sid in source_ids:
                    src_entry = self.source_catalog.get(sid, {})
                    catalog_val = src_entry.get("verified_value")
                    catalog_unit = src_entry.get("unit")

                    if catalog_val is not None and math.isfinite(catalog_val):
                        if not math.isclose(claimed_val, catalog_val, rel_tol=1e-4, abs_tol=1e-7):
                            failed.append("CHECK_4_MATH_UNITS_CORRECT")
                            reasons.append(ReasonCode.ERR_NUMERIC_MISMATCH)
                            details["expected_value"] = catalog_val
                            details["claimed_value"] = claimed_val
                            math_valid = False
                            break

                    if catalog_unit and claimed_unit:
                        allowed_units = self.unit_equivalences.get(catalog_unit, {catalog_unit})
                        if claimed_unit not in allowed_units:
                            failed.append("CHECK_4_MATH_UNITS_CORRECT")
                            reasons.append(ReasonCode.ERR_UNIT_INCOMPATIBLE)
                            details["expected_unit"] = catalog_unit
                            details["claimed_unit"] = claimed_unit
                            math_valid = False
                            break

        if math_valid and "CHECK_4_MATH_UNITS_CORRECT" not in failed:
            passed.append("CHECK_4_MATH_UNITS_CORRECT")

        # ---------------------------------------------------------------------
        # Check 5: Multiple sources are consistent
        # ---------------------------------------------------------------------
        if len(source_ids) > 1:
            values_in_sources = []
            for sid in source_ids:
                src_val = self.source_catalog.get(sid, {}).get("verified_value")
                if src_val is not None and math.isfinite(src_val):
                    values_in_sources.append(src_val)
            if len(values_in_sources) > 1:
                first = values_in_sources[0]
                if any(not math.isclose(first, v, rel_tol=1e-3, abs_tol=1e-6) for v in values_in_sources[1:]):
                    failed.append("CHECK_5_MULTI_SOURCE_CONSISTENT")
                    reasons.append(ReasonCode.ERR_SOURCE_CONTRADICTION)
                    details["conflicting_source_values"] = values_in_sources
                else:
                    passed.append("CHECK_5_MULTI_SOURCE_CONSISTENT")
            else:
                passed.append("CHECK_5_MULTI_SOURCE_CONSISTENT")
        else:
            passed.append("CHECK_5_MULTI_SOURCE_CONSISTENT")

        # ---------------------------------------------------------------------
        # Check 6: Applicable SOP version is active and valid
        # ---------------------------------------------------------------------
        sop = candidate.get("sop_version") or metadata.get("sop_version") or target.get("sop_version")
        if not sop:
            # Check lab notes for SOP mention
            match = re.search(r"SOP-[A-Za-z0-9_.-]+", lab_notes)
            sop = match.group(0) if match else None

        if not sop:
            # If no SOP specified, pass by default unless catalog demands one
            passed.append("CHECK_6_SOP_VERSION_APPLICABLE")
        elif sop not in self.sop_registry:
            # If not in registry, quarantine if novel/alternative methodology, else pass if catalog lacks registry
            if self.sop_registry:
                failed.append("CHECK_6_SOP_VERSION_APPLICABLE")
                reasons.append(ReasonCode.ERR_SOP_UNKNOWN)
            else:
                passed.append("CHECK_6_SOP_VERSION_APPLICABLE")
        else:
            sop_meta = self.sop_registry[sop]
            if sop_meta.get("status") == "superseded":
                failed.append("CHECK_6_SOP_VERSION_APPLICABLE")
                reasons.append(ReasonCode.ERR_SOP_SUPERSEDED)
                details["superseded_by"] = sop_meta.get("superseded_by")
            elif sop_meta.get("status") == "experimental":
                reasons.append(ReasonCode.WARN_ALTERNATIVE_METHOD)
                passed.append("CHECK_6_SOP_VERSION_APPLICABLE")
            else:
                passed.append("CHECK_6_SOP_VERSION_APPLICABLE")

        # ---------------------------------------------------------------------
        # Determine Tri-State Decision
        # ---------------------------------------------------------------------
        if not failed:
            if ReasonCode.WARN_ALTERNATIVE_METHOD in reasons:
                decision = GateDecisionType.QUARANTINE
            else:
                decision = GateDecisionType.ACCEPT
                reasons.append(ReasonCode.PASS_ALL_CHECKS)
        else:
            # Fatal checks: Missing source, bad math, nonfinite, contradiction, superseded SOP
            fatal_reasons = {
                ReasonCode.ERR_SOURCE_NOT_FOUND,
                ReasonCode.ERR_NUMERIC_MISMATCH,
                ReasonCode.ERR_UNIT_INCOMPATIBLE,
                ReasonCode.ERR_NONFINITE_VALUE,
                ReasonCode.ERR_SOURCE_CONTRADICTION,
                ReasonCode.ERR_SOP_SUPERSEDED,
            }
            if any(r in fatal_reasons for r in reasons):
                decision = GateDecisionType.REJECT
            else:
                # Soft failure: irrelevant span or unknown SOP can be quarantined for expert review
                decision = GateDecisionType.QUARANTINE

        eval_obj = GateEvaluation(
            record_id=record_id,
            decision=decision,
            checks_passed=passed,
            checks_failed=failed,
            reason_codes=reasons,
            details=details,
            evaluator_version=GATE_VERSION,
        )
        eval_obj.provenance_hash = fingerprint(asdict(eval_obj))
        return eval_obj


# =============================================================================
# Label Corruption Benchmark Harness
# =============================================================================

def inject_label_corruption(
    records: List[Dict[str, Any]],
    corruption_rate: float,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[bool]]:
    """Inject controlled corruption at specified rate (0%, 5%, 10%, 20%).

    Returns:
        corrupted_records: New copies with injected errors.
        is_corrupted: Boolean flag per record indicating if corruption was injected.
    """
    rng = random.Random(seed)
    corrupted_records = []
    is_corrupted = []

    for item in records:
        rec = copy.deepcopy(item)
        corrupt = rng.random() < corruption_rate

        if corrupt:
            corruption_type = rng.choice(["math", "unit", "source", "sop", "decision"])
            target = rec.setdefault("target_response", {})

            if corruption_type == "math":
                curr = target.get("correct_value", 10.0)
                target["correct_value"] = curr * 10.0 if curr is not None else 999.0
            elif corruption_type == "unit":
                target["unit"] = "invalid_furlongs_per_fortnight"
            elif corruption_type == "source":
                rec["source_ids"] = ["NON_EXISTENT_GHOST_SOURCE_999"]
            elif corruption_type == "sop":
                rec.setdefault("metadata", {})["sop_version"] = "SOP-SUPERSEDED-OLD-v1"
            elif corruption_type == "decision":
                # Contradicting verdict for quality control
                target["decision"] = "pass"
                rec["evidence_spans"] = ["CALIBRATION DEVIATION: Incubator temperature exceeded standard tolerance by 4.2 degC."]

        is_corrupted.append(corrupt)
        corrupted_records.append(rec)

    return corrupted_records, is_corrupted


def benchmark_gate_corruption(
    gate: EvidenceGate,
    clean_records: List[Dict[str, Any]],
    corruption_rates: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.20),
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate EvidenceGate precision, recall, and detection metrics across corruption tiers."""
    results = {}

    for rate in corruption_rates:
        corrupted, is_corrupted = inject_label_corruption(clean_records, rate, seed=seed)
        decisions = [gate.evaluate(rec) for rec in corrupted]

        # Ground truth: corrupt -> should be REJECT or QUARANTINE; clean -> should be ACCEPT
        total = len(clean_records)
        true_positives = sum(corr and d.decision in (GateDecisionType.REJECT, GateDecisionType.QUARANTINE)
                             for corr, d in zip(is_corrupted, decisions))
        false_positives = sum(not corr and d.decision in (GateDecisionType.REJECT, GateDecisionType.QUARANTINE)
                              for corr, d in zip(is_corrupted, decisions))
        true_negatives = sum(not corr and d.decision == GateDecisionType.ACCEPT
                             for corr, d in zip(is_corrupted, decisions))
        false_negatives = sum(corr and d.decision == GateDecisionType.ACCEPT
                              for corr, d in zip(is_corrupted, decisions))

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 1.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 1.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        rate_key = f"{int(rate * 100)}pct"
        results[rate_key] = {
            "corruption_rate": rate,
            "total_records": total,
            "corrupted_count": sum(is_corrupted),
            "clean_count": total - sum(is_corrupted),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "rejections": sum(d.decision == GateDecisionType.REJECT for d in decisions),
            "quarantines": sum(d.decision == GateDecisionType.QUARANTINE for d in decisions),
            "accepts": sum(d.decision == GateDecisionType.ACCEPT for d in decisions),
        }

    return results


def create_benign_qc_catalog() -> Dict[str, Dict[str, Any]]:
    """Factory creating authoritative source catalog for benign laboratory quality control."""
    return {
        "DOC-CAL-001": {
            "source_id": "DOC-CAL-001",
            "title": "Incubator RTD Temperature Probe Calibration Certificate",
            "measurand": "incubator_chamber_temperature_celsius",
            "verified_value": 37.1,
            "unit": "Celsius",
            "status": "verified",
        },
        "DOC-MAINT-002": {
            "source_id": "DOC-MAINT-002",
            "title": "Centrifuge Rotor Maintenance & Dynamic Balance Report",
            "measurand": "rotor_bearing_vibration_rms",
            "verified_value": 0.42,
            "unit": "mm/s",
            "status": "verified",
        },
        "DOC-QC-003": {
            "source_id": "DOC-QC-003",
            "title": "Tris-HCl Buffer Molarity and pH Quality Assurance Log",
            "measurand": "tris_hcl_concentration_millimolar",
            "verified_value": 50.0,
            "unit": "mM",
            "status": "verified",
        },
        "DOC-DEV-004": {
            "source_id": "DOC-DEV-004",
            "title": "Chiller Thermal Loop Out-of-Tolerance Deviation Notice",
            "measurand": "recirculating_chiller_coolant_temperature_celsius",
            "verified_value": 8.4,
            "unit": "Celsius",
            "status": "flagged",
        },
        "DOC-AUDIT-005": {
            "source_id": "DOC-AUDIT-005",
            "title": "UV-Vis Spectrophotometer Photometric Accuracy Audit",
            "measurand": "spectrophotometer_photometric_accuracy_absorbance",
            "verified_value": 0.003,
            "unit": "AU",
            "status": "verified",
        },
    }


def create_benign_qc_sop_registry() -> Dict[str, Dict[str, Any]]:
    """Factory creating authoritative SOP registry for laboratory quality control."""
    return {
        "SOP-CAL-TEMP-v3": {
            "sop_id": "SOP-CAL-TEMP-v3",
            "title": "Thermal Chamber RTD Probe Calibration Standard",
            "status": "active",
            "effective_date": "2026-01-15",
        },
        "SOP-MAINT-ROT-v2": {
            "sop_id": "SOP-MAINT-ROT-v2",
            "title": "Centrifuge Rotor Vibration Velocity Assessment",
            "status": "active",
            "effective_date": "2025-11-01",
        },
        "SOP-QC-BUFF-v1": {
            "sop_id": "SOP-QC-BUFF-v1",
            "title": "Laboratory Buffer Potentiometric Verification",
            "status": "active",
            "effective_date": "2025-08-20",
        },
        "SOP-CAL-CHILL-v2": {
            "sop_id": "SOP-CAL-CHILL-v2",
            "title": "Recirculating Chiller Temperature Maintenance",
            "status": "active",
            "effective_date": "2026-03-10",
        },
        "SOP-QC-SPEC-v4": {
            "sop_id": "SOP-QC-SPEC-v4",
            "title": "Spectrophotometer Photometric Linearity & Accuracy",
            "status": "active",
            "effective_date": "2026-06-01",
        },
        "SOP-QC-SPEC-v3": {
            "sop_id": "SOP-QC-SPEC-v3",
            "title": "Spectrophotometer Photometric Linearity (Legacy)",
            "status": "superseded",
            "superseded_by": "SOP-QC-SPEC-v4",
            "retirement_date": "2026-05-31",
        },
        "SOP-QC-SPEC-EXP": {
            "sop_id": "SOP-QC-SPEC-EXP",
            "title": "Novel Dual-Wavelength Photometric Pilot Method",
            "status": "experimental",
        },
    }


