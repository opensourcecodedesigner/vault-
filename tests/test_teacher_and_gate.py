"""Unit tests for VAULT Teacher LLM Interface and Six-Check Evidence Gate.

Validates:
1. Teacher interface correctness, fixture generation, provenance, and hardware memory guard.
2. Strict adherence to authoritative truth ledger records without scientific fact overrides.
3. Six-check evidence gate:
   - Check 1: Missing source detection -> ERR_SOURCE_NOT_FOUND
   - Check 2: Empty/irrelevant span detection -> ERR_SPAN_EMPTY / ERR_SPAN_IRRELEVANT
   - Check 3: Unsupported claim / conflicting verdict -> ERR_CLAIM_UNSUPPORTED
   - Check 4: Mathematical mismatch, nonfinite values, incompatible units -> ERR_NUMERIC_MISMATCH / ERR_UNIT_INCOMPATIBLE
   - Check 5: Multi-source contradiction -> ERR_SOURCE_CONTRADICTION
   - Check 6: Superseded SOP -> ERR_SOP_SUPERSEDED; experimental method -> QUARANTINE
4. Controlled label corruption benchmark across 0%, 5%, 10%, 20% corruption tiers.
5. Invariance: No silent mutation of input candidate facts.
"""
from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vault_ml.evidence_gate import (
    EvidenceGate,
    GateDecisionType,
    ReasonCode,
    benchmark_gate_corruption,
    create_benign_qc_catalog,
    create_benign_qc_sop_registry,
    inject_label_corruption,
)
from vault_ml.teacher import (
    TEACHER_LICENSE,
    TEACHER_MODEL_ID,
    TEACHER_REVISION,
    TeacherClient,
    TeacherHardwareError,
    TruthLedgerRecord,
    batch_generate_teacher_dataset,
    create_benign_qc_records,
)


class TeacherInterfaceTests(unittest.TestCase):
    """Test modular teacher interface behavior, fixture mode, and hardware guards."""

    def setUp(self):
        self.records = create_benign_qc_records()
        self.client = TeacherClient(mode="fixture")

    def test_teacher_fixture_generation_benign_qc(self):
        """Teacher fixture mode synthesizes candidate questions and notes from truth records."""
        for rec in self.records:
            candidate = self.client.generate_candidate(rec)
            self.assertEqual(candidate.ledger_id, rec.ledger_id)
            self.assertEqual(candidate.scenario_family, rec.scenario_family)
            self.assertTrue(len(candidate.candidate_question) > 10)
            self.assertIn(rec.source_id, candidate.source_ids)
            self.assertEqual(candidate.target_response["decision"], rec.decision)
            self.assertEqual(candidate.target_response["correct_value"], rec.verified_value)
            self.assertEqual(candidate.target_response["unit"], rec.unit)
            self.assertEqual(candidate.target_response["measurand"], rec.measurand)

    def test_teacher_preserves_truth_verdict_strictly(self):
        """Authoritative truth ledger records can never be silently overridden by the teacher."""
        for rec in self.records:
            candidate = self.client.generate_candidate(rec)
            target = candidate.target_response
            # Verifies exact numerical match
            if rec.verified_value is not None:
                self.assertAlmostEqual(target["correct_value"], rec.verified_value, places=6)
            else:
                self.assertIsNone(target["correct_value"])
            self.assertEqual(target["unit"], rec.unit)
            self.assertEqual(target["decision"], rec.decision)

    def test_teacher_provenance_and_immutability(self):
        """Outputs include model metadata, generator flag, and deterministic content hash."""
        rec = self.records[0]
        out1 = self.client.generate_candidate(rec)
        out2 = self.client.generate_candidate(rec)
        self.assertEqual(out1.content_hash, out2.content_hash)
        self.assertEqual(out1.metadata["teacher_model_id"], TEACHER_MODEL_ID)
        self.assertEqual(out1.metadata["teacher_revision"], TEACHER_REVISION)
        self.assertEqual(out1.metadata["teacher_license"], TEACHER_LICENSE)
        self.assertEqual(out1.metadata["generator"], "deterministic_fixture_non_llm")

    def test_teacher_hardware_guard(self):
        """TeacherClient detects when physical VRAM is below the 8B co-existence budget."""
        with patch("torch.cuda.is_available", return_value=True):
            class FakeProps:
                total_memory = 6 * (1024 ** 3)  # 6 GB
            with patch("torch.cuda.get_device_properties", return_value=FakeProps()):
                with self.assertRaises(TeacherHardwareError):
                    TeacherClient.verify_local_hardware_budget()

    def test_teacher_batch_generation_manifest(self):
        """Batch generation outputs candidates and writes immutable manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "candidates.json"
            results = batch_generate_teacher_dataset(self.records, self.client, out_path)
            self.assertEqual(len(results), len(self.records))
            self.assertTrue(out_path.exists())


class EvidenceGateTests(unittest.TestCase):
    """Test deterministic six-check evidence gate and machine-readable reason codes."""

    def setUp(self):
        self.catalog = create_benign_qc_catalog()
        self.sop_registry = create_benign_qc_sop_registry()
        self.gate = EvidenceGate(source_catalog=self.catalog, sop_registry=self.sop_registry)
        self.teacher = TeacherClient(mode="fixture")
        self.qc_records = create_benign_qc_records()
        self.clean_candidates = [
            self.teacher.generate_candidate(r).__dict__ for r in self.qc_records
        ]

    def test_gate_accepts_clean_benign_records(self):
        """All clean benign QC candidates pass all 6 checks with ACCEPT decision."""
        # Test clean records that have corresponding catalog and SOP entries
        for cand in self.clean_candidates:
            res = self.gate.evaluate(cand)
            self.assertEqual(res.decision, GateDecisionType.ACCEPT)
            self.assertIn("CHECK_1_SOURCE_EXISTS", res.checks_passed)
            self.assertIn("CHECK_2_SPAN_RELEVANT", res.checks_passed)
            self.assertIn("CHECK_3_CLAIM_SUPPORTED", res.checks_passed)
            self.assertIn("CHECK_4_MATH_UNITS_CORRECT", res.checks_passed)
            self.assertIn("CHECK_5_MULTI_SOURCE_CONSISTENT", res.checks_passed)
            self.assertIn("CHECK_6_SOP_VERSION_APPLICABLE", res.checks_passed)
            self.assertIn(ReasonCode.PASS_ALL_CHECKS, res.reason_codes)

    def test_gate_check_1_missing_source(self):
        """Missing or uncited source produces REJECT with ERR_SOURCE_NOT_FOUND."""
        cand = copy.deepcopy(self.clean_candidates[0])
        cand["source_ids"] = ["DOC-NON-EXISTENT-999"]
        res = self.gate.evaluate(cand)
        self.assertEqual(res.decision, GateDecisionType.REJECT)
        self.assertIn("CHECK_1_SOURCE_EXISTS", res.checks_failed)
        self.assertIn(ReasonCode.ERR_SOURCE_NOT_FOUND, res.reason_codes)

    def test_gate_check_2_empty_or_irrelevant_span(self):
        """Empty evidence span produces ERR_SPAN_EMPTY."""
        cand = copy.deepcopy(self.clean_candidates[0])
        cand["evidence_spans"] = ["   "]
        res = self.gate.evaluate(cand)
        self.assertIn("CHECK_2_SPAN_RELEVANT", res.checks_failed)
        self.assertIn(ReasonCode.ERR_SPAN_EMPTY, res.reason_codes)

    def test_gate_check_3_unsupported_claim(self):
        """Claiming pass when evidence records out-of-tolerance deviation produces ERR_CLAIM_UNSUPPORTED."""
        cand = copy.deepcopy(self.clean_candidates[0])
        cand["target_response"]["decision"] = "pass"
        cand["evidence_spans"] = ["CALIBRATION DEVIATION: Incubator temperature out_of_tolerance."]
        res = self.gate.evaluate(cand)
        self.assertIn("CHECK_3_CLAIM_SUPPORTED", res.checks_failed)
        self.assertIn(ReasonCode.ERR_CLAIM_UNSUPPORTED, res.reason_codes)

    def test_gate_check_4_incorrect_math(self):
        """Numerical mismatch produces REJECT with ERR_NUMERIC_MISMATCH."""
        cand = copy.deepcopy(self.clean_candidates[0])
        # Catalog has 37.1 Celsius, mutate to 42.5
        cand["target_response"]["correct_value"] = 42.5
        res = self.gate.evaluate(cand)
        self.assertEqual(res.decision, GateDecisionType.REJECT)
        self.assertIn("CHECK_4_MATH_UNITS_CORRECT", res.checks_failed)
        self.assertIn(ReasonCode.ERR_NUMERIC_MISMATCH, res.reason_codes)

    def test_gate_check_4_incompatible_unit(self):
        """Incompatible measurement unit produces REJECT with ERR_UNIT_INCOMPATIBLE."""
        cand = copy.deepcopy(self.clean_candidates[0])
        cand["target_response"]["unit"] = "kg/m3"
        res = self.gate.evaluate(cand)
        self.assertEqual(res.decision, GateDecisionType.REJECT)
        self.assertIn("CHECK_4_MATH_UNITS_CORRECT", res.checks_failed)
        self.assertIn(ReasonCode.ERR_UNIT_INCOMPATIBLE, res.reason_codes)

    def test_gate_check_4_nonfinite_numeric_value(self):
        """NaN or Inf numeric values produce REJECT with ERR_NONFINITE_VALUE."""
        cand = copy.deepcopy(self.clean_candidates[0])
        cand["target_response"]["correct_value"] = float("nan")
        res = self.gate.evaluate(cand)
        self.assertEqual(res.decision, GateDecisionType.REJECT)
        self.assertIn(ReasonCode.ERR_NONFINITE_VALUE, res.reason_codes)

    def test_gate_check_5_conflicting_measurements(self):
        """Contradictory measurements across cited sources produce REJECT with ERR_SOURCE_CONTRADICTION."""
        cand = copy.deepcopy(self.clean_candidates[0])
        # Cite both incubator (37.1 C) and chiller (8.4 C)
        cand["source_ids"] = ["DOC-CAL-001", "DOC-DEV-004"]
        res = self.gate.evaluate(cand)
        self.assertEqual(res.decision, GateDecisionType.REJECT)
        self.assertIn("CHECK_5_MULTI_SOURCE_CONSISTENT", res.checks_failed)
        self.assertIn(ReasonCode.ERR_SOURCE_CONTRADICTION, res.reason_codes)

    def test_gate_check_6_outdated_sop_version(self):
        """Referencing a superseded SOP version produces REJECT with ERR_SOP_SUPERSEDED."""
        cand = copy.deepcopy(self.clean_candidates[4])  # Spectrophotometer case
        cand["metadata"]["sop_version"] = "SOP-QC-SPEC-v3"  # Superseded
        cand["lab_notes"] = "Test performed under SOP-QC-SPEC-v3"
        res = self.gate.evaluate(cand)
        self.assertEqual(res.decision, GateDecisionType.REJECT)
        self.assertIn("CHECK_6_SOP_VERSION_APPLICABLE", res.checks_failed)
        self.assertIn(ReasonCode.ERR_SOP_SUPERSEDED, res.reason_codes)

    def test_gate_quarantine_alternative_methodology(self):
        """Novel or experimental SOP produces QUARANTINE with WARN_ALTERNATIVE_METHOD."""
        cand = copy.deepcopy(self.clean_candidates[4])
        cand["metadata"]["sop_version"] = "SOP-QC-SPEC-EXP"  # Experimental
        cand["lab_notes"] = "Pilot evaluation under SOP-QC-SPEC-EXP"
        res = self.gate.evaluate(cand)
        self.assertEqual(res.decision, GateDecisionType.QUARANTINE)
        self.assertIn(ReasonCode.WARN_ALTERNATIVE_METHOD, res.reason_codes)

    def test_no_silent_mutation_of_authoritative_facts(self):
        """EvidenceGate.evaluate() never mutates input candidate data."""
        original = copy.deepcopy(self.clean_candidates[0])
        mutated_target = copy.deepcopy(self.clean_candidates[0])
        _ = self.gate.evaluate(mutated_target)
        self.assertEqual(original, mutated_target)

    def test_controlled_label_corruption_benchmark(self):
        """Benchmark gate performance across 0%, 5%, 10%, 20% controlled corruption."""
        # Replicate benign QC candidates to create a 40-record evaluation corpus
        corpus = [copy.deepcopy(c) for _ in range(8) for c in self.clean_candidates]
        benchmark = benchmark_gate_corruption(self.gate, corpus, corruption_rates=(0.0, 0.05, 0.10, 0.20), seed=42)

        self.assertIn("0pct", benchmark)
        self.assertIn("5pct", benchmark)
        self.assertIn("10pct", benchmark)
        self.assertIn("20pct", benchmark)

        # At 0% corruption, precision and clean count should be 100%
        self.assertEqual(benchmark["0pct"]["corrupted_count"], 0)
        self.assertEqual(benchmark["0pct"]["false_positives"], 0)
        self.assertEqual(benchmark["0pct"]["accepts"], len(corpus))

        # At 20% corruption, every injected corrupted record should be rejected/quarantined
        self.assertEqual(benchmark["20pct"]["false_negatives"], 0)
        self.assertEqual(benchmark["20pct"]["recall"], 1.0)
        self.assertEqual(benchmark["20pct"]["f1_score"], 1.0)


class ControlledDatasetPipelineTests(unittest.TestCase):
    """Test scenario-family isolation and controlled dataset construction (Phase 3)."""

    def test_scenario_family_isolation_strictly_disjoint(self):
        """Train, dev, and heldout evaluation scenario families have zero overlap."""
        from vault_ml.dataset_generator import (
            DEV_SCENARIO_FAMILIES,
            HELDOUT_SCENARIO_FAMILIES,
            TRAINING_SCENARIO_FAMILIES,
            assert_scenario_family_isolation,
        )
        assert_scenario_family_isolation()
        self.assertTrue(TRAINING_SCENARIO_FAMILIES.isdisjoint(HELDOUT_SCENARIO_FAMILIES))
        self.assertTrue(DEV_SCENARIO_FAMILIES.isdisjoint(HELDOUT_SCENARIO_FAMILIES))
        self.assertTrue(TRAINING_SCENARIO_FAMILIES.isdisjoint(DEV_SCENARIO_FAMILIES))

    def test_controlled_datasets_construction_and_sizes(self):
        """Pipeline builds M1, M2, M2_control, M2_safety, and smoke datasets."""
        from vault_ml.dataset_generator import generate_controlled_datasets
        with tempfile.TemporaryDirectory() as tmpdir:
            datasets = generate_controlled_datasets(output_root=Path(tmpdir))
            self.assertIn("M1", datasets)
            self.assertIn("M2", datasets)
            self.assertIn("M2_control", datasets)
            self.assertIn("M2_safety", datasets)
            self.assertIn("smoke", datasets)

            # M1 is unfiltered (16 records)
            self.assertEqual(len(datasets["M1"]), 16)
            # M2 is filtered by evidence gate (12 records)
            self.assertEqual(len(datasets["M2"]), 12)
            self.assertTrue(all(x["verifier_approved"] for x in datasets["M2"]))
            # M2_control has the EXACT SAME count as M2 (12 records)
            self.assertEqual(len(datasets["M2_control"]), len(datasets["M2"]))
            # M2_safety has M2 plus benign safety supervision
            self.assertEqual(len(datasets["M2_safety"]), len(datasets["M2"]) + 2)

            # Verify manifest was written
            manifest_files = list(Path(tmpdir).glob("*/manifest.json")) + list(Path(tmpdir).glob("manifest.json"))
            self.assertTrue(len(manifest_files) > 0)


if __name__ == "__main__":
    unittest.main()

