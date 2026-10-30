from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
import tempfile
import threading
import unittest
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.us_quant.research_dashboard import (  # noqa: E402
    DashboardError, DashboardHandler, PublishedInputs, build_snapshot, gates, public_projection,
)


class ProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.current = build_snapshot(ROOT)

    def test_actual_inventory_and_scope_counts(self) -> None:
        data = self.current
        progress = json.loads((ROOT / "docs/research_progress.json").read_text())
        state = json.loads((ROOT / progress["recurring_research_program"]["cycle_snapshot"]).read_text())
        self.assertEqual(data["summary"]["completed"], state["total_evaluated_configurations"])
        self.assertEqual(data["summary"]["reviews"], len(state["candidate_reviews"]))
        self.assertEqual(data["summary"]["legacy"], 84)
        self.assertEqual(data["summary"]["registered"], len(state["registered_candidates"]))
        self.assertEqual(len(data["strategies"]), 84 + len(state["registered_candidates"]))
        self.assertEqual(data["summary"]["technical_failures"], 2)
        self.assertEqual(data["summary"]["definitions"], 6)
        self.assertEqual(data["summary"]["economic_families"], 5)
        budget = progress["approved_bounded_optimization_round"]
        self.assertEqual(data["budget"]["cap"], 6)
        self.assertEqual(data["budget"]["used"], budget["registered_configurations_consumed"])
        self.assertEqual(data["budget"]["remaining"], 6 - data["budget"]["used"])

    def test_ranking_is_not_a_champion_or_a_legacy_winner(self) -> None:
        data = self.current
        eligible = [row for row in data["strategies"] if row["eligible_for_current_ranking"]
                    and row["max_drawdown"] <= .15 and row["min_sharpe"] is not None]
        expected = sorted(eligible, key=lambda row: (-row["min_sharpe"], row["id"]))[0]
        self.assertEqual(data["leader_id"], expected["id"])
        self.assertFalse(expected["forward_validated"])
        self.assertFalse(data["order_authority"])
        self.assertTrue(all(not row["eligible_for_current_ranking"] for row in data["strategies"] if row["scope"] == "legacy"))

    def test_strict_thresholds_are_not_rounded(self) -> None:
        benchmark = {"cagr": .15}
        values = {"sharpe": 1.0, "max_drawdown": .15, "cagr": .15}
        result = gates(values, benchmark, benchmark, "stress")
        self.assertFalse(result["sharpe_above_1"])
        self.assertTrue(result["drawdown_at_most_15pct"])
        self.assertFalse(result["beats_spy"])
        self.assertFalse(result["also_beats_base_cost_spy"])
        result = gates({**values, "sharpe": .999999999}, benchmark, benchmark, "base")
        self.assertFalse(result["sharpe_above_1"])

    def test_input_failure_and_real_forward_days_are_separate(self) -> None:
        progress = json.loads((ROOT / "docs/research_progress.json").read_text())
        reference = progress["daily_operational_execution"]["actual_run"]
        actual = json.loads((ROOT / reference).read_text())
        self.assertEqual(self.current["daily"]["new_sessions"], 0)
        self.assertEqual(self.current["daily"]["evidence"], reference)
        self.assertEqual(self.current["daily"]["session"], actual["market_session"])
        self.assertEqual(self.current["daily"]["status"], actual["status"])
        self.assertEqual(self.current["daily"]["reason"], actual["reason"])
        self.assertEqual(self.current["daily"]["latest_accepted_session"], "2026-10-12")
        for model in self.current["forward"]:
            self.assertEqual(model["sessions"], 1)
            self.assertIsNone(model["sharpe"])
            self.assertIsNone(model["cagr"])
        operations = [row for row in self.current["history"] if row["kind"] == "operation"]
        contract = json.loads((ROOT / "config/publication_allowlist.json").read_text())
        references = {
            name for name in contract["files"]
            if Path(name).name.startswith(("research_daily_actual_run_", "research_daily_recovery_"))
            and "_preservation_" not in name
        }
        self.assertEqual({row["id"] for row in operations}, references)
        expected_failures = 0
        for name in references:
            record = json.loads((ROOT / name).read_text())
            execution = record["execution_record"] if "execution_record" in record else record
            expected_failures += "failed" in execution["status"]
        self.assertEqual(len([row for row in operations if "failed" in row["status"]]), expected_failures)

    def test_projection_never_publishes_private_input_or_account_rows(self) -> None:
        raw = json.dumps(self.current)
        for forbidden in ("frozen_files", "market-open.csv", "raw_close", "account_number",
                          "receipt_chain_head", "webhook", "broker_positions", "target_weights"):
            self.assertNotIn('"' + forbidden + '"', raw)
        self.assertTrue(all(name.startswith(("config/", "docs/", "research/workbench/config/",
                                            "research/workbench/evidence/"))
                            for name in self.current["input_sha256"]))

    def fixture(self) -> Path:
        temporary = tempfile.TemporaryDirectory(prefix="research_dashboard_")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        contract = json.loads((ROOT / "config/publication_allowlist.json").read_text())
        names = set(self.current["input_sha256"]) | {
            "config/publication_allowlist.json", contract["bundle_manifest"],
        }
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, path)
        return root

    def change(self, root: Path, name: str, mutate) -> None:
        path = root / name
        value = json.loads(path.read_text())
        mutate(value)
        path.write_text(json.dumps(value))

    def change_ssd_result(self, root: Path, mutate) -> None:
        progress = json.loads((root / "docs/research_progress.json").read_text())
        node = progress["ssd_continuous_research_v1"]
        self.change(root, node["first_pair_results"], mutate)
        node["first_pair_results_sha256"] = hashlib.sha256((root / node["first_pair_results"]).read_bytes()).hexdigest()
        (root / "docs/research_progress.json").write_text(json.dumps(progress))

    def change_ssd_catalog(self, root: Path, mutate) -> None:
        progress = json.loads((root / "docs/research_progress.json").read_text())
        node = progress["ssd_continuous_research_v1"]["factor_catalog_extension"]
        self.change(root, node["evidence"], mutate)
        node["sha256"] = hashlib.sha256((root / node["evidence"]).read_bytes()).hexdigest()
        (root / "docs/research_progress.json").write_text(json.dumps(progress))

    def test_ssd_catalog_is_separate_from_original_definitions_and_strategy_counts(self) -> None:
        ssd = self.current["ssd_research"]
        catalog = ssd["factor_catalog_extension"]
        self.assertEqual(self.current["summary"]["definitions"], 6)
        self.assertEqual(self.current["summary"]["economic_families"], 5)
        self.assertEqual(self.current["queued"]["definitions"], 2)
        self.assertEqual(catalog["new_definition_count"], 2)
        self.assertEqual(catalog["combined_definition_count"], 8)
        self.assertEqual(catalog["combined_family_count"], 7)
        self.assertEqual(catalog["strategy_evaluations_by_admission"], 0)
        self.assertEqual(catalog["source_sessions"], 2805)
        self.assertEqual(catalog["preserved_asset_count"], 12)
        self.assertEqual(catalog["appended_symbols"], ["IJR", "PKW"])
        added = {row["id"] for row in catalog["new_factors"]}
        self.assertEqual(added, {"size_exposure_ijr_v1", "net_buyback_exposure_pkw_v1"})
        for factor in self.current["factors"]:
            if factor["id"] in added:
                self.assertEqual(factor["status"], "queued")
                self.assertEqual(factor["strategy_count"], 0)
                self.assertEqual(factor["ssd_registration"]["evidence"], catalog["evidence"])
            else:
                self.assertNotIn("ssd_registration", factor)
        root = self.fixture()
        self.change(root, "docs/research_progress.json",
                    lambda value: value["ssd_continuous_research_v1"].pop("factor_catalog_extension"))
        without = build_snapshot(root)
        self.assertNotIn("factor_catalog_extension", without["ssd_research"])
        for key in ("registered", "completed", "reviews", "pending", "technical_failures",
                    "account_paths", "qualified", "total_legacy_plus_ssd_completed"):
            self.assertEqual(without["ssd_research"][key], ssd[key])
        self.assertEqual(without["summary"], self.current["summary"])

    def test_ssd_catalog_hash_change_is_not_silently_trusted(self) -> None:
        root = self.fixture()
        node = json.loads((root / "docs/research_progress.json").read_text())
        reference = node["ssd_continuous_research_v1"]["factor_catalog_extension"]
        self.change(root, reference["evidence"], lambda value: value.update(new_definition_count=3))
        with self.assertRaisesRegex(DashboardError, "frozen_policy_hash_mismatch"):
            build_snapshot(root)

    def test_ssd_catalog_cannot_claim_changed_identity_clock_scope_or_authority(self) -> None:
        mutations = [
            lambda value: value.update(new_definition_count=3),
            lambda value: value.update(combined_definition_count=9),
            lambda value: value.update(combined_family_count=8),
            lambda value: value.update(new_strategy_evaluations=1),
            lambda value: value.update(new_strategy_evaluations=False),
            lambda value: value.update(schema_version=True),
            lambda value: value.update(independent_alpha_or_forward_proof=True),
            lambda value: value.update(original_ledger_written_or_migrated=True),
            lambda value: value.update(order_authority=True),
            lambda value: value.update(week="2026-W42"),
            lambda value: value["new_factors"][0]["factor"].update(family="momentum"),
            lambda value: value["new_factors"][0].update(semantic_sha256="0" * 64),
            lambda value: value["new_factors"][0].update(registered_at="2026-10-10T00:00:00Z"),
            lambda value: value["legacy"].update(original_cycle_for_new_week_exists=True),
            lambda value: value["admission"].update(checked_at="2026-10-10T21:31:12Z"),
            lambda value: value["admission"].update(original_price_arrays_exactly_preserved=False),
            lambda value: value["admission"].update(original_risk_free_reference_and_array_preserved=1),
            lambda value: value.pop("registered_before_new_candidate_selection"),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index):
                root = self.fixture()
                self.change_ssd_catalog(root, mutate)
                with self.assertRaisesRegex(DashboardError, "ssd_catalog_contract_mismatch"):
                    build_snapshot(root)

    def test_ssd_counts_benchmarks_and_legacy_quota_are_separate(self) -> None:
        ssd = self.current["ssd_research"]
        first = ssd["cohorts"][0]
        self.assertEqual((first["registered"], first["completed"], first["qualified"]), (2, 2, 0))
        for key in ("registered", "completed", "pending", "technical_failures", "account_paths"):
            self.assertEqual(ssd[key], sum(cohort[key] for cohort in ssd["cohorts"]))
        self.assertIsNone(ssd["configuration_count_cap_before_deadline"])
        self.assertEqual((self.current["budget"]["cap"], self.current["budget"]["remaining"]), (6, 0))
        self.assertEqual(self.current["summary_scope"], "preserved_pre_ssd_research")
        original = json.loads((ROOT / first["registration_evidence"]).read_text())
        original_ids = {row["candidate_id"] for row in original["entries"]}
        for candidate in (row for row in ssd["candidates"] if row["id"] in original_ids):
            self.assertEqual(candidate["status"], "rejected_historical")
            self.assertEqual(len(candidate["paths"]), 4)
            for path in candidate["paths"]:
                self.assertEqual(path["gates"]["beats_qqq"],
                                 path["metrics"]["cagr"] > path["qqq_benchmark"]["cagr"])
                self.assertEqual(path["gates"]["cagr_above_12pct"], path["metrics"]["cagr"] > .12)
                self.assertFalse(path["gates"]["sharpe_above_1"])
                self.assertFalse(path["gates"]["drawdown_at_most_15pct"])
                self.assertGreater(path["execution"]["total_cost_usd"], 0)

    def synthetic_ssd_cohort(self, *, completed: bool = False) -> Path:
        root = self.fixture()
        progress = json.loads((root / "docs/research_progress.json").read_text())
        node = progress["ssd_continuous_research_v1"]
        packet = json.loads((root / node["first_pair_registration"]).read_text())
        study_id = "synthetic_commitment_study"
        packet.update(schema_version=2, study_id=study_id, independent_forward_validation=False,
                      candidate_accounts_or_metrics_computed=False)
        identifiers = ("synthetic_commitment_a", "synthetic_commitment_b")
        for identifier, entry in zip(identifiers, packet["entries"]):
            entry["candidate_id"] = entry["specification"]["candidate"]["id"] = identifier
            entry["specification"]["frozen"]["study_id"] = study_id
        policy_path = "research/workbench/config/synthetic-commitment.json"
        policy = {"study_id": study_id, "candidates": [row["specification"]["candidate"] for row in packet["entries"]]}
        (root / policy_path).write_text(json.dumps(policy))
        policy_sha = hashlib.sha256((root / policy_path).read_bytes()).hexdigest()
        for entry in packet["entries"]:
            entry["specification"]["frozen"]["policy"] = {"path": "config/synthetic-commitment.json", "sha256": policy_sha}
        packet_path = "research/workbench/evidence/synthetic_commitment_registration.json"
        (root / packet_path).write_text(json.dumps(packet))
        packet_sha = hashlib.sha256((root / packet_path).read_bytes()).hexdigest()
        item = {"study_id": study_id, "registration": packet_path, "registration_sha256": packet_sha,
                "results": None, "results_sha256": None, "preregistration_commit": None}
        total = {"registered": 4, "completed": 2, "reviews": 2, "pending": 2,
                 "technical_failures": 0, "qualified": 0, "account_paths": 8,
                 "total_legacy_plus_ssd_completed": 180}
        allowed = [policy_path, packet_path]
        if completed:
            result = json.loads((root / node["first_pair_results"]).read_text())
            result.update(schema_version=2, study_id=study_id, new_completed_strategy_evaluations=2,
                          new_pending_candidates=0)
            result["registration_packet"] = {"path": packet_path[len("research/workbench/"):], "sha256": packet_sha}
            result["actual_publication_before_accounts"]["registration_packet"] = result["registration_packet"]
            result["actual_publication_before_accounts"]["published_commit"] = "b" * 40
            for identifier, review in zip(identifiers, result["reviews"]):
                review["candidate_id"] = identifier
            result_path = "research/workbench/evidence/synthetic_commitment_results.json"
            (root / result_path).write_text(json.dumps(result))
            item.update(results=result_path, results_sha256=hashlib.sha256((root / result_path).read_bytes()).hexdigest(),
                        preregistration_commit="b" * 40)
            total.update(completed=4, reviews=4, pending=0, account_paths=16,
                         total_legacy_plus_ssd_completed=182)
            allowed.append(result_path)
        node.update(additional_cohorts=[item], aggregate_counts=total)
        (root / "docs/research_progress.json").write_text(json.dumps(progress))
        self.change(root, "config/publication_allowlist.json", lambda contract: contract["files"].extend(allowed))
        return root

    def test_ssd_new_pending_cohort_does_not_invent_accounts_or_replace_old_results(self) -> None:
        root = self.synthetic_ssd_cohort()
        result = build_snapshot(root)
        ssd = result["ssd_research"]
        self.assertEqual((ssd["registered"], ssd["completed"], ssd["pending"]), (4, 2, 2))
        self.assertEqual(ssd["status"], "cohort_evaluations_pending")
        self.assertIsNone(ssd["completed_at"])
        self.assertIsNone(ssd["study_id"])
        self.assertEqual(result["summary"], self.current["summary"])
        self.assertEqual(result["forward"], self.current["forward"])
        for candidate in ssd["candidates"][2:]:
            self.assertEqual(candidate["status"], "pending")
            self.assertFalse(candidate["full_gate_pass"])
            self.assertEqual(len(candidate["paths"]), 4)
            self.assertTrue(all(path["metrics"] is None for path in candidate["paths"]))

    def synthetic_preparation_failure(self) -> Path:
        root = self.synthetic_ssd_cohort()
        progress = json.loads((root / "docs/research_progress.json").read_text())
        node = progress["ssd_continuous_research_v1"]
        item = node["additional_cohorts"][0]
        path = root / item["registration"]
        packet = json.loads(path.read_text())
        packet.update(
            schema_version=3, record_type="registered_preparation_failure",
            target_proof_completed=False, new_completed_configurations=0,
            new_completed_strategy_evaluations=0, new_reviews=0,
            new_technical_failures=2, new_pending_candidates=0,
            new_historical_point_gate_passes=0, new_account_paths=0,
        )
        for entry in packet["entries"]:
            entry.update(
                target_proof=None, target_proof_at=None, status="technical_failure",
                failed_at=packet["created_at"], failure_event_sha256="f" * 64,
                failure_stage="ordered_path_preparation", failure_reason="synthetic numerical failure",
                candidate_accounts_started=False, partial_outputs_retained=True, paths=[],
            )
        path.write_text(json.dumps(packet))
        item["registration_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        node["aggregate_counts"].update(pending=0, technical_failures=2)
        (root / "docs/research_progress.json").write_text(json.dumps(progress))
        return root

    def test_ssd_preparation_failure_keeps_registration_without_inventing_target_proof(self) -> None:
        result = build_snapshot(self.synthetic_preparation_failure())
        ssd = result["ssd_research"]
        self.assertEqual((ssd["registered"], ssd["completed"], ssd["technical_failures"],
                          ssd["pending"], ssd["account_paths"]), (4, 2, 2, 0, 8))
        for candidate in ssd["candidates"][2:]:
            self.assertEqual(candidate["status"], "technical_failure")
            self.assertFalse(candidate["full_gate_pass"])
            self.assertTrue(all(path["metrics"] is None and path["execution"] is None
                                and path["gates"] == {} for path in candidate["paths"]))
        self.assertEqual(result["summary"], self.current["summary"])
        self.assertEqual(result["forward"], self.current["forward"])

    def test_ssd_preparation_failure_rejects_proof_account_count_or_clock_fabrication(self) -> None:
        mutations = [
            lambda packet: packet["entries"][0].update(target_proof={"invented": True}),
            lambda packet: packet["entries"][0].update(candidate_accounts_started=True),
            lambda packet: packet["entries"][0].update(failed_at="2000-01-01T00:00:00Z"),
            lambda packet: packet["entries"][0].update(paths=[{"invented": True}]),
            lambda packet: packet.update(new_completed_strategy_evaluations=1),
            lambda packet: packet.update(candidate_accounts_or_metrics_computed=True),
            lambda packet: packet["entries"][0].update(failure_stage="account_evaluation"),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                root = self.synthetic_preparation_failure()
                progress = json.loads((root / "docs/research_progress.json").read_text())
                item = progress["ssd_continuous_research_v1"]["additional_cohorts"][0]
                path = root / item["registration"]
                packet = json.loads(path.read_text())
                mutate(packet)
                path.write_text(json.dumps(packet))
                item["registration_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                (root / "docs/research_progress.json").write_text(json.dumps(progress))
                with self.assertRaises(DashboardError):
                    build_snapshot(root)

    def covariance_preparation_failure(self, *, stage: str, study_id: str) -> Path:
        root = self.synthetic_preparation_failure()
        progress = json.loads((root / "docs/research_progress.json").read_text())
        item = progress["ssd_continuous_research_v1"]["additional_cohorts"][0]
        path = root / item["registration"]
        packet = json.loads(path.read_text())
        packet["study_id"] = item["study_id"] = study_id
        for entry in packet["entries"]:
            entry["specification"]["frozen"]["study_id"] = study_id
            entry["failure_stage"] = stage
        path.write_text(json.dumps(packet))
        item["registration_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (root / "docs/research_progress.json").write_text(json.dumps(progress))
        return root

    def test_ssd_covariance_failure_has_exact_stage_without_target_or_account(self) -> None:
        root = self.covariance_preparation_failure(
            stage="expert_full_covariance_preparation",
            study_id="ssd_expert_full_covariance_risk_budget_20261030_v1",
        )
        result = build_snapshot(root)
        cohort = result["ssd_research"]["cohorts"][-1]
        self.assertTrue(cohort["terminal_preparation_failure_without_target_proof"])
        self.assertEqual((cohort["completed"], cohort["account_paths"]), (0, 0))
        candidates = result["ssd_research"]["candidates"][2:]
        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(c["status"] == "technical_failure" for c in candidates))
        self.assertEqual(result["forward"], self.current["forward"])

    def test_ssd_new_preparation_stage_cannot_be_mislabelled_or_broadened(self) -> None:
        cases = [
            ("ordered_path_preparation", "ssd_expert_full_covariance_risk_budget_20261030_v1"),
            ("expert_full_covariance_preparation", "unregistered_other_study"),
            ("account_evaluation", "ssd_expert_full_covariance_risk_budget_20261030_v1"),
            ("ordered_path_preparation", "ssd_expert_headroom_stress_20261030_v1"),
            ("account_evaluation", "ssd_expert_headroom_stress_20261030_v1"),
            ("expert_full_covariance_preparation", "ssd_expert_headroom_stress_20261030_v1"),
            ("expert_headroom_stress_transition_preparation", "unregistered_other_study"),
            ("expert_headroom_stress_transition_preparation",
             "ssd_expert_full_covariance_risk_budget_20261030_v1"),
        ]
        for stage, study_id in cases:
            with self.subTest(stage=stage, study_id=study_id):
                root = self.covariance_preparation_failure(stage=stage, study_id=study_id)
                with self.assertRaisesRegex(DashboardError, "ssd_preparation_failure_record_mismatch"):
                    build_snapshot(root)

    def test_ssd_headroom_stress_failure_keeps_both_rules_without_target_or_account(self) -> None:
        root = self.covariance_preparation_failure(
            stage="expert_headroom_stress_transition_preparation",
            study_id="ssd_expert_headroom_stress_20261030_v1",
        )
        result = build_snapshot(root)
        cohort = result["ssd_research"]["cohorts"][-1]
        self.assertTrue(cohort["terminal_preparation_failure_without_target_proof"])
        self.assertEqual((cohort["registered"], cohort["technical_failures"],
                          cohort["completed"], cohort["account_paths"]), (2, 2, 0, 0))
        for candidate in result["ssd_research"]["candidates"][2:]:
            self.assertEqual(candidate["failure_stage"],
                             "expert_headroom_stress_transition_preparation")
            self.assertFalse(candidate["full_gate_pass"])
            self.assertTrue(all(path["metrics"] is None and path["execution"] is None
                                and path["gates"] == {} for path in candidate["paths"]))
        self.assertEqual(result["summary"], self.current["summary"])
        self.assertEqual(result["forward"], self.current["forward"])

    def change_pending_ssd_registration(self, root: Path, mutate) -> None:
        progress = json.loads((root / "docs/research_progress.json").read_text())
        item = progress["ssd_continuous_research_v1"]["additional_cohorts"][0]
        self.assertIsNone(item["results"])
        self.change(root, item["registration"], mutate)
        item["registration_sha256"] = hashlib.sha256(
            (root / item["registration"]).read_bytes()).hexdigest()
        (root / "docs/research_progress.json").write_text(json.dumps(progress))

    def test_ssd_explicit_packet_account_boundary_covers_entries_without_duplicate_flag(self) -> None:
        root = self.synthetic_ssd_cohort()
        def mutate(packet):
            self.assertIs(packet["candidate_accounts_or_metrics_computed"], False)
            for entry in packet["entries"]:
                entry["target_proof"].pop("candidate_accounts_or_metrics_computed")
        self.change_pending_ssd_registration(root, mutate)
        ssd = build_snapshot(root)["ssd_research"]
        self.assertEqual((ssd["completed"], ssd["pending"], ssd["account_paths"]), (2, 2, 8))
        self.assertTrue(all(path["metrics"] is None
                            for row in ssd["candidates"][2:] for path in row["paths"]))

    def test_ssd_absent_or_conflicting_account_boundary_is_not_assumed_safe(self) -> None:
        for damage in ("missing_packet", "true_packet", "null_packet", "true_entry", "null_entry"):
            with self.subTest(damage=damage):
                root = self.synthetic_ssd_cohort()
                def mutate(packet):
                    key = "candidate_accounts_or_metrics_computed"
                    if damage == "missing_packet":
                        packet.pop(key)
                    elif damage.endswith("packet"):
                        packet[key] = True if damage == "true_packet" else None
                    else:
                        packet["entries"][0]["target_proof"][key] = (
                            True if damage == "true_entry" else None)
                self.change_pending_ssd_registration(root, mutate)
                with self.assertRaisesRegex(DashboardError, "ssd_cohort_registration_mismatch"):
                    build_snapshot(root)

    def test_ssd_second_completed_cohort_is_counted_without_reclassifying_first(self) -> None:
        root = self.synthetic_ssd_cohort(completed=True)
        result = build_snapshot(root)
        self.assertEqual(result["ssd_research"]["completed"], 4)
        self.assertEqual(result["ssd_research"]["account_paths"], 16)
        self.assertEqual(result["ssd_research"]["status"], "cohort_evaluations_finished")
        self.assertEqual(result["ssd_research"]["completed_at"],
                         max(row["completed_at"] for row in result["ssd_research"]["cohorts"]))
        self.assertEqual(result["ssd_research"]["cohorts"][0]["completed"], 2)
        self.assertEqual(result["ssd_research"]["total_legacy_plus_ssd_completed"], 182)
        self.assertEqual(result["budget"], self.current["budget"])

    def test_ssd_partial_technical_failure_is_not_completed_or_silently_removed(self) -> None:
        root = self.synthetic_ssd_cohort(completed=True)
        progress = json.loads((root / "docs/research_progress.json").read_text())
        node = progress["ssd_continuous_research_v1"]
        item = node["additional_cohorts"][0]
        source = json.loads((root / item["results"]).read_text())
        failed = source["reviews"][0]
        failed.update(status="technical_failure", failure_event_sha256="f" * 64, failure_stage="synthetic",
                      all_four_paths_same_candidate_and_frozen_parameters=False)
        failed["paths"] = failed["paths"][:1]
        source.update(new_completed_configurations=1, new_completed_strategy_evaluations=1,
                      new_reviews=1, new_technical_failures=1, new_account_paths=5)
        (root / item["results"]).write_text(json.dumps(source))
        item["results_sha256"] = hashlib.sha256((root / item["results"]).read_bytes()).hexdigest()
        node["aggregate_counts"].update(completed=3, reviews=3, technical_failures=1,
                                        account_paths=13, total_legacy_plus_ssd_completed=181)
        (root / "docs/research_progress.json").write_text(json.dumps(progress))
        actual = build_snapshot(root)["ssd_research"]
        self.assertEqual((actual["completed"], actual["technical_failures"], actual["account_paths"]), (3, 1, 13))
        self.assertEqual(actual["candidates"][2]["status"], "technical_failure")
        self.assertFalse(actual["candidates"][2]["full_gate_pass"])
        self.assertEqual(sum(path["metrics"] is not None for path in actual["candidates"][2]["paths"]), 1)

    def test_ssd_aggregate_counts_cannot_stay_at_first_pair_or_double_count_candidates(self) -> None:
        root = self.synthetic_ssd_cohort()
        self.change(root, "docs/research_progress.json",
                    lambda value: value["ssd_continuous_research_v1"]["aggregate_counts"].update(registered=2))
        with self.assertRaisesRegex(DashboardError, "ssd_aggregate_count_mismatch"):
            build_snapshot(root)
        root = self.synthetic_ssd_cohort()
        self.change(root, "docs/research_progress.json",
                    lambda value: value["ssd_continuous_research_v1"]["additional_cohorts"].append(
                        copy.deepcopy(value["ssd_continuous_research_v1"]["additional_cohorts"][0])))
        with self.assertRaisesRegex(DashboardError, "ssd_duplicate_cohort_or_candidate"):
            build_snapshot(root)

    def test_ssd_new_cohort_requires_its_own_publication_packet_and_complete_review(self) -> None:
        for damage in ("packet", "missing_path", "count", "false_pass"):
            with self.subTest(damage=damage):
                root = self.synthetic_ssd_cohort(completed=True)
                progress = json.loads((root / "docs/research_progress.json").read_text())
                item = progress["ssd_continuous_research_v1"]["additional_cohorts"][0]
                result = json.loads((root / item["results"]).read_text())
                if damage == "packet":
                    result["actual_publication_before_accounts"]["registration_packet"] = {"path": "old.json", "sha256": "0" * 64}
                elif damage == "missing_path":
                    result["reviews"][0]["paths"].pop()
                elif damage == "count":
                    result["new_completed_configurations"] = 1
                else:
                    result["reviews"][0]["status"] = "historical_point_gates_passed"
                (root / item["results"]).write_text(json.dumps(result))
                item["results_sha256"] = hashlib.sha256((root / item["results"]).read_bytes()).hexdigest()
                (root / "docs/research_progress.json").write_text(json.dumps(progress))
                with self.assertRaises(DashboardError):
                    build_snapshot(root)

    def test_ssd_hash_changed_receipt_is_not_silently_accepted(self) -> None:
        root = self.fixture()
        node = json.loads((root / "docs/research_progress.json").read_text())["ssd_continuous_research_v1"]
        self.change(root, node["first_pair_results"],
                    lambda result: result["reviews"][0]["paths"][0]["metrics"].update(cagr=.99))
        with self.assertRaisesRegex(DashboardError, "frozen_policy_hash_mismatch"):
            build_snapshot(root)

    def test_ssd_strict_boundaries_cannot_be_reported_as_passes(self) -> None:
        first = self.current["ssd_research"]["candidates"][0]["paths"][0]
        for metric, value, gate in (("sharpe", 1.0, "sharpe_above_1"),
                                    ("cagr", .12, "cagr_above_12pct"),
                                    ("cagr", first["qqq_benchmark"]["cagr"], "beats_qqq")):
            with self.subTest(gate=gate):
                root = self.fixture()
                def mutate(result):
                    path = result["reviews"][0]["paths"][0]
                    path["metrics"][metric] = value
                    path["gates"][gate] = True
                self.change_ssd_result(root, mutate)
                with self.assertRaisesRegex(DashboardError, "ssd_gate_metric_mismatch"):
                    build_snapshot(root)

    def test_ssd_cannot_drop_a_path_or_the_qqq_gate(self) -> None:
        root = self.fixture()
        self.change_ssd_result(root, lambda result: result["reviews"][0]["paths"].pop())
        with self.assertRaisesRegex(DashboardError, "ssd_four_formal_paths_required"):
            build_snapshot(root)
        root = self.fixture()
        self.change_ssd_result(root, lambda result: result["reviews"][0]["paths"][0]["gates"].pop("beats_qqq"))
        with self.assertRaisesRegex(DashboardError, "ssd_gate_metric_mismatch"):
            build_snapshot(root)

    def test_ssd_false_fees_borrowing_or_gross_exposure_are_rejected(self) -> None:
        for key, value in (("minimum_cash_usd", -1.0), ("maximum_gross_exposure", 1.01),
                           ("total_cost_usd", 0.0), ("model_order_count", 1.5)):
            with self.subTest(key=key):
                root = self.fixture()
                self.change_ssd_result(root, lambda result: result["reviews"][0]["paths"][0]["execution"].update({key: value}))
                with self.assertRaisesRegex(DashboardError, "ssd_execution_disclosure_mismatch"):
                    build_snapshot(root)

    def test_ssd_cannot_restore_a_quota_or_claim_forward_success(self) -> None:
        for key, value in (("configuration_count_cap_before_deadline", 6),
                           ("independent_forward_validation", True),
                           ("investment_objective_verified", True), ("order_authority", True)):
            with self.subTest(key=key):
                root = self.fixture()
                self.change(root, "docs/research_progress.json",
                            lambda progress: progress["ssd_continuous_research_v1"].update({key: value}))
                with self.assertRaisesRegex(DashboardError, "ssd_protocol_or_authority_mismatch"):
                    build_snapshot(root)

    def test_ssd_receipts_do_not_reclassify_the_legacy_ranking(self) -> None:
        root = self.fixture()
        self.change(root, "docs/research_progress.json", lambda progress: progress.pop("ssd_continuous_research_v1"))
        legacy = build_snapshot(root)
        self.assertNotIn("ssd_research", legacy)
        for key in ("summary", "budget", "leader_id", "qualified_ids", "strategies", "forward"):
            self.assertEqual(legacy[key], self.current[key])

    def test_ssd_missing_benchmark_and_wrong_counts_are_rejected(self) -> None:
        root = self.fixture()
        self.change(root, "docs/research_progress.json",
                    lambda progress: progress["ssd_continuous_research_v1"].update(new_reviews=3))
        with self.assertRaisesRegex(DashboardError, "ssd_actual_count_mismatch"):
            build_snapshot(root)
        root = self.fixture()
        progress = json.loads((root / "docs/research_progress.json").read_text())
        node = progress["ssd_continuous_research_v1"]
        self.change(root, node["benchmark_evidence"], lambda value: value["paths"].pop())
        node["benchmark_evidence_sha256"] = hashlib.sha256((root / node["benchmark_evidence"]).read_bytes()).hexdigest()
        (root / "docs/research_progress.json").write_text(json.dumps(progress))
        with self.assertRaisesRegex(DashboardError, "ssd_eight_full_capital_benchmarks_required"):
            build_snapshot(root)

    def test_a_new_review_is_derived_not_hard_coded(self) -> None:
        root = self.fixture()
        reference = self.current["source_snapshot"]
        state = json.loads((root / reference).read_text())
        existing = next(row for row in state["registered_candidates"] if row["id"] == self.current["leader_id"])
        review = next(row for row in state["candidate_reviews"] if row["candidate_id"] == existing["id"])
        extra_spec, extra_review = copy.deepcopy(existing), copy.deepcopy(review)
        extra_spec["id"] = extra_review["candidate_id"] = "synthetic_future_strategy"
        extra_spec["configuration"] = {"id": extra_spec["id"]}
        for path in extra_review["paths"]:
            path["metrics"]["sharpe"] += .05
        state["registered_candidates"].append(extra_spec)
        state["candidate_reviews"].append(extra_review)
        state["total_evaluated_configurations"] += 1
        state["new_evaluated_strategy_count"] += 1
        (root / reference).write_text(json.dumps(state))
        new_count = len(state["candidate_reviews"])
        self.change(root, "docs/research_progress.json",
                    lambda p: p["recurring_research_program"].update(new_strategy_evaluations=new_count))
        def update_combined_count(progress):
            ssd = progress.get("ssd_continuous_research_v1", {})
            if "aggregate_counts" in ssd:
                ssd["aggregate_counts"]["total_legacy_plus_ssd_completed"] += 1
        self.change(root, "docs/research_progress.json", update_combined_count)
        audit = json.loads((root / "docs/research_progress.json").read_text())["source_bound_review"]["audit"]
        self.change(root, audit, lambda a: a.update(reviewed_candidates=new_count, regenerated_strategy_paths=4 * new_count))
        refreshed = build_snapshot(root)
        self.assertEqual(refreshed["summary"]["completed"], self.current["summary"]["completed"] + 1)
        self.assertEqual(refreshed["leader_id"], "synthetic_future_strategy")

    def test_wrong_gate_and_wrong_budget_fail_explicitly(self) -> None:
        root = self.fixture()
        self.change(root, self.current["source_snapshot"],
                    lambda state: state["candidate_reviews"][0]["paths"][0]["gates"].update(sharpe_above_1=True))
        with self.assertRaisesRegex(DashboardError, "gate_metric_mismatch"):
            build_snapshot(root)
        root = self.fixture()
        self.change(root, "docs/research_progress.json",
                    lambda p: p["approved_bounded_optimization_round"].update(registered_configurations_consumed=0))
        with self.assertRaisesRegex(DashboardError, "budget_mismatch"):
            build_snapshot(root)

    def test_private_and_parent_paths_are_rejected_even_if_allowlisted(self) -> None:
        inputs = PublishedInputs(ROOT)
        for path in ("../README.md", "/etc/passwd", "data/accounts.json",
                     "research/workbench/runtime/research-program.sqlite3"):
            inputs.allowed.add(path)
            with self.assertRaises(DashboardError):
                inputs.read(path)

    def test_private_fields_and_credentials_never_enter_projection(self) -> None:
        with self.assertRaisesRegex(DashboardError, "private_projection_field"):
            public_projection({"configuration": {"account_number": "synthetic"}})
        with self.assertRaisesRegex(DashboardError, "credential_projection"):
            public_projection({"description": "ghp_" + "A" * 35})

    def test_formal_date_mismatch_is_not_a_rankable_result(self) -> None:
        root = self.fixture()
        self.change(root, self.current["source_snapshot"],
                    lambda state: state["candidate_reviews"][0]["paths"][0]["metrics"].update(start="2017-01-01"))
        with self.assertRaisesRegex(DashboardError, "formal_window_changed"):
            build_snapshot(root)

    def test_final_round_review_cannot_replace_a_strategy_metric(self) -> None:
        if self.current["round_review"] is None:
            self.skipTest("The completed-round review has not been published yet.")
        for field in ("minimum_four_path_sharpe", "maximum_four_path_drawdown",
                      "ten_year_base_cagr", "ten_year_stress_cagr"):
            with self.subTest(field=field):
                root = self.fixture()
                reference = self.current["round_review"]["evidence"]
                self.change(root, reference,
                            lambda value: value["configurations"][0].update({field: 99}))
                with self.assertRaisesRegex(DashboardError, "round_review_metric_mismatch"):
                    build_snapshot(root)

    def test_changed_source_and_symlink_fail_closed(self) -> None:
        root = self.fixture()
        path = root / "docs/research_progress.json"
        inputs = PublishedInputs(root)
        inputs.read(path.relative_to(root).as_posix())
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(DashboardError, "changed_during_read"):
            inputs.unchanged()
        path.unlink()
        path.symlink_to(ROOT / "docs/research_progress.json")
        with self.assertRaisesRegex(DashboardError, "path_rejected"):
            PublishedInputs(root).read("docs/research_progress.json")

    def test_projection_leaves_research_inputs_unchanged(self) -> None:
        expected = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in self.current["input_sha256"]}
        build_snapshot(ROOT)
        self.assertEqual(expected, {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                    for name in expected})


class HTTPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), partial(DashboardHandler, root=ROOT))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def test_api_and_assets_are_local_read_only_and_uncached(self) -> None:
        with urlopen(self.url + "/api/research") as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])
            progress = json.loads((ROOT / "docs/research_progress.json").read_text())
            self.assertEqual(json.load(response)["summary"]["completed"],
                             progress["recurring_research_program"]["total_evaluated_configurations"])
        for route in ("/", "/app.js", "/styles.css"):
            with urlopen(self.url + route) as response:
                self.assertEqual(response.status, 200)

    def test_private_routes_writes_and_foreign_hosts_are_rejected(self) -> None:
        for path, method, expected in (
            ("/runtime/research-program.sqlite3", "GET", 404),
            ("/evidence/research/workbench/data/close.json", "GET", 503),
            ("/api/research", "POST", 405), ("/api/research", "DELETE", 405),
        ):
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(self.url + path, method=method))
            self.assertEqual(error.exception.code, expected)
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(self.url + "/api/research", headers={"Host": "malicious.example"}))
        self.assertEqual(error.exception.code, 403)


if __name__ == "__main__":
    unittest.main()
