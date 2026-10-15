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
