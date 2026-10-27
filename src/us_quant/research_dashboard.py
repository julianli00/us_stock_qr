from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "web/research"
WORKBENCH = "research/workbench/"
LEGACY_SOURCES = (
    "all_52_summary.json",
    "factor_round_20261007_results.json",
    "allocation_round_20261007_results.json",
    "implementation_round_20261007_results.json",
    "portfolio_refinement_20261007_results.json",
    "multifactor_stability_20261010_results.json",
)
CELLS = ((10, "base"), (10, "stress"), (5, "base"), (5, "stress"))
METRICS = ("cagr", "sharpe", "max_drawdown", "start", "end", "sessions")
PRIVATE_KEYS = {"account_number", "account_id", "holdings", "positions", "shares",
                "quantity", "api_key", "access_token", "password", "webhook_url"}
SECRET = re.compile(r"hooks\.slack(?:-gov)?\.com/services/|discord(?:app)?\.com/api/webhooks/"
                    r"|gh[pousr]_[A-Za-z0-9]{30,}|xox[baprs]-[A-Za-z0-9-]{25,}")
FACTOR_NAMES = {
    "price_momentum": "价格动量",
    "value_exposure": "价值",
    "quality_exposure": "质量",
    "low_volatility_exposure": "低波动",
    "cashflow_accrual_quality_v1": "现金流应计质量",
    "conservative_asset_growth_v1": "保守资产增长",
    "size": "小盘规模",
    "share_issuance": "净股份回购",
}
STATUS_NAMES = {
    "rejected_historical": "历史未达标",
    "historically_qualified": "历史候选",
    "pending": "已注册，待评估",
    "technical_failure": "技术失败",
    "legacy": "旧协议记录",
}


class DashboardError(RuntimeError):
    """Stable, public error codes; never expose file contents or private paths."""


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def decode(raw: bytes) -> dict:
    try:
        value = json.loads(raw)
    except (UnicodeError, ValueError):
        raise DashboardError("invalid_published_json") from None
    if not isinstance(value, dict):
        raise DashboardError("published_object_required")
    return value


def text(value: Any, limit: int = 12000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise DashboardError("invalid_public_text")
    return value


def number(value: Any, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise DashboardError("invalid_published_metric")
    return float(value)


def public_projection(value: Any) -> None:
    if isinstance(value, dict):
        if PRIVATE_KEYS.intersection(str(key).lower() for key in value):
            raise DashboardError("private_projection_field_rejected")
        for child in value.values():
            public_projection(child)
    elif isinstance(value, list):
        for child in value:
            public_projection(child)
    elif isinstance(value, str) and SECRET.search(value):
        raise DashboardError("credential_projection_rejected")


class PublishedInputs:
    def __init__(self, root: Path):
        self.root = root.resolve()
        contract = decode((self.root / "config/publication_allowlist.json").read_bytes())
        self.allowed = set(contract["files"])
        manifest = self.root / contract["bundle_manifest"]
        if manifest.is_symlink() or not manifest.resolve().is_relative_to(self.root):
            raise DashboardError("publication_manifest_path_rejected")
        raw = manifest.read_bytes()
        if digest(raw) != contract["bundle_manifest_sha256"]:
            raise DashboardError("publication_manifest_hash_mismatch")
        self.allowed.update(
            (manifest.parent / name).relative_to(self.root).as_posix()
            for name in decode(raw)["file_sha256"]
        )
        self.hashes: dict[str, str] = {}

    def read(self, name: str, expected: str | None = None) -> dict:
        if (
            not isinstance(name, str) or name not in self.allowed
            or not name.endswith(".json") or Path(name).is_absolute()
            or ".." in Path(name).parts
            or not name.startswith(("config/", "docs/", WORKBENCH + "config/", WORKBENCH + "evidence/"))
        ):
            raise DashboardError("unpublished_input_rejected")
        path = self.root / name
        if (
            any(part.is_symlink() for part in (path, *path.parents) if part != self.root.parent)
            or not path.resolve().is_relative_to(self.root)
        ):
            raise DashboardError("published_input_path_rejected")
        raw = path.read_bytes()
        actual = digest(raw)
        if expected is not None and actual != expected:
            raise DashboardError("frozen_policy_hash_mismatch")
        if name in self.hashes and self.hashes[name] != actual:
            raise DashboardError("published_inputs_changed_during_read")
        self.hashes[name] = actual
        return decode(raw)

    def unchanged(self) -> None:
        for name, expected in self.hashes.items():
            if digest((self.root / name).read_bytes()) != expected:
                raise DashboardError("published_inputs_changed_during_read")


def metrics(source: dict) -> dict:
    result = {}
    for name in METRICS:
        if name in ("start", "end"):
            result[name] = text(source[name], 32)
        else:
            result[name] = number(source[name], nullable=name == "sharpe")
    if result["sessions"] <= 0 or not 0 <= result["max_drawdown"] <= 1:
        raise DashboardError("invalid_published_metric_period")
    return result


def gates(own: dict, benchmark: dict, base_benchmark: dict, scenario: str) -> dict:
    result = {
        "sharpe_above_1": own["sharpe"] is not None and own["sharpe"] > 1,
        "drawdown_at_most_15pct": own["max_drawdown"] <= 0.15,
        "beats_spy": own["cagr"] > benchmark["cagr"],
    }
    if scenario == "stress":
        result["also_beats_base_cost_spy"] = own["cagr"] > base_benchmark["cagr"]
    return result


def policy_for(inputs: PublishedInputs, spec: dict) -> tuple[dict | None, str | None]:
    matches = []
    for name, expected in spec["frozen_files"].items():
        if name.startswith("config/") and name.endswith(".json"):
            relative = WORKBENCH + name
            if relative not in inputs.allowed:
                raise DashboardError("frozen_policy_not_published")
            policy = inputs.read(relative, expected)
            if any(row.get("id") == spec["id"] for row in policy.get("candidates", []) if isinstance(row, dict)):
                matches.append((policy, relative))
    if len(matches) > 1:
        raise DashboardError("ambiguous_candidate_policy")
    return matches[0] if matches else (None, None)


def recurring_strategy(inputs: PublishedInputs, spec: dict, review: dict | None, failure: dict | None) -> dict:
    policy, reference = (None, None) if failure is not None else policy_for(inputs, spec)
    method = (policy or {}).get("methodology", {})
    if not isinstance(method, dict):
        raise DashboardError("invalid_published_methodology")
    description = {
        name: text(method[name])
        for name in ("hypothesis", "rule", "allocation", "budget", "target_rule", "execution", "limitations", "risks")
        if isinstance(method.get(name), str) and method[name].strip()
    }
    leverage = spec["asset_leverage"]
    if not isinstance(leverage, dict) or not leverage:
        raise DashboardError("asset_identity_missing")
    result = {
        "id": text(spec["id"], 120),
        "title": (policy or {}).get("study_id", spec["id"]).replace("_", " "),
        "scope": "recurring",
        "status": "pending",
        "reviewed_at": None,
        "factor_ids": list(spec["factor_ids"]),
        "assets": sorted(leverage),
        "unleveraged": all(number(value) == 1 for value in leverage.values()),
        "configuration": spec["configuration"],
        "description": description,
        "policy_evidence": reference,
        "history_status": spec["history_status"],
        "paths": [],
        "eligible_for_current_ranking": False,
        "full_gate_pass": False,
        "min_sharpe": None,
        "max_drawdown": None,
        "forward_validated": False,
    }
    if failure is not None:
        result.update(status="technical_failure", reviewed_at=failure["at"],
                      failure_reason=text(failure.get("reason", failure.get("error", "已登记的技术失败；未取得完整四路径结果。"))))
        result["description"]["limitations"] = "此失败属于保留的旧代码版本。当前修订后的配置不是当时的规则，不读取当前配置来改写旧版本，也不补造四路径结果。"
    if review is None:
        return result
    paths = review["paths"]
    if len(paths) != 4 or {(row["years"], row["scenario"]) for row in paths} != set(CELLS):
        raise DashboardError("four_formal_paths_required")
    result.update(status=review["status"], reviewed_at=text(review["at"], 80))
    for years, scenario in CELLS:
        path = next(row for row in paths if (row["years"], row["scenario"]) == (years, scenario))
        base = next(row for row in paths if (row["years"], row["scenario"]) == (years, "base"))
        own, spy, base_spy = metrics(path["metrics"]), metrics(path["benchmark"]), metrics(base["benchmark"])
        if (
            own["sessions"] != (2512 if years == 10 else 1254)
            or own["start"] != ("2016-10-06" if years == 10 else "2021-10-06")
            or own["end"] != "2026-10-05"
            or any(own[name] != spy[name] for name in ("start", "end", "sessions"))
        ):
            raise DashboardError("formal_window_changed")
        calculated = gates(own, spy, base_spy, scenario)
        if calculated != path["gates"]:
            raise DashboardError("published_gate_metric_mismatch")
        result["paths"].append({"years": years, "scenario": scenario, "metrics": own,
                                "benchmark": spy, "gates": calculated, "status": "completed"})
    qualified = all(all(path["gates"].values()) for path in result["paths"])
    if qualified is not review["historical_gates_passed"]:
        raise DashboardError("published_review_verdict_mismatch")
    result.update(
        full_gate_pass=qualified,
        eligible_for_current_ranking=result["unleveraged"],
        min_sharpe=min(path["metrics"]["sharpe"] for path in result["paths"])
        if all(path["metrics"]["sharpe"] is not None for path in result["paths"]) else None,
        max_drawdown=max(path["metrics"]["max_drawdown"] for path in result["paths"]),
        accounting="funded-v2" if "accounting_engine" in review else "legacy-original",
    )
    return result


def legacy_strategies(inputs: PublishedInputs) -> list[dict]:
    result = []
    for filename in LEGACY_SOURCES:
        relative = WORKBENCH + "evidence/" + filename
        source = inputs.read(relative)
        entries = source.get("candidates", source.get("configurations"))
        if isinstance(entries, dict):
            entries = list(entries.values())
        if not isinstance(entries, list) or not entries:
            raise DashboardError("legacy_inventory_missing")
        for entry in entries:
            definition = entry.get("definition", {})
            identifier = entry.get("candidate", definition.get("id"))
            if filename == "portfolio_refinement_20261007_results.json" and identifier == "unchanged_baseline":
                continue
            paths = []
            for years, scenario in CELLS:
                window = entry["windows"][f"{years}y"]
                saved = window.get(scenario)
                own = saved.get("metrics", saved) if isinstance(saved, dict) else {}
                complete = all(name in own for name in METRICS)
                paths.append({
                    "years": years, "scenario": scenario,
                    "status": "completed" if complete else "incomplete",
                    "metrics": metrics(own) if complete else None,
                    "gates": window.get(scenario + "_gates", (saved or {}).get("gates", {})),
                })
            result.append({
                "id": f"legacy:{filename}:{text(identifier, 120)}",
                "original_id": identifier, "title": text(identifier, 120).replace("_", " "),
                "scope": "legacy", "status": "legacy",
                "reviewed_at": source.get("created_at"),
                "factor_ids": [], "assets": [], "configuration": definition,
                "description": {"limitations": "旧研究的验收协议、资产杠杆或账务版本不同；保留原数值，不加入当前无杠杆候选排名。"},
                "policy_evidence": relative, "history_status": "exposed_history",
                "original_assessment": {
                    key: value for key, value in entry.items()
                    if key.endswith("_pass") or key.endswith("_passes") or key == "promotion_assessment"
                },
                "paths": paths, "eligible_for_current_ranking": False, "full_gate_pass": False,
                "min_sharpe": None, "max_drawdown": None, "forward_validated": False,
                "unleveraged": None,
            })
    if len(result) != 84 or len({row["id"] for row in result}) != 84:
        raise DashboardError("legacy_inventory_count_mismatch")
    return result


def ssd_first_pair(inputs: PublishedInputs, node: dict) -> dict:
    protocol = inputs.read(node["protocol"], node["protocol_sha256"])
    benchmark_source = inputs.read(node["benchmark_evidence"], node["benchmark_evidence_sha256"])
    packet = inputs.read(node["first_pair_registration"], node["first_pair_registration_sha256"])
    source = inputs.read(node["first_pair_results"], node["first_pair_results_sha256"])
    if (
        protocol["protocol_id"] != "ssd_dual_benchmark_net_cagr_v1"
        or protocol["hard_gates_per_path"] != {
            "net_excess_sharpe_strictly_greater_than": 1.0,
            "net_cagr_strictly_greater_than": 0.12,
            "maximum_drawdown_at_most": 0.15,
            "net_cagr_strictly_greater_than_same_cost_spy": True,
            "net_cagr_strictly_greater_than_same_cost_qqq": True,
            "stress_also_beats_base_cost_spy": True,
            "all_four_paths_same_candidate_and_parameters": True,
        }
        or node["deadline_utc"] != protocol["authorization_deadline_utc"]
        or node["configuration_count_cap_before_deadline"] is not None
        or node["ordinary_per_batch_approval_required"] is not False
        or any(value.get(key) is not False for value in (node, source)
               for key in ("order_authority", "investment_objective_verified", "independent_forward_validation"))
        or source["original_records_reclassified"] is not False
        or source["raw_prices_documents_targets_or_daily_accounts_exported"] is not False
        or source["actual_publication_before_accounts"]["published_commit"] != node["first_pair_preregistration_commit"]
        or source["registration_packet"]["sha256"] != node["first_pair_registration_sha256"]
        or source["protocol"]["sha256"] != node["protocol_sha256"]
    ):
        raise DashboardError("ssd_protocol_or_authority_mismatch")
    benchmarks = {}
    for row in benchmark_source["paths"]:
        key = (row["years"], row["scenario"], row["symbol"])
        if key in benchmarks:
            raise DashboardError("ssd_duplicate_benchmark")
        benchmarks[key] = metrics(row["metrics"])
    if set(benchmarks) != {(years, scenario, symbol) for years, scenario in CELLS for symbol in ("SPY", "QQQ")}:
        raise DashboardError("ssd_eight_full_capital_benchmarks_required")
    registered = {row["candidate_id"]: row for row in packet["entries"]}
    reviews = {row["candidate_id"]: row for row in source["reviews"]}
    if (
        len(registered) != len(packet["entries"]) or len(reviews) != len(source["reviews"])
        or set(registered) != set(reviews) or source["study_id"] != packet["study_id"]
        or source["new_technical_failures"] != 0
    ):
        raise DashboardError("ssd_complete_pair_identity_mismatch")
    candidates = []
    for identifier, review in reviews.items():
        if (
            len(review["paths"]) != 4
            or {(row["years"], row["scenario"]) for row in review["paths"]} != set(CELLS)
            or review["all_four_paths_same_candidate_and_frozen_parameters"] is not True
            or review["independent_forward_validation"] is not False
            or review["order_authority"] is not False
        ):
            raise DashboardError("ssd_four_formal_paths_required")
        paths = []
        for years, scenario in CELLS:
            row = next(item for item in review["paths"] if (item["years"], item["scenario"]) == (years, scenario))
            own = metrics(row["metrics"])
            spy, qqq, base_spy = (benchmarks[(years, scenario, "SPY")],
                                  benchmarks[(years, scenario, "QQQ")], benchmarks[(years, "base", "SPY")])
            if (
                own["sessions"] != (2512 if years == 10 else 1254)
                or own["start"] != ("2016-10-06" if years == 10 else "2021-10-06")
                or own["end"] != "2026-10-05"
                or any(own[key] != reference[key] for reference in (spy, qqq, base_spy)
                       for key in ("start", "end", "sessions"))
                or row["benchmark_net_cagr"] != {"SPY": spy["cagr"], "QQQ": qqq["cagr"]}
            ):
                raise DashboardError("ssd_formal_window_or_benchmark_mismatch")
            calculated = {
                **gates(own, spy, base_spy, scenario),
                "cagr_above_12pct": own["cagr"] > 0.12,
                "beats_qqq": own["cagr"] > qqq["cagr"],
            }
            if calculated != row["gates"] or row["all_point_gates_pass"] is not all(calculated.values()):
                raise DashboardError("ssd_gate_metric_mismatch")
            execution = {key: number(row["execution"][key])
                         for key in protocol["execution_disclosure"]["required_fields"]}
            if (
                any(value < 0 for value in execution.values())
                or any(execution[key] != int(execution[key]) for key in ("model_order_count", "rebalance_session_count"))
                or row["execution"]["model_orders_are_broker_orders"] is not False
                or number(row["execution"]["minimum_cash_usd"]) < -1e-8
                or not 0 <= number(row["execution"]["maximum_gross_exposure"]) <= 1 + 1e-10
                or abs(execution["total_cost_usd"] - execution["fixed_commission_usd"]
                       - execution["proportional_cost_usd"]) > 1e-6
                or execution["fixed_commission_usd"] != execution["model_order_count"]
            ):
                raise DashboardError("ssd_execution_disclosure_mismatch")
            paths.append({"years": years, "scenario": scenario, "metrics": own,
                          "benchmark": spy, "qqq_benchmark": qqq, "gates": calculated,
                          "execution": execution, "status": "completed"})
        qualified = all(all(path["gates"].values()) for path in paths)
        if review["status"] != ("historical_point_gates_passed" if qualified else "rejected"):
            raise DashboardError("ssd_review_verdict_mismatch")
        candidates.append({
            "id": identifier, "title": identifier.replace("_", " "), "scope": "ssd_v1",
            "status": "historically_qualified" if qualified else "rejected_historical",
            "full_gate_pass": qualified, "reviewed_at": text(review["reviewed_at"], 80),
            "paths": paths,
        })
    counts = {
        "new_registered_candidates": len(registered),
        "new_reviews": len(reviews),
        "new_technical_failures": 0,
        "new_historical_point_gate_passes": sum(row["full_gate_pass"] for row in candidates),
    }
    if (
        any(type(value.get(key)) is not int or value[key] != count
            for value in (node, source) for key, count in counts.items())
        or node["new_completed_strategy_configurations"] != len(candidates)
        or source["new_completed_configurations"] != len(candidates)
        or packet["new_registered_candidates"] != len(registered)
        or node["new_account_paths"] != 4 * len(candidates)
        or source["new_account_paths"] != 4 * len(candidates)
    ):
        raise DashboardError("ssd_actual_count_mismatch")
    deadline = datetime.fromisoformat(node["deadline_utc"].replace("Z", "+00:00"))
    if deadline.tzinfo is None:
        raise DashboardError("ssd_deadline_timezone_required")
    return {
        "status": node["status"], "deadline_beijing": node["deadline_beijing"],
        "authorization_window_open": datetime.now(timezone.utc) < deadline,
        "configuration_count_cap_before_deadline": None, "registered": len(registered),
        "completed": len(candidates), "qualified": counts["new_historical_point_gate_passes"],
        "candidates": candidates, "results_evidence": node["first_pair_results"],
        "registration_evidence": node["first_pair_registration"],
        "benchmark_evidence": node["benchmark_evidence"],
        "completed_at": source["actual_completed_at"],
        "independent_forward_validation": False, "order_authority": False,
    }


def build_snapshot(root: Path = ROOT) -> dict:
    inputs = PublishedInputs(root)
    progress = inputs.read("docs/research_progress.json")
    current = progress["recurring_research_program"]
    state_reference = current["cycle_snapshot"]
    state = inputs.read(state_reference)
    policy = inputs.read(WORKBENCH + "config/research-program.json")
    specs, reviews = state["registered_candidates"], state["candidate_reviews"]
    failures = {row["candidate_id"]: row for row in state["technical_failure_records"]}
    by_review = {row["candidate_id"]: row for row in reviews}
    if (
        len(by_review) != len(reviews) or len({row["id"] for row in specs}) != len(specs)
        or state["total_evaluated_configurations"] != 84 + len(reviews)
        or len(reviews) != current["new_strategy_evaluations"]
        or any(state.get(key) is not False for key in ("order_authority", "automatic_live_deployment", "independent_forward_validation"))
    ):
        raise DashboardError("published_state_count_or_authority_mismatch")
    strategies = [
        recurring_strategy(inputs, spec, by_review.get(spec["id"]), failures.get(spec["id"]))
        for spec in specs
    ]
    legacy = legacy_strategies(inputs)
    candidates = [
        row for row in strategies
        if row["eligible_for_current_ranking"] and row["max_drawdown"] <= 0.15
        and row["min_sharpe"] is not None
    ]
    candidates.sort(key=lambda row: (-row["min_sharpe"], row["id"]))
    leader = candidates[0] if candidates else None
    qualified = [row["id"] for row in strategies if row["full_gate_pass"]]
    if len(qualified) != state["historical_qualified_candidate_count"]:
        raise DashboardError("qualification_count_mismatch")
    round_state = progress["approved_bounded_optimization_round"]
    used = sum(
        spec["configuration"].get("optimization_round_id") == "user_confirmed_etf_restructure_20261013"
        for spec in specs
    )
    if (
        used != round_state["registered_configurations_consumed"]
        or round_state["maximum_registered_configurations"] != 6
        or round_state["registered_configuration_slots_remaining"] != 6 - used
        or used > 6
    ):
        raise DashboardError("bounded_budget_mismatch")
    round_review = None
    if round_state.get("review"):
        saved_review = inputs.read(round_state["review"])
        rows = saved_review["configurations"]
        tagged = {spec["id"] for spec in specs
                  if spec["configuration"].get("optimization_round_id") == "user_confirmed_etf_restructure_20261013"}
        if (
            used != 6 or saved_review["registered_slots_remaining"] != 0
            or {row["candidate_id"] for row in rows} != tagged or len(rows) != 6
            or saved_review["new_strategy_evaluations_by_this_review"] != 0
            or saved_review["historically_qualified_configurations"] != sum(row["historical_gates_passed"] for row in rows)
        ):
            raise DashboardError("completed_round_review_mismatch")
        for row in rows:
            actual = next(strategy for strategy in strategies if strategy["id"] == row["candidate_id"])
            base = next(path for path in actual["paths"] if (path["years"], path["scenario"]) == (10, "base"))
            stress = next(path for path in actual["paths"] if (path["years"], path["scenario"]) == (10, "stress"))
            if (
                abs(row["minimum_four_path_sharpe"] - actual["min_sharpe"]) > 1e-10
                or abs(row["maximum_four_path_drawdown"] - actual["max_drawdown"]) > 1e-10
                or abs(row["ten_year_base_cagr"] - base["metrics"]["cagr"]) > 1e-10
                or abs(row["ten_year_stress_cagr"] - stress["metrics"]["cagr"]) > 1e-10
                or row["historical_gates_passed"] is not actual["full_gate_pass"]
                or row["status"] != actual["status"]
            ):
                raise DashboardError("round_review_metric_mismatch")
        round_review = {
            "evidence": round_state["review"],
            "qualified": saved_review["historically_qualified_configurations"],
            "rows": [{key: row[key] for key in
                      ("candidate_id", "ten_year_base_cagr", "ten_year_stress_cagr",
                       "minimum_four_path_sharpe", "maximum_four_path_drawdown", "status")}
                     for row in rows],
            "additional_trials_allowed": False,
        }
    factors = []
    for item in state["factor_proposals"]:
        factors.append({
            key: item[key]
            for key in ("id", "family", "name", "definition", "rationale", "limitations", "direction", "inputs", "sources")
        })
        factors[-1].update(title=FACTOR_NAMES.get(item["id"], item["name"]),
                          status="registered", strategy_count=sum(item["id"] in row["factor_ids"] for row in strategies))
    if len(factors) != state["known_factor_definition_count"]:
        raise DashboardError("factor_catalog_count_mismatch")
    queued_policy = inputs.read(WORKBENCH + "config/factor-family-expansion.json")
    for item in queued_policy["factors"]:
        factors.append({
            key: item[key]
            for key in ("id", "family", "name", "definition", "rationale", "limitations", "direction", "inputs", "sources")
        })
        factors[-1].update(title=FACTOR_NAMES.get(item["family"], item["name"]),
                          status="queued", strategy_count=0)
    daily_reference = progress["daily_operational_execution"]["actual_run"]
    daily = inputs.read(daily_reference)
    forward = []
    for node, count_key in (
        ("prospective_research_accounts", "actual_observed_model_sessions"),
        ("prospective_dollar_guard", "forward_model_sessions"),
    ):
        saved = progress[node]
        count = saved[count_key]
        if count < 63 and (saved["reported_forward_sharpe"] is not None or saved["reported_forward_cagr"] is not None):
            raise DashboardError("premature_forward_metrics")
        forward.append({
            "id": saved["source_candidate_id"], "sessions": count,
            "latest_session": saved["latest_model_session"],
            "minimum_sessions": 63, "sharpe": saved["reported_forward_sharpe"],
            "cagr": saved["reported_forward_cagr"], "historical_status": "rejected_historical",
        })
    history = []
    for row in strategies + legacy:
        if row["reviewed_at"]:
            history.append({
                "id": row["id"], "at": row["reviewed_at"],
                "kind": "strategy", "status": row["status"], "title": row["title"],
            })
    for name in sorted(inputs.allowed):
        if name.startswith(WORKBENCH + "evidence/research_daily_") and (
            "/research_daily_actual_run_" in name or "/research_daily_recovery_" in name
        ) and "_preservation_" not in name:
            record = inputs.read(name)
            if "execution_record" in record:
                execution = record["execution_record"]
                history.append({
                    "id": name, "kind": "operation", "evidence": name,
                    "at": execution["started_at"], "status": execution["status"],
                    "title": "原流程连续性检查：已有输入，未新增收益日",
                    "session": execution["steps"][0]["result"]["latest_session"],
                })
                continue
            history.append({
                "id": name, "kind": "operation", "evidence": name,
                "at": record.get("actual_pipeline_started_at_utc", record.get("actual_original_pipeline_started_at_utc")),
                "status": record["status"], "title": record.get("reason", "原日流程及独立美元流程真实恢复"),
                "session": record["market_session"],
            })
    if any(not row["at"] for row in history):
        raise DashboardError("historical_event_timestamp_missing")
    history.sort(key=lambda row: (row["at"], row["id"]), reverse=True)
    audit = inputs.read(current["cycle_snapshot"].rsplit("/", 1)[0] + "/" +
                        Path(progress["source_bound_review"]["audit"]).name)
    if audit["reviewed_candidates"] != len(reviews) or audit["regenerated_strategy_paths"] != 4 * len(reviews):
        raise DashboardError("current_source_audit_count_mismatch")
    snapshot = {
        "schema_version": 1, "read_at": datetime.now(timezone.utc).isoformat(),
        "updated_on": progress["updated_on"], "history_end": "2026-10-05",
        "source_snapshot": state_reference, "source_event_head": state["event_chain_sha256"],
        "summary": {
            "completed": state["total_evaluated_configurations"], "legacy": len(legacy),
            "reviews": len(reviews), "registered": len(specs), "technical_failures": len(failures),
            "definitions": state["known_factor_definition_count"], "economic_families": state["economic_factor_family_count"],
            "qualified": len(qualified), "pending": len(state["pending_candidate_ids"]),
            "source_audit_paths": audit["regenerated_strategy_paths"],
        },
        "goals": policy["goals"], "status_names": STATUS_NAMES, "factors": factors,
        "strategies": strategies + legacy, "history": history, "forward": forward,
        "leader_id": leader["id"] if leader else None, "qualified_ids": qualified,
        "ranking": "仅当前四路径完整、全部资产1倍、最大回撤均不超过15%的研究配置；按四路径最低净超额夏普降序。旧84配置不混入排名。历史候选不是独立前瞻证明。",
        "budget": {"cap": 6, "used": used, "remaining": 6 - used, "completed_batches": round_state["completed_batches"]},
        "round_review": round_review,
        "queued": {"definitions": current["queued_new_factor_definitions"],
                   "strategies": current["prepared_unregistered_strategy_candidates"],
                   "not_before": current["queued_factor_eligibility"]},
        "fundamental_data_blockers": state["latest_cycle"]["data_blockers"],
        "daily": {
            "status": daily["status"], "session": daily["market_session"],
            "reason": daily.get("reason"), "evidence": daily_reference,
            "latest_accepted_session": progress["prospective_data_archive"]["latest_accepted_session"],
            "new_sessions": progress["daily_operational_execution"]["new_market_or_model_observations"],
        },
        "warnings": [
            "全部为历史研究及模拟记录，不连接券商，不提交订单。",
            "ETF数量、策略配置数量和经济因子家族数量不是独立alpha数量。",
            "15%回撤是验收门槛，不是硬止损或未来回撤保证。",
            "前瞻未满63个实际收益日不报告夏普；63日也不是长期有效性证明。",
            "旧84配置保留当时结果和协议；不能和当前四路径门槛拼接，失败不删除。",
        ],
        "order_authority": False,
    }
    if "ssd_continuous_research_v1" in progress:
        snapshot["ssd_research"] = ssd_first_pair(inputs, progress["ssd_continuous_research_v1"])
        snapshot["summary_scope"] = "preserved_pre_ssd_research"
        snapshot["ranking"] = "原 SPY 四路径口径的低回撤研究对照，不按新 SSD 双基准规则重新认定；旧记录、失败和原有排序保持。"
        snapshot["warnings"].append("历史六配置轮次已用完不限制新 SSD 授权；新口径与原记录分开展示，不混用通过指标。")
    inputs.unchanged()
    snapshot["input_sha256"] = dict(sorted(inputs.hashes.items()))
    public_projection(snapshot)
    try:
        json.dumps(snapshot, allow_nan=False)
    except ValueError:
        raise DashboardError("nonfinite_dashboard_output") from None
    return snapshot


class DashboardHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, root: Path, **kwargs):
        self.root = root
        super().__init__(*args, **kwargs)

    def response(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; script-src 'self'; style-src 'self'; "
                         "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
                         "base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self) -> None:
        host = self.headers.get("Host", "")
        if host not in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}:
            self.response(403, b'{"error":"local_host_required"}', "application/json")
            return
        path = unquote(urlsplit(self.path).path)
        try:
            if path == "/api/research":
                body = json.dumps(build_snapshot(self.root), ensure_ascii=False, allow_nan=False).encode()
                self.response(200, body, "application/json; charset=utf-8")
            elif path in ("/", "/index.html", "/app.js", "/styles.css"):
                name = "index.html" if path == "/" else path[1:]
                types = {"html": "text/html", "js": "text/javascript", "css": "text/css"}
                asset = self.root / "web/research" / name
                if asset.is_symlink():
                    raise DashboardError("asset_path_rejected")
                self.response(200, asset.read_bytes(), types[asset.suffix[1:]] + "; charset=utf-8")
            elif path.startswith("/evidence/"):
                source = path[len("/evidence/"):]
                body = PublishedInputs(self.root).read(source)
                public_projection(body)
                self.response(200, json.dumps(body, ensure_ascii=False, allow_nan=False).encode(), "application/json; charset=utf-8")
            else:
                self.response(404, b'{"error":"not_found"}', "application/json")
        except DashboardError as exc:
            self.response(503, json.dumps({"error": str(exc)}).encode(), "application/json")
        except (OSError, KeyError, TypeError, ValueError):
            self.response(503, b'{"error":"published_evidence_unavailable_or_inconsistent"}', "application/json")

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_POST(self) -> None:
        self.response(405, b'{"error":"read_only_dashboard"}', "application/json")

    do_PUT = do_POST
    do_PATCH = do_POST
    do_DELETE = do_POST

    def log_message(self, format: str, *args) -> None:
        print(f"Research dashboard: {args[1] if len(args) > 1 else 'request'}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local, read-only published research ledger. No data collection or orders.")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--check", action="store_true", help="Validate projection without starting a server.")
    args = parser.parse_args()
    try:
        snapshot = build_snapshot()
        if args.check:
            print(json.dumps({"dashboard": "verified", **snapshot["summary"], "leader": snapshot["leader_id"]}))
            return
        if not 0 <= args.port <= 65535:
            raise DashboardError("invalid_local_port")
        with ThreadingHTTPServer(("127.0.0.1", args.port), partial(DashboardHandler, root=ROOT)) as server:
            print(f"Research dashboard ready: http://127.0.0.1:{server.server_port}", flush=True)
            server.serve_forever()
    except DashboardError as exc:
        parser.exit(2, f"Research dashboard blocked: {exc}.\n")
    except (OSError, KeyError, TypeError, ValueError):
        parser.exit(2, "Research dashboard blocked: published evidence unavailable or inconsistent; run --check after repairing the published metadata.\n")


if __name__ == "__main__":
    main()
