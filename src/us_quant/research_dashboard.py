from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timedelta, timezone
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit
from zoneinfo import ZoneInfo

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


def ssd_benchmarks(source: dict) -> dict:
    benchmarks = {}
    for row in source["paths"]:
        key = (row["years"], row["scenario"], row["symbol"])
        if key in benchmarks:
            raise DashboardError("ssd_duplicate_benchmark")
        benchmarks[key] = metrics(row["metrics"])
    if set(benchmarks) != {(years, scenario, symbol) for years, scenario in CELLS for symbol in ("SPY", "QQQ")}:
        raise DashboardError("ssd_eight_full_capital_benchmarks_required")
    return benchmarks


def ssd_path(protocol: dict, benchmarks: dict, row: dict) -> dict:
    years, scenario = row["years"], row["scenario"]
    if (years, scenario) not in CELLS or row["path_key"] != f"{years}y-{scenario}":
        raise DashboardError("ssd_formal_path_identity_mismatch")
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
    return {"years": years, "scenario": scenario, "metrics": own,
            "benchmark": spy, "qqq_benchmark": qqq, "gates": calculated,
            "execution": execution, "status": "completed"}


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
    benchmarks = ssd_benchmarks(benchmark_source)
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
            paths.append(ssd_path(protocol, benchmarks, row))
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
        "study_id": source["study_id"],
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


def ssd_preparation_failure(inputs: PublishedInputs, item: dict, node: dict, packet: dict) -> dict:
    if (
        packet.get("schema_version") != 3
        or packet.get("record_type") != "registered_preparation_failure"
        or packet.get("program_id") != "ssd_continuous_research_v1"
        or packet.get("study_id") != item["study_id"]
        or any(item[key] is not None for key in ("results", "results_sha256", "preregistration_commit"))
        or any(packet.get(key) is not False for key in (
            "candidate_accounts_or_metrics_computed", "target_proof_completed",
            "investment_objective_verified", "independent_forward_validation", "order_authority"))
    ):
        raise DashboardError("ssd_preparation_failure_boundary_mismatch")
    entries = packet["entries"]
    created = datetime.fromisoformat(packet["created_at"])
    if (
        not entries or len({row["candidate_id"] for row in entries}) != len(entries)
        or created.tzinfo is None or created > datetime.now(timezone.utc)
    ):
        raise DashboardError("ssd_preparation_failure_identity_mismatch")
    candidates, failures = [], []
    for entry in entries:
        specification = entry["specification"]
        frozen = specification["frozen"]
        policy_ref = frozen["policy"]
        policy = inputs.read(WORKBENCH + policy_ref["path"], policy_ref["sha256"])
        registered = datetime.fromisoformat(entry["registered_at"])
        failed = datetime.fromisoformat(entry["failed_at"])
        if (
            specification["candidate"]["id"] != entry["candidate_id"]
            or frozen["study_id"] != item["study_id"]
            or frozen["protocol"] != {
                "path": node["protocol"][len(WORKBENCH):], "sha256": node["protocol_sha256"]}
            or policy["candidates"] != [row["specification"]["candidate"] for row in entries]
            or registered.tzinfo is None or failed.tzinfo is None
            or not registered <= failed <= created
            or not re.fullmatch(r"[0-9a-f]{64}", entry["registered_event_sha256"])
            or not re.fullmatch(r"[0-9a-f]{64}", entry["failure_event_sha256"])
            or entry["status"] != "technical_failure"
            or entry["failure_stage"] != "ordered_path_preparation"
            or not isinstance(entry["failure_reason"], str) or not entry["failure_reason"]
            or entry["target_proof"] is not None or entry["target_proof_at"] is not None
            or entry["candidate_accounts_started"] is not False
            or entry["partial_outputs_retained"] is not True or entry["paths"] != []
        ):
            raise DashboardError("ssd_preparation_failure_record_mismatch")
        candidates.append({
            "id": entry["candidate_id"], "title": entry["candidate_id"].replace("_", " "),
            "scope": "ssd_v1", "study_id": item["study_id"], "status": "technical_failure",
            "full_gate_pass": False, "reviewed_at": None,
            "failure_stage": entry["failure_stage"], "failed_at": entry["failed_at"],
            "paths": [{"years": years, "scenario": scenario, "metrics": None,
                       "execution": None, "gates": {}, "status": "technical_failure"}
                      for years, scenario in CELLS],
        })
        failures.append(entry["failed_at"])
    expected = {
        "new_registered_candidates": len(entries), "new_completed_configurations": 0,
        "new_completed_strategy_evaluations": 0, "new_reviews": 0,
        "new_technical_failures": len(entries), "new_pending_candidates": 0,
        "new_historical_point_gate_passes": 0, "new_account_paths": 0,
    }
    if any(type(packet.get(key)) is not int or packet[key] != value for key, value in expected.items()):
        raise DashboardError("ssd_preparation_failure_count_mismatch")
    return {
        "registered": len(entries), "completed": 0, "reviews": 0,
        "technical_failures": len(entries), "pending": 0, "qualified": 0, "account_paths": 0,
        "study_id": item["study_id"], "candidates": candidates,
        "registration_evidence": item["registration"], "results_evidence": None,
        "completed_at": max(failures, key=datetime.fromisoformat),
        "terminal_preparation_failure_without_target_proof": True,
    }


def ssd_cohort(inputs: PublishedInputs, item: dict, node: dict, protocol: dict, benchmarks: dict) -> dict:
    packet = inputs.read(item["registration"], item["registration_sha256"])
    if packet.get("schema_version") == 3:
        return ssd_preparation_failure(inputs, item, node, packet)
    registered = {row["candidate_id"]: row for row in packet["entries"]}
    if (
        packet["schema_version"] != 2 or packet["program_id"] != "ssd_continuous_research_v1"
        or packet["study_id"] != item["study_id"]
        or not registered or len(registered) != len(packet["entries"])
        or packet["new_registered_candidates"] != len(registered)
        or any(packet.get(key) is not False for key in
               ("order_authority", "investment_objective_verified", "independent_forward_validation",
                "candidate_accounts_or_metrics_computed"))
    ):
        raise DashboardError("ssd_cohort_registration_mismatch")
    created = datetime.fromisoformat(packet["created_at"])
    if created.tzinfo is None or created > datetime.now(timezone.utc):
        raise DashboardError("ssd_cohort_registration_clock_mismatch")
    for identifier, entry in registered.items():
        specification = entry["specification"]
        frozen = specification["frozen"]
        policy_ref = frozen["policy"]
        policy = inputs.read(WORKBENCH + policy_ref["path"], policy_ref["sha256"])
        registered_at, proved_at = (datetime.fromisoformat(entry[key])
                                    for key in ("registered_at", "target_proof_at"))
        if (
            specification["candidate"]["id"] != identifier or frozen["study_id"] != item["study_id"]
            or frozen["protocol"] != {"path": node["protocol"][len(WORKBENCH):], "sha256": node["protocol_sha256"]}
            or policy["candidates"] != [row["specification"]["candidate"] for row in registered.values()]
            or registered_at.tzinfo is None or proved_at.tzinfo is None
            or not registered_at <= proved_at <= created
            or entry["target_proof"]["registered_event_sha256"] != entry["registered_event_sha256"]
            or ("candidate_accounts_or_metrics_computed" in entry["target_proof"]
                and entry["target_proof"]["candidate_accounts_or_metrics_computed"] is not False)
        ):
            raise DashboardError("ssd_cohort_registration_mismatch")
    source = None
    if item["results"] is not None:
        source = inputs.read(item["results"], item["results_sha256"])
        if (
            source["schema_version"] != 2 or source["study_id"] != packet["study_id"]
            or source["registration_packet"] != {
                "path": item["registration"][len(WORKBENCH):], "sha256": item["registration_sha256"]}
            or source["protocol"] != {"path": node["protocol"][len(WORKBENCH):], "sha256": node["protocol_sha256"]}
            or source["actual_publication_before_accounts"]["published_commit"] != item["preregistration_commit"]
            or source["actual_publication_before_accounts"]["registration_packet"] != source["registration_packet"]
            or not re.fullmatch(r"[0-9a-f]{40}", item["preregistration_commit"] or "")
            or any(source.get(key) is not False for key in (
                "order_authority", "investment_objective_verified", "independent_forward_validation",
                "original_records_reclassified", "raw_prices_documents_targets_or_daily_accounts_exported"))
        ):
            raise DashboardError("ssd_cohort_result_boundary_mismatch")
        reviews = {row["candidate_id"]: row for row in source["reviews"]}
        if len(reviews) != len(source["reviews"]) or set(reviews) != set(registered):
            raise DashboardError("ssd_cohort_result_identity_mismatch")
    else:
        if item["results_sha256"] is not None or item["preregistration_commit"] is not None:
            raise DashboardError("ssd_cohort_pending_reference_mismatch")
        reviews = {identifier: {"status": "pending", "paths": []} for identifier in registered}
    candidates = []
    for identifier, review in reviews.items():
        status = review["status"]
        raw_paths = review["paths"]
        keys = {(row["years"], row["scenario"]) for row in raw_paths}
        complete = status in {"rejected", "historical_point_gates_passed"}
        if (
            status not in {"pending", "technical_failure", "rejected", "historical_point_gates_passed"}
            or len(keys) != len(raw_paths) or not keys.issubset(set(CELLS))
            or (complete and (keys != set(CELLS) or review["all_four_paths_same_candidate_and_frozen_parameters"] is not True))
            or (source is not None and any(review.get(key) is not False
                                          for key in ("independent_forward_validation", "order_authority")))
            or (status == "technical_failure" and not re.fullmatch(r"[0-9a-f]{64}", review["failure_event_sha256"] or ""))
        ):
            raise DashboardError("ssd_cohort_path_or_failure_mismatch")
        known = {(row["years"], row["scenario"]): ssd_path(protocol, benchmarks, row) for row in raw_paths}
        qualified = complete and all(all(path["gates"].values()) for path in known.values())
        if complete and status != ("historical_point_gates_passed" if qualified else "rejected"):
            raise DashboardError("ssd_review_verdict_mismatch")
        paths = [known.get((years, scenario), {"years": years, "scenario": scenario,
                 "metrics": None, "execution": None, "gates": {}, "status": status}) for years, scenario in CELLS]
        candidates.append({
            "id": identifier, "title": identifier.replace("_", " "), "scope": "ssd_v1",
            "study_id": item["study_id"],
            "status": ("historically_qualified" if qualified else "rejected_historical") if complete else status,
            "full_gate_pass": qualified, "reviewed_at": review.get("reviewed_at"), "paths": paths,
        })
    counts = {
        "registered": len(registered),
        "completed": sum(row["status"] in {"historically_qualified", "rejected_historical"} for row in candidates),
        "technical_failures": sum(row["status"] == "technical_failure" for row in candidates),
        "pending": sum(row["status"] == "pending" for row in candidates),
        "qualified": sum(row["full_gate_pass"] for row in candidates),
        "account_paths": sum(path["metrics"] is not None for row in candidates for path in row["paths"]),
    }
    counts["reviews"] = counts["completed"]
    if source is not None:
        expected = {
            "new_registered_candidates": counts["registered"], "new_completed_configurations": counts["completed"],
            "new_completed_strategy_evaluations": counts["completed"], "new_reviews": counts["reviews"],
            "new_technical_failures": counts["technical_failures"], "new_pending_candidates": counts["pending"],
            "new_historical_point_gate_passes": counts["qualified"], "new_account_paths": counts["account_paths"],
        }
        if any(type(source.get(key)) is not int or source[key] != value for key, value in expected.items()):
            raise DashboardError("ssd_cohort_actual_count_mismatch")
    return {
        **counts, "study_id": item["study_id"], "candidates": candidates,
        "registration_evidence": item["registration"], "results_evidence": item["results"],
        "completed_at": source["actual_completed_at"] if source is not None else None,
    }


def ssd_research(inputs: PublishedInputs, node: dict, legacy_completed: int) -> dict:
    first = ssd_first_pair(inputs, node)
    first.update(reviews=first["completed"], pending=0, technical_failures=0,
                 account_paths=sum(len(row["paths"]) for row in first["candidates"]))
    cohorts = [first]
    additional = node.get("additional_cohorts", [])
    if additional:
        protocol = inputs.read(node["protocol"], node["protocol_sha256"])
        benchmarks = ssd_benchmarks(inputs.read(node["benchmark_evidence"], node["benchmark_evidence_sha256"]))
        cohorts.extend(ssd_cohort(inputs, item, node, protocol, benchmarks) for item in additional)
    candidates = [row for cohort in cohorts for row in cohort["candidates"]]
    if (len({row["id"] for row in candidates}) != len(candidates)
            or len({cohort["study_id"] for cohort in cohorts}) != len(cohorts)):
        raise DashboardError("ssd_duplicate_cohort_or_candidate")
    totals = {key: sum(cohort[key] for cohort in cohorts) for key in
              ("registered", "completed", "reviews", "pending", "technical_failures", "qualified", "account_paths")}
    totals["total_legacy_plus_ssd_completed"] = legacy_completed + totals["completed"]
    if "additional_cohorts" in node and (
        node.get("aggregate_counts") != totals
        or any(type(value) is not int for value in node["aggregate_counts"].values())
    ):
        raise DashboardError("ssd_aggregate_count_mismatch")
    return {
        **first, **totals, "candidates": candidates,
        "snapshot_schema_version": 2, "program_id": "ssd_continuous_research_v1",
        "study_id": first["study_id"] if len(cohorts) == 1 else None,
        "status": "cohort_evaluations_pending" if totals["pending"] else "cohort_evaluations_finished",
        "completed_at": None if totals["pending"] else max(
            (cohort["completed_at"] for cohort in cohorts),
            key=lambda value: datetime.fromisoformat(value)),
        "single_cohort_reference_fields_scope": "preserved_first_equity_first_pair",
        "cohorts": [{key: value for key, value in cohort.items() if key != "candidates"} for cohort in cohorts],
    }


def ssd_factor_catalog(inputs: PublishedInputs, node: dict, original: dict) -> dict | None:
    if "factor_catalog_extension" not in node:
        return None
    try:
        reference = node["factor_catalog_extension"]
        packet = inputs.read(reference["evidence"], reference["sha256"])
        policy = inputs.read(WORKBENCH + packet["policy"]["path"], packet["policy"]["sha256"])
        proposal_ref = policy["factor_proposal"]
        proposals = inputs.read(WORKBENCH + proposal_ref["path"], proposal_ref["sha256"])
        program_ref = policy["preserved_program_policy"]
        program = inputs.read(WORKBENCH + program_ref["path"], program_ref["sha256"])
        legacy, admission, rows = packet["legacy"], packet["admission"], packet["new_factors"]
        original_families = {row["id"]: row["family"] for row in original["factor_proposals"]}
        expected = {row["id"]: row for row in proposals["factors"]}
        added = {row["factor"]["id"]: row for row in rows}
        checked = datetime.fromisoformat(admission["checked_at"])
        admitted = datetime.fromisoformat(admission["recorded_at"])
        registered = datetime.fromisoformat(packet["registered_at"])
        counts = (
            packet["new_definition_count"], packet["combined_definition_count"],
            packet["combined_family_count"], packet["new_strategy_evaluations"],
            legacy["definition_count"], legacy["family_count"], admission["sessions"],
        )
        false_keys = (
            "historical_W42_plans_reclassified", "independent_alpha_or_forward_proof",
            "original_ledger_written_or_migrated", "strategy_accounts_or_metrics_computed",
            "order_authority",
        )
        if (
            packet["schema_version"] != 1 or policy["schema_version"] != 1
            or type(packet["schema_version"]) is not int
            or type(policy["schema_version"]) is not int
            or packet["catalog_id"] != policy["catalog_id"]
            or policy["data_scope"] != "factor_etf_portfolio"
            or any(packet[key] is not False for key in false_keys)
            or packet["registered_before_new_candidate_selection"] is not True
            or any(policy[key] is not False for key in
                   ("rewrite_legacy_ledger_or_plans", "new_market_requests",
                    "strategy_accounts_or_metrics", "order_authority"))
            or any(policy[key] is not True for key in
                   ("preserve_existing_price_cell_text", "reuse_exact_original_risk_free_reference"))
            or policy["configuration_count_limit_before_deadline"] is not None
            or policy["protocol"] != {
                "path": node["protocol"].removeprefix(WORKBENCH),
                "sha256": node["protocol_sha256"],
            }
            or policy["source_audit"] != packet["source_audit"]
            or any(type(value) is not int for value in counts)
            or packet["new_strategy_evaluations"] != 0
            or len(rows) != len(added) or set(added) != set(expected)
            or packet["new_definition_count"] != len(rows)
            or len(rows) != policy["maximum_new_definitions_this_cycle"]
            or len(rows) != program["maximum_new_factors_per_cycle"]
            or set(added).intersection(original_families)
            or len(legacy["factors"]) != len(original_families)
            or {row["id"]: row["family"] for row in legacy["factors"]} != original_families
            or legacy["definition_count"] != len(original_families)
            or legacy["family_count"] != len(set(original_families.values()))
            or legacy["event_head_sha256"] != original["event_chain_sha256"]
            or legacy["written_or_migrated"] is not False
            or legacy["original_cycle_for_new_week_exists"] is not False
            or packet["combined_definition_count"] != len(original_families) + len(rows)
            or packet["combined_family_count"] != len(
                set(original_families.values()) | {row["factor"]["family"] for row in rows})
            or any(row["factor"] != expected[identifier]
                   or row["registered_at"] != packet["registered_at"]
                   or row["semantic_sha256"] != policy["factor_semantic_sha256"][identifier]
                   for identifier, row in added.items())
            or any(clock.tzinfo is None for clock in (checked, admitted, registered))
            or not checked <= admitted <= registered <= datetime.now(timezone.utc)
            or registered - checked > timedelta(days=8)
            or policy["maximum_readiness_age_days"] != 8
            or admission["readiness_age_limit_days"] != 8
            or packet["week"] != registered.astimezone(
                ZoneInfo(policy["cycle_timezone"])).strftime("%G-W%V")
            or admission["sessions"] != 2805
            or admission["preserved_symbols"] != [
                "SPY", "IEF", "GLD", "BIL", "MTUM", "VLUE", "QUAL", "USMV",
                "TLT", "QQQ", "XLK", "SOXX"]
            or admission["appended_symbols"] != policy["appended_symbols"]
            or policy["appended_symbols"] != ["IJR", "PKW"]
            or admission["original_price_arrays_exactly_preserved"] is not True
            or admission["original_risk_free_reference_and_array_preserved"] is not True
        ):
            raise DashboardError("ssd_catalog_contract_mismatch")
        return {
            "catalog_id": packet["catalog_id"], "registered_at": packet["registered_at"],
            "week": packet["week"], "evidence": reference["evidence"],
            "preserved_definition_count": legacy["definition_count"],
            "new_definition_count": len(rows),
            "combined_definition_count": packet["combined_definition_count"],
            "combined_family_count": packet["combined_family_count"],
            "source_sessions": admission["sessions"],
            "preserved_asset_count": len(admission["preserved_symbols"]),
            "appended_symbols": admission["appended_symbols"],
            "strategy_evaluations_by_admission": 0,
            "new_factors": [
                {"id": identifier, "family": row["factor"]["family"],
                 "title": FACTOR_NAMES.get(row["factor"]["family"], row["factor"]["name"]),
                 "registered_at": row["registered_at"]}
                for identifier, row in added.items()
            ],
        }
    except (KeyError, TypeError, ValueError):
        raise DashboardError("ssd_catalog_contract_mismatch") from None


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
        snapshot["ssd_research"] = ssd_research(
            inputs, progress["ssd_continuous_research_v1"], snapshot["summary"]["completed"])
        catalog = ssd_factor_catalog(inputs, progress["ssd_continuous_research_v1"], state)
        if catalog is not None:
            snapshot["ssd_research"]["factor_catalog_extension"] = catalog
            registrations = {row["id"]: row for row in catalog["new_factors"]}
            for factor in factors:
                if factor["id"] in registrations:
                    factor["ssd_registration"] = {
                        "registered_at": registrations[factor["id"]]["registered_at"],
                        "evidence": catalog["evidence"],
                    }
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
