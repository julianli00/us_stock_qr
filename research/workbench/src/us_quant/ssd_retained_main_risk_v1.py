from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from importlib.metadata import distributions
from pathlib import Path
from types import FunctionType
from xml.etree import ElementTree

import pandas as pd

from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import retained_main_risk_v1 as rules
from us_quant import ssd_expert_adanormalhedge_v1 as saved
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.storage import new_output_directory, utc_now, write_json

ROOT = rules.ROOT
FOLDER = ROOT / "reports/ssd-continuous-research-v1/retained-main-risk-20261108-v1"
CATALOGUE = FOLDER / "source-inputs-v1.json"
INTENT = FOLDER / "analysis-intent-v1.json"
OUTPUT = FOLDER / "diagnostic-v1.json"
PUBLIC = ROOT / "evidence/retained_main_risk_20261108_v1.json"
OWN = (
    "config/retained-main-risk-v1.json",
    "src/us_quant/retained_main_risk_v1.py",
    "src/us_quant/ssd_retained_main_risk_v1.py",
    "tests/test_retained_main_risk_v1.py",
    "RETAINED_MAIN_RISK_V1.md",
    "tests/conftest.py",
)
content_path, file_digest, read_json = saved.content_path, saved.file_digest, saved.read_json
reference, checked_reference = saved.reference, saved.checked_reference


def _guard() -> None:
    for path in (lifecycle.AUTHORIZATION, lifecycle.original.POLICY):
        content_path(path)
    lifecycle.runtime_guard()


def _policy() -> dict:
    result = read_json(rules.POLICY)
    rules.validate(result)
    return result


def _progress(phase: str, **extra) -> None:
    actual = saved.counts()
    rules.require(
        all(key not in extra or extra[key] == value for key, value in actual.items()),
        "A diagnostic cannot overwrite authoritative financial counters.",
    )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "retained_main_risk_" + phase,
            "study_id": rules.STUDY,
            "active_compute_job": None,
            **actual,
            **extra,
            "new_financial_trials": 0,
            "investment_objective_verified": False,
            "order_authority": False,
        },
    )


def dependencies():
    function = FunctionType(
        saved.dependencies.__code__,
        {**saved.dependencies.__globals__, **globals()},
        argdefs=saved.dependencies.__defaults__,
    )
    function.__kwdefaults__ = saved.dependencies.__kwdefaults__
    return function()


def _packages():
    rows = sorted([[d.metadata["Name"], d.version] for d in distributions()])
    rules.require(
        len(rows) == 58 and len({name.lower().replace("_", "-") for name, _ in rows}) == 57,
        "Keep the accepted exact package inventory; do not restore or loosen dependency pins.",
    )
    return rows


def _permitted_refs(baseline: dict):
    references = baseline["protected_reference_metadata"]
    live = [ref for ref in references if not saved.protected_reference(ref)]
    rules.require(
        len(references) == 16 and len(live) == 15,
        "Keep sixteen archival references distinct from fifteen permitted live checks.",
    )
    for ref in live:
        path = content_path(ROOT.parents[1] / ref["path"])
        rules.require(
            file_digest(path) == ref["sha256"] and oct(path.stat().st_mode & 0o777) == ref["mode"],
            "An allowed preserved input changed; do not restore it.",
        )
    user_paths = (
        "GLOSSARY.md",
        "research/workbench/config/real-estate-source-intake-v1.json",
        "research/workbench/config/real-estate-structure-review-v1.json",
        "research/workbench/config/yen-currency-source-intake-v1.json",
        "research/workbench/config/yen-currency-structure-review-v1.json",
    )
    permitted = {ref["path"]: ref for ref in live}
    for name in user_paths:
        ref = permitted[name]
        rules.require(not saved.protected_reference(ref), "Never revisit prohibited content.")
        path = content_path(ROOT.parents[1] / name)
        rules.require(file_digest(path) == ref["sha256"], "A permitted user file changed.")
    return {"archival_references": 16, "permitted_live": 15, "permitted_user_live": 5}


def capture():
    _guard()
    with lifecycle.research_job():
        policy = _policy()
        rules.require(not FOLDER.exists(), "Preserve an existing capture or partial identity.")
        new_output_directory(FOLDER)
        write_new_receipt(FOLDER / "capture-started-v1.json", {"started_at": utc_now()})
        closed = read_json(checked_reference(policy["closed_financial_cycle"]))
        counts = saved.counts()
        rules.require(
            counts == closed["actual_authoritative_counts"]
            and counts["current_journal_event_count"] == 1554,
            "Use the actual closed J1554 financial predecessor without changing counts.",
        )
        financial = read_json(checked_reference(closed["private_result"]))
        source = read_json(checked_reference(financial["source_forecast_proof"]))
        rules.require(
            source["source_decisions"] == 121
            and source["complete_source_intervals"] == 119
            and source["input_resolution_records"] == 242,
            "Keep the complete original source tape, not a selected segment.",
        )
        capital = {}
        for key, ref in financial["review"]["paths"].items():
            receipt = read_json(checked_reference(ref))
            artifacts = {Path(row["path"]).name: row for row in receipt["artifacts"]}
            capital[key] = {
                "complete": ref,
                "account": artifacts["account.csv"],
                "held_weights": artifacts["weights.csv"],
                "desired": artifacts["desired.csv"],
                "start": receipt["metrics"]["start"],
                "end": receipt["metrics"]["end"],
                "rows": receipt["metrics"]["sessions"],
                "saved_max_drawdown": receipt["metrics"]["max_drawdown"],
            }
        preservation = read_json(saved.FOLDER / "preservation-baseline-v1.json")
        catalogue = {
            "captured_at": utc_now(),
            "study_id": rules.STUDY,
            "actual_authoritative_counts": counts,
            "closed_financial_cycle": policy["closed_financial_cycle"],
            "private_financial_result": closed["private_result"],
            "source_records": source["retained_records"],
            "source_proof": financial["source_forecast_proof"],
            "close": financial["frozen"]["market"]["close"],
            "capital": capital,
            "protected_reference_metadata": preservation["preserved_repo_refs"],
            "new_risk_statistics_computed": False,
            "new_financial_trials": 0,
            "prior_protected_content_boundary_error_preserved": True,
        }
        for ref in [catalogue["source_records"], catalogue["close"]]:
            checked_reference(ref)
        for item in capital.values():
            for name in ("account", "held_weights", "desired"):
                checked_reference(item[name])
        catalogue["permitted_checks"] = _permitted_refs(catalogue)
        write_new_receipt(CATALOGUE, catalogue)
        _progress(
            "captured",
            next_step="synthetic_validation_then_freeze",
            last_complete_financial_cycle=policy["closed_financial_cycle"],
        )
        return reference(CATALOGUE)


class ArchiveFeed(rules.MemoryFeed):
    def __init__(self, path: Path):
        points = []
        with content_path(path).open(encoding="utf-8") as stream:
            while line := stream.readline():
                before = json.loads(line)
                after_line = stream.readline()
                rules.require(bool(after_line), "Retain both halves of a stored source decision.")
                after = json.loads(after_line)
                day = before["decision_session"]
                audit = after["audit"]
                rules.require(
                    before["phase"] == "before_decision"
                    and after["phase"] == "after_decision"
                    and after["decision_session"] == audit["decision_session"] == day
                    and before["source_native_known_at"] == audit["source_native_known_at"],
                    "Stored source order and known-at coordinates differ.",
                )
                points.append(
                    rules.SourcePoint(
                        pd.Timestamp(day),
                        tuple(audit["expert_weights"]),
                        tuple(
                            tuple(float.fromhex(v) for v in row)
                            for row in before["source_targets_hex"]
                        ),
                        tuple(float.fromhex(v) for v in after["unified_target_hex"]),
                        tuple(pd.Timestamp(d) for d in before["source_native_known_at"]),
                        audit["complete_source_intervals"],
                    )
                )
        super().__init__(tuple(points))


def _csv(ref: dict) -> pd.DataFrame:
    return pd.read_csv(
        checked_reference(ref), index_col=0, parse_dates=True, float_precision="round_trip"
    )


def freeze():
    _guard()
    with lifecycle.research_job():
        rules.require(
            not INTENT.exists() and not (FOLDER / "freeze-started-v1.json").exists(),
            "Preserve the first complete or partial freeze.",
        )
        catalogue, policy = read_json(CATALOGUE), _policy()
        rules.require(
            saved.counts() == catalogue["actual_authoritative_counts"],
            "The financial baseline changed before diagnostic freeze.",
        )
        proof = read_json(FOLDER / "validation-source-v1.json")
        suites = (
            ElementTree.parse(checked_reference(proof["test_outputs"][0]))
            .getroot()
            .findall(".//testsuite")
        )
        rules.require(
            proof["exit_code"] == 0
            and proof["synthetic_only"] is True
            and proof["implementation_and_test_inputs"] == [reference(ROOT / p) for p in OWN]
            and suites
            and not any(
                int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
            ),
            "Freeze actual successful synthetic Interface checks and their exact inputs.",
        )
        names = {c.get("name").split("[", 1)[0] for s in suites for c in s.findall("testcase")}
        rules.require(
            {
                "test_archive_and_memory_cross_same_interface",
                "test_future_does_not_change_then_known_prefix",
                "test_old_pipelines_never_called",
                "test_signed_hedging_share_not_clipped",
                "test_zero_variance_is_undefined",
                "test_retains_reference_before_failure",
                "test_archive_guard_rejects_before_open",
            }.issubset(names),
            "Check real saved-record, causality, arithmetic and prohibited-reader Seams.",
        )
        write_new_receipt(FOLDER / "freeze-started-v1.json", {"started_at": utc_now()})
        write_new_receipt(
            INTENT,
            {
                "frozen_at": utc_now(),
                "criteria": policy,
                "catalogue": reference(CATALOGUE),
                "implementation_and_test_inputs": [reference(ROOT / p) for p in OWN],
                "complete_dependency_inputs": dependencies(),
                "validation": reference(FOLDER / "validation-source-v1.json"),
                "package_versions": _packages(),
                "python_version": sys.version,
                "actual_authoritative_counts": saved.counts(),
                "new_risk_statistics_computed": False,
                "new_financial_trials": 0,
                "protected_archival_reference_is_new_live_verification": False,
            },
        )
        _progress("frozen", next_step="actual_checkpoint_then_one_retained_risk_audit")
        return reference(INTENT)


def _checked_intent():
    intent = read_json(INTENT)
    rules.validate(intent["criteria"])
    checked_reference(intent["catalogue"])
    for ref in intent["implementation_and_test_inputs"] + intent["complete_dependency_inputs"]:
        checked_reference(ref)
    checked_reference(intent["validation"])
    rules.require(
        _packages() == intent["package_versions"],
        "The diagnostic's frozen package inventory changed.",
    )
    rules.require(
        saved.counts() == intent["actual_authoritative_counts"],
        "A read-only diagnosis cannot reset or hide financial events.",
    )
    return intent


def audit():
    _guard()
    with lifecycle.research_job():
        intent = _checked_intent()
        rules.require(
            not OUTPUT.exists() and not (FOLDER / "audit-started-v1.json").exists(),
            "Never retry a complete or partial frozen diagnostic.",
        )
        checkpoint = read_json(FOLDER / "actual-prefunding-checkpoint-v1.json")
        rules.require(
            checkpoint["entire_state_readback_equal"] is True
            and checkpoint["intent"] == reference(INTENT),
            "Require the actual successful post-freeze checkpoint before descriptive outcomes.",
        )
        write_new_receipt(
            FOLDER / "audit-started-v1.json",
            {
                "started_at": utc_now(),
                "pid": os.getpid(),
                "intent": reference(INTENT),
                "checkpoint": reference(FOLDER / "actual-prefunding-checkpoint-v1.json"),
            },
        )
        catalogue = read_json(checked_reference(intent["catalogue"]))
        _progress("running", active_compute_job={"pid": os.getpid(), "study": rules.STUDY})
        close = _csv(catalogue["close"]).loc[:, list(rules.ASSETS)]
        feed = ArchiveFeed(checked_reference(catalogue["source_records"]))
        rules.require(len(feed.origins) == 121, "Preserve every original source decision.")
        results, prefixes = {}, []
        tape = content_path(FOLDER / "risk-input-resolution-v1.jsonl")
        with tape.open("x", encoding="utf-8") as stream:
            for key, item in catalogue["capital"].items():
                lifecycle.runtime_guard()
                evidence = rules.PathEvidence(
                    key,
                    _csv(item["account"]),
                    _csv(item["held_weights"]),
                    _csv(item["desired"]),
                    close,
                )
                rules.require(
                    len(evidence.account) == item["rows"]
                    and str(evidence.account.index[0].date()) == item["start"]
                    and str(evidence.account.index[-1].date()) == item["end"],
                    "Keep all four full original funded windows.",
                )

                def retain(row, *, cutoff=None):
                    stream.write(
                        json.dumps(
                            {**row, "diagnostic_prefix": cutoff},
                            allow_nan=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )

                result = rules.audit_path(evidence, feed, retain=retain)
                rules.require(
                    abs(
                        result["summary"]["maximum_drawdown_from_saved_NAV"]
                        - item["saved_max_drawdown"]
                    )
                    <= rules.TOLERANCE,
                    "Saved main NAV disagrees with the closed financial drawdown context.",
                )
                results[key] = result
                for cutoff in intent["criteria"]["diagnostic_prefix_cutoffs"]:
                    lifecycle.runtime_guard()
                    day = pd.Timestamp(cutoff)
                    if day < evidence.account.index[0]:
                        prefixes.append(
                            {
                                "path": key,
                                "cutoff": cutoff,
                                "status": "not_applicable_before_funding",
                            }
                        )
                        continue
                    short = rules.audit_path(
                        replace(
                            evidence,
                            account=evidence.account.loc[:day],
                            held_weights=evidence.held_weights.loc[:day],
                        ),
                        feed,
                        retain=lambda row, cutoff=cutoff: retain(row, cutoff=cutoff),
                    )
                    expected = {d: h for d, h in result["daily_digests"].items() if d <= cutoff}
                    rules.require(
                        short["daily_digests"] == expected,
                        "Future records changed a then-known diagnostic prefix.",
                    )
                    prefixes.append(
                        {
                            "path": key,
                            "cutoff": cutoff,
                            "status": "then_known_daily_statistics_exact",
                            "days_checked": len(expected),
                        }
                    )
        protection = _permitted_refs(catalogue)
        _checked_intent()
        for ref in [catalogue["source_records"], catalogue["close"]]:
            checked_reference(ref)
        for item in catalogue["capital"].values():
            for name in ("account", "held_weights", "desired"):
                checked_reference(item[name])
        rules.require(
            saved.counts() == catalogue["actual_authoritative_counts"],
            "Read-only risk outcomes cannot create financial trials.",
        )
        write_new_receipt(
            OUTPUT,
            {
                "completed_at": utc_now(),
                "study_id": rules.STUDY,
                "intent": reference(INTENT),
                "catalogue": reference(CATALOGUE),
                "paths": results,
                "prefix_checks": prefixes,
                "retained_records": reference(tape),
                "permitted_checks": protection,
                "actual_authoritative_counts": saved.counts(),
                "new_financial_trials": 0,
                "old_learner_account_BT_or_prefix_runs": 0,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        _progress(
            "complete",
            next_step="sanitized_aggregate_then_cycle_closure",
            last_complete_financial_cycle=catalogue["closed_financial_cycle"],
        )
        return reference(OUTPUT)


def _context(row):
    if row is None:
        return None
    risk = row.get("risk")
    return {
        key: row[key]
        for key in (
            "date",
            "drawdown",
            "gross",
            "source_origin",
            "source_complete_intervals",
            "known_returns",
            "source_probability_hhi",
            "actual_invested_asset_hhi",
            "actual_to_saved_mixture_L1",
            "source_probability_leader",
            "risk_status",
        )
        if key in row
    } | {
        "actual_sample_variance": None if risk is None else risk["actual_variance"],
        "source_mixture_sample_variance": None if risk is None else risk["proposed_variance"],
    }


def export():
    _guard()
    with lifecycle.research_job():
        _checked_intent()
        result = read_json(OUTPUT)
        paths = {}
        for key, row in result["paths"].items():
            worst = row["contexts"]["worst_drawdown"]
            paths[key] = {
                "summary": row["summary"],
                "first_strict_15pct_breach": _context(row["contexts"]["first_strict_15pct_breach"]),
                "worst_drawdown": None
                if worst is None
                else {
                    "peak_date": worst["peak"]["date"],
                    "trough": _context(worst["trough"]),
                    "model_cost_since_peak": worst["model_cost_since_peak"],
                    "turnover_since_peak": worst["turnover_since_peak"],
                    "model_tickets_since_peak": worst["model_tickets_since_peak"],
                },
                "maximum_independent_risk_error": row["maximum_independent_risk_error"],
                "maximum_recorded_mixture_error": row["maximum_recorded_mixture_error"],
            }
        write_new_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "record_type": "retained_main_risk_source_selection_diagnostic",
                "study_id": rules.STUDY,
                "completed_at": result["completed_at"],
                "private_result": reference(OUTPUT),
                "intent": reference(INTENT),
                "paths": paths,
                "prefix_checks": result["prefix_checks"],
                "actual_authoritative_counts": result["actual_authoritative_counts"],
                "new_financial_trials": 0,
                "old_learner_account_BT_or_prefix_runs": 0,
                "raw_prices_targets_covariances_daily_books_or_states_exported": False,
                "source_risk_contribution_is_realized_main_PnL_or_wealth_bound": False,
                "full_window_drawdown_contexts_are_hindsight_not_prefix_causal": True,
                "sample_covariance_is_a_future_drawdown_certificate": False,
                "financial_gates_changed": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        return reference(PUBLIC)


def main():
    parser = argparse.ArgumentParser(description="Read-only preserved main-risk diagnosis.")
    parser.add_argument("action", choices=("capture", "freeze", "audit", "export"))
    args = parser.parse_args()
    try:
        result = globals()[args.action]()
    except (QuantError, PermissionError, OSError, ValueError) as error:
        if FOLDER.exists() and FOLDER.resolve() == FOLDER:
            path = FOLDER / f"{args.action}-failure-v1.json"
            if not path.exists():
                write_new_receipt(
                    path,
                    {
                        "failed_at": utc_now(),
                        "stage": args.action,
                        "error": str(error),
                        "new_financial_trials": 0,
                    },
                )
        raise
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
