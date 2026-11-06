from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import hierarchical_risk_transfer_v1 as rules
from us_quant import ssd_five_hierarchical_risk_v1 as source
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_style_activity_financial_v1 import dependencies as _dependencies
from us_quant.storage import file_digest, read_json, utc_now, write_json

ROOT, base = source.ROOT, source.base
lifecycle, reference, unique_refs = source.lifecycle, source.reference, source.unique_refs
FOLDER = base.REPORTS / "hierarchical-risk-transfer-20261106-v1"
POLICY = ROOT / "config/hierarchical-risk-transfer-v1.json"
INTENT, RESULT = FOLDER / "analysis-intent-v1.json", FOLDER / "diagnostic-result-v1.json"
PUBLIC = ROOT / "evidence/hierarchical_risk_transfer_20261106_v1.json"
OWN = (
    "config/hierarchical-risk-transfer-v1.json",
    "src/us_quant/hierarchical_risk_transfer_v1.py",
    "src/us_quant/ssd_hierarchical_risk_transfer_v1.py",
    "tests/test_hierarchical_risk_transfer_v1.py",
    "HIERARCHICAL_RISK_TRANSFER_V1.md",
    "tests/conftest.py",
)


def policy():
    expected = {
        "schema_version": 1,
        "study_id": "ssd_hierarchical_risk_transfer_20261106_v1",
        "actual_native_event_seq": 184910,
        "prior_journal_sequence": 1393,
        "prior_journal_head_sha256": "3d13b71e0935bdc60678c7eddcc57721c02b4d226168cde314104bdd7997e3e9",
        "prior_completed_configurations": 295,
        "source_catalogue": {
            "path": (
                "reports/ssd-continuous-research-v1/hierarchical-risk-transfer-20261106-v1/"
                "source-input-catalogue-v1.json"
            ),
            "sha256": "21122f0080e544cab06e81118fc9de851aaf8f4d23750f0f1ec39ad2bb0d14f5",
        },
        "forecast": "unchanged_recorded_hierarchical_annualized_NET_RF_volatility",
        "forecast_daily_variance": "recorded_volatility_squared_divided_by_252",
        "label_horizon": "following_complete_exchange_calendar_month",
        "label_channels": list(rules.CHANNELS),
        "canonical_variance": "scalar_fsum_ddof1_exact_constant_zero",
        "independent_variance_relative_tolerance": 1e-10,
        "independent_variance_absolute_tolerance": 1e-18,
        "recorded_forecast_covariance_check_tolerance": 1e-12,
        "full_calendar_month_required_for_both_labels": True,
        "initial_five_year_partial_month_kept_unlabelled": True,
        "terminal_partial_month_kept_unlabelled": True,
        "zero_variance_is_safe_success": False,
        "zero_forecast_variance_ratio": None,
        "all_121_origins_and_four_path_contexts_retained": True,
        "label_gap_is_causal_fee_or_source_profit_attribution": False,
        "new_covariance_fits_targets_accounts_or_financial_trials": 0,
        "new_model_or_allocation_selected": False,
        "new_source_requests_or_RF_views": 0,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    value = read_json(POLICY)
    rules.require(
        value == expected and all(type(value[k]) is type(v) for k, v in expected.items()),
        "Freeze exact forecast/label scopes without a window, floor or financial waiver.",
    )
    return value


def progress(phase, **fields):
    counts = source.current_counts(base.journal_events(base.JOURNAL))
    rules.require(
        all(k not in fields or fields[k] == v for k, v in counts.items()),
        "A calibration cannot replace authoritative financial counts.",
    )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "hierarchical_risk_transfer_" + phase,
            "study_id": policy()["study_id"],
            "active_compute_job": None,
            **counts,
            **fields,
            "new_financial_trials": 0,
            "order_authority": False,
        },
    )


def sources():
    catalogue = read_json(base.checked_reference(policy()["source_catalogue"]))
    previous = read_json(base.checked_reference(catalogue["previous_completed_cycle"]))
    return unique_refs(
        [
            *previous["exact_evidence_references"],
            *catalogue["source_references"],
            policy()["source_catalogue"],
            *catalogue["main_account_refs"].values(),
            *[
                catalogue[k]
                for k in (
                    "hierarchy_policy",
                    "hierarchy_freeze",
                    "hierarchy_information_audits",
                    "hierarchy_input_resolved_tape",
                    "hierarchy_results",
                    "previous_completed_cycle",
                )
            ],
            *[
                reference(FOLDER / name)
                for name in (
                    "capture-preflight-v1.py",
                    "actual-invocation-v1.json",
                    "design-decision-v1.json",
                    "preservation-baseline-v1.json",
                )
            ],
        ]
    )


def dependencies():
    return FunctionType(
        _dependencies.__code__,
        {**_dependencies.__globals__, **globals(), "prior": source},
        argdefs=_dependencies.__defaults__,
    )()


def check_frozen():
    frozen = read_json(INTENT)
    for ref in (
        frozen["source_references"]
        + frozen["implementation_dependencies"]
        + frozen["validation_evidence"]
    ):
        base.checked_reference(ref)
    baseline = read_json(FOLDER / "preservation-baseline-v1.json")
    rules.require(
        sys.version == baseline["python_version"], "The accepted full Python version changed."
    )
    actual_packages = sorted([d.metadata["Name"], d.version] for d in distributions())
    rules.require(
        actual_packages == baseline["package_versions"], "Frozen package versions changed."
    )
    for ref in baseline["preserved_repo_refs"]:
        rules.require(
            file_digest(base.REPO / ref["path"]) == ref["sha256"],
            "A preserved user or legacy source changed.",
        )
    return frozen


def freeze():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Keep the first complete or partial risk-transfer freeze.")
    with lifecycle.research_job():
        declared = policy()
        history = base.journal_events(base.JOURNAL)
        rules.require(
            len(history) == 1393 and history[-1]["sha256"] == declared["prior_journal_head_sha256"],
            "Freeze against actual closed J1393, not stale progress.",
        )
        design = read_json(FOLDER / "design-decision-v1.json")
        rules.require(
            design["actual_native_event_seq"] == 184910
            and design["new_future_risk_labels_or_calibration_computed"] is False,
            "The comparison must be decided before its new labels.",
        )
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        validation = read_json(FOLDER / "validation-source-v1.json")
        xml, log = [base.checked_reference(r) for r in validation["test_outputs"]]
        suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
        rules.require(
            validation["implementation_and_test_inputs"] == [reference(ROOT / p) for p in OWN]
            and validation["exit_code"] == 0
            and suites
            and validation["passed_tests"] == sum(int(s.get("tests", 0)) for s in suites)
            and not any(
                int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
            ),
            "Bind all source/main scope, month censoring, variance and prefix checks.",
        )
        refs, deps = sources(), dependencies()
        for ref in [*refs, *deps]:
            base.checked_reference(ref)
        write_new_receipt(
            INTENT,
            {
                "frozen_at": utc_now(),
                "study_id": declared["study_id"],
                "policy": reference(POLICY),
                "source_references": refs,
                "implementation_dependencies": unique_refs(
                    [*deps, *[reference(ROOT / p) for p in OWN]]
                ),
                "validation_evidence": [
                    reference(xml),
                    reference(log),
                    reference(FOLDER / "validation-source-v1.json"),
                ],
                "new_future_risk_labels_or_calibration_computed": False,
                "new_financial_trials": 0,
                "new_models_choices_targets_or_accounts": 0,
            },
        )
        progress(
            "frozen",
            intent=reference(INTENT),
            next_step="run_one_paired_scope_calibration_without_source_or_account_replay",
        )
    return reference(INTENT)


def load_inputs():
    catalogue = read_json(base.checked_reference(policy()["source_catalogue"]))
    _, experts, _, _, clocks, _ = source.source_inputs()
    rules.require(clocks == catalogue["source_clocks"], "Original source clocks or views changed.")
    observations = [e.observe(pd.Timestamp("2026-10-05"), 2512) for e in experts]
    rf = observations[0].risk_free
    rules.require(
        all(len(o.net_returns) == 2512 and o.risk_free.equals(rf) for o in observations),
        "Reuse full exact formal source NET/RF, not a normalized or new view.",
    )
    net = pd.DataFrame({o.identity.identifier: o.net_returns for o in observations})
    mains = {}
    for key in rules.PATHS:
        frame = pd.read_csv(
            base.checked_reference(catalogue["main_account_refs"][key]),
            index_col=0,
            parse_dates=True,
            float_precision="round_trip",
        )
        previous = frame["equity"].shift(1)
        previous.iloc[0] = 10000.0
        rules.require(
            np.allclose(frame["equity"] / previous - 1, frame["return"], rtol=0, atol=1e-12)
            and (frame["equity"] > 0).all()
            and (frame["cash"] >= 0).all()
            and frame["gross_exposure"].between(0, 1).all(),
            "A main label must be its unchanged cash-funded net capital record.",
        )
        mains[key] = frame
    audits = read_json(base.checked_reference(catalogue["hierarchy_information_audits"]))[
        "decisions"
    ]
    rules.require(len(audits) == 121, "Keep all original forecast and unscored origins.")
    return audits, net, rf, mains


def verify_trace(path, rows, audits, net, rf, mains):
    with path.open() as stream:
        for row in rows:
            first, second = stream.readline(), stream.readline()
            rules.require(bool(first and second), "Keep a truncated calibration trace as failure.")
            before, after = json.loads(first), json.loads(second)
            audit = audits[row["decision_index"]]
            rules.require(
                before["record"] == "risk_transfer_input"
                and after["record"] == "risk_transfer_resolved"
                and before["original_audit"] == audit
                and after["result"] == row
                and before["case"]["path_key"] == row["path_key"]
                and before["case"]["decision_session"] == row["decision_session"],
                "Retain each exact original forecast and resolved path case.",
            )
            if row["forecast_available"]:
                wanted = rules.future_calendar(pd.Timestamp(row["decision_session"]))
                sd, md = (
                    wanted.intersection(net.index),
                    wanted.intersection(mains[row["path_key"]].index),
                )
                rules.require(
                    before["required_dates"] == wanted.strftime("%Y-%m-%d").tolist()
                    and before["source_dates"] == sd.strftime("%Y-%m-%d").tolist()
                    and before["main_dates"] == md.strftime("%Y-%m-%d").tolist()
                    and before["source_NET_binary64_hex"]
                    == [[float(v).hex() for v in a] for a in net.loc[sd].to_numpy()]
                    and before["source_RF_binary64_hex"] == [float(v).hex() for v in rf.loc[sd]]
                    and before["main_NET_binary64_hex"]
                    == [float(v).hex() for v in mains[row["path_key"]].loc[md, "return"]]
                    and before["main_RF_binary64_hex"]
                    == [float(v).hex() for v in mains[row["path_key"]].loc[md, "risk_free"]],
                    "Full original source/main values and censoring dates changed.",
                )
            if row["label"] is not None:
                predicted, weights, _ = rules.forecast_variance(audit)
                checked = rules.assess(
                    predicted,
                    weights,
                    net.loc[wanted].to_numpy(),
                    rf.loc[wanted].to_numpy(),
                    mains[row["path_key"]].loc[wanted, "return"].to_numpy(),
                )
                rules.require(
                    checked == row["label"], "A verified label differs from saved inputs."
                )
        rules.require(not stream.readline(), "Unexpected extra calibration input records.")
    return {
        "actual_input_and_resolved_record_count": 2 * len(rows),
        "exact_source_main_forecast_values_and_statuses_checked": True,
    }


def run():
    lifecycle.runtime_guard()
    if RESULT.exists() or (FOLDER / "attempt-v1.json").exists():
        raise QuantError("Do not replay or repair a complete or partial risk-transfer identity.")
    with lifecycle.research_job():
        check_frozen()
        before = source.current_counts(base.journal_events(base.JOURNAL))
        write_new_receipt(
            FOLDER / "attempt-v1.json",
            {
                "started_at": utc_now(),
                "pid": os.getpid(),
                "intent": reference(INTENT),
            },
        )
        started = time.monotonic()
        try:
            audits, net, rf, mains = load_inputs()
            trace = FOLDER / "actual-inputs-resolved-v1.jsonl"
            with trace.open("x") as stream:

                def retain(row):
                    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())

                rows = rules.diagnose(
                    audits, net, rf, mains, before_case=lifecycle.runtime_guard, retain=retain
                )
            rules.require(len(rows) == 484, "Keep all121origins in all four contexts.")
            verified = verify_trace(trace, rows, audits, net, rf, mains)
            cuts = []
            for text in ("2017-03-31", "2021-09-30", "2021-10-29", "2023-12-29", "2026-09-30"):
                lifecycle.runtime_guard()
                day = pd.Timestamp(text)
                early_audits = [a for a in audits if pd.Timestamp(a["decision_session"]) <= day]
                early = rules.diagnose(
                    early_audits,
                    net.loc[:day],
                    rf.loc[:day],
                    {k: frame.loc[:day] for k, frame in mains.items()},
                )
                by_case = {(r["decision_session"], r["path_key"]): r for r in early}
                count = 0
                for full in rows:
                    if full["status"] == "complete_paired_risk_label" and (
                        pd.Timestamp(full["required_label_end"]) <= day
                    ):
                        rules.require(
                            by_case[full["decision_session"], full["path_key"]] == full,
                            "Future rows changed an already complete past risk label.",
                        )
                        count += 1
                cuts.append({"cutoff": text, "complete_past_cases_exactly_matched": count})
            after = source.current_counts(base.journal_events(base.JOURNAL))
            rules.require(after == before, "Risk calibration changed a financial trial count.")
            result = {
                "completed_at": utc_now(),
                "case_rows": rows,
                "summary": rules.summarize(rows),
                "verification": verified,
                "prefix_checks": cuts,
                "actual_authoritative_counts": after,
                "new_financial_trials": 0,
                "wall_seconds": time.monotonic() - started,
                "Darwin_peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "input_trace": reference(trace),
                "original_forecast_targets_or_accounts_replayed": False,
            }
            write_new_receipt(RESULT, result)
            write_new_receipt(
                PUBLIC,
                {
                    "schema_version": 1,
                    "study_id": policy()["study_id"],
                    "completed_at": result["completed_at"],
                    **result["summary"],
                    "prefix_checks": cuts,
                    "actual_authoritative_counts": after,
                    "case_count": len(rows),
                    "input_resolved_record_count": verified[
                        "actual_input_and_resolved_record_count"
                    ],
                    "original_forecast_targets_or_accounts_replayed": False,
                    "raw_prices_NET_returns_or_daily_book_rows_exported": False,
                    "order_authority": False,
                },
            )
            progress(
                "diagnosis_complete",
                diagnostic=reference(RESULT),
                public_result=reference(PUBLIC),
                next_step="publish_scoped_risk_findings_without_financial_metric_splicing",
            )
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            write_new_receipt(
                FOLDER / "failure-v1.json",
                {
                    "failed_at": utc_now(),
                    "pid": os.getpid(),
                    "error": str(error),
                    "partial_inputs_preserved": True,
                    "financial_trials_added": 0,
                    "old_source_or_frozen_algorithm_rescue": False,
                },
            )
            progress("terminal_failure_preserved", next_step="do_not_retry_or_retune_this_identity")
            raise
    return {"private": reference(RESULT), "public": reference(PUBLIC)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "run": run}[args.action]()))
