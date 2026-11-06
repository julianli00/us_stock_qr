from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import current_opinion_risk_calibration_v1 as rules
from us_quant import ssd_growth_macro_defense_hedge_v1 as source
from us_quant.calendar import sessions
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_nested_growth_allocation_audit_v1 import pairs
from us_quant.ssd_style_activity_financial_v1 import dependencies as _dependencies
from us_quant.storage import read_json, utc_now, write_json

base, ROOT, lifecycle = source.base, source.ROOT, source.lifecycle
reference, unique_refs = source.reference, source.unique_refs
FOLDER = base.REPORTS / "current-opinion-risk-calibration-20261106-v1"
POLICY = ROOT / "config/current-opinion-risk-calibration-v1.json"
INTENT = FOLDER / "diagnostic-implementation-freeze-v1.json"
RESULTS = FOLDER / "results-v1.json"
PUBLIC = ROOT / "evidence/current_opinion_risk_calibration_20261106_v1.json"
OWN = (
    "config/current-opinion-risk-calibration-v1.json",
    "src/us_quant/current_opinion_risk_calibration_v1.py",
    "src/us_quant/ssd_current_opinion_risk_calibration_v1.py",
    "tests/test_current_opinion_risk_calibration_v1.py",
    "tests/conftest.py",
    "CURRENT_OPINION_RISK_CALIBRATION_V1.md",
)


def dependencies():
    return FunctionType(
        _dependencies.__code__,
        {**_dependencies.__globals__, **globals()},
        argdefs=_dependencies.__defaults__,
    )()


def policy():
    value = read_json(POLICY)
    expected = {
        "schema_version": 1,
        "study_id": "ssd_current_opinion_risk_calibration_20261106_v1",
        "actual_native_event_seq": 181145,
        "prior_journal_sequence": 1320,
        "history_sessions": 126,
        "future_label_sessions": 21,
        "sample_covariance_ddof": 1,
        "numeric_variance_agreement_tolerance": 1e-12,
        "models": list(rules.MODELS),
        "recorded_outer_views": list(rules.VIEWS),
        "source_inputs": {
            "path": str((FOLDER / "source-inputs-v1.json").relative_to(ROOT)),
            "sha256": "a228cb81d09de8e74521fee6e5250c087f0e0294aca8630274e5fe2677ec0b59",
        },
        "unmatured_history": "retain_decision_unscored_no_fallback",
        "incomplete_future_label": "retain_forecast_unlabelled_no_filling",
        "current_opinion_projection_includes_fees_or_future_policy_changes": False,
        "labels_are_main_accounts_or_tradable_portfolio_returns": False,
        "new_allocation_or_model_selection": False,
        "new_financial_trials": 0,
        "new_source_requests_or_RF_views": 0,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    rules.require(
        value == expected and all(type(value[k]) is type(v) for k, v in expected.items()),
        "Freeze both risk descriptions, original source scope and separate future-label Interface.",
    )
    return value


def progress(phase, **fields):
    counts = source.current_counts(base.journal_events(base.JOURNAL))
    rules.require(
        all(k not in fields or fields[k] == v for k, v in counts.items()),
        "Risk-only calibration cannot modify authoritative financial counts.",
    )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "current_opinion_risk_calibration_" + phase,
            "study_id": policy()["study_id"],
            "active_compute_job": None,
            **counts,
            **fields,
            "order_authority": False,
        },
    )


def source_refs():
    inputs = read_json(base.checked_reference(policy()["source_inputs"]))
    previous = read_json(base.checked_reference(inputs["previous_completed_cycle"]))
    return unique_refs(
        [
            *previous["exact_evidence_references"],
            *[
                inputs[k]
                for k in (
                    "previous_completed_cycle",
                    "quartet_implementation_freeze",
                    "quartet_input_resolved_tape",
                    "quartet_information_audits",
                    "quartet_prepared_targets",
                )
            ],
            *inputs["quartet_registration_packets"],
            *[
                reference(FOLDER / name)
                for name in (
                    "source-inputs-v1.json",
                    "design-decision-v1.json",
                    "actual-invocation-v1.json",
                    "preservation-baseline-v1.json",
                    "capture-preflight-v1.py",
                    "reader-contract-preflight-v1.json",
                )
            ],
        ]
    )


def check_frozen():
    frozen = read_json(INTENT)
    for ref in (
        frozen["implementation_dependencies"]
        + frozen["source_references"]
        + frozen["validation_evidence"]
    ):
        base.checked_reference(ref)
    baseline = read_json(FOLDER / "preservation-baseline-v1.json")
    rules.require(
        frozen["implementation_and_test_inputs"] == [reference(ROOT / p) for p in OWN]
        and sys.version == baseline["python_version"]
        and [list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())]
        == baseline["package_versions"],
        "Keep frozen code, source identities and the accepted complete environment.",
    )
    return frozen


def freeze():
    lifecycle.runtime_guard()
    rules.require(
        not INTENT.exists() and not (FOLDER / "freeze-attempt-v1.json").exists(),
        "Keep the first complete or partial calibration freeze.",
    )
    with lifecycle.research_job():
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        policy()
        baseline = read_json(FOLDER / "preservation-baseline-v1.json")
        rules.require(
            source.current_counts(base.journal_events(base.JOURNAL)) == baseline["actual_counts"]
            and baseline["actual_counts"]["current_journal_event_count"] == 1320,
            "Use the unchanged actual J1320/289 financial baseline.",
        )
        validation = read_json(FOLDER / "validation-source-v1.json")
        xml, _ = [base.checked_reference(r) for r in validation["test_outputs"]]
        suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
        rules.require(
            validation["implementation_and_test_inputs"] == [reference(ROOT / p) for p in OWN]
            and validation["exit_code"] == 0
            and validation["synthetic_only"] is True
            and suites
            and validation["passed_tests"] == sum(int(s.get("tests", 0)) for s in suites)
            and not any(
                int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
            ),
            "Freeze successful causal-prefix and separate-label synthetic checks.",
        )
        refs = source_refs()
        for ref in refs:
            base.checked_reference(ref)
        write_new_receipt(
            INTENT,
            {
                "frozen_at": utc_now(),
                "policy": reference(POLICY),
                "implementation_and_test_inputs": [reference(ROOT / p) for p in OWN],
                "implementation_dependencies": dependencies(),
                "source_references": refs,
                "validation_evidence": [
                    *validation["test_outputs"],
                    reference(FOLDER / "validation-source-v1.json"),
                ],
                "actual_authoritative_counts": baseline["actual_counts"],
                "new_risk_predictions_or_future_labels_computed": False,
                "new_financial_trials": 0,
                "order_authority": False,
            },
        )
        progress("frozen", implementation_freeze=reference(INTENT), new_financial_trials=0)
    return reference(INTENT)


def prediction(before, after, asset_returns):
    observations = before["observations"]
    day = pd.Timestamp(before["as_of"])
    rules.require(
        len(observations) == 4
        and tuple(o["id"] for o in observations) == rules.EXPERT_IDS
        and all(
            o["as_of"] == before["as_of"]
            and o["target_known_at"] <= before["as_of"]
            and tuple(o["asset_order"]) == rules.ASSETS
            and o["score_mature"] == (len(o["NET_returns"]) == rules.HISTORY)
            and len(o["NET_returns"]) <= rules.HISTORY
            for o in observations
        )
        and after["audit"]["decision_session"] == before["as_of"],
        "Keep exact source order, opinions, clocks and maturity without source recomputation.",
    )
    if not all(o["score_mature"] for o in observations):
        return {"as_of": before["as_of"], "status": "unscored_immature_source_history"}
    index = pd.DatetimeIndex(observations[0]["return_dates"])
    rules.require(
        all(
            o["return_dates"] == observations[0]["return_dates"]
            and o["RF"] == observations[0]["RF"]
            for o in observations
        ),
        "Preserve the common admitted assessment RF without inventing a native RF bridge.",
    )
    net = pd.DataFrame(
        np.array([o["NET_returns"] for o in observations]).T,
        index=index,
        columns=rules.EXPERT_IDS,
    )
    result = rules.forecast(
        day,
        asset_returns.loc[index],
        net,
        pd.Series(observations[0]["RF"], index=index),
        np.array([o["target"] for o in observations]),
        {
            "equal": np.array(after["audit"]["equal_expert_weights"]),
            "recorded_rolling": np.array(after["audit"]["rolling_expert_weights"]),
        },
    )
    return {"as_of": before["as_of"], "status": "forecast_available", "forecast": result}


def calibrate():
    lifecycle.runtime_guard()
    rules.require(
        not RESULTS.exists() and not (FOLDER / "calibration-attempt-v1.json").exists(),
        "Preserve the first complete or partial actual calibration; do not retry it.",
    )
    with lifecycle.research_job():
        frozen = check_frozen()
        write_new_receipt(FOLDER / "calibration-attempt-v1.json", {"started_at": utc_now()})
        inputs = read_json(base.checked_reference(policy()["source_inputs"]))
        data, experts, _, _, clocks, _ = source.source_inputs()
        rules.require(clocks == inputs["source_clocks"], "Retain actual source/admission clocks.")
        expected = sessions("2016-10-06", "2026-10-05")
        full = [expert.observe(data.close.index[-1], len(expected)) for expert in experts]
        rules.require(
            tuple(o.identity.identifier for o in full) == rules.EXPERT_IDS
            and all(
                o.net_returns.index.equals(expected)
                and o.risk_free.equals(data.risk_free.loc[expected])
                for o in full
            ),
            "Labels require the existing complete source NET series and admitted exact RF view.",
        )
        full_net = pd.DataFrame(
            {o.identity.identifier: o.net_returns for o in full}, columns=rules.EXPERT_IDS
        )
        assets = pd.DataFrame(
            {
                symbol: [
                    float(data.close[symbol].iloc[i]) / float(data.close[symbol].iloc[i - 1]) - 1
                    for i in range(1, len(data.close))
                ]
                for symbol in rules.ASSETS
            },
            index=data.close.index[1:],
        ).loc[expected]
        retained = list(pairs(inputs["quartet_input_resolved_tape"]))
        rows, scored, predictions = [], [], []
        log_path = FOLDER / "actual-inputs-resolved-v1.jsonl"
        with log_path.open("x") as stream:

            def retain(row):
                stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

            try:
                for before, after in retained:
                    lifecycle.runtime_guard()
                    day = pd.Timestamp(before["as_of"])
                    for observation in before["observations"]:
                        dates = pd.DatetimeIndex(observation["return_dates"])
                        rules.require(
                            full_net.loc[dates, observation["id"]].tolist()
                            == observation["NET_returns"]
                            and data.risk_free.loc[dates].tolist() == observation["RF"],
                            "Saved source history and full label reader must be exactly identical.",
                        )
                    past = assets.loc[:day].tail(rules.HISTORY)
                    retain(
                        {
                            "record": "forecast_input",
                            "source_input": before,
                            "outer_audit": after["audit"],
                            "asset_return_dates": [str(x.date()) for x in past.index],
                            "asset_returns": past.to_numpy().tolist(),
                            "new_prediction_not_yet_computed": True,
                        }
                    )
                    result = prediction(before, after, assets)
                    predictions.append(result)
                    retain({"record": "forecast_resolved", "result": result})
                    future = full_net.loc[full_net.index > day].iloc[: rules.HORIZON]
                    if result["status"] == "forecast_available":
                        if len(future) == rules.HORIZON:
                            rf = data.risk_free.loc[future.index]
                            retain(
                                {
                                    "record": "future_label_input",
                                    "as_of": before["as_of"],
                                    "dates": [str(x.date()) for x in future.index],
                                    "NET_returns": future.to_numpy().tolist(),
                                    "RF": rf.tolist(),
                                    "label_not_yet_computed": True,
                                }
                            )
                            labels = rules.assess(result["forecast"], future, rf)
                            result = {
                                **result,
                                "label_status": "complete_next_21_sessions",
                                "calibration": labels,
                            }
                            scored.extend(labels)
                        else:
                            result = {
                                **result,
                                "label_status": "incomplete_future_retained_unlabelled",
                                "available_future_sessions": len(future),
                            }
                        retain({"record": "label_resolved", "result": result})
                    rows.append(result)
            except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
                write_new_receipt(
                    FOLDER / "calibration-failure-v1.json",
                    {
                        "failed_at": utc_now(),
                        "error": str(error),
                        "completed_origins": len(rows),
                        "partial_evidence_preserved": True,
                        "terminal_no_retry_or_repair": True,
                        "new_financial_trials": 0,
                    },
                )
                progress(
                    "terminal_failure_preserved", next_step="do_not_retry_or_patch_frozen_code"
                )
                raise
        rules.require(len(rows) == 121, "Keep every original initial/month-end source origin.")
        prefixes = []
        for position in (0, 6, 12, 60, 120):
            day = pd.Timestamp(retained[position][0]["as_of"])
            early = [prediction(b, a, assets.loc[:day]) for b, a in retained[: position + 1]]
            rules.require(
                early == predictions[: position + 1], "Future data changed a past forecast."
            )
            prefixes.append(str(day.date()))
        summaries = []
        for view in rules.VIEWS:
            for model in rules.MODELS:
                selected = [r for r in scored if r["view"] == view and r["model"] == model]
                errors = [r["absolute_variance_error"] for r in selected]
                require_count = len(selected)
                rules.require(
                    require_count > 0, "No labelled forecast can receive an empty summary."
                )
                summaries.append(
                    {
                        "view": view,
                        "model": model,
                        "labelled_origins": require_count,
                        "mean_absolute_daily_variance_error": math.fsum(errors) / require_count,
                        "median_absolute_daily_variance_error": statistics.median(errors),
                        "mean_signed_daily_variance_error": math.fsum(
                            r["signed_variance_error"] for r in selected
                        )
                        / require_count,
                        "underforecast_count": sum(r["underforecast"] for r in selected),
                    }
                )
        counts = source.current_counts(base.journal_events(base.JOURNAL))
        rules.require(
            counts == frozen["actual_authoritative_counts"], "No financial journal changes."
        )
        write_new_receipt(
            RESULTS,
            {
                "completed_at": utc_now(),
                "implementation_freeze": reference(INTENT),
                "source_origins": len(rows),
                "unscored_origins": sum(row["status"] != "forecast_available" for row in rows),
                "forecast_origins": len(rows)
                - sum(row["status"] != "forecast_available" for row in rows),
                "incomplete_future_origins": sum(
                    "available_future_sessions" in row for row in rows
                ),
                "forecast_prefixes": prefixes,
                "summary": summaries,
                "origins": rows,
                "actual_input_resolved_tape": reference(log_path),
                "actual_authoritative_counts": counts,
                "new_financial_trials": 0,
                "new_model_selected_or_pool_or_financial_acceptance_changed": False,
                "source_strategy_account_BT_or_CAGR_Sharpe_replayed": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        progress(
            "actual_calibration_complete", private_result=reference(RESULTS), new_financial_trials=0
        )
    return reference(RESULTS)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(RESULTS)
        write_new_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "study_id": policy()["study_id"],
                **{
                    k: saved[k]
                    for k in (
                        "completed_at",
                        "source_origins",
                        "unscored_origins",
                        "forecast_origins",
                        "incomplete_future_origins",
                        "forecast_prefixes",
                        "summary",
                        "actual_authoritative_counts",
                        "new_financial_trials",
                    )
                },
                "current_opinion_projection_includes_fees_or_future_policy_changes": False,
                "labels_are_main_accounts_or_tradable_portfolio_returns": False,
                "new_model_selected_or_pool_or_financial_acceptance_changed": False,
                "raw_prices_targets_returns_or_daily_rows_exported": False,
                "source_strategy_account_BT_or_CAGR_Sharpe_replayed": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
    return reference(PUBLIC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "calibrate", "export"))
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "calibrate": calibrate, "export": export}[args.action]()))
