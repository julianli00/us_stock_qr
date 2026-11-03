from __future__ import annotations

import argparse
import ast
import itertools
import json
import math
import os
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import ssd_eight_expert_meta_v1 as source
from us_quant.calendar import is_month_end, sessions
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.storage import file_digest, read_json, utc_now

ROOT, BASE = source.ROOT, source.base
FOLDER = BASE.REPORTS / "pair-risk-forecast-diagnosis-20261103-v1"
DESIGN = FOLDER / "design-decision-v1.json"
INTENT = FOLDER / "analysis-intent-v1.json"
OUTPUT = FOLDER / "diagnostic-v1.json"
SUMMARY = FOLDER / "summary-v1.json"
PUBLIC = ROOT / "evidence/pair_risk_forecast_diagnosis_20261103_v1.json"
OWN = (
    "src/us_quant/pair_risk_forecast_diagnosis_v1.py",
    "tests/test_pair_risk_forecast_diagnosis_v1.py",
    "tests/conftest.py",
)


def future_calendar(day: pd.Timestamp) -> pd.DatetimeIndex:
    if not is_month_end(day):
        raise QuantError("A risk forecast origin must be a genuine completed exchange month.")
    month = day.to_period("M") + 1
    return sessions(str(month.start_time.date()), str(month.end_time.date()))


def scalar_covariance(x, y):
    a, b = math.fsum(x) / len(x), math.fsum(y) / len(y)
    if all(v == x[0] for v in x) or all(v == y[0] for v in y):
        return 0.0
    return math.fsum((u - a) * (v - b) for u, v in zip(x, y, strict=True)) / (len(x) - 1)


def risk_case(net: np.ndarray, rf: np.ndarray, weights: np.ndarray, forecast: dict) -> dict:
    if (
        net.ndim != 2
        or net.shape[1] != 2
        or len(net) < 2
        or rf.shape != (len(net),)
        or weights.shape != (2,)
        or not np.isfinite(net).all()
        or not np.isfinite(rf).all()
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or abs(float(weights.sum()) - 1) > 1e-12
    ):
        raise QuantError(
            "Use exact complete future NET/RF pairs and original fixed source weights."
        )
    contributions = np.asarray(forecast["training_marginal_variance_contributions"], dtype=float)
    recorded_variance = math.fsum(contributions.tolist())
    if (
        contributions.shape != (8,)
        or not np.isfinite(contributions).all()
        or recorded_variance <= 0
    ):
        raise QuantError("A scored source pair requires its positive recorded training risk.")
    values = net - rf[:, None]
    portfolio = values @ weights
    x, y = values[:, 0].tolist(), values[:, 1].tolist()
    covariance = np.cov(values, rowvar=False, ddof=1)
    constant = [all(v == row[0] for v in row) for row in (x, y)]
    for i, yes in enumerate(constant):
        if yes:
            covariance[i, :] = 0.0
            covariance[:, i] = 0.0
    scalar = np.array(
        [
            [scalar_covariance(x, x), scalar_covariance(x, y)],
            [scalar_covariance(y, x), scalar_covariance(y, y)],
        ]
    )
    if not np.allclose(covariance, scalar, rtol=1e-10, atol=1e-18):
        raise QuantError("Independent source covariance does not match the fixed diagnostic.")
    if np.all(portfolio == portfolio[0]):
        return {
            "status": "complete_future_zero_portfolio_risk",
            "future_source_sessions": len(net),
            "recorded_training_variance": recorded_variance,
            "future_portfolio_variance": 0.0,
            "forecast_variance_ratio": None,
            "ERC_relative_imbalance": None,
            "zero_risk_is_safe_success": False,
            "variance_floor_or_source_normalization": False,
        }
    realized = float(np.var(portfolio, ddof=1))
    scalar_portfolio = [
        math.fsum(float(weights[i]) * float(row[i]) for i in range(2)) for row in values
    ]
    oracle = scalar_covariance(scalar_portfolio, scalar_portfolio)
    if realized <= 0 or oracle <= 0 or not math.isclose(realized, oracle, rel_tol=1e-10, abs_tol=0):
        raise QuantError("Independent strictly positive future variance agreement failed.")
    marginal = weights * (covariance @ weights)
    if not math.isclose(float(marginal.sum()), realized, rel_tol=1e-10, abs_tol=0):
        raise QuantError("Future marginal source risk does not sum to its diagnostic variance.")
    correlation = (
        None
        if any(constant)
        else float(covariance[0, 1] / math.sqrt(covariance[0, 0] * covariance[1, 1]))
    )
    return {
        "status": "complete_future_risk_evaluated",
        "future_source_sessions": len(net),
        "recorded_training_variance": recorded_variance,
        "forecast_variance_derived_by_new_scalar_sum_of_stored_contributions": True,
        "future_portfolio_variance": realized,
        "forecast_variance_ratio": realized / recorded_variance,
        "recorded_training_correlation": forecast["training_correlation"],
        "future_pair_correlation": correlation,
        "negative_recorded_correlation_became_nonnegative": (
            forecast["training_correlation"] < 0 and correlation >= 0
            if correlation is not None
            else None
        ),
        "future_marginal_variance_contributions": marginal.tolist(),
        "ERC_relative_imbalance": abs(float(marginal[0] - marginal[1])) / realized,
        "exact_constant_future_source_series": constant,
        "independent_scalar_variance_difference": abs(realized - oracle),
        "independent_scalar_covariance_maximum_difference": float(abs(covariance - scalar).max()),
        "independent_scalar_variance_relative_difference": abs(realized - oracle) / oracle,
        "new_diagnostic_covariance_not_recovered_original_memory": True,
        "statistical_source_mixture_is_tradable_main_account": False,
        "future_risk_is_bound_or_independent_forward_proof": False,
        "variance_floor_or_source_normalization": False,
    }


def diagnose(audits, source_net: pd.DataFrame, rf: pd.Series, before_case=None, retain=None):
    if (
        source_net.empty
        or not isinstance(source_net.index, pd.DatetimeIndex)
        or list(source_net.columns) != list(source.rules.EXPERT_IDS)
        or not source_net.index.equals(rf.index)
        or not source_net.index.equals(sessions(source_net.index[0], source_net.index[-1]))
        or not np.isfinite(source_net.to_numpy()).all()
        or not np.isfinite(rf).all()
    ):
        raise QuantError("Keep the full original common source-NET/RF assessment calendar.")
    decisions, cases = [], []
    for audit in audits:
        day = pd.Timestamp(audit["decision_session"])
        selection = audit["selection"]
        if not selection["scores_available"]:
            decisions.append(
                {
                    "decision_session": str(day.date()),
                    "status": "original_unscored_no_forecast",
                    "original_reason": selection["reason"],
                    "source_pairs_retained": 0,
                }
            )
            continue
        wanted = future_calendar(day)
        present = wanted.intersection(source_net.index)
        complete = wanted.equals(present)
        pairs = selection["pairs"]
        expected = {
            tuple(sorted((source.rules.EXPERT_IDS[i], source.rules.EXPERT_IDS[j])))
            for i, j in itertools.combinations(range(8), 2)
        }
        if len(pairs) != 28 or {tuple(p["pair"]) for p in pairs} != expected:
            raise QuantError("Retain all28original pairs,not selected-only forecasts.")
        decisions.append(
            {
                "decision_session": str(day.date()),
                "status": "future_month_complete" if complete else "future_month_incomplete",
                "required_future_sessions": len(wanted),
                "available_future_sessions": len(present),
                "required_end": str(wanted[-1].date()),
                "source_pairs_retained": 28,
            }
        )
        for pair in pairs:
            if before_case is not None:
                before_case()
            names = pair["pair"]
            indices = [source.rules.EXPERT_IDS.index(n) for n in names]
            row = {
                "decision_session": str(day.date()),
                "pair": names,
                "was_original_selected_pair": names == selection["selected_pair"],
                "required_future_start": str(wanted[0].date()),
                "required_future_end": str(wanted[-1].date()),
                "required_future_sessions": len(wanted),
                "available_future_sessions": len(present),
            }
            values = source_net.loc[present, names].to_numpy()
            full_weights = np.asarray(pair.get("expert_weights", [0.0] * 8), dtype=float)
            if "unscorable_reason" not in pair and (
                full_weights.shape != (8,)
                or not np.isfinite(full_weights).all()
                or any(full_weights[k] != 0 for k in range(8) if k not in indices)
            ):
                raise QuantError("A stored pair cannot hide a third source or changed coordinates.")
            weights = full_weights[indices]
            if retain is not None:
                retain(
                    {
                        "record": "new_diagnostic_input",
                        "case": row,
                        "stored_pair_record": pair,
                        "future_dates": present.strftime("%Y-%m-%d").tolist(),
                        "original_source_NET_binary64_hex": [
                            [float(v).hex() for v in r] for r in values
                        ],
                        "original_source_RF_binary64_hex": [
                            float(v).hex() for v in rf.loc[present]
                        ],
                    }
                )
            if not complete:
                result = {
                    **row,
                    "status": "future_month_incomplete_not_evaluated",
                    "partial_or_newer_snapshot_data_filled": False,
                    "forecast_variance_ratio": None,
                    "ERC_relative_imbalance": None,
                }
            elif "unscorable_reason" in pair:
                result = {
                    **row,
                    "status": "original_pair_unscorable_not_evaluated",
                    "original_reason": pair["unscorable_reason"],
                    "forecast_variance_ratio": None,
                    "ERC_relative_imbalance": None,
                }
            else:
                result = {**row, **risk_case(values, rf.loc[present].to_numpy(), weights, pair)}
            cases.append(result)
            if retain is not None:
                retain({"record": "new_diagnostic_resolved", "result": result})
    return decisions, cases


def aggregate(cases):
    result = {}
    for label, selected in (
        ("all_complete_source_pairs", None),
        ("original_selected_pairs", True),
        ("original_nonselected_pairs", False),
    ):
        rows = [
            r
            for r in cases
            if r["status"] == "complete_future_risk_evaluated"
            and (selected is None or r["was_original_selected_pair"] == selected)
        ]
        if not rows:
            result[label] = {"pairs": 0, "status": "no_complete_evaluable_pairs"}
            continue
        ratios = np.asarray([r["forecast_variance_ratio"] for r in rows])
        imbalances = np.asarray([r["ERC_relative_imbalance"] for r in rows])
        negative = [
            r
            for r in rows
            if r["recorded_training_correlation"] < 0 and r["future_pair_correlation"] is not None
        ]
        result[label] = {
            "pairs": len(rows),
            "variance_ratio_median": float(np.median(ratios)),
            "variance_ratio_p90": float(np.quantile(ratios, 0.9)),
            "variance_ratio_max": float(ratios.max()),
            "future_variance_above_recorded_count": int((ratios > 1).sum()),
            "relative_ERC_imbalance_median": float(np.median(imbalances)),
            "relative_ERC_imbalance_p90": float(np.quantile(imbalances, 0.9)),
            "relative_ERC_imbalance_max": float(imbalances.max()),
            "negative_correlation_origins": len(negative),
            "negative_to_nonnegative_correlation_count": sum(
                r["negative_recorded_correlation_became_nonnegative"] for r in negative
            ),
            "independent_sample_or_pvalue_or_risk_bound": False,
        }
    return result


def dependencies():
    return FunctionType(
        source.seven.admitted.import_template.dependencies.__code__,
        {**globals(), "ast": ast, "prior": source, "base": BASE},
    )()


def freeze():
    lifecycle.runtime_guard()
    if INTENT.exists():
        raise QuantError("Preserve the complete or partial diagnostic freeze.")
    with lifecycle.research_job():
        os.umask(0o077)
        if (
            file_digest(DESIGN)
            != "c0c7280136a82a5c5d2af143e75b3fa5698ea873702dc77bd2fd20d2c05c952a"
        ):
            raise QuantError("Preserve the actual original pre-outcome diagnostic scheme.")
        history = BASE.journal_events(BASE.JOURNAL)
        if (
            len(history) != 722
            or history[-1]["sha256"] != read_json(DESIGN)["current_journal_head"]
        ):
            raise QuantError("Freeze one diagnostic against the actual722-event source state.")
        _, _, frozen, _ = source.inputs()
        source.check_frozen(frozen)
        deps = dependencies()
        sources = source.unique_refs(
            [
                *frozen["code"],
                *frozen["policies"],
                *frozen["prior_evidence"],
                *frozen["validation_evidence"],
                *frozen["benchmark_artifacts"],
                *frozen["market"].values(),
                BASE.reference(source.FOLDER / "information-audits-v1.json"),
                BASE.reference(source.FOLDER / "completion-v1.json"),
                BASE.reference(source.INTENT),
                BASE.reference(source.PUBLIC),
                BASE.reference(DESIGN),
                BASE.reference(FOLDER / "actual-invocation-v1.json"),
                BASE.reference(FOLDER / "actual-resume-invocation-v1.json"),
                BASE.reference(FOLDER / "calendar-count-clarification-before-review-v1.json"),
            ]
        )
        xml = FOLDER / "offline-tests-v1.xml"
        suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
        proof = read_json(FOLDER / "validation-source-v1.json")
        if (
            not suites
            or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
            or proof["passed_tests"] != sum(int(s.get("tests", 0)) for s in suites)
            or proof["implementation_and_test_inputs"] != [BASE.reference(ROOT / p) for p in OWN]
            or proof["outputs"]
            != [BASE.reference(xml), BASE.reference(FOLDER / "offline-tests-v1.log")]
            or proof["exit_code"] != 0
        ):
            raise QuantError("Complete successful exact source-calendar/statistical oracle tests.")
        for row in sources:
            BASE.checked_reference(row)
        write_new_receipt(
            INTENT,
            {
                "frozen_at": utc_now(),
                "source_references": sources,
                "implementation_dependencies": [
                    *deps,
                    BASE.reference(ROOT / "PAIR_RISK_FORECAST_DIAGNOSIS_V1.md"),
                ],
                "validation_refs": [
                    BASE.reference(xml),
                    BASE.reference(FOLDER / "offline-tests-v1.log"),
                    BASE.reference(FOLDER / "validation-source-v1.json"),
                ],
                "package_versions": sorted(
                    (d.metadata["Name"], d.version) for d in distributions()
                ),
                "actual_new_diagnostic_values_computed": False,
                "new_financial_trials": 0,
                "new_rule_or_parameter_selected": False,
            },
        )
    return BASE.reference(INTENT)


def run():
    lifecycle.runtime_guard()
    if OUTPUT.exists() or (FOLDER / "attempt-v1.json").exists():
        raise QuantError("Never repeat a complete or partial fixed diagnostic identity.")
    intent = read_json(INTENT)
    for row in (
        intent["source_references"]
        + intent["implementation_dependencies"]
        + intent["validation_refs"]
    ):
        BASE.checked_reference(row)
    if intent["package_versions"] != [
        list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())
    ]:
        raise QuantError("The frozen diagnostic environment changed.")
    with lifecycle.research_job():
        os.umask(0o077)
        before = source.current_counts(BASE.journal_events(BASE.JOURNAL))
        write_new_receipt(
            FOLDER / "attempt-v1.json", {"started_at": utc_now(), "intent": BASE.reference(INTENT)}
        )
        try:
            _, experts, _, _ = source.inputs()
            observations = [e.observe(pd.Timestamp("2026-10-05"), 2512) for e in experts]
            rf = observations[0].risk_free
            if any(not o.risk_free.equals(rf) or len(o.net_returns) != 2512 for o in observations):
                raise QuantError("Reuse exact complete source shadow NET/RF,not new source views.")
            net = pd.DataFrame({o.identity.identifier: o.net_returns for o in observations})
            audits = read_json(source.FOLDER / "information-audits-v1.json")["decisions"]
            log = FOLDER / "actual-new-inputs-resolved-v1.jsonl"
            with log.open("x") as stream:

                def retain(row):
                    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())

                decisions, cases = diagnose(
                    audits, net, rf, before_case=lifecycle.runtime_guard, retain=retain
                )
            if len(decisions) != 121 or len(cases) != 114 * 28:
                raise QuantError("Preserve121origins/all28pairs for every mature source decision.")
            aggregates = aggregate(cases)
            statuses = {
                status: sum(r["status"] == status for r in cases)
                for status in sorted({r["status"] for r in cases})
            }
            result = {
                "completed_at": utc_now(),
                "intent": BASE.reference(INTENT),
                "decision_states": decisions,
                "new_diagnostic_cases": cases,
                "aggregates": aggregates,
                "pair_case_status_counts": statuses,
                "inputs_resolved": BASE.reference(log),
                "new_diagnostic_reconstruction_not_original_captured_future_risk": True,
                "source_scoring_target_or_account_pipeline_replayed": False,
                "statistical_source_mixture_is_funded_main_account": False,
                "partial_future_month_or_old_forward_missing_days_filled": False,
                "new_financial_trials_or_metrics_or_rules_selected": 0,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
            }
            if source.current_counts(BASE.journal_events(BASE.JOURNAL)) != before:
                raise QuantError("Risk diagnostics cannot change financial trial counts.")
            for row in intent["source_references"]:
                BASE.checked_reference(row)
            write_new_receipt(OUTPUT, result)
            summary = {
                "completed_at": utc_now(),
                "diagnostic": BASE.reference(OUTPUT),
                "original_decisions": len(decisions),
                "source_pair_records": len(cases),
                "full_future_origins": sum(
                    d["status"] == "future_month_complete" for d in decisions
                ),
                "incomplete_future_origins": sum(
                    d["status"] == "future_month_incomplete" for d in decisions
                ),
                "original_unscored_origins": sum(
                    d["status"] == "original_unscored_no_forecast" for d in decisions
                ),
                "aggregates": aggregates,
                "all_source_inputs_unchanged": True,
                "pair_case_status_counts": statuses,
                "actual_financial_counts_unchanged": before,
                "statistical_risk_not_main_PNL_or_new_strategy": True,
                "new_source_requests_or_financial_trials": 0,
                "new_policy_parameter_or_window_selected": False,
                "independent_samples_or_pvalues_or_bounds_claimed": False,
                "investment_objective_verified": False,
            }
            write_new_receipt(SUMMARY, summary)
            write_new_receipt(PUBLIC, summary)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            write_new_receipt(
                FOLDER / "failure-v1.json",
                {
                    "failed_at": utc_now(),
                    "error": str(error),
                    "partial_inputs_preserved": True,
                    "new_financial_trials": 0,
                },
            )
            raise
    return BASE.reference(SUMMARY)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One fixed cached source risk-forecast diagnostic,no trading."
    )
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    print(json.dumps(freeze() if args.action == "freeze" else run(), indent=2))
