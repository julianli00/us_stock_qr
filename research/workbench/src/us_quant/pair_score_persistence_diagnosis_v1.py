from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from collections import Counter
from importlib.metadata import distributions
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import pair_risk_forecast_diagnosis_v1 as risk
from us_quant import ssd_macro_expert_inclusion_v1 as saved
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.storage import digest_json, read_json, utc_now

ROOT, BASE = saved.ROOT, saved.base
FOLDER = BASE.REPORTS / "pair-score-persistence-diagnosis-20261104-v1"
DESIGN = FOLDER / "design-decision-v1.json"
INTENT = FOLDER / "analysis-intent-v1.json"
RESULT = FOLDER / "diagnostic-result-v1.json"
SUMMARY = FOLDER / "summary-v1.json"
PUBLIC = ROOT / "evidence/pair_score_persistence_diagnosis_20261104_v1.json"
DIAGNOSIS = "ssd_pair_score_persistence_diagnosis_20261104_v1"
OLD = BASE.REPORTS / "pair-risk-forecast-diagnosis-20261103-v1"
TRACE = OLD / "actual-new-inputs-resolved-v1.jsonl"
TRACE_SHA = "afedb2e7fca5234496386b5ec709a740871764262442add1b0f8e0002ade243c"
IDS = saved.template.rules.EXPERT_IDS
OWN = (
    "src/us_quant/pair_score_persistence_diagnosis_v1.py",
    "tests/test_pair_score_persistence_diagnosis_v1.py",
    "tests/conftest.py",
    "PAIR_SCORE_PERSISTENCE_DIAGNOSIS_V1.md",
)


def assess(before, resolved):
    case, old = before["case"], resolved["result"]
    pair, forecast = case["pair"], before["stored_pair_record"]
    if (
        before["record"] != "new_diagnostic_input"
        or resolved["record"] != "new_diagnostic_resolved"
        or any(old[key] != value for key, value in case.items())
        or len(pair) != 2
        or pair != sorted(set(pair))
        or not set(pair).issubset(IDS)
        or forecast["pair"] != pair
    ):
        raise QuantError("Keep the exact original pair, source flags and input/resolved identity.")
    wanted = risk.future_calendar(pd.Timestamp(case["decision_session"]))
    dates = pd.DatetimeIndex(before["future_dates"])
    if (
        case["required_future_start"] != str(wanted[0].date())
        or case["required_future_end"] != str(wanted[-1].date())
        or case["required_future_sessions"] != len(wanted)
        or case["available_future_sessions"] != len(dates)
        or not dates.equals(wanted[: len(dates)])
        or len(before["original_source_NET_binary64_hex"]) != len(dates)
        or len(before["original_source_RF_binary64_hex"]) != len(dates)
    ):
        raise QuantError("Only the complete original following exchange month may be assessed.")
    row = {
        **case,
        "source_input_resolved_sha256": digest_json([before, resolved]),
        "past_score_key": forecast.get("selection_score_key"),
        "future_reference_assessment": None,
    }
    if not dates.equals(wanted):
        if old["status"] != "future_month_incomplete_not_evaluated":
            raise QuantError("An incomplete future month cannot be an evaluated source outcome.")
        return {**row, "status": "future_month_incomplete_not_assessed"}
    if old["status"] in {
        "original_pair_unscorable_not_evaluated",
        "complete_future_zero_portfolio_risk",
    }:
        return {**row, "status": old["status"]}
    if old["status"] != "complete_future_risk_evaluated":
        raise QuantError("Unknown old risk status cannot become a valid future score.")
    weights = forecast["expert_weights"]
    if (
        len(weights) != 8
        or any(not math.isfinite(w) or w < 0 for w in weights)
        or abs(math.fsum(weights) - 1) > 1e-12
        or any(w != 0 for name, w in zip(IDS, weights, strict=True) if name not in pair)
        or forecast["cost_proxy_is_actual_main_account_cost"] is not False
        or forecast["selection_score"]
        != min(forecast["train_net_risk_score"], forecast["validation_net_risk_score"])
        or forecast["selection_score_key"] != round(forecast["selection_score"], 12)
    ):
        raise QuantError(
            "Use original fixed pair weights and recorded score keys, not a new choice."
        )
    indices = [IDS.index(name) for name in pair]
    chosen_weights = np.asarray([weights[i] for i in indices], dtype=float)
    net = np.asarray(
        [
            [float.fromhex(v) for v in values]
            for values in before["original_source_NET_binary64_hex"]
        ]
    )
    rf = np.asarray([float.fromhex(v) for v in before["original_source_RF_binary64_hex"]])
    variance = old["future_portfolio_variance"]
    proxy = forecast["estimated_incremental_switch_cost_fraction"]
    if (
        net.shape != (len(dates), 2)
        or not np.isfinite(net).all()
        or not np.isfinite(rf).all()
        or not math.isfinite(variance)
        or variance <= 0
        or not math.isfinite(proxy)
        or proxy < 0
    ):
        raise QuantError(
            "Use finite exact cached future NET/RF and positive stored risk, not a floor."
        )
    excess = net - rf[:, None]
    scalar_values = [
        math.fsum(float(w) * float(v) for w, v in zip(chosen_weights, values, strict=True))
        for values in excess
    ]
    mean = math.fsum(scalar_values) / len(scalar_values)
    numpy_mean = float(np.mean(excess @ chosen_weights))
    deviation = math.sqrt(variance)
    assessment = math.sqrt(252) * (mean - proxy / 21) / deviation
    independent = math.sqrt(252) * (numpy_mean - proxy / 21) / deviation
    error = abs(assessment - independent)
    if not math.isfinite(assessment) or error > 1e-8:
        raise QuantError("The fixed independent scalar assessment check failed.")
    return {
        **row,
        "status": "complete_future_reference_assessed",
        "future_reference_assessment": assessment,
        "future_source_NET_RF_mean": mean,
        "already_stored_future_portfolio_variance": variance,
        "recorded_incremental_switch_proxy": proxy,
        "independent_normalized_assessment_difference": error,
        "new_variance_or_covariance_estimate": False,
        "assessment_is_main_account_Sharpe_or_paid_fee": False,
    }


def tau_b(x, y):
    if len(x) != len(y) or len(x) < 2:
        raise QuantError("Ordinal comparison requires the complete paired assessment population.")
    concordant = discordant = x_ties = y_ties = both_ties = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        a = (x[i] > x[j]) - (x[i] < x[j])
        b = (y[i] > y[j]) - (y[i] < y[j])
        if a == 0 and b == 0:
            both_ties += 1
        elif a == 0:
            x_ties += 1
        elif b == 0:
            y_ties += 1
        elif a == b:
            concordant += 1
        else:
            discordant += 1
    denominator = math.sqrt((concordant + discordant + x_ties) * (concordant + discordant + y_ties))
    return {
        "tau_b": (concordant - discordant) / denominator if denominator else None,
        "status": "defined" if denominator else "undefined_tied_population",
        "concordant": concordant,
        "discordant": discordant,
        "past_only_ties": x_ties,
        "future_only_ties": y_ties,
        "both_ties": both_ties,
    }


def month(rows):
    expected = {tuple(sorted(pair)) for pair in itertools.combinations(IDS, 2)}
    if (
        len(rows) != 28
        or {tuple(row["pair"]) for row in rows} != expected
        or len({row["decision_session"] for row in rows}) != 1
        or sum(row["was_original_selected_pair"] for row in rows) != 1
    ):
        raise QuantError("Retain all28 distinct original pairs and the one actual selected flag.")
    result = {
        "decision_session": rows[0]["decision_session"],
        "pair_status_counts": dict(Counter(r["status"] for r in rows)),
        "evaluable_pairs": 0,
        "selected_future_midrank_percentile": None,
        "tau": None,
    }
    if all(r["status"] == "future_month_incomplete_not_assessed" for r in rows):
        return {**result, "status": "future_month_incomplete_not_ranked"}
    if any(r["status"] == "future_month_incomplete_not_assessed" for r in rows):
        raise QuantError(
            "A common following-month clock cannot be complete for selected pairs only."
        )
    evaluated = [r for r in rows if r["status"] == "complete_future_reference_assessed"]
    selected = next(r for r in rows if r["was_original_selected_pair"])
    result["evaluable_pairs"] = len(evaluated)
    if len(evaluated) >= 2:
        result["tau"] = tau_b(
            [r["past_score_key"] for r in evaluated],
            [r["future_reference_assessment"] for r in evaluated],
        )
    if selected["future_reference_assessment"] is None:
        return {**result, "status": "selected_future_assessment_undefined"}
    value = selected["future_reference_assessment"]
    result["selected_future_midrank_percentile"] = (
        sum(r["future_reference_assessment"] < value for r in evaluated)
        + 0.5 * sum(r["future_reference_assessment"] == value for r in evaluated)
    ) / len(evaluated)
    return {**result, "status": "complete_month_original_choice_assessed"}


def read_pairs(path: Path):
    with path.open() as stream:
        while line := stream.readline():
            following = stream.readline()
            if not following:
                raise QuantError("Keep a truncated old diagnostic stream as a failure.")
            try:
                before, resolved = json.loads(line), json.loads(following)
            except json.JSONDecodeError as error:
                raise QuantError(
                    "Malformed retained source data cannot become an empty outcome."
                ) from error
            yield before, resolved


def _references():
    if BASE.reference(TRACE)["sha256"] != TRACE_SHA:
        raise QuantError("The complete already-saved future pair assessment inputs changed.")
    old = read_json(OLD / "analysis-intent-v1.json")
    current = read_json(
        BASE.REPORTS
        / "legacy-price-forecast-expert-eligibility-20261104-v1/eligibility-intent-v1.json"
    )
    source = saved.unique_refs(
        [
            *old["source_references"],
            *(
                BASE.reference(p)
                for p in (
                    DESIGN,
                    FOLDER / "actual-invocation-v1.json",
                    TRACE,
                    OLD / "analysis-intent-v1.json",
                    OLD / "completion-v1.json",
                    OLD / "diagnostic-v1.json",
                    OLD / "summary-v1.json",
                    OLD / "calendar-count-clarification-before-review-v1.json",
                    BASE.REPORTS
                    / "six-expert-objective-alignment-20261030-v1/analysis-design-v1.json",
                    BASE.REPORTS / "six-expert-objective-alignment-20261030-v1/summary-v2.json",
                    BASE.REPORTS
                    / "expert-contest-credit-diagnosis-20261030-v1/design-decision-v1.json",
                )
            ),
        ]
    )
    dependencies = saved.unique_refs(
        [
            *old["implementation_dependencies"],
            *current["implementation_dependencies"],
            *(BASE.reference(ROOT / p) for p in OWN),
        ]
    )
    return source, dependencies


def freeze():
    with lifecycle.research_job():
        if INTENT.exists():
            raise QuantError("Do not refreeze the complete or partial fixed diagnostic identity.")
        os.umask(0o077)
        design = read_json(DESIGN)
        if (
            design["diagnosis_id"] != DIAGNOSIS
            or design["actual_continuation_event"] != 150260
            or design["actual_new_mean_or_ordinal_values_computed"] is not False
        ):
            raise QuantError("Keep the original single ordinal-persistence interpretation.")
        sources, dependencies = _references()
        xml, log = FOLDER / "offline-tests-v1.xml", FOLDER / "offline-tests-v1.log"
        proof = read_json(FOLDER / "validation-source-v1.json")
        suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
        if (
            not suites
            or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
            or proof["passed_tests"] != sum(int(s.get("tests", 0)) for s in suites)
            or proof["implementation_and_test_inputs"] != [BASE.reference(ROOT / p) for p in OWN]
            or proof["outputs"] != [BASE.reference(xml), BASE.reference(log)]
            or proof["exit_code"] != 0
        ):
            raise QuantError(
                "Bind the exact mean, full-month, tie, censoring and no-pipeline checks."
            )
        for ref in sources + dependencies:
            BASE.checked_reference(ref)
        counts = saved.current_counts(BASE.journal_events(BASE.JOURNAL))
        if counts["current_journal_event_count"] != 852:
            raise QuantError("Freeze against the actual closed852-event financial baseline.")
        write_new_receipt(
            INTENT,
            {
                "frozen_at": utc_now(),
                "diagnosis_id": DIAGNOSIS,
                "source_references": sources,
                "implementation_dependencies": dependencies,
                "validation_evidence": [
                    BASE.reference(xml),
                    BASE.reference(log),
                    BASE.reference(FOLDER / "validation-source-v1.json"),
                ],
                "package_versions": sorted(
                    (d.metadata["Name"], d.version) for d in distributions()
                ),
                "financial_counts": counts,
                "actual_new_mean_or_ordinal_values_computed": False,
                "maximum_actual_stream_passes": 1,
                "new_financial_trials": 0,
            },
        )
    return BASE.reference(INTENT)


def aggregate(months):
    complete = [m for m in months if m["status"] != "future_month_incomplete_not_ranked"]
    ranks = [
        m["selected_future_midrank_percentile"]
        for m in complete
        if m["selected_future_midrank_percentile"] is not None
    ]
    values = [m["tau"]["tau_b"] for m in complete if m["tau"] and m["tau"]["tau_b"] is not None]
    return {
        "mature_past_origins_retained": len(months),
        "complete_future_origins": len(complete),
        "censored_origins_retained": len(months) - len(complete),
        "future_selected_rank_defined_origins": len(ranks),
        "selected_future_midrank_median": float(np.median(ranks)) if ranks else None,
        "selected_future_midrank_mean": math.fsum(ranks) / len(ranks) if ranks else None,
        "selected_future_midrank_below_half_count": sum(v < 0.5 for v in ranks),
        "selected_undefined_origins_retained": len(complete) - len(ranks),
        "tau_defined_origins": len(values),
        "tau_median": float(np.median(values)) if values else None,
        "tau_mean": math.fsum(values) / len(values) if values else None,
        "tau_positive_origins": sum(v > 0 for v in values),
        "tau_negative_origins": sum(v < 0 for v in values),
        "tau_exact_zero_origins": sum(v == 0 for v in values),
        "tau_undefined_origins_retained": len(complete) - len(values),
        "pair_status_counts": dict(
            sum((Counter(m["pair_status_counts"]) for m in months), Counter())
        ),
    }


def public_report(result):
    return {
        "schema_version": 1,
        "diagnosis_id": DIAGNOSIS,
        "completed_at": result["completed_at"],
        "analysis_intent_sha256": result["intent"]["sha256"],
        "aggregates": result["aggregates"],
        "original_decision_states_preserved": 121,
        "original_unscored_origins_preserved": 7,
        "original_pair_records_preserved": 3192,
        "maximum_independent_scalar_assessment_difference": result[
            "maximum_independent_scalar_error"
        ],
        "ordinal_interpretation_introduced_by_this_diagnosis": True,
        "original_score_claimed_calibrated_forecast": False,
        "new_variance_covariance_chooser_target_or_account_pipeline": False,
        "reference_assessment_is_main_Sharpe_or_paid_fee": False,
        "censored_source_data_filled": False,
        "independent_samples_significance_or_main_PNL_attribution_claimed": False,
        "post_outcome_adaptive_selection_bias_preserved": True,
        "actual_financial_counts_unchanged": result["financial_counts_unchanged"],
        "new_financial_trials": 0,
        "new_source_requests": 0,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }


def run():
    lifecycle.runtime_guard()
    if RESULT.exists() or (FOLDER / "attempt-v1.json").exists():
        raise QuantError("Keep the first complete or partial diagnostic; never repair and retry.")
    intent = read_json(INTENT)
    if (
        intent["diagnosis_id"] != DIAGNOSIS
        or intent["actual_new_mean_or_ordinal_values_computed"] is not False
        or intent["maximum_actual_stream_passes"] != 1
        or intent["package_versions"]
        != [list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())]
    ):
        raise QuantError("The exact diagnostic identity or environment changed.")
    for ref in (
        intent["source_references"]
        + intent["implementation_dependencies"]
        + intent["validation_evidence"]
    ):
        BASE.checked_reference(ref)
    with lifecycle.research_job():
        os.umask(0o077)
        counts = saved.current_counts(BASE.journal_events(BASE.JOURNAL))
        if counts != intent["financial_counts"]:
            raise QuantError("The fixed diagnosis cannot hide a new financial event.")
        write_new_receipt(
            FOLDER / "attempt-v1.json",
            {
                "started_at": utc_now(),
                "pid": os.getpid(),
                "intent": BASE.reference(INTENT),
            },
        )
        months, pending, cases, maximum = [], [], 0, 0.0
        trace = FOLDER / "new-assessment-records-v1.jsonl"
        try:
            original = read_json(OLD / "diagnostic-v1.json")
            states = original["decision_states"]
            if (
                len(states) != 121
                or sum(s["status"] == "original_unscored_no_forecast" for s in states) != 7
            ):
                raise QuantError("Keep all121original states, including seven unscored origins.")
            with trace.open("x") as stream:
                for before, resolved in read_pairs(TRACE):
                    lifecycle.runtime_guard()
                    if cases >= len(original["new_diagnostic_cases"]) or (
                        resolved["result"] != original["new_diagnostic_cases"][cases]
                    ):
                        raise QuantError("The preserved risk result and raw stream disagree.")
                    row = assess(before, resolved)
                    maximum = max(
                        maximum, row.get("independent_normalized_assessment_difference", 0.0)
                    )
                    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                    stream.flush()
                    pending.append(row)
                    cases += 1
                    if len(pending) == 28:
                        months.append(month(pending))
                        pending = []
            if pending or cases != 3192 or len(months) != 114:
                raise QuantError("Do not shorten the28pair-by114origin source population.")
            summary = aggregate(months)
            if (
                summary["complete_future_origins"] != 113
                or summary["censored_origins_retained"] != 1
            ):
                raise QuantError("Keep113fullfuturemonths and the one unassessed final month.")
            if saved.current_counts(BASE.journal_events(BASE.JOURNAL)) != counts:
                raise QuantError("This diagnostic cannot change financial trial counts.")
            for ref in intent["source_references"] + intent["implementation_dependencies"]:
                BASE.checked_reference(ref)
            result = {
                "completed_at": utc_now(),
                "intent": BASE.reference(INTENT),
                "source_decision_states": states,
                "monthly_ordinal_assessments": months,
                "derived_records": BASE.reference(trace),
                "aggregates": summary,
                "maximum_independent_scalar_error": maximum,
                "financial_counts_unchanged": counts,
                "all_frozen_source_and_code_bytes_unchanged": True,
            }
            write_new_receipt(RESULT, result)
            write_new_receipt(SUMMARY, public_report(result))
            write_new_receipt(PUBLIC, public_report(result))
        except (QuantError, OSError, ValueError, KeyError, TypeError, IndexError) as error:
            write_new_receipt(
                FOLDER / "failure-v1.json",
                {
                    "failed_at": utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "complete_cases_retained": cases,
                    "complete_months_retained": len(months),
                    "partial_trace_preserved": True,
                    "new_financial_trials": 0,
                },
            )
            raise
    return BASE.reference(SUMMARY)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One cached pair-score ordinal persistence diagnosis."
    )
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    print(json.dumps(freeze() if args.action == "freeze" else run(), indent=2))
