from __future__ import annotations

import argparse
import json
import os
import sys
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import execution_gap_envelope_v1 as rules
from us_quant import ssd_confidence_regret_scope_v1 as prior
from us_quant.candidate_receipts import write_new_receipt
from us_quant.research_program import review_market
from us_quant.ssd_style_activity_financial_v1 import dependencies as _dependencies
from us_quant.storage import digest_json, read_json, utc_now, write_json

base, ROOT, lifecycle = prior.base, prior.ROOT, prior.lifecycle
reference, unique_refs = prior.reference, prior.unique_refs
FOLDER = base.REPORTS / "execution-gap-envelope-20261106-v1"
INTENT = FOLDER / "analysis-intent-v1.json"
OUTPUT = FOLDER / "diagnostic-v1.json"
PUBLIC = ROOT / "evidence/execution_gap_envelope_20261106_v1.json"
OWN = (
    "config/execution-gap-envelope-v1.json",
    "src/us_quant/execution_gap_envelope_v1.py",
    "src/us_quant/ssd_execution_gap_envelope_v1.py",
    "tests/test_execution_gap_envelope_v1.py",
    "EXECUTION_GAP_ENVELOPE_V1.md",
    "tests/conftest.py",
)


def dependencies():
    return FunctionType(
        _dependencies.__code__,
        {**_dependencies.__globals__, **globals()},
        argdefs=_dependencies.__defaults__,
    )()


counts = prior.counts


def progress(phase, **fields):
    actual = counts()
    rules.require(
        all(k not in fields or fields[k] == v for k, v in actual.items()),
        "A finite scenario reference cannot rewrite financial totals.",
    )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "execution_gap_envelope_" + phase,
            "study_id": read_json(rules.POLICY)["study_id"],
            "active_compute_job": None,
            **actual,
            **fields,
            "new_financial_trials": 0,
            "investment_objective_verified": False,
            "order_authority": False,
        },
    )


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    catalogue = read_json(base.checked_reference(policy["source_inputs"]))
    closed = read_json(base.checked_reference(catalogue["last_completed_cycle"]))
    rules.require(
        counts() == closed["actual_authoritative_counts"]
        and counts()["current_journal_event_count"] == 1490
        and set(catalogue["four_saved_contexts"]) == set(rules.PATHS)
        and catalogue["new_execution_gap_intervals_or_scenarios_computed"] is False
        and catalogue["prior_headroom_V1_repaired_retried_or_executed"] is False
        and catalogue["source_learner_or_target_or_account_or_prior_diagnosis_replayed"] is False,
        "Keep all original source states and terminal failures without running an old controller.",
    )
    refs = unique_refs(
        [
            *closed["exact_evidence_references"],
            policy["source_inputs"],
            *[
                v
                for v in catalogue.values()
                if isinstance(v, dict) and set(v) == {"path", "sha256"}
            ],
            *catalogue["source_market"].values(),
            *[
                v
                for c in catalogue["four_saved_contexts"].values()
                for v in c.values()
                if isinstance(v, dict) and set(v) == {"path", "sha256"}
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
        ],
    )
    for ref in refs:
        base.checked_reference(ref)
    return catalogue, refs


def freeze():
    lifecycle.runtime_guard()
    rules.require(
        not INTENT.exists() and not (FOLDER / "freeze-attempt-v1.json").exists(),
        "Preserve the first complete or partial execution-gap reference freeze.",
    )
    with lifecycle.research_job():
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, refs = source_inputs()
        validation = read_json(FOLDER / "validation-source-v1.json")
        suites = (
            ElementTree.parse(base.checked_reference(validation["test_outputs"][0]))
            .getroot()
            .findall(".//testsuite")
        )
        rules.require(
            validation["implementation_and_test_inputs"] == [reference(ROOT / p) for p in OWN]
            and validation["exit_code"] == 0
            and validation["synthetic_only"] is True
            and suites
            and validation["passed_tests"] == sum(int(s.get("tests", 0)) for s in suites)
            and not any(
                int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
            ),
            "Require successful prefill, cost-bound, exact-witness and known-price prefix checks.",
        )
        write_new_receipt(
            INTENT,
            {
                "frozen_at": utc_now(),
                "criteria": read_json(rules.POLICY),
                "source_references": refs,
                "complete_dependency_inputs": dependencies(),
                "implementation_and_test_inputs": [reference(ROOT / p) for p in OWN],
                "validation_evidence": [
                    *validation["test_outputs"],
                    reference(FOLDER / "validation-source-v1.json"),
                ],
                "python_version": sys.version,
                "package_versions": sorted(
                    (d.metadata["Name"], d.version) for d in distributions()
                ),
                "actual_authoritative_counts": counts(),
                "new_execution_gap_intervals_or_scenarios_computed": False,
                "new_financial_trials": 0,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="checkpoint_then_resolve_only_saved_decision_references")
    return reference(INTENT)


def analyze():
    lifecycle.runtime_guard()
    rules.require(
        not OUTPUT.exists() and not (FOLDER / "analysis-started-v1.json").exists(),
        "Do not replay a complete or partial actual reference inspection.",
    )
    with lifecycle.research_job():
        intent = read_json(INTENT)
        for ref in (
            intent["source_references"]
            + intent["complete_dependency_inputs"]
            + intent["implementation_and_test_inputs"]
            + intent["validation_evidence"]
        ):
            base.checked_reference(ref)
        rules.require(
            sys.version == intent["python_version"]
            and counts() == intent["actual_authoritative_counts"]
            and [list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())]
            == intent["package_versions"],
            "The frozen source, environment or financial history changed.",
        )
        catalogue, refs = source_inputs()
        rules.require(refs == intent["source_references"], "Keep every exact source reference.")
        start = read_json(FOLDER / "actual-analysis-start-v1.json")
        rules.require(
            start["actual_continuity_checkpoint_readback"]["saved_at"] < start["started_at"],
            "An actual saved checkpoint must precede new empirical intervals.",
        )
        write_new_receipt(
            FOLDER / "analysis-started-v1.json",
            {"started_at": utc_now(), "pid": os.getpid(), "analysis_intent": reference(INTENT)},
        )
        market = review_market({"market": catalogue["source_market"]}, ROOT)
        rules.require(
            tuple(market.close.columns) == tuple(rules.ASSETS),
            "The original fourteen-asset market cannot be relabeled.",
        )
        archive = rules.KnownPriceArchive(market.open, market.close)
        nominal = pd.read_csv(
            base.checked_reference(catalogue["source_nominal_targets"]),
            index_col=0,
            parse_dates=True,
            float_precision="round_trip",
        )
        rules.require(
            nominal.index.equals(market.close.index)
            and nominal.columns.equals(market.close.columns),
            "The original nominal source keeps the full market coordinates.",
        )
        active = nominal.dropna(how="all")
        directory = FOLDER / "actual-scenario-inputs-v1"
        directory.mkdir(mode=0o700)
        results, cases, contexts = [], [], {}
        trace = FOLDER / "actual-inputs-resolved-v1.jsonl"
        with trace.open("x") as stream:

            def retain(row):
                stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

            for key in rules.PATHS:
                context = catalogue["four_saved_contexts"][key]
                account, weights = [
                    pd.read_csv(
                        base.checked_reference(context[name]),
                        index_col=0,
                        parse_dates=True,
                        float_precision="round_trip",
                    )
                    for name in ("account.csv", "weights.csv")
                ]
                expected = rules.sessions(
                    "2016-10-06" if key.startswith("10") else "2021-10-06", "2026-10-05"
                )
                rules.require(
                    account.index.equals(expected)
                    and weights.index.equals(expected)
                    and tuple(weights.columns) == tuple(rules.ASSETS),
                    "No original formal return row or holding coordinate may be lost.",
                )
                decisions = read_json(base.checked_reference(context["decisions.json"]))[
                    "decisions"
                ]
                rules.require(
                    len(decisions) == intent["criteria"]["known_main_contexts"][key],
                    "Inspect every saved source decision,not selected drawdown dates.",
                )
                own = []
                previous_day = None
                for decision in decisions:
                    lifecycle.runtime_guard()
                    day = pd.Timestamp(decision["signal_session"])
                    initial = decision["initial_capital"]
                    rules.require(
                        initial is (previous_day is None) and (initial or day > previous_day),
                        "Use the actual initial/month-end source decision order.",
                    )
                    if initial:
                        values, cash, nav, peak = np.zeros(14), 10000.0, 10000.0, 10000.0
                    else:
                        nav, cash = (
                            float(account.loc[day, "equity"]),
                            float(account.loc[day, "cash"]),
                        )
                        peak = max(10000.0, float(account.loc[:day, "equity"].max()))
                        values = weights.loc[day].to_numpy() * nav
                    mark = rules.MainMark(day, values, cash, nav, peak)
                    known = active.loc[:day]
                    rules.require(
                        not known.empty, "A future nominal target cannot seed an earlier reference."
                    )
                    target = known.iloc[-1].to_numpy()
                    sample = archive.observe(day)
                    path = directory / f"{len(results):04d}.npz"
                    np.savez_compressed(
                        path,
                        starts=sample.starts.strftime("%Y-%m-%d").to_numpy(dtype="U10"),
                        wait_closes=sample.wait_closes.strftime("%Y-%m-%d").to_numpy(dtype="U10"),
                        ends=sample.ends.strftime("%Y-%m-%d").to_numpy(dtype="U10"),
                        prefill=sample.prefill,
                        opening=sample.opening,
                        fill_mark=sample.fill_mark,
                    )
                    path.chmod(0o600)
                    source = {
                        "path_key": key,
                        "as_of": str(day.date()),
                        "initial_source_capital": initial,
                        "source_decision": decision,
                        "known_nominal_session": str(known.index[-1].date()),
                        "asset_values_hex": [float(v).hex() for v in values],
                        "cash_hex": cash.hex(),
                        "NAV_hex": nav.hex(),
                        "peak_hex": peak.hex(),
                        "nominal_hex": [float(v).hex() for v in target],
                        "scenario_operands": reference(path),
                        "original_case_refs": context,
                        "original_market": catalogue["source_market"],
                        "hypothetical_full_fill_is_original_request": False,
                    }
                    retain({"phase": "reference_input", "source": source})
                    result = rules.resolve(target, mark, sample)
                    checked = rules.verify(target, mark, sample, result)
                    retain(
                        {"phase": "reference_resolved", "result": result, "independent": checked}
                    )
                    results.append({"path_key": key, "as_of": str(day.date()), **result, **checked})
                    cases.append((target.copy(), mark, reference(path)))
                    own.append(results[-1])
                    previous_day = day
                feasible = [row for row in own if row["status"] == "finite_interval_certified"]
                fractions = [
                    row["interval"]["largest_certified_nominal_fraction"] for row in feasible
                ]
                contexts[key] = {
                    "saved_main_decisions": len(own),
                    "conditional_reference_status_counts": {
                        status: sum(row["status"] == status for row in own)
                        for status in (
                            "outside_at_current_mark",
                            "unavoidable_prefill_reference_breach",
                            "uncertified_conservative_cost_bound",
                            "finite_interval_certified",
                            "no_finite_sufficient_interval",
                        )
                    },
                    "feasible_display_fraction_min_mean_max": (
                        [min(fractions), float(np.mean(fractions)), max(fractions)]
                        if fractions
                        else None
                    ),
                    "fully_known_historical_windows_examined": sum(
                        row["known_historical_scenarios"] for row in own
                    ),
                    "maximum_independent_scalar_money_error_USD": max(
                        row["maximum_independent_scalar_money_error_USD"] for row in own
                    ),
                    "finite_support_is_future_or_full_month_guarantee": False,
                    "actual_source_band_or_request_policy_reexecuted": False,
                }
            rules.require(len(results) == 364, "Keep every original saved main decision context.")
            prefixes = []
            for index in (0, 12, 120, 242, 363):
                target, mark, saved_sample = cases[index]
                with np.load(base.checked_reference(saved_sample), allow_pickle=False) as arrays:
                    sample = rules.StressSample(
                        mark.as_of,
                        pd.DatetimeIndex(arrays["starts"]),
                        pd.DatetimeIndex(arrays["wait_closes"]),
                        pd.DatetimeIndex(arrays["ends"]),
                        arrays["prefill"],
                        arrays["opening"],
                        arrays["fill_mark"],
                    )
                prefix = rules.KnownPriceArchive(
                    market.open.loc[: mark.as_of], market.close.loc[: mark.as_of]
                ).observe(mark.as_of)
                rules.require(
                    all(
                        np.array_equal(getattr(prefix, name), getattr(sample, name))
                        for name in ("prefill", "opening", "fill_mark")
                    ),
                    "Future prices changed a historical execution-gap scenario.",
                )
                earlier = rules.resolve(target, mark, prefix)
                original = {
                    k: v
                    for k, v in results[index].items()
                    if k
                    not in (
                        "path_key",
                        "maximum_independent_scalar_money_error_USD",
                        "exact_represented_interval_and_witness_verified",
                        "future_or_full_month_guarantee_verified",
                    )
                }
                rules.require(
                    digest_json(earlier) == digest_json(original),
                    "A later market suffix changed a known reference interval.",
                )
                prefixes.append(
                    {
                        "case_index": index,
                        "as_of": str(mark.as_of.date()),
                        "exact_known_price_prefix": True,
                    }
                )
        rules.require(
            counts() == intent["actual_authoritative_counts"],
            "A reference inspection cannot append a financial lifecycle event.",
        )
        write_new_receipt(
            OUTPUT,
            {
                "completed_at": utc_now(),
                "study_id": intent["criteria"]["study_id"],
                "analysis_intent": reference(INTENT),
                "source_complete_financial_result": catalogue["source_financial_public_result"],
                "four_original_contexts": contexts,
                "saved_main_contexts": len(results),
                "actual_input_resolution_records": reference(trace),
                "actual_input_resolution_record_count": 728,
                "known_price_prefixes": prefixes,
                "actual_authoritative_counts": counts(),
                "prior_headroom_V1_repaired_retried_or_executed": False,
                "source_learner_or_target_or_account_or_prior_diagnosis_replayed": False,
                "new_target_vectors_or_accounts": 0,
                "new_financial_trials": 0,
                "finite_reference_is_future_or_full_month_guarantee": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        progress(
            "actual_analysis_complete",
            private_result=reference(OUTPUT),
            analysis_intent=reference(INTENT),
            next_step="export_sanitized_reference_counts_once",
        )
    return reference(OUTPUT)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(OUTPUT)
        write_new_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                **{
                    k: v
                    for k, v in saved.items()
                    if k not in ("analysis_intent", "actual_input_resolution_records")
                },
                "raw_prices_holdings_targets_or_dated_case_rows_exported": False,
            },
        )
    return reference(PUBLIC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "analyze", "export"))
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "analyze": analyze, "export": export}[args.action]()))
