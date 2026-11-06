from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import pandas as pd

from us_quant import funded_meta_experts_v1 as rules
from us_quant import ssd_eight_expert_meta_v1 as shared
from us_quant.candidate_receipts import write_new_receipt
from us_quant.ssd_style_activity_financial_v1 import dependencies as _dependencies
from us_quant.storage import (
    digest_json,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
)

base, ROOT, lifecycle = shared.base, shared.ROOT, shared.lifecycle
reference, unique_refs = shared.reference, shared.unique_refs
FOLDER = base.REPORTS / "funded-meta-expert-eligibility-20261106-v1"
INTENT = FOLDER / "technical-implementation-freeze-v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC = ROOT / "evidence/funded_meta_expert_eligibility_20261106_v1.json"
OWN = (
    "config/funded-meta-expert-eligibility-v1.json",
    "src/us_quant/funded_meta_experts_v1.py",
    "src/us_quant/ssd_funded_meta_expert_eligibility_v1.py",
    "tests/test_funded_meta_experts_v1.py",
    "tests/conftest.py",
    "FUNDED_META_EXPERT_ELIGIBILITY_V1.md",
)


def dependencies():
    return FunctionType(
        _dependencies.__code__,
        {**_dependencies.__globals__, **globals()},
        argdefs=_dependencies.__defaults__,
    )()


def source_refs():
    records = read_json(base.checked_reference(read_json(rules.POLICY)["source_records"]))[
        "records"
    ]
    rules.require(
        tuple(r["id"] for r in records) == rules.IDS, "Review all four declared programs."
    )
    refs = [
        reference(FOLDER / name)
        for name in (
            "source-records-v1.json",
            "design-decision-v1.json",
            "actual-invocation-v1.json",
            "preservation-baseline-v1.json",
            "preflight-failure-v1.json",
            "capture-preflight-v1.py",
            "complete-source-preflight-v2.py",
        )
    ]
    for record in records:
        rules.check_record(record)
        refs.extend(
            record[k]
            for k in (
                "registration",
                "results",
                "source_cycle",
                "source_implementation_freeze",
                "initial_opinion_policy",
                "resolved_opinion_tape",
            )
        )
        packet = read_json(base.checked_reference(record["registration"]))
        entry = packet["entries"][0]
        frozen = entry["specification"]["frozen"]
        refs.extend(
            frozen["code"]
            + frozen["policies"]
            + frozen["prior_evidence"]
            + frozen["validation_evidence"]
            + frozen["benchmark_artifacts"]
            + frozen["volatility_sources"]
            + list(frozen["market"].values())
            + list(entry["target_proof"][k] for k in ("targets", "proof"))
        )
        for ref in record["completed_paths"].values():
            refs.append(ref)
            refs.extend(read_json(base.checked_reference(ref))["artifacts"])
    return records, unique_refs(refs)


def check_frozen():
    frozen = read_json(INTENT)
    for ref in (
        frozen["implementation_dependencies"]
        + frozen["source_references"]
        + frozen["validation_evidence"]
    ):
        base.checked_reference(ref)
    rules.require(
        frozen["implementation_and_test_inputs"] == [reference(ROOT / p) for p in OWN],
        "Preserve the exact first frozen source-scoped review Implementation.",
    )
    baseline = read_json(FOLDER / "preservation-baseline-v1.json")
    rules.require(
        sys.version == baseline["python_version"]
        and [list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())]
        == baseline["package_versions"],
        "The accepted full Python version and all package rows must remain exact.",
    )
    return frozen


def progress(phase, **fields):
    counts = shared.current_counts(base.journal_events(base.JOURNAL))
    rules.require(
        all(k not in fields or fields[k] == v for k, v in counts.items()),
        "A technical expert review cannot alter financial trial counts.",
    )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "funded_meta_expert_eligibility_" + phase,
            "study_id": read_json(rules.POLICY)["study_id"],
            "active_compute_job": None,
            **counts,
            **fields,
            "order_authority": False,
        },
    )


def freeze():
    lifecycle.runtime_guard()
    rules.require(
        not INTENT.exists() and not (FOLDER / "freeze-attempt-v1.json").exists(),
        "Preserve the first complete or partial technical freeze.",
    )
    with lifecycle.research_job():
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        policy = read_json(rules.POLICY)
        rules.validate(policy)
        baseline = read_json(FOLDER / "preservation-baseline-v1.json")
        rules.require(
            shared.current_counts(base.journal_events(base.JOURNAL)) == baseline["actual_counts"]
            and baseline["actual_counts"]["current_journal_event_count"] == 1297,
            "Freeze this review against its actual unchanged J1297 baseline.",
        )
        validation = read_json(FOLDER / "validation-source-v1.json")
        xml, _ = [base.checked_reference(ref) for ref in validation["test_outputs"]]
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
            "Bind passing real-Interface, source-clock, exact-RF, funding and prefix checks.",
        )
        records, refs = source_refs()
        for ref in refs:
            base.checked_reference(ref)
        deps = dependencies()
        write_new_receipt(
            INTENT,
            {
                "frozen_at": utc_now(),
                "policy": reference(rules.POLICY),
                "implementation_and_test_inputs": [reference(ROOT / p) for p in OWN],
                "implementation_dependencies": deps,
                "source_references": refs,
                "validation_evidence": [
                    *validation["test_outputs"],
                    reference(FOLDER / "validation-source-v1.json"),
                ],
                "source_actual_introduction_clocks": [
                    {"id": r["id"], "actual_source_registered_at": r["introduced_at"]}
                    for r in records
                ],
                "actual_authoritative_counts": baseline["actual_counts"],
                "new_observation_review_started": False,
                "new_financial_trials": 0,
                "global_pool_changed": False,
                "order_authority": False,
            },
        )
        progress("frozen", implementation_freeze=reference(INTENT), new_financial_trials=0)
    return reference(INTENT)


def observation_payload(observation):
    return {
        "id": observation.identity.identifier,
        "as_of": str(observation.as_of.date()),
        "target_known_at": (
            str(observation.target_known_at.date())
            if observation.target_known_at is not None
            else None
        ),
        "target": observation.target.tolist() if observation.target is not None else None,
        "return_dates": observation.net_returns.index.strftime("%Y-%m-%d").tolist(),
        "NET": observation.net_returns.tolist(),
        "RF": observation.risk_free.tolist(),
        "score_mature": observation.score_mature,
    }


def check_observations(expert, record, technical, retain):
    targets = rules.frame(technical["source_target_reference"])
    receipt = read_json(base.checked_reference(record["completed_paths"]["10y-base"]))
    shadow_ref = next(r for r in receipt["artifacts"] if r["path"].endswith("/account.csv"))
    shadow = rules.frame(shadow_ref)
    active = targets.dropna(how="all")
    native_dates = active.index.tolist()
    dates = pd.DatetimeIndex([rules.INITIAL, *shadow.index])
    hashes = {}
    for i, day in enumerate(dates):
        if i % 128 == 0:
            lifecycle.runtime_guard()
        observed = expert.observe(day, 126)
        position = bisect.bisect_right(native_dates, day) - 1
        known = native_dates[position] if position >= 0 else None
        expected = shadow.loc[shadow.index <= day].iloc[-126:]
        rules.require(
            observed.as_of == day
            and observed.identity.identifier == record["id"]
            and observed.target_known_at == known
            and observed.target.equals(active.loc[known])
            and observed.net_returns.equals(expected["return"])
            and observed.risk_free.equals(expected["risk_free"])
            and observed.score_mature is (len(expected) == 126),
            "Independent raw-source observation slices or opinion dates changed.",
        )
        context = expert.event_context(day)
        rules.require(
            context["native_policy_opinion_refreshed_today"] == (day in active.index)
            and context["capital_initialization_itself_is_policy_refresh"] is False,
            "Observation, source opinion refresh and funding initialization remain distinct.",
        )
        payload = observation_payload(observed)
        hashes[day] = digest_json(payload)
        retain(
            {
                "candidate_id": record["id"],
                "as_of": str(day.date()),
                "lookback": 126,
                "source_target_reference": technical["source_target_reference"],
                "source_shadow_reference": shadow_ref,
                "observed_payload_sha256": hashes[day],
                "observed_source_rows": len(expected),
                "event_context": context,
                "raw_source_slice_exact": True,
            }
        )
    prefixes = []
    for cutoff in (
        shadow.index[0],
        pd.Timestamp("2016-10-31"),
        shadow.index[125],
        pd.Timestamp("2021-10-05"),
        shadow.index[-1],
    ):
        lifecycle.runtime_guard()
        early = rules.ArchivedInitialMonthlyExpert(
            expert.identity, targets.loc[:cutoff], shadow.loc[:cutoff], tuple(rules.ASSETS)
        )
        for day in dates[dates <= cutoff]:
            rules.require(
                digest_json(observation_payload(early.observe(day, 126))) == hashes[day],
                "A later source row changed a previous observation or maturity state.",
            )
        prefixes.append(str(cutoff.date()))
    return {
        "all_formal_observations_plus_initial_checked": len(dates),
        "initial_empty_history_explicit": not expert.observe(rules.INITIAL, 126).score_mature,
        "exact_prefix_cutoffs": prefixes,
        "source_shadow_reference": shadow_ref,
        "five_year_funding_anchor_context": expert.event_context("2021-10-05"),
        "first_policy_opinion_context": expert.event_context(rules.INITIAL),
    }


def review():
    lifecycle.runtime_guard()
    rules.require(
        not RESULTS.exists() and not (FOLDER / "review-attempt-v1.json").exists(),
        "Preserve this first complete or partial actual review; do not replay it.",
    )
    with lifecycle.research_job():
        frozen = check_frozen()
        write_new_receipt(FOLDER / "review-attempt-v1.json", {"started_at": utc_now()})
        records = read_json(base.checked_reference(read_json(rules.POLICY)["source_records"]))[
            "records"
        ]
        results = []
        log = FOLDER / "actual-observations-v1.jsonl"
        with log.open("x") as stream:

            def retain(row):
                stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

            for record in records:
                lifecycle.runtime_guard()
                expert, technical = rules.load_expert(record, tuple(rules.ASSETS))
                observed = check_observations(expert, record, technical, retain)
                result = {
                    "reviewed_at": utc_now(),
                    "record": record,
                    "technical": technical,
                    "observation_checks": observed,
                    "status": "technically_admitted_scoped_initial_plus_monthly_not_selected",
                    "financial_status_preserved": technical["source_financial_verdict_preserved"],
                    "pool_membership_or_final_qualification": False,
                }
                path = FOLDER / "experts" / (record["id"] + ".json")
                if not path.parent.exists():
                    new_output_directory(path.parent)
                write_new_receipt(path, result)
                results.append(result)
        counts = shared.current_counts(base.journal_events(base.JOURNAL))
        rules.require(
            counts == frozen["actual_authoritative_counts"],
            "Technical reuse cannot append a financial lifecycle event.",
        )
        write_new_receipt(
            RESULTS,
            {
                "completed_at": utc_now(),
                "implementation_freeze": reference(INTENT),
                "experts": results,
                "actual_observation_tape": reference(log),
                "actual_authoritative_counts": counts,
                "source_family_count": 1,
                "new_financial_trials": 0,
                "new_source_requests": 0,
                "new_RF_or_nominal_target_or_assessment_views": 0,
                "source_generator_rank_fit_shadow_account_or_benchmark_replayed": False,
                "global_pool_changed": False,
                "pool_representative_selected": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        progress(
            "actual_review_complete",
            private_result=reference(RESULTS),
            implementation_freeze=reference(INTENT),
            new_financial_trials=0,
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
                "study_id": read_json(rules.POLICY)["study_id"],
                "completed_at": saved["completed_at"],
                "expert_reviews": [
                    {
                        "id": r["record"]["id"],
                        "mechanism_group": rules.GROUP,
                        "source_registered_at": r["record"]["introduced_at"],
                        "new_scoped_review_at": r["reviewed_at"],
                        "status": r["status"],
                        "source_financial_status_preserved": r["financial_status_preserved"],
                        "native_policy_opinions": r["technical"]["native_policy_opinions"],
                        "native_RF_exact_canonical": r["technical"]["native_RF_exact_canonical"],
                        "original_full_funded_paths": r["technical"]["original_full_funded_paths"],
                        "observation_checks": {
                            k: v
                            for k, v in r["observation_checks"].items()
                            if k != "source_shadow_reference"
                        },
                    }
                    for r in saved["experts"]
                ],
                "source_family_count": 1,
                "new_financial_trials": 0,
                "new_RF_or_nominal_target_or_assessment_views": 0,
                "all_financial_statuses_and_original_dates_preserved": True,
                "raw_prices_targets_returns_or_daily_holdings_exported": False,
                "global_pool_changed": False,
                "pool_representative_selected": False,
                "actual_authoritative_counts": saved["actual_authoritative_counts"],
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
    return reference(PUBLIC)


def main():
    parser = argparse.ArgumentParser(description="Scoped complete funded meta-expert eligibility.")
    parser.add_argument("action", choices=("freeze", "review", "export"))
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "review": review, "export": export}[args.action]()))


if __name__ == "__main__":
    main()
