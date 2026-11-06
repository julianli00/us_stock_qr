from __future__ import annotations

import argparse
import json
import os
import sys
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

from us_quant import expert_opinion_clock_review_v1 as rules
from us_quant import ssd_protective_six_adaptive_hedge_v1 as pool
from us_quant.calendar import sessions
from us_quant.candidate_receipts import write_new_receipt
from us_quant.ssd_style_activity_financial_v1 import dependencies as _dependencies
from us_quant.storage import digest_json, read_json, utc_now, write_json

base, ROOT, lifecycle = pool.base, pool.ROOT, pool.lifecycle
reference, unique_refs = pool.reference, pool.unique_refs
FOLDER = base.REPORTS / "expert-opinion-clock-review-20261106-v1"
INTENT = FOLDER / "review-intent-v1.json"
OUTPUT = FOLDER / "review-v1.json"
PUBLIC = ROOT / "evidence/expert_opinion_clock_review_20261106_v1.json"
OWN = (
    "config/expert-opinion-clock-review-v1.json",
    "src/us_quant/expert_opinion_clock_review_v1.py",
    "src/us_quant/ssd_expert_opinion_clock_review_v1.py",
    "tests/test_expert_opinion_clock_review_v1.py",
    "EXPERT_OPINION_CLOCK_REVIEW_V1.md",
    "tests/conftest.py",
)


def dependencies():
    return FunctionType(
        _dependencies.__code__,
        {**_dependencies.__globals__, **globals()},
        argdefs=_dependencies.__defaults__,
    )()


def counts():
    return pool.current_counts(base.journal_events(base.JOURNAL))


def progress(phase, **fields):
    actual = counts()
    rules.require(
        all(k not in fields or fields[k] == v for k, v in actual.items()),
        "Opinion-clock review cannot alter financial counters.",
    )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "expert_opinion_clock_review_" + phase,
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
    source = read_json(base.checked_reference(policy["source_inputs"]))
    data, experts, _, refs, clocks, _ = pool.source_inputs()
    rules.require(
        source["expert_ids"] == list(pool.rules.EXPERT_IDS)
        and tuple(e.identity.identifier for e in experts) == pool.rules.EXPERT_IDS
        and source["source_clocks"] == clocks
        and source["declared_decision_information_clock"] == pool.rules.CLOCK
        and source["existing_adapter_classes"] == [type(e).__name__ for e in experts]
        and source["new_daily_clock_labels_or_financial_targets_computed"] is False
        and source["new_source_or_RF_or_event_Adapter"] is False,
        "Keep all original six source identities,admission contexts and existing Adapters.",
    )
    closed = read_json(base.checked_reference(source["last_completed_cycle"]))
    rules.require(
        closed["actual_authoritative_counts"] == counts(),
        "The actual financial predecessor cannot change during source-clock review.",
    )
    refs = unique_refs(
        [
            *refs,
            *closed["exact_evidence_references"],
            policy["source_inputs"],
            reference(ROOT / "src/us_quant/expert_initial_monthly_execution_v1.py"),
            reference(ROOT / "src/us_quant/legacy_feedback_expert_v1.py"),
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
    for ref in refs:
        base.checked_reference(ref)
    return data, experts, refs


def freeze():
    lifecycle.runtime_guard()
    rules.require(
        not INTENT.exists() and not (FOLDER / "freeze-attempt-v1.json").exists(),
        "Preserve the first complete or partial clock-review freeze.",
    )
    with lifecycle.research_job():
        actual = counts()
        rules.require(
            actual["current_journal_event_count"] == 1478,
            "Freeze against the actual unchanged J1478 predecessor.",
        )
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, _, refs = source_inputs()
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
            "Bind successful timestamp,maturity,signed-zero,scope and prefix checks.",
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
                "actual_authoritative_counts": actual,
                "new_daily_clock_labels_or_financial_targets_computed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="checkpoint_then_observe_existing_source_archives_once")
    return reference(INTENT)


def review():
    lifecycle.runtime_guard()
    rules.require(
        not OUTPUT.exists() and not (FOLDER / "review-started-v1.json").exists(),
        "Never replay the first complete or partial source-clock review.",
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
            "The frozen source environment or predecessor changed.",
        )
        data, experts, refs = source_inputs()
        rules.require(refs == intent["source_references"], "Keep every exact source clock input.")
        start = read_json(FOLDER / "actual-review-start-v1.json")
        rules.require(
            start["actual_continuity_checkpoint_readback"]["saved_at"] < start["started_at"],
            "A real saved preanalysis checkpoint must precede new source-clock labels.",
        )
        write_new_receipt(
            FOLDER / "review-started-v1.json",
            {"started_at": utc_now(), "pid": os.getpid(), "review_intent": reference(INTENT)},
        )
        calendar = sessions("2016-10-05", "2026-10-05")
        rules.require(len(calendar) == 2513, "Preserve every original source-observation date.")
        previous = {e.identity.identifier: None for e in experts}
        per_source = {e.identity.identifier: [] for e in experts}
        trace = FOLDER / "actual-inputs-resolved-v1.jsonl"
        with trace.open("x") as stream:
            for index, day in enumerate(calendar):
                lifecycle.runtime_guard()
                expected_returns = calendar[1 : index + 1][-126:]
                for expert in experts:
                    observed = expert.observe(day, 126)
                    rules.require(
                        observed.as_of == day
                        and observed.target is not None
                        and observed.target_known_at is not None
                        and tuple(observed.target.index) == tuple(rules.ASSETS)
                        and observed.net_returns.index.equals(expected_returns)
                        and observed.risk_free.index.equals(expected_returns)
                        and observed.risk_free.equals(data.risk_free.loc[expected_returns]),
                        "A missing,foreign,future or changed original source window cannot fill.",
                    )
                    identity = observed.identity.identifier
                    row = {
                        "expert_id": identity,
                        "mechanism_group": observed.identity.mechanism_group,
                        "introduced_at": observed.identity.introduced_at,
                        "as_of": str(day.date()),
                        "target_known_at": str(observed.target_known_at.date()),
                        "asset_order": list(observed.target.index),
                        "target_hex": [float(v).hex() for v in observed.target],
                        "assessment_RF_matches_original_view": True,
                        "source_window_exact_calendar": True,
                        "source_return_count": len(observed.net_returns),
                        "source_score_mature": observed.score_mature,
                        "historical_counterfactual": observed.historical_counterfactual,
                    }
                    stream.write(
                        json.dumps(
                            {"phase": "source_observation_input", "source": row},
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    stream.flush()
                    resolved = rules.inspect(row, previous[identity])
                    previous[identity] = row
                    per_source[identity].append(resolved)
                    stream.write(
                        json.dumps(
                            {"phase": "clock_resolved", "labels": resolved},
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    stream.flush()
                os.fsync(stream.fileno())
        observations = sum(len(rows) for rows in per_source.values())
        rules.require(observations == 15078, "Do not omit a source or a covered session.")
        prefixes = []
        for cut in (1, 21, 126, len(calendar) // 2, len(calendar)):
            lifecycle.runtime_guard()
            prior_rows = dict.fromkeys(previous)
            with trace.open() as stream:
                for _ in range(cut * 6):
                    source = json.loads(stream.readline())["source"]
                    saved = json.loads(stream.readline())["labels"]
                    identity = source["expert_id"]
                    check = rules.inspect(source, prior_rows[identity])
                    rules.require(
                        digest_json(check) == digest_json(saved),
                        "Future retained observations changed earlier clock labels.",
                    )
                    prior_rows[identity] = source
            prefixes.append(
                {
                    "sessions": cut,
                    "through": str(calendar[cut - 1].date()),
                    "observations": cut * 6,
                    "exact_retained_prefix": True,
                }
            )
        rules.require(
            counts() == intent["actual_authoritative_counts"],
            "Source-clock inspection cannot append a financial lifecycle event.",
        )
        write_new_receipt(
            OUTPUT,
            {
                "completed_at": utc_now(),
                "review_intent": reference(INTENT),
                "summary": rules.summarize(per_source),
                "exact_prefixes": prefixes,
                "actual_input_resolved_records": reference(trace),
                "actual_input_resolved_record_count": observations * 2,
                "actual_authoritative_counts": counts(),
                "new_financial_trials": 0,
                "source_model_target_shadow_account_BT_replayed": False,
                "new_source_or_event_Adapter": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        progress(
            "actual_review_complete",
            private_result=reference(OUTPUT),
            review_intent=reference(INTENT),
            next_step="export_sanitized_clock_counts_once",
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
                "study_id": read_json(rules.POLICY)["study_id"],
                **{
                    k: v
                    for k, v in saved.items()
                    if k not in ("review_intent", "actual_input_resolved_records")
                },
                "raw_prices_targets_returns_or_dated_source_rows_exported": False,
            },
        )
    return reference(PUBLIC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "review", "export"))
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "review": review, "export": export}[args.action]()))
