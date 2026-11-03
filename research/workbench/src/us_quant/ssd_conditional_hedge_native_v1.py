from __future__ import annotations

import argparse
import ast
import json
import os
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import conditional_hedge_expert_v1 as law
from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import ssd_expert_target_consensus_v1 as template
from us_quant.calendar import previous_session
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.research_program import review_market
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.storage import (
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)
from us_quant.strategy_experts_v1 import load_expert

base, ROOT, source = template.base, template.ROOT, template.source
POLICY = ROOT / "config/conditional-hedge-native-books-v1.json"
FOLDER = base.REPORTS / "conditional-hedge-expert-20261103-v1"
PRIMARY = FOLDER / "primary-implementation-freeze-v1.json"
INTENT = FOLDER / "native-financial-freeze-v1.json"
PREPARED = FOLDER / "native-prepared-targets-v1.json"
PACKET = ROOT / "evidence/conditional_hedge_expert_20261103_registration_v1.json"
RESULTS = FOLDER / "native-results-v1.json"
PUBLIC = ROOT / "evidence/conditional_hedge_expert_20261103_results_v1.json"
SOURCE_RESULTS = FOLDER / "source-result-contract-v1.json"
ADMISSION = FOLDER / "technical-admission-v1.json"
PUBLIC_ADMISSION = ROOT / "evidence/conditional_hedge_expert_20261103_admission_v1.json"
PARENT = (
    base.REPORTS / "expert-target-consensus-20261103-v1/financial-implementation-freeze-v1.json"
)
ID = law.ID
CANDIDATE = {"id": ID, "construction": "complete_monthly_conditional_hedge_policy"}
OWN = (
    "src/us_quant/ssd_conditional_hedge_native_v1.py",
    "tests/test_ssd_conditional_hedge_native_v1.py",
    "src/us_quant/conditional_hedge_expert_v1.py",
    "tests/test_conditional_hedge_expert_v1.py",
    "src/us_quant/november_research_lifecycle_v1.py",
    "tests/test_expert_initial_monthly_execution_v1.py",
    "tests/conftest.py",
    "config/conditional-hedge-expert-v1.json",
    "config/conditional-hedge-native-books-v1.json",
)
reference, unique_refs = base.reference, source.unique_refs


def rebound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


def dependencies():
    walker = source.admitted.import_template.dependencies
    return FunctionType(
        walker.__code__, {**globals(), "ast": ast, "prior": source}, argdefs=walker.__defaults__
    )()


def events(kind):
    return rebound(template.events)(kind)


def progress(phase, **fields):
    counts = current_counts(base.journal_events(base.JOURNAL))
    if any(key in fields and fields[key] != value for key, value in counts.items()):
        raise QuantError("Native-source progress cannot override actual trial counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "conditional_hedge_native_" + phase,
            "study_id": read_json(POLICY)["study_id"],
            "active_compute_job": None,
            **fields,
            **counts,
        },
    )


def check_primary():
    if file_digest(PRIMARY) != "e815f5bf767157d25a48f78712d94fa5378814445bfe3a8aa7ba5edfbe856ced":
        raise QuantError("The completed primary-law freeze cannot change.")
    for row in read_json(PRIMARY)["implementation_and_test_inputs"]:
        base.checked_reference(row)
    law.validate(read_json(law.POLICY))
    policy = read_json(POLICY)
    if (
        policy["candidates"] != [CANDIDATE]
        or policy["primary_law"] != reference(law.POLICY)
        or policy["primary_freeze_sha256"] != file_digest(PRIMARY)
        or policy["prior_journal_sequence"] != 687
        or policy["prior_journal_head_sha256"]
        != "5befe4cab00c6d5524632e73a9245a71f18713a603fb67102ee47cdfc63bc831"
        or policy["order_authority"] is not False
    ):
        raise QuantError("The native declaration may not change law, history or authority.")


def check_frozen(frozen):
    base.check_frozen_files(frozen)
    for row in frozen["prior_evidence"] + frozen["validation_evidence"]:
        base.checked_reference(row)


def validation_refs():
    xml = FOLDER / "offline-native-tests-v1.xml"
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    if not suites or any(
        int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
    ):
        raise QuantError("Complete successful native lifecycle and accounting checks first.")
    proof = read_json(FOLDER / "native-validation-source-v1.json")
    if (
        proof["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
        or proof["passed_tests"] != sum(int(s.get("tests", 0)) for s in suites)
        or proof["exit_code"] != 0
        or proof["test_outputs"]
        != [reference(xml), reference(FOLDER / "offline-native-tests-v1.log")]
    ):
        raise QuantError(
            "The successful tests do not bind these actual native implementation bytes."
        )
    return [
        reference(xml),
        reference(FOLDER / "offline-native-tests-v1.log"),
        reference(FOLDER / "native-validation-source-v1.json"),
    ]


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "native-freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the complete or partial first native freeze.")
    with lifecycle.research_job():
        check_primary()
        write_new_receipt(FOLDER / "native-freeze-attempt-v1.json", {"started_at": utc_now()})
        history = base.journal_events(base.JOURNAL)
        if (
            len(history) != 687
            or history[-1]["sha256"] != read_json(POLICY)["prior_journal_head_sha256"]
        ):
            raise QuantError("The native-source experiment needs its exact pre-outcome baseline.")
        parent = read_json(PARENT)["frozen"]
        check_frozen(parent)
        data = review_market({"market": parent["market"]}, ROOT)
        if tuple(data.close.columns) != law.UNIVERSE:
            raise QuantError("Use the unchanged original complete fourteen-asset market.")
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": read_json(POLICY)["study_id"],
            "policy": reference(POLICY),
            "primary_signal_policy": reference(law.POLICY),
            "expert_pool_definition_role": "one_new_complete_source_policy_not_pool_expansion",
            "construction": "complete_monthly_conditional_cross_asset_hedge_policy",
            "code": unique_refs(
                [*parent["code"], *[r for r in deps if r["path"].startswith(("src/", "tests/"))]]
            ),
            "policies": unique_refs(
                [*parent["policies"], *[r for r in deps if r["path"].startswith("config/")]]
            ),
            "prior_evidence": unique_refs(
                [
                    *parent["prior_evidence"],
                    reference(PRIMARY),
                    reference(PARENT),
                    reference(FOLDER / "design-decision-v1.json"),
                    reference(FOLDER / "actual-invocation-v1.json"),
                ]
            ),
            "validation_evidence": unique_refs(
                [*parent["validation_evidence"], *validation_refs()]
            ),
            "new_economic_factor_definitions": 0,
            "new_source_requests": 0,
            "new_shadow_account_runs": 0,
            "original_return_windows_extended": False,
        }
        check_frozen(frozen)
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "frozen": frozen,
                "complete_dependency_inputs": deps,
                "package_versions": sorted(
                    (d.metadata["Name"], d.version) for d in distributions()
                ),
                "source_price_sessions": len(data.close),
                "new_real_targets_or_accounts_computed": False,
                "registered": False,
                "pool_changed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="register_before_real_targets")
    return reference(INTENT)


def inputs():
    lifecycle.runtime_guard()
    check_primary()
    intent = read_json(INTENT)
    for row in intent["complete_dependency_inputs"]:
        base.checked_reference(row)
    if intent["package_versions"] != [
        list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())
    ]:
        raise QuantError("The isolated source-book package environment changed.")
    frozen = intent["frozen"]
    check_frozen(frozen)
    frozen = {
        **frozen,
        "prior_evidence": unique_refs([*frozen["prior_evidence"], reference(INTENT)]),
    }
    data = review_market({"market": frozen["market"]}, ROOT)
    benchmarks = [
        row
        for row in frozen["benchmark_artifacts"]
        if row["path"].endswith("/benchmarks-20261027-v1/results.json")
    ]
    if len(benchmarks) != 1:
        raise QuantError("Reuse the uniquely frozen existing dual benchmark, never rerun it.")
    return data, None, frozen, read_json(base.checked_reference(benchmarks[0]))


def verify_targets(close, target, audits, records):
    expected_dates = pd.DatetimeIndex(
        [day for i, day in enumerate(close.index) if i >= 252 and law.is_month_end(day)]
    )
    if (
        not target.index.equals(close.index)
        or not target.columns.equals(close.columns)
        or not target.dropna(how="all").index.equals(expected_dates)
        or len(audits) != len(expected_dates)
        or len(records) != 2 * len(audits)
    ):
        raise QuantError("Native source recommendations must retain their whole genuine lattice.")
    maximum = 0.0
    for i, day in enumerate(expected_dates):
        before, after = records[2 * i : 2 * i + 2]
        if (
            before["record"] != "decision_input"
            or after["record"] != "decision_resolved"
            or before["decision_session"] != str(day.date())
            or before["information_clock"] != law.CLOCK
            or after["audit"] != audits[i]
            or after["complete_target"] != target.loc[day].tolist()
            or after["asset_order"] != list(close.columns)
        ):
            raise QuantError("Retained inputs must precede and match every actual native decision.")
        position = close.index.get_loc(day)
        window = close.iloc[position - 63 : position + 1].loc[:, ["SPY", *law.DEFENSE]]
        if (
            before["window_dates"] != window.index.strftime("%Y-%m-%d").tolist()
            or before["used_source_prices_binary64_hex"]
            != {s: [float(v).hex() for v in window[s]] for s in window.columns}
            or before["SPY_year_prior_source_date"] != str(close.index[position - 252].date())
            or before["SPY_year_prior_price_binary64_hex"]
            != float(close["SPY"].iloc[position - 252]).hex()
        ):
            raise QuantError("Source prices or availability changed before native target proof.")
        oracle, selected = law.independent_decision(close, position)
        if selected != audits[i]["selected_assets"]:
            raise QuantError("The independent integer covariance-sign decision disagrees.")
        maximum = max(maximum, float(abs(oracle - target.loc[day]).max()))
    if maximum > 1e-12:
        raise QuantError("Independent source target composition differs.")
    return {
        "genuine_month_end_decisions": len(audits),
        "retained_input_resolved_records": len(records),
        "maximum_independent_integer_target_difference": maximum,
    }


def prepare(data, unused, frozen):
    if PREPARED.exists():
        old = read_json(PREPARED)
        if old["frozen_specification_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting prior native preparation.")
        for row in old["entries"][ID].values():
            base.checked_reference(row)
        return old["entries"][ID]
    if (FOLDER / "native-preparation-attempt-v1.json").exists():
        raise QuantError("An incomplete source-target preparation is not automatically repeated.")
    if len(events("registered")) != 1 or events("registered")[0]["body"] != {
        "candidate": CANDIDATE,
        "frozen": frozen,
    }:
        raise QuantError("Actual native registration must precede all real source targets.")
    write_new_receipt(
        FOLDER / "native-preparation-attempt-v1.json",
        {"started_at": utc_now(), "frozen_sha256": digest_json(frozen)},
    )
    log = FOLDER / "native-actual-inputs-resolved-v1.jsonl"
    with log.open("x") as stream:

        def retain(row):
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        targets, audits = law.build_targets(
            data.close,
            read_json(law.POLICY),
            before_decision=lifecycle.runtime_guard,
            retain=retain,
        )
    records = [json.loads(line) for line in log.read_text().splitlines()]
    verification = verify_targets(data.close, targets, audits, records)
    prefixes = []
    for index in sorted({0, 6, 12, len(audits) // 2, len(audits) - 1}):
        day = pd.Timestamp(audits[index]["decision_session"])
        early, earlier = law.build_targets(
            data.close.loc[:day], read_json(law.POLICY), before_decision=lifecycle.runtime_guard
        )
        if not early.equals(targets.loc[:day]) or earlier != audits[: index + 1]:
            raise QuantError("Future inputs changed a native historical recommendation.")
        prefixes.append(str(day.date()))
    coverage = []
    active = targets.dropna(how="all")
    for window in read_json(base.PROTOCOL)["windows"]:
        anchor = previous_session(window["start"])
        known = active.loc[:anchor].tail(1)
        rows = pd.concat([known, active.loc[anchor + pd.Timedelta(days=1) : window["end"]]])
        factors = [symbol for symbol in law.CORE if (rows[symbol] > 1e-10).any()]
        if known.empty or len(factors) < 3:
            raise QuantError(
                "Both unchanged funded horizons need defined anchors and genuine factors."
            )
        coverage.append(
            {
                "years": window["years"],
                "actual_factor_symbols": factors,
                "known_native_recommendation_at_anchor": str(known.index[0].date()),
            }
        )
    directory = FOLDER / "native-targets"
    new_output_directory(directory)
    write_text_atomic(directory / "targets.csv", targets.to_csv(float_format="%.17g"))
    write_new_receipt(FOLDER / "native-information-audits-v1.json", {"decisions": audits})
    write_new_receipt(
        directory / "proof.json",
        {
            "checked_at": utc_now(),
            "frozen_specification_sha256": digest_json(frozen),
            **verification,
            "prefix_truncations": prefixes,
            "formal_factor_coverage": coverage,
            "actual_input_resolved_records": reference(log),
            "information_audits": reference(FOLDER / "native-information-audits-v1.json"),
            "whole_native_prehistory_retained": True,
            "synthetic_initial_recommendation_added": False,
            "candidate_accounts_or_metrics_computed": False,
            "original_source_signal_recomputed": False,
            "first_new_policy_targets_computed": True,
            "independent_forward_validation": False,
        },
    )
    entry = {
        "targets": reference(directory / "targets.csv"),
        "proof": reference(directory / "proof.json"),
    }
    write_new_receipt(
        PREPARED, {"frozen_specification_sha256": digest_json(frozen), "entries": {ID: entry}}
    )
    check_frozen(frozen)
    return entry


def record_failure(stage, error):
    return rebound(template.record_failure)(stage, error)


def register():
    return rebound(template.register)()


def publication(frozen, commit):
    return rebound(template.publication)(frozen, commit)


def evaluate_path(data, targets, frozen, window, scenario, benchmarks):
    return rebound(template.evaluate_path)(data, targets, frozen, window, scenario, benchmarks)


def evaluate(commit):
    return rebound(template.evaluate)(commit)


def export():
    result = read_json(RESULTS)
    frozen = result["frozen"]
    paths = result["paths"]
    if set(paths) != base.PATH_KEYS or len(events("review")) != 1:
        raise QuantError("Publish only the complete actual native review.")
    review = {
        **result["review"],
        "candidate_id": ID,
        "paths": [paths[key] for key in sorted(paths)],
    }
    contract = {
        "schema_version": 2,
        "study_id": frozen["study_id"],
        "registration_packet": reference(PACKET),
        "protocol": {"path": "config/dual-benchmark-v1.json", "sha256": base.PROTOCOL_SHA256},
        "completed_at": result["completed_at"],
        "reviews": [review],
        "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
        "native_price_prehistory_preserved": True,
        "raw_prices_targets_or_daily_book_rows_exported": False,
        "technical_admission_not_automatic_from_financial_review": True,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    source.common.retain_receipt(SOURCE_RESULTS, contract)
    source.common.retain_receipt(PUBLIC, contract)
    return reference(PUBLIC)


def admit():
    lifecycle.runtime_guard()
    if ADMISSION.exists() or (FOLDER / "technical-admission-attempt-v1.json").exists():
        raise QuantError("Preserve the complete or failed first technical admission.")
    with lifecycle.research_job():
        data, _, frozen, _ = inputs()
        before = current_counts(base.journal_events(base.JOURNAL))
        write_new_receipt(
            FOLDER / "technical-admission-attempt-v1.json",
            {"started_at": utc_now(), "frozen_sha256": digest_json(frozen)},
        )
        record = {
            "id": ID,
            "mechanism_group": "conditional_cross_asset_hedge_suitability",
            "adapter": "monthly",
            "introduced_at": events("registered")[0]["at"],
            "registered_event_sha256": events("registered")[0]["sha256"],
            "registration": reference(PACKET),
            "results": reference(SOURCE_RESULTS),
            "completed_paths": events("review")[0]["body"]["paths"],
        }
        try:
            expert, technical = load_expert(record, tuple(data.close.columns))
            shadow = pd.read_csv(
                base.checked_reference(technical["shadow_reference"]),
                index_col=0,
                parse_dates=True,
                float_precision="round_trip",
            )
            if not np.array_equal(
                shadow["risk_free"].to_numpy(), data.risk_free.loc[shadow.index].to_numpy()
            ):
                raise QuantError("New native RF must equal the unchanged active common RF exactly.")
            dates = [
                previous_session(window["start"]) for window in read_json(base.PROTOCOL)["windows"]
            ]
            dates.extend(
                day for day in data.close.index if day >= dates[0] and law.is_month_end(day)
            )
            dates.append(data.close.index[-1])
            observed = []
            for day in sorted(set(dates)):
                observation = expert.observe(day, 126)
                if (
                    observation.target is None
                    or observation.target_known_at is None
                    or observation.target_known_at > day
                    or not observation.risk_free.equals(
                        data.risk_free.loc[observation.risk_free.index]
                    )
                ):
                    raise QuantError(
                        "Native recommendations, exact RF or decision-time availability failed."
                    )
                observed.append(
                    {
                        "decision_session": str(day.date()),
                        "target_known_at": str(observation.target_known_at.date()),
                        "NET_observation_count": len(observation.net_returns),
                        "score_mature": observation.score_mature,
                    }
                )
            if current_counts(base.journal_events(base.JOURNAL)) != before:
                raise QuantError("Technical admission cannot create another financial trial.")
            packet = {
                "admitted_at": utc_now(),
                "record": record,
                "technical": technical,
                "observation_checks": observed,
                "native_RF_exact_common": True,
                "new_RF_or_recommendation_or_assessment_view": False,
                "funding_and_independent_BT_checked_all_four_paths": True,
                "standalone_financial_verdict_preserved": events("review")[0]["body"]["status"],
                "technical_admission_is_final_qualification": False,
                "new_technically_available_representative": ID,
                "technically_available_total_after_this_admission": 8,
                "active_pool_representatives_still7": True,
                "existing_pool_changed": False,
                "new_source_requests": 0,
                "new_strategy_evaluations": 0,
                "actual_authoritative_counts": before,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            }
            write_new_receipt(ADMISSION, packet)
            write_new_receipt(PUBLIC_ADMISSION, packet)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            write_new_receipt(
                FOLDER / "technical-admission-failure-v1.json",
                {
                    "failed_at": utc_now(),
                    "error": str(error),
                    "partial_evidence_preserved": True,
                    "pool_changed": False,
                },
            )
            raise
        progress(
            "technical_admission_complete_pool_unchanged",
            next_step="freeze_separate_pool_inclusion_if_justified",
        )
    return reference(ADMISSION)


def main():
    parser = argparse.ArgumentParser(description="New native hedge-policy study, no broker orders.")
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export", "admit"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use an actual remotely published preregistration SHA.")
    actions = {"freeze": initialize, "register": register, "export": export, "admit": admit}
    print(
        json.dumps(
            evaluate(args.commit) if args.action == "evaluate" else actions[args.action](), indent=2
        )
    )


if __name__ == "__main__":
    main()
