from __future__ import annotations

import argparse
import ast
import json
import math
import os
from fractions import Fraction
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import expert_native_arrival_equal_v1 as rules
from us_quant import expert_native_event_execution_v1 as event_execution
from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import ssd_eight_expert_meta_v1 as pool_source
from us_quant import ssd_expert_target_consensus_v1 as financial
from us_quant.calendar import is_month_end, previous_session
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
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

base, ROOT, source = pool_source.base, pool_source.ROOT, pool_source.seven
reference, unique_refs = base.reference, pool_source.unique_refs
FOLDER = base.REPORTS / "expert-native-arrival-equal-20261103-v1"
PRIMARY = FOLDER / "primary-implementation-freeze-v1.json"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
CLOCK_CONTRACT = FOLDER / "source-clock-contract-v1.json"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
PREPARED = FOLDER / "prepared-target-bundle-v1.json"
RESULTS = FOLDER / "results.json"
PACKET = ROOT / "evidence/expert_native_arrival_20261103_registration_v1.json"
PUBLIC = ROOT / "evidence/expert_native_arrival_20261103_results_v1.json"
PRIMARY_SHA = "26eaa8ebf99997aa56a545460bcce9120521766fc203ab467dc8be6ce9a74112"
DESIGN_SHA = "7cf39d0857c5f17258b626063952227f49636c992e26ab011ce13751c059da9b"
CANDIDATE = rules.CANDIDATES[0]
ID = CANDIDATE["id"]
OWN = (
    "config/expert-native-arrival-equal-v1.json",
    "config/expert-native-event-execution-v1.json",
    "src/us_quant/expert_native_arrival_equal_v1.py",
    "src/us_quant/expert_native_event_execution_v1.py",
    "src/us_quant/ssd_expert_native_arrival_equal_v1.py",
    "tests/test_expert_native_arrival_equal_v1.py",
    "tests/test_expert_native_event_execution_v1.py",
    "tests/test_ssd_expert_native_arrival_equal_v1.py",
    "tests/test_expert_initial_monthly_execution_v1.py",
    "tests/conftest.py",
)


class _NativeFeedbackAdapter:
    @staticmethod
    def verify_feedback(data, targets, result, issued, decisions, window, scenario):
        return event_execution.verify_feedback(
            data,
            targets,
            result,
            issued,
            decisions,
            read_json(event_execution.POLICY),
            window["start"],
            window["end"],
            scenario["cost_bps_per_side"],
            scenario["delay_sessions"],
        )


class _NativeFundingExecutionAdapter:
    """Satisfy the frozen funding Interface, not the original monthly implementation."""

    monthly = event_execution.monthly
    original = _NativeFeedbackAdapter

    @staticmethod
    def run(data, targets, candidate, policy, start, end, cost, delay):
        if candidate != event_execution.monthly.CANDIDATES[0]:
            raise QuantError("The legacy execution token cannot select another policy or band.")
        return event_execution.run(data, targets, policy, start, end, cost, delay)


execution = _NativeFundingExecutionAdapter()


def check_primary():
    if (
        file_digest(PRIMARY) != PRIMARY_SHA
        or file_digest(DESIGN) != DESIGN_SHA
        or file_digest(CLOCK_CONTRACT) != rules.CLOCK_CONTRACT_SHA
    ):
        raise QuantError("Keep the actual native-clock interpretation and pre-outcome law freeze.")
    saved = read_json(PRIMARY)
    for row in (
        saved["primary_implementation_and_test_inputs"]
        + saved["complete_primary_dependency_inputs"]
        + saved["cached_source_reference_hashes"]
        + [saved["validation"], saved["design"], saved["actual_invocation"]]
    ):
        base.checked_reference(row)
    rules.validate(read_json(rules.POLICY))
    event_execution.validate(read_json(event_execution.POLICY))
    return saved


def events(kind):
    return [
        row
        for row in base.journal_events(base.JOURNAL)
        if row["candidate_id"] == ID and row["kind"] == kind
    ]


def progress(phase, **fields):
    counts = current_counts(base.journal_events(base.JOURNAL))
    if any(key in fields and fields[key] != value for key, value in counts.items()):
        raise QuantError("Actual counts, failures and old financial history cannot be overridden.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "native_arrival_" + phase.removeprefix("consensus_"),
            "study_id": read_json(rules.POLICY)["study_id"],
            "active_compute_job": None,
            **fields,
            **counts,
        },
    )


def dependencies():
    walker = source.admitted.import_template.dependencies
    return FunctionType(
        walker.__code__,
        {**globals(), "ast": ast, "prior": pool_source},
        argdefs=walker.__defaults__,
    )()


def source_inputs():
    check_primary()
    data, experts, parent, refs, clocks, benchmarks = pool_source.source_inputs()
    old_inventory = (
        base.REPORTS
        / "expert-minimax-opportunity-packet-bridge-20261103-v2"
        / "all-saved-financial-private-inventory-v2.json"
    )
    refs = unique_refs(
        [
            *refs,
            *read_json(CLOCK_CONTRACT)["source_references"],
            reference(PRIMARY),
            reference(DESIGN),
            reference(INVOCATION),
            reference(CLOCK_CONTRACT),
            reference(pool_source.rules.POLICY),
            reference(pool_source.PUBLIC),
            reference(old_inventory),
        ]
    )
    for row in refs:
        base.checked_reference(row)
    return data, experts, parent, refs, clocks, benchmarks


def check_frozen(frozen):
    base.check_frozen_files(frozen)
    for row in frozen["prior_evidence"] + frozen["validation_evidence"]:
        base.checked_reference(row)


def validation_refs():
    path = FOLDER / "financial-validation-source-v1.json"
    saved = read_json(path)
    if (
        saved["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
        or saved["exit_code"] != 0
    ):
        raise QuantError("Financial tests must bind these exact native-clock and funding sources.")
    xml, log = [base.checked_reference(row) for row in saved["test_outputs"]]
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    if (
        xml.parent != FOLDER
        or log.parent != FOLDER
        or not suites
        or saved["passed_tests"] < 25
        or sum(int(s.get("tests", 0)) for s in suites) != saved["passed_tests"]
        or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
    ):
        raise QuantError("Complete the real packet, native source, execution and accounting tests.")
    return [*saved["test_outputs"], reference(path)]


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "financial-freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the first complete or partial native financial freeze.")
    with lifecycle.research_job():
        write_new_receipt(FOLDER / "financial-freeze-attempt-v1.json", {"started_at": utc_now()})
        _, _, parent, refs, clocks, _ = source_inputs()
        history = base.journal_events(base.JOURNAL)
        if len(history) != 737 or history[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError("Freeze against the actual 737-event pre-outcome baseline.")
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": read_json(rules.POLICY)["study_id"],
            "policy": reference(rules.POLICY),
            "expert_pool": reference(pool_source.rules.POLICY),
            "execution_policy": reference(event_execution.POLICY),
            "construction": "equal_eighths_at_native_policy_opinion_refresh_union",
            "original_policy_role": "unchanged_money_band_cost_delay_native_candidate_clock",
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in [*deps, *refs] if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / "EXPERT_NATIVE_ARRIVAL_EQUAL_V1.md"),
                    reference(ROOT / "EXPERT_NATIVE_ARRIVAL_ORCHESTRATION_V1.md"),
                ]
            ),
            "policies": unique_refs(
                [
                    *parent["policies"],
                    *[r for r in [*deps, *refs] if r["path"].startswith("config/")],
                ]
            ),
            "prior_evidence": unique_refs([*parent["prior_evidence"], *refs]),
            "validation_evidence": unique_refs(
                [*parent["validation_evidence"], *validation_refs()]
            ),
            "source_policy_and_expert_admission_clocks": clocks,
            "decision_information_clock": rules.CLOCK,
            "source_native_request_or_execution_dates_copied": False,
            "new_shadow_account_runs": 0,
            "new_source_requests": 0,
        }
        check_frozen(frozen)
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "frozen": frozen,
                "source_references": refs,
                "complete_dependency_inputs": deps,
                "package_versions": sorted(
                    (d.metadata["Name"], d.version) for d in distributions()
                ),
                "new_real_aggregate_targets_or_funded_accounts_computed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="register_before_first_real_aggregate_target")
    return reference(INTENT)


def inputs():
    lifecycle.runtime_guard()
    intent = read_json(INTENT)
    for row in intent["source_references"] + intent["complete_dependency_inputs"]:
        base.checked_reference(row)
    if intent["package_versions"] != [
        list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())
    ]:
        raise QuantError("The accepted isolated environment changed after freezing.")
    data, experts, _, refs, clocks, benchmarks = source_inputs()
    if (
        refs != intent["source_references"]
        or clocks != intent["frozen"]["source_policy_and_expert_admission_clocks"]
    ):
        raise QuantError("Cached source opinions, input versions or admission scope changed.")
    frozen = {
        **intent["frozen"],
        "prior_evidence": unique_refs([*intent["frozen"]["prior_evidence"], reference(INTENT)]),
    }
    check_frozen(frozen)
    return data, experts, frozen, benchmarks


def verify_retained(path, data, experts, targets, audits):
    dates = data.close.loc["2016-10-05":].index
    if (
        not targets.index.equals(data.close.index)
        or not targets.columns.equals(data.close.columns)
        or not targets.dropna(how="all").index.equals(dates)
        or [a["decision_session"] for a in audits] != dates.strftime("%Y-%m-%d").tolist()
    ):
        raise QuantError("Preserve every verified native event and the original full lattice.")
    monthly = pd.DatetimeIndex([day for day in data.close.index if is_month_end(day)])
    maximum_target_error = 0.0
    source_refreshes = [0] * 8
    with path.open() as stream:
        for day, audit in zip(dates, audits, strict=True):
            query, before, after = [json.loads(stream.readline()) for _ in range(3)]
            last_month = monthly[monthly <= day][-1]
            expected_queries = [day if i == 3 else last_month for i in range(8)]
            if (
                query["record"] != "opinion_query"
                or query["meta_as_of"] != str(day.date())
                or query["information_clock"] != rules.CLOCK
                or query["source_ids"] != list(rules.EXPERT_IDS)
                or query["native_query_sessions"] != [str(d.date()) for d in expected_queries]
                or before["record"] != "opinion_input"
                or before["meta_as_of"] != str(day.date())
                or before["equal_expert_weights_binary64_hex"] != [0.125.hex()] * 8
                or before["source_NET_or_RF_used_to_rank_or_allocate"] is not False
                or after["record"] != "opinion_resolved"
                or after["meta_target_emitted"] is not True
                or after["meta_as_of"] != str(day.date())
                or after["audit"] != audit
                or before["source_opinions"] != audit["source_opinions"]
                or len(before["source_opinions"]) != 8
            ):
                raise QuantError("Retain actual native queries and raw opinions before aggregates.")
            actual_sources = []
            fresh = []
            for i, (row, expert, native_day) in enumerate(
                zip(before["source_opinions"], experts, expected_queries, strict=True)
            ):
                native = expert.observe(native_day, 2)
                arrived = native_day == day
                if (
                    row["id"] != rules.EXPERT_IDS[i]
                    or row["mechanism_group"] != rules.MECHANISMS[i]
                    or row["introduced_at"] != native.identity.introduced_at
                    or row["source_observed_as_of"] != str(native_day.date())
                    or row["source_known_at"] != str(native.target_known_at.date())
                    or row["policy_refreshed_at"] != str(native_day.date())
                    or row["arrived_today"] is not arrived
                    or row["native_opinion_clock"] != rules.SOURCE_CLOCKS[i]
                    or row["asset_order"] != list(targets.columns)
                    or row["target_binary64_hex"] != [float(x).hex() for x in native.target]
                    or row["source_native_order_or_execution_claimed"] is not False
                    or native.target_known_at != native_day
                ):
                    raise QuantError("An opinion, source clock or native request role changed.")
                actual_sources.append([float.fromhex(x) for x in row["target_binary64_hex"]])
                if arrived:
                    fresh.append(row["id"])
                    if day > dates[0]:
                        source_refreshes[i] += 1
            if (
                audit["expert_weights"] != [0.125] * 8
                or audit["fresh_source_opinion_ids"] != fresh
                or audit["decision_role"]
                != ("initial_capital" if day == dates[0] else "native_opinion_refresh")
                or audit["decision_information_clock"] != rules.CLOCK
                or audit["counterfactual_historical_expert_availability"] is not True
                or any(
                    audit[k] is not False
                    for k in (
                        "source_scores_used",
                        "source_requests_or_held_positions_copied",
                        "shadow_wealth_or_protection_guarantee_transferred",
                        "independent_forward_validation",
                    )
                )
            ):
                raise QuantError(
                    "The fixed equal policy, information clock or source role changed."
                )
            row = targets.loc[day]
            if (
                not np.isfinite(row).all()
                or (row < 0).any()
                or abs(float(row.sum()) - 0.98) > 1e-12
                or after["target_binary64_hex"] != [float(x).hex() for x in row]
            ):
                raise QuantError("The actual cash-funded target differs from retained output.")
            for j, value in enumerate(row):
                exact = sum((Fraction(source[j]) for source in actual_sources), Fraction(0)) / 8
                error = abs(float(exact) - float(value))
                maximum_target_error = max(maximum_target_error, error)
                if not math.isfinite(error) or error > 1e-12:
                    raise QuantError("Independent rational equal-opinion composition disagrees.")
        if stream.readline():
            raise QuantError("Unexpected extra or partial native opinion records.")
    return {
        "native_decisions_verified": len(audits),
        "actual_query_input_resolved_records": 3 * len(audits),
        "source_opinion_snapshots_verified": 8 * len(audits),
        "source_refresh_counts_excluding_initial": source_refreshes,
        "maximum_independent_rational_target_difference": maximum_target_error,
        "raw_dense_known_at_not_used_as_native_clock": True,
        "source_native_requests_or_positions_copied": False,
        "original_source_signal_recomputed": False,
    }


def prepare(data, experts, frozen):
    if PREPARED.exists():
        saved = read_json(PREPARED)
        if saved["frozen_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting prior native-arrival targets.")
        for row in saved["prepared"].values():
            base.checked_reference(row)
        return saved["prepared"]
    if len(events("registered")) != 1 or (FOLDER / "preparation-attempt-v1.json").exists():
        raise QuantError("Register once before new targets and preserve partial preparations.")
    write_new_receipt(
        FOLDER / "preparation-attempt-v1.json",
        {"started_at": utc_now(), "frozen_sha256": digest_json(frozen)},
    )
    log = FOLDER / "actual-opinion-inputs-resolved-v1.jsonl"
    with log.open("x") as stream:

        def retain(row):
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        targets, audits = rules.build_targets(
            experts,
            data.close.index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            before_decision=lifecycle.runtime_guard,
            retain=retain,
        )
    proof = verify_retained(log, data, experts, targets, audits)
    if (
        len(audits) != 2513
        or proof["actual_query_input_resolved_records"] != 7539
        or proof["source_refresh_counts_excluding_initial"]
        != [120, 120, 120, 2512, 120, 120, 120, 120]
    ):
        raise QuantError("Keep every proved native opinion clock and formal decision date.")
    prefixes = []
    for stamp in ("2016-10-05", "2016-10-06", "2016-10-31", "2021-10-05", "2026-10-05"):
        day = pd.Timestamp(stamp)
        early, earlier = rules.build_targets(
            experts,
            data.close.loc[:day].index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            before_decision=lifecycle.runtime_guard,
        )
        if not early.equals(targets.loc[:day]) or earlier != [
            row for row in audits if row["decision_session"] <= stamp
        ]:
            raise QuantError("Future observations changed an earlier native policy decision.")
        prefixes.append(stamp)
    coverage = []
    active = targets.dropna(how="all")
    for window in read_json(base.PROTOCOL)["windows"]:
        anchor = previous_session(window["start"])
        known = active.loc[:anchor].tail(1)
        rows = pd.concat([known, active.loc[anchor + pd.Timedelta(days=1) : window["end"]]])
        factors = [
            symbol
            for symbol in ("MTUM", "VLUE", "QUAL", "USMV", "IJR", "PKW")
            if (rows[symbol] > 1e-10).any()
        ]
        if known.empty or len(factors) < 3:
            raise QuantError("Both original horizons need known anchors and genuine factors.")
        coverage.append({"years": window["years"], "actual_factor_symbols": factors})
    directory = FOLDER / "targets"
    new_output_directory(directory)
    write_text_atomic(directory / "targets.csv", targets.to_csv(float_format="%.17g"))
    restored = pd.read_csv(
        directory / "targets.csv", index_col=0, parse_dates=True, float_precision="round_trip"
    )
    if not restored.equals(targets):
        raise QuantError("Stored native targets must preserve the actual binary64 values.")
    write_new_receipt(FOLDER / "information-audits-v1.json", {"decisions": audits})
    write_new_receipt(
        directory / "proof.json",
        {
            "checked_at": utc_now(),
            "frozen_specification_sha256": digest_json(frozen),
            **proof,
            "prefix_truncations": prefixes,
            "formal_factor_coverage": coverage,
            "actual_input_resolved_records": reference(log),
            "candidate_accounts_or_metrics_computed": False,
            "original_source_signal_recomputed": False,
            "source_shadow_or_control_account_replayed": False,
            "independent_forward_validation": False,
        },
    )
    prepared = {
        "targets": reference(directory / "targets.csv"),
        "proof": reference(directory / "proof.json"),
    }
    write_new_receipt(PREPARED, {"frozen_sha256": digest_json(frozen), "prepared": prepared})
    return prepared


def bound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


def register():
    return bound(financial.register)()


def publication(frozen, commit):
    return bound(financial.publication)(frozen, commit)


def record_failure(stage, error):
    return bound(financial.record_failure)(stage, error)


def evaluate_path(*args):
    return bound(financial.evaluate_path)(*args)


def evaluate(commit):
    return bound(financial.evaluate)(commit)


def export():
    saved = read_json(RESULTS)
    if set(saved["paths"]) != base.PATH_KEYS:
        raise QuantError("Publish every original path of the same frozen native-arrival rule.")
    control = read_json(base.checked_reference(reference(pool_source.PUBLIC)))["candidates"][
        "expert_pool_eight_equal_weight_v1"
    ]
    body = {
        "schema_version": 1,
        "study_id": read_json(rules.POLICY)["study_id"],
        "candidate_id": ID,
        "completed_at": saved["completed_at"],
        "status": saved["review"]["status"],
        "paths": saved["paths"],
        "cached_monthly_equal8_control": control,
        "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
        "original_active_seven_and_available_eight_unchanged": True,
        "native_opinion_clocks_not_source_execution_copying": True,
        "source_generators_shadows_controls_or_old_accounts_replayed": False,
        "new_source_requests": 0,
        "raw_source_prices_targets_or_daily_rows_exported": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    source.common.retain_receipt(PUBLIC, body)
    return reference(PUBLIC)


def main():
    parser = argparse.ArgumentParser(
        description="Frozen native opinion equal-expert research only."
    )
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use an actual remotely published preregistration SHA.")
    actions = {"freeze": initialize, "register": register, "export": export}
    print(
        json.dumps(evaluate(args.commit) if args.action == "evaluate" else actions[args.action]())
    )


if __name__ == "__main__":
    main()
