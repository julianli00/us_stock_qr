from __future__ import annotations

import argparse
import ast
import itertools
import json
import math
import os
from fractions import Fraction
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import expert_instrument_concentration_v1 as rules
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
FOLDER = base.REPORTS / "expert-instrument-concentration-20261103-v1"
PRIMARY = FOLDER / "primary-implementation-freeze-v1.json"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
PREPARED = FOLDER / "prepared-target-bundle-v1.json"
PACKET = ROOT / "evidence/expert_instrument_concentration_20261103_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC = ROOT / "evidence/expert_instrument_concentration_20261103_results_v1.json"
PRIMARY_SHA = "692a1d89b69b9c079fc39bd51e78d9bca4cce1580a23cad0636f45711dafc2e4"
DESIGN_SHA = "c17375819095f8f94cd365541618ee4024fccd096776eb9edea0e3eae201962f"
CANDIDATE = rules.CANDIDATES[0]
ID = CANDIDATE["id"]
OWN = (
    "config/expert-instrument-concentration-v1.json",
    "src/us_quant/expert_instrument_concentration_v1.py",
    "src/us_quant/ssd_expert_instrument_concentration_v1.py",
    "tests/test_expert_instrument_concentration_v1.py",
    "tests/test_ssd_expert_instrument_concentration_v1.py",
    "tests/test_expert_native_arrival_equal_v1.py",
    "tests/conftest.py",
)


def check_primary():
    if file_digest(PRIMARY) != PRIMARY_SHA or file_digest(DESIGN) != DESIGN_SHA:
        raise QuantError("Keep the actual pre-outcome instrument-composition law.")
    saved = read_json(PRIMARY)
    for row in (
        saved["primary_implementation_and_test_inputs"]
        + saved["complete_primary_dependency_inputs"]
        + [saved["validation"], saved["design"], saved["actual_invocation"]]
    ):
        base.checked_reference(row)
    rules.validate(read_json(rules.POLICY))
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
        raise QuantError("Keep actual registrations, completions and failures without resetting.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "instrument_concentration_" + phase.removeprefix("consensus_"),
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
    native = base.REPORTS / "expert-native-arrival-equal-20261103-v1"
    refs = unique_refs(
        [
            *refs,
            reference(PRIMARY),
            reference(DESIGN),
            reference(INVOCATION),
            reference(pool_source.rules.POLICY),
            reference(pool_source.PUBLIC),
            reference(native / "source-clock-contract-v1.json"),
            reference(native / "primary-implementation-freeze-v1.json"),
            reference(native / "all-saved-financial-private-inventory-v1.json"),
            reference(native / "completion-v1.json"),
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
        raise QuantError("Successful canonical checks must bind these exact source bytes.")
    xml, log = [base.checked_reference(row) for row in saved["test_outputs"]]
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    if (
        xml.parent != FOLDER
        or log.parent != FOLDER
        or not suites
        or saved["passed_tests"] < 20
        or sum(int(s.get("tests", 0)) for s in suites) != saved["passed_tests"]
        or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
    ):
        raise QuantError("Complete all current-target, rational-HHI, clock and real-packet checks.")
    return [*saved["test_outputs"], reference(path)]


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "financial-freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the first complete or partial instrument financial freeze.")
    with lifecycle.research_job():
        write_new_receipt(FOLDER / "financial-freeze-attempt-v1.json", {"started_at": utc_now()})
        _, _, parent, refs, clocks, _ = source_inputs()
        history = base.journal_events(base.JOURNAL)
        if len(history) != 749 or history[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError("Freeze against the actual 749-event pre-outcome baseline.")
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": read_json(rules.POLICY)["study_id"],
            "policy": reference(rules.POLICY),
            "expert_pool": reference(pool_source.rules.POLICY),
            "construction": "equal_pair_minimum_current_ETF_instrument_concentration",
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in [*deps, *refs] if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / "EXPERT_INSTRUMENT_CONCENTRATION_V1.md"),
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
                "actual_new_instrument_HHI_pair_selection_targets_or_funds_computed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="register_before_first_real_composition_score")
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
        raise QuantError("Source opinions, original inputs or admission scope changed.")
    frozen = {
        **intent["frozen"],
        "prior_evidence": unique_refs([*intent["frozen"]["prior_evidence"], reference(INTENT)]),
    }
    check_frozen(frozen)
    return data, experts, frozen, benchmarks


def _close(actual, expected):
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
        raise QuantError("Independent current composition arithmetic disagrees.")


def verify_retained(path, data, experts, targets, audits):
    initial = pd.Timestamp("2016-10-05")
    dates = pd.DatetimeIndex(
        [initial, *[day for day in data.close.index if day > initial and is_month_end(day)]]
    )
    months = pd.DatetimeIndex([day for day in data.close.index if is_month_end(day)])
    if (
        not targets.index.equals(data.close.index)
        or not targets.columns.equals(data.close.columns)
        or not targets.dropna(how="all").index.equals(dates)
        or [a["decision_session"] for a in audits] != dates.strftime("%Y-%m-%d").tolist()
    ):
        raise QuantError("Keep every original capital/month-end decision and asset coordinate.")
    previous = None
    target_gap = concentration_gap = 0.0
    with path.open() as stream:
        for day, audit in zip(dates, audits, strict=True):
            before, after = json.loads(stream.readline()), json.loads(stream.readline())
            if (
                before["record"] != "decision_input"
                or before["as_of"] != str(day.date())
                or before["decision_information_clock"] != rules.CLOCK
                or before["previous_expert_weights_binary64_hex"]
                != ([float(x).hex() for x in previous] if previous is not None else None)
                or before["source_opinions"] != audit["source_opinions"]
                or len(before["source_opinions"]) != 8
                or after["record"] != "decision_resolved"
                or after["audit"] != audit
            ):
                raise QuantError("Source operands must precede all new concentration cases.")
            native_days = [day if i == 3 else months[months <= day][-1] for i in range(8)]
            matrix = []
            for i, (row, expert, native_day) in enumerate(
                zip(before["source_opinions"], experts, native_days, strict=True)
            ):
                native = expert.observe(native_day, 2)
                if (
                    row["id"] != rules.EXPERT_IDS[i]
                    or row["mechanism_group"] != rules.MECHANISMS[i]
                    or row["introduced_at"] != native.identity.introduced_at
                    or row["source_query_as_of"] != str(native_day.date())
                    or row["source_known_at"] != str(native.target_known_at.date())
                    or row["policy_refreshed_at"] != str(native_day.date())
                    or row["asset_order"] != list(targets.columns)
                    or row["target_binary64_hex"] != [float(x).hex() for x in native.target]
                    or row["source_native_order_or_execution_claimed"] is not False
                    or native.target_known_at != native_day
                ):
                    raise QuantError(
                        "The exact original instrument opinion or native clock changed."
                    )
                matrix.append([float.fromhex(x) for x in row["target_binary64_hex"]])
            rows = audit["selection"]["pair_records"]
            records = {tuple(row["pair"]): row for row in rows}
            if len(rows) != 28 or len(records) != 28:
                raise QuantError("Every complete pair concentration case must be retained.")
            ranking = []
            for first, second in itertools.combinations(range(8), 2):
                names = tuple(sorted((rules.EXPERT_IDS[first], rules.EXPERT_IDS[second])))
                row = records[names]
                weights = [0.5 if i in (first, second) else 0.0 for i in range(8)]
                exact = [
                    (Fraction(matrix[first][j]) + Fraction(matrix[second][j])) / 2
                    for j in range(len(matrix[0]))
                ]
                values = [float(x) for x in exact]
                concentration = math.fsum((x / 0.98) ** 2 for x in values)
                rational = float(sum((x / Fraction(0.98)) ** 2 for x in exact))
                concentration_gap = max(concentration_gap, abs(concentration - rational))
                _close(concentration, rational)
                cost = 0.0
                changed = 0
                if previous is not None:
                    movement = [
                        math.fsum((weights[i] - previous[i]) * matrix[i][j] for i in range(8))
                        for j in range(len(matrix[0]))
                    ]
                    changed = sum(abs(x) > 1e-12 for x in movement)
                    cost = 5 * math.fsum(abs(x) for x in movement) + changed
                if (
                    row["expert_weights"] != weights
                    or row["combined_target_binary64_hex"] != [x.hex() for x in values]
                    or row["switch_changed_asset_coordinates"] != changed
                    or row["initial_no_previous_allocation"] is not (previous is None)
                ):
                    raise QuantError(
                        "A pair coordinate, initial prior or changed-asset count differs."
                    )
                _close(row["instrument_concentration"], concentration)
                _close(row["switch_cost_proxy_usd"], cost)
                if row["concentration_key_12dp"] != round(concentration, 12) or row[
                    "switch_cost_key_12dp"
                ] != round(cost, 12):
                    raise QuantError("Independent concentration/cost tie keys disagree.")
                ranking.append((round(concentration, 12), round(cost, 12), names, weights))
            key, cost_key, names, weights = min(ranking, key=lambda row: row[:3])
            selected = audit["selection"]
            if (
                selected["selected_pair"] != list(names)
                or selected["selected_concentration_key_12dp"] != key
                or selected["selected_switch_cost_key_12dp"] != cost_key
                or audit["expert_weights"] != weights
                or audit["previous_expert_weights"] != previous
                or audit["source_return_scores_used"] is not False
                or audit["source_orders_positions_or_wealth_floor_copied"] is not False
                or audit["counterfactual_historical_expert_availability"] is not True
                or audit["independent_forward_validation"] is not False
                or selected["source_return_or_covariance_history_used"] is not False
                or selected["instrument_HHI_is_economic_independence_or_future_risk"] is not False
            ):
                raise QuantError(
                    "The full-menu structural ranking or financial evidence role changed."
                )
            actual = targets.loc[day]
            if (
                not np.isfinite(actual).all()
                or (actual < 0).any()
                or abs(float(actual.sum()) - 0.98) > 1e-12
                or after["target_binary64_hex"] != [float(x).hex() for x in actual]
            ):
                raise QuantError("The actual unified target differs from the retained output.")
            for j, value in enumerate(actual):
                exact = sum(
                    (Fraction(weights[i]) * Fraction(matrix[i][j]) for i in range(8)),
                    Fraction(0),
                )
                gap = abs(float(exact) - float(value))
                target_gap = max(target_gap, gap)
                if gap > 1e-12:
                    raise QuantError("Independent rational unified-target composition disagrees.")
            previous = weights
        if stream.readline():
            raise QuantError("Unexpected extra or partial instrument-selection records.")
    return {
        "verified_decisions": len(audits),
        "actual_input_resolved_records": 2 * len(audits),
        "all28_current_pair_concentration_records_verified": True,
        "maximum_independent_rational_target_difference": target_gap,
        "maximum_independent_rational_concentration_difference": concentration_gap,
        "initial_previous_allocation_absent_not_fabricated": True,
        "original_source_signal_recomputed": False,
    }


def prepare(data, experts, frozen):
    if PREPARED.exists():
        saved = read_json(PREPARED)
        if saved["frozen_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting prior composition targets.")
        for row in saved["prepared"].values():
            base.checked_reference(row)
        return saved["prepared"]
    if len(events("registered")) != 1 or (FOLDER / "preparation-attempt-v1.json").exists():
        raise QuantError("Register once before current HHI and preserve partial preparations.")
    write_new_receipt(
        FOLDER / "preparation-attempt-v1.json",
        {"started_at": utc_now(), "frozen_sha256": digest_json(frozen)},
    )
    log = FOLDER / "actual-decision-inputs-resolved-v1.jsonl"
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
    if len(audits) != 121 or proof["actual_input_resolved_records"] != 242:
        raise QuantError("Keep the full original initial/month-end decision lattice.")
    prefixes = []
    for position in (0, 6, 12, 60, 120):
        day = pd.Timestamp(audits[position]["decision_session"])
        early, earlier = rules.build_targets(
            experts,
            data.close.loc[:day].index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            before_decision=lifecycle.runtime_guard,
        )
        if not early.equals(targets.loc[:day]) or earlier != audits[: position + 1]:
            raise QuantError("Future opinions changed a previous composition selection.")
        prefixes.append(str(day.date()))
    active = targets.dropna(how="all")
    coverage = []
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
            raise QuantError(
                "Both original horizons need known anchors and genuine factor exposure."
            )
        coverage.append({"years": window["years"], "actual_factor_symbols": factors})
    directory = FOLDER / "targets"
    new_output_directory(directory)
    write_text_atomic(directory / "targets.csv", targets.to_csv(float_format="%.17g"))
    restored = pd.read_csv(
        directory / "targets.csv", index_col=0, parse_dates=True, float_precision="round_trip"
    )
    if not restored.equals(targets):
        raise QuantError("Stored composition target values must round-trip exactly.")
    write_new_receipt(FOLDER / "information-audits-v1.json", {"decisions": audits})
    write_new_receipt(
        directory / "proof.json",
        {
            "checked_at": utc_now(),
            "frozen_specification_sha256": digest_json(frozen),
            **proof,
            "prefix_truncations": prefixes,
            "formal_factor_coverage": coverage,
            "actual_input_resolved_records_reference": reference(log),
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
        raise QuantError("Publish every original path, not passing fragments.")
    controls = read_json(base.checked_reference(reference(pool_source.PUBLIC)))["candidates"]
    body = {
        "schema_version": 1,
        "study_id": read_json(rules.POLICY)["study_id"],
        "candidate_id": ID,
        "completed_at": saved["completed_at"],
        "status": saved["review"]["status"],
        "paths": saved["paths"],
        "cached_eight_controls": controls,
        "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
        "original_active_seven_and_available_eight_unchanged": True,
        "instrument_concentration_not_stock_PIT_overlap_or_future_risk": True,
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
        description="Frozen current instrument-composition research only."
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
