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

from us_quant import expert_minimax_opportunity_v1 as rules
from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import ssd_eight_expert_meta_v1 as pool_source
from us_quant import ssd_expert_target_consensus_v1 as financial
from us_quant.calendar import is_month_end, next_session, previous_session, sessions
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
FOLDER = base.REPORTS / "expert-minimax-opportunity-20261103-v1"
PRIMARY = FOLDER / "primary-implementation-freeze-v1.json"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
PREPARED = FOLDER / "prepared-target-bundle-v1.json"
PACKET = ROOT / "evidence/expert_minimax_opportunity_20261103_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC = ROOT / "evidence/expert_minimax_opportunity_20261103_results_v1.json"
CANDIDATE = rules.CANDIDATES[0]
ID = CANDIDATE["id"]
PRIMARY_SHA = "b413f22133ae184378f2bac6aaf0dd46ad8823b62ea1a903fb31902b0ba35325"
DESIGN_SHA = "d5fb89f3bed9708b9d6f09cf87a23e42db44deaba6cf3b7db99d393b208565c2"
OWN = (
    "config/expert-minimax-opportunity-v1.json",
    "src/us_quant/expert_minimax_opportunity_v1.py",
    "src/us_quant/ssd_expert_minimax_opportunity_v1.py",
    "tests/test_expert_minimax_opportunity_v1.py",
    "tests/test_ssd_expert_minimax_opportunity_v1.py",
    "tests/conftest.py",
)


def check_primary():
    if file_digest(PRIMARY) != PRIMARY_SHA or file_digest(DESIGN) != DESIGN_SHA:
        raise QuantError("Preserve the actual pre-outcome opportunity law and primary freeze.")
    primary = read_json(PRIMARY)
    for row in (
        primary["primary_implementation_and_test_inputs"]
        + primary["complete_primary_dependency_inputs"]
        + [primary["validation"], primary["design"], primary["actual_invocation"]]
    ):
        base.checked_reference(row)
    rules.validate(read_json(rules.POLICY))
    return primary


def events(kind):
    return [
        row
        for row in base.journal_events(base.JOURNAL)
        if row["kind"] == kind and row["candidate_id"] == ID
    ]


def progress(phase, **fields):
    counts = current_counts(base.journal_events(base.JOURNAL))
    if any(key in fields and fields[key] != value for key, value in counts.items()):
        raise QuantError("A study cannot override actual authoritative financial counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "minimax_opportunity_" + phase.removeprefix("consensus_"),
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
        {**globals(), "prior": pool_source, "ast": ast},
        argdefs=walker.__defaults__,
    )()


def source_inputs():
    check_primary()
    data, experts, parent, refs, clocks, benchmarks = pool_source.source_inputs()
    if tuple(item.identity.identifier for item in experts) != rules.EXPERT_IDS:
        raise QuantError("Use the same eight admitted complete source identities, not a ninth.")
    refs = unique_refs(
        [
            *refs,
            reference(pool_source.rules.POLICY),
            reference(pool_source.INTENT),
            reference(pool_source.RESULTS),
            reference(pool_source.PUBLIC),
            reference(pool_source.FOLDER / "completion-v1.json"),
            reference(pool_source.FOLDER / "all-saved-financial-private-inventory-v1.json"),
            reference(PRIMARY),
            reference(DESIGN),
            reference(INVOCATION),
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
    xml = FOLDER / "offline-financial-tests-v2.xml"
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    if (
        not suites
        or sum(int(s.get("tests", 0)) for s in suites) != 28
        or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
    ):
        raise QuantError("Complete the opportunity, input-oracle and financial-seam checks.")
    path = FOLDER / "financial-validation-source-v1.json"
    proof = read_json(path)
    outputs = [reference(xml), reference(FOLDER / "offline-financial-tests-v2.log")]
    if (
        proof["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
        or proof["test_outputs"] != outputs
        or proof["passed_tests"] != sum(int(s.get("tests", 0)) for s in suites)
        or proof["exit_code"] != 0
    ):
        raise QuantError("Successful canonical tests must bind these exact financial code bytes.")
    return [*outputs, reference(path)]


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "financial-freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the complete or partial first financial freeze.")
    with lifecycle.research_job():
        write_new_receipt(FOLDER / "financial-freeze-attempt-v1.json", {"started_at": utc_now()})
        _, _, parent, refs, clocks, _ = source_inputs()
        history = base.journal_events(base.JOURNAL)
        if len(history) != 722 or history[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError("Freeze against the actual 722-event pre-outcome baseline.")
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": read_json(rules.POLICY)["study_id"],
            "policy": reference(rules.POLICY),
            "expert_pool": reference(rules.POLICY),
            "construction": "eight_complete_experts_worst_mature_month_opportunity",
            "expert_pool_definition_role": "study_specific_same_eight_original_seven_immutable",
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in [*deps, *refs] if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / "EXPERT_MINIMAX_OPPORTUNITY_V1.md"),
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
                "new_month_relatives_scores_targets_or_funded_accounts_computed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="register_before_first_actual_month_score_or_target")
    return reference(INTENT)


def inputs():
    lifecycle.runtime_guard()
    intent = read_json(INTENT)
    for row in intent["source_references"] + intent["complete_dependency_inputs"]:
        base.checked_reference(row)
    if intent["package_versions"] != [
        list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())
    ]:
        raise QuantError("The frozen isolated source environment changed.")
    data, experts, _, refs, clocks, benchmarks = source_inputs()
    if (
        refs != intent["source_references"]
        or clocks != intent["frozen"]["source_policy_and_expert_admission_clocks"]
    ):
        raise QuantError("Frozen sources, native accounting or admission clocks changed.")
    frozen = {
        **intent["frozen"],
        "prior_evidence": unique_refs([*intent["frozen"]["prior_evidence"], reference(INTENT)]),
    }
    check_frozen(frozen)
    return data, experts, frozen, benchmarks


def _close(actual, expected, tolerance=1e-12):
    if not math.isfinite(actual) or not math.isclose(
        actual, expected, rel_tol=0, abs_tol=tolerance
    ):
        raise QuantError("Independent opportunity arithmetic disagrees with the retained result.")


def verify_retained(path, experts, targets, audits):
    initial = pd.Timestamp("2016-10-05")
    expected_dates = pd.DatetimeIndex(
        [initial, *[day for day in targets.index if day > initial and is_month_end(day)]]
    )
    declared = pd.DatetimeIndex([pd.Timestamp(row["decision_session"]) for row in audits])
    if (
        targets.empty
        or not targets.index.equals(sessions(targets.index[0], targets.index[-1]))
        or not declared.equals(expected_dates)
        or not targets.columns.is_unique
        or targets.loc[~targets.index.isin(expected_dates)].notna().any().any()
        or not np.isfinite(targets.loc[expected_dates].to_numpy()).all()
        or (targets.loc[expected_dates] < 0).any().any()
        or (abs(targets.loc[expected_dates].sum(axis=1) - 0.98) > 1e-12).any()
    ):
        raise QuantError("The complete target clock and funded asset lattice must be preserved.")
    previous = [0.125] * 8
    history, intervals = [], []
    anchor = None
    target_gap = relative_gap = 0.0
    mature = 0
    with path.open() as stream:
        for position, audit in enumerate(audits):
            before, after = json.loads(stream.readline()), json.loads(stream.readline())
            day = pd.Timestamp(audit["decision_session"])
            if (
                before["record"] != "decision_input"
                or before["decision_index"] != position
                or before["as_of"] != audit["decision_session"]
                or before["decision_information_clock"] != rules.CLOCK
                or before["previous_expert_weights_binary64_hex"] != [x.hex() for x in previous]
                or before["previous_matured_months_binary64_hex"] != history
                or before["previous_completed_intervals"] != intervals
                or after["record"] != "decision_resolved"
                or after["audit"] != audit
                or len(before["observations"]) != 8
                or audit["decision_information_clock"] != rules.CLOCK
                or audit["counterfactual_historical_expert_availability"] is not True
                or audit["source_intramonth_feedback_or_shadow_wealth_guarantee_transferred"]
                is not False
                or audit["independent_forward_validation"] is not False
            ):
                raise QuantError(
                    "Every source interval must be retained before its new derivation."
                )
            if before["new_interval_left_anchor"] != (str(anchor.date()) if anchor else None):
                raise QuantError("The full source-month anchor changed.")
            observed = [expert.observe(day, 2512) for expert in experts]
            if tuple(targets.columns) != tuple(observed[0].target.index):
                raise QuantError("The unified target uses a different asset coordinate order.")
            matrix = []
            products = []
            for item, native in zip(before["observations"], observed, strict=True):
                if (
                    item["id"] != native.identity.identifier
                    or item["mechanism_group"] != native.identity.mechanism_group
                    or item["introduced_at"] != native.identity.introduced_at
                    or item["as_of"] != str(day.date())
                    or item["target_known_at"] != str(native.target_known_at.date())
                    or item["asset_order"] != list(native.target.index)
                    or item["target_binary64_hex"] != [float(x).hex() for x in native.target]
                    or item["full_prefix_return_count"] != len(native.net_returns)
                    or item["source_delivery_score_mature"] != native.score_mature
                ):
                    raise QuantError("Retained raw opinion differs from the frozen source.")
                matrix.append([float.fromhex(x) for x in item["target_binary64_hex"]])
                net = (
                    native.net_returns.loc[native.net_returns.index > anchor]
                    if anchor is not None
                    else native.net_returns.iloc[:0]
                )
                rf = (
                    native.risk_free.loc[native.risk_free.index > anchor]
                    if anchor is not None
                    else native.risk_free.iloc[:0]
                )
                if (
                    item["new_NET_dates"] != net.index.strftime("%Y-%m-%d").tolist()
                    or item["new_RF_dates"] != rf.index.strftime("%Y-%m-%d").tolist()
                    or item["new_NET_binary64_hex"] != [float(x).hex() for x in net]
                    or item["new_RF_binary64_hex"] != [float(x).hex() for x in rf]
                ):
                    raise QuantError("Source NET or exact RF was changed, filled or omitted.")
                if anchor is not None:
                    expected = sessions(next_session(anchor), day)
                    if not net.index.equals(expected) or not rf.index.equals(expected):
                        raise QuantError("A partial or missing source month cannot be assessed.")
                    products.append(math.prod(1 + float(value) for value in net))
            actual_months = audit["source_month_relatives_binary64_hex"]
            if anchor is not None:
                if len(actual_months) != len(history) + 1 or actual_months[:-1] != history:
                    raise QuantError("Do not drop, reset or append future assessment months.")
                for actual, expected in zip(actual_months[-1], products, strict=True):
                    value = float.fromhex(actual)
                    if value <= 0 or not math.isclose(value, expected, rel_tol=1e-12, abs_tol=0):
                        raise QuantError(
                            "Independent scalar compounding disagrees with source NET."
                        )
                    relative_gap = max(relative_gap, abs(value - expected) / value)
                intervals.append([str(anchor.date()), str(day.date())])
            elif actual_months:
                raise QuantError("The initial partial source month cannot train the rule.")
            history = actual_months
            if is_month_end(day):
                anchor = day
            if audit["completed_intervals"] != intervals:
                raise QuantError("Matured interval clocks cannot change.")
            selection = audit["selection"]
            if not history:
                expected_weights = [0.125] * 8
                if selection["score_mature"] or selection["pair_records"]:
                    raise QuantError("Initial immature states must remain explicitly unscored.")
            else:
                mature += 1
                values = [[float.fromhex(x) for x in row] for row in history]
                candidates = []
                records = {tuple(row["pair"]): row for row in selection["pair_records"]}
                if (
                    not selection["score_mature"]
                    or len(records) != 28
                    or len(selection["pair_records"]) != 28
                    or selection["completed_intervals"] != len(history)
                ):
                    raise QuantError("All 28 pairs must survive each mature assessment.")
                for first, second in itertools.combinations(range(8), 2):
                    names = tuple(sorted((rules.EXPERT_IDS[first], rules.EXPERT_IDS[second])))
                    row = records[names]
                    weights = [0.5 if i in (first, second) else 0.0 for i in range(8)]
                    losses = []
                    for month in values:
                        pair = float((Fraction(month[first]) + Fraction(month[second])) / 2)
                        losses.append(math.log(max(month) / pair))
                    changes = [
                        math.fsum((weights[i] - previous[i]) * matrix[i][j] for i in range(8))
                        for j in range(len(matrix[0]))
                    ]
                    changed = sum(abs(x) > 1e-12 for x in changes)
                    cost = 5 * math.fsum(abs(x) for x in changes) + changed
                    penalty = -math.log1p(-cost / 10000)
                    score = max(losses) + penalty
                    if (
                        row["expert_weights"] != weights
                        or row["switch_changed_asset_coordinates"] != changed
                    ):
                        raise QuantError("A pair coordinate or cost count changed.")
                    if len(row["monthly_opportunity_losses"]) != len(losses):
                        raise QuantError(
                            "Every mature month must contribute its recorded opportunity loss."
                        )
                    if row["worst_month_indices"] != [
                        i for i, loss in enumerate(losses) if loss == max(losses)
                    ]:
                        raise QuantError("The retained worst opportunity month indices differ.")
                    for actual, expected in zip(
                        row["monthly_opportunity_losses"], losses, strict=True
                    ):
                        _close(actual, expected)
                    for name, value in (
                        ("worst_opportunity_loss", max(losses)),
                        ("switch_cost_proxy_usd", cost),
                        ("one_time_switch_log_penalty", penalty),
                        ("raw_score", score),
                    ):
                        _close(row[name], value)
                    if row["score_key_12dp"] != round(score, 12):
                        raise QuantError(
                            "The independently ranked 12-decimal opportunity key differs."
                        )
                    candidates.append((round(score, 12), names, weights))
                key, names, expected_weights = min(candidates, key=lambda x: (x[0], x[1]))
                if (
                    selection["selected_pair"] != list(names)
                    or selection["selected_score_key_12dp"] != key
                ):
                    raise QuantError("Independent complete-menu selection disagrees.")
            if audit["expert_weights"] != expected_weights:
                raise QuantError("Statistical expert fractions changed after selection.")
            actual_target = targets.loc[day]
            if after["target_binary64_hex"] != [float(x).hex() for x in actual_target]:
                raise QuantError("Resolved target differs from the actual output matrix.")
            for j, actual in enumerate(actual_target):
                exact = sum(
                    (Fraction(expected_weights[i]) * Fraction(matrix[i][j]) for i in range(8)),
                    Fraction(0),
                )
                gap = abs(float(exact) - float(actual))
                target_gap = max(target_gap, gap)
                if gap > 1e-12:
                    raise QuantError("Independent rational unified-target composition disagrees.")
            previous = expected_weights
        if stream.readline():
            raise QuantError("Unexpected extra or partial retained decision records.")
    return {
        "verified_decisions": len(audits),
        "verified_actual_input_resolved_records": 2 * len(audits),
        "mature_decisions": mature,
        "unscored_decisions": len(audits) - mature,
        "maximum_independent_scalar_relative_difference": relative_gap,
        "maximum_independent_rational_target_difference": target_gap,
        "all_28_pairs_per_mature_decision_verified": True,
        "source_generator_or_old_account_replayed": False,
    }


def prepare(data, experts, frozen):
    if PREPARED.exists():
        saved = read_json(PREPARED)
        if saved["frozen_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting prior opportunity targets.")
        for row in saved["prepared"].values():
            base.checked_reference(row)
        return saved["prepared"]
    if len(events("registered")) != 1 or (FOLDER / "preparation-attempt-v1.json").exists():
        raise QuantError("Register once before new scores and preserve partial preparations.")
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
    proof = verify_retained(log, experts, targets, audits)
    if (
        len(audits) != 121
        or proof["mature_decisions"] != 119
        or proof["unscored_decisions"] != 2
        or len(audits[-1]["completed_intervals"]) != 119
    ):
        raise QuantError("Keep all original decisions and every complete matured source month.")
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
            raise QuantError("Future observations changed a prior opportunity allocation.")
        prefixes.append(str(day.date()))
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
        raise QuantError("Stored target binary64 values must round-trip exactly.")
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
        raise QuantError("Export every original funded path, not selected passing metrics.")
    controls = read_json(base.checked_reference(reference(pool_source.PUBLIC)))
    body = {
        "schema_version": 1,
        "study_id": read_json(rules.POLICY)["study_id"],
        "candidate_id": ID,
        "completed_at": saved["completed_at"],
        "status": saved["review"]["status"],
        "registered_event_sha256": events("registered")[0]["sha256"],
        "paths": saved["paths"],
        "cached_eight_controls": controls["candidates"],
        "source_experts_shadows_controls_or_accounts_replayed": False,
        "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
        "study_specific_same_eight_original_seven_unchanged": True,
        "new_source_requests": 0,
        "raw_source_daily_rows_or_targets_exported": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    source.common.retain_receipt(PUBLIC, body)
    return reference(PUBLIC)


def main():
    parser = argparse.ArgumentParser(description="Frozen monthly expert opportunity research only.")
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
