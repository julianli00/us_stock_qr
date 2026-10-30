from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from fractions import Fraction
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import expert_contextual_contest_v1 as rules
from us_quant import expert_initial_monthly_execution_v1 as execution
from us_quant import ssd_expert_hierarchical_risk_v1 as import_template
from us_quant import ssd_four_expert_meta_v1 as participation
from us_quant import ssd_six_expert_meta_v1 as source
from us_quant.calendar import is_month_end, next_session, previous_session, sessions
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import DEADLINE, research_job, runtime_guard
from us_quant.research_weekly import verify_publication
from us_quant.storage import (
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_text_atomic,
)

base, common, current, ROOT = source.base, source.common, source.current, source.ROOT
FOLDER = base.REPORTS / "expert-contextual-contest-20261030-v1"
INTENT = FOLDER / "research-intent-v1.json"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
TARGETS = ROOT / "data/ssd-expert-contextual-contest-20261030-v1"
PREPARED = FOLDER / "prepared-target-bundle.json"
PACKET = ROOT / "evidence/expert_contextual_contest_20261030_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/expert_contextual_contest_20261030_results_v1.json"
IDS = tuple(candidate["id"] for candidate in rules.CANDIDATES)
OWN = (
    "config/expert-contextual-contest-v1.json",
    "src/us_quant/expert_contextual_contest_v1.py",
    "src/us_quant/ssd_expert_contextual_contest_v1.py",
    "tests/test_expert_contextual_contest_v1.py",
    "tests/test_ssd_expert_contextual_contest_v1.py",
)
SHARED_CODE = "src/us_quant/strategy_expert_pool_v4.py"
PARENT_SPEC = "3e0ad10d2c93fe24bf2ed09440272afb90d8add407adc8a2c13ae6c2c6c1d290"
reference, unique_refs = source.reference, source.unique_refs
append_event, matching_events = source.append_event, source.matching_events
check_frozen, published_inputs, frame = source.check_frozen, source.published_inputs, source.frame


def dependency_inputs():
    traversal = FunctionType(import_template.dependency_inputs.__code__, {**globals(), "ast": ast})
    return traversal()


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    design = read_json(DESIGN)
    packet = read_json(base.checked_reference(policy["source_registration"]))
    parent = packet["entries"][0]["specification"]["frozen"]
    if (
        digest_json(parent) != PARENT_SPEC
        or design["policy"] != reference(rules.POLICY)
        or design["actual_invocation"] != reference(INVOCATION)
        or design["candidates"] != list(rules.CANDIDATES)
        or design["source_specification_sha256"] != PARENT_SPEC
        or design["source_price_contexts_month_credit_or_new_allocation_weights_computed"]
        is not False
        or design["candidate_accounts_or_metrics_computed"] is not False
    ):
        raise QuantError("Keep the actual pre-outcome context design and exact six-expert source.")
    check_frozen(parent)
    source.checked_packet(parent)
    data, experts, old_policy, _, old_refs, clocks, admission_refs, benchmarks = (
        source.source_inputs()
    )
    if any(
        policy[key] != old_policy[key]
        for key in (
            "expert_ids",
            "mechanism_groups",
            "lookback_sessions",
            "initial_decision_session",
            "actual_weight_trade_band",
            "invested_target_budget",
            "cash_reserve",
        )
    ):
        raise QuantError(
            "Change the allocation law,not valid experts,source maturity or execution."
        )
    controls = read_json(base.checked_reference(policy["source_controls"]))
    if (
        controls["registration_packet"] != policy["source_registration"]
        or {row["candidate_id"] for row in controls["reviews"]} != set(source.IDS)
        or any(row["status"] != "rejected" or len(row["paths"]) != 4 for row in controls["reviews"])
    ):
        raise QuantError(
            "Preserve all original exposed six-expert controls and financial rejections."
        )
    refs = unique_refs(
        [
            *old_refs,
            *admission_refs,
            *design["source_references"],
            reference(source.INTENT),
            *[
                policy[key]
                for key in (
                    "source_registration",
                    "source_controls",
                    "previous_growth_results",
                    "prior_terminal_reversion_failure",
                    "prior_local_numerical_evidence",
                )
            ],
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    if file_digest(ROOT / "runtime/research-program.sqlite3") != base.LEGACY_SHA256:
        raise QuantError("The original immutable ledger changed.")
    return data, experts, policy, parent, refs, clocks, benchmarks


def validation_references():
    xml = FOLDER / "offline-study-tests-v1.xml"
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    names = {
        case.get("name", "").split("[", 1)[0] for s in suites for case in s.findall("testcase")
    }
    required = {
        "test_fixed_contextual_rules_are_not_a_repaired_wealth_integral",
        "test_exact_context_and_zero_relative_return_ties",
        "test_exact_month_keys_and_integer_tied_credit",
        "test_positive_posterior_prior_and_context_counts",
        "test_joint_loss_tilt_and_no_loss_information_are_explicit",
        "test_initial_month_clock_maturity_and_persistent_counts",
        "test_future_prices_targets_and_returns_preserve_prior_outputs",
        "test_independent_integer_counts_and_decimal_risk_composition",
        "test_independent_verifier_rejects_context_credit_loss_clock_and_weight_tampering",
        "test_inputs_are_saved_before_a_failing_credit_or_risk_update",
        "test_cached_sources_never_call_closed_entrypoints",
        "test_complete_import_closure_is_bound_without_mutating_the_template",
        "test_publication_checks_all_dependencies_before_fence",
        "test_failed_publication_cannot_start_main_capital",
        "test_canonical_export_preserves_original_eight_paths_without_private_rows",
        "test_original_monthly_execution_is_preserved",
    }
    if not required.issubset(names) or any(
        int(suite.get(key, 0)) for suite in suites for key in ("failures", "errors", "skipped")
    ):
        raise QuantError("Require completed context,credit,causality,risk and publication checks.")
    outputs = [reference(xml), reference(FOLDER / "offline-study-tests-v1.log")]
    receipt = FOLDER / "validation-source-receipt-v1.json"
    proof = read_json(receipt)
    if (
        proof["implementation_and_test_inputs"]
        != [
            reference(ROOT / path)
            for path in (*OWN, "tests/test_expert_initial_monthly_execution_v1.py")
        ]
        or proof["canonical_test_outputs"] != outputs
        or proof["passed_test_count"] != sum(int(suite.get("tests", 0)) for suite in suites)
        or proof["exit_code"] != 0
    ):
        raise QuantError("Validation must bind the exact current implementation and test bytes.")
    return [*outputs, reference(receipt)]


def initialize():
    runtime_guard()
    if INTENT.exists():
        raise QuantError("Preserve a complete or partial implementation freeze.")
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        _, _, _, parent, refs, clocks, _ = source_inputs()
        events = base.journal_events(base.JOURNAL)
        counts = current.current_counts(events)
        invocation = read_json(INVOCATION)
        if (
            len(events) != 560
            or events[-1]["sha256"] != rules.PRIOR_HEAD
            or counts["new_registered_candidates"] != 54
            or counts["new_completed_strategy_evaluations"] != 46
            or counts["new_technical_failures"] != 8
            or counts["new_pending_candidates"] != 0
            or invocation["data"]["source"] != "system"
            or invocation["data"]["delivery"] != "idle"
            or read_json(DESIGN)["actual_continuation_event"] != 54426
        ):
            raise QuantError("Freeze against the actual unchanged journal and scheduled origin.")
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "actual_continuation_event": 54426,
                "actual_delivery": "system_idle",
                "design": reference(DESIGN),
                "actual_invocation": reference(INVOCATION),
                "model_and_test_files": [reference(ROOT / path) for path in OWN],
                "validation_evidence": validation_references(),
                "source_references": refs,
                "source_specification_sha256": digest_json(parent),
                "source_policy_and_expert_admission_clocks": clocks,
                "real_new_contexts_credits_allocations_or_targets_computed": False,
                "candidate_accounts_or_metrics_computed": False,
                "source_expert_targets_fits_or_accounts_replayed": False,
                "new_shadow_account_runs": 0,
                "new_source_requests": 0,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        current.progress(
            "expert_contextual_contest_frozen",
            active_compute_job=None,
            expert_contextual_contest_intent=reference(INTENT),
            pending_steps=["register_both_rules_before_real_context_credit_or_allocation_updates"],
        )
        return reference(INTENT)


def inputs():
    data, experts, policy, parent, refs, clocks, benchmarks = source_inputs()
    intent = read_json(INTENT)
    if (
        intent["model_and_test_files"] != [reference(ROOT / path) for path in OWN]
        or intent["validation_evidence"] != validation_references()
        or intent["source_references"] != refs
        or intent["source_specification_sha256"] != PARENT_SPEC
        or intent["source_policy_and_expert_admission_clocks"] != clocks
        or intent["design"] != reference(DESIGN)
        or intent["actual_invocation"] != reference(INVOCATION)
        or intent["real_new_contexts_credits_allocations_or_targets_computed"] is not False
        or intent["candidate_accounts_or_metrics_computed"] is not False
    ):
        raise QuantError("The actual pre-outcome contextual implementation freeze changed.")
    code, policies = dependency_inputs()
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": reference(rules.POLICY),
        "code": unique_refs(
            [*parent["code"], *code, *[reference(ROOT / path) for path in OWN[1:]]]
        ),
        "policies": unique_refs([*parent["policies"], *policies, reference(rules.POLICY)]),
        "selection_policy": reference(rules.POLICY),
        "expert_pool_definition_role": "unchanged_six_representatives_categorical_context_contest",
        "source_policy_and_expert_admission_clocks": clocks,
        "prior_completed_configurations": 224,
        "prior_journal_sequence": 560,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "construction": "unconditional_vs_contextual_categorical_counts_same_co_loss_tilt",
        "archive_import_dependencies_are_additional_strategy_experts": False,
        "public_archive_policy_pins_are_additional_active_allocators": False,
        "prior_evidence": unique_refs(
            [
                *parent["prior_evidence"],
                *refs,
                reference(DESIGN),
                reference(INVOCATION),
                reference(INTENT),
            ]
        ),
        "validation_evidence": unique_refs(
            [*parent["validation_evidence"], *validation_references()]
        ),
        "new_shadow_account_runs": 0,
        "new_shadow_recommendation_artifacts": 0,
        "new_source_requests": 0,
    }
    check_frozen(frozen)
    return data, experts, frozen, benchmarks


def close_array(actual, expected, message, tolerance=rules.TOLERANCE):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if (
        a.shape != b.shape
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or float(abs(a - b).max(initial=0)) > tolerance
    ):
        raise QuantError(message)


def scalar(value):
    numerator, denominator = float(value).as_integer_ratio()
    return Fraction(numerator, denominator)


def encode_fraction(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def independent_context(data, day):
    position = data.close.index.get_loc(day)
    if position < 126:
        raise QuantError("The independent context lacks127prior admitted closes.")
    previous = data.close.index[position - 126]
    record = {
        "as_of": str(day.date()),
        "earlier_session": str(previous.date()),
        "earlier_prices": {s: float(data.close.loc[previous, s]) for s in ("SPY", "IEF", "BIL")},
        "current_prices": {s: float(data.close.loc[day, s]) for s in ("SPY", "IEF", "BIL")},
    }
    bits = []
    for symbol in ("SPY", "IEF"):
        a, b = record["current_prices"][symbol].as_integer_ratio()
        c, d = record["earlier_prices"]["BIL"].as_integer_ratio()
        e, f = record["current_prices"]["BIL"].as_integer_ratio()
        g, h = record["earlier_prices"][symbol].as_integer_ratio()
        bits.append(a * c * f * h > e * g * b * d)
    return 2 * int(bits[0]) + int(bits[1]), record


def independent_credit(observed, known):
    keys = []
    for item in observed:
        numerator = denominator = 1
        for value in item.net_returns.loc[known]:
            n, d = float(value).as_integer_ratio()
            numerator *= n + d
            denominator *= d
            divisor = math.gcd(numerator, denominator)
            numerator //= divisor
            denominator //= divisor
        key, remainder = divmod(numerator * 10**12, denominator)
        if 2 * remainder > denominator or 2 * remainder == denominator and key % 2:
            key += 1
        keys.append(key)
    winners = [i for i in range(6) if keys[i] == max(keys)]
    return keys, winners, [60 // len(winners) if i in winners else 0 for i in range(6)]


def independent_risk(observed, fractions):
    losses = [Fraction(0)] * 6
    selected = []
    for row in range(126):
        x = [
            scalar(item.net_returns.iloc[row]) - scalar(item.risk_free.iloc[row])
            for item in observed
        ]
        reference_return = sum((fractions[i] * x[i] for i in range(6)), Fraction(0))
        if reference_return < 0:
            selected.append(str(observed[0].net_returns.index[row].date()))
            losses = [losses[i] + max(-x[i], Fraction(0)) for i in range(6)]
    if selected:
        losses = [value / len(selected) for value in losses]
        reference = sum((fractions[i] * losses[i] for i in range(6)), Fraction(0))
        if reference <= 0:
            raise QuantError("The independent common-loss reference is not positive.")
        ratios = [value / reference for value in losses]
        reason = "observed_joint_excess_loss_tilt"
    else:
        reference = Fraction(0)
        ratios = [Fraction(0)] * 6
        reason = "no_common_excess_loss_dates_posterior_unchanged_no_safety_claim"
    with localcontext(Context(prec=50, rounding=ROUND_HALF_EVEN)):
        raw = []
        for q, ratio in zip(fractions, ratios, strict=True):
            probability = Decimal(q.numerator) / Decimal(q.denominator)
            exponent = Decimal(ratio.numerator) / Decimal(ratio.denominator)
            raw.append(probability * (-exponent).exp())
        mass = sum(raw, Decimal(0))
        weights = np.array([float(value / mass) for value in raw])
    return weights, {
        "posterior_rational_fractions": [encode_fraction(value) for value in fractions],
        "common_loss_source_sessions": selected,
        "component_nonoffsetting_loss_means": [encode_fraction(value) for value in losses],
        "reference_nonoffsetting_loss": encode_fraction(reference),
        "component_relative_losses": [encode_fraction(value) for value in ratios],
        "risk_tilt_reason": reason,
        "source_risk_proxy_is_main_loss_or_guarantee": False,
        "positive_hedge_gains_credited_in_proxy": False,
    }


def learning_records(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def verify_targets(data, experts, targets, audits, records):
    dates = rules.decision_sessions(data.close.index, read_json(rules.POLICY))
    if (
        set(targets) != set(IDS)
        or len(audits) != len(dates)
        or any(
            not target.index.equals(data.close.index)
            or not target.columns.equals(data.close.columns)
            or not target.dropna(how="all").index.equals(dates)
            for target in targets.values()
        )
    ):
        raise QuantError("The full original initial/month-end target calendar changed.")
    iterator = iter(records)
    counts = [[0] * 6 for _ in range(4)]
    months = rows = 0
    previous_day = previous_context = None
    previous_main = {name: np.full(6, 1 / 6) for name in IDS}
    maximum_weight = maximum_target = 0.0
    for position, (day, audit) in enumerate(zip(dates, audits, strict=True)):
        observed = [expert.observe(day, 126) for expert in experts]
        if (
            tuple(item.identity.identifier for item in observed) != rules.EXPERT_IDS
            or tuple(item.identity.mechanism_group for item in observed) != rules.MECHANISMS
            or any(
                item.as_of != day
                or item.target is None
                or item.target_known_at is None
                or item.target_known_at > day
                or len(item.net_returns) > 126
                or item.score_mature != (len(item.net_returns) == 126)
                or not item.net_returns.index.equals(item.risk_free.index)
                or not np.isfinite(item.net_returns).all()
                or not np.isfinite(item.risk_free).all()
                or (item.net_returns <= -1).any()
                for item in observed
            )
            or any(
                not item.net_returns.index.equals(observed[0].net_returns.index)
                or not item.risk_free.equals(observed[0].risk_free)
                for item in observed
            )
        ):
            raise QuantError("Independent source identities,availability or maturity disagree.")
        context, context_record = independent_context(data, day)
        entry = next(iterator, None)
        if (
            entry is None
            or entry["stage"] != "input"
            or entry["session"] != str(day.date())
            or entry["context_input"] != context_record
            or entry["previous_decision"]
            != (None if previous_day is None else str(previous_day.date()))
            or entry["previous_context"] != previous_context
            or entry["credit_state_before"] != counts
            or entry["complete_months_before"] != months
            or entry["source_return_sessions"]
            != observed[0].net_returns.index.strftime("%Y-%m-%d").tolist()
            or entry["source_NET_returns"] != [item.net_returns.tolist() for item in observed]
            or entry["source_RF"] != observed[0].risk_free.tolist()
        ):
            raise QuantError("The retained pre-update context or categorical operands changed.")
        rows += 1
        update = None
        partial = previous_day is not None and not is_month_end(previous_day)
        if previous_day is not None and is_month_end(previous_day):
            known = sessions(next_session(previous_day), day)
            keys, winners, units = independent_credit(observed, known)
            update = {
                "winner_relative_keys": keys,
                "winner_indices": winners,
                "credit_units": units,
                "previous_context": previous_context,
                "first_source_session": str(known[0].date()),
                "last_source_session": str(known[-1].date()),
                "source_sessions": len(known),
            }
            counts[previous_context] = [counts[previous_context][i] + units[i] for i in range(6)]
            months += 1
        mature = all(len(item.net_returns) == 126 for item in observed)
        if (
            audit["decision_session"] != str(day.date())
            or audit["decision_role"]
            != ("initial_capital" if position == 0 else "completed_exchange_month_end")
            or audit["context_input"] != context_record
            or audit["current_context"] != context
            or audit["initial_partial_source_month_not_assimilated"] is not partial
            or audit["monthly_credit_update"] != update
            or audit["credit_state_after"] != counts
            or audit["complete_learning_months"] != months
            or audit["source_observation_counts"] != [len(item.net_returns) for item in observed]
            or audit["source_assessment_mature"] is not mature
            or audit["expert_target_known_at"]
            != [str(item.target_known_at.date()) for item in observed]
            or audit["preserved_archive_identity_introduced_at"]
            != [item.identity.introduced_at for item in observed]
            or audit["learning_state_is_main_wealth"] is not False
            or audit["source_shadow_wealth_or_protection_floor_transferred"] is not False
            or audit["counterfactual_historical_expert_availability"] is not True
            or audit["independent_forward_validation"] is not False
        ):
            raise QuantError(
                "Independent context,monthly credit,clock or capital separation changed."
            )
        for index, name in enumerate(IDS):
            where = f"{day.date()}:{name}"
            numerator = (
                [sum(row[i] for row in counts) for i in range(6)] if index == 0 else counts[context]
            )
            fraction = tuple(Fraction(value + 60, sum(numerator) + 360) for value in numerator)
            allocation = audit["allocations"][name]
            if (
                allocation["counted_complete_months"] != sum(numerator) // 60
                or allocation["categorical_forecast_is_prior_only"] is not (sum(numerator) == 0)
                or allocation["posterior_rational_fractions"]
                != [encode_fraction(q) for q in fraction]
                or allocation["reason"]
                != (
                    "mature_categorical_and_joint_loss_assessment"
                    if mature
                    else "unscored_initial_equal_weights"
                )
            ):
                raise QuantError(
                    "The independent categorical posterior or prior-only label changed."
                )
            if mature:
                weights, risk = independent_risk(observed, fraction)
                actual_risk = dict(allocation["co_loss_assessment"])
                close_array(
                    actual_risk.pop("expert_weights_after_tilt"),
                    weights,
                    f"{where}:independent fixed50digit loss weights disagree.",
                )
                if actual_risk != risk:
                    raise QuantError(f"{where}:exact common-loss dates,means or ratios disagree.")
            else:
                weights = np.full(6, 1 / 6)
                if allocation["co_loss_assessment"] is not None:
                    raise QuantError("Unmatured source history cannot fabricate a loss assessment.")
            close_array(
                allocation["previous_applied_expert_weights"],
                previous_main[name],
                "Previous main intentions changed.",
            )
            close_array(
                allocation["applied_expert_weights"],
                weights,
                f"{where}:independent fractions disagree.",
            )
            maximum_weight = max(
                maximum_weight,
                float(abs(np.asarray(allocation["applied_expert_weights"]) - weights).max()),
            )
            expected = [
                math.fsum(float(weights[i]) * float(observed[i].target[s]) for i in range(6))
                for s in data.close.columns
            ]
            actual = targets[name].loc[day].to_numpy()
            close_array(actual, expected, f"{where}:independent underlying target disagrees.")
            maximum_target = max(maximum_target, float(abs(actual - expected).max()))
            allocated = next(iterator, None)
            if allocated != {
                "stage": "allocation_resolved",
                "session": str(day.date()),
                "candidate_id": name,
                "allocation": allocation,
                "unified_target": actual.tolist(),
            }:
                raise QuantError(f"{where}:resolved allocation operands were not retained.")
            rows += 1
            previous_main[name] = weights.copy()
        resolved = next(iterator, None)
        if resolved != {"stage": "resolved", "session": str(day.date()), "audit": audit}:
            raise QuantError("The retained resolved state changed before independent verification.")
        rows += 1
        previous_day, previous_context = day, context
    if next(iterator, None) is not None:
        raise QuantError("Extra contextual learning records cannot be hidden.")
    return {
        "all_monthly_decisions_checked": len(audits),
        "complete_source_months_checked": months,
        "unscored_initial_decisions": sum(
            not audit["source_assessment_mature"] for audit in audits
        ),
        "retained_input_output_records_checked": rows,
        "maximum_independent_expert_weight_difference": maximum_weight,
        "maximum_independent_scalar_target_difference": maximum_target,
        "context_and_credit_integer_keys_counts_and_loss_assessment_exact": True,
        "independent_risk_transform_significant_decimal_digits": 50,
        "main_account_or_old_source_pipeline_replayed": False,
    }


def prepare(data, experts, frozen):
    if PREPARED.exists():
        saved = read_json(PREPARED)
        if saved["frozen_specification_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting prior preparation rather than regenerate it.")
        for entry in saved["entries"].values():
            for ref in entry.values():
                base.checked_reference(ref)
        return saved["entries"]
    attempt = FOLDER / "preparation-attempt-v1.json"
    if attempt.exists():
        raise QuantError("Preserve an unresolved preparation;no silent retry.")
    write_new_receipt(attempt, {"started_at": utc_now(), "frozen_sha256": digest_json(frozen)})
    new_output_directory(TARGETS)
    learning = TARGETS / "learning-input-output-records.jsonl"
    with learning.open("x") as stream:

        def retain(row):
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()

        targets, audits = rules.build_targets(
            experts,
            data.close,
            read_json(rules.POLICY),
            before_decision=runtime_guard,
            record_learning=retain,
        )
    write_new_receipt(TARGETS / "information-audits.json", {"decisions": audits})
    for name in IDS:
        new_output_directory(TARGETS / name)
        write_text_atomic(
            TARGETS / name / "targets.csv", targets[name].to_csv(float_format="%.17g")
        )
    independent = verify_targets(data, experts, targets, audits, learning_records(learning))
    if (
        len(audits) != 121
        or independent["complete_source_months_checked"] != 119
        or independent["unscored_initial_decisions"] != 7
    ):
        raise QuantError(
            "Retain all original decisions,complete source months and explicit maturity."
        )
    prefixes = []
    for position in (0, 6, 12, 60, 120):
        runtime_guard()
        day = pd.Timestamp(audits[position]["decision_session"])
        early, earlier = rules.build_targets(
            experts,
            data.close.loc[:day],
            read_json(rules.POLICY),
            before_decision=runtime_guard,
        )
        if (
            any(not early[n].equals(targets[n].loc[:day]) for n in IDS)
            or earlier != audits[: position + 1]
        ):
            raise QuantError(
                "Future prices or source observations changed a prior contextual allocation."
            )
        prefixes.append(str(day.date()))
    entries = {}
    for name in IDS:
        coverage = []
        for window in read_json(base.PROTOCOL)["windows"]:
            anchor = previous_session(window["start"])
            known = targets[name].ffill().loc[anchor : window["end"]]
            active = [
                symbol
                for symbol in participation.source.donor.rules.CORE
                if (known[symbol] > 1e-10).any()
            ]
            if anchor not in known.index or known.isna().any().any() or len(active) < 3:
                raise QuantError(
                    "Keep every formal anchor and measured registered-factor coverage."
                )
            coverage.append({"years": window["years"], "actual_factor_symbols": active})
        directory = TARGETS / name
        write_new_receipt(
            directory / "proof.json",
            {
                "checked_at": utc_now(),
                "frozen_specification_sha256": digest_json(frozen),
                "decision_count": len(audits),
                **independent,
                "learning_records": reference(learning),
                "learning_inputs_saved_before_updates": True,
                "information_audits": reference(TARGETS / "information-audits.json"),
                "prefix_truncations": prefixes,
                "formal_factor_coverage": coverage,
                "source_policy_and_expert_admission_clocks": frozen[
                    "source_policy_and_expert_admission_clocks"
                ],
                "candidate_accounts_or_metrics_computed": False,
                "candidate_metrics_scope": (
                    "formal main paths,not posterior counts or co-loss statistics"
                ),
                "original_source_signal_recomputed": False,
                "new_shadow_account_runs": 0,
                "new_shadow_recommendation_artifacts": 0,
                "independent_forward_validation": False,
            },
        )
        entries[name] = {
            "targets": reference(directory / "targets.csv"),
            "proof": reference(directory / "proof.json"),
        }
    check_frozen(frozen)
    write_new_receipt(
        PREPARED, {"frozen_specification_sha256": digest_json(frozen), "entries": entries}
    )
    return entries


def checked_packet(frozen):
    packet = read_json(PACKET)
    entries = [common.packet_entry(candidate, frozen) for candidate in rules.CANDIDATES]
    created = pd.Timestamp(packet["created_at"])
    if (
        packet["schema_version"] != 2
        or packet["program_id"] != "ssd_continuous_research_v1"
        or packet["study_id"] != frozen["study_id"]
        or packet["entries"] != entries
        or packet["new_registered_candidates"] != 2
        or created.tzinfo is None
        or pd.isna(created)
        or created > pd.Timestamp(utc_now())
        or any(pd.Timestamp(entry["target_proof_at"]) > created for entry in entries)
        or any(
            created > pd.Timestamp(event["at"])
            for name in IDS
            for event in matching_events("path_started", name)
        )
        or any(
            packet.get(key) is not False
            for key in (
                "candidate_accounts_or_metrics_computed",
                "investment_objective_verified",
                "independent_forward_validation",
                "order_authority",
            )
        )
    ):
        raise QuantError("The complete actual pre-account contextual registration packet changed.")
    return packet


def failure(stage, identifiers, error):
    expired = pd.Timestamp(utc_now()) >= pd.Timestamp(DEADLINE)
    if not expired:
        for name in identifiers:
            if (
                matching_events("registered", name)
                and not matching_events("review", name)
                and not matching_events("technical_failure", name)
            ):
                append_event(
                    base.JOURNAL,
                    "technical_failure",
                    name,
                    {
                        "stage": stage,
                        "reason": str(error),
                        "partial_outputs_retained": True,
                        "candidate_accounts_started": bool(matching_events("path_started", name)),
                    },
                )
    current.progress(
        "expert_contextual_contest_expired" if expired else "expert_contextual_contest_blocked",
        active_compute_job=None,
        active_candidate=None,
        active_path=None,
        current_contextual_contest_failure=str(error),
        pending_steps=["preserve_terminal_or_interrupted_contextual_version_without_retry"],
    )


def register():
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, experts, frozen, _ = inputs()
        old = [
            event
            for event in base.journal_events(base.JOURNAL)
            if event["candidate_id"] not in IDS
            and not (
                event["kind"] == "publication"
                and event["body"].get("study_id") == frozen["study_id"]
            )
        ]
        if len(old) != 560 or old[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError("Preserve every old completed and failed financial identity.")
        if any(matching_events("technical_failure", name) for name in IDS):
            raise QuantError("A terminal registered version cannot be silently retried.")
        for candidate in rules.CANDIDATES:
            found = matching_events("registered", candidate["id"])
            body = {"candidate": candidate, "frozen": frozen}
            if found and (len(found) != 1 or found[0]["body"] != body):
                raise QuantError("Registered code,policy and inputs cannot change.")
            if not found:
                runtime_guard()
                append_event(base.JOURNAL, "registered", candidate["id"], body)
        pending = [name for name in IDS if not matching_events("target_proof", name)]
        try:
            if pending:
                current.progress(
                    "preparing_registered_expert_contextual_contest",
                    active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
                )
                prepared = prepare(data, experts, frozen)
                for name in pending:
                    runtime_guard()
                    append_event(
                        base.JOURNAL,
                        "target_proof",
                        name,
                        {
                            "registered_event_sha256": matching_events("registered", name)[0][
                                "sha256"
                            ],
                            **prepared[name],
                        },
                    )
            common.retain_receipt(
                PACKET,
                {
                    "schema_version": 2,
                    "program_id": "ssd_continuous_research_v1",
                    "study_id": frozen["study_id"],
                    "created_at": utc_now(),
                    "entries": [
                        common.packet_entry(candidate, frozen) for candidate in rules.CANDIDATES
                    ],
                    "new_registered_candidates": 2,
                    "new_economic_hypotheses": 2,
                    "unconditional_control_and_contextual_rule_share_other_laws": True,
                    "candidate_accounts_or_metrics_computed": False,
                    "candidate_metrics_scope": (
                        "formal main paths,not posterior counts or co-loss statistics"
                    ),
                    "raw_inputs_and_target_matrices_remain_private": True,
                    "investment_objective_verified": False,
                    "independent_forward_validation": False,
                    "order_authority": False,
                },
                ("created_at",),
            )
            checked_packet(frozen)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("expert_contextual_contest_preparation", pending, error)
            raise
        except KeyboardInterrupt:
            current.progress(
                "expert_contextual_contest_preparation_interrupted", active_compute_job=None
            )
            raise
        current.progress(
            "expert_contextual_contest_registered_waiting_publication",
            active_compute_job=None,
            expert_contextual_contest_registration=reference(PACKET),
            pending_steps=["publish_exact_contextual_packet_before_all_eight_accounts"],
        )
        return reference(PACKET)


def publication(frozen, commit):
    dependencies = published_inputs(frozen, commit)
    contract = read_json(base.REPO / "config/publication_allowlist.json")
    explicit = {
        ref["path"]
        for ref in frozen["code"] + frozen["policies"]
        if f"research/workbench/{ref['path']}" in contract["files"]
    }
    own = verify_publication(ROOT, reference(PACKET), explicit, commit)
    found = [
        event
        for event in matching_events("publication")
        if event["body"].get("study_id") == frozen["study_id"]
    ]
    if found:
        if len(found) != 1 or found[0]["body"]["publication"]["published_commit"] != commit:
            raise QuantError("Resume the original publication fence,not a later patch.")
        return found[0]["body"]["publication"]
    receipt = {**own, "complete_dependency_publication": dependencies}
    append_event(
        base.JOURNAL, "publication", None, {"study_id": frozen["study_id"], "publication": receipt}
    )
    return receipt


def states(frozen):
    result = {}
    for name in IDS:
        reviews, failed, completed = (
            matching_events(kind, name)
            for kind in ("review", "technical_failure", "path_completed")
        )
        if (
            len({event["body"]["path_key"] for event in completed}) != len(completed)
            or len(reviews) > 1
            or len(failed) > 1
        ):
            raise QuantError(
                "Duplicate completion or terminal events cannot select another winner."
            )
        paths = {
            event["body"]["path_key"]: base.verify_path(
                event["body"]["receipt"], frozen, name, event["body"]["path_key"]
            )
            for event in completed
        }
        if reviews:
            if (
                failed
                or set(paths) != base.PATH_KEYS
                or reviews[0]["body"]["paths"] != paths
                or reviews[0]["body"]["all_four_paths_same_candidate_and_frozen_parameters"]
                is not True
            ):
                raise QuantError("A complete review requires every actual original path.")
            result[name] = reviews[0]["body"]
        else:
            result[name] = {
                "candidate_id": name,
                "status": "technical_failure" if failed else "pending",
                "paths": paths,
                "failure_event_sha256": failed[0]["sha256"] if failed else None,
                "failure_stage": failed[0]["body"]["stage"] if failed else None,
            }
    return result


def evaluate(commit):
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, _, frozen, benchmarks = inputs()
        checked_packet(frozen)
        if RESULTS.exists():
            saved = read_json(RESULTS)
            if saved["frozen_specification_sha256"] != digest_json(frozen) or saved[
                "candidates"
            ] != states(frozen):
                raise QuantError("Read preserved outcomes instead of replaying completed accounts.")
            return reference(RESULTS)
        active = IDS
        try:
            published = publication(frozen, commit)
            current.progress(
                "evaluating_expert_contextual_contest",
                active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            )
            for name in IDS:
                active = (name,)
                if matching_events("review", name) or matching_events("technical_failure", name):
                    continue
                target = frame(matching_events("target_proof", name)[0]["body"]["targets"])
                paths = {}
                for window in read_json(base.PROTOCOL)["windows"]:
                    for scenario in read_json(base.PROTOCOL)["accounts"]["scenarios"]:
                        runtime_guard()
                        check_frozen(frozen)
                        key = f"{window['years']}y-{scenario['id']}"
                        current.progress(
                            "evaluating_expert_contextual_contest",
                            active_candidate=name,
                            active_path=key,
                        )
                        paths[key] = execution.evaluate_path(
                            data, target, frozen, name, window, scenario, benchmarks
                        )
                actual_families = participation.actual_factor_participation(paths, frozen)
                passed = all(path["all_point_gates_pass"] for path in paths.values()) and all(
                    item["minimum_three_registered_families_pass"]
                    for item in actual_families.values()
                )
                append_event(
                    base.JOURNAL,
                    "review",
                    name,
                    {
                        "candidate_id": name,
                        "reviewed_at": utc_now(),
                        "status": "historical_point_gates_passed" if passed else "rejected",
                        "paths": paths,
                        "actual_factor_participation": actual_families,
                        "all_four_paths_same_candidate_and_frozen_parameters": True,
                        "full_search_selection_adjusted_alpha_verified": False,
                        "independent_forward_validation": False,
                        "investment_objective_verified": False,
                        "automatic_live_deployment": False,
                        "order_authority": False,
                    },
                )
            events = base.journal_events(base.JOURNAL)
            write_new_receipt(
                RESULTS,
                {
                    "schema_version": 2,
                    "study_id": frozen["study_id"],
                    "finished_at": utc_now(),
                    "frozen_specification_sha256": digest_json(frozen),
                    "candidates": states(frozen),
                    "publication": published,
                    "legacy": frozen["legacy"],
                    "actual_journal_event_count": len(events),
                    "actual_journal_head_sha256": events[-1]["sha256"],
                    "global_counts_at_completion": common.counts(events),
                    "new_shadow_account_runs": 0,
                    "new_shadow_recommendation_artifacts": 0,
                    "investment_objective_verified": False,
                    "order_authority": False,
                },
            )
            current.progress(
                "expert_contextual_contest_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                expert_contextual_contest_results=reference(RESULTS),
                pending_steps=["publish_all_eight_contextual_contest_outcomes"],
            )
            return reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("expert_contextual_contest_evaluation", active, error)
            raise
        except KeyboardInterrupt:
            current.progress(
                "expert_contextual_contest_evaluation_interrupted", active_compute_job=None
            )
            raise


def public_result():
    base.storage_guard()
    _, _, frozen, benchmarks = inputs()
    checked_packet(frozen)
    actual, saved = states(frozen), read_json(RESULTS)
    if saved["candidates"] != actual or any(row["status"] == "pending" for row in actual.values()):
        raise QuantError("Publish actual complete outcomes or retained technical failures only.")
    lookup = {
        (row["years"], row["scenario"], row["symbol"]): row["metrics"]["cagr"]
        for row in benchmarks["benchmarks"]
    }
    reviews = []
    for name, row in actual.items():
        paths = []
        for key, path in row["paths"].items():
            year, scenario = key.split("-")
            years = int(year[:-1])
            paths.append(
                {
                    **{
                        field: path[field]
                        for field in (
                            "path_key",
                            "metrics",
                            "execution",
                            "gates",
                            "all_point_gates_pass",
                            "excess_net_cagr_vs_spy",
                            "excess_net_cagr_vs_qqq",
                            "stress_vs_base_cost_qqq_diagnostic",
                            "independent_bt_max_equity_difference_usd",
                            "independent_monthly_feedback",
                        )
                    },
                    "years": years,
                    "scenario": scenario,
                    "benchmark_net_cagr": {
                        symbol: lookup[years, scenario, symbol] for symbol in ("SPY", "QQQ")
                    },
                    "actual_factor_participation": row.get("actual_factor_participation", {}).get(
                        key
                    ),
                }
            )
        reviews.append(
            {
                "candidate_id": name,
                "status": row["status"],
                "paths": paths,
                "reviewed_at": row.get("reviewed_at"),
                "failure_event_sha256": row.get("failure_event_sha256"),
                "failure_stage": row.get("failure_stage"),
                "all_four_paths_same_candidate_and_frozen_parameters": row.get(
                    "all_four_paths_same_candidate_and_frozen_parameters", False
                ),
                "independent_forward_validation": False,
                "order_authority": False,
            }
        )
    counts = common.counts(
        [event for event in base.journal_events(base.JOURNAL) if event["candidate_id"] in IDS]
    )
    common.retain_receipt(
        PUBLIC_RESULTS,
        {
            "schema_version": 2,
            "study_id": frozen["study_id"],
            "created_at": utc_now(),
            "actual_completed_at": saved["finished_at"],
            "actual_journal_event_count": saved["actual_journal_event_count"],
            "actual_journal_head_sha256": saved["actual_journal_head_sha256"],
            "actual_publication_before_accounts": saved["publication"],
            "registration_packet": reference(PACKET),
            "protocol": frozen["protocol"],
            "private_complete_results": reference(RESULTS),
            "reviews": reviews,
            **counts,
            "new_completed_configurations": counts["new_completed_strategy_evaluations"],
            "global_counts_at_completion": saved["global_counts_at_completion"],
            "original_records_reclassified": False,
            "raw_prices_documents_targets_or_daily_accounts_exported": False,
            "new_shadow_account_runs": 0,
            "new_shadow_recommendation_artifacts": 0,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "order_authority": False,
            "automatic_live_deployment": False,
        },
        ("created_at",),
    )
    return reference(PUBLIC_RESULTS)


def main():
    parser = argparse.ArgumentParser(
        description="Unconditional/contextual expert contests;no trading."
    )
    parser.add_argument(
        "command", choices=("init", "register", "evaluate", "publish-result", "status")
    )
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("Contextual expert research cannot request live sources.")

    sys.addaudithook(offline)
    try:
        if args.command == "init":
            result = initialize()
        elif args.command == "register":
            result = register()
        elif args.command == "evaluate":
            if args.commit is None:
                raise QuantError("Provide the original actual preregistration publication SHA.")
            result = evaluate(args.commit)
        elif args.command == "publish-result":
            result = public_result()
        else:
            result = current.current_counts(base.journal_events(base.JOURNAL))
        print(json.dumps(result, indent=2, allow_nan=False))
    except QuantError as error:
        parser.exit(2, f"Contextual expert research blocked: {error}\n")


if __name__ == "__main__":
    main()
