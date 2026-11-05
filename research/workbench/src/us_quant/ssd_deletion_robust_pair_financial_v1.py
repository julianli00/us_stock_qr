from __future__ import annotations

import argparse
import itertools
import json
import math
from fractions import Fraction
from importlib.metadata import distributions
from types import FunctionType

import numpy as np

from us_quant import expert_deletion_robust_pair_v1 as rules
from us_quant import ssd_distinct_opinion_pair_financial_v1 as single
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.storage import read_json, utc_now, write_json

base, ROOT, source, lifecycle = single.base, single.ROOT, single.source, single.lifecycle
cached, financial = single.cached, single.financial
reference, unique_refs = single.reference, single.unique_refs
FOLDER = base.REPORTS / "deletion-robust-pair-financial-20261105-v1"
POLICY = ROOT / "config/deletion-robust-pair-financial-v1.json"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
PREPARED = FOLDER / "prepared-target-bundle-v1.json"
RESULTS = FOLDER / "results.json"
PACKET = ROOT / "evidence/deletion_robust_pair_financial_20261105_registration_v1.json"
PUBLIC = ROOT / "evidence/deletion_robust_pair_financial_20261105_results_v1.json"
ID = rules.ID
CANDIDATE = {"id": ID, "weighting": "conditional_single_return_deletion_lower_envelope"}
OWN = (
    "config/deletion-robust-pair-financial-v1.json",
    "src/us_quant/ssd_deletion_robust_pair_financial_v1.py",
    "tests/test_ssd_deletion_robust_pair_financial_v1.py",
    "DELETION_ROBUST_PAIR_FINANCIAL_V1.md",
    *rules.OWN,
    *rules.FIXTURE_INPUTS,
    "tests/conftest.py",
    "tests/test_ssd_expert_instrument_concentration_v1.py",
    "tests/test_expert_native_arrival_equal_v1.py",
)


def bound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


def events(kind):
    return [
        e
        for e in base.journal_events(base.JOURNAL)
        if e["kind"] == kind and e["candidate_id"] == ID
    ]


def check_frozen(frozen):
    single.check_frozen(frozen)


def dependencies():
    return bound(cached.previous.dependencies)()


def progress(phase, **fields):
    counts = current_counts(base.journal_events(base.JOURNAL))
    if any(k in fields and fields[k] != v for k, v in counts.items()):
        raise QuantError("Robust financial progress cannot override actual journal counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "deletion_robust_financial_" + phase.removeprefix("consensus_"),
            "study_id": read_json(POLICY)["study_id"],
            "active_compute_job": None,
            **counts,
            **fields,
        },
    )


def check_primary():
    policy = read_json(POLICY)
    expected = {
        "schema_version": 1,
        "study_id": "ssd_deletion_robust_pair_financial_20261105_v1",
        "candidate_id": ID,
        "primary_policy": {
            "path": "config/expert-deletion-robust-pair-v1.json",
            "sha256": "cf4c0451ec94c57a1e7c7f478c28c849172509c24234e238dfab8550359b3c51",
        },
        "primary_freeze": {
            "path": (
                "reports/ssd-continuous-research-v1/deletion-robust-pair-primary-20261105-v1/"
                "primary-implementation-freeze-v1.json"
            ),
            "sha256": "98bd1738e63ab6783f302547d890a658fb5ce12c4c6815f1cdce54a8918509af",
        },
        "primary_completion": {
            "path": (
                "reports/ssd-continuous-research-v1/deletion-robust-pair-primary-20261105-v1/"
                "completion-v1.json"
            ),
            "sha256": "eec842ccbfe77de221b2a2f64ca4c1e90a38262476ee3fa18e467cdab12d7288",
        },
        "cached_same_nine_results": {
            "path": "evidence/lagged_spy_price_impact_expert_inclusion_20261105_results_v1.json",
            "sha256": "5e0a59929a2e4166c0227478a4e1d9fcb1a41eb3a97b42ebdbc2a7c50961952b",
        },
        "original_financial_protocol": {
            "path": "config/dual-benchmark-v1.json",
            "sha256": "1b5b2572089ffa53746b01e72ff8303e00d6b5e5e3f815844f93834d0521e469",
        },
        "prior_journal_sequence": 1074,
        "prior_journal_head_sha256": (
            "7a2074649982220a74e62d2339527d5353eb7138f671992a58d3755114f3b050"
        ),
        "prior_completed_configurations": 268,
        "candidate_count": 1,
        "static_target_packet_schema": 2,
        "new_source_requests": 0,
        "new_shadow_account_runs": 0,
        "global_pool_changed": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if policy != expected or any(type(policy[k]) is not type(v) for k, v in expected.items()):
        raise QuantError("Use the unchanged98bd primary and original financial protocol.")
    for k in (
        "primary_policy",
        "primary_freeze",
        "primary_completion",
        "cached_same_nine_results",
        "original_financial_protocol",
    ):
        base.checked_reference(policy[k])
    frozen = read_json(base.checked_reference(policy["primary_freeze"]))
    for ref in frozen["implementation_and_test_inputs"] + frozen["complete_dependency_inputs"]:
        base.checked_reference(ref)
    if (
        frozen["actual_registered"] is not False
        or frozen["real_new_policy_targets_scores_accounts_or_financial_outcomes_computed"]
        is not False
    ):
        raise QuantError("Preserve the true preparation-only historical primary snapshot.")
    rules.validate(read_json(rules.POLICY))
    return policy


def source_inputs():
    return bound(single.source_inputs)()


def validation_refs():
    return bound(single.validation_refs)()


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the first complete or partial financial freeze.")
    with lifecycle.research_job():
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, _, parent, refs, clocks, _ = source_inputs()
        policy = read_json(POLICY)
        counts = current_counts(base.journal_events(base.JOURNAL))
        if (
            counts["current_journal_event_count"] != 1074
            or counts["current_journal_head_sha256"] != policy["prior_journal_head_sha256"]
            or counts["total_legacy_plus_ssd_completed_configurations"] != 268
        ):
            raise QuantError("Keep the actual J1074/268 pre-outcome baseline.")
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": policy["study_id"],
            "policy": reference(rules.POLICY),
            "financial_program_policy": reference(POLICY),
            "expert_pool": reference(rules.source.POLICY),
            "construction": "conditional_single_return_deletion_lower_envelope_pair",
            "prior_journal_sequence": 1074,
            "prior_journal_head_sha256": policy["prior_journal_head_sha256"],
            "prior_completed_configurations": 268,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in [*deps, *refs] if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / "DELETION_ROBUST_PAIR_FINANCIAL_V1.md"),
                    reference(ROOT / "EXPERT_DELETION_ROBUST_PAIR_V1.md"),
                ]
            ),
            "policies": unique_refs(
                [
                    *parent["policies"],
                    *[r for r in [*deps, *refs] if r["path"].startswith("config/")],
                ]
            ),
            "prior_evidence": unique_refs(
                [
                    *parent["prior_evidence"],
                    *refs,
                    *[
                        reference(FOLDER / p)
                        for p in (
                            "baseline-v1.json",
                            "design-decision-v1.json",
                            "actual-invocation-v1.json",
                        )
                    ],
                ]
            ),
            "validation_evidence": unique_refs(
                [*parent["validation_evidence"], *validation_refs()]
            ),
            "source_policy_and_expert_admission_clocks": clocks,
            "new_shadow_account_runs": 0,
            "new_source_requests": 0,
            "new_RF_or_source_scope_waiver": False,
        }
        check_frozen(frozen)
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "frozen": frozen,
                "complete_dependency_inputs": deps,
                "source_references": refs,
                "package_versions": sorted(
                    (d.metadata["Name"], d.version) for d in distributions()
                ),
                "real_new_policy_scores_targets_or_funds_computed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="actual_register_before_real_new_targets")
    return reference(INTENT)


def inputs():
    return bound(single.inputs)()


def _numpy_score(values, cost):
    array = np.asarray(values, dtype=float)
    if np.ptp(array) == 0:
        return None
    value = float(np.sqrt(252) * (np.mean(array) - cost / 21) / np.std(array, ddof=1))
    if not math.isfinite(value):
        raise QuantError("Independent conditional score is nonfinite.")
    return value


def _catalogue(observed, previous, selection, assets):
    known = observed[0].net_returns.index
    values = [(o.net_returns - o.risk_free).tolist() for o in observed]
    means = [math.fsum(v[:84]) / 84 for v in values]
    deviations = [
        math.sqrt(math.fsum((x - m) ** 2 for x in v[:84]) / 83)
        for v, m in zip(values, means, strict=True)
    ]
    expected = {
        tuple(sorted((rules.EXPERT_IDS[i], rules.EXPERT_IDS[j])))
        for i, j in itertools.combinations(range(9), 2)
    }
    pairs = selection["pairs"]
    if len(known) != 126 or len(pairs) != 36 or {tuple(p["pair"]) for p in pairs} != expected:
        raise QuantError("Retain all36original pair records and complete126-observation samples.")
    max_weight = max_score = max_profile = 0.0
    checked_profiles = 0
    for pair in pairs:
        i, j = [rules.EXPERT_IDS.index(n) for n in pair["pair"]]
        reason = (
            "same_mechanism_group"
            if observed[i].identity.mechanism_group == observed[j].identity.mechanism_group
            else "zero_training_variance"
            if min(deviations[i], deviations[j]) <= 0
            else None
        )
        weights = np.zeros(9)
        if reason is None:
            weights[i], weights[j] = (
                deviations[j] / (deviations[i] + deviations[j]),
                deviations[i] / (deviations[i] + deviations[j]),
            )
            portfolio = [math.fsum(weights[k] * values[k][t] for k in range(9)) for t in range(126)]
            if np.std(portfolio[:84], ddof=1) <= 0 or np.std(portfolio[84:], ddof=1) <= 0:
                reason = "nonpositive_or_nonfinite_pair_risk"
        if reason is not None:
            if (
                pair.get("unscorable_reason") != reason
                or pair["conditional_deletion_admissible"] is not False
                or pair["conditional_deletion_unscorable_reason"] != "original_" + reason
                or pair["conditional_profile"] is not None
            ):
                raise QuantError("An original unscorable pair cannot be rescued.")
            continue
        actual = np.asarray(pair["expert_weights"])
        if (
            actual.shape != (9,)
            or not np.isfinite(actual).all()
            or (actual < 0).any()
            or abs(math.fsum(actual) - 1) > 1e-12
            or "unscorable_reason" in pair
        ):
            raise QuantError("Retain complete finite original ERC weights.")
        max_weight = max(max_weight, float(max(abs(actual - weights))))
        covariance = (
            math.fsum((values[i][t] - means[i]) * (values[j][t] - means[j]) for t in range(84)) / 83
        )
        contribution = np.zeros(9)
        contribution[i] = weights[i] * (weights[i] * deviations[i] ** 2 + weights[j] * covariance)
        contribution[j] = weights[j] * (weights[j] * deviations[j] ** 2 + weights[i] * covariance)
        if (
            abs(pair["training_correlation"] - covariance / (deviations[i] * deviations[j])) > 1e-10
            or max(abs(np.asarray(pair["training_marginal_variance_contributions"]) - contribution))
            > 1e-12
        ):
            raise QuantError("Independent full-training covariance or marginal risk differs.")
        change = [
            abs(
                math.fsum(
                    (actual[k] - previous[k]) * float(observed[k].target[s]) for k in range(9)
                )
            )
            for s in assets
        ]
        cost = math.fsum(change) * 5 / 10000 + sum(v > 1e-12 for v in change) / 10000
        if (
            abs(cost - pair["estimated_incremental_switch_cost_fraction"]) > 1e-12
            or pair["estimated_incremental_changed_assets"] != sum(v > 1e-12 for v in change)
            or pair["cost_proxy_is_actual_main_account_cost"] is not False
        ):
            raise QuantError("The original nominal switch proxy is not actual main-account cost.")
        source_scores = [
            single.scalar_score(portfolio[:84], cost),
            single.scalar_score(portfolio[84:], cost),
        ]
        max_score = max(
            max_score,
            abs(source_scores[0] - pair["train_net_risk_score"]),
            abs(source_scores[1] - pair["validation_net_risk_score"]),
        )
        if pair["selection_score"] != min(
            pair["train_net_risk_score"], pair["validation_net_risk_score"]
        ) or pair["selection_score_key"] != round(pair["selection_score"], 12):
            raise QuantError("Preserve the original point-score metadata separately.")
        fixed = [math.fsum(actual[k] * values[k][t] for k in range(9)) for t in range(126)]
        train, valid = fixed[:84], fixed[84:]
        recorded_cost = pair["estimated_incremental_switch_cost_fraction"]
        expected_profile = [
            _numpy_score(train, recorded_cost),
            _numpy_score(valid, recorded_cost),
            *[_numpy_score(np.delete(train, k), recorded_cost) for k in range(84)],
            *[_numpy_score(np.delete(valid, k), recorded_cost) for k in range(42)],
        ]
        profile = pair["conditional_profile"]
        if (
            len(profile["training_one_return_deleted_scores"]) != 84
            or len(profile["validation_one_return_deleted_scores"]) != 42
        ):
            raise QuantError("Every full and single-return-deleted score must be retained.")
        stored = [
            profile["full_training_score"],
            profile["full_validation_score"],
            *profile["training_one_return_deleted_scores"],
            *profile["validation_one_return_deleted_scores"],
        ]
        undefined = []
        for index, (expected_score, actual_score) in enumerate(
            zip(expected_profile, stored, strict=True)
        ):
            checked_profiles += 1
            if (expected_score is None) != (actual_score is None):
                raise QuantError("Conditional exact-zero-risk classifications differ.")
            if expected_score is None:
                undefined.append(index)
            else:
                if not math.isfinite(actual_score):
                    raise QuantError("A retained conditional sample score is nonfinite.")
                max_profile = max(max_profile, abs(expected_score - actual_score))
        admissible = not undefined
        if (
            profile["undefined_sample_positions_full_train_valid_then_train_valid_deletions"]
            != undefined
            or pair["conditional_deletion_admissible"] != admissible
            or pair["conditional_deletion_unscorable_reason"]
            != (None if admissible else "constant_full_or_single_return_deleted_sample")
            or profile["conditional_lower_envelope"] != (min(stored) if admissible else None)
            or profile["conditional_lower_envelope_key"]
            != (round(profile["conditional_lower_envelope"], 12) if admissible else None)
        ):
            raise QuantError("The exact full128-score envelope, key or admissibility differs.")
    point = [p for p in pairs if "selection_score_key" in p]
    old = min(point, key=lambda p: (-p["selection_score_key"], p["pair"])) if point else None
    usable = [p for p in pairs if p["conditional_deletion_admissible"]]
    if (
        selection["same_state_original_catalogue_selected_pair"] != (old["pair"] if old else None)
        or selection["same_state_original_catalogue_scores_available"] != bool(point)
        or selection["conditionally_admissible_pairs"] != len(usable)
    ):
        raise QuantError("Original catalogue and conditional-score availability differ.")
    if point and [
        selection[k] for k in ("train_start", "train_end", "validation_start", "validation_end")
    ] != [str(known[i].date()) for i in (0, 83, 84, 125)]:
        raise QuantError("Keep the original chronological84/42mature split.")
    if usable:
        # Rank verified stored keys, not independently re-rounded binary64 buckets.
        best = min(
            usable,
            key=lambda p: (-p["conditional_profile"]["conditional_lower_envelope_key"], p["pair"]),
        )
        if (
            selection["selected_pair"] != best["pair"]
            or selection["scores_available"] is not True
            or selection["reason"] != "conditional_single_return_deletion_lower_envelope"
        ):
            raise QuantError("The independently checked robust-key lexical winner differs.")
        chosen = np.asarray(best["expert_weights"])
    else:
        if (
            selection["selected_pair"] is not None
            or selection["scores_available"] is not False
            or selection["reason"] != "no_conditionally_admissible_pair_explicit_equal_weights"
        ):
            raise QuantError("The mature equal-nine fallback cannot claim scoring success.")
        chosen = np.full(9, 1 / 9)
    if max_weight > 1e-10 or max_score > 1e-8 or max_profile > 1e-8:
        raise QuantError("Original independent weight, point-score or profile tolerances failed.")
    return chosen, max_weight, max_score, max_profile, checked_profiles


def verify_retained(path, data, experts, targets, audits):
    dates = rules.source.decision_sessions(data.close.index, read_json(rules.source.POLICY))
    if (
        not targets.index.equals(data.close.index)
        or not targets.columns.equals(data.close.columns)
        or not targets.dropna(how="all").index.equals(dates)
        or [a["decision_session"] for a in audits] != dates.strftime("%Y-%m-%d").tolist()
    ):
        raise QuantError("Keep every original information date and complete target coordinate.")
    previous = np.full(9, 1 / 9)
    max_target = max_weight = max_score = max_profile = 0.0
    source_values = profile_values = 0
    with path.open() as stream:
        for index, (day, audit) in enumerate(zip(dates, audits, strict=True)):
            before, after = json.loads(stream.readline()), json.loads(stream.readline())
            observed = [e.observe(day, 126) for e in experts]
            if (
                before["record"] != "decision_input"
                or after["record"] != "decision_resolved"
                or before["decision_index"] != index
                or after["decision_index"] != index
                or before["as_of"] != str(day.date())
                or before["information_clock"] != rules.CLOCK
                or before["previous_weights_binary64_hex"] != [float(v).hex() for v in previous]
                or len(before["observations"]) != 9
                or audit["previous_rolling_expert_weights"] != previous.tolist()
                or audit["decision_information_clock"] != rules.CLOCK
                or audit["shadow_wealth_added_to_main_capital"] is not False
                or audit["source_feedback_floor_transferred_to_main_account"] is not False
                or after["audit"] != audit
                or after["asset_order"] != list(targets.columns)
            ):
                raise QuantError(
                    "Retained operands, actual previous state or capital separation changed."
                )
            for i, (row, o) in enumerate(zip(before["observations"], observed, strict=True)):
                if (
                    row["id"] != rules.EXPERT_IDS[i]
                    or row["mechanism_group"] != rules.MECHANISMS[i]
                    or row["introduced_at"] != o.identity.introduced_at
                    or row["target_known_at"] != str(o.target_known_at.date())
                    or o.target_known_at > day
                    or row["asset_order"] != list(targets.columns)
                    or row["target_binary64_hex"] != [float(v).hex() for v in o.target]
                    or row["return_dates"] != o.net_returns.index.strftime("%Y-%m-%d").tolist()
                    or row["NET_returns_binary64_hex"] != [float(v).hex() for v in o.net_returns]
                    or row["RF_binary64_hex"] != [float(v).hex() for v in o.risk_free]
                    or row["score_mature"] != o.score_mature
                ):
                    raise QuantError("Retain every exact admitted source target, NET/RF and clock.")
                source_values += len(o.net_returns) + len(o.risk_free)
            selection = audit["selection"]
            mature = all(o.score_mature for o in observed)
            if (
                selection["NET_history_mature"] != mature
                or selection["pair_covariance_and_weights_refitted_after_deletion"] is not False
                or selection["current_opinion_exclusion_used"] is not False
                or selection["conditional_envelope_is_probability_or_risk_guarantee"] is not False
                or selection["safety_or_success_claimed"] is not False
            ):
                raise QuantError("Keep the exact98bd conditional score law and explicit limits.")
            if mature:
                if any(
                    not o.net_returns.index.equals(observed[0].net_returns.index)
                    or not o.risk_free.equals(observed[0].risk_free)
                    or o.net_returns.index[-1] > day
                    for o in observed
                ):
                    raise QuantError("The mature source histories or canonical RF are not common.")
                chosen, weight, score, profile, checked = _catalogue(
                    observed, previous, selection, data.close.columns
                )
                max_weight, max_score, max_profile = (
                    max(max_weight, weight),
                    max(max_score, score),
                    max(max_profile, profile),
                )
                profile_values += checked
            else:
                if (
                    selection["pairs"]
                    or selection["selected_pair"] is not None
                    or selection["scores_available"] is not False
                    or selection["reason"] != "unscored_initial_equal_weights"
                    or selection["conditionally_admissible_pairs"] != 0
                ):
                    raise QuantError("An immature source cannot acquire successful robust scores.")
                chosen = np.full(9, 1 / 9)
            if not np.allclose(chosen, audit["rolling_expert_weights"], rtol=0, atol=1e-12):
                raise QuantError("Issued statistical weights differ from their verified choice.")
            actual = targets.loc[day]
            if after["complete_target_binary64_hex"] != [float(v).hex() for v in actual]:
                raise QuantError("Resolved target operands differ from the actual unified target.")
            expected = [
                float(
                    sum(
                        Fraction.from_float(float(chosen[i]))
                        * Fraction.from_float(float(o.target[s]))
                        for i, o in enumerate(observed)
                    )
                )
                for s in data.close.columns
            ]
            max_target = max(max_target, float(max(abs(actual - expected))))
            previous = chosen
        if stream.readline():
            raise QuantError("Unexpected extra or partial retained decision records.")
    if max_target > 1e-10:
        raise QuantError("Original independent unified-target tolerance failed.")
    return {
        "verified_decisions": len(audits),
        "actual_input_resolved_records": 2 * len(audits),
        "roundtrip_NET_RF_binary64_values_checked": source_values,
        "independent_conditional_sample_scores_checked": profile_values,
        "all36_original_catalogues_and_all128_conditional_profile_scores_verified": True,
        "maximum_independent_target_difference": max_target,
        "maximum_independent_expert_weight_difference": max_weight,
        "maximum_independent_net_score_difference": max_score,
        "maximum_independent_conditional_profile_difference": max_profile,
        "original_source_signal_recomputed": False,
    }


def prepare(*args):
    return bound(single.prepare)(*args)


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
    return bound(single.export)()


def main():
    parser = argparse.ArgumentParser(
        description="Frozen conditional robust pair finance, no orders."
    )
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use an actual remotely published exact preregistration SHA.")
    actions = {"freeze": initialize, "register": register, "export": export}
    print(
        json.dumps(evaluate(args.commit) if args.action == "evaluate" else actions[args.action]())
    )


if __name__ == "__main__":
    main()
