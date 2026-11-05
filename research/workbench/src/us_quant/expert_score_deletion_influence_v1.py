from __future__ import annotations

import itertools
import math
from datetime import date

import numpy as np

from us_quant import expert_distinct_opinion_pair_v1 as source
from us_quant.config import QuantError

ROOT = source.ROOT
POLICY = ROOT / "config/expert-score-deletion-influence-v1.json"
ID = "ssd_expert_score_deletion_influence_20261105_v1"
EXPECTED = {
    "schema_version": 1,
    "study_id": ID,
    "source_completion": {
        "path": (
            "reports/ssd-continuous-research-v1/distinct-opinion-pair-financial-20261105-v1/"
            "completion-v1.json"
        ),
        "sha256": "75db4b85d2b81afb9f81c59971357aba01db490206be5ba6636bbe24731bd806",
    },
    "source_trace": {
        "path": (
            "reports/ssd-continuous-research-v1/distinct-opinion-pair-financial-20261105-v1/"
            "actual-decision-inputs-resolved-v1.jsonl"
        ),
        "sha256": "66fe4b839bb9e16dcb618f0fa88d83a74e0795c7a76507caa0d6a12e1ae7f230",
    },
    "source_decisions": 121,
    "expert_count": 9,
    "asset_count": 14,
    "train_observations": 84,
    "validation_observations": 42,
    "annualization": 252,
    "cost_amortization_sessions": 21,
    "selection_key_decimals": 12,
    "independent_score_absolute_tolerance": 1e-8,
    "pair_weights_and_nominal_switch_proxy_fixed": True,
    "current_opinion_eligibility_fixed": True,
    "covariance_refitted_after_deletion": False,
    "exact_constant_sample_is_unscorable": True,
    "new_trading_policy_or_acceptance_rule": False,
    "new_financial_evaluations": 0,
    "order_authority": False,
}


def validate(policy):
    if policy != EXPECTED or any(type(policy[k]) is not type(v) for k, v in EXPECTED.items()):
        raise QuantError(
            "Keep the declared conditional deletion experiment, not a new portfolio rule."
        )


def _hex(values):
    result = [float.fromhex(value) for value in values]
    if not all(math.isfinite(v) for v in result):
        raise QuantError("Retained operands must be finite binary64 values.")
    return result


def _score(values, cost):
    if len(values) < 2:
        raise QuantError("A sample risk score needs at least two observations.")
    if min(values) == max(values):
        return None
    mean = math.fsum(values) / len(values)
    variance = math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1)
    if variance <= 0 or not math.isfinite(variance):
        raise QuantError("Nonconstant direct scalar sample variance is invalid.")
    result = math.sqrt(252) * (mean - cost / 21) / math.sqrt(variance)
    if not math.isfinite(result):
        raise QuantError("A nonconstant sample produced a nonfinite risk score.")
    return result


def _oracle(values, cost):
    array = np.asarray(values, dtype=float)
    if np.ptp(array) == 0:
        return None
    result = float(np.sqrt(252) * (np.mean(array) - cost / 21) / np.std(array, ddof=1))
    if not math.isfinite(result):
        raise QuantError("The independent moment calculation is nonfinite.")
    return result


def _winner(rows):
    ordered = sorted(rows, key=lambda row: (-row["key"], row["pair"]))
    if not ordered:
        return {"pair": None, "key": None, "runner_up_key_margin": None}
    return {
        **ordered[0],
        "runner_up_key_margin": (
            ordered[0]["key"] - ordered[1]["key"] if len(ordered) > 1 else None
        ),
    }


def diagnose(before: dict, after: dict, policy: dict) -> dict:
    validate(policy)
    audit = after["audit"]
    selection = audit["selection"]
    observed = before["observations"]
    day = date.fromisoformat(before["as_of"])
    if (
        before["record"] != "decision_input"
        or after["record"] != "decision_resolved"
        or before["decision_index"] != after["decision_index"]
        or audit["decision_session"] != before["as_of"]
        or before["information_clock"] != source.CLOCK
        or audit["decision_information_clock"] != source.CLOCK
        or [o["id"] for o in observed] != list(source.EXPERT_IDS)
        or [o["mechanism_group"] for o in observed] != list(source.MECHANISMS)
        or after["asset_order"] != list(source.UNIVERSE)
    ):
        raise QuantError("Use the complete frozen source record, clock and nine-expert identity.")
    previous = _hex(before["previous_weights_binary64_hex"])
    if len(previous) != 9 or previous != audit["previous_rolling_expert_weights"]:
        raise QuantError("The retained statistical previous state changed.")
    dates = observed[0]["return_dates"]
    if dates != sorted(set(dates)) or any(date.fromisoformat(d) > day for d in dates):
        raise QuantError("Retain chronological observations available at the decision.")
    excess, targets = [], []
    for row in observed:
        net, rf, target = (
            _hex(row[k])
            for k in ("NET_returns_binary64_hex", "RF_binary64_hex", "target_binary64_hex")
        )
        if (
            row["return_dates"] != dates
            or len(net) != len(rf)
            or len(net) != len(dates)
            or row["RF_binary64_hex"] != observed[0]["RF_binary64_hex"]
            or row["asset_order"] != list(source.UNIVERSE)
            or len(target) != 14
            or date.fromisoformat(row["target_known_at"]) > day
            or min(target) < 0
            or abs(math.fsum(target) - 0.98) > 1e-12
            or type(row["score_mature"]) is not bool
        ):
            raise QuantError("Retain all source NET/RF, complete opinions and known-at clocks.")
        excess.append([n - r for n, r in zip(net, rf, strict=True)])
        targets.append(target)
    mature = all(o["score_mature"] for o in observed)
    if selection["NET_history_mature"] != mature or (mature and len(dates) != 126):
        raise QuantError("Score maturity must not become eligible-score availability.")
    result = {
        "decision_index": before["decision_index"],
        "decision_session": before["as_of"],
        "source_selected_pair": selection["selected_pair"],
        "source_reason": selection["reason"],
        "source_catalogue_rows": len(selection["pairs"]),
        "conditional_weights_and_proxy_fixed": True,
        "covariance_refitted": False,
        "new_portfolio_targets_accounts_or_financial_acceptance": False,
    }
    if not mature:
        if (
            selection["pairs"]
            or selection["selected_pair"] is not None
            or selection["scores_available"]
        ):
            raise QuantError("An immature source cannot have a scored winner.")
        return {
            **result,
            "state": "source_unscored",
            "deletions": [],
            "profiles": [],
            "conditional_full_sample_winner": None,
            "source_vs_kernel_baseline_winner_differs": None,
            "source_key_bucket_differences": None,
            "maximum_independent_score_gap": None,
            "maximum_source_baseline_score_gap": None,
            "oracle_sample_score_checks": 0,
        }
    expected_pairs = {
        tuple(sorted((source.EXPERT_IDS[i], source.EXPERT_IDS[j])))
        for i, j in itertools.combinations(range(9), 2)
    }
    pairs = selection["pairs"]
    if len(pairs) != 36 or {tuple(p["pair"]) for p in pairs} != expected_pairs:
        raise QuantError("Retain all36original scored/unscorable pair records.")
    max_gap = source_gap = 0.0
    oracle_checks = bucket_differences = 0

    def checked_score(values, cost):
        nonlocal max_gap, oracle_checks
        scalar, numpy = _score(values, cost), _oracle(values, cost)
        oracle_checks += 1
        if (scalar is None) != (numpy is None):
            raise QuantError("Independent zero-variance classifications disagree.")
        if scalar is not None:
            gap = abs(scalar - numpy)
            max_gap = max(max_gap, gap)
            if gap > policy["independent_score_absolute_tolerance"]:
                raise QuantError("Independent sample risk-score calculation differs.")
        return scalar

    profiles = []
    for pair in pairs:
        names = pair["pair"]
        i, j = [source.EXPERT_IDS.index(name) for name in names]
        allowed = targets[i] != targets[j]
        if (
            pair["eligible_by_current_opinion"] != allowed
            or pair["exact_same_current_opinion"] == allowed
        ):
            raise QuantError("The frozen exact current-opinion menu changed.")
        if "unscorable_reason" in pair:
            if any(k in pair for k in ("selection_score", "selection_score_key")):
                raise QuantError("An originally unscorable pair cannot acquire a stored score.")
            profiles.append(
                {
                    "pair": names,
                    "eligible": allowed,
                    "source_unscorable_reason": pair["unscorable_reason"],
                }
            )
            continue
        weights = pair["expert_weights"]
        cost = pair["estimated_incremental_switch_cost_fraction"]
        if (
            len(weights) != 9
            or not all(math.isfinite(v) and v >= 0 for v in weights)
            or abs(math.fsum(weights) - 1) > 1e-12
            or any(weights[k] != 0 for k in range(9) if k not in (i, j))
            or not math.isfinite(cost)
            or cost < 0
            or pair["cost_proxy_is_actual_main_account_cost"] is not False
        ):
            raise QuantError("Keep complete recorded pair weights and nominal, not actual, costs.")
        values = [math.fsum(weights[k] * excess[k][t] for k in range(9)) for t in range(126)]
        train, valid = values[:84], values[84:]
        first, second = checked_score(train, cost), checked_score(valid, cost)
        if first is None or second is None:
            raise QuantError("An originally scored pair has a constant full-sample portfolio.")
        source_gap = max(
            source_gap,
            abs(first - pair["train_net_risk_score"]),
            abs(second - pair["validation_net_risk_score"]),
        )
        if (
            source_gap > policy["independent_score_absolute_tolerance"]
            or pair["selection_score"]
            != min(pair["train_net_risk_score"], pair["validation_net_risk_score"])
            or pair["selection_score_key"] != round(pair["selection_score"], 12)
        ):
            raise QuantError(
                "The retained full-sample catalogue disagrees with its fixed operands."
            )
        key = round(min(first, second), 12)
        bucket_differences += key != pair["selection_score_key"]
        profiles.append(
            {
                "pair": names,
                "eligible": allowed,
                "full_train_score": first,
                "full_validation_score": second,
                "full_key": key,
                "training_one_return_deleted_scores": [
                    checked_score(train[:k] + train[k + 1 :], cost) for k in range(84)
                ],
                "validation_one_return_deleted_scores": [
                    checked_score(valid[:k] + valid[k + 1 :], cost) for k in range(42)
                ],
            }
        )
    retained = [
        {"pair": p["pair"], "key": p["selection_score_key"]}
        for p in pairs
        if p["eligible_by_current_opinion"] and "selection_score_key" in p
    ]
    if _winner(retained)["pair"] != selection["selected_pair"]:
        raise QuantError("The actual retained eligible-key winner changed.")
    full = _winner(
        [
            {"pair": p["pair"], "key": p["full_key"]}
            for p in profiles
            if p["eligible"] and "full_key" in p
        ]
    )
    deletions = []
    for index, observed_at in enumerate(dates):
        candidates, degenerate = [], []
        for profile in profiles:
            if not profile["eligible"] or "full_key" not in profile:
                continue
            if index < 84:
                left = profile["training_one_return_deleted_scores"][index]
                right = profile["full_validation_score"]
            else:
                left = profile["full_train_score"]
                right = profile["validation_one_return_deleted_scores"][index - 84]
            if left is None or right is None:
                degenerate.append(profile["pair"])
                continue
            candidates.append({"pair": profile["pair"], "key": round(min(left, right), 12)})
        winner = _winner(candidates)
        deletions.append(
            {
                "deleted_observation": observed_at,
                "panel": "training" if index < 84 else "validation",
                "conditional_winner": winner["pair"],
                "conditional_winner_key": winner["key"],
                "runner_up_key_margin": winner["runner_up_key_margin"],
                "eligible_pairs_with_degenerate_deleted_sample": degenerate,
                "no_scorable_conditional_pair": not candidates,
            }
        )
    return {
        **result,
        "state": "mature_conditional_catalogue",
        "profiles": profiles,
        "deletions": deletions,
        "conditional_full_sample_winner": full["pair"],
        "source_vs_kernel_baseline_winner_differs": full["pair"] != selection["selected_pair"],
        "source_key_bucket_differences": int(bucket_differences),
        "maximum_independent_score_gap": max_gap,
        "maximum_source_baseline_score_gap": source_gap,
        "oracle_sample_score_checks": oracle_checks,
    }
