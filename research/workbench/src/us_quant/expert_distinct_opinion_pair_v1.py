from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import lagged_spy_price_impact_expert_inclusion_v1 as source
from us_quant.conditional_hedge_expert_v1 import UNIVERSE
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import ExpertObservation, StrategyExpert

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/expert-distinct-opinion-pair-v1.json"
ID = "expert_pool_nine_distinct_opinion_pair_v1"
EXPERT_IDS, MECHANISMS, CLOCK = source.EXPERT_IDS, source.MECHANISMS, source.CLOCK
score_catalogue, combine = source.choose_pair, source.combine
OWN = (
    "config/expert-distinct-opinion-pair-v1.json",
    "src/us_quant/expert_distinct_opinion_pair_v1.py",
    "tests/test_expert_distinct_opinion_pair_v1.py",
    "EXPERT_DISTINCT_OPINION_PAIR_V1.md",
)


def validate(policy):
    expected = {
        "schema_version": 1,
        "study_id": "ssd_expert_distinct_opinion_pair_primary_20261105_v1",
        "candidate_id": ID,
        "source_pool_policy": {
            "path": "config/lagged-spy-price-impact-expert-inclusion-v1.json",
            "sha256": "d089bb061fc164dc328c00caa716d07bfb08ccf07d4b6620c817d815006f0118",
        },
        "source_cohort_completion": {
            "path": (
                "reports/ssd-continuous-research-v1/"
                "lagged-spy-price-impact-expert-inclusion-20261105-v1/completion-v1.json"
            ),
            "sha256": "c2229fe31184f00bbb2d5d31f1b0130afef8ac0d6f1e8de398c7b486038572ab",
        },
        "source_geometry_completion": {
            "path": (
                "reports/ssd-continuous-research-v1/"
                "expert-opinion-geometry-audit-20261105-v1/completion-v1.json"
            ),
            "sha256": "b5923bf313da96e278a8e777fc76026155bfbb9e666fa1b95754578771ccd6d1",
        },
        "prior_journal_sequence": 1062,
        "prior_journal_head_sha256": (
            "ee118c53ed06e85f2df9a35116a5446e0a73c9a655742f21b587921e63dd298a"
        ),
        "prior_completed_configurations": 267,
        "expert_count": 9,
        "source_pool_scope": "same_study_specific_nine_not_global_roster_promotion",
        "universe": list(UNIVERSE),
        "lookback_sessions": 126,
        "training_sessions": 84,
        "validation_sessions": 42,
        "selected_experts": 2,
        "all_pair_catalogue_size": 36,
        "eligibility": (
            "exclude_only_exact_numeric_equal_current_complete_asset_opinions_signed_zero_equal"
        ),
        "approximate_geometry_or_rank_threshold_used": False,
        "cold_start": "original_unscored_equal_nine",
        "no_eligible_scorable_pair": "explicit_equal_nine_known_opinions_not_safety_or_success",
        "scorer": "unchanged_original_train_ERC_weaker_NET_period_score_and_switch_proxy",
        "score_decimal_places": 12,
        "switch_cost_amortization_sessions": 21,
        "invested_target_budget": 0.98,
        "idle_cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "decision_information_clock": CLOCK,
        "new_source_requests": 0,
        "new_shadow_account_runs": 0,
        "source_experts_invalidated": False,
        "global_pool_changed": False,
        "real_new_policy_targets_accounts_or_financial_outcomes_computed": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if (
        not isinstance(policy, dict)
        or policy != expected
        or any(type(policy[k]) is not type(v) for k, v in expected.items())
    ):
        raise QuantError("Freeze one exact current-opinion menu change without scorer rescue.")


def _opinions(observed):
    if (
        tuple(o.identity.identifier for o in observed) != EXPERT_IDS
        or tuple(o.identity.mechanism_group for o in observed) != MECHANISMS
        or any(
            not isinstance(o.target, pd.Series)
            or tuple(o.target.index) != UNIVERSE
            or o.target.dtype.kind not in "fi"
            or not np.isfinite(o.target.to_numpy()).all()
            or (o.target < 0).any()
            or abs(math.fsum(float(v) for v in o.target) - 0.98) > 1e-12
            for o in observed
        )
    ):
        raise QuantError("Retain all nine complete opinions and the exact14-asset source scope.")
    return np.stack([o.target.to_numpy() for o in observed])


def choose_pair(observed: list[ExpertObservation], previous: np.ndarray):
    matrix = _opinions(observed)
    if (
        not isinstance(previous, np.ndarray)
        or previous.shape != (9,)
        or previous.dtype.kind not in "fi"
        or not np.isfinite(previous).all()
        or (previous < 0).any()
        or abs(math.fsum(float(v) for v in previous) - 1) > 1e-12
    ):
        raise QuantError("Keep the actual previous statistical allocation, not main capital.")
    original_weights, original = score_catalogue(observed, previous)
    eligible = {
        tuple(sorted((EXPERT_IDS[a], EXPERT_IDS[b]))): not np.array_equal(matrix[a], matrix[b])
        for a, b in itertools.combinations(range(9), 2)
    }
    mature = all(o.score_mature for o in observed)
    added = {
        "NET_history_mature": mature,
        "same_state_original_catalogue_selected_pair": original["selected_pair"],
        "same_state_original_catalogue_scores_available": original["scores_available"],
        "current_opinion_pair_eligibility": [
            {"pair": list(pair), "exact_same_current_opinion": not allowed, "eligible": allowed}
            for pair, allowed in sorted(eligible.items())
        ],
        "current_opinion_difference_is_independent_risk": False,
        "source_experts_invalidated": False,
    }
    if not mature:
        return original_weights, {**original, **added, "eligible_scorable_pairs": 0}
    pairs = original["pairs"]
    if len(pairs) != 36 or {tuple(row["pair"]) for row in pairs} != set(eligible):
        raise QuantError("Retain every original mature pair score and unscorable reason.")
    records = [
        {
            **row,
            "exact_same_current_opinion": not eligible[tuple(row["pair"])],
            "eligible_by_current_opinion": eligible[tuple(row["pair"])],
        }
        for row in pairs
    ]
    usable = [
        row
        for row in records
        if row["eligible_by_current_opinion"] and "unscorable_reason" not in row
    ]
    result = {**original, **added, "pairs": records, "eligible_scorable_pairs": len(usable)}
    if not usable:
        reason = (
            "no_distinct_current_opinion_pair_equal_weights"
            if not any(eligible.values())
            else "no_scorable_distinct_current_opinion_pair_equal_weights"
        )
        return np.full(9, 1 / 9), {
            **result,
            "reason": reason,
            "selected_pair": None,
            "scores_available": False,
            "safety_or_success_claimed": False,
        }
    chosen = min(usable, key=lambda row: (-row["selection_score_key"], row["pair"]))
    return np.asarray(chosen["expert_weights"]), {
        **result,
        "reason": "original_scored_pair_with_exact_current_opinion_distinction",
        "selected_pair": chosen["pair"],
        "scores_available": True,
        "safety_or_success_claimed": False,
    }


def build_targets(
    experts: tuple[StrategyExpert, ...],
    calendar: pd.DatetimeIndex,
    universe: tuple[str, ...],
    policy: dict,
    before_decision=None,
    retain=None,
):
    validate(policy)
    if universe != UNIVERSE or tuple(e.identity.identifier for e in experts) != EXPERT_IDS:
        raise QuantError("Use the unchanged fixed nine sources and original14asset universe.")
    dates = source.decision_sessions(calendar, read_json(source.POLICY))
    target = pd.DataFrame(np.nan, calendar, universe)
    audits, previous = [], np.full(9, 1 / 9)
    for position, day in enumerate(dates):
        if before_decision is not None:
            before_decision()
        observed = [e.observe(day, 126) for e in experts]
        if retain is not None:
            retain(
                {
                    "record": "decision_input",
                    "decision_index": position,
                    "as_of": str(day.date()),
                    "information_clock": CLOCK,
                    "previous_weights_binary64_hex": [float(v).hex() for v in previous],
                    "observations": [
                        {
                            "id": o.identity.identifier,
                            "mechanism_group": o.identity.mechanism_group,
                            "introduced_at": o.identity.introduced_at,
                            "target_known_at": str(o.target_known_at.date())
                            if o.target_known_at is not None
                            else None,
                            "target_binary64_hex": [float(v).hex() for v in o.target]
                            if o.target is not None
                            else None,
                            "asset_order": list(o.target.index) if o.target is not None else None,
                            "return_dates": o.net_returns.index.strftime("%Y-%m-%d").tolist(),
                            "NET_returns_binary64_hex": [float(v).hex() for v in o.net_returns],
                            "RF_binary64_hex": [float(v).hex() for v in o.risk_free],
                            "score_mature": o.score_mature,
                        }
                        for o in observed
                    ],
                }
            )
        if position > 0 and any(not len(o.net_returns) for o in observed):
            raise QuantError("Missing post-start NET history cannot receive a fallback.")
        weights, selection = choose_pair(observed, previous)
        target.loc[day] = combine(observed, weights)
        record = {
            "decision_session": str(day.date()),
            "decision_information_clock": CLOCK,
            "expert_target_known_at": [str(o.target_known_at.date()) for o in observed],
            "preserved_archive_identity_introduced_at": [
                o.identity.introduced_at for o in observed
            ],
            "previous_rolling_expert_weights": previous.tolist(),
            "rolling_expert_weights": weights.tolist(),
            "selection": selection,
            "counterfactual_historical_expert_availability": True,
            "shadow_wealth_added_to_main_capital": False,
            "source_feedback_floor_transferred_to_main_account": False,
            "independent_forward_validation": False,
        }
        audits.append(record)
        if retain is not None:
            retain(
                {
                    "record": "decision_resolved",
                    "decision_index": position,
                    "audit": record,
                    "asset_order": list(universe),
                    "complete_target_binary64_hex": [float(v).hex() for v in target.loc[day]],
                }
            )
        previous = weights
    return {ID: target}, audits
