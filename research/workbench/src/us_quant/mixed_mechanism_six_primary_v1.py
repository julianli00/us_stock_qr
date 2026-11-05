from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import strategy_expert_pool_eight_v1 as original
from us_quant import strategy_expert_pool_v2 as scorer
from us_quant.calendar import is_month_end, sessions
from us_quant.conditional_hedge_expert_v1 import UNIVERSE
from us_quant.config import QuantError

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/mixed-mechanism-six-primary-v1.json"
EXPERT_IDS = (
    "equity_first_month_end_commitment_v1",
    "semiconductor_ratchet_protection",
    "conditional_covariance_hedge_expert_v1",
    "calendar_seasonal_six_core_bil_v1",
    "monthly_style_reversal_v1",
    "fixed_four_sleeve_rebalance_v1",
)
MECHANISMS = (
    "committed_price_trend_defense",
    "wealth_feedback_protection",
    "conditional_cross_asset_hedge_suitability",
    "fixed_calendar_seasonal_equity_participation",
    "monthly_cross_sectional_style_reversal",
    "fixed_cross_asset_capital_rebalancing",
)
CLOCK = "after_16_15_ET_index_close_before_later_open_execution"
CANDIDATES = (
    {"id": "expert_pool_mixed_mechanism_six_equal_v1", "weighting": "equal"},
    {
        "id": "expert_pool_mixed_mechanism_six_rolling_pair_v1",
        "weighting": "train_erc_validate_net",
    },
)
IDS = tuple(row["id"] for row in CANDIDATES)
score_catalogue, combine = scorer.choose_pair, scorer.combine
OWN = (
    "config/mixed-mechanism-six-primary-v1.json",
    "src/us_quant/mixed_mechanism_six_primary_v1.py",
    "src/us_quant/mixed_mechanism_six_sources_v1.py",
    "tests/test_mixed_mechanism_six_primary_v1.py",
    "MIXED_MECHANISM_SIX_PRIMARY_V1.md",
)
FIXTURE_INPUTS = ("tests/test_expert_distinct_opinion_pair_v1.py", "tests/conftest.py")


def validate(policy):
    expected = {
        "schema_version": 1,
        "study_id": "ssd_mixed_mechanism_six_primary_20261105_v1",
        "original_eight_policy": {
            "path": "config/strategy-expert-pool-eight-v1.json",
            "sha256": "d778bd86a6ed4d9c813336be188b4b071b86860d912da2eaed5b6338665375f3",
        },
        "calendar_policy": {
            "path": "config/calendar-expert-inclusion-v1.json",
            "sha256": "22861134145bedd04f4489f4c1a6578298cfb234c35f31c1c3472c002ebd0cbf",
        },
        "reversal_policy": {
            "path": "config/reversal-expert-inclusion-v1.json",
            "sha256": "1b50a4fb04e80a318fdd47909fb1218bf3b6a0f7d61e611055e3d0af9e5fd8f9",
        },
        "fixed_policy": {
            "path": "config/fixed-four-sleeve-expert-inclusion-v1.json",
            "sha256": "1b78e2761ff3ffd88057338697ec9707911bc79a87baddbf6406a072906c45c8",
        },
        "preceding_financial_completion": {
            "path": (
                "reports/ssd-continuous-research-v1/deletion-robust-pair-financial-20261105-v1/"
                "completion-v1.json"
            ),
            "sha256": "84e744408654e22ea9e929227318cc818455eacfc7eddf6251725316a2a7f6e9",
        },
        "prior_journal_sequence": 1086,
        "prior_journal_head_sha256": (
            "83abc48588a2405d4389ba1d7097e805e29b04b5b63d4dd02ea0e2607a0a2f6a"
        ),
        "prior_completed_configurations": 269,
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "expert_count": 6,
        "initial_decision_session": "2016-10-05",
        "lookback_sessions": 126,
        "training_sessions": 84,
        "validation_sessions": 42,
        "selected_experts": 2,
        "mature_pair_catalogue_size": 15,
        "score_decimal_places": 12,
        "switch_cost_amortization_sessions": 21,
        "scorer": "unchanged_original_train_ERC_weaker_NET_period_and_nominal_switch_proxy",
        "equal_control_weights": "one_sixth_each_complete_expert",
        "cold_start": "original_unscored_equal_six",
        "exact_or_approximate_opinion_exclusion": False,
        "conditional_deletion_lower_envelope": False,
        "invested_target_budget": 0.98,
        "idle_cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "decision_information_clock": CLOCK,
        "candidates": list(CANDIDATES),
        "candidate_count_in_this_study": 2,
        "study_size_is_a_total_research_quota": False,
        "source_generators_or_old_accounts_replayed": False,
        "new_source_requests": 0,
        "new_shadow_account_runs": 0,
        "source_RF_view_or_admission_scope_changed": False,
        "global_pool_changed": False,
        "mechanism_labels_are_independent_risks": False,
        "actual_registered_at_preparation": False,
        "first_real_meta_derivation_requires_actual_registrations": True,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if (
        not isinstance(policy, dict)
        or policy != expected
        or any(type(policy[k]) is not type(v) for k, v in expected.items())
    ):
        raise QuantError("Freeze the deliberate six mechanisms and unchanged original scoring law.")


def assemble(original_eight, calendar, reversal, fixed):
    if tuple(e.identity.identifier for e in original_eight) != original.EXPERT_IDS:
        raise QuantError("Select representatives only from the exact complete original eight.")
    experts = (original_eight[0], original_eight[3], original_eight[7], calendar, reversal, fixed)
    if (
        tuple(e.identity.identifier for e in experts) != EXPERT_IDS
        or tuple(e.identity.mechanism_group for e in experts) != MECHANISMS
        or len({e.identity.identifier for e in experts}) != 6
    ):
        raise QuantError(
            "Retain the six complete identities, not substituted or duplicated sources."
        )
    return experts


def choose_pair(observed, previous):
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
        raise QuantError(
            "Retain every complete source opinion in the original14-asset coordinates."
        )
    if not isinstance(previous, np.ndarray) or previous.shape != (6,):
        raise QuantError("Keep the six-dimensional statistical previous state.")
    weights, selection = score_catalogue(observed, previous)
    mature = all(o.score_mature for o in observed)
    if mature and (
        len(selection["pairs"]) != 15
        or {tuple(p["pair"]) for p in selection["pairs"]}
        != {
            tuple(sorted((EXPERT_IDS[i], EXPERT_IDS[j])))
            for i, j in itertools.combinations(range(6), 2)
        }
    ):
        raise QuantError(
            "Retain every original scored or unscorable pair in this six-member roster."
        )
    return weights, {
        **selection,
        "NET_history_mature": mature,
        "source_experts_invalidated": False,
        "mechanism_labels_are_independent_risks": False,
        "current_opinion_exclusion_used": False,
        "conditional_deletion_lower_envelope_used": False,
    }


def build_targets(experts, calendar, universe, policy, before_decision=None, retain=None):
    validate(policy)
    initial = pd.Timestamp("2016-10-05")
    if (
        universe != UNIVERSE
        or tuple(e.identity.identifier for e in experts) != EXPERT_IDS
        or not isinstance(calendar, pd.DatetimeIndex)
        or calendar.empty
        or not calendar.is_unique
        or not calendar.is_monotonic_increasing
        or initial not in calendar
        or not calendar.equals(sessions(calendar[0], calendar[-1]))
    ):
        raise QuantError("Use the whole original decision lattice, six sources and14assets.")
    dates = pd.DatetimeIndex([initial, *[d for d in calendar if d > initial and is_month_end(d)]])
    targets = {identifier: pd.DataFrame(np.nan, calendar, universe) for identifier in IDS}
    equal, previous = np.full(6, 1 / 6), np.full(6, 1 / 6)
    audits = []
    for index, day in enumerate(dates):
        if before_decision is not None:
            before_decision()
        observed = [e.observe(day, 126) for e in experts]
        if retain is not None:
            retain(
                {
                    "record": "decision_input",
                    "decision_index": index,
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
        if index > 0 and any(not len(o.net_returns) for o in observed):
            raise QuantError("Missing post-start NET history cannot receive a successful fallback.")
        weights, selection = choose_pair(observed, previous)
        targets[IDS[0]].loc[day] = combine(observed, equal)
        targets[IDS[1]].loc[day] = combine(observed, weights)
        audit = {
            "decision_session": str(day.date()),
            "decision_information_clock": CLOCK,
            "expert_target_known_at": [str(o.target_known_at.date()) for o in observed],
            "preserved_archive_identity_introduced_at": [
                o.identity.introduced_at for o in observed
            ],
            "equal_expert_weights": equal.tolist(),
            "previous_rolling_expert_weights": previous.tolist(),
            "rolling_expert_weights": weights.tolist(),
            "selection": selection,
            "counterfactual_historical_expert_availability": True,
            "shadow_wealth_added_to_main_capital": False,
            "source_feedback_floor_transferred_to_main_account": False,
            "independent_forward_validation": False,
        }
        audits.append(audit)
        if retain is not None:
            retain(
                {
                    "record": "decision_resolved",
                    "decision_index": index,
                    "audit": audit,
                    "asset_order": list(universe),
                    "complete_targets_binary64_hex": {
                        identifier: [float(v).hex() for v in frame.loc[day]]
                        for identifier, frame in targets.items()
                    },
                }
            )
        previous = weights
    return targets, audits
