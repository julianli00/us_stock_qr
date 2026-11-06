from __future__ import annotations

import math
from pathlib import Path
from types import FunctionType

import numpy as np
import pandas as pd

from us_quant import commitment_macro_defense_hedge_v1 as prior
from us_quant import strategy_expert_pool_v5 as calendar_source
from us_quant.config import QuantError
from us_quant.storage import read_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/expert-total-target-cost-v1.json"
EXPERT_IDS, MECHANISMS, CLOCK = prior.EXPERT_IDS, prior.MECHANISMS, prior.CLOCK
ID = "expert_pool_commitment_total_target_cost_pair_v1"
CANDIDATES = ({"id": ID, "weighting": "train_erc_validate_net_total_nominal_target_cost"},)
PRIOR_HEAD = "66ec26b9a626e4b4ef2a4ae7447e2f492676fbb8343282850879a97991b81796"
combine = prior.combine


def expected_policy():
    source = read_json(prior.POLICY)
    inherited = (
        "protocol_id",
        "expert_ids",
        "mechanism_groups",
        "original_source_slots",
        "direct_commitment_source_slot",
        "initial_decision_session",
        "lookback_sessions",
        "training_sessions",
        "validation_sessions",
        "selected_experts",
        "mature_pair_count",
        "selection_score_decimal_places",
        "shadow_reference_path",
        "reference_capital_usd",
        "switch_cost_bps_per_side",
        "switch_commission_per_changed_asset_usd",
        "switch_cost_amortization_sessions",
        "changed_asset_numeric_tolerance",
        "invested_target_budget",
        "cash_reserve",
        "actual_weight_trade_band",
        "minimum_economic_factor_families",
        "decision_information_clock",
        "admitted_legacy_defensive_views_reused_unchanged",
        "all_source_native_RF_claimed_equal_without_view",
        "mechanism_labels_are_independent_risks",
        "nested_growth_controller_included",
        "current_opinion_risk_forecast_promoted",
        "new_source_requests",
        "new_shadow_account_runs",
        "new_source_RF_views",
        "study_size_is_a_total_research_quota",
        "global_pool_changed",
        "independent_forward_validation",
        "investment_objective_verified",
        "order_authority",
    )
    return {
        **{k: source[k] for k in inherited},
        "schema_version": 1,
        "study_id": "ssd_expert_total_target_cost_20261106_v1",
        "actual_native_event_seq": 182343,
        "prior_journal_sequence": 1343,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_completed_configurations": 291,
        "source_inputs": {
            "path": (
                "reports/ssd-continuous-research-v1/expert-total-target-cost-20261106-v1/"
                "source-inputs-v1.json"
            ),
            "sha256": "f48860ecd93ff2b1b00e4506d3c902ea406e6963a552c901b54cb300f53cfce3",
        },
        "cost_reference": "previous_origin_final_desired_meta_target",
        "cost_proxy_is_actual_holdings_NAV_or_main_fees": False,
        "source_NET_fees_already_in_history": True,
        "coefficient_window_or_risk_weighting_tuned": False,
        "candidate_count_in_this_study": 1,
        "candidates": list(CANDIDATES),
        "cached_equal_and_rolling_controls_replayed": False,
    }


def validate(policy):
    expected = expected_policy()
    if policy != expected or any(type(policy[k]) is not type(v) for k, v in expected.items()):
        raise QuantError("Version only the nominal cost reference, not the pool or fees/windows.")


def nominal_cost(target: pd.Series, previous_target: pd.Series):
    if (
        not isinstance(target, pd.Series)
        or not isinstance(previous_target, pd.Series)
        or not target.index.is_unique
        or not target.index.equals(previous_target.index)
        or not np.isfinite(target).all()
        or not np.isfinite(previous_target).all()
        or (target < 0).any()
        or (previous_target < 0).any()
        or abs(math.fsum(target) - 0.98) > 1e-12
        or abs(math.fsum(previous_target) - 0.98) > 1e-12
    ):
        raise QuantError("Compare exact funded desired targets, never invented actual holdings.")
    delta = abs(target - previous_target).to_numpy()
    return float(delta.sum() * 5 / 10000 + (delta > 1e-12).sum() / 10000), int(
        (delta > 1e-12).sum()
    )


def choose_pair(observed, previous_weights, previous_target):
    if previous_target is None and observed[0].as_of != pd.Timestamp("2016-10-05"):
        raise QuantError("A post-initial decision needs its actual preceding desired target.")
    source_weights, original = prior.choose_pair(observed, previous_weights)
    if not original["scores_available"]:
        return source_weights, {
            **original,
            "cost_reference": "previous_origin_final_desired_meta_target",
            "cost_proxy_is_actual_main_account_cost": False,
        }
    if previous_target is None:
        raise QuantError("A mature assessment cannot invent a previous desired target.")
    excess = np.column_stack([(o.net_returns - o.risk_free).to_numpy() for o in observed])
    records, usable = [], []
    replaced = {
        "train_net_risk_score",
        "validation_net_risk_score",
        "selection_score",
        "selection_score_key",
        "estimated_incremental_switch_cost_fraction",
        "estimated_incremental_changed_assets",
    }
    for source_row in original["pairs"]:
        row = {k: v for k, v in source_row.items() if k not in replaced}
        if "expert_weights" in source_row:
            weights = np.asarray(source_row["expert_weights"])
            target = combine(observed, weights)
            cost, changed = nominal_cost(target, previous_target)
            scores = []
            for sample in (excess[:84], excess[84:]):
                portfolio = sample @ weights
                sigma = float(np.std(portfolio, ddof=1))
                if sigma <= 0 or not math.isfinite(sigma):
                    raise QuantError("The unchanged scored pair must keep finite positive risk.")
                scores.append(float(math.sqrt(252) * (np.mean(portfolio) - cost / 21) / sigma))
            if not np.isfinite(scores).all():
                raise QuantError("Nonfinite total-target cost scores cannot receive a fallback.")
            row.update(
                {
                    "train_net_risk_score": scores[0],
                    "validation_net_risk_score": scores[1],
                    "selection_score": min(scores),
                    "selection_score_key": round(min(scores), 12),
                    "estimated_total_nominal_target_cost_fraction": cost,
                    "estimated_total_nominal_changed_assets": changed,
                    "same_origin_incremental_cost_fraction_reference": source_row[
                        "estimated_incremental_switch_cost_fraction"
                    ],
                    "same_origin_incremental_changed_assets_reference": source_row[
                        "estimated_incremental_changed_assets"
                    ],
                    "same_origin_reference_train_net_risk_score": source_row[
                        "train_net_risk_score"
                    ],
                    "same_origin_reference_validation_net_risk_score": source_row[
                        "validation_net_risk_score"
                    ],
                    "cost_reference": "previous_origin_final_desired_meta_target",
                    "cost_proxy_is_actual_main_account_cost": False,
                }
            )
            usable.append(row)
        records.append(row)
    if not usable:
        raise QuantError("A previously scored menu cannot become an empty successful selection.")
    chosen = min(usable, key=lambda row: (-row["selection_score_key"], row["pair"]))
    return np.asarray(chosen["expert_weights"]), {
        **original,
        "reason": "rolling_trained_pair_total_nominal_target_cost",
        "pairs": records,
        "selected_pair": chosen["pair"],
        "cost_reference": "previous_origin_final_desired_meta_target",
        "cost_proxy_is_actual_main_account_cost": False,
    }


def decision_sessions(calendar, policy):
    return FunctionType(
        calendar_source.decision_sessions.__code__,
        {**calendar_source.decision_sessions.__globals__, **globals()},
        argdefs=calendar_source.decision_sessions.__defaults__,
    )(calendar, policy)


def snapshot(observed):
    return [
        {
            "id": o.identity.identifier,
            "mechanism_group": o.identity.mechanism_group,
            "introduced_at": o.identity.introduced_at,
            "as_of": str(o.as_of.date()),
            "target_known_at": str(o.target_known_at.date())
            if o.target_known_at is not None
            else None,
            "target": o.target.tolist() if o.target is not None else None,
            "asset_order": list(o.target.index) if o.target is not None else None,
            "return_dates": o.net_returns.index.strftime("%Y-%m-%d").tolist(),
            "NET_returns": o.net_returns.tolist(),
            "RF": o.risk_free.tolist(),
            "score_mature": o.score_mature,
        }
        for o in observed
    ]


def build_targets(experts, calendar, universe, policy, before_decision=None, retain=None):
    validate(policy)
    dates = decision_sessions(calendar, policy)
    if (
        len(experts) != 4
        or len(universe) != len(set(universe))
        or tuple(e.identity.identifier for e in experts) != EXPERT_IDS
    ):
        raise QuantError("Keep the completed direct quartet and original unique coordinates.")
    target_frame = pd.DataFrame(np.nan, index=calendar, columns=universe)
    previous_weights, previous_target = np.full(4, 0.25), None
    audits = []
    for index, day in enumerate(dates):
        if before_decision is not None:
            before_decision()
        observed = [e.observe(day, 126) for e in experts]
        if tuple(o.identity.mechanism_group for o in observed) != MECHANISMS or any(
            o.target is None or tuple(o.target.index) != universe for o in observed
        ):
            raise QuantError("No missing expert or asset coordinate may be dropped or cash filled.")
        if retain is not None:
            retain(
                {
                    "record": "decision_input",
                    "decision_index": index,
                    "as_of": str(day.date()),
                    "information_clock": CLOCK,
                    "previous_weights": previous_weights.tolist(),
                    "previous_weights_binary64_hex": [float(x).hex() for x in previous_weights],
                    "previous_nominal_target": previous_target.tolist()
                    if previous_target is not None
                    else None,
                    "previous_nominal_target_is_actual_holdings": False,
                    "observations": snapshot(observed),
                    "binary64_values_truncated_to_fixed_decimals": False,
                }
            )
        weights, selection = choose_pair(observed, previous_weights, previous_target)
        target = combine(observed, weights)
        target_frame.loc[day] = target
        audit = {
            "decision_session": str(day.date()),
            "decision_role": "initial_capital" if index == 0 else "completed_exchange_month_end",
            "decision_information_clock": CLOCK,
            "expert_target_known_at": [str(o.target_known_at.date()) for o in observed],
            "preserved_archive_identity_introduced_at": [
                o.identity.introduced_at for o in observed
            ],
            "expert_shadow_last_observation": [
                str(o.net_returns.index[-1].date()) if len(o.net_returns) else None
                for o in observed
            ],
            "equal_expert_weights": [0.25] * 4,
            "previous_rolling_expert_weights": previous_weights.tolist(),
            "rolling_expert_weights": weights.tolist(),
            "previous_nominal_meta_target": previous_target.tolist()
            if previous_target is not None
            else None,
            "previous_nominal_target_is_actual_holdings": False,
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
                    "combined_targets": {ID: target.tolist()},
                }
            )
        previous_weights, previous_target = weights, target.copy()
    return {ID: target_frame}, audits
