from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import conditional_volatility_quintet_v1 as source
from us_quant.config import QuantError
from us_quant.retained_pair_equal_allocation_v1 import ASSETS
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import ExpertObservation, StrategyExpert

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/median-equity-budget-v1.json"
EXPERT_IDS, MECHANISMS, CLOCK = source.EXPERT_IDS, source.MECHANISMS, source.CLOCK
EQUITY = ("SPY", "MTUM", "VLUE", "QUAL", "USMV", "QQQ", "XLK", "SOXX", "IJR", "PKW")
TOLERANCE = 1e-12
PRIOR_HEAD = "8e105f6426b15eebe69269399662d031962bf83131570973d8d6ec385848aa63"
ID = "expert_pool_median_equity_budget_v1"
CANDIDATES = (
    {"id": ID, "weighting": "recorded_quintet_rolling_with_median_opinion_equity_budget"},
)


def expected_policy():
    old = read_json(source.POLICY)
    inherited = (
        "protocol_id",
        "expert_ids",
        "mechanism_groups",
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
        "current_opinion_risk_forecast_promoted",
        "total_target_cost_reference_promoted",
        "admitted_legacy_defensive_views_reused_unchanged",
        "all_source_native_RF_claimed_equal_without_view",
        "new_source_requests",
        "new_shadow_account_runs",
        "new_source_RF_views",
        "global_pool_changed",
        "study_size_is_a_total_research_quota",
        "independent_forward_validation",
        "investment_objective_verified",
        "order_authority",
    )
    return {
        **{k: old[k] for k in inherited},
        "schema_version": 1,
        "study_id": "ssd_median_equity_budget_20261106_v1",
        "authorization_id": "november-research-20261103-v1",
        "actual_native_event_seq": 183607,
        "prior_journal_sequence": 1369,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_completed_configurations": 293,
        "source_catalogue": {
            "path": (
                "reports/ssd-continuous-research-v1/median-equity-budget-20261106-v1/"
                "source-input-catalogue-v1.json"
            ),
            "sha256": "158182622bb8b6a73485136d103e8560ebd7d1ceb9006f2789a0dd47a0d0d2a3",
        },
        "upstream_candidate_id": source.CANDIDATES[1]["id"],
        "budget_operator": "median_current_complete_source_equity_allocation",
        "equity_symbols": list(EQUITY),
        "removed_equity_destination": "BIL",
        "relative_equity_weights_preserved": True,
        "other_non_equity_targets_preserved": True,
        "upstream_selection_state_recomputed_or_changed": False,
        "source_score_maturity_required_for_budget": False,
        "actual_holdings_or_drawdown_bound_claimed": False,
        "candidate_count_in_this_study": 1,
        "candidates": list(CANDIDATES),
    }


def validate(policy):
    expected = expected_policy()
    if policy != expected or any(type(policy[k]) is not type(v) for k, v in expected.items()):
        raise QuantError("Freeze one majority budget, not another quantile, pool or ranking law.")


def _vector(target):
    if (
        not isinstance(target, pd.Series)
        or tuple(target.index) != ASSETS
        or target.dtype.kind not in "fi"
        or not np.isfinite(target).all()
        or (target < 0).any()
        or (target > 0.98 + TOLERANCE).any()
        or abs(math.fsum(float(v) for v in target) - 0.98) > TOLERANCE
    ):
        raise QuantError("Require one complete finite long-only original14-asset target.")


def checked_opinions(observed: list[ExpertObservation], day: pd.Timestamp):
    if (
        not isinstance(day, pd.Timestamp)
        or pd.isna(day)
        or day.tzinfo is not None
        or day != day.normalize()
        or len(observed) != 5
        or any(not isinstance(o, ExpertObservation) for o in observed)
        or tuple(o.identity.identifier for o in observed) != EXPERT_IDS
        or tuple(o.identity.mechanism_group for o in observed) != MECHANISMS
        or any(
            o.as_of != day
            or not isinstance(o.target_known_at, pd.Timestamp)
            or pd.isna(o.target_known_at)
            or o.target_known_at.tzinfo is not None
            or o.target_known_at > day
            for o in observed
        )
    ):
        raise QuantError("All five complete source opinions must be known on this exact clock.")
    for opinion in observed:
        _vector(opinion.target)


def cap_target(original: pd.Series, observed: list[ExpertObservation], day: pd.Timestamp):
    _vector(original)
    checked_opinions(observed, day)
    if original.name != day:
        raise QuantError("Use the upstream target recorded for this exact decision only.")
    source_equity = [math.fsum(float(o.target[symbol]) for symbol in EQUITY) for o in observed]
    budget = sorted(source_equity)[2]
    before = math.fsum(float(original[symbol]) for symbol in EQUITY)
    target = original.copy()
    binding = before - budget > TOLERANCE
    if binding:
        scale = budget / before
        for symbol in EQUITY:
            target[symbol] = float(original[symbol]) * scale
        removed = before - math.fsum(float(target[symbol]) for symbol in EQUITY)
        target["BIL"] = float(original["BIL"]) + removed
    after = math.fsum(float(target[symbol]) for symbol in EQUITY)
    _vector(target)
    if (
        after - budget > TOLERANCE
        or after - before > TOLERANCE
        or any(target[s] != original[s] for s in ("IEF", "GLD", "TLT"))
        or target["BIL"] < original["BIL"]
    ):
        raise QuantError("Conserve the budget and non-equity targets; never add equity.")
    return target, {
        "decision_session": str(day.date()),
        "decision_information_clock": CLOCK,
        "source_equity_allocations": source_equity,
        "source_equity_binary64_hex": [v.hex() for v in source_equity],
        "median_equity_allocation": budget,
        "input_equity_allocation": before,
        "output_equity_allocation": after,
        "cap_binding": binding,
        "equity_transferred_to_BIL": before - after,
        "source_score_mature": [o.score_mature for o in observed],
        "score_maturity_required_for_budget": False,
        "upstream_selection_state_recomputed_or_changed": False,
        "actual_holdings_or_drawdown_bound_claimed": False,
    }


def independent_target(original, observed, day):
    _vector(original)
    checked_opinions(observed, day)

    def fraction(value):
        return Fraction.from_float(float(value))

    amounts = [float(sum((fraction(o.target[s]) for s in EQUITY), Fraction())) for o in observed]
    budget = sorted(amounts)[2]
    total = float(sum((fraction(original[s]) for s in EQUITY), Fraction()))
    expected = original.copy()
    if total - budget > TOLERANCE:
        values = {s: fraction(original[s]) * fraction(budget) / fraction(total) for s in EQUITY}
        for symbol, value in values.items():
            expected[symbol] = float(value)
        removed = fraction(total) - sum((fraction(expected[s]) for s in EQUITY), Fraction())
        expected["BIL"] = float(fraction(original["BIL"]) + removed)
    return expected, budget


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
            "asset_order": list(o.target.index) if o.target is not None else None,
            "target_binary64_hex": (
                [float(v).hex() for v in o.target] if o.target is not None else None
            ),
            "source_score_mature": o.score_mature,
        }
        for o in observed
    ]


def decision_sessions(calendar, policy):
    validate(policy)
    return source.decision_sessions(calendar, read_json(source.POLICY))


def build_targets(
    experts: tuple[StrategyExpert, ...],
    calendar: pd.DatetimeIndex,
    universe: tuple[str, ...],
    policy: dict,
    cached: pd.DataFrame,
    before_decision: Callable[[], None] | None = None,
    retain: Callable[[dict], None] | None = None,
):
    dates = decision_sessions(calendar, policy)
    if (
        len(experts) != 5
        or universe != ASSETS
        or not cached.index.equals(calendar)
        or tuple(cached.columns) != universe
        or not cached.dropna(how="all").index.equals(dates)
        or (cached.isna().any(axis=1) & ~cached.isna().all(axis=1)).any()
    ):
        raise QuantError("Keep the full original lattice and every cached upstream decision.")
    targets = pd.DataFrame(np.nan, index=calendar, columns=universe)
    audits = []
    for position, day in enumerate(dates):
        if before_decision is not None:
            before_decision()
        observed = [e.observe(day, 126) for e in experts]
        original = cached.loc[day]
        if retain is not None:
            retain(
                {
                    "record": "decision_input",
                    "decision_index": position,
                    "as_of": str(day.date()),
                    "observations": snapshot(observed),
                    "asset_order": list(universe),
                    "upstream_target_binary64_hex": [float(v).hex() for v in original],
                    "upstream_scores_weights_or_target_recomputed": False,
                }
            )
        target, audit = cap_target(original, observed, day)
        targets.loc[day] = target
        audits.append(audit)
        if retain is not None:
            retain(
                {
                    "record": "decision_resolved",
                    "decision_index": position,
                    "audit": audit,
                    "asset_order": list(universe),
                    "combined_targets": {ID: target.tolist()},
                }
            )
    return {ID: targets}, audits
