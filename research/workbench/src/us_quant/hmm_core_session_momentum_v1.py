from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.core_activity_momentum_v1 import CORE, SHARES, UNIVERSE, rank
from us_quant.daily_variance_disagreement_v1 import FACTOR_IDS
from us_quant.data import MarketData

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/hmm-core-session-momentum-v1.json"
PRIOR_HEAD = "bf398d358dd17166d6c404323c1889673af61810d77c36a5b05b159be45416c7"
CANDIDATES = (
    {"id": "hmm_core_close_momentum_control_v1", "score": "close_to_close"},
    {"id": "hmm_core_overnight_momentum_v1", "score": "overnight"},
    {"id": "hmm_core_intraday_momentum_v1", "score": "intraday"},
)
METHODS = tuple(row["score"] for row in CANDIDATES)


def validate(policy):
    expected = {
        "schema_version": 1,
        "study_id": "ssd_hmm_core_session_momentum_20261029_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 201,
        "prior_registered_ssd_configurations": 24,
        "prior_journal_sequence": 275,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "source_candidate": "causal_two_state_growth_bil_v1",
        "lookback_sessions": 126,
        "skip_recent_sessions": 21,
        "score_return_sessions": 105,
        "ranking_score_decimal_places": 12,
        "ranked_factor_shares": list(SHARES),
        "original_calendar_months": 131,
        "source_target_months": 122,
        "eligible_source_months": 122,
        "additional_source_months_dropped": 0,
        "undefined_prehistory_calendar_months": 9,
        "core_symbols": list(CORE),
        "factor_ids": list(FACTOR_IDS),
        "new_economic_factor_definitions": 0,
        "cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "candidate_count_in_this_study": 3,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
    }
    if any(
        policy.get(key) != value or type(policy.get(key)) is not type(value)
        for key, value in expected.items()
    ) or any(
        policy.get("methodology", {}).get(key) is not False
        for key in (
            "source_generator_recomputed",
            "paper_replication_or_new_independent_alpha_claimed",
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("Keep the three frozen session methods, source budget and original gates.")


def score_components(closes, opens):
    close, opening = np.asarray(closes, dtype=float), np.asarray(opens, dtype=float)
    if (
        close.shape != (106,)
        or opening.shape != (105,)
        or not all(np.isfinite(a).all() and (a > 0).all() for a in (close, opening))
    ):
        raise QuantError(
            "Each component requires all105matched returns and positive adjusted prices."
        )
    log_close, log_open = np.log(close), np.log(opening)
    result = {
        "close_to_close": float(np.diff(log_close).mean()),
        "overnight": float((log_open - log_close[:-1]).mean()),
        "intraday": float((log_close[1:] - log_open).mean()),
    }
    if (
        not all(math.isfinite(value) for value in result.values())
        or abs(result["close_to_close"] - result["overnight"] - result["intraday"]) > 1e-12
    ):
        raise QuantError(
            "The signed adjusted session components must reconcile to their daily sum."
        )
    return result


def source_core_budget(row):
    if (
        tuple(row.index) != UNIVERSE
        or not np.isfinite(row).all()
        or (row < 0).any()
        or abs(float(row.sum()) - 0.98) > 1e-12
    ):
        raise QuantError("The saved HMM carrier must remain a fully specified funded target.")
    budget = math.fsum(float(row[name]) for name in CORE)
    if (
        not 0.147 / 2 - 1e-12 <= budget <= 0.98 / 2 + 1e-12
        or abs(float(row["SOXX"]) - budget) > 1e-12
        or abs(float(row["BIL"]) - (0.98 - 2 * budget)) > 1e-12
        or any(abs(float(row[name]) - budget / 4) > 1e-12 for name in CORE)
        or any(row[name] != 0 for name in row.index if name not in {*CORE, "SOXX", "BIL"})
    ):
        raise QuantError("Do not change the stored HMM floor, equal core, SOXX or BIL allocation.")
    return budget


def eligible_dates(data, desired):
    dates = desired.dropna(how="all").index
    if not len(dates) or any(data.close.index.get_loc(day) < 126 for day in dates):
        raise QuantError("Do not drop a source month or invent missing126session prehistory.")
    return pd.DatetimeIndex(dates)


def build_targets(data: MarketData, desired: pd.DataFrame, policy: dict):
    validate(policy)
    data.validate()
    if (
        tuple(data.close.columns) != UNIVERSE
        or not data.close.index.equals(sessions(data.close.index[0], data.close.index[-1]))
        or not desired.index.equals(data.close.index)
        or not desired.columns.equals(data.close.columns)
        or (desired.isna().any(axis=1) & ~desired.isna().all(axis=1)).any()
        or not all(is_month_end(day) for day in desired.dropna(how="all").index)
    ):
        raise QuantError("Keep the complete market and exact sparse source decision calendar.")
    dates = eligible_dates(data, desired)
    targets = {
        candidate["id"]: pd.DataFrame(np.nan, index=desired.index, columns=desired.columns)
        for candidate in CANDIDATES
    }
    audits = {candidate["id"]: [] for candidate in CANDIDATES}
    for day in dates:
        pos = data.close.index.get_loc(day)
        budget = source_core_budget(desired.loc[day])
        components = {
            name: score_components(
                data.close[name].iloc[pos - 126 : pos - 20],
                data.open[name].iloc[pos - 125 : pos - 20],
            )
            for name in CORE
        }
        for candidate in CANDIDATES:
            method, identifier = candidate["score"], candidate["id"]
            values = {name: components[name][method] for name in CORE}
            order = rank(values)
            row = desired.loc[day].copy()
            for name, share in zip(order, SHARES, strict=True):
                row[name] = budget * share
            if abs(math.fsum(float(row[name]) for name in CORE) - budget) > 1e-12:
                raise QuantError("Session ranking cannot change the aggregate core budget.")
            targets[identifier].loc[day] = row
            audits[identifier].append(
                {
                    "decision_session": str(day.date()),
                    "first_prior_close": str(data.close.index[pos - 126].date()),
                    "last_return_session": str(data.close.index[pos - 21].date()),
                    "return_sessions": 105,
                    "method": method,
                    "scores": values,
                    "component_scores": components,
                    "ranking_scores": {name: round(values[name], 12) for name in CORE},
                    "ranking": order,
                    "source_core_budget": budget,
                    "all_noncore_targets_preserved": True,
                    "hmm_refitted_or_states_recomputed": False,
                    "partial_session_trading_account_claimed": False,
                }
            )
    return targets, audits


def independent_row(data, old, day, method):
    pos = data.close.index.get_loc(day)
    if pos < 126 or method not in METHODS:
        raise QuantError(
            "Independent component scoring needs the complete frozen window and method."
        )
    components = {}
    for name in CORE:
        values = {key: [] for key in METHODS}
        for i in range(pos - 125, pos - 20):
            previous = float(data.close[name].iloc[i - 1])
            opening = float(data.open[name].iloc[i])
            closing = float(data.close[name].iloc[i])
            if not all(math.isfinite(x) and x > 0 for x in (previous, opening, closing)):
                raise QuantError(
                    "Invalid observed prices cannot be replaced by zero-return sessions."
                )
            values["close_to_close"].append(math.log(closing / previous))
            values["overnight"].append(math.log(opening / previous))
            values["intraday"].append(math.log(closing / opening))
        components[name] = {key: math.fsum(values[key]) / 105 for key in METHODS}
    scores = {name: components[name][method] for name in CORE}
    order = sorted(CORE, key=lambda name: (-round(scores[name], 12), name))
    budget = math.fsum(float(old[name]) for name in CORE)
    row = old.copy()
    for name, share in zip(order, SHARES, strict=True):
        row[name] = budget * share
    return row, scores, order, components
