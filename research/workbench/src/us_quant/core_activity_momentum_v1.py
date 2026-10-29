from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.anfci_residual_confirmation_v1 import CORE, UNIVERSE, donor_budget
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.daily_variance_disagreement_v1 import FACTOR_IDS
from us_quant.data import MarketData

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/core-activity-momentum-v1.json"
PRIOR_HEAD = "284115a28a56ec402ef90f43dc9c55ddaf3223119c2bb055ab8e2025ff1513e5"
SHARES = (0.35, 0.30, 0.20, 0.15)
CANDIDATES = (
    {"id": "two_segment_core_price_momentum_control_v1", "score": "uniform_time"},
    {"id": "two_segment_core_activity_momentum_v1", "score": "reported_price_volume"},
)


def validate(policy):
    expected = {
        "schema_version": 1,
        "study_id": "ssd_core_activity_momentum_20261029_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 198,
        "prior_journal_sequence": 240,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "source_candidate": "dollar_option_two_segment_variation_v1",
        "lookback_sessions": 126,
        "skip_recent_sessions": 21,
        "score_return_sessions": 105,
        "ranking_score_decimal_places": 12,
        "ranked_factor_shares": list(SHARES),
        "original_source_months": 131,
        "eligible_source_months": 128,
        "undefined_prehistory_source_months": 3,
        "core_symbols": list(CORE),
        "factor_ids": list(FACTOR_IDS),
        "new_economic_factor_definitions": 0,
        "cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "candidate_count_in_this_study": 2,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
    }
    if any(
        policy.get(k) != v or type(policy.get(k)) is not type(v) for k, v in expected.items()
    ) or any(
        policy.get("methodology", {}).get(k) is not False
        for k in (
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("Keep the fixed paired activity question, ranks and original acceptance.")


def score(closes, raw_closes, volumes, method):
    close, raw, volume = (np.asarray(x, dtype=float) for x in (closes, raw_closes, volumes))
    if (
        close.shape != (106,)
        or raw.shape != (105,)
        or volume.shape != (105,)
        or not all(np.isfinite(x).all() and (x > 0).all() for x in (close, raw, volume))
        or method not in {"uniform_time", "reported_price_volume"}
    ):
        raise QuantError("Require the identical105return/activity dates and complete prices.")
    returns = np.diff(np.log(close))
    activity = np.ones(105) if method == "uniform_time" else raw * volume
    if not np.isfinite(activity).all() or (activity <= 0).any():
        raise QuantError("Invalid activity cannot be replaced by uniform weights.")
    weights = activity / activity.max()
    value = float(np.dot(weights, returns) / weights.sum())
    if not np.isfinite(value):
        raise QuantError("A nonfinite mean-log score cannot receive a valid rank.")
    return value


def rank(scores):
    if set(scores) != set(CORE) or not all(math.isfinite(value) for value in scores.values()):
        raise QuantError("All four actual factor scores must be present.")
    return sorted(CORE, key=lambda name: (-round(float(scores[name]), 12), name))


def eligible_dates(data, desired):
    return pd.DatetimeIndex(
        [day for day in desired.dropna(how="all").index if data.close.index.get_loc(day) >= 126]
    )


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
        raise QuantError("Preserve the full calendar and exact sparse parent targets.")
    dates = eligible_dates(data, desired)
    if dates.empty:
        raise QuantError("No original decision has the complete declared score prehistory.")
    targets, audits = {}, {}
    for candidate in CANDIDATES:
        target = pd.DataFrame(np.nan, index=desired.index, columns=desired.columns)
        records = []
        for day in dates:
            pos = data.close.index.get_loc(day)
            prior = data.close.index[pos - 126]
            end = data.close.index[pos - 21]
            row = desired.loc[day].copy()
            donor_budget(row)
            budget = float(row.loc[list(CORE)].sum())
            values = {
                name: score(
                    data.close[name].iloc[pos - 126 : pos - 20],
                    data.raw_close[name].iloc[pos - 125 : pos - 20],
                    data.volume[name].iloc[pos - 125 : pos - 20],
                    candidate["score"],
                )
                for name in CORE
            }
            order = rank(values)
            for name, share in zip(order, SHARES, strict=True):
                row[name] = budget * share
            if abs(float(row.loc[list(CORE)].sum()) - budget) > 1e-12:
                raise QuantError("Ranking cannot alter the original aggregate style budget.")
            target.loc[day] = row
            records.append(
                {
                    "decision_session": str(day.date()),
                    "first_prior_close": str(prior.date()),
                    "last_score_and_activity_session": str(end.date()),
                    "return_sessions": 105,
                    "method": candidate["score"],
                    "scores": values,
                    "ranking_scores": {name: round(values[name], 12) for name in CORE},
                    "ranking": order,
                    "source_core_budget": budget,
                    "all_noncore_targets_preserved": True,
                    "activity_is_actual_flow_or_constituent_turnover": False,
                }
            )
        targets[candidate["id"]], audits[candidate["id"]] = target, records
    return targets, audits


def independent_row(data, old, day, method):
    pos = data.close.index.get_loc(day)
    if pos < 126 or method not in {"uniform_time", "reported_price_volume"}:
        raise QuantError("Independent scoring cannot fill missing126session prehistory.")
    scores = {}
    for name in CORE:
        values, weights = [], []
        for i in range(pos - 125, pos - 20):
            values.append(
                math.log(float(data.close[name].iloc[i]) / float(data.close[name].iloc[i - 1]))
            )
            weights.append(
                1.0
                if method == "uniform_time"
                else float(data.raw_close[name].iloc[i]) * float(data.volume[name].iloc[i])
            )
        normalizer = max(weights)
        if (
            not math.isfinite(normalizer)
            or normalizer <= 0
            or not all(math.isfinite(value) and value > 0 for value in weights)
        ):
            raise QuantError("Independent activity weights must be finite and positive.")
        weights = [value / normalizer for value in weights]
        scores[name] = math.fsum(
            w * value for w, value in zip(weights, values, strict=True)
        ) / math.fsum(weights)
    order = sorted(CORE, key=lambda name: (-round(scores[name], 12), name))
    budget = math.fsum(float(old[name]) for name in CORE)
    row = old.copy()
    for name, share in zip(order, SHARES, strict=True):
        row[name] = budget * share
    return row, scores, order
