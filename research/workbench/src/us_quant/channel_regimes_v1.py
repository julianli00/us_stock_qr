from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.equity_first_regimes_v1 import COMPOSITION, UNIVERSE
from us_quant.semiconductor_trend_transfer import equity_signal

POLICY = Path(__file__).resolve().parents[2] / "config/channel-regimes-v1.json"
CANDIDATES = (
    {"id": "equity_channel63x21_goldtrend200_v1", "gold_rule": "sma200"},
    {"id": "equity_channel63x21_goldchannel63x21_v1", "gold_rule": "channel63x21"},
)
PRIOR_JOURNAL_HEAD = "c0001df2d730df1d92824c1cd4b55a032818b0c478e6b76ba5712432a43b3239"


def validate(policy: dict) -> None:
    required = {
        "schema_version": 1,
        "study_id": "ssd_channel_regimes_20261027_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 182,
        "prior_new_lane_registered_candidates": 4,
        "prior_new_lane_completed_configurations": 4,
        "prior_journal_sequence": 47,
        "prior_journal_head_sha256": PRIOR_JOURNAL_HEAD,
        "new_economic_factor_definitions": 0,
        "minimum_economic_factor_families": 3,
        "entry_prior_observations": 63,
        "exit_prior_observations": 21,
        "gold_trend_observations": 200,
        "common_warmup_observations": 253,
        "cash_reserve": 0.02,
        "candidate_count_in_this_study": 2,
        "candidates": list(CANDIDATES),
    }
    if (
        any(policy.get(key) != value for key, value in required.items())
        or policy.get("study_size_is_a_total_research_quota") is not False
        or any(
            policy.get("methodology", {}).get(key) is not False
            for key in (
                "independent_forward_validation",
                "investment_objective_verified",
                "order_authority",
                "automatic_live_deployment",
            )
        )
    ):
        raise QuantError("Keep the exact fixed channel pair, prior trials and authority boundary.")


def validate_inputs(data: MarketData) -> None:
    data.validate()
    if not data.close.columns.is_unique or set(data.close.columns) != UNIVERSE:
        raise QuantError("Require the original complete twelve-asset admitted panel.")


def channel_state(level: pd.Series) -> pd.Series:
    if (
        not level.index.is_unique
        or not level.index.is_monotonic_increasing
        or not np.isfinite(level).all()
        or (level <= 0).any()
    ):
        raise QuantError("A channel needs ordered, unique, finite positive observed levels.")
    upper = level.shift(1).rolling(63, min_periods=63).max()
    lower = level.shift(1).rolling(21, min_periods=21).min()
    state = False
    result = pd.Series(np.nan, index=level.index, dtype=float)
    for position in range(63, len(level)):
        if level.iloc[position] > upper.iloc[position]:
            state = True
        elif level.iloc[position] < lower.iloc[position]:
            state = False
        result.iloc[position] = float(state)
    return result


def build_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    validate(policy)
    validate_inputs(data)
    equity = channel_state(equity_signal(data.close))
    gold_channel = channel_state(data.close["GLD"])
    gold_trend = data.close["GLD"] > data.close["GLD"].rolling(200, min_periods=200).mean()
    result = {}
    for candidate in CANDIDATES:
        stock = equity.eq(1)
        gold = gold_channel.eq(1) if candidate["gold_rule"] == "channel63x21" else gold_trend
        target = pd.DataFrame(0.0, index=data.close.index, columns=data.close.columns)
        for symbol, share in COMPOSITION.items():
            target[symbol] = 0.98 * share * stock
        target["GLD"] = 0.98 * (~stock & gold)
        target["BIL"] = 0.98 * (~stock & ~gold)
        target.iloc[:252] = np.nan
        result[candidate["id"]] = target
    return result


def independent_targets(data: MarketData, policy: dict) -> dict[str, pd.DataFrame]:
    validate(policy)
    validate_inputs(data)
    columns = list(data.close.columns)
    positions = {symbol: columns.index(symbol) for symbol in COMPOSITION}
    gold_column, bill_column = columns.index("GLD"), columns.index("BIL")
    prices = data.close.to_numpy()
    equity_levels = [1.0]
    equity_on = gold_on = False
    result = {
        candidate["id"]: np.full(prices.shape, np.nan, dtype=float) for candidate in CANDIDATES
    }
    for position in range(len(prices)):
        if position:
            change = sum(
                share * (prices[position, column] / prices[position - 1, column] - 1)
                for symbol, share in COMPOSITION.items()
                for column in (positions[symbol],)
            )
            equity_levels.append(equity_levels[-1] * (1 + change))
        if position < 63:
            continue
        current = equity_levels[-1]
        if current > max(equity_levels[position - 63 : position]):
            equity_on = True
        elif current < min(equity_levels[position - 21 : position]):
            equity_on = False
        gold = prices[position, gold_column]
        if gold > max(prices[position - 63 : position, gold_column]):
            gold_on = True
        elif gold < min(prices[position - 21 : position, gold_column]):
            gold_on = False
        if position < 252:
            continue
        gold_trend = gold > sum(prices[position - 199 : position + 1, gold_column]) / 200
        for candidate in CANDIDATES:
            row = np.zeros(len(columns))
            if equity_on:
                for symbol, share in COMPOSITION.items():
                    row[positions[symbol]] = 0.98 * share
            else:
                defense = gold_on if candidate["gold_rule"] == "channel63x21" else gold_trend
                row[gold_column if defense else bill_column] = 0.98
            result[candidate["id"]][position] = row
    return {
        identifier: pd.DataFrame(values, index=data.close.index, columns=data.close.columns)
        for identifier, values in result.items()
    }
