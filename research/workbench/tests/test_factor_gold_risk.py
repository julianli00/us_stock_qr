from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.calendar import is_month_end, next_session
from us_quant.config import QuantError
from us_quant.factor_gold_risk import monthly_targets, run_candidate, validate
from us_quant.multifactor_stability import FACTORS
from us_quant.storage import read_json


@pytest.fixture
def policy():
    return read_json(Path(__file__).parents[1] / "config/factor-gold-risk.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2017-06-30", ("SPY", "IEF", "GLD", "BIL", *FACTORS))


def test_fixed_sleeves_preserve_all_four_factors_without_leverage(market, policy):
    validate(policy)
    targets = monthly_targets(market, policy).dropna(how="all")
    assert all(is_month_end(day) for day in targets.index)
    assert np.allclose(targets.sum(axis=1), 0.98)
    assert (targets >= 0).all().all()
    equity = targets.loc[:, list(FACTORS)]
    assert np.allclose(equity.div(equity.sum(axis=1), axis=0), 0.25)
    assert equity.sum(axis=1).between(0.98 * 0.30 - 1e-12, 0.98 * 0.70 + 1e-12).all()
    assert targets[["SPY", "IEF"]].eq(0).all().all()


@pytest.mark.parametrize("candidate_index", [0, 1])
@pytest.mark.parametrize("delay,cost", [(1, 5), (2, 20)])
def test_dynamic_risk_decisions_replay_and_never_overlap(
    market, policy, candidate_index, delay, cost
):
    baseline = monthly_targets(market, policy)
    run, targets, decisions = run_candidate(
        market,
        baseline,
        policy["candidates"][candidate_index],
        policy,
        "2016-10-06",
        "2017-06-30",
        cost,
        delay,
    )
    assert (run.frame["cash"] >= 0).all()
    assert (run.weights.sum(axis=1) <= 1 + 1e-12).all()
    assert len({d["signal_session"] for d in decisions}) == len(decisions)
    for current, following in zip(decisions, decisions[1:], strict=False):
        assert following["signal_session"] >= current["execution_session"]
    for item in decisions:
        execution = pd.Timestamp(item["signal_session"])
        for _ in range(delay):
            execution = next_session(execution)
        assert str(execution.date()) == item["execution_session"]
    if delay == 2:
        assert run.frame.iloc[0]["equity"] == 10000
        assert run.frame.iloc[0]["orders"] == 0
    assert np.allclose(targets.dropna(how="all").sum(axis=1), 0.98)


def test_future_changes_do_not_change_earlier_daily_forecasts_or_decisions(market, policy):
    baseline = monthly_targets(market, policy)
    boundary = pd.Timestamp("2017-01-31")
    original = run_candidate(
        market, baseline, policy["candidates"][1], policy, "2016-10-06", "2017-06-30", 20, 2
    )
    close, opening = market.close.copy(), market.open.copy()
    later = close.index > boundary
    multipliers = np.linspace(1.0, 1.5, later.sum())
    close.loc[later, "GLD"] *= multipliers
    opening.loc[later, "GLD"] *= multipliers
    changed = replace(market, close=close, open=opening, raw_close=close.copy())
    after = run_candidate(
        changed,
        monthly_targets(changed, policy),
        policy["candidates"][1],
        policy,
        "2016-10-06",
        "2017-06-30",
        20,
        2,
    )
    pd.testing.assert_frame_equal(original[0].frame.loc[:boundary], after[0].frame.loc[:boundary])
    pd.testing.assert_frame_equal(original[1].loc[:boundary], after[1].loc[:boundary])
    assert [d for d in original[2] if d["signal_session"] <= str(boundary.date())] == [
        d for d in after[2] if d["signal_session"] <= str(boundary.date())
    ]


def test_no_result_driven_changes_to_targets_or_number_of_cases(policy):
    policy["candidates"][1]["daily_volatility_target"] = 0.15
    with pytest.raises(QuantError):
        validate(policy)
