from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.backtest import simulate
from us_quant.bt_audit import independent_equity
from us_quant.config import QuantError
from us_quant.dual_horizon import seed_window
from us_quant.factor_gold_risk import monthly_targets
from us_quant.factor_momentum_comparison import BASELINE, build_targets, validate_policy
from us_quant.factor_research import factor_scores
from us_quant.multifactor_stability import FACTORS
from us_quant.storage import file_digest, read_json
from us_quant.strategy_replay import registered_generator, registered_targets

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/factor-momentum-comparison.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2018-06-29", ("SPY", "IEF", "GLD", "BIL", *FACTORS))


def test_all_four_factors_retain_bounded_shares_and_exact_matched_budgets(market, policy):
    targets = build_targets(market, policy)
    baseline = monthly_targets(market, read_json(BASELINE))
    control, residual = (targets[row["id"]].dropna(how="all") for row in policy["candidates"])
    assert control.index.equals(residual.index)
    assert control.index[0] <= pd.Timestamp("2016-09-30")
    assert control.index.min() > market.close.index[251]
    pd.testing.assert_frame_equal(
        control[["SPY", "IEF", "GLD", "BIL"]], residual[["SPY", "IEF", "GLD", "BIL"]]
    )
    for frame in (control, residual):
        equity = frame.loc[:, list(FACTORS)].sum(axis=1)
        np.testing.assert_allclose(equity, baseline.loc[frame.index, list(FACTORS)].sum(axis=1))
        shares = frame.loc[:, list(FACTORS)].div(equity, axis=0)
        np.testing.assert_allclose(
            np.sort(shares.to_numpy(), axis=1),
            np.tile([0.15, 0.20, 0.30, 0.35], (len(frame), 1)),
        )
        assert (frame >= 0).all().all()
        np.testing.assert_allclose(frame.sum(axis=1), 0.98, rtol=0, atol=1e-12)
    assert not control.loc[:, list(FACTORS)].equals(residual.loc[:, list(FACTORS)])


def test_future_prices_rates_and_ohlc_cannot_change_past_targets(market, policy):
    original = build_targets(market, policy)
    cutoff = pd.Timestamp("2017-06-30")
    close, opening = market.close.copy(), market.open.copy()
    future = close.index > cutoff
    close.loc[future, ["SPY", "QUAL", "GLD"]] *= np.linspace(1, 1.4, future.sum())[:, None]
    opening.loc[future, ["SPY", "QUAL", "GLD"]] *= np.linspace(1, 1.4, future.sum())[:, None]
    rates = market.risk_free.copy()
    rates.loc[future] += 0.0001
    changed = replace(market, open=opening, close=close, raw_close=close.copy(), risk_free=rates)
    updated = build_targets(changed, policy)
    for name in original:
        pd.testing.assert_frame_equal(original[name].loc[:cutoff], updated[name].loc[:cutoff])


def test_only_preceding_information_is_used_at_a_fresh_account_anchor(market, policy):
    cutoff = pd.Timestamp("2016-10-05")
    full = build_targets(market, policy)
    past = replace(
        market,
        open=market.open.loc[:cutoff],
        close=market.close.loc[:cutoff],
        raw_close=market.raw_close.loc[:cutoff],
        volume=market.volume.loc[:cutoff],
        risk_free=market.risk_free.loc[:cutoff],
    )
    truncated = build_targets(past, policy)
    for name in full:
        expected = seed_window(full[name], "2016-10-06").loc[cutoff]
        observed = seed_window(truncated[name], "2016-10-06").loc[cutoff]
        pd.testing.assert_series_equal(expected, observed)


def test_residual_scores_remove_a_fitted_beta_but_holdings_do_not_hedge(market, policy):
    history = market.close.copy()
    daily = history.pct_change(fill_method=None).iloc[1:]
    equity_excess = daily["SPY"] - market.risk_free.loc[daily.index]
    for symbol, beta in zip(FACTORS, (0.5, 0.8, 1.1, 1.4)):
        returns = market.risk_free.loc[daily.index] + 0.0002 + beta * equity_excess
        history.loc[daily.index, symbol] = 100 * np.cumprod(1 + returns.to_numpy())
        history.loc[history.index[0], symbol] = 100
    scores = factor_scores(history, market.risk_free, policy["score_parameters"])
    assert scores["momentum"].nunique() > 1
    np.testing.assert_allclose(scores["residual"], 0, rtol=0, atol=1e-8)


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_registered_source_replay_and_independent_accounts(market, policy, cost, delay):
    candidate = policy["candidates"][1]
    source = ROOT / "src/us_quant/factor_momentum_comparison.py"
    config = ROOT / "config/factor-momentum-comparison.json"
    spec = {
        "id": candidate["id"],
        "configuration": candidate,
        "frozen_files": {
            path.relative_to(ROOT).as_posix(): file_digest(path)
            for path in (
                source,
                config,
                BASELINE,
                source.with_name("factor_gold_risk.py"),
                source.with_name("factor_research.py"),
            )
        },
    }
    assert registered_generator(spec, ROOT)[0] == "factor_momentum_comparison"
    missing_helper = {
        **spec,
        "frozen_files": {
            key: value
            for key, value in spec["frozen_files"].items()
            if key != "src/us_quant/factor_research.py"
        },
    }
    with pytest.raises(QuantError, match="missing a frozen helper"):
        registered_generator(missing_helper, ROOT)
    first, last = "2016-10-06", "2017-03-31"
    targets = registered_targets(spec, market, first, last, cost, delay, ROOT, {})
    expected = seed_window(build_targets(market, policy)[candidate["id"]], first)
    pd.testing.assert_frame_equal(targets, expected)
    own = simulate(
        market, targets, first, last, initial_capital=10000, cost_bps=cost, commission=1, delay=delay
    )
    independent = independent_equity(
        market, targets, first, last, capital=10000, cost_bps=cost, commission=1, delay=delay
    )
    np.testing.assert_allclose(own.frame["equity"], independent["equity"], rtol=0, atol=1e-8)
    assert own.frame["cost"].sum() > 0
    assert own.frame["cash"].min() >= 0
    if delay == 1:
        assert own.frame["orders"].iloc[0] > 0
    else:
        assert own.frame["orders"].iloc[0] == 0


@pytest.mark.parametrize("change", ["ranked_factor_shares", "score_parameters", "cash_reserve"])
def test_post_result_parameter_changes_are_rejected(policy, change):
    if change == "ranked_factor_shares":
        policy[change] = [0.4, 0.3, 0.2, 0.1]
    elif change == "score_parameters":
        policy[change]["parameters"]["residual_fit_sessions"] = 126
    else:
        policy[change] = 0
    with pytest.raises(QuantError):
        validate_policy(policy)


def test_substituted_fund_or_missing_price_is_not_a_usable_input(market, policy):
    invalid = replace(market, volume=market.volume.assign(MTUM=0))
    with pytest.raises(QuantError):
        build_targets(invalid, policy)
