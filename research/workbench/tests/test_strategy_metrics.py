from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from us_quant.calendar import is_month_end, sessions
from us_quant.config import Candidate, QuantError
from us_quant.metrics import acceptance, block_bootstrap, performance
from us_quant.strategy import monthly_signals, target_weights


def test_no_lookahead_from_future_price_changes(market_factory, small_config):
    data = market_factory(symbols=small_config.symbols)
    cutoff = pd.Timestamp("2021-06-30")
    original = monthly_signals(data.close, small_config.candidates[1], small_config)
    altered = data.close.copy()
    altered.loc[altered.index > cutoff, "QQQ"] *= 50
    changed = monthly_signals(altered, small_config.candidates[1], small_config)
    pd.testing.assert_frame_equal(original.loc[:cutoff], changed.loc[:cutoff])


def test_prefix_results_and_adjustment_scale_are_invariant(market_factory, small_config):
    close = market_factory(symbols=small_config.symbols).close
    candidate = small_config.candidates[1]
    original = monthly_signals(close, candidate, small_config)
    prefix = monthly_signals(close.loc[:"2021-06-15"], candidate, small_config)
    pd.testing.assert_frame_equal(original.loc[prefix.index], prefix)
    scaled = close * pd.Series(np.arange(1, len(close.columns) + 1), index=close.columns)
    pd.testing.assert_series_equal(
        target_weights(close, candidate, small_config),
        target_weights(scaled, candidate, small_config),
        atol=1e-12,
        rtol=1e-12,
    )
    assert prefix.loc["2021-06-15"].isna().all()


def test_all_candidates_obey_risk_caps(market_factory, small_config):
    close = market_factory(symbols=small_config.symbols).close
    for candidate in small_config.candidates:
        signals = monthly_signals(close, candidate, small_config)
        observed = signals.dropna()
        assert all(is_month_end(day) for day in observed.index)
        assert (observed >= 0).all().all()
        assert (observed <= candidate.max_weight + 1e-10).all().all()
        assert (observed.sum(axis=1) <= 1 - small_config.cash_reserve + 1e-10).all()


def test_warmup_and_ineligible_assets_go_to_cash(market_factory, config):
    close = market_factory().close
    candidate = Candidate("test", ("SPY", "QQQ"), "momentum", 2, 0.5, 0.1)
    assert target_weights(close.iloc[:10], candidate, config).sum() == 0
    assert target_weights(close * 0 + 100, candidate, config).sum() == 0
    assert target_weights(close, None, config).sum() == 0


def test_first_loss_is_included_in_drawdown():
    index = sessions("2020-02-03", "2020-02-04")
    result = performance(pd.Series([-0.1, 0.0], index=index), pd.Series(0.0, index=index))
    assert result["max_drawdown"] == pytest.approx(0.1)
    assert result["total_return"] == pytest.approx(-0.1)


def test_sharpe_subtracts_nonzero_risk_free_and_cash_is_undefined():
    index = sessions("2020-01-02", "2020-12-31")
    returns = pd.Series(np.tile([0.002, -0.001], len(index) // 2 + 1)[: len(index)], index=index)
    zero = performance(returns, pd.Series(0.0, index=index))
    positive = performance(returns, pd.Series(0.0002, index=index))
    assert positive["sharpe"] < zero["sharpe"]
    assert performance(returns * 0, returns * 0)["sharpe"] is None


def test_acceptance_is_strict_and_requires_aligned_benchmark(config):
    item = {
        "start": "2022-01-03",
        "end": "2026-09-28",
        "sessions": 1000,
        "cagr": 0.20,
        "sharpe": 1.0,
        "max_drawdown": 0.15,
    }
    gates = acceptance(item, item, config)
    assert gates == {
        "cagr_above_target": False,
        "sharpe_above_target": False,
        "drawdown_within_limit": True,
        "beats_primary_benchmark": False,
        "enough_sessions": True,
    }
    with pytest.raises(QuantError, match="mismatched"):
        acceptance(item, {**item, "end": "2025-01-01"}, config)


def test_bootstrap_is_paired_and_reproducible(market_factory):
    data = market_factory()
    returns = data.close["SPY"].pct_change().iloc[1:]
    kwargs = {"samples": 100, "block": 21, "seed": 17}
    first = block_bootstrap(returns, returns, data.risk_free.loc[returns.index], **kwargs)
    second = block_bootstrap(returns, returns, data.risk_free.loc[returns.index], **kwargs)
    assert first == second
    assert first["excess_cagr_vs_spy_95pct_interval"] == [0.0, 0.0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("cost_bps_per_side", float("nan")),
        ("commission_per_order", float("nan")),
        ("execution_delay_sessions", 0),
        ("trend_lookback", 2.5),
    ],
)
def test_config_rejects_invalid_research_settings(config, field, value):
    with pytest.raises(QuantError):
        replace(config, **{field: value})
