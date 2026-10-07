from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.calendar import is_month_end
from us_quant.config import QuantError
from us_quant.dual_horizon import load_protocol
from us_quant.factor_research import (
    build_signals,
    factor_scores,
    fingerprint,
    goal_gates,
    information_discreteness,
    inverse_variance_scale,
    snapshot_descriptor,
    validate_policy,
    verify_registration,
)
from us_quant.storage import digest_json, file_digest, read_json, utc_now, write_json


@pytest.fixture
def policy():
    return read_json(Path(__file__).parents[1] / "config/factor-research.json")


@pytest.fixture
def comparison():
    return load_protocol(Path(__file__).parents[1] / "config/dual-horizon.json")


@pytest.fixture
def factor_market(market_factory, policy, comparison):
    return market_factory(
        start="2010-01-04",
        end="2014-03-14",
        symbols=tuple(comparison.symbols) + tuple(policy["supplement_symbols"]),
    )


def test_round_preserves_all_configurations_and_both_goal_contracts(policy):
    validate_policy(policy)
    assert len(policy["candidates"]) == 8
    assert policy["prior_disclosed_configurations"] == 52
    assert policy["retained_secondary_goals"]["max_drawdown_at_most"] == 0.15


@pytest.mark.parametrize(
    "change",
    [
        {"cash_reserve": 0},
        {"capital_usd": 1000000},
        {"base_delay_sessions": 0},
        {"stress_delay_sessions": 1},
        {"cost_bps_per_side": 0},
        {"commission_per_order": float("nan")},
        {"prior_disclosed_configurations": 0},
        {"primary_goal": {"net_excess_sharpe_strictly_above": 0.8}},
    ],
)
def test_no_silent_goal_cost_or_authority_relaxation(policy, change):
    policy.update(change)
    with pytest.raises(QuantError):
        validate_policy(policy)


def test_information_discreteness_distinguishes_continuous_and_jump_returns():
    returns = pd.DataFrame({"continuous": [0.01] * 10, "jump": [0.0] * 9 + [1.01**10 - 1]})
    values = information_discreteness(returns)
    assert values["continuous"] == -1
    assert values["jump"] == pytest.approx(-0.1)


def test_variance_management_is_not_inverse_volatility():
    assert inverse_variance_scale(0.4**2, 0.2) == pytest.approx(0.25)
    assert inverse_variance_scale(0.1**2, 0.2) == 1
    assert inverse_variance_scale(0.0, 0.2) == 1
    with pytest.raises(QuantError):
        inverse_variance_scale(float("nan"), 0.2)


def test_residual_factor_removes_exact_market_loading(policy, factor_market):
    close = factor_market.close.copy()
    returns = close.pct_change(fill_method=None).fillna(0)
    rf = factor_market.risk_free
    dependent = 0.0001 + 1.3 * (returns["SPY"] - rf) + rf
    close["XLK"] = 100 * (1 + dependent).cumprod()
    scores = factor_scores(close, rf, policy)
    assert abs(scores.loc["XLK", "residual"]) < 1e-8


def test_residual_warmup_and_alignment_are_required(policy, factor_market):
    with pytest.raises(QuantError, match="warmup"):
        factor_scores(factor_market.close.iloc[:100], factor_market.risk_free.iloc[:100], policy)
    with pytest.raises(QuantError, match="aligned"):
        factor_scores(factor_market.close, factor_market.risk_free.iloc[1:], policy)


def test_signals_are_month_end_cash_funded_and_do_not_use_future_data(policy, factor_market):
    original = build_signals(factor_market, policy)
    boundary = pd.Timestamp("2013-09-30")
    changed_close = factor_market.close.copy()
    later = changed_close.index > boundary
    changed_close.loc[later] *= np.exp(np.linspace(0, 0.3, later.sum()))[:, None]
    changed = replace(
        factor_market,
        close=changed_close,
        raw_close=changed_close.copy(),
        open=changed_close * 0.999,
    )
    changed_signals = build_signals(changed, policy)
    for identifier, frame in original.items():
        defined = frame.dropna(how="all")
        assert all(is_month_end(day) for day in defined.index)
        assert defined.index[-1] == pd.Timestamp("2014-02-28")
        assert np.allclose(defined.sum(axis=1), 0.98)
        assert (defined >= 0).all().all()
        pd.testing.assert_frame_equal(
            frame.loc[:boundary], changed_signals[identifier].loc[:boundary]
        )
    components = policy["candidates"][3]["components"]
    expected = sum(original[name] for name in components) / len(components)
    pd.testing.assert_frame_equal(original["sector_fixed_factor_ensemble"], expected)


def sample_metrics(cagr=0.15, sharpe=1.1):
    return {
        "start": "2016-10-06",
        "end": "2026-10-05",
        "sessions": 2512,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": 0.2,
    }


def test_latest_goal_is_separate_from_old_return_and_drawdown_targets(policy):
    result = goal_gates(sample_metrics(), sample_metrics(cagr=0.12), policy)
    assert all(result.values())
    assert sample_metrics()["cagr"] < policy["retained_secondary_goals"]["net_cagr_strictly_above"]


@pytest.mark.parametrize(
    "cagr,sharpe,expected",
    [
        (0.15, 1.0, False),
        (0.12, 1.2, False),
        (0.15, None, False),
        (0.15, 1.00001, True),
    ],
)
def test_goal_thresholds_are_strict(policy, cagr, sharpe, expected):
    result = goal_gates(sample_metrics(cagr, sharpe), sample_metrics(cagr=0.12), policy)
    assert all(result.values()) is expected


def test_mismatched_benchmark_window_is_not_an_outperformance_result(policy):
    benchmark = sample_metrics()
    benchmark["start"] = "2017-01-03"
    with pytest.raises(QuantError, match="same benchmark"):
        goal_gates(sample_metrics(), benchmark, policy)


def test_registration_binds_code_policy_and_comparison(policy, comparison):
    record = {
        "registered_at": utc_now(),
        "policy_sha256": digest_json(policy),
        "comparison_sha256": digest_json(asdict(comparison)),
        "implementation_sha256": fingerprint(),
        "candidate_ids": [item["id"] for item in policy["candidates"]],
        "windows": comparison.windows(),
        "total_disclosed_after_round": 60,
        "order_authority": False,
        "history_previously_exposed": True,
    }
    verify_registration(policy, comparison, record)
    changed = deepcopy(policy)
    changed["parameters"]["variance_sessions"] = 63
    with pytest.raises(QuantError, match="changed"):
        verify_registration(changed, comparison, record)
    record["implementation_sha256"] = "0" * 64
    with pytest.raises(QuantError, match="changed"):
        verify_registration(policy, comparison, record)


def test_snapshot_requires_original_hashes_and_retrieval_time(tmp_path, comparison):
    (tmp_path / "raw").mkdir()
    (tmp_path / "SPY.csv").write_text("observed fixture")
    (tmp_path / "raw/SPY.json").write_text("{}")
    manifest = {
        "data_start": comparison.data_start,
        "data_end": comparison.as_of,
        "retrieved_at": "2026-10-06T05:16:59+00:00",
        "files": {name: file_digest(tmp_path / name) for name in ("SPY.csv", "raw/SPY.json")},
    }
    write_json(tmp_path / "manifest.json", manifest)
    descriptor = snapshot_descriptor(tmp_path, {"SPY"}, comparison)
    assert descriptor["retrieved_at"] == manifest["retrieved_at"]
    (tmp_path / "SPY.csv").write_text("revised fixture")
    with pytest.raises(QuantError, match="revised"):
        snapshot_descriptor(tmp_path, {"SPY"}, comparison)
