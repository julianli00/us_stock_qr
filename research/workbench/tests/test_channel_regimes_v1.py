from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from us_quant import channel_regimes_v1 as rules
from us_quant import equity_first_regimes_v1 as execution
from us_quant.bt_audit import independent_equity
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import execution_statistics
from us_quant.storage import read_json


@pytest.fixture
def study(market_factory):
    data = market_factory(
        start="2018-01-02", end="2021-12-31", symbols=tuple(sorted(execution.UNIVERSE))
    )
    return data, read_json(rules.POLICY)


def test_prior_only_barriers_strict_ties_state_memory_and_complete_warmup():
    levels = pd.Series([100.0] * 63 + [101.0, 100.0, 99.0, 100.0, 101.0, 102.0])
    state = rules.channel_state(levels)
    assert state.iloc[:63].isna().all()
    assert state.iloc[63:].tolist() == [1, 1, 0, 0, 0, 1]
    assert rules.channel_state(pd.Series([100.0] * 70)).iloc[63:].eq(0).all()


@pytest.mark.parametrize("damage", ["nan", "infinity", "zero", "duplicate", "unordered"])
def test_unknown_or_invalid_levels_cannot_become_a_cash_default(damage):
    level = pd.Series(np.arange(100, 200, dtype=float))
    if damage in {"nan", "infinity", "zero"}:
        level.iloc[70] = {"nan": np.nan, "infinity": np.inf, "zero": 0}[damage]
    elif damage == "duplicate":
        level.index = [0] * len(level)
    else:
        level = level.iloc[::-1]
    with pytest.raises(QuantError):
        rules.channel_state(level)


def test_full_targets_agree_with_separate_sequential_price_oracle(study):
    data, policy = study
    actual, oracle = rules.build_targets(data, policy), rules.independent_targets(data, policy)
    for identifier, target in actual.items():
        np.testing.assert_allclose(target, oracle[identifier], rtol=0, atol=1e-12, equal_nan=True)
        assert target.iloc[:252].isna().all().all()
        assert target.iloc[252:].notna().all().all()
        np.testing.assert_allclose(target.iloc[252:].sum(axis=1), 0.98, rtol=0, atol=1e-12)
        assert (target.iloc[252:] >= 0).all().all()
        assert target.iloc[252:]["SOXX"].gt(0).any()
        assert target.iloc[252:]["SOXX"].eq(0).any()


@pytest.mark.parametrize("position", [63, 199, 252, 300, 600, 900])
def test_future_prices_cannot_revise_any_state_or_target(study, position):
    data, policy = study
    earlier = replace(
        data,
        **{
            name: getattr(data, name).iloc[: position + 1]
            for name in ("open", "close", "raw_close", "volume", "risk_free")
        },
    )
    full, prefix = rules.build_targets(data, policy), rules.build_targets(earlier, policy)
    for identifier in full:
        pd.testing.assert_frame_equal(full[identifier].iloc[: position + 1], prefix[identifier])


def test_gold_state_is_maintained_even_while_equities_are_eligible(study):
    data, policy = study
    prices = (
        pd.DataFrame(
            np.repeat(
                np.exp(np.arange(len(data.close))[:, None] * 0.001), len(data.close.columns), axis=1
            ),
            index=data.close.index,
            columns=data.close.columns,
        )
        * 100
    )
    prices.loc[prices.index[310] :, list(execution.COMPOSITION)] *= 0.5
    data = replace(data, close=prices, open=prices, raw_close=prices)
    target = rules.build_targets(data, policy)[rules.CANDIDATES[1]["id"]]
    assert target.iloc[309]["SOXX"] == pytest.approx(0.49)
    assert target.iloc[310]["SOXX"] == 0
    assert target.iloc[310]["GLD"] == pytest.approx(0.98)


@pytest.mark.parametrize(
    "key,value",
    [
        ("entry_prior_observations", 42),
        ("exit_prior_observations", 10),
        ("cash_reserve", 0),
        ("common_warmup_observations", 64),
        ("minimum_economic_factor_families", 2),
        ("prior_completed_configurations", 180),
    ],
)
def test_fixed_rule_cannot_accept_a_new_horizon_or_rewrite_prior_trials(study, key, value):
    _, policy = study
    changed = deepcopy(policy)
    changed[key] = value
    with pytest.raises(QuantError):
        rules.validate(changed)


@pytest.mark.parametrize("candidate", rules.CANDIDATES)
@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_unchanged_funded_executor_and_pending_requests_match_independent_bt(
    study, candidate, cost, delay
):
    data, policy = study
    targets = rules.build_targets(data, policy)[candidate["id"]]
    account = execution.simulate_targets(
        data,
        targets,
        read_json(execution.POLICY),
        "2020-01-02",
        "2020-06-30",
        cost_bps=cost,
        delay=delay,
    )
    independent = independent_equity(
        data,
        account.issued,
        "2020-01-02",
        "2020-06-30",
        capital=10000,
        cost_bps=cost,
        commission=1,
        delay=delay,
    )
    np.testing.assert_allclose(
        account.result.frame["equity"], independent["equity"], rtol=0, atol=1e-7
    )
    stats = execution_statistics(account.result, data, 10000, cost, 1)
    assert stats["minimum_cash_usd"] >= 0
    assert stats["maximum_gross_exposure"] <= 1
    assert stats["model_order_count"] > 0
    if delay == 2:
        assert account.result.frame.iloc[0]["equity"] == 10000
        assert account.decisions[1]["reason"] == "pending_request_preserved"
