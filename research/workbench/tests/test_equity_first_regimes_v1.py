from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from us_quant.bt_audit import independent_equity
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import execution_statistics
from us_quant.equity_first_regimes_v1 import (
    CANDIDATES,
    POLICY,
    UNIVERSE,
    build_targets,
    independent_targets,
    simulate_targets,
    validate,
)
from us_quant.storage import read_json


@pytest.fixture
def study(market_factory):
    data = market_factory(start="2018-01-02", end="2021-12-31", symbols=tuple(sorted(UNIVERSE)))
    vix = pd.Series(
        20 + 6 * np.sin(np.arange(len(data.close)) / 37),
        index=data.close.index,
        name="VIX",
    )
    return data, vix, read_json(POLICY)


def test_complete_targets_match_sequential_oracle_and_keep_funding(study):
    data, vix, policy = study
    actual = build_targets(data, vix, policy)
    oracle = independent_targets(data, vix, policy)
    assert set(actual) == {candidate["id"] for candidate in CANDIDATES}
    for identifier, targets in actual.items():
        np.testing.assert_allclose(targets, oracle[identifier], rtol=0, atol=1e-12, equal_nan=True)
        assert targets.iloc[:252].isna().all().all()
        assert targets.iloc[252:].notna().all().all()
        np.testing.assert_allclose(targets.iloc[252:].sum(axis=1), 0.98, rtol=0, atol=1e-12)
        assert (targets.iloc[252:] >= 0).all().all()


@pytest.mark.parametrize("position", [252, 300, 600])
def test_future_prices_or_vix_cannot_revise_an_earlier_target(study, position):
    data, vix, policy = study
    day = data.close.index[position]
    truncated = replace(
        data,
        **{
            name: getattr(data, name).loc[:day]
            for name in ("open", "close", "raw_close", "volume", "risk_free")
        },
    )
    full, prefix = build_targets(data, vix, policy), build_targets(truncated, vix.loc[:day], policy)
    for identifier in full:
        pd.testing.assert_frame_equal(full[identifier].loc[:day], prefix[identifier])


def test_variance_reference_excludes_current_observation_and_never_leverages(study):
    data, _, policy = study
    rising = (
        pd.DataFrame(
            np.repeat(
                np.exp(np.arange(len(data.close))[:, None] * 0.001), len(data.close.columns), axis=1
            ),
            index=data.close.index,
            columns=data.close.columns,
        )
        * 100
    )
    data = replace(data, close=rising, open=rising, raw_close=rising)
    vix = pd.Series(20.0, index=rising.index)
    vix.iloc[600] = 40
    targets = build_targets(data, vix, policy)
    plain, capped = (targets[candidate["id"]] for candidate in CANDIDATES)
    assert plain.iloc[600]["SOXX"] == pytest.approx(0.49)
    assert capped.iloc[600]["SOXX"] == pytest.approx(0.49 / 4)
    assert capped.iloc[599]["SOXX"] == pytest.approx(0.49)
    assert (capped.dropna(how="all").sum(axis=1) <= 1).all()


def test_missing_vix_is_not_a_silent_cash_or_other_series_substitute(study):
    data, vix, policy = study
    vix.iloc[500] = np.nan
    with pytest.raises(QuantError, match="actual aligned VIX"):
        build_targets(data, vix, policy)


def test_nonfinite_variance_cannot_become_a_safe_oracle_default(study):
    data, vix, policy = study
    vix.iloc[500] = 1e200
    for implementation in (build_targets, independent_targets):
        with pytest.raises(QuantError, match="nonfinite variance"):
            implementation(data, vix, policy)


@pytest.mark.parametrize(
    "field,value",
    [
        ("weight_band", 0.02),
        ("trend_observations", 150),
        ("cash_reserve", 0),
        ("minimum_economic_factor_families", 2),
    ],
)
def test_fixed_pair_does_not_accept_post_outcome_parameter_changes(study, field, value):
    _, _, policy = study
    changed = deepcopy(policy)
    changed[field] = value
    with pytest.raises(QuantError):
        validate(changed)


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
@pytest.mark.parametrize("candidate", list(CANDIDATES))
def test_new_daily_band_uses_real_queued_fills_and_independent_bt(study, cost, delay, candidate):
    data, vix, policy = study
    targets = build_targets(data, vix, policy)[candidate["id"]]
    result = simulate_targets(
        data, targets, policy, "2020-01-02", "2020-06-30", cost_bps=cost, delay=delay
    )
    independent = independent_equity(
        data,
        result.issued,
        "2020-01-02",
        "2020-06-30",
        capital=10000,
        cost_bps=cost,
        commission=1,
        delay=delay,
    )
    np.testing.assert_allclose(
        result.result.frame["equity"], independent["equity"], rtol=0, atol=1e-7
    )
    disclosure = execution_statistics(result.result, data, 10000, cost, 1)
    assert disclosure["minimum_cash_usd"] >= 0
    assert disclosure["maximum_gross_exposure"] <= 1
    assert disclosure["model_order_count"] > 0
    if delay == 2:
        assert result.result.frame.iloc[0]["equity"] == 10000
        assert result.result.frame.iloc[0]["orders"] == 0
        assert result.decisions[1]["reason"] == "pending_request_preserved"
        assert result.decisions[1]["execution_session"] is None
