from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end, sessions
from us_quant.commitment_duration_v1 import (
    CANDIDATES,
    POLICY,
    SLOTS,
    build_targets,
    independent_targets,
)
from us_quant.config import QuantError
from us_quant.equity_first_regimes_v1 import COMPOSITION, UNIVERSE, simulate_targets
from us_quant.equity_first_regimes_v1 import POLICY as EXECUTION_POLICY
from us_quant.storage import read_json


@pytest.fixture
def source():
    index = sessions("2019-10-01", "2020-06-30")
    values = pd.DataFrame(0.0, index=index, columns=sorted(UNIVERSE))
    for position in range(len(values)):
        if position % 3 == 0:
            for symbol, share in COMPOSITION.items():
                values.loc[index[position], symbol] = 0.98 * share
        else:
            values.loc[index[position], "GLD" if position % 3 == 1 else "BIL"] = 0.98
    values.iloc[:7] = np.nan
    return values


def test_fixed_rules_match_deque_oracle_and_conserve_funding(source):
    policy = read_json(POLICY)
    actual, oracle = build_targets(source, policy), independent_targets(source, policy)
    for identifier, target in actual.items():
        np.testing.assert_allclose(target, oracle[identifier], rtol=0, atol=1e-12, equal_nan=True)
        assert target.iloc[: 7 + SLOTS - 1].isna().all().all()
        defined = target.dropna(how="all")
        assert not defined.empty
        assert (defined >= 0).all().all()
        np.testing.assert_allclose(defined.sum(axis=1), 0.98, rtol=0, atol=1e-12)
        assert (defined[["SPY", "QQQ", "IEF", "TLT", "XLK"]] == 0).all().all()


@pytest.mark.parametrize("position", [5, 7, 26, 27, 38, 64, 107, 120])
def test_prefix_truncation_never_turns_last_observation_into_month_end(source, position):
    policy = read_json(POLICY)
    full, prefix = build_targets(source, policy), build_targets(source.iloc[: position + 1], policy)
    for identifier in full:
        pd.testing.assert_frame_equal(full[identifier].iloc[: position + 1], prefix[identifier])


def test_monthly_commitment_uses_actual_holiday_calendar_not_last_available_row(source):
    policy = read_json(POLICY)
    assert is_month_end(pd.Timestamp("2019-11-29"))
    assert not is_month_end(pd.Timestamp("2019-12-20"))
    target = build_targets(source.loc[:"2019-12-20"], policy)[CANDIDATES[0]["id"]]
    pd.testing.assert_series_equal(
        target.loc["2019-12-20"], source.loc["2019-11-29"], check_names=False
    )
    pd.testing.assert_series_equal(
        target.loc["2019-11-29"], source.loc["2019-11-29"], check_names=False
    )


def test_twenty_one_slots_mean_targets_not_returns_or_replicated_latest_signal(source):
    target = build_targets(source, read_json(POLICY))[CANDIDATES[1]["id"]]
    assert target.iloc[26].isna().all()
    np.testing.assert_allclose(target.iloc[27], source.iloc[7:28].mean(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(target.iloc[28], source.iloc[8:29].mean(), rtol=0, atol=1e-12)
    assert target.iloc[27]["GLD"] == pytest.approx(0.98 / 3)
    assert target.iloc[27]["SOXX"] == pytest.approx(0.49 / 3)


def test_source_and_policy_are_not_mutated(source):
    policy = read_json(POLICY)
    original, original_policy = source.copy(), deepcopy(policy)
    build_targets(source, policy)
    independent_targets(source, policy)
    pd.testing.assert_frame_equal(source, original)
    assert policy == original_policy


@pytest.mark.parametrize("implementation", [build_targets, independent_targets])
@pytest.mark.parametrize(
    "damage", ["gap", "partial", "calendar", "duplicate", "cap", "negative", "inf"]
)
def test_invalid_source_does_not_silently_create_safe_targets(source, implementation, damage):
    if damage == "gap":
        source.iloc[60] = np.nan
    elif damage == "partial":
        source.iloc[60, 0] = np.nan
    elif damage == "calendar":
        source = source.drop(source.index[60])
    elif damage == "duplicate":
        source = pd.concat([source.iloc[:61], source.iloc[60:]])
    elif damage == "cap":
        source.iloc[60] *= 0.5
    elif damage == "negative":
        source.iloc[60, 0] = -0.01
    else:
        source.iloc[60, 0] = np.inf
    with pytest.raises(QuantError):
        implementation(source, read_json(POLICY))


@pytest.mark.parametrize(
    "field,value",
    [
        ("common_source_intent_warmup", 20),
        ("source_candidate_id", "equity_first_vix_variance_band025_v1"),
        ("study_size_is_a_total_research_quota", True),
        ("prior_completed_configurations", 178),
    ],
)
def test_fixed_rules_cannot_be_tuned_after_outcomes(source, field, value):
    policy = read_json(POLICY)
    policy[field] = value
    with pytest.raises(QuantError, match="fixed commitment"):
        build_targets(source, policy)


@pytest.mark.parametrize("candidate", list(CANDIDATES))
@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_aggregate_intents_execute_in_one_real_funded_account(
    source, market_factory, candidate, cost, delay
):
    data = market_factory(
        start=str(source.index[0].date()),
        end=str(source.index[-1].date()),
        symbols=tuple(source.columns),
    )
    targets = build_targets(source, read_json(POLICY))[candidate["id"]]
    run = simulate_targets(
        data,
        targets,
        read_json(EXECUTION_POLICY),
        "2020-01-02",
        "2020-03-31",
        cost_bps=cost,
        delay=delay,
    )
    independent = independent_equity(
        data,
        run.issued,
        "2020-01-02",
        "2020-03-31",
        capital=10000,
        cost_bps=cost,
        commission=1,
        delay=delay,
    )
    np.testing.assert_allclose(run.result.frame["equity"], independent["equity"], rtol=0, atol=1e-7)
    assert (run.result.frame["cash"] >= -1e-8).all()
    if delay == 2:
        assert run.result.frame.iloc[0]["equity"] == 10000
        assert run.decisions[1]["reason"] == "pending_request_preserved"
