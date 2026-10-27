from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from us_quant import equity_first_regimes_v1 as execution
from us_quant import option_exit_latch_v1 as rules
from us_quant.bt_audit import independent_equity
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import execution_statistics
from us_quant.storage import read_json


def source_frame(index):
    source = pd.DataFrame(0.0, index=index, columns=sorted(execution.UNIVERSE))
    for symbol, share in execution.COMPOSITION.items():
        source[symbol] = 0.98 * share
    return source


def terms_frame(index):
    return pd.DataFrame({"VIX": 15.0, "VIX3M": 20.0}, index=index)


def target(source, terms):
    return rules.build_targets(source, terms, read_json(rules.POLICY))[rules.CANDIDATES[0]["id"]]


def test_initial_off_daily_exit_and_only_actual_month_end_restoration():
    index = sessions("2019-01-02", "2019-04-02")
    source, terms = source_frame(index), terms_frame(index)
    terms.loc["2019-02-05", "VIX"] = 20.0
    terms.loc["2019-02-28", "VIX"] = 20.0
    actual = target(source, terms)
    assert actual.loc["2019-01-30", "BIL"] == 0.98
    assert actual.loc["2019-01-31", "SOXX"] == 0.49
    assert actual.loc["2019-02-04", "SOXX"] == 0.49
    assert actual.loc["2019-02-05":"2019-03-28", "BIL"].eq(0.98).all()
    assert actual.loc["2019-03-29", "SOXX"] == 0.49
    assert actual.loc["2019-04-01", "SOXX"] == 0.49


def test_exit_applies_to_gold_and_restores_latest_saved_source_not_old_equity():
    index = sessions("2019-01-02", "2019-04-02")
    source, terms = source_frame(index), terms_frame(index)
    source.loc["2019-02-28":, :] = 0.0
    source.loc["2019-02-28":, "GLD"] = 0.98
    terms.loc["2019-02-15", "VIX"] = 25.0
    terms.loc["2019-03-05", "VIX"] = 25.0
    actual = target(source, terms)
    assert actual.loc["2019-02-15", "BIL"] == 0.98
    assert actual.loc["2019-02-28", "GLD"] == 0.98
    assert actual.loc["2019-03-05":"2019-03-28", "BIL"].eq(0.98).all()
    assert actual.loc["2019-03-29", "GLD"] == 0.98
    assert actual.loc["2019-03-29", "SOXX"] == 0.0


def test_state_observes_prehistory_without_erasing_source_warmup():
    index = sessions("2019-01-02", "2019-03-15")
    source, terms = source_frame(index), terms_frame(index)
    source.loc[:"2019-02-07"] = np.nan
    terms.loc["2019-02-05", "VIX"] = 25.0
    actual = target(source, terms)
    assert actual.loc[:"2019-02-07"].isna().all().all()
    assert actual.loc["2019-02-08":"2019-02-27", "BIL"].eq(0.98).all()
    assert actual.loc["2019-02-28", "SOXX"] == 0.49


@pytest.mark.parametrize("end", ["2019-01-15", "2019-01-31", "2019-02-14", "2019-02-28"])
def test_prefix_end_cannot_be_mistaken_for_month_end_or_revise_state(end):
    index = sessions("2019-01-02", "2019-03-29")
    source, terms = source_frame(index), terms_frame(index)
    terms.loc["2019-02-05", "VIX"] = 25.0
    full = target(source, terms)
    pd.testing.assert_frame_equal(full.loc[:end], target(source.loc[:end], terms.loc[:end]))


@pytest.mark.parametrize(
    "damage", ["missing", "nan", "zero", "infinity", "renamed", "source_gap", "nonmonth_change"]
)
def test_invalid_or_unknown_inputs_never_default_to_safe_cash(damage):
    index = sessions("2019-01-02", "2019-03-29")
    source, terms = source_frame(index), terms_frame(index)
    if damage == "missing":
        terms = terms.iloc[1:]
    elif damage in {"nan", "zero", "infinity"}:
        terms.iloc[10, 0] = {"nan": np.nan, "zero": 0, "infinity": np.inf}[damage]
    elif damage == "renamed":
        terms = terms.rename(columns={"VIX": "VIX9D"})
    elif damage == "source_gap":
        source.iloc[10] = np.nan
    else:
        source.loc["2019-02-05":, :] = 0.0
        source.loc["2019-02-05":, "BIL"] = 0.98
    with pytest.raises(QuantError):
        target(source, terms)


@pytest.mark.parametrize(
    "key,value",
    [
        ("option_ratio_risk_threshold", 1.1),
        ("initial_risk_off", False),
        ("cash_reserve", 0.0),
        ("prior_completed_configurations", 182),
        ("minimum_economic_factor_families", 2),
        ("candidate_count_in_this_study", 2),
    ],
)
def test_fixed_rule_cannot_accept_refitted_thresholds_or_rewrite_trials(key, value):
    policy = deepcopy(read_json(rules.POLICY))
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_full_oracle_funding_and_pending_execution_agree(market_factory, cost, delay):
    data = market_factory(
        start="2018-01-02", end="2021-12-31", symbols=tuple(sorted(execution.UNIVERSE))
    )
    source, terms = source_frame(data.close.index), terms_frame(data.close.index)
    terms["VIX"] = 18 + 7 * np.sin(np.arange(len(terms)) / 13)
    policy = read_json(rules.POLICY)
    actual = rules.build_targets(source, terms, policy)[rules.CANDIDATES[0]["id"]]
    oracle = rules.independent_targets(source, terms, policy)[rules.CANDIDATES[0]["id"]]
    pd.testing.assert_frame_equal(actual, oracle)
    np.testing.assert_allclose(actual.sum(axis=1), 0.98, rtol=0, atol=1e-12)
    account = execution.simulate_targets(
        data,
        actual,
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
