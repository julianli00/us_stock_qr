from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from us_quant import family_core_expansion_v1 as rules
from us_quant.bt_audit import independent_equity
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.equity_first_regimes_v1 import COMPOSITION, simulate_targets
from us_quant.equity_first_regimes_v1 import POLICY as EXECUTION_POLICY
from us_quant.storage import read_json


@pytest.fixture
def policy():
    return read_json(rules.POLICY)


def source_frame():
    dates = sessions("2020-01-02", "2020-05-29")
    frame = pd.DataFrame(0.0, index=dates, columns=rules.EXISTING)
    for symbol, share in COMPOSITION.items():
        frame[symbol] = 0.98 * share
    frame.iloc[:3] = np.nan
    frame.iloc[20:40] = 0.0
    frame.iloc[20:40, frame.columns.get_loc("GLD")] = 0.98
    frame.iloc[40:60] = 0.0
    frame.iloc[40:60, frame.columns.get_loc("BIL")] = 0.98
    return frame


def test_only_factor_half_changes_without_new_clock_or_risk_budget(policy):
    source = source_frame()
    before = source.copy(deep=True)
    target = rules.build_targets(source, policy)[rules.CANDIDATES[0]["id"]]
    assert tuple(target.columns) == rules.UNIVERSE
    assert target.index.equals(source.index)
    assert target.isna().all(axis=1).equals(source.isna().all(axis=1))
    unchanged = [symbol for symbol in rules.EXISTING if symbol not in rules.FACTORS]
    pd.testing.assert_frame_equal(target[unchanged], source[unchanged], check_exact=True)
    np.testing.assert_allclose(target.iloc[3][list(rules.FAMILIES)], 0.49 / 6, rtol=0, atol=1e-15)
    assert target.iloc[3]["SOXX"] == 0.49
    assert (target.iloc[20:60][list(rules.FAMILIES)] == 0.0).all().all()
    np.testing.assert_allclose(target.dropna().sum(axis=1), 0.98, rtol=0, atol=1e-12)
    pd.testing.assert_frame_equal(source, before, check_exact=True)


def test_independent_soxx_budget_oracle_and_prefixes_match(policy):
    source = source_frame()
    target = rules.build_targets(source, policy)[rules.CANDIDATES[0]["id"]]
    oracle = rules.independent_targets(source, policy)[rules.CANDIDATES[0]["id"]]
    np.testing.assert_allclose(target, oracle, rtol=0, atol=1e-12, equal_nan=True)
    for stop in (1, 3, 21, 40, 61, len(source) - 1):
        prefix = rules.build_targets(source.iloc[:stop], policy)[rules.CANDIDATES[0]["id"]]
        pd.testing.assert_frame_equal(prefix, target.iloc[:stop], check_exact=True)
    reordered = source.loc[:, list(reversed(source.columns))]
    pd.testing.assert_frame_equal(
        rules.build_targets(reordered, policy)[rules.CANDIDATES[0]["id"]], target, check_exact=True
    )


@pytest.mark.parametrize(
    "damage",
    ["partial", "gap", "negative", "composition", "calendar", "unknown", "duplicate", "infinite"],
)
def test_invalid_sources_do_not_become_safe_targets(policy, damage):
    source = source_frame()
    if damage == "partial":
        source.iloc[3, 0] = np.nan
    elif damage == "gap":
        source.iloc[10] = np.nan
    elif damage == "negative":
        source.iloc[3, source.columns.get_loc("SPY")] = -1e-13
    elif damage == "composition":
        source.iloc[3, source.columns.get_loc("SOXX")] = 0.4
    elif damage == "calendar":
        source = source.drop(source.index[10])
    elif damage == "unknown":
        source["OTHER"] = 0.0
    elif damage == "duplicate":
        source.columns = ["SPY", *list(source.columns)[1:-1], "SPY"]
    else:
        source.iloc[3, source.columns.get_loc("SOXX")] = np.inf
    with pytest.raises(QuantError):
        rules.build_targets(source, policy)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_candidate_id", "equity_first_21slot_commitment_v1"),
        ("source_targets_sha256", "0" * 64),
        ("catalog_publication_commit", "0" * 40),
        ("equal_core_family_count", 5),
        ("source_stock_share_preserved_SOXX", 0.4),
        ("cash_reserve", 0.0),
        ("weight_band", 0.02),
        ("candidate_count_in_this_study", True),
        ("used_economic_factor_families", 4),
        ("new_economic_factor_definitions_in_this_study", 2),
        ("study_size_is_a_total_research_quota", True),
    ],
)
def test_source_and_fixed_rule_cannot_be_tuned_silently(policy, field, value):
    changed = deepcopy(policy)
    changed[field] = value
    with pytest.raises(QuantError):
        rules.build_targets(source_frame(), changed)


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_unchanged_execution_and_independent_accounting_support_synthetic_fourteen_assets(
    policy, cost, delay
):
    source = source_frame()
    target = rules.build_targets(source, policy)[rules.CANDIDATES[0]["id"]]
    prices = pd.DataFrame(100.0, index=source.index, columns=rules.UNIVERSE)
    data = MarketData(
        prices.copy(),
        prices.copy(),
        prices.copy(),
        prices.copy() * 100,
        pd.Series(0.0, index=source.index, name="risk_free"),
    )
    start, end = (str(source.index[position].date()) for position in (70, 76))
    result = simulate_targets(
        data, target, read_json(EXECUTION_POLICY), start, end, cost_bps=cost, delay=delay
    )
    independent = independent_equity(
        data,
        result.issued,
        start,
        end,
        capital=10000,
        cost_bps=cost,
        commission=1,
        delay=delay,
    )
    assert result.result.weights.columns.tolist() == list(rules.UNIVERSE)
    assert result.result.frame["orders"].sum() == 7
    assert result.result.frame["cash"].min() >= 0
    assert result.result.frame["gross_exposure"].max() <= 1
    assert result.result.frame.iloc[delay - 1]["orders"] == 7
    if delay == 2:
        assert result.result.frame.iloc[0]["equity"] == 10000
    assert independent.index.equals(result.result.frame.index)
    assert independent.columns.tolist() == ["equity", "return"]
    np.testing.assert_allclose(
        result.result.frame["equity"], independent["equity"], rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        result.result.frame["return"], independent["return"], rtol=0, atol=1e-12
    )
