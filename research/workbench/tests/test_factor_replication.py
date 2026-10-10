from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.config import QuantError
from us_quant.factor_gold_risk import monthly_targets as original_targets
from us_quant.factor_gold_risk import run_candidate
from us_quant.factor_replication import FACTORS, monthly_targets, run, validate_policy
from us_quant.storage import read_json

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/factor-implementation-replication.json")


@pytest.fixture
def markets(market_factory):
    before = market_factory(
        "2016-01-04",
        "2017-06-30",
        ("SPY", "IEF", "GLD", "BIL", "MTUM", "VLUE", "QUAL", "USMV"),
    )
    rename = {"QUAL": "SPHQ"}
    after = replace(
        before,
        open=before.open.rename(columns=rename),
        close=before.close.rename(columns=rename),
        raw_close=before.raw_close.rename(columns=rename),
        volume=before.volume.rename(columns=rename),
    )
    return before, after


def test_fund_identity_change_keeps_allocation_rule_exactly_equivalent_on_equal_inputs(
    markets, policy
):
    before, after = markets
    original = read_json(ROOT / "config/factor-gold-risk.json")
    actual = monthly_targets(after, policy).rename(columns={"SPHQ": "QUAL"})
    pd.testing.assert_frame_equal(original_targets(before, original), actual)
    weights = monthly_targets(after, policy).dropna(how="all")
    assert np.allclose(weights.sum(axis=1), 0.98)
    equities = weights.loc[:, list(FACTORS)]
    assert np.allclose(equities.div(equities.sum(axis=1), axis=0), 0.25)


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("cost,delay", [(5, 1), (20, 2)])
def test_original_and_replicated_accounting_match_with_identical_return_inputs(
    markets, policy, index, cost, delay
):
    before, after = markets
    original = read_json(ROOT / "config/factor-gold-risk.json")
    reference, targets, _ = run_candidate(
        before,
        original_targets(before, original),
        original["candidates"][index],
        original,
        "2016-10-06",
        "2017-06-30",
        cost,
        delay,
    )
    actual, decisions = run(
        after, policy["candidates"][index], policy, "2016-10-06", "2017-06-30", cost, delay
    )
    pd.testing.assert_frame_equal(reference.frame, actual.frame)
    pd.testing.assert_frame_equal(targets, decisions.rename(columns={"SPHQ": "QUAL"}))
    assert (actual.frame["cash"] >= 0).all()
    assert (actual.weights.sum(axis=1) <= 1 + 1e-12).all()


def test_new_fund_prices_cannot_change_past_decisions(markets, policy):
    _, market = markets
    cut = pd.Timestamp("2017-01-31")
    prior, issued = run(market, policy["candidates"][1], policy, "2016-10-06", "2017-06-30", 20, 2)
    closing, opening = market.close.copy(), market.open.copy()
    after = closing.index > cut
    scale = np.linspace(1, 1.5, after.sum())
    closing.loc[after, "SPHQ"] *= scale
    opening.loc[after, "SPHQ"] *= scale
    updated = replace(market, close=closing, open=opening, raw_close=closing.copy())
    actual, decisions = run(
        updated, policy["candidates"][1], policy, "2016-10-06", "2017-06-30", 20, 2
    )
    pd.testing.assert_frame_equal(prior.frame.loc[:cut], actual.frame.loc[:cut])
    pd.testing.assert_frame_equal(issued.loc[:cut], decisions.loc[:cut])


def test_no_fund_or_parameter_tuning_hidden_in_replication(policy):
    validate_policy(policy)
    policy["factor_symbols"][1] = "SPVU"
    with pytest.raises(QuantError):
        validate_policy(policy)
