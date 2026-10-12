from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from us_quant import producer_sector_sources as sources
from us_quant.config import QuantError
from us_quant.storage import read_json

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/producer-sector-sources.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10","2016-06-30",sources.ORIGINAL)


def extra(data):
    panels = {}
    for name in ("open","close","raw_close","volume"):
        frame = getattr(data,name).copy()
        for symbol in sources.ADDITIONAL:
            frame[symbol] = frame["SPY"]
        panels[name] = frame
    return replace(data,**panels,risk_free=data.risk_free.copy())


def test_extra_funds_never_replace_any_original_portfolio_price_or_rate(market):
    sector = extra(market)
    before = {name:getattr(market,name).copy() for name in ("open","close","raw_close","volume","risk_free")}
    result = sources.extend_market(market,sector)
    assert len(result.close.columns) == 14
    for name in ("open","close","raw_close","volume"):
        pd.testing.assert_frame_equal(getattr(result,name).loc[:,list(sources.ORIGINAL)],before[name])
        pd.testing.assert_frame_equal(getattr(market,name),before[name])
    pd.testing.assert_series_equal(result.risk_free,before["risk_free"])


def test_separate_signal_xlk_version_is_not_substituted_into_portfolio(market):
    sector = extra(market)
    sector.close["XLK"] *= 1.00001
    sector.open["XLK"] *= 1.00001
    result = sources.extend_market(market,sector)
    pd.testing.assert_series_equal(result.close["XLK"],market.close["XLK"])
    assert not result.close["XLK"].equals(sector.close["XLK"])


@pytest.mark.parametrize("problem",["missing_extra","missing_core","changed_core","changed_rate","bad_date","negative"])
def test_incomplete_or_substituted_sector_sources_fail_without_fill_defaults(market, problem):
    sector = extra(market)
    if problem in ("missing_extra","missing_core"):
        name = "XLE" if problem == "missing_extra" else "MTUM"
        sector = replace(sector,**{key:getattr(sector,key).drop(columns=name)
                                  for key in ("open","close","raw_close","volume")})
    elif problem == "changed_core":
        sector.close["MTUM"] *= 1.00001
    elif problem == "changed_rate":
        sector = replace(sector,risk_free=sector.risk_free+0.0001)
    elif problem == "bad_date":
        sector = replace(sector,**{key:getattr(sector,key).iloc[1:]
                                  for key in ("open","close","raw_close","volume","risk_free")})
    else:
        sector.close.loc[sector.close.index[0],"XLE"] = 0.0
    with pytest.raises(QuantError):
        sources.extend_market(market,sector)


@pytest.mark.parametrize("field,value",[("additional_assets",["XLE"]),("as_of","2026-08-31"),
                                        ("portfolio_targets_or_returns_computed",True),
                                        ("new_registered_candidates",2),("new_economic_factor_definitions",2),
                                        ("existing_W42_queue_changed",True),("order_authority",True)])
def test_source_only_study_cannot_claim_trials_or_change_scope(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        sources.validate(policy)


@pytest.mark.parametrize("field,value",[("new_quote_requests",2),("access_or_rate_limit_bypass",True),
                                        ("no_price_source_retrieval_time_rewrite",False)])
def test_original_quote_access_and_versions_cannot_be_repaired_silently(policy, field, value):
    policy["source_contract"][field] = value
    with pytest.raises(QuantError):
        sources.validate(policy)


@pytest.mark.parametrize("field",["energy_materials_funds_are_guaranteed_inflation_hedges",
                                 "sector_funds_are_new_independent_economic_factor_families",
                                 "source_readiness_is_strategy_or_investment_qualification"])
def test_source_identity_is_not_an_economic_alpha_or_hedge_guarantee(policy, field):
    policy["limits"][field] = True
    with pytest.raises(QuantError):
        sources.validate(policy)
