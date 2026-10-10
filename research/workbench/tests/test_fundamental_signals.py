from __future__ import annotations

from copy import deepcopy

import pytest

from us_quant.config import QuantError
from us_quant.fundamental_signals import fundamental_signals


def source():
    def fact(value, end="2020-12-31", start=None):
        row = {
            "val": value,
            "end": end,
            "filed": "2021-02-26",
            "form": "10-K",
            "accn": "0000000001-21-000001",
        }
        if start:
            row["start"] = start
        return row

    return {
        "cik": 1,
        "entityName": "Synthetic factor test",
        "facts": {
            "us-gaap": {
                "Assets": {"units": {"USD": [fact(100, "2019-12-31"), fact(120)]}},
                "Liabilities": {"units": {"USD": [fact(60)]}},
                "NetIncomeLoss": {"units": {"USD": [fact(11, start="2020-01-01")]}},
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {"USD": [fact(16.5, start="2020-01-01")]}
                },
            }
        },
    }


def test_two_distinct_features_reuse_period_aligned_filing_evidence():
    result = fundamental_signals(source(), "2021-03-01")
    assert result["factor_values"] == pytest.approx(
        {
            "cashflow_accrual_quality_v1": 0.05,
            "conservative_asset_growth_v1": -0.20,
        }
    )
    assert not result["objective_verified"] and not result["strategy_qualified"]
    assert not result["full_historical_universe_verified"] and not result["order_authority"]


def test_future_restatements_do_not_change_prior_factor_values():
    original = source()
    changed = deepcopy(original)
    for data in changed["facts"]["us-gaap"].values():
        later = deepcopy(data["units"]["USD"][-1])
        later.update(filed="2022-02-28", accn="0000000001-22-000001", val=9999)
        data["units"]["USD"].append(later)
    assert (
        fundamental_signals(original, "2021-03-01")["factor_values"]
        == (fundamental_signals(changed, "2021-03-01")["factor_values"])
    )


def test_filing_lag_and_missing_prior_assets_are_not_bypassed():
    with pytest.raises(QuantError):
        fundamental_signals(source(), "2021-02-26")
    data = source()
    data["facts"]["us-gaap"]["Assets"]["units"]["USD"].pop(0)
    with pytest.raises(QuantError):
        fundamental_signals(data, "2021-03-01")
