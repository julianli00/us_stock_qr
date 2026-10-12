from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.config import QuantError
from us_quant.factor_gold_risk import monthly_targets
from us_quant.multifactor_stability import FACTORS
from us_quant.storage import file_digest, read_json, write_json
from us_quant.volatility_term_risk import build_targets, load_terms, validate


@pytest.fixture
def policy():
    return read_json(Path(__file__).parents[1] / "config/volatility-term-risk.json")


@pytest.fixture
def market(market_factory):
    return market_factory("2015-08-10", "2017-06-30", ("SPY", "IEF", "GLD", "BIL", *FACTORS))


def test_inversion_uses_exact_threshold_and_preserves_gold_factor_proportions(market, policy):
    terms = pd.DataFrame({"VIX": 15.0, "VIX3M": 20.0}, index=market.close.index)
    terms.loc["2016-10-10":, "VIX"] = 20.0
    half = build_targets(market, terms, policy["candidates"][0], policy)
    exit_weights = build_targets(market, terms, policy["candidates"][1], policy)
    base = monthly_targets(
        market, read_json(Path(__file__).parents[1] / "config/factor-gold-risk.json")
    ).ffill()
    np.testing.assert_allclose(
        half.loc["2016-10-10", list(FACTORS)], base.loc["2016-10-10", list(FACTORS)] * 0.5
    )
    assert exit_weights.loc["2016-10-10", list(FACTORS)].sum() == 0
    assert half.loc["2016-10-10", "GLD"] == exit_weights.loc["2016-10-10", "GLD"]
    for frame in (half, exit_weights):
        assert np.allclose(frame.dropna(how="all").sum(axis=1), 0.98)
        assert (frame.dropna(how="all") >= 0).all().all()
        assert "VIX" not in frame.columns and "VIX3M" not in frame.columns


def test_future_index_observations_cannot_change_past_signals(market, policy):
    terms = pd.DataFrame({"VIX": 15.0, "VIX3M": 20.0}, index=market.close.index)
    before = build_targets(market, terms, policy["candidates"][0], policy)
    cutoff = pd.Timestamp("2017-01-31")
    changed = terms.copy()
    changed.loc[changed.index > cutoff, "VIX"] = 40
    after = build_targets(market, changed, policy["candidates"][0], policy)
    pd.testing.assert_frame_equal(before.loc[:cutoff], after.loc[:cutoff])


def test_term_inputs_must_be_complete_and_cannot_be_forward_filled(tmp_path):
    index = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    records = []
    for symbol in ("VIX", "VIX3M"):
        dates = index if symbol == "VIX" else index[:2]
        path = tmp_path / f"{symbol}.csv"
        pd.DataFrame({"DATE": dates.strftime("%m/%d/%Y"), "CLOSE": 20.0}).to_csv(path, index=False)
        records.append(
            {
                "symbol": symbol,
                "url": f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv",
                "sha256": file_digest(path),
            }
        )
    write_json(tmp_path / "manifest.json", {"sources": records})
    with pytest.raises(QuantError, match="no forward filling"):
        load_terms(tmp_path, index)


def test_threshold_cannot_be_tuned_after_results(policy):
    policy["risk_off_threshold"] = 0.95
    with pytest.raises(QuantError):
        validate(policy)
