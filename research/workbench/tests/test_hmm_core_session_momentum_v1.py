from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_quant import hmm_core_session_momentum_v1 as rules
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json


def test_signed_components_add_to_full_daily_return():
    close = 100 * np.exp(np.r_[0, np.cumsum(np.full(105, 0.001))])
    opening = close[:-1] * np.exp(0.003)
    values = rules.score_components(close, opening)
    assert values["close_to_close"] == pytest.approx(0.001, abs=1e-15)
    assert values["overnight"] == pytest.approx(0.003, abs=1e-15)
    assert values["intraday"] == pytest.approx(-0.002, abs=1e-15)
    assert values["close_to_close"] == pytest.approx(
        values["overnight"] + values["intraday"], abs=1e-15
    )


@pytest.mark.parametrize("flat", ["overnight", "intraday"])
def test_flat_session_has_zero_score_not_a_replacement_daily_score(flat):
    close = 100 * np.exp(np.arange(106) * 0.001)
    values = rules.score_components(close, close[:-1] if flat == "overnight" else close[1:])
    assert values[flat] == 0
    assert values["intraday" if flat == "overnight" else "overnight"] == values["close_to_close"]


def test_common_adjusted_price_scale_does_not_change_components():
    close = 100 * np.exp(np.arange(106) * 0.001)
    opening = close[:-1] * np.exp(0.003)
    a, b = rules.score_components(close, opening), rules.score_components(close * 10, opening * 10)
    for method in rules.METHODS:
        assert a[method] == pytest.approx(b[method], abs=1e-15)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_invalid_open_is_fatal_even_for_the_daily_control(bad):
    opening = np.full(105, 100.0)
    opening[12] = bad
    with pytest.raises(QuantError):
        rules.score_components(np.full(106, 100.0), opening)


def test_incomplete_session_windows_are_not_silently_truncated():
    with pytest.raises(QuantError):
        rules.score_components(np.full(106, 100.0), np.full(104, 100.0))


def fixture(end="2016-11-30"):
    index = sessions("2015-08-10", end)
    close = pd.DataFrame(100.0, index=index, columns=rules.UNIVERSE)
    opening = close.copy()
    for j, name in enumerate(rules.CORE, 1):
        close[name] *= np.exp(np.arange(len(index)) * j * 0.0001)
        opening[name] = close[name].shift(1).fillna(close[name].iloc[0]) * np.exp(j * 0.0002)
    data = MarketData(opening, close, close.copy(), close * 1000, pd.Series(0.0, index=index))
    desired = pd.DataFrame(np.nan, index=index, columns=close.columns)
    row = pd.Series(0.0, index=close.columns)
    row["SOXX"], row["BIL"] = 0.35, 0.28
    row.loc[list(rules.CORE)] = 0.35 / 4
    for i, day in enumerate(index):
        if i >= 252 and is_month_end(day):
            desired.loc[day] = row
    return data, desired


def test_all_source_months_and_noncore_targets_are_preserved_without_hmm(monkeypatch):
    from us_quant import hmm_growth_bil_v1, volatility_hmm_v1

    monkeypatch.setattr(volatility_hmm_v1, "fit", lambda *a, **k: pytest.fail("No HMM fit"))
    monkeypatch.setattr(
        hmm_growth_bil_v1, "build_targets", lambda *a, **k: pytest.fail("No parent generation")
    )
    data, desired = fixture()
    targets, audits = rules.build_targets(data, desired, read_json(rules.POLICY))
    noncore = desired.columns.difference(rules.CORE)
    for candidate in rules.CANDIDATES:
        target = targets[candidate["id"]]
        assert target.dropna().index.equals(desired.dropna().index)
        assert target.loc[:"2016-07-29"].isna().all().all()
        for day in target.dropna().index:
            oracle, _, order, _ = rules.independent_row(
                data, desired.loc[day], day, candidate["score"]
            )
            np.testing.assert_allclose(target.loc[day], oracle, rtol=0, atol=1e-12)
            assert target.loc[day, noncore].equals(desired.loc[day, noncore])
            assert (
                list(target.loc[day, list(rules.CORE)].sort_values(ascending=False).index) == order
            )
            assert target.loc[day, list(rules.CORE)].sum() == pytest.approx(0.35)
        assert audits[candidate["id"]][0]["hmm_refitted_or_states_recomputed"] is False
    assert (
        audits[rules.CANDIDATES[0]["id"]][0]["ranking"]
        != audits[rules.CANDIDATES[2]["id"]][0]["ranking"]
    )


def test_skipped_recent_prices_do_not_change_current_scores():
    data, desired = fixture()
    before, audits_before = rules.build_targets(data, desired, read_json(rules.POLICY))
    data.open.iloc[-21:, data.open.columns.get_loc("MTUM")] *= 1.2
    data.close.iloc[-21:, data.close.columns.get_loc("MTUM")] *= 1.3
    after, audits_after = rules.build_targets(data, desired, read_json(rules.POLICY))
    for identifier in before:
        pd.testing.assert_series_equal(before[identifier].iloc[-1], after[identifier].iloc[-1])
        assert audits_before[identifier][-1] == audits_after[identifier][-1]


def test_altered_hmm_carrier_is_not_accepted_as_another_budget():
    data, desired = fixture()
    day = desired.dropna().index[0]
    desired.loc[day, "GLD"] = 0.01
    desired.loc[day, "BIL"] -= 0.01
    with pytest.raises(QuantError, match="stored HMM"):
        rules.build_targets(data, desired, read_json(rules.POLICY))


def test_unsupported_early_source_month_is_not_dropped():
    data, desired = fixture()
    desired.loc[pd.Timestamp("2015-08-31")] = desired.dropna().iloc[0]
    with pytest.raises(QuantError, match="drop a source month"):
        rules.build_targets(data, desired, read_json(rules.POLICY))


def test_original_rank_precision_keeps_lexical_ties():
    assert rules.rank({"MTUM": 0.0, "QUAL": 1e-14, "VLUE": -0.01, "USMV": -0.01}) == [
        "MTUM",
        "QUAL",
        "USMV",
        "VLUE",
    ]


@pytest.mark.parametrize(
    "key,value",
    [
        ("lookback_sessions", 252),
        ("skip_recent_sessions", 0),
        ("ranking_score_decimal_places", 8),
        ("ranked_factor_shares", [0.4, 0.3, 0.2, 0.1]),
        ("candidate_count_in_this_study", 2),
    ],
)
def test_fixed_session_study_cannot_be_retuned(key, value):
    policy = read_json(rules.POLICY)
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)
