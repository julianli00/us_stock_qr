from copy import deepcopy
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from us_quant import style_activity_participation_v1 as law
from us_quant import style_activity_source_v1 as source
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json


def examples():
    dates = sessions("2015-08-10", "2017-10-05")
    t = np.arange(len(dates))
    prices = np.column_stack(
        [100 + 0.03 * t + np.sin(t / (4 + i)) * (i + 1) for i in range(len(law.UNIVERSE))]
    )
    close = pd.DataFrame(prices, index=dates, columns=law.UNIVERSE)
    volume = pd.DataFrame(100000.0, index=dates, columns=law.UNIVERSE)
    return MarketData(
        close.copy(),
        close.copy(),
        close.copy(),
        volume,
        pd.Series(0.00001, index=dates, name="risk_free"),
    )


def test_signed_activity_balance_matches_fraction_oracle_not_return_magnitude_momentum():
    steps = np.r_[np.ones(42), -np.ones(21)]
    close = np.r_[100, 100 + np.cumsum(steps)]
    raw, volume = np.arange(63) + 10.0, (np.arange(63) + 1.0) * 100
    expected = sum(
        Fraction(float(raw[i])) * Fraction(float(volume[i])) * int(steps[i]) for i in range(63)
    ) / sum(Fraction(float(raw[i])) * Fraction(float(volume[i])) for i in range(63))
    assert law.balance(close, raw, volume) == pytest.approx(float(expected), abs=1e-14)
    assert law.balance(close, raw, volume * 10) == pytest.approx(float(expected), abs=1e-14)
    constant = np.ones(63)
    assert law.balance(close, constant, constant) == pytest.approx(1 / 3)
    assert law.balance(np.arange(64) + 100.0, constant, constant) == 1
    assert law.balance(200.0 - np.arange(64), constant, constant) == -1
    assert law.balance(np.full(64, 100.0), constant, constant) == 0


@pytest.mark.parametrize(
    "kind",
    [
        "shape",
        "NaN",
        "zero_close",
        "zero_volume",
        "negative",
        "fractional_volume",
        "object",
        "overflow",
    ],
)
def test_missing_invalid_or_degenerate_inputs_are_fatal_not_cash_fallback(kind):
    close, raw, volume = np.arange(64) + 100.0, np.full(63, 100.0), np.full(63, 1000.0)
    if kind == "shape":
        close = close[:-1]
    elif kind == "NaN":
        close[0] = np.nan
    elif kind == "zero_close":
        close[0] = 0
    elif kind == "zero_volume":
        volume[0] = 0
    elif kind == "negative":
        raw[0] = -1
    elif kind == "fractional_volume":
        volume[0] = 0.5
    elif kind == "object":
        raw = raw.astype(str)
    else:
        raw[:], volume[:] = 1e308, 1e308
    with pytest.raises(QuantError):
        law.balance(close, raw, volume)


def test_positive_count_selection_lexical_ties_and_full_unleveraged_asset_vector():
    scores = dict(zip(law.CORE, (0.2, 0.3, 0.4, 0.5), strict=True))
    target, audit = law.allocation(scores)
    assert audit["selected_assets"] == ["USMV", "QUAL", "VLUE"]
    assert tuple(target.index) == law.UNIVERSE
    assert (target >= 0).all() and target.sum() == pytest.approx(0.98)
    assert target["BIL"] == 0
    tied, result = law.allocation(dict.fromkeys(law.CORE, 0.25))
    assert result["selected_assets"] == sorted(law.CORE)[:3]
    assert (tied.loc[result["selected_assets"]] == 0.98 / 3).all()


def test_positive_eligibility_is_unrounded_and_rank_precision_does_not_change_zero_threshold():
    scores = dict(zip(law.CORE, (1e-14, 2e-14, 3e-14, 0.0), strict=True))
    target, result = law.allocation(scores)
    assert result["strictly_positive_style_count"] == 3
    assert result["selected_assets"] == sorted(law.CORE[:3])
    assert target["BIL"] == 0


@pytest.mark.parametrize("positive_count", [0, 1, 2])
def test_defined_insufficient_participation_is_distinct_from_missing_input(positive_count):
    scores = {s: 0.1 if i < positive_count else 0.0 for i, s in enumerate(law.CORE)}
    target, result = law.allocation(scores)
    assert target["BIL"] == 0.98 and target.sum() == 0.98
    assert result["reason"] == "defined_insufficient_positive_activity_participation"
    assert result["invalid_input_defaulted_to_cash"] is False
    del scores[law.CORE[0]]
    with pytest.raises(QuantError):
        law.allocation(scores)


def test_native_lag_full_target_prefix_and_no_fake_capital_anchor_opinion():
    data = examples()
    day = pd.Timestamp("2016-09-30")
    changed = deepcopy(data)
    for field in ("open", "close", "raw_close"):
        getattr(changed, field).loc[day:, list(law.CORE)] *= 1.2
    changed.volume.loc[day:, list(law.CORE)] *= 2
    policy = read_json(law.POLICY)
    records = []
    full, audits = law.build_targets(data, policy, retain=records.append)
    later, after = law.build_targets(changed, policy)
    prefix = MarketData(
        *[
            getattr(data, k).loc[:day].copy()
            for k in ("open", "close", "raw_close", "volume", "risk_free")
        ]
    )
    early, before = law.build_targets(prefix, policy)
    assert early.equals(full.loc[:day])
    assert early.equals(later.loc[:day])
    assert before == [a for a in audits if a["decision_session"] <= str(day.date())]
    assert before == [a for a in after if a["decision_session"] <= str(day.date())]
    assert len(records) == 2 * len(audits)
    assert full.loc["2016-10-05"].isna().all()
    assert not full.loc["2016-09-30"].isna().any()
    assert len(records[0]["inputs_binary64_hex"][law.CORE[0]]["adjusted_closes"]) == 64
    assert all(a["last_input_session"] < a["decision_session"] for a in audits)
    assert all(a["price_warmup_implies_mature_NET_score"] is False for a in audits)


def test_no_ordinary_nonmonthly_or_unwarmed_source_opinion():
    data = examples()
    with pytest.raises(QuantError):
        law.decision(data, pd.Timestamp("2016-10-05"))
    with pytest.raises(QuantError):
        law.decision(data, pd.Timestamp("2015-09-30"))
    short = MarketData(
        *[
            getattr(data, k).iloc[:40].copy()
            for k in ("open", "close", "raw_close", "volume", "risk_free")
        ]
    )
    with pytest.raises(QuantError):
        law.build_targets(short, read_json(law.POLICY))


def test_existing_admission_field_comparison_preserves_exact_volume_and_original_tolerances():
    data = examples()
    old = {k: getattr(data, k).copy() for k in ("open", "close", "raw_close", "volume")}
    checked = source.compare_fields(old, data)
    assert checked["reported_share_volume_exact_and_positive_integer"] is True
    assert checked["old_activity_source_RF_imported_or_normalized"] is False
    assert checked["old_policy_native_RF_claimed_equal_to_current_native_RF"] is False
    old["volume"].iloc[2, old["volume"].columns.get_loc("MTUM")] += 1
    with pytest.raises(QuantError):
        source.compare_fields(old, data)


def test_old_activity_native_RF_is_not_imported_compared_or_rewritten():
    data = examples()
    before = data.risk_free.copy()
    old = {k: getattr(data, k).copy() for k in ("open", "close", "raw_close", "volume")}
    source.compare_fields(old, data)
    assert data.risk_free.equals(before)
    old["risk_free"] = pd.DataFrame({"risk_free": 0.5}, index=data.close.index)
    with pytest.raises(QuantError):
        source.compare_fields(old, data)


@pytest.mark.parametrize(
    "key,value",
    [
        ("return_and_activity_observations", 126),
        ("input_lag_sessions", 0),
        ("minimum_positive_activity_balances", 2),
        ("selected_styles_when_participating", 4),
        ("positive_balance_threshold", -0.01),
        ("old_activity_source_RF_imported_or_normalized", True),
        ("source_field_scope_expanded_beyond_original_four", True),
        ("expert_technically_admitted", True),
        ("prior_completed_configurations", 269),
    ],
)
def test_no_window_lag_scope_RF_membership_or_prior_rescue(key, value):
    policy = deepcopy(read_json(law.POLICY))
    policy[key] = value
    with pytest.raises(QuantError):
        law.validate(policy)
