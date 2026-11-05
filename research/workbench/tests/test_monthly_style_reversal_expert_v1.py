import copy

import numpy as np
import pandas as pd
import pytest

from us_quant import monthly_style_reversal_expert_v1 as law
from us_quant.calendar import is_month_end, market_calendar, sessions
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import (
    ArchivedDenseExpert,
    ArchivedMonthlyExpert,
    ExpertIdentity,
)


def market(start="2018-01-02", end="2020-12-31"):
    dates = sessions(start, end)
    trend = 1 + np.arange(len(dates))[:, None] * np.linspace(0.0001, 0.0014, 14)
    return pd.DataFrame(100 * trend, dates, law.UNIVERSE)


def month_position(close):
    return int(close.index.get_loc(pd.Timestamp("2019-04-30")))


def test_two_weakest_styles_match_independent_scalar_close_ratios_and_full_budget():
    close = market()
    position = month_position(close)
    relatives = dict(zip(law.CORE, (1.08, 0.91, 1.03, 0.97, 1.01, 1.06), strict=True))
    for symbol, relative in relatives.items():
        close.loc[close.index[position], symbol] = (
            close.loc[close.index[position - 21], symbol] * relative
        )
    target, audit = law.decision(close, position)
    independent = {
        symbol: float(close[symbol].iloc[position]) / float(close[symbol].iloc[position - 21]) - 1
        for symbol in law.CORE
    }
    ranked = sorted(independent, key=lambda symbol: (round(independent[symbol], 12), symbol))
    expected = pd.Series(
        [0.49 if symbol in ("VLUE", "USMV") else 0.0 for symbol in law.UNIVERSE],
        index=law.UNIVERSE,
    )
    pd.testing.assert_series_equal(target, expected)
    assert audit["simple_total_returns"] == independent
    assert audit["ranking_order"] == ranked
    assert audit["selected_assets"] == ["VLUE", "USMV"]
    assert target.sum() == 0.98 and (target >= 0).all() and target["BIL"] == 0


def test_all_equal_returns_use_lexical_not_declared_core_order():
    close = market()
    close.loc[:, :] = 100.0
    target, audit = law.decision(close, month_position(close))
    assert audit["ranking_order"] == sorted(law.CORE)
    assert audit["selected_assets"] == ["IJR", "MTUM"]
    assert target["IJR"] == target["MTUM"] == 0.49


def test_frozen_twelve_decimal_ranking_is_not_an_outcome_dependent_tolerance():
    close = market()
    close.loc[:, :] = 100.0
    position = month_position(close)
    close.loc[close.index[position], "VLUE"] = 100 * (1 - 1e-14)
    _, audit = law.decision(close, position)
    assert audit["simple_total_returns"]["VLUE"] < 0
    assert all(value == 0 for value in audit["rounded_ranking_values"].values())
    assert audit["selected_assets"] == ["IJR", "MTUM"]


@pytest.mark.parametrize("bad", [None, True, False, "21", 21.0, -1, 0, 20, 10_000])
def test_invalid_or_unmatured_positions_are_not_safe_or_filled_opinions(bad):
    with pytest.raises(QuantError):
        law.decision(market(), bad)


def test_only_genuine_month_ends_generate_opinions_and_calendar_is_bounded():
    close = market()
    with pytest.raises(QuantError, match="month end"):
        law.decision(close, int(close.index.get_loc(pd.Timestamp("2019-04-29"))))
    last = sessions(
        market_calendar().last_session - pd.Timedelta(days=60), market_calendar().last_session
    )
    frame = pd.DataFrame(100.0, last, law.UNIVERSE)
    with pytest.raises(QuantError, match="month end"):
        law.decision(frame, len(frame) - 1)
    with pytest.raises(QuantError, match="following session"):
        law.build_targets(frame, read_json(law.POLICY))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("lookback_return_sessions", 20),
        ("source_price_warmup_levels", 21),
        ("selected_styles", 3),
        ("selected_styles", True),
        ("ranking_decimal_places", 8),
        ("prior_completed_configurations", 228),
        ("prior_completed_configurations", 256),
        ("actual_weight_trade_band", 0.01),
        ("new_traded_assets", 1),
        ("parameter_search", True),
    ],
)
def test_fixed_law_scope_and_authoritative_prior_cannot_drift(key, value):
    policy = copy.deepcopy(read_json(law.POLICY))
    policy[key] = value
    with pytest.raises(QuantError):
        law.validate(policy)


def test_preparation_does_not_claim_admission_qualification_or_order_authority():
    policy = read_json(law.POLICY)
    law.validate(policy)
    for key in policy["methodology"]:
        changed = copy.deepcopy(policy)
        changed["methodology"][key] = True
        with pytest.raises(QuantError):
            law.validate(changed)
    changed = copy.deepcopy(policy)
    changed["new_optional_window"] = 42
    with pytest.raises(QuantError):
        law.validate(changed)


@pytest.mark.parametrize(
    "failure",
    ["gap", "missing", "zero", "infinity", "reorder", "boolean", "nullable", "timezone", "NaT"],
)
def test_invalid_market_is_not_filled_or_reclassified_as_bill_safety(failure):
    close = market()
    if failure == "gap":
        close = close.drop(close.index[10])
    elif failure == "missing":
        close.iloc[10, 0] = np.nan
    elif failure == "zero":
        close.iloc[10, 0] = 0.0
    elif failure == "infinity":
        close.iloc[10, 0] = np.inf
    elif failure == "reorder":
        close = close.loc[:, list(reversed(close.columns))]
    elif failure == "boolean":
        close = close > 0
    elif failure == "nullable":
        close = close.astype("Float64")
    elif failure == "timezone":
        close.index = close.index.tz_localize("UTC")
    else:
        close.index = close.index.where(close.index != close.index[10], pd.NaT)
    with pytest.raises(QuantError):
        law.build_targets(close, read_json(law.POLICY))


def test_non_dataframe_and_out_of_calendar_input_fail_explicitly():
    for bad in (None, {}, pd.DataFrame()):
        with pytest.raises(QuantError):
            law.build_targets(bad, read_json(law.POLICY))
    close = pd.DataFrame(100.0, pd.date_range("1990-01-01", periods=30), law.UNIVERSE)
    with pytest.raises(QuantError):
        law.build_targets(close, read_json(law.POLICY))


@pytest.mark.parametrize("failure", ["overflow", "underflow", "nonpositive_relative"])
def test_unrepresentable_price_ratio_is_not_normalized_into_a_valid_forecast(failure):
    close = market()
    position = month_position(close)
    start, end = close.index[position - 21], close.index[position]
    if failure == "overflow":
        close.loc[start, "MTUM"], close.loc[end, "MTUM"] = 1e-300, 1e300
    elif failure == "underflow":
        close.loc[start, "MTUM"], close.loc[end, "MTUM"] = 1e300, 1e-300
    else:
        close.loc[start, "MTUM"], close.loc[end, "MTUM"] = 1e20, 1.0
    with pytest.raises(QuantError):
        law.decision(close, position)


def test_native_lattice_warmup_and_input_before_resolution_are_complete():
    close = market()
    records, called = [], []
    targets, audits = law.build_targets(
        close,
        read_json(law.POLICY),
        before_decision=lambda: called.append("guarded"),
        retain=records.append,
    )
    expected = pd.DatetimeIndex(
        [day for position, day in enumerate(close.index) if position >= 21 and is_month_end(day)]
    )
    assert targets.index.equals(close.index) and targets.columns.equals(close.columns)
    assert targets.dropna(how="all").index.equals(expected)
    assert targets.iloc[:21].isna().all().all()
    assert len(audits) == len(expected) == len(called)
    assert len(records) == 2 * len(audits)
    for index, day in enumerate(expected):
        position = int(close.index.get_loc(day))
        before, after = records[2 * index : 2 * index + 2]
        assert before["record"] == "decision_input"
        assert (
            before["window_dates"]
            == close.index[position - 21 : position + 1].strftime("%Y-%m-%d").tolist()
        )
        for symbol in law.CORE:
            values = [
                float.fromhex(value) for value in before["used_source_prices_binary64_hex"][symbol]
            ]
            assert values == close[symbol].iloc[position - 21 : position + 1].tolist()
        assert after["record"] == "decision_resolved" and after["audit"] == audits[index]
        assert after["complete_target"] == targets.loc[day].tolist()


def test_prefix_equality_and_future_perturbations_do_not_change_known_opinions():
    close = market()
    policy = read_json(law.POLICY)
    targets, audits = law.build_targets(close, policy)
    for position in (20, 21, 125, 252, len(close) - 1):
        day = close.index[position]
        prefix, earlier = law.build_targets(close.iloc[: position + 1], policy)
        pd.testing.assert_frame_equal(prefix, targets.loc[:day])
        assert earlier == [row for row in audits if pd.Timestamp(row["decision_session"]) <= day]
    position = month_position(close)
    original, audit = law.decision(close, position)
    changed = close.copy()
    changed.iloc[position + 1 :] = np.nan
    target, other = law.decision(changed, position)
    pd.testing.assert_series_equal(target, original)
    assert other == audit


def test_exact_window_ignores_earlier_prices_and_per_asset_units():
    close = market()
    position = month_position(close)
    target, audit = law.decision(close, position)
    changed = close.copy()
    changed.iloc[: position - 21] *= 10000
    earlier, same = law.decision(changed, position)
    pd.testing.assert_series_equal(earlier, target)
    assert same == audit
    scaled = close * np.arange(1, len(close.columns) + 1)
    other, units = law.decision(scaled, position)
    pd.testing.assert_series_equal(other, target)
    assert units["ranking_order"] == audit["ranking_order"]


def test_existing_monthly_and_dense_adapters_keep_opinion_and_score_clocks_separate():
    close = market()
    targets, _ = law.build_targets(close, read_json(law.POLICY))
    dense_targets = pd.DataFrame(0.0, close.index, close.columns)
    dense_targets.loc[:, list(law.CORE)] = 0.98 / len(law.CORE)
    shadow = pd.DataFrame(
        {
            "equity": 10000.0,
            "return": 0.0,
            "risk_free": 0.0,
            "cash": 10000.0,
            "gross_exposure": 0.0,
            "cost": 0.0,
            "orders": 0.0,
        },
        index=close.index[252:],
    )
    monthly = ArchivedMonthlyExpert(
        ExpertIdentity(law.ID, "monthly_style_reversal", "2026-11-05T05:43:44Z"),
        targets,
        shadow,
        law.UNIVERSE,
    )
    dense = ArchivedDenseExpert(
        ExpertIdentity("synthetic_daily_core", "synthetic_fixture", "2026-11-05T05:43:44Z"),
        dense_targets,
        shadow,
        law.UNIVERSE,
    )
    assert monthly.observe(close.index[0], 126).target is None
    before_book = monthly.observe(close.index[251], 126)
    assert before_book.target is not None
    assert len(before_book.net_returns) == 0 and not before_book.score_mature
    cold = monthly.observe(shadow.index[124], 126)
    mature = monthly.observe(shadow.index[125], 126)
    assert len(cold.net_returns) == 125 and not cold.score_mature
    assert len(mature.net_returns) == 126 and mature.score_mature
    assert mature.target_known_at <= mature.as_of
    assert dense.observe(shadow.index[125], 126).target_known_at == shadow.index[125]
