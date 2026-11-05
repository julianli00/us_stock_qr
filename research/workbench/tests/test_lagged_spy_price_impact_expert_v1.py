from copy import deepcopy
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from us_quant import lagged_spy_price_impact_expert_v1 as law
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import ArchivedMonthlyExpert, ExpertIdentity


@pytest.fixture
def data(market_factory):
    return market_factory(start="2015-01-02", end="2020-12-31", symbols=law.UNIVERSE)


def arrays(volume_multiplier=1.0):
    close = np.power(1.5, np.arange(274, dtype=float))
    raw, volume = np.full(273, 100.0), np.full(273, 1000.0)
    volume[-21:] *= volume_multiplier
    return close, raw, volume


@pytest.mark.parametrize("scale,risk_off", [(0.5, True), (1.0, False), (2.0, False)])
def test_independent_fractional_oracle_and_strict_equal_boundary(scale, risk_off):
    close, raw, volume = arrays(scale)
    measured = law.price_impact_state(close, raw, volume)
    exact = [
        abs(Fraction.from_float(float(close[i + 1])) / Fraction.from_float(float(close[i])) - 1)
        / (Fraction.from_float(float(raw[i])) * Fraction.from_float(float(volume[i])))
        for i in range(273)
    ]
    reference, current = sum(exact[:252], Fraction()) / 252, sum(exact[252:], Fraction()) / 21
    assert measured["reference_impact_mean"] == pytest.approx(float(reference), rel=1e-14)
    assert measured["current_impact_mean"] == pytest.approx(float(current), rel=1e-14)
    assert measured["rounded_relative_ratio"] == round(float(current / reference), 12)
    assert measured["risk_off_triggered"] is risk_off


@pytest.mark.parametrize("delta,expected", [(0.4e-12, False), (2e-12, True)])
def test_predeclared_ratio_precision_not_float_boundary_rescue(delta, expected):
    close, raw, volume = arrays()
    volume[-21:] /= 1 + delta
    assert law.price_impact_state(close, raw, volume)["risk_off_triggered"] is expected


def test_uniform_reported_activity_scale_changes_units_not_relative_state():
    close, raw, volume = arrays(0.5)
    first, scaled = (
        law.price_impact_state(close, raw, volume),
        law.price_impact_state(close, raw, volume * 10),
    )
    assert scaled["reference_impact_mean"] == pytest.approx(first["reference_impact_mean"] / 10)
    assert scaled["rounded_relative_ratio"] == first["rounded_relative_ratio"]
    assert scaled["risk_off_triggered"] == first["risk_off_triggered"]


def test_absolute_negative_returns_are_not_negative_illiquidity():
    _, raw, volume = arrays()
    state = law.price_impact_state(np.power(0.75, np.arange(274, dtype=float)), raw, volume)
    assert state["reference_impact_mean"] == pytest.approx(0.25 / 100000)
    assert state["current_impact_mean"] == pytest.approx(0.25 / 100000)
    assert state["rounded_relative_ratio"] == 1.0 and state["risk_off_triggered"] is False


@pytest.mark.parametrize(
    "failure", ["shape", "zero_volume", "negative_price", "NaN", "boolean", "flat"]
)
def test_missing_invalid_or_degenerate_inputs_are_not_bill_fallback(failure):
    close, raw, volume = arrays()
    if failure == "shape":
        close = close[1:]
    elif failure == "zero_volume":
        volume[0] = 0
    elif failure == "negative_price":
        raw[0] = -1
    elif failure == "NaN":
        close[1] = np.nan
    elif failure == "boolean":
        volume = volume > 0
    else:
        close[:] = 1.0
    with pytest.raises(QuantError):
        law.price_impact_state(close, raw, volume)


@pytest.mark.parametrize(
    "key,value",
    [
        ("input_lag_sessions", 0),
        ("current_impact_sessions", 42),
        ("nonoverlapping_reference_impact_sessions", 126),
        ("risk_off_ratio_strictly_greater_than", 1.1),
        ("relative_ratio_decimal_places", 8),
        ("actual_weight_trade_band", 0.01),
        ("prior_completed_configurations", 262),
    ],
)
def test_no_window_lag_threshold_band_or_current_count_retune(key, value):
    policy = deepcopy(read_json(law.POLICY))
    policy[key] = value
    with pytest.raises(QuantError):
        law.validate(policy)


def test_source_warmup_does_not_invent_an_initial_opinion(data):
    policy = read_json(law.POLICY)
    targets, audits = law.build_targets(data, policy)
    expected = pd.DatetimeIndex(
        [day for index, day in enumerate(data.close.index) if index >= 274 and is_month_end(day)]
    )
    assert targets.dropna(how="all").index.equals(expected)
    assert targets.iloc[:274].isna().all().all()
    assert len(audits) == len(expected)
    with pytest.raises(QuantError, match="warmup"):
        law.decision(data, pd.Timestamp("2015-12-31"))


def test_nonoverlapping_windows_end_before_own_decision_bar(data):
    _, audit = law.decision(data, pd.Timestamp("2016-09-30"))
    assert audit["reference_input_end"] < audit["current_input_start"]
    assert audit["last_input_session"] == "2016-09-29"
    assert audit["last_input_session"] < audit["decision_session"]
    assert audit["reference_impact_observations"] == 252
    assert audit["current_impact_observations"] == 21
    assert audit["decision_session_own_market_bar_used_to_choose_weight"] is False


def test_own_and_future_valid_market_bars_do_not_choose_past_opinion(data):
    day = pd.Timestamp("2016-09-30")
    target, audit = law.decision(data, day)
    changed = MarketData(
        data.open.copy(),
        data.close.copy(),
        data.raw_close.copy(),
        data.volume.copy(),
        data.risk_free.copy(),
    )
    changed.close.loc[day:, "SPY"] *= 1.01
    changed.raw_close.loc[day:, "SPY"] *= 1.01
    changed.volume.loc[day:, "SPY"] *= 3
    other, later = law.decision(changed, day)
    pd.testing.assert_series_equal(other, target)
    assert later == audit


def test_fixed_core_or_actual_bill_target_and_idle_budget(data):
    day = pd.Timestamp("2016-09-30")
    _, initial = law.decision(data, day)
    position = int(data.close.index.get_loc(day))
    stressed = MarketData(
        data.open.copy(),
        data.close.copy(),
        data.raw_close.copy(),
        data.volume.copy(),
        data.risk_free.copy(),
    )
    stressed.volume.iloc[position - 21 : position, stressed.volume.columns.get_loc("SPY")] *= (
        max(initial["relative_impact_ratio"], 1e-10) / 10
    )
    target, audit = law.decision(stressed, day)
    assert audit["risk_off_triggered"] is True
    assert target["BIL"] == 0.98 and target.sum() == 0.98
    assert target.loc[list(law.CORE)].sum() == 0
    assert audit["proxy_is_measured_market_depth_flow_or_execution_cost"] is False


def test_retained_input_precedes_resolution_and_references_only_lagged_actual_values(data):
    records, guarded = [], []
    targets, audits = law.build_targets(
        data,
        read_json(law.POLICY),
        retain=records.append,
        before_decision=lambda: guarded.append(True),
    )
    assert len(records) == 2 * len(audits) and len(guarded) == len(audits)
    for i, audit in enumerate(audits):
        before, after = records[2 * i : 2 * i + 2]
        assert before["record"] == "decision_input" and after["record"] == "decision_resolved"
        assert len(before["adjusted_close_binary64_hex"]) == 274
        assert len(before["reported_close_binary64_hex"]) == 273
        assert len(before["reported_share_volume_binary64_hex"]) == 273
        assert before["impact_return_session_dates"][-1] == audit["last_input_session"]
        assert before["impact_return_session_dates"][-1] < before["decision_session"]
        assert after["audit"] == audit
        assert (
            after["complete_target"]
            == targets.loc[pd.Timestamp(audit["decision_session"])].tolist()
        )


def test_input_evidence_survives_a_degenerate_decision_without_safe_fallback(data):
    data.close.loc[:, "SPY"] = 100.0
    records = []
    with pytest.raises(QuantError, match="degenerate"):
        law.build_targets(data, read_json(law.POLICY), retain=records.append)
    assert len(records) == 1 and records[0]["record"] == "decision_input"


@pytest.mark.parametrize("failure", ["gap", "volume", "remap"])
def test_admitted_market_quality_and_full_universe_cannot_be_bypassed(data, failure):
    if failure == "gap":
        day = data.close.index[20]
        data = MarketData(
            data.open.drop(day),
            data.close.drop(day),
            data.raw_close.drop(day),
            data.volume.drop(day),
            data.risk_free.drop(day),
        )
    elif failure == "volume":
        data.volume.iloc[0, data.volume.columns.get_loc("SPY")] = 0
    else:
        columns = list(reversed(data.close.columns))
        data = MarketData(
            data.open.loc[:, columns],
            data.close.loc[:, columns],
            data.raw_close.loc[:, columns],
            data.volume.loc[:, columns],
            data.risk_free,
        )
    with pytest.raises(QuantError):
        law.build_targets(data, read_json(law.POLICY))


def test_prefix_causality_and_NET_maturity_are_distinct_from_price_warmup(data):
    targets, audits = law.build_targets(data, read_json(law.POLICY))
    day = pd.Timestamp("2017-04-28")
    earlier = MarketData(
        data.open.loc[:day],
        data.close.loc[:day],
        data.raw_close.loc[:day],
        data.volume.loc[:day],
        data.risk_free.loc[:day],
    )
    prefix, first = law.build_targets(earlier, read_json(law.POLICY))
    pd.testing.assert_frame_equal(prefix, targets.loc[:day])
    assert first == [row for row in audits if pd.Timestamp(row["decision_session"]) <= day]
    dates = sessions("2016-10-06", "2020-12-31")
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
        index=dates,
    )
    expert = ArchivedMonthlyExpert(
        ExpertIdentity(law.ID, "lagged_market_price_impact_participation", "2026-11-05T09:27:44Z"),
        targets,
        shadow,
        law.UNIVERSE,
    )
    assert expert.observe(pd.Timestamp("2015-12-31"), 126).target is None
    assert len(expert.observe(pd.Timestamp("2016-10-05"), 126).net_returns) == 0
    cold, mature = [expert.observe(dates[i], 126) for i in (124, 125)]
    assert len(cold.net_returns) == 125 and not cold.score_mature
    assert len(mature.net_returns) == 126 and mature.score_mature
