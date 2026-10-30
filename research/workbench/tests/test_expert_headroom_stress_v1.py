from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from us_quant import expert_headroom_stress_v1 as model
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end, sessions
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.factor_validation import independent_metrics
from us_quant.storage import read_json


def synthetic_market(flat=False):
    dates = sessions("2019-06-03", "2020-04-30")
    phase = np.arange(len(dates))
    close = pd.DataFrame(
        {
            symbol: np.full(len(dates), 100.0)
            if flat
            else 100
            * np.exp(
                np.cumsum(
                    np.full(len(dates), 0.00005)
                    if symbol == "BIL"
                    else 0.0002 + 0.01 * np.sin(phase / (3 + i))
                )
            )
            for i, symbol in enumerate(model.UNIVERSE)
        },
        index=dates,
    )
    opening = close.shift(1)
    opening.iloc[0] = close.iloc[0]
    data = MarketData(
        opening,
        close,
        close.copy(),
        close * 0 + 1e6,
        pd.Series(0.00005, index=dates, name="risk_free"),
    )
    nominal = pd.DataFrame(np.nan, index=dates, columns=model.UNIVERSE)
    for day in dates:
        if day == pd.Timestamp("2020-01-30") or (
            day > pd.Timestamp("2020-01-30") and is_month_end(day)
        ):
            nominal.loc[day] = 0.0
            nominal.loc[day, "GLD" if day.month == 2 else "SOXX"] = 0.98
    return data, nominal


def run(data, nominal, cost=5.0, delay=1, start="2020-01-31", end="2020-04-30"):
    return model.run(
        data, nominal, model.CANDIDATES[0], read_json(model.POLICY), start, end, cost, delay
    )


@pytest.mark.parametrize(
    "intercepts,slopes,budget,beta,feasible,lower,upper",
    [
        ([0.01, 0.12], [0.2, -0.1], 0.1, 0.45, True, 0.2, 0.45),
        ([0.2], [0.0], 0.1, 0.0, False, 0.0, 1.0),
        ([0.0, 0.2], [0.2, -0.1], 0.05, 0.0, False, 1.5, 0.25),
        ([0.2], [-0.2], 0.1, 1.0, True, 0.5, 1.0),
        ([0.2], [0.1], 0.1, 0.0, False, 0.0, -1.0),
        ([-0.1], [-0.1], 0.0, 1.0, True, 0.0, 1.0),
    ],
)
def test_affine_interval_handles_both_slope_signs_and_contradictions(
    intercepts, slopes, budget, beta, feasible, lower, upper
):
    actual, certificate = model.affine_interval(np.array(intercepts), np.array(slopes), budget)
    assert actual == pytest.approx(beta)
    assert certificate["feasible"] is feasible
    assert certificate["lower_fraction"] == pytest.approx(lower)
    assert certificate["upper_fraction"] == pytest.approx(upper)
    if slopes == [0.0]:
        assert certificate["zero_slope_violation_scenario"] == 0
    if feasible:
        assert certificate["maximum_scenario_residual_at_applied_fraction"] <= 1e-10
    json.dumps(certificate, allow_nan=False)


@pytest.mark.parametrize("delay", [1, 2])
def test_review_horizon_uses_next_real_month_end_and_original_delay(delay):
    review, execution, horizon = model.review_horizon("2020-01-30", delay)
    assert review == pd.Timestamp("2020-01-31")
    assert execution == pd.Timestamp("2020-02-03" if delay == 1 else "2020-02-04")
    assert horizon == delay + 1
    review, execution, horizon = model.review_horizon("2020-01-31", delay)
    assert review == pd.Timestamp("2020-02-28")
    assert execution == pd.Timestamp("2020-03-02" if delay == 1 else "2020-03-03")
    assert horizon == len(sessions("2020-02-03", execution))
    with pytest.raises(QuantError, match="delay"):
        model.review_horizon("2020-01-31", True)


@pytest.mark.parametrize(
    "nav,gross", [(10000.0, 0.0), (10000.0, 0.98), (9000.0, 0.98), (8490.0, 0.98)]
)
@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_independent_scalar_stress_scenarios_fraction_and_funding_match(nav, gross, cost, delay):
    data, nominal = synthetic_market()
    day = pd.Timestamp("2020-01-31")
    current = pd.Series(0.0, index=model.UNIVERSE)
    current["BIL"] = gross
    state = model.MainCloseState(day, nav, 10000.0, current)
    target, audit = model.resolve(nominal.loc[day], state, data.close.loc[:day], cost, delay)
    checked = model.verify_resolution(
        nominal.loc[day], state, data.close.loc[:day], cost, delay, target, audit
    )
    assert checked["maximum_target_difference"] < 1e-10
    assert checked["maximum_scenario_coefficient_difference"] < 1e-10
    assert len(audit["scenarios"]) == 127 - audit["known_calendar_horizon_sessions"]
    assert target.sum() == pytest.approx(0.98)
    assert 0 <= audit["applied_fraction"] <= 1
    assert target["BIL"] >= nominal.loc[day, "BIL"]
    assert audit["reference_is_actual_future_loss_or_drawdown_guarantee"] is False
    json.dumps(audit, allow_nan=False)


def test_signed_headroom_and_current_close_allowance_are_not_charged_fees():
    data, nominal = synthetic_market(flat=True)
    day = pd.Timestamp("2020-01-30")
    current = pd.Series(0.0, index=model.UNIVERSE)
    current["SOXX"] = 0.98
    state = model.MainCloseState(day, 8490.0, 10000.0, current)
    target, audit = model.resolve(nominal.loc[day], state, data.close.loc[:day], 20.0, 2)
    assert audit["signed_main_headroom_USD"] == -10
    assert audit["current_close_cost_allowance_USD"] == pytest.approx(8490 * 1.96 * 20 / 10000 + 14)
    assert audit["available_stress_loss_fraction"] < 0
    assert audit["feasible"] is False and target["BIL"] == pytest.approx(0.98)
    assert audit["allowance_is_deducted_fee_or_guaranteed_future_open_cap"] is False
    result, _, decisions = run(data, nominal)
    first = result.frame.iloc[0]
    assert first["equity"] == pytest.approx(10000 - first["cost"])
    assert first["cost"] < decisions[0]["stress"]["current_close_cost_allowance_USD"]


def test_short_history_is_explicitly_unscored_and_bad_prices_are_errors():
    data, nominal = synthetic_market()
    day = pd.Timestamp("2020-01-31")
    state = model.MainCloseState(day, 10000.0, 10000.0, pd.Series(0.0, index=model.UNIVERSE))
    history = data.close.loc[:day].tail(126)
    target, audit = model.resolve(nominal.loc[day], state, history, 5.0, 1)
    assert audit["feasible"] is None and audit["stress_sample_available"] is False
    assert audit["certificate"] is None and audit["scenarios"] == []
    assert target["BIL"] == pytest.approx(0.98)
    model.verify_resolution(nominal.loc[day], state, history, 5.0, 1, target, audit)
    for value in (np.nan, 0.0, -1.0):
        broken = data.close.loc[:day].copy()
        broken.iloc[-1, 0] = value
        with pytest.raises(QuantError, match="known-close"):
            model.resolve(nominal.loc[day], state, broken, 5.0, 1)
    with pytest.raises(QuantError, match="known-close"):
        model.resolve(
            nominal.loc[day], state, data.close.loc[:day].drop(day - pd.Timedelta(days=1)), 5.0, 1
        )
    with pytest.raises(QuantError, match="known-close"):
        model.resolve(nominal.loc[day], state, data.close, 5.0, 1)


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_actual_monthly_feedback_keeps_continuous_main_peak_and_full_calendar(cost, delay):
    data, nominal = synthetic_market()
    result, issued, decisions = run(data, nominal, cost, delay)
    proof = model.verify_feedback(
        data, nominal, result, issued, decisions, "2020-01-31", "2020-04-30", cost, delay
    )
    expected = ["2020-01-30", "2020-01-31", "2020-02-28", "2020-03-31", "2020-04-30"]
    assert [r["signal_session"] for r in decisions] == expected
    assert proof["checked_decision_states"] == 5
    assert proof["initial_capital_and_peak_are_fresh_main_10000"] is True
    assert proof["source_shadow_wealth_or_peak_imported"] is False
    assert result.frame.index.equals(sessions("2020-01-31", "2020-04-30"))
    assert (result.frame["cash"] >= 0).all() and (result.frame["gross_exposure"] <= 1).all()
    for row in decisions[1:]:
        day = row["signal_session"]
        assert row["known_main_NAV_USD"] == result.frame.loc[day, "equity"]
        assert row["known_main_peak_USD"] == max(10000, result.frame.loc[:day, "equity"].max())
    if delay == 2:
        assert result.frame.iloc[0]["equity"] == 10000


def test_initial_pending_request_is_retained_without_overwrite():
    data, nominal = synthetic_market()
    nominal.loc["2020-01-31"] = 0.0
    nominal.loc["2020-01-31", "GLD"] = 0.98
    _, issued, rows = run(data, nominal, 20.0, 2)
    assert rows[0]["execution_session"] == "2020-02-03"
    assert rows[1]["reason"] == "pending_request"
    assert rows[1]["source_nominal_session"] == "2020-01-31"
    assert rows[1]["desired_reference_target"][model.UNIVERSE.index("GLD")] > 0
    assert rows[1]["pending_executions"] == ["2020-02-03"]
    assert rows[1]["request_issued"] is False and issued.loc["2020-01-31"].isna().all()
    assert issued.loc["2020-01-30", "SOXX"] > 0


def test_fresh_later_account_never_inherits_source_or_long_main_peak():
    data, nominal = synthetic_market()
    long, _, _ = run(data, nominal)
    later, _, rows = run(data, nominal, start="2020-03-02")
    assert long.frame.loc["2020-02-28", "equity"] != 10000
    assert rows[0]["known_main_NAV_USD"] == rows[0]["known_main_peak_USD"] == 10000
    assert rows[0]["signal_session"] == "2020-02-28"
    assert later.frame.index[0] == pd.Timestamp("2020-03-02")


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_prefix_and_future_perturbations_preserve_known_monthly_feedback(cost, delay):
    data, nominal = synthetic_market()
    end = pd.Timestamp("2020-03-13")
    full, full_issued, full_rows = run(data, nominal, cost, delay)
    prefix = MarketData(
        data.open.loc[:end],
        data.close.loc[:end],
        data.raw_close.loc[:end],
        data.volume.loc[:end],
        data.risk_free.loc[:end],
    )
    truncated, truncated_issued, truncated_rows = run(
        prefix, nominal.loc[:end], cost, delay, end=str(end.date())
    )
    pd.testing.assert_frame_equal(full.frame.loc[:end], truncated.frame, check_exact=True)
    pd.testing.assert_frame_equal(full_issued.loc[:end], truncated_issued, check_exact=True)
    assert [r for r in full_rows if r["signal_session"] <= str(end.date())] == truncated_rows
    close, opening, later_nominal = data.close.copy(), data.open.copy(), nominal.copy()
    close.loc[close.index > end] *= 1.4
    opening.loc[opening.index > end] *= 1.4
    later_nominal.loc["2020-03-31":] = np.nan
    for day in later_nominal.loc["2020-03-31":].index:
        if is_month_end(day):
            later_nominal.loc[day] = 0.0
            later_nominal.loc[day, "BIL"] = 0.98
    changed = MarketData(opening, close, close.copy(), data.volume, data.risk_free)
    after, after_issued, after_rows = run(changed, later_nominal, cost, delay)
    pd.testing.assert_frame_equal(full.frame.loc[:end], after.frame.loc[:end], check_exact=True)
    pd.testing.assert_frame_equal(full_issued.loc[:end], after_issued.loc[:end], check_exact=True)
    assert [r for r in after_rows if r["signal_session"] <= str(end.date())] == truncated_rows


def test_holding_band_does_not_force_below_band_reference_orders():
    data, nominal = synthetic_market(flat=True)
    for day in nominal.dropna().index:
        nominal.loc[day] = 0.0
        nominal.loc[day, "SOXX"] = 0.98
    nominal.loc["2020-02-28", ["SOXX", "GLD"]] = [0.97, 0.01]
    _, issued, rows = run(data, nominal)
    row = next(r for r in rows if r["signal_session"] == "2020-02-28")
    assert row["reason"] == "hold" and row["request_issued"] is False
    assert row["maximum_observed_target_gap"] < 0.025
    assert row["desired_reference_target"][model.UNIVERSE.index("GLD")] == 0.01
    assert issued.loc["2020-02-28"].isna().all()


def test_delayed_gap_can_breach_reference_without_a_drawdown_guarantee():
    data, nominal = synthetic_market(flat=True)
    for day in nominal.dropna().index:
        nominal.loc[day] = 0.0
        nominal.loc[day, "SOXX"] = 0.98
    close = data.close.copy()
    close.loc["2020-02-03":, "SOXX"] = 70.0
    shocked = MarketData(close.copy(), close, close.copy(), data.volume, data.risk_free)
    result, _, rows = run(shocked, nominal)
    dd = 1 - result.frame["equity"] / result.frame["equity"].cummax().clip(lower=10000)
    assert dd.max() > 0.15
    assert not any(r["signal_session"] == "2020-02-03" for r in rows)
    after_breach = next(r for r in rows if r["signal_session"] == "2020-02-28")
    assert after_breach["stress"]["signed_main_headroom_USD"] < 0
    assert after_breach["stress"]["feasible"] is False
    assert after_breach["stress"]["reference_is_actual_future_loss_or_drawdown_guarantee"] is False


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_funded_replay_and_independent_bt_match_the_new_closed_loop(cost, delay):
    data, nominal = synthetic_market()
    result, issued, _ = run(data, nominal, cost, delay)
    replay = simulate(
        data,
        issued,
        "2020-01-31",
        "2020-04-30",
        initial_capital=10000,
        cost_bps=cost,
        commission=1,
        delay=delay,
    )
    np.testing.assert_allclose(result.frame, replay.frame, rtol=0, atol=1e-8)
    np.testing.assert_allclose(result.weights, replay.weights, rtol=0, atol=1e-12)
    independent = independent_equity(
        data,
        issued,
        "2020-01-31",
        "2020-04-30",
        capital=10000,
        cost_bps=cost,
        commission=1,
        delay=delay,
    )
    _, difference = independent_metrics(
        result.frame, independent, data.risk_free.loc[result.frame.index], 10000
    )
    assert difference < 1e-8


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_final_delayed_request_does_not_force_terminal_liquidation(cost, delay):
    data, nominal = synthetic_market(flat=True)
    result, issued, rows = run(data, nominal, cost, delay, end="2020-02-28")
    assert rows[-1]["request_issued"] is True
    assert rows[-1]["execution_session"] > "2020-02-28"
    assert issued.loc["2020-02-28", "GLD"] == pytest.approx(0.98)
    assert result.frame.iloc[-1]["orders"] == result.frame.iloc[-1]["cost"] == 0
    assert result.weights.iloc[-1]["SOXX"] > 0.9


@pytest.mark.parametrize(
    "damage", ["state", "NaN", "date", "coefficient", "bound", "witness", "target"]
)
def test_tampered_stress_state_scenario_or_certificate_is_rejected(damage):
    data, nominal = synthetic_market()
    day = pd.Timestamp("2020-01-31")
    state = model.MainCloseState(day, 10000.0, 10000.0, pd.Series(0.0, index=model.UNIVERSE))
    target, original = model.resolve(nominal.loc[day], state, data.close.loc[:day], 5.0, 1)
    audit = copy.deepcopy(original)
    if damage == "state":
        audit["known_main_peak_USD"] = 11000
    elif damage == "NaN":
        audit["applied_fraction"] = np.nan
    elif damage == "date":
        audit["scenarios"][0]["last_price_session"] = "2026-10-05"
    elif damage == "coefficient":
        audit["scenarios"][0]["nonBIL_fraction_loss_slope"] += 0.01
    elif damage == "bound":
        audit["certificate"]["upper_fraction"] -= 0.1
    elif damage == "witness":
        audit["certificate"]["binding_upper_scenario"] = 100000
    else:
        target = target.copy()
        target["SOXX"] -= 0.001
        target["BIL"] += 0.001
    with pytest.raises(QuantError):
        model.verify_resolution(
            nominal.loc[day], state, data.close.loc[:day], 5.0, 1, target, audit
        )


def test_unissued_reference_and_nonmonthly_requests_cannot_be_rewritten():
    data, nominal = synthetic_market(flat=True)
    result, issued, rows = run(data, nominal)
    held = next(r for r in rows if r["reason"] == "hold")
    altered = copy.deepcopy(rows)
    row = next(r for r in altered if r["signal_session"] == held["signal_session"])
    row["desired_reference_target"][model.UNIVERSE.index("SOXX")] -= 0.001
    row["desired_reference_target"][model.UNIVERSE.index("BIL")] += 0.001
    with pytest.raises(QuantError):
        model.verify_feedback(
            data, nominal, result, issued, altered, "2020-01-31", "2020-04-30", 5.0, 1
        )
    changed = issued.copy()
    changed.loc["2020-02-04"] = 0.0
    changed.loc["2020-02-04", "BIL"] = 0.98
    with pytest.raises(QuantError, match="unique original"):
        model.verify_feedback(
            data, nominal, result, changed, rows, "2020-01-31", "2020-04-30", 5.0, 1
        )


def test_policy_cannot_claim_a_guarantee_or_import_nonfunded_state():
    policy = read_json(model.POLICY)
    policy["methodology"]["main_drawdown_guarantee"] = True
    with pytest.raises(QuantError):
        model.validate(policy)
    data, nominal = synthetic_market()
    for nav, peak in ((0, 10000), (10000, 9999), (np.nan, 10000)):
        state = model.MainCloseState(
            pd.Timestamp("2020-01-31"), nav, peak, pd.Series(0.0, index=model.UNIVERSE)
        )
        with pytest.raises(QuantError, match="main NAV"):
            model.resolve(nominal.loc["2020-01-31"], state, data.close.loc[:"2020-01-31"], 5.0, 1)
