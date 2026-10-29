from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from us_quant import expert_master_protection_v1 as adapted
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end, sessions
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.factor_validation import independent_metrics
from us_quant.semiconductor_portfolio_protection import protected_target
from us_quant.storage import read_json

SYMBOLS = (
    "SPY",
    "IEF",
    "TLT",
    "GLD",
    "BIL",
    "MTUM",
    "VLUE",
    "QUAL",
    "USMV",
    "QQQ",
    "XLK",
    "SOXX",
    "IJR",
    "PKW",
)


def synthetic_market():
    dates = sessions("2020-01-02", "2020-04-30")
    phase = np.arange(len(dates))
    close = pd.DataFrame(
        {
            name: 100 * np.exp(np.cumsum(0.0002 + 0.004 * np.sin(phase / (2 + i))))
            for i, name in enumerate(SYMBOLS)
        },
        index=dates,
    )
    opening = close.shift(1)
    opening.iloc[0] = close.iloc[0]
    data = MarketData(
        opening,
        close,
        close.copy(),
        pd.DataFrame(1e6, index=dates, columns=SYMBOLS),
        pd.Series(0.00005, index=dates, name="risk_free"),
    )
    nominal = pd.DataFrame(np.nan, index=dates, columns=SYMBOLS)
    for day in dates:
        if day == pd.Timestamp("2020-01-30") or is_month_end(day):
            nominal.loc[day] = 0.0
            if day.month == 2:
                nominal.loc[day, ["GLD", "BIL"]] = [0.49, 0.49]
            else:
                nominal.loc[day, ["SOXX", "MTUM", "IJR", "PKW"]] = [0.49, 0.245, 0.1225, 0.1225]
    return data, nominal


def run(data, nominal, cost=5.0, delay=1, start="2020-01-31", end="2020-04-30"):
    return adapted.run(
        data,
        nominal,
        adapted.CANDIDATES[0],
        read_json(adapted.POLICY),
        start,
        end,
        cost,
        delay,
    )


@pytest.mark.parametrize("nav,peak", [(10000, 10000), (9400, 11000), (8500, 10000), (8400, 10000)])
@pytest.mark.parametrize("bil", [0.0, 0.49, 0.98])
def test_scalar_oracle_matches_original_law_without_transferring_wealth(nav, peak, bil):
    nominal = pd.Series((0.98 - bil) / 13, index=SYMBOLS)
    nominal["BIL"] = bil
    expected, _ = adapted.scalar_target(nominal, nav, peak)
    actual = protected_target(nominal, nav, peak)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    assert actual["BIL"] + 1e-12 >= bil
    assert actual.sum() == pytest.approx(0.98)
    assert (actual >= 0).all()


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_feedback_uses_actual_net_main_state_and_preserves_every_formal_day(cost, delay):
    data, nominal = synthetic_market()
    result, issued, decisions = run(data, nominal, cost, delay)
    proof = adapted.verify_feedback(
        data, nominal, result, issued, decisions, "2020-01-31", "2020-04-30", delay
    )
    assert proof["checked_decision_states"] == len(result.frame) + 1
    assert proof["source_shadow_wealth_or_peak_imported"] is False
    assert proof["main_drawdown_guarantee"] is False
    assert result.frame.index.equals(sessions("2020-01-31", "2020-04-30"))
    assert decisions[0]["known_nav_usd"] == decisions[0]["known_peak_usd"] == 10000
    assert decisions[1]["known_nav_usd"] == result.frame.iloc[0]["equity"]
    assert (result.frame["cash"] >= 0).all()
    assert (result.frame["gross_exposure"] <= 1).all()
    if delay == 2:
        assert result.frame.iloc[0]["equity"] == 10000
        assert result.frame.iloc[0]["cost"] == result.frame.iloc[0]["orders"] == 0


def test_monthly_update_during_pending_request_is_retained_not_overwritten():
    data, nominal = synthetic_market()
    nominal.loc["2020-01-31"] = 0.0
    nominal.loc["2020-01-31", "BIL"] = 0.98
    _, issued, decisions = run(data, nominal, 20.0, 2)
    by_day = {row["signal_session"]: row for row in decisions}
    assert by_day["2020-01-30"]["execution_session"] == "2020-02-03"
    assert by_day["2020-01-31"]["reason"] == "pending_request"
    assert by_day["2020-01-31"]["source_nominal_session"] == "2020-01-31"
    assert issued.loc["2020-01-31"].isna().all()
    assert issued.loc["2020-01-30", "SOXX"] == pytest.approx(0.49)
    assert by_day["2020-02-03"]["request_issued"] is True
    assert by_day["2020-02-03"]["execution_session"] == "2020-02-05"
    assert issued.loc["2020-02-03", "BIL"] == pytest.approx(0.98)


def test_fresh_later_account_does_not_inherit_the_long_accounts_wealth_peak():
    data, nominal = synthetic_market()
    long, _, _ = run(data, nominal)
    later, _, rows = run(data, nominal, start="2020-03-02")
    assert long.frame.loc["2020-02-28", "equity"] != 10000
    assert rows[0]["signal_session"] == rows[0]["source_nominal_session"] == "2020-02-28"
    assert rows[0]["known_nav_usd"] == rows[0]["known_peak_usd"] == 10000
    assert later.frame.index[0] == pd.Timestamp("2020-03-02")


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_prefix_and_future_perturbations_cannot_change_known_feedback(cost, delay):
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
    close, opening = data.close.copy(), data.open.copy()
    close.loc[close.index > end] *= 1.15
    opening.loc[opening.index > end] *= 1.15
    changed = MarketData(opening, close, close.copy(), data.volume, data.risk_free)
    after, after_issued, after_rows = run(changed, nominal, cost, delay)
    pd.testing.assert_frame_equal(full.frame.loc[:end], after.frame.loc[:end], check_exact=True)
    pd.testing.assert_frame_equal(full_issued.loc[:end], after_issued.loc[:end], check_exact=True)
    assert [r for r in after_rows if r["signal_session"] <= str(end.date())] == truncated_rows


def test_delayed_gap_can_breach_the_floor_without_inventing_a_guarantee():
    data, nominal = synthetic_market()
    close = pd.DataFrame(100.0, index=data.close.index, columns=SYMBOLS)
    close.loc["2020-02-03":, "SOXX"] = 70.0
    opening = close.copy()
    nominal.loc[nominal.notna().all(axis=1)] = 0.0
    nominal.loc[nominal.notna().all(axis=1), "SOXX"] = 0.98
    shocked = MarketData(opening, close, close.copy(), data.volume, data.risk_free)
    result, issued, rows = run(shocked, nominal)
    dd = 1 - result.frame["equity"] / result.frame["equity"].cummax().clip(lower=10000)
    assert dd.max() > 0.15
    breach = next(r for r in rows if r["signal_session"] == "2020-02-03")
    assert breach["scale"] == 0
    assert issued.loc["2020-02-03", "BIL"] == pytest.approx(0.98)


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_funded_replay_and_independent_bt_match_the_actual_closed_loop(cost, delay):
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
    bt = independent_equity(
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
        result.frame, bt, data.risk_free.loc[result.frame.index], 10000
    )
    assert difference < 1e-8


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_unexecuted_final_request_is_retained_without_terminal_liquidation(cost, delay):
    data, nominal = synthetic_market()
    prices = pd.DataFrame(100.0, index=data.close.index, columns=SYMBOLS)
    fixed = MarketData(prices, prices.copy(), prices.copy(), data.volume, data.risk_free)
    nominal.loc[nominal.notna().all(axis=1)] = 0.0
    nominal.loc[nominal.notna().all(axis=1), "SOXX"] = 0.98
    nominal.loc["2020-02-28"] = 0.0
    nominal.loc["2020-02-28", "GLD"] = 0.98
    result, issued, rows = run(fixed, nominal, cost, delay, end="2020-02-28")
    assert rows[-1]["request_issued"] is True
    assert rows[-1]["execution_session"] > "2020-02-28"
    assert issued.loc["2020-02-28", "GLD"] > 0.9
    assert result.frame.iloc[-1]["orders"] == result.frame.iloc[-1]["cost"] == 0
    assert result.weights.iloc[-1]["SOXX"] > 0.9
    bt = independent_equity(
        fixed,
        issued,
        "2020-01-31",
        "2020-02-28",
        capital=10000,
        cost_bps=cost,
        commission=1,
        delay=delay,
    )
    assert abs(result.frame["equity"] - bt["equity"]).max() < 1e-8


@pytest.mark.parametrize("damage", ["future_peak", "false_clock", "unissued_target"])
def test_independent_verifier_rejects_rewritten_actual_state(damage):
    data, nominal = synthetic_market()
    result, issued, original_rows = run(data, nominal, 20.0, 2)
    rows = copy.deepcopy(original_rows)
    issued = issued.copy()
    if damage == "future_peak":
        rows[1]["known_peak_usd"] = 99999
    elif damage == "false_clock":
        rows[0]["execution_session"] = "2020-01-31"
    else:
        issued.loc["2020-01-31"] = issued.loc["2020-01-30"]
    with pytest.raises(QuantError, match="feedback|pending request"):
        adapted.verify_feedback(data, nominal, result, issued, rows, "2020-01-31", "2020-04-30", 2)


@pytest.mark.parametrize("damage", ["future_initial", "nonmonthly", "partial", "negative"])
def test_invalid_or_future_nominal_information_does_not_become_a_successful_default(damage):
    data, nominal = synthetic_market()
    if damage == "future_initial":
        nominal.loc[:"2020-01-30"] = np.nan
    elif damage == "nonmonthly":
        nominal.loc["2020-02-04"] = nominal.loc["2020-01-30"]
    elif damage == "partial":
        nominal.loc["2020-01-30", "SOXX"] = np.nan
    else:
        nominal.loc["2020-01-30", "SOXX"] = -0.1
    with pytest.raises(QuantError, match="nominal|monthly|missingness"):
        run(data, nominal)


def test_fixed_policy_cannot_claim_a_floor_guarantee_or_new_execution_scenario():
    policy = read_json(adapted.POLICY)
    policy["methodology"]["main_drawdown_guarantee"] = True
    with pytest.raises(QuantError, match="promised"):
        adapted.validate(policy)
    data, nominal = synthetic_market()
    with pytest.raises(QuantError, match="execution scenarios"):
        run(data, nominal, 5.0, 2)
