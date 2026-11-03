from __future__ import annotations

import pandas as pd
import pytest

from us_quant import consensus_episode_diagnosis_v1 as diag
from us_quant.calendar import sessions
from us_quant.config import QuantError


def curve(values):
    dates = sessions("2020-01-02", "2020-01-15")[: len(values)]
    return pd.Series(values, index=dates, dtype=float)


def test_initial_capital_peak_precedes_first_loss_without_fabricating_a_close():
    result = diag.episode(curve([9000, 8000, 8500]))
    assert result["peak_session"] == "2019-12-31"
    assert result["trough_session"] == "2020-01-03"
    assert result["maximum_drawdown"] == pytest.approx(0.2)
    assert result["peak_is_initial_capital_anchor"] is True
    assert result["recovery_session"] is None


def test_ties_keep_first_peak_first_worst_trough_and_first_recovery():
    result = diag.episode(curve([11000, 11000, 8800, 8800, 11000, 12000]))
    assert result["peak_session"] == "2020-01-02"
    assert result["trough_session"] == "2020-01-06"
    assert result["recovery_session"] == "2020-01-08"
    assert result["maximum_drawdown"] == pytest.approx(0.2)


def test_future_peak_never_becomes_a_past_episode_peak():
    result = diag.episode(curve([11000, 8800, 15000, 14500]))
    assert result["peak_session"] == "2020-01-02"
    assert result["trough_session"] == "2020-01-03"


def test_no_loss_is_explicit_and_never_uses_an_invalid_input_fallback():
    result = diag.episode(curve([10000, 11000, 12000]))
    assert result["status"] == "no_positive_drawdown"
    assert result["peak_session"] is result["trough_session"] is None
    with pytest.raises(QuantError):
        diag.episode(curve([]))


def test_shared_date_change_keeps_the_control_own_prior_peak_without_rebasing():
    equity = curve([20000, 15000, 18000, 14000])
    result = diag.shared_window(equity, "2020-01-03", "2020-01-06")
    assert result["stored_close_to_close_change"] == pytest.approx(0.2)
    assert result["own_retained_peak_drawdown_at_end"] == pytest.approx(0.1)
    assert result["new_account_or_rebased_equity_created"] is False


def test_different_own_episodes_are_not_relabelled_as_one_market_event():
    curves = {
        "new": curve([11000, 8800, 12000, 11000]),
        "primary": curve([10000, 11000, 12000, 8400]),
        "secondary_context": curve([10000, 10000, 9500, 10000]),
    }
    saved = {role: diag.episode(nav)["maximum_drawdown"] for role, nav in curves.items()}
    result = diag.compare(curves, saved)
    assert result["new_and_primary_own_episode_clocks_identical"] is False
    assert result["shared_window_change_is_a_replacement_maximum_drawdown"] is False
    assert result["causal_fee_or_unique_loss_cause_identified"] is False
    assert result["fee_free_or_other_counterfactual_account_computed"] is False


@pytest.mark.parametrize("kind", ["nan", "negative", "duplicate", "gap", "reversed"])
def test_bad_stored_NAV_or_calendar_is_rejected_without_filling(kind):
    nav = curve([10000, 11000, 9000, 10500])
    if kind == "nan":
        nav.iloc[1] = float("nan")
    elif kind == "negative":
        nav.iloc[1] = -1
    elif kind == "duplicate":
        nav.index = pd.DatetimeIndex([nav.index[0], nav.index[0], *nav.index[2:]])
    elif kind == "gap":
        nav = nav.drop(nav.index[1])
    else:
        nav = nav.iloc[::-1]
    with pytest.raises(QuantError):
        diag.episode(nav)


def test_missing_controls_or_changed_saved_metric_stops_the_fixed_scheme():
    data = {role: curve([10000, 9000, 9500]) for role in diag.ROLES}
    saved = {role: 0.1 for role in diag.ROLES}
    saved["primary"] = 0.05
    with pytest.raises(QuantError, match="disagrees"):
        diag.compare(data, saved)
    with pytest.raises(QuantError, match="both complete"):
        diag.compare({"new": data["new"]}, {"new": 0.1})
