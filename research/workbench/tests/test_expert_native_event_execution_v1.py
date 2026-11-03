import copy

import numpy as np
import pandas as pd
import pytest
from test_expert_initial_monthly_execution_v1 import synthetic_market

from us_quant import expert_initial_monthly_execution_v1 as old
from us_quant import expert_native_event_execution_v1 as native
from us_quant import observed_weight_execution as monthly
from us_quant.bt_audit import independent_equity
from us_quant.config import QuantError
from us_quant.storage import read_json


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_old_monthly_behavior_is_exactly_preserved_on_monthly_lattice(cost, delay):
    data, target = synthetic_market()
    before, before_issued, before_decisions = old.run(
        data,
        target,
        monthly.CANDIDATES[0],
        read_json(monthly.POLICY),
        "2016-10-06",
        "2016-12-29",
        cost,
        delay,
    )
    after, issued, decisions = native.run(
        data, target, read_json(native.POLICY), "2016-10-06", "2016-12-29", cost, delay
    )
    pd.testing.assert_frame_equal(before.frame, after.frame, check_exact=True)
    pd.testing.assert_frame_equal(before.weights, after.weights, check_exact=True)
    pd.testing.assert_frame_equal(before_issued, issued, check_exact=True)
    assert [r["signal_session"] for r in decisions] == [
        r["signal_session"] for r in before_decisions
    ]
    assert [r["reason"] for r in decisions] == [r["reason"] for r in before_decisions]
    proof = native.verify_feedback(
        data,
        target,
        after,
        issued,
        decisions,
        read_json(native.POLICY),
        "2016-10-06",
        "2016-12-29",
        cost,
        delay,
    )
    assert proof["complete_frame_and_positions_reconciled_to_cash_engine"]


def daily_targets():
    data, _ = synthetic_market()
    target = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    for i, day in enumerate(data.close.loc["2016-10-05":].index):
        target.loc[day] = 0.0
        target.loc[day, ("SOXX", "BIL", "QUAL")[i % 3]] = 0.98
    return data, target


@pytest.mark.parametrize("cost,delay,fill", [(5.0, 1, "2016-10-06"), (20.0, 2, "2016-10-07")])
def test_native_event_account_keeps_full_clock_and_matches_independent_bt(cost, delay, fill):
    data, target = daily_targets()
    policy = read_json(native.POLICY)
    result, issued, decisions = native.run(
        data, target, policy, "2016-10-06", "2016-10-20", cost, delay
    )
    assert decisions[0]["signal_session"] == "2016-10-05"
    assert decisions[0]["execution_session"] == fill
    assert str(result.frame.index[result.frame["orders"] > 0][0].date()) == fill
    assert (result.frame["cash"] >= 0).all() and (result.frame["gross_exposure"] <= 1).all()
    assert result.frame["risk_free"].equals(data.risk_free.loc[result.frame.index])
    proof = native.verify_feedback(
        data, target, result, issued, decisions, policy, "2016-10-06", "2016-10-20", cost, delay
    )
    independent = independent_equity(
        data,
        issued,
        "2016-10-06",
        "2016-10-20",
        capital=10000,
        cost_bps=cost,
        commission=1,
        delay=delay,
    )
    assert abs(independent["equity"] - result.frame["equity"]).max() < 1e-6
    assert proof["maximum_observed_gap_error"] <= 1e-12
    if delay == 2:
        assert result.frame.iloc[0]["equity"] == 10000
        assert decisions[1]["reason"] == "pending_request_preserved"
        assert decisions[1]["execution_session"] is None
        assert issued.loc["2016-10-06"].isna().all()


def test_repeated_fresh_opinion_does_not_automatically_rebalance():
    data, target = daily_targets()
    target.loc["2016-10-05":] = 0.0
    target.loc["2016-10-05":, "BIL"] = 0.98
    result, issued, decisions = native.run(
        data, target, read_json(native.POLICY), "2016-10-06", "2016-10-20", 5.0, 1
    )
    assert len(issued.dropna(how="all")) == 1
    assert all(row["reason"] == "hold" for row in decisions[1:])
    assert (result.frame.iloc[1:]["orders"] == 0).all()


def test_pending_request_cannot_be_cancelled_or_replaced():
    data, target = daily_targets()
    policy = read_json(native.POLICY)
    result, issued, decisions = native.run(
        data, target, policy, "2016-10-06", "2016-10-20", 20.0, 2
    )
    changed = copy.deepcopy(decisions)
    changed[1]["reason"] = "observed_weight_gap"
    changed[1]["execution_session"] = "2016-10-10"
    with pytest.raises(QuantError, match="preserved-queue"):
        native.verify_feedback(
            data, target, result, issued, changed, policy, "2016-10-06", "2016-10-20", 20.0, 2
        )


def test_held_position_tampering_is_not_hidden_by_unchanged_total_equity():
    data, target = daily_targets()
    policy = read_json(native.POLICY)
    result, issued, decisions = native.run(data, target, policy, "2016-10-06", "2016-10-20", 5.0, 1)
    result.weights.loc["2016-10-06", "SPY"] += 0.00001
    with pytest.raises(QuantError):
        native.verify_feedback(
            data, target, result, issued, decisions, policy, "2016-10-06", "2016-10-20", 5.0, 1
        )


def test_future_target_cannot_initialize_missing_capital_and_cost_delay_cannot_change():
    data, target = daily_targets()
    target.loc[:"2016-10-06"] = np.nan
    with pytest.raises(QuantError, match="future opinion"):
        native.run(data, target, read_json(native.POLICY), "2016-10-06", "2016-10-20", 5.0, 1)
    with pytest.raises(QuantError, match="original base/stress"):
        native.run(data, target, read_json(native.POLICY), "2016-10-06", "2016-10-20", 20.0, 1)


def test_no_terminal_liquidation_or_retroactive_stress_fill():
    data, target = daily_targets()
    result, issued, decisions = native.run(
        data, target, read_json(native.POLICY), "2016-10-06", "2016-10-07", 20.0, 2
    )
    assert len(result.frame) == 2
    assert result.frame.iloc[0]["cost"] == 0
    assert decisions[-1]["execution_session"] == "2016-10-11"
    assert pd.Timestamp(decisions[-1]["execution_session"]) > result.frame.index[-1]
    assert result.frame.iloc[-1]["gross_exposure"] > 0
