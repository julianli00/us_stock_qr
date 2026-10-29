from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from us_quant.calendar import next_session, previous_session, sessions
from us_quant.config import QuantError
from us_quant.legacy_feedback_expert_v1 import feedback_observations
from us_quant.strategy_experts_v1 import ArchivedDenseExpert, ExpertIdentity


def saved_history(delay=1, navs=None):
    calendar = sessions("2019-01-31", "2019-03-06")
    dates = sessions("2019-03-01", "2019-03-06")
    anchor = previous_session(dates[0])
    columns = ["SOXX", "GLD", "BIL"]
    monthly = pd.DataFrame(np.nan, index=calendar, columns=columns)
    monthly.loc["2019-01-31"] = [0.98, 0, 0]
    monthly.loc["2019-02-28"] = [0.98, 0, 0]
    equity = pd.Series(navs or [10000.0] * len(dates), index=dates)
    prior = equity.shift(1)
    prior.iloc[0] = 10000
    shadow = pd.DataFrame(
        {
            "equity": equity,
            "return": equity / prior - 1,
            "risk_free": 0.0,
            "cash": 200.0,
            "gross_exposure": 0.98,
            "cost": 0.0,
            "orders": 0,
        }
    )
    weights = pd.DataFrame(np.tile([0.98, 0, 0], (len(dates), 1)), index=dates, columns=columns)
    issued = pd.DataFrame(np.nan, index=calendar, columns=columns)
    issued.loc[anchor] = [0.98, 0, 0]
    pending = next_session(anchor) if delay == 1 else next_session(next_session(anchor))
    states = []
    for day in dates:
        states.append(
            {
                "session": str(day.date()),
                "nav_usd": 10000.0,
                "peak_usd": 10000.0,
                "floor_usd": 8500.0,
                "scale": 1.0,
                "source_bil_weight": 0.0,
                "desired_bil_weight": 0.0,
                "monthly_requested": False,
                "pending_executions": [str(pending.date())] if day < pending else [],
            }
        )
    return monthly, shadow, weights, issued, states, delay


@pytest.mark.parametrize("delay", [1, 2])
def test_real_initial_anchor_and_every_daily_observation_are_kept(delay):
    inputs = saved_history(delay)
    result, audit = feedback_observations(*inputs)
    anchor = previous_session(inputs[1].index[0])
    assert result.loc[: previous_session(anchor)].isna().all().all()
    assert result.dropna().index.equals(inputs[0].loc[anchor:].index)
    assert audit["first_known_recommendation"] == "2019-02-28"
    assert audit["archived_requests_checked"] == 1
    assert audit["shadow_accounts_replayed"] is False
    assert audit["main_drawdown_guarantee"] is False


def test_existing_dense_adapter_separates_recommendation_availability_and_score_maturity():
    inputs = saved_history()
    dense, _ = feedback_observations(*inputs)
    expert = ArchivedDenseExpert(
        ExpertIdentity("saved_feedback", "wealth_feedback", "2026-10-29T17:00:00Z"),
        dense,
        inputs[1],
        ("SOXX", "GLD", "BIL", "IJR", "PKW"),
    )
    initial = expert.observe(pd.Timestamp("2019-02-28"), 126)
    assert initial.target_known_at == pd.Timestamp("2019-02-28")
    assert initial.target["SOXX"] == 0.98
    assert initial.target["IJR"] == initial.target["PKW"] == 0
    assert len(initial.net_returns) == 0 and not initial.score_mature
    assert expert.observe(pd.Timestamp("2019-02-27"), 126).target is None


@pytest.mark.parametrize("delay", [1, 2])
def test_prefix_causality_without_regenerating_a_shadow_account(delay):
    inputs = saved_history(delay)
    full, _ = feedback_observations(*inputs)
    for count in (1, 2, 3):
        end = inputs[1].index[count - 1]
        prefix, _ = feedback_observations(
            *(frame.loc[:end] for frame in inputs[:4]), inputs[4][:count], delay
        )
        pd.testing.assert_frame_equal(prefix, full.loc[:end])


def test_unissued_desired_recommendation_does_not_become_a_native_trade():
    monthly, shadow, weights, issued, states, delay = saved_history()
    shadow.loc[:, "equity"] = 9990.0
    shadow.loc[:, "return"] = [9990 / 10000 - 1, 0, 0, 0]
    scale = (9990 - 8500) / (0.15 * 9990)
    for row in states:
        row.update(nav_usd=9990.0, scale=scale, desired_bil_weight=0.98 * (1 - scale))
    dense, audit = feedback_observations(monthly, shadow, weights, issued, states, delay)
    assert dense.loc[shadow.index[0], "BIL"] > 0
    assert issued.loc[shadow.index[0]].isna().all()
    assert audit["archived_requests_checked"] == 1


@pytest.mark.parametrize(
    "field", ["nav_usd", "peak_usd", "floor_usd", "scale", "desired_bil_weight"]
)
def test_changed_saved_feedback_state_is_not_an_admissible_prefix(field):
    inputs = list(saved_history())
    inputs[4] = copy.deepcopy(inputs[4])
    inputs[4][0][field] += 1
    with pytest.raises(QuantError, match="prefix"):
        feedback_observations(*inputs)


def test_missing_initial_intent_cannot_be_a_safe_default():
    inputs = list(saved_history())
    inputs[3].loc[:, :] = np.nan
    with pytest.raises(QuantError, match="complete funded"):
        feedback_observations(*inputs)


def test_pending_request_cannot_be_overwritten():
    inputs = list(saved_history(2))
    inputs[3].loc[inputs[1].index[0]] = [0, 0.98, 0]
    with pytest.raises(QuantError, match="hold or pending"):
        feedback_observations(*inputs)


def test_month_end_intent_is_known_while_the_old_native_request_is_pending():
    calendar = sessions("2018-12-31", "2019-02-05")
    dates = sessions("2019-01-31", "2019-02-05")
    columns = ["SOXX", "GLD", "BIL"]
    monthly = pd.DataFrame(np.nan, index=calendar, columns=columns)
    monthly.loc["2018-12-31"] = [0.98, 0, 0]
    monthly.loc["2019-01-31"] = [0, 0.98, 0]
    shadow = pd.DataFrame({"equity": 10000.0}, index=dates)
    weights = pd.DataFrame(
        [[0, 0, 0], [0.98, 0, 0], [0.98, 0, 0], [0, 0.98, 0]],
        index=dates,
        columns=columns,
    )
    issued = pd.DataFrame(np.nan, index=calendar, columns=columns)
    issued.loc["2019-01-30"] = [0.98, 0, 0]
    issued.loc["2019-02-01"] = [0, 0.98, 0]
    states = [
        {
            "session": str(day.date()),
            "nav_usd": 10000.0,
            "peak_usd": 10000.0,
            "floor_usd": 8500.0,
            "scale": 1.0,
            "source_bil_weight": 0.0,
            "desired_bil_weight": 0.0,
            "monthly_requested": i == 0,
            "pending_executions": [pending] if pending else [],
        }
        for i, (day, pending) in enumerate(
            zip(dates, ["2019-02-01", "2019-02-05", "2019-02-05", None], strict=True)
        )
    ]
    dense, audit = feedback_observations(monthly, shadow, weights, issued, states, 2)
    assert dense.loc["2019-01-31", "GLD"] == 0.98
    assert issued.loc["2019-01-31"].isna().all()
    assert audit["archived_requests_checked"] == 2


def test_missing_native_session_is_not_silently_carried():
    inputs = list(saved_history())
    inputs[1] = inputs[1].drop(inputs[1].index[1])
    inputs[2] = inputs[2].loc[inputs[1].index]
    inputs[4] = [r for r in inputs[4] if r["session"] in inputs[1].index.strftime("%Y-%m-%d")]
    with pytest.raises(QuantError, match="aligned"):
        feedback_observations(*inputs)
