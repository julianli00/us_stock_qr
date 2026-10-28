from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_quant import anfci_residual_confirmation_v1 as rules
from us_quant.calendar import previous_session, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json


def fixture():
    dates = sessions("2022-08-01", "2023-02-10")
    close = pd.DataFrame(100.0, index=dates, columns=rules.UNIVERSE)
    desired = pd.DataFrame(np.nan, index=dates, columns=rules.UNIVERSE)
    opportunities = pd.to_datetime(
        ["2022-08-31", "2022-09-30", "2022-10-31", "2022-11-30", "2022-12-30", "2023-01-31"]
    )
    observations = [
        "2022-08-26",
        "2022-09-23",
        "2022-10-28",
        "2022-11-25",
        "2022-12-09",
        "2023-01-27",
    ]
    values = [0.2, 0.0, -0.2, 0.2, 10.0, -0.3]
    guarded = [True, True, True, False, False, False]
    records = []
    for day, observed, value, half in zip(
        opportunities, observations, values, guarded, strict=True
    ):
        budget = 0.49 if half else 0.98
        desired.loc[day] = 0.0
        desired.loc[day, list(rules.CORE)] = budget / 16
        desired.loc[day, ["SOXX", "GLD", "BIL"]] = [budget / 4, budget / 2, 0.98 - budget]
        age = (day - pd.Timestamp(observed)).days
        records.append(
            {
                "query_session": str(day.date()),
                "vintage_cutoff": str(previous_session(day).date()),
                "observation_date": observed,
                "age_calendar_days": age,
                "fresh": age <= 14,
                "actual_latest_value": value,
            }
        )
    data = MarketData(
        close.copy(),
        close,
        close.copy(),
        close * 1000,
        pd.Series(0.0, index=dates, name="risk_free"),
    )
    return data, desired, records


def test_positive_confirmation_equality_release_and_explicit_abstention():
    data, desired, records = fixture()
    result, audits = rules.build_targets(data, desired, records, read_json(rules.POLICY))
    target = result[rules.CANDIDATES[0]["id"]]
    for row in records:
        day = pd.Timestamp(row["query_session"])
        np.testing.assert_allclose(
            target.loc[day],
            rules.independent_row(desired.loc[day], row, day),
            rtol=0,
            atol=1e-12,
            equal_nan=True,
        )
    assert target.loc["2022-08-31", "BIL"] == 0.49
    assert target.loc["2022-09-30", "BIL"] == 0
    assert target.loc["2022-10-31", "BIL"] == 0
    assert target.loc["2022-11-30", "BIL"] == 0
    assert target.loc["2022-12-30"].isna().all()
    pause = [row for row in audits if not row["candidate_target_created"]]
    assert len(pause) == 1 and pause[0]["accepted_ANFCI_value"] is None
    assert pause[0]["original_dollar_guard_active"] is False
    assert pause[0]["source_abstention_reason"] == "anfci_exceeds_frozen_14day_age"
    assert (target.dropna()[list(rules.CORE)] > 0).all().all()


def test_future_records_do_not_change_prior_targets():
    data, desired, records = fixture()
    policy = read_json(rules.POLICY)
    full, _ = rules.build_targets(data, desired, records, policy)
    end = pd.Timestamp("2022-11-30")
    prefix = MarketData(
        data.open.loc[:end],
        data.close.loc[:end],
        data.raw_close.loc[:end],
        data.volume.loc[:end],
        data.risk_free.loc[:end],
    )
    early, _ = rules.build_targets(
        prefix,
        desired.loc[:end],
        [r for r in records if r["query_session"] <= str(end.date())],
        policy,
    )
    pd.testing.assert_frame_equal(
        early[rules.CANDIDATES[0]["id"]], full[rules.CANDIDATES[0]["id"]].loc[:end]
    )


@pytest.mark.parametrize(
    "kind", ["missing", "duplicate", "cutoff", "age", "fresh", "nonfinite", "budget"]
)
def test_invalid_input_never_becomes_standby(kind):
    data, desired, records = fixture()
    if kind == "missing":
        records.pop(0)
    elif kind == "duplicate":
        records.append(records[0].copy())
    elif kind == "cutoff":
        records[0]["vintage_cutoff"] = records[0]["query_session"]
    elif kind == "age":
        records[0]["age_calendar_days"] += 1
    elif kind == "fresh":
        records[4]["fresh"] = True
    elif kind == "nonfinite":
        records[4]["actual_latest_value"] = float("nan")
    else:
        desired.loc["2022-12-30", "BIL"] = 0.1
    with pytest.raises(QuantError):
        rules.build_targets(data, desired, records, read_json(rules.POLICY))


@pytest.mark.parametrize(
    "key,value",
    [
        ("anfci_threshold", 0.2),
        ("maximum_observation_age_calendar_days", 21),
        ("stale_input_behavior", "hold_cash"),
        ("require_fresh_anfci_before_every_monthly_decision", False),
        ("source_full_225_query_admission_passed", True),
        ("actual_weight_trade_band", 0.01),
    ],
)
def test_policy_does_not_rescue_data_or_change_gates(key, value):
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)
