from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import expert_anfci_scope_v1 as scope
from us_quant.calendar import is_month_end, previous_session, sessions
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import ArchivedMonthlyExpert, ExpertIdentity


def synthetic():
    policy = read_json(
        Path(scope.__file__).resolve().parents[2] / "config/anfci-residual-confirmation-v1.json"
    )
    calendar = sessions("2015-11-02", "2026-10-02")
    dates = pd.DatetimeIndex([day for day in calendar if is_month_end(day)])
    assert len(dates) == 131
    auxiliary = [day for day in calendar if day not in dates][:94]
    query_dates = sorted([*dates, *auxiliary])
    stale = {pd.Timestamp("2022-12-30"), auxiliary[0]}
    readings = []
    for day in query_dates:
        cutoff = previous_session(day)
        observed = cutoff - pd.Timedelta(days=(cutoff.weekday() - 4) % 7)
        if day in stale:
            observed -= pd.Timedelta(days=21)
        age = (day - observed).days
        readings.append(
            {
                "query_session": str(day.date()),
                "vintage_cutoff": str(cutoff.date()),
                "observation_date": str(observed.date()),
                "age_calendar_days": age,
                "fresh": day not in stale,
                "actual_latest_value": 0.123,
            }
        )
    lookup = {row["query_session"]: row for row in readings}
    target = pd.DataFrame(
        np.nan, index=calendar, columns=["SPY", "BIL", "MTUM", "VLUE", "QUAL", "USMV"]
    )
    audits = []
    for day in dates:
        row = lookup[str(day.date())]
        if row["fresh"]:
            target.loc[day] = [0, 0, 0.245, 0.245, 0.245, 0.245]
        audits.append(
            {
                "decision_session": str(day.date()),
                "vintage_cutoff": row["vintage_cutoff"],
                "observation_date": row["observation_date"],
                "age_calendar_days": row["age_calendar_days"],
                "fresh_required_reading": row["fresh"],
                "candidate_target_created": row["fresh"],
                "stale_numeric_reading_used": False,
                "source_abstention_reason": None
                if row["fresh"]
                else "anfci_exceeds_frozen_14day_age",
                "accepted_ANFCI_value": 0.123 if row["fresh"] else None,
                "residual_stress_confirmed": False if row["fresh"] else None,
            }
        )
    source = {
        "readiness": {
            "source_admitted": False,
            "actual_vintage_count": 225,
            "fresh_query_count": 223,
            "stale_query_count": 2,
            "freshness_limit_calendar_days": 14,
        },
        "manifest": {
            "source_admitted": False,
            "validation": policy["anfci_validation"],
            "batches": [],
        },
        "validation": {"source_admitted": False, "batches": [], "query_readings": readings},
    }
    proof = {
        "original_monthly_opportunities": 131,
        "fresh_target_decisions": 130,
        "source_abstention_count": 1,
        "source_abstention_dates": ["2022-12-30"],
        "source_full_225_query_admission_passed": False,
        "source_age_limit_relaxed": False,
        "stale_numeric_readings_used": False,
        "formal_returns_removed": False,
    }
    return policy, source, {"targets": target, "proof": proof, "audits": audits}


def test_original_scope_and_sparse_abstention_are_preserved():
    result = scope.assess_saved(*synthetic())
    assert result["status"] == "not_admitted_to_current_six_expert_pipeline"
    assert result["source_admitted_flags_preserved"] == dict.fromkeys(
        ("readiness", "manifest", "validation"), False
    )
    assert (
        result["native_monthly_opportunities"] == 131 and result["native_fresh_target_rows"] == 130
    )
    assert result["current_dense_monthly_adapter_missing_dates"] == ["2022-12-30"]
    assert result["no_new_request_is_rebalance_to_carried_target"] is False
    assert result["new_expert_admitted"] is False


@pytest.mark.parametrize(
    "kind", ["source_true", "missing_flag", "age_limit", "missing_query", "retimed_cutoff"]
)
def test_source_false_flags_and_age_limit_cannot_be_waived(kind):
    policy, source, decisions = synthetic()
    if kind == "source_true":
        source["manifest"]["source_admitted"] = True
    elif kind == "missing_flag":
        source["readiness"].pop("source_admitted")
    elif kind == "age_limit":
        policy["maximum_observation_age_calendar_days"] = 21
    elif kind == "missing_query":
        source["validation"]["query_readings"].pop()
    else:
        source["validation"]["query_readings"][0]["vintage_cutoff"] = "2015-11-02"
    with pytest.raises(QuantError):
        scope.assess_saved(policy, source, decisions)


@pytest.mark.parametrize(
    "kind", ["carried", "zero", "fake_fresh", "missing_audit", "changed_proof", "partial_row"]
)
def test_gap_cannot_be_carried_zero_filled_or_retimed(kind):
    policy, source, decisions = synthetic()
    target = decisions["targets"]
    if kind == "carried":
        target.loc["2022-12-30"] = target.ffill().loc["2022-12-30"]
    elif kind == "zero":
        target.loc["2022-12-30"] = 0
    elif kind == "fake_fresh":
        next(row for row in decisions["audits"] if row["decision_session"] == "2022-12-30")[
            "fresh_required_reading"
        ] = True
    elif kind == "missing_audit":
        decisions["audits"].pop()
    elif kind == "changed_proof":
        decisions["proof"]["source_abstention_dates"] = []
    else:
        target.loc["2022-12-30", "SPY"] = 0
    with pytest.raises(QuantError):
        scope.assess_saved(policy, source, decisions)


def test_current_original_monthly_adapter_rejects_the_same_gap():
    _, _, decisions = synthetic()
    target = decisions["targets"]
    shadow = pd.DataFrame(
        {
            "equity": 10000.0,
            "return": 0.0,
            "risk_free": 0.0,
            "cash": 200.0,
            "gross_exposure": 0.98,
            "cost": 0.0,
            "orders": 0,
        },
        index=target.index,
    )
    with pytest.raises(QuantError, match="every genuine month end"):
        ArchivedMonthlyExpert(
            ExpertIdentity("synthetic", "synthetic_financing", "2026-10-30T00:00:00+00:00"),
            target,
            shadow,
            (*scope.UNIVERSE, "IJR", "PKW"),
        )


def test_review_preserves_all_original_inputs():
    policy, source, decisions = synthetic()
    old_policy, old_source, old_decisions = deepcopy((policy, source, decisions))
    scope.assess_saved(policy, source, decisions)
    assert policy == old_policy and source == old_source
    pd.testing.assert_frame_equal(decisions["targets"], old_decisions["targets"])
    assert (
        decisions["proof"] == old_decisions["proof"]
        and decisions["audits"] == old_decisions["audits"]
    )
