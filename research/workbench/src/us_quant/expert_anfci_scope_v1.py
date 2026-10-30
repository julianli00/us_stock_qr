from __future__ import annotations

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end, previous_session, sessions
from us_quant.config import QuantError
from us_quant.daily_variance_disagreement_v1 import UNIVERSE

CANDIDATE = "two_segment_anfci_residual_confirmation_v1"


def assess_saved(policy: dict, source: dict, decisions: dict) -> dict:
    fixed = {
        "study_id": "ssd_anfci_residual_confirmation_20261028_v1",
        "maximum_observation_age_calendar_days": 14,
        "vintage_cutoff": "previous_NYSE_session",
        "required_anfci_information": "current_own_vintage_level_only",
        "require_fresh_anfci_before_every_monthly_decision": True,
        "stale_input_behavior": "explicit_no_new_request_keep_actual_positions",
        "source_monthly_opportunities": 131,
        "expected_fresh_decisions": 130,
        "expected_source_abstentions": 1,
        "source_full_225_query_admission_passed": False,
    }
    if not isinstance(policy, dict) or any(
        policy.get(key) != value or type(policy.get(key)) is not type(value)
        for key, value in fixed.items()
    ):
        raise QuantError("Preserve the original current-level scope,age limit and abstention law.")
    if not isinstance(source, dict) or set(source) != {"readiness", "manifest", "validation"}:
        raise QuantError("Require all three original source-admission documents.")
    if any(
        not isinstance(value, dict) or value.get("source_admitted") is not False
        for value in source.values()
    ):
        raise QuantError("The original full-source nonadmission cannot be overridden.")
    readiness, manifest, validation = (
        source[key] for key in ("readiness", "manifest", "validation")
    )
    if (
        readiness.get("actual_vintage_count") != 225
        or readiness.get("fresh_query_count") != 223
        or readiness.get("stale_query_count") != 2
        or readiness.get("freshness_limit_calendar_days") != 14
        or manifest.get("validation") != policy["anfci_validation"]
        or manifest.get("batches") != validation.get("batches")
        or not isinstance(validation.get("query_readings"), list)
        or len(validation["query_readings"]) != 225
    ):
        raise QuantError("Preserve every original source query and its failed full-scope gate.")
    readings = {}
    for row in validation["query_readings"]:
        if not isinstance(row, dict):
            raise QuantError("A saved source query must retain its dated record.")
        day = pd.Timestamp(row["query_session"])
        cutoff, observed = (
            pd.Timestamp(row["vintage_cutoff"]),
            pd.Timestamp(row["observation_date"]),
        )
        age = (day - observed).days
        if (
            pd.isna(day)
            or pd.isna(cutoff)
            or pd.isna(observed)
            or day in readings
            or cutoff != previous_session(day)
            or observed > cutoff
            or observed.weekday() != 4
            or type(row["age_calendar_days"]) is not int
            or row["age_calendar_days"] != age
            or age < 0
            or type(row["fresh"]) is not bool
            or row["fresh"] != (age <= 14)
        ):
            raise QuantError("Original source cutoff,weekly date,age or eligibility changed.")
        readings[day] = row
    if sum(not row["fresh"] for row in readings.values()) != 2:
        raise QuantError("Do not remove either stale source query.")
    if not isinstance(decisions, dict) or set(decisions) != {"targets", "proof", "audits"}:
        raise QuantError("Require saved native targets,proof and all original opportunity audits.")
    target, proof, audits = (decisions[key] for key in ("targets", "proof", "audits"))
    if (
        not isinstance(target, pd.DataFrame)
        or target.empty
        or not isinstance(target.index, pd.DatetimeIndex)
        or not target.index.is_unique
        or not target.index.is_monotonic_increasing
        or target.index.tz is not None
        or not target.index.equals(sessions(target.index[0], target.index[-1]))
        or not target.columns.is_unique
        or not set(target.columns).issubset((*UNIVERSE, "IJR", "PKW"))
        or (target.isna().any(axis=1) & ~target.isna().all(axis=1)).any()
        or not isinstance(proof, dict)
        or not isinstance(audits, list)
        or len(audits) != 131
    ):
        raise QuantError("Preserve the native source calendar and complete coordinate support.")
    active = target.dropna(how="all")
    if (
        active.empty
        or not np.isfinite(active.to_numpy()).all()
        or (active < 0).any().any()
        or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
    ):
        raise QuantError("A native target cannot be clipped,zero-filled or silently normalized.")
    observed_dates, fresh_dates, abstentions = [], [], []
    for audit in audits:
        day = pd.Timestamp(audit["decision_session"])
        if (
            day not in readings
            or day not in target.index
            or day in observed_dates
            or not is_month_end(day)
        ):
            raise QuantError("Every saved source opportunity needs its original dated query.")
        query = readings[day]
        fresh = query["fresh"]
        if (
            audit.get("fresh_required_reading") is not fresh
            or audit.get("candidate_target_created") is not fresh
            or audit.get("vintage_cutoff") != query["vintage_cutoff"]
            or audit.get("observation_date") != query["observation_date"]
            or audit.get("age_calendar_days") != query["age_calendar_days"]
            or audit.get("stale_numeric_reading_used") is not False
        ):
            raise QuantError("Native eligibility must match the saved query without retiming.")
        if fresh:
            if target.loc[day].isna().any() or audit.get("source_abstention_reason") is not None:
                raise QuantError("A fresh original opportunity lost its native recommendation.")
            fresh_dates.append(day)
        else:
            if (
                not target.loc[day].isna().all()
                or audit.get("accepted_ANFCI_value") is not None
                or audit.get("residual_stress_confirmed") is not None
                or audit.get("source_abstention_reason") != "anfci_exceeds_frozen_14day_age"
            ):
                raise QuantError(
                    "Preserve explicit no-new-request;never carry a target into the gap."
                )
            abstentions.append(
                {
                    "decision_session": str(day.date()),
                    "observation_date": query["observation_date"],
                    "age_calendar_days": query["age_calendar_days"],
                    "meaning": "no_new_request_keep_actual_positions",
                }
            )
        observed_dates.append(day)
    dates, fresh = pd.DatetimeIndex(observed_dates), pd.DatetimeIndex(fresh_dates)
    gap_dates = [row["decision_session"] for row in abstentions]
    if (
        not dates.is_monotonic_increasing
        or len(fresh) != 130
        or len(abstentions) != 1
        or not active.index.equals(fresh)
        or proof.get("original_monthly_opportunities") != 131
        or proof.get("fresh_target_decisions") != 130
        or proof.get("source_abstention_count") != 1
        or proof.get("source_abstention_dates") != gap_dates
        or any(
            proof.get(key) is not False
            for key in (
                "source_full_225_query_admission_passed",
                "source_age_limit_relaxed",
                "stale_numeric_readings_used",
                "formal_returns_removed",
            )
        )
    ):
        raise QuantError(
            "The original proof,all opportunities and actual abstention must reconcile."
        )
    expected = pd.DatetimeIndex(
        [day for day in target.loc[active.index[0] :].index if is_month_end(day)]
    )
    if not dates.equals(expected):
        raise QuantError("The native opportunity calendar cannot omit a genuine month end.")
    missing = expected.difference(active.index)
    if missing.strftime("%Y-%m-%d").tolist() != gap_dates:
        raise QuantError("An unexplained source gap cannot become a neutral eligibility result.")
    return {
        "candidate_id": CANDIDATE,
        "status": "not_admitted_to_current_six_expert_pipeline",
        "source_admitted_flags_preserved": {
            key: value["source_admitted"] for key, value in source.items()
        },
        "original_full_scope_queries": 225,
        "original_fresh_queries": 223,
        "original_stale_queries": 2,
        "native_current_level_only_scope_is_blanket_source_approval": False,
        "native_monthly_opportunities": len(dates),
        "native_fresh_target_rows": len(active),
        "native_source_abstentions": abstentions,
        "first_native_target": str(active.index[0].date()),
        "last_native_target": str(active.index[-1].date()),
        "current_dense_monthly_adapter_requires_all_post_warmup_month_ends": True,
        "current_dense_monthly_adapter_supports_native_target_calendar": False,
        "current_dense_monthly_adapter_missing_dates": gap_dates,
        "reason_codes": [
            "full_source_nonadmission_remains_without_new_expert_specific_scope_approval",
            "native_abstention_not_representable_by_existing_dense_after_warmup_adapter",
        ],
        "no_new_request_is_rebalance_to_carried_target": False,
        "target_forward_filled_zero_filled_or_retimed": False,
        "new_availability_adapter_or_assessment_view_created": False,
        "original_standalone_verdict_reclassified_as_technical_invalidity": False,
        "new_expert_admitted": False,
        "current_expert_pool_changed": False,
        "new_financial_trials": 0,
        "new_source_requests": 0,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
    }
