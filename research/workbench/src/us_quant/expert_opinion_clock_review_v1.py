from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end
from us_quant.nested_growth_allocation_audit_v1 import require
from us_quant.protective_source_mass_audit_v1 import ASSETS, _hex
from us_quant.saved_inclusion_execution_v1 import _target

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/expert-opinion-clock-review-v1.json"


def validate(policy):
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_expert_opinion_clock_review_20261106_v1",
        "actual_native_event_seq": 193058,
        "prior_journal_sequence": 1478,
        "source_inputs": {
            "path": (
                "reports/ssd-continuous-research-v1/expert-opinion-clock-review-20261106-v1/"
                "source-inputs-v1.json"
            ),
            "sha256": "dd8dbaab062dc907900efc9d8151b33a0a860b03c8bee6b24f52bce051b5f1e7",
        },
        "initial_observation_session": "2016-10-05",
        "last_observation_session": "2026-10-05",
        "decision_sessions": 2513,
        "source_count": 6,
        "source_lookback_sessions": 126,
        "expected_observations": 15078,
        "target_equality": "exact_numeric_signed_zeros_equal_no_threshold",
        "observation_clock": "unchanged_pinned_source_information_clock",
        "existing_main_clock": "initial_seed_then_month_end_only",
        "known_opinion_rows_are_native_requests": False,
        "daily_availability_is_native_event_completeness": False,
        "source_assessment_RF_view_changed": False,
        "new_source_or_event_Adapter": False,
        "new_rankings_targets_accounts_or_financial_trials": 0,
        "global_pool_changed": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    require(
        policy == fixed and all(type(policy[k]) is type(v) for k, v in fixed.items()),
        "Freeze source observation scope without promoting observations to native events.",
    )


def inspect(row, previous):
    day, known = row["as_of"], row["target_known_at"]
    require(
        type(day) is str
        and type(known) is str
        and str(date.fromisoformat(day)) == day
        and str(date.fromisoformat(known)) == known
        and known <= day,
        "A source recommendation must have an explicit past-known clock.",
    )
    require(
        tuple(row["asset_order"]) == tuple(ASSETS)
        and row["assessment_RF_matches_original_view"] is True
        and row["source_window_exact_calendar"] is True
        and type(row["source_return_count"]) is int
        and 0 <= row["source_return_count"] <= 126
        and type(row["source_score_mature"]) is bool
        and row["source_score_mature"] is (row["source_return_count"] == 126),
        "Keep original source coordinates,assessment view and explicit source maturity.",
    )
    target = _target(_hex(row["target_hex"], len(ASSETS)), len(ASSETS))
    changed = advanced = False
    initial = previous is None
    if not initial:
        require(
            row["expert_id"] == previous["expert_id"]
            and row["mechanism_group"] == previous["mechanism_group"]
            and row["introduced_at"] == previous["introduced_at"]
            and day > previous["as_of"]
            and known >= previous["target_known_at"],
            "Source identity or known-at time cannot move backward or reset.",
        )
        old = _target(_hex(previous["target_hex"], len(ASSETS)), len(ASSETS))
        changed = not np.array_equal(target, old)
        advanced = known > previous["target_known_at"]
        require(
            advanced or not changed,
            "An unchanged known timestamp cannot silently acquire another numeric opinion.",
        )
    same_day_new = advanced and known == day
    delayed_new = advanced and known < day
    month_end = bool(is_month_end(pd.Timestamp(day)))
    return {
        "expert_id": row["expert_id"],
        "as_of": day,
        "initial_observation": initial,
        "initial_observes_prior_dated_opinion": initial and known < day,
        "carried_known_timestamp": not initial and not advanced,
        "new_same_day_known_row": same_day_new,
        "newly_observed_prior_dated_row": delayed_new,
        "numeric_target_changed": changed,
        "new_same_day_row_same_numeric_target": same_day_new and not changed,
        "off_month_end_new_same_day_row": same_day_new and not month_end,
        "off_month_end_changed_opinion": same_day_new and not month_end and changed,
        "source_score_mature": row["source_score_mature"],
        "source_return_count": row["source_return_count"],
        "known_row_is_native_request_or_execution": False,
        "new_main_target_or_rebalance_generated": False,
        "native_event_completeness_or_new_source_admission_verified": False,
    }


def summarize(per_source):
    keys = (
        "initial_observation",
        "initial_observes_prior_dated_opinion",
        "carried_known_timestamp",
        "new_same_day_known_row",
        "newly_observed_prior_dated_row",
        "numeric_target_changed",
        "new_same_day_row_same_numeric_target",
        "off_month_end_new_same_day_row",
        "off_month_end_changed_opinion",
        "source_score_mature",
    )
    return {
        "sources": {
            name: {
                "observations": len(rows),
                **{key: sum(row[key] for row in rows) for key in keys},
            }
            for name, rows in per_source.items()
        },
        "all_observations": sum(len(rows) for rows in per_source.values()),
        "current_monthly_execution_accepts_extra_non_month_end_targets": False,
        "future_consumer_requires_separately_frozen_execution_and_native_clock_review": True,
        "source_known_rows_are_native_requests": False,
        "global_membership_or_source_scope_changed": False,
        "new_rankings_targets_accounts_or_financial_trials": 0,
    }
