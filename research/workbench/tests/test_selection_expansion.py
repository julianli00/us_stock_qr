from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.config import QuantError
from us_quant.selection_expansion import (
    matrices,
    validate_cohort,
    validate_policy,
    verify_original_prefix,
)
from us_quant.selection_validation import joint_maximum
from us_quant.storage import read_json

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/selection-expansion.json")


def cohort():
    ids = [f"candidate_{index:02}" for index in range(36)]
    state = {
        "candidate_reviews": [{"candidate_id": name} for name in ids],
        "total_evaluated_configurations": 120,
        "pending_candidate_ids": [],
        "event_chain_sha256": "1" * 64,
    }
    audit = {
        "reviewed_candidates": 36,
        "regenerated_strategy_paths": 144,
        "ledger_unchanged": True,
        "event_chain_sha256": "1" * 64,
        "reviews": state["candidate_reviews"],
    }
    return state, audit, ids[:22]


def test_expansion_keeps_exact_22_candidate_prefix_and_does_not_claim_the_earlier_84(policy):
    state, audit, original = cohort()
    validate_policy(policy)
    validate_cohort(state, audit, original, policy)
    assert policy["included_candidate_count"] == 36
    assert policy["earlier_configurations_not_in_joint_inference"] == 84
    assert policy["total_disclosed_configurations"] == 120
    assert (
        policy["cost_comparisons"]
        == read_json(ROOT / "config/selection-validation.json")["cost_comparisons"]
    )


@pytest.mark.parametrize(
    "problem",
    ["dropped_failure", "duplicate", "old_order", "audit_head", "pending", "incomplete_paths"],
)
def test_changed_cohort_or_incomplete_review_paths_cannot_be_silently_excluded(policy, problem):
    state, audit, original = cohort()
    if problem == "dropped_failure":
        state["candidate_reviews"] = state["candidate_reviews"][:-1]
    elif problem == "duplicate":
        state["candidate_reviews"][-1] = state["candidate_reviews"][0]
    elif problem == "old_order":
        state["candidate_reviews"][0], state["candidate_reviews"][1] = (
            state["candidate_reviews"][1],
            state["candidate_reviews"][0],
        )
    elif problem == "audit_head":
        audit["event_chain_sha256"] = "2" * 64
    elif problem == "pending":
        state["pending_candidate_ids"] = ["unreviewed"]
    else:
        audit["regenerated_strategy_paths"] = 143
    with pytest.raises(QuantError):
        validate_cohort(state, audit, original, policy)


@pytest.mark.parametrize(
    "field,value",
    [
        ("included_candidate_count", 35),
        ("bootstrap_samples", 10000),
        ("block_sessions", 63),
        ("seed", 20261012),
        ("familywise_alpha", 0.10),
        ("capital_usd", 100000),
        ("earlier_configurations_not_in_joint_inference", 0),
        ("order_authority", True),
        ("change_research_qualification_policy", True),
    ],
)
def test_settings_scope_costs_and_authority_cannot_be_retuned(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        validate_policy(policy)


def test_reused_statistic_preserves_old_prefix_and_nonshrinking_common_maximum_bound():
    rng = np.random.default_rng(29)
    values = rng.normal(0, 0.002, (137, 108))
    frame = pd.DataFrame(values, columns=[f"candidate::{index}" for index in range(108)])
    before = joint_maximum(frame.iloc[:, :66], samples=128, block=21, seed=20261011, alpha=0.025)
    expanded = joint_maximum(frame, samples=128, block=21, seed=20261011, alpha=0.025)
    verify_original_prefix(before, deepcopy(before))
    assert expanded["comparisons"] == 108
    assert (
        expanded["bootstrap_maximum_critical_daily_log_advantage"]
        >= before["bootstrap_maximum_critical_daily_log_advantage"] - 1e-15
    )
    invalid = deepcopy(before)
    invalid["scope_limited_omnibus_p_value"] += 0.01
    with pytest.raises(QuantError, match="did not reproduce"):
        verify_original_prefix(before, invalid)


def test_every_review_and_all_three_existing_cost_comparisons_enter_the_matrix(policy, monkeypatch):
    state, audit, original = cohort()
    dates = pd.date_range("2020-01-01", periods=42)
    accounts = {
        "base": {
            "strategy": pd.DataFrame({"return": 0.002}, index=dates),
            "spy": pd.DataFrame({"return": 0.001}, index=dates),
        },
        "stress": {
            "strategy": pd.DataFrame({"return": 0.0015}, index=dates),
            "spy": pd.DataFrame({"return": 0.0008}, index=dates),
        },
    }
    calls = []

    def reviewed(bundle, review, years, settings):
        calls.append((review["candidate_id"], years))
        return accounts

    monkeypatch.setattr(
        "us_quant.selection_expansion.original_bundle", lambda review: ({}, {"sha256": "1" * 64})
    )
    monkeypatch.setattr("us_quant.selection_expansion.verified_accounts", reviewed)
    frames, sources = matrices(state, policy)
    assert len(sources) == 36 and len(calls) == 72
    assert frames["10"].shape == frames["5"].shape == (42, 108)
    assert frames["10"].columns[0] == "candidate_00::base_vs_base"
    assert frames["10"].columns[65] == "candidate_21::stress_vs_base"
    assert frames["10"].columns[66] == "candidate_22::base_vs_base"
    assert np.isfinite(frames["10"]).all().all()
