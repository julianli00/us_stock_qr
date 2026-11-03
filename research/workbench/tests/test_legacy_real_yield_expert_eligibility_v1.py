import copy

import numpy as np
import pandas as pd
import pytest

from us_quant import legacy_real_yield_expert_eligibility_v1 as rules
from us_quant.calendar import is_month_end, previous_session, sessions
from us_quant.config import QuantError
from us_quant.storage import digest_json


def source(*, funding_seed=False, rf_gap=False):
    context = sessions("2015-08-10", "2026-10-05")
    target = pd.DataFrame(np.nan, index=context, columns=rules.ASSETS)
    active = [day for day in context if is_month_end(day)]
    target.loc[active] = [0.98 / 8] * 8
    common = pd.Series(0.0001, index=context, name="risk_free")
    frames = {}
    paths = []
    reviews = []
    for years, scenario in rules.PATHS:
        start = "2016-10-06" if years == 10 else "2021-10-06"
        dates = sessions(start, "2026-10-05")
        nav = 10000 * np.cumprod(np.full(len(dates), 1.0003))
        book = pd.DataFrame(
            {
                "equity": nav,
                "return": 0.0003,
                "cash": nav * 0.02,
                "gross_exposure": 0.98,
                "turnover": 0.0,
                "cost": 0.0,
                "orders": 0,
                "risk_free": 0.0001,
            },
            index=dates,
        )
        if rf_gap:
            book.iloc[0, book.columns.get_loc("risk_free")] = np.nextafter(0.0001, 1.0)
        targets = target.copy()
        if funding_seed:
            targets.loc[previous_session(start)] = [0.98 / 8] * 8
        frames[years, scenario] = {
            "strategy": book,
            "strategy_bt": book[["equity"]].copy(),
            "weights": pd.DataFrame(0.98 / 8, index=dates, columns=rules.ASSETS),
            "targets": targets,
        }
        paths.append(
            {
                "years": years,
                "scenario": scenario,
                "capital_usd": 10000,
                "commission_per_order": 1,
                "cost_bps": 5.0 if scenario == "base" else 20.0,
                "delay_sessions": 1 if scenario == "base" else 2,
            }
        )
        reviews.append(
            {
                "years": years,
                "scenario": scenario,
                "market_and_target_replay_passed": True,
                "independent_bt_reexecuted": True,
                "max_independent_equity_error_usd": 0.0,
            }
        )
    snapshot = {"review": {"paths": reviews}, "registered_at": "2026-10-10T15:53:49.512619Z"}
    return snapshot, {"paths": paths}, frames, common


def test_genuine_monthly_anchor_uses_known_prior_request_not_full_context_first_active():
    result = rules.check(*source())
    assert result["status"] == "eligible_not_admitted"
    assert all(row["direct_monthly_calendar_compatible"] for row in result["paths"])
    assert all(not row["formal_anchor_literal_active_row"] for row in result["paths"])
    assert all(row["native_requests_before_formal_anchor_preserved"] > 0 for row in result["paths"])


def test_original_funding_seed_is_retained_and_blocks_direct_monthly_not_deleted():
    result = rules.check(*source(funding_seed=True))
    assert result["status"] == "not_admitted_current_Interface"
    assert all(len(row["full_native_nonmonthend_dates"]) == 1 for row in result["paths"])
    assert all(not row["direct_monthly_calendar_compatible"] for row in result["paths"])
    assert result["native_funding_seed_rows_not_deleted_to_manufacture_monthly_compatibility"]
    assert result["new_RF_recommendation_assessment_view_or_Adapter"] is False


def test_one_ulp_RF_difference_is_a_real_exact_identity_block_without_normalization():
    result = rules.check(*source(rf_gap=True))
    assert all(row["RF_mismatch_count"] == 1 for row in result["paths"])
    assert any("RF" in row["reason"] for row in result["blockers"])
    assert result["new_expert_admitted"] is False


def test_cash_and_independent_accounting_failure_are_not_financial_score_exclusions():
    values = source()
    values[2][10, "base"]["strategy"].iloc[0, 2] = -1
    result = rules.check(*values)
    assert any(row["reason"] == "native_cash_funding_or_accounting" for row in result["blockers"])
    assert result["old_standalone_financial_rejection_preserved"]


@pytest.mark.parametrize("kind", ["missing_path", "missing_target", "nan", "wrong_delay"])
def test_other_source_qualification_failures_stop_without_fill_or_source_pipeline(kind):
    snapshot, bundle, frames, common = source()
    frames = copy.deepcopy(frames)
    if kind == "missing_path":
        del frames[10, "base"]
    elif kind == "missing_target":
        frames[10, "base"]["targets"] = frames[10, "base"]["targets"].astype(object)
        frames[10, "base"]["targets"].loc[:] = ""
    elif kind == "nan":
        frames[10, "base"]["strategy"].iloc[0, 0] = np.nan
    else:
        bundle["paths"][0]["delay_sessions"] = 0
    with pytest.raises((QuantError, TypeError, ValueError)):
        rules.check(snapshot, bundle, frames, common)


def identity_source():
    policy = {
        "publication_delay_sessions": 2,
        "maximum_observation_age_days": 7,
        "real_yield_change_sessions": 63,
        "rising_real_yield_factor_shares": [0.15, 0.35, 0.15, 0.35],
        "nonrising_real_yield_factor_shares": [0.35, 0.15, 0.35, 0.15],
        "candidates": [{"id": rules.CANDIDATE, "curve_defense": False}],
    }
    spec = {
        "id": rules.CANDIDATE,
        "configuration": policy["candidates"][0],
        "market": {"original": "unchanged"},
        "leveraged_products_allowed": False,
        "order_authority": False,
        "asset_leverage": {s: 1.0 for s in rules.ASSETS},
    }
    bundle = {
        "candidate_spec_sha256": digest_json(spec),
        "market": spec["market"],
        "order_authority": False,
        "leveraged_products_allowed": False,
        "completed_at": "2026-10-10T15:54:14Z",
        "paths": [{"years": y, "scenario": s} for y, s in rules.PATHS],
    }
    snapshot = {
        "candidate_id": rules.CANDIDATE,
        "technical_failures": 0,
        "events": [{"sequence": 21}, {"sequence": 23}],
        "specification": spec,
        "spec_sha256": digest_json(spec),
        "evidence_sha256": digest_json(bundle),
        "registered_at": "2026-10-10T15:53:49Z",
        "reviewed_at": "2026-10-10T15:55:00Z",
        "review": {"status": "rejected_historical"},
    }
    registration = {
        "candidates": [
            {
                "spec": spec,
                "registration": {
                    "registered_at": snapshot["registered_at"],
                    "spec_sha256": snapshot["spec_sha256"],
                },
            }
        ]
    }
    results = {"candidates": {rules.CANDIDATE: snapshot["review"]}}
    return snapshot, registration, results, bundle, policy


def test_original_real_registration_review_and_policy_identity_is_not_return_reselection():
    result = rules.check_identity(*identity_source())
    assert result["actual_source_events"] == [21, 23]
    assert result["original_standalone_rejection_preserved"]


def test_wrong_macro_law_identity_cannot_enter_source_eligibility():
    inputs = list(identity_source())
    inputs[-1]["real_yield_change_sessions"] = 126
    with pytest.raises(QuantError, match="original candidate21"):
        rules.check_identity(*inputs)


def test_source_review_dependency_scope_captures_current_code_without_old_entrypoints():
    paths = {row["path"] for row in rules.dependencies()}
    assert set(rules.OWN).issubset(paths)
    assert "src/us_quant/strategy_experts_v1.py" in paths
    assert "src/us_quant/november_research_lifecycle_v1.py" in paths
