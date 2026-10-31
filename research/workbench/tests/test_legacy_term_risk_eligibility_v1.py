import copy

import numpy as np
import pandas as pd
import pytest

from us_quant import legacy_term_risk_eligibility_v1 as rules
from us_quant.calendar import is_month_end, previous_session, sessions
from us_quant.config import QuantError
from us_quant.storage import read_json


def synthetic_books():
    frames, records, reviews = {}, [], []
    rf = pd.Series(0.00005, index=sessions("2016-10-05", "2026-10-05"), name="risk_free")
    for years, scenario in rules.PATHS:
        start = "2016-10-06" if years == 10 else "2021-10-06"
        dates = sessions(start, "2026-10-05")
        anchor = previous_session(start)
        equity = 10000 * np.cumprod(np.full(len(dates), 1.0001))
        book = pd.DataFrame(
            {
                "equity": equity,
                "return": 0.0001,
                "cash": equity * 0.02,
                "gross_exposure": 0.98,
                "turnover": 0.0,
                "cost": 0.0,
                "orders": 0,
                "risk_free": rf.loc[dates],
            },
            index=dates,
        )
        weights = pd.DataFrame(0.98 / 8, index=dates, columns=rules.ASSETS)
        target_dates = sessions(anchor, "2026-10-05")
        targets = pd.DataFrame(np.nan, index=target_dates, columns=rules.ASSETS)
        active = [anchor, *[day for day in target_dates if is_month_end(day)], target_dates[2]]
        targets.loc[active] = 0.98 / 8
        frames[years, scenario] = {
            "strategy": book,
            "strategy_bt": book.loc[:, ["equity", "return"]].copy(),
            "weights": weights,
            "targets": targets,
        }
        records.append(
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
    return {"review": {"paths": reviews}}, {"paths": records}, frames, rf


@pytest.fixture(scope="module")
def books():
    return synthetic_books()


def test_cached_funding_is_separate_from_calendar_eligibility(books):
    checked = rules.check(*books)
    assert checked["status"] == "not_admitted_current_Interface"
    assert checked["new_expert_admitted"] is False
    assert checked["active_pool_changed"] is False
    assert all(row["native_cash_funding_and_accounting_checks_pass"] for row in checked["paths"])
    assert all(row["native_RF_exact_active_common_reader"] for row in checked["paths"])
    assert all(not row["missing_meta_decision_dates"] for row in checked["paths"])
    assert all(not row["existing_dense_Adapter_calendar_compatible"] for row in checked["paths"])
    assert all(not row["existing_monthly_Adapter_calendar_compatible"] for row in checked["paths"])
    assert checked["source_policy_technically_invalid_merely_for_sparse_requests"] is False


def test_RF_exact_mismatch_is_explicit_without_replacement(books):
    snapshot, bundle, frames, rf = books
    changed = rf.copy()
    changed.loc["2016-10-06"] += 1e-15
    checked = rules.check(snapshot, bundle, frames, changed)
    affected = [row for row in checked["paths"] if row["RF_mismatch_count"]]
    assert len(affected) == 2
    assert all(row["RF_mismatch_count"] == 1 for row in affected)
    assert all(row["first_RF_mismatch_date"] == "2016-10-06" for row in affected)
    assert checked["native_RF_redefined_or_new_common_RF_view_created"] is False
    assert any(row["reason"] == "native_RF_not_exact_active_common" for row in checked["blockers"])


@pytest.mark.parametrize(
    "field", ["missing_date", "partial_target", "target_budget", "negative_cash", "gross_leverage"]
)
def test_source_shape_budget_or_funding_failures_are_not_silent_success(books, field):
    changed = copy.deepcopy(books)
    data = changed[2][10, "base"]
    if field == "missing_date":
        data["strategy"] = data["strategy"].iloc[1:]
    elif field == "partial_target":
        data["targets"].iloc[0, 0] = np.nan
    elif field == "target_budget":
        data["targets"].iloc[0, 0] += 0.1
    elif field == "negative_cash":
        data["strategy"].iloc[0, data["strategy"].columns.get_loc("cash")] = -0.01
    elif field == "gross_leverage":
        data["strategy"].iloc[0, data["strategy"].columns.get_loc("gross_exposure")] = 1.01
    if field in ("negative_cash", "gross_leverage"):
        checked = rules.check(*changed)
        assert checked["status"] == "not_admitted_current_Interface"
        assert not checked["paths"][0]["native_cash_funding_and_accounting_checks_pass"]
        assert any(
            row["reason"] == "native_cash_funding_or_accounting" for row in checked["blockers"]
        )
    else:
        with pytest.raises(QuantError):
            rules.check(*changed)


def test_existing_dense_calendar_can_be_reported_eligible_without_admission(books):
    changed = copy.deepcopy(books)
    for data in changed[2].values():
        data["targets"].loc[:] = 0.98 / 8
    checked = rules.check(*changed)
    assert checked["status"] == "eligible_not_admitted"
    assert checked["new_expert_admitted"] is False
    assert all(row["existing_dense_Adapter_calendar_compatible"] for row in checked["paths"])


def test_original_identity_is_preserved_without_return_rank_selection():
    from us_quant import ssd_legacy_term_risk_eligibility_v1 as study

    design = read_json(study.DESIGN)
    snapshot = read_json(study.base.checked_reference(design["source_identity_snapshot"]))
    registration = read_json(study.base.checked_reference(design["source_registration"]))
    results = read_json(study.base.checked_reference(design["source_results"]))
    bundle = read_json(study.base.checked_reference(design["source_bundle"]))
    policy = read_json(study.base.checked_reference(design["source_policy"]))
    checked = rules.check_identity(snapshot, registration, results, bundle, policy)
    assert checked["original_candidate_event_sequence"] == 17
    assert checked["original_review_event_sequence"] == 19
    assert checked["original_standalone_verdict_preserved"] == "rejected_historical"
    altered = copy.deepcopy(policy)
    altered["candidates"].reverse()
    with pytest.raises(QuantError, match="source identity"):
        rules.check_identity(snapshot, registration, results, bundle, altered)
