import copy

import numpy as np
import pandas as pd
import pytest
from test_legacy_term_risk_eligibility_v1 import synthetic_books

from us_quant import legacy_term_risk_eligibility_v2 as rules
from us_quant.calendar import sessions
from us_quant.config import QuantError


def books_with_prehistory():
    snapshot, bundle, frames, rf = synthetic_books()
    prefix = sessions("2015-11-30", "2016-10-04")[:25]
    for data in frames.values():
        targets = data["targets"]
        whole = pd.DataFrame(
            np.nan, index=sessions("2015-08-10", "2026-10-05"), columns=targets.columns
        )
        whole.loc[targets.index] = targets
        whole.loc[prefix] = 0.98 / 8
        data["targets"] = whole
    return snapshot, bundle, frames, rf


@pytest.fixture(scope="module")
def books():
    return books_with_prehistory()


def test_full_prehistory_and_defined_formal_anchor_are_separate(books):
    before = {key: data["targets"].copy() for key, data in books[2].items()}
    checked = rules.check(*books)
    assert checked["four_native_cached_paths_checked"] == 4
    for row in checked["paths"][:2]:
        assert row["native_first_active_request"] == "2015-11-30"
        assert row["actual_formal_anchor"] == "2016-10-05"
        assert row["formal_anchor_row_is_defined"] is True
        assert row["native_requests_before_formal_anchor_preserved"] == 25
        assert row["formal_source_initial_capital_usd"] == 10000
        assert row["formal_first_return_initial_capital_reconciliation_difference"] < 1e-12
    for key, frame in before.items():
        pd.testing.assert_frame_equal(frame, books[2][key]["targets"])
    assert checked["old_V1_first_active_assumption_or_failure_rewritten"] is False
    assert checked["new_expert_admitted"] is False


@pytest.mark.parametrize("mode", ["all_missing", "partial"])
def test_missing_or_partial_formal_anchor_is_not_filled(books, mode):
    changed = copy.deepcopy(books)
    target = changed[2][10, "base"]["targets"]
    if mode == "all_missing":
        target.loc["2016-10-05"] = np.nan
    else:
        target.loc["2016-10-05", "SPY"] = np.nan
    with pytest.raises(QuantError):
        rules.check(*changed)
    assert target.loc["2016-10-05"].isna().any()


@pytest.mark.parametrize("mode", ["capital", "budget"])
def test_formal_capital_or_anchor_budget_cannot_change(books, mode):
    changed = copy.deepcopy(books)
    if mode == "capital":
        changed[1]["paths"][0]["capital_usd"] = 20000
    else:
        changed[2][10, "base"]["targets"].loc["2016-10-05", "SPY"] += 0.1
    with pytest.raises(QuantError):
        rules.check(*changed)


def test_RF_funding_and_native_Adapter_checks_remain_complete(books):
    snapshot, bundle, frames, rf = copy.deepcopy(books)
    rf.loc["2016-10-06"] += 1e-15
    frames[10, "stress"]["strategy"].loc["2016-10-06", "cash"] = -0.01
    frames[5, "base"]["strategy_bt"].iloc[0, 0] += 0.01
    checked = rules.check(snapshot, bundle, frames, rf)
    assert len(checked["paths"]) == 4
    assert checked["status"] == "not_admitted_current_Interface"
    assert any(row["RF_mismatch_count"] == 1 for row in checked["paths"])
    assert (
        sum(row["native_cash_funding_and_accounting_checks_pass"] for row in checked["paths"]) == 2
    )
    assert all(
        not row["existing_dense_full_native_Adapter_calendar_compatible"]
        for row in checked["paths"]
    )
    assert checked["new_Adapter_or_recommendation_view_created"] is False
    assert checked["native_RF_redefined_or_new_common_RF_view_created"] is False


def test_native_calendar_is_not_filtered_to_formal_window(books):
    changed = copy.deepcopy(books)
    for data in changed[2].values():
        data["targets"].loc["2015-11-30":] = 0.98 / 8
    checked = rules.check(*changed)
    assert all(
        row["existing_dense_full_native_Adapter_calendar_compatible"] for row in checked["paths"]
    )
    assert all(row["native_first_active_request"] == "2015-11-30" for row in checked["paths"])
    assert checked["status"] == "eligible_not_admitted"
    assert checked["new_expert_admitted"] is False


def test_source_prehistory_invalid_budget_is_not_hidden(books):
    changed = copy.deepcopy(books)
    changed[2][10, "base"]["targets"].loc["2015-11-30", "SPY"] += 0.1
    with pytest.raises(QuantError, match="full-source"):
        rules.check(*changed)


def test_V1_checker_remains_unchanged_and_rejects_prehistory(books):
    from us_quant import legacy_term_risk_eligibility_v1 as old

    with pytest.raises(QuantError, match="actual initial anchor"):
        old.check(*books)
    checked = rules.check(*books)
    assert checked["new_time_range_contract_checked_without_source_crop"] is True
