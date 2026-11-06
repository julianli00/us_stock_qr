from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_saved_main_book_decomposition_v1 import examples as original

from us_quant import quintet_learning_loss_accounting_v1 as rules
from us_quant.config import QuantError
from us_quant.storage import read_json


def examples():
    account, weights, opened, closed, _ = original()
    op = pd.DataFrame(100.0, index=opened.index, columns=rules.ASSETS)
    cl = op.copy()
    for source, symbol in (("A", "SPY"), ("B", "GLD")):
        op[symbol], cl[symbol] = opened[source], closed[source]
    books = {}
    for role, scale in zip(rules.ROLES, (1.0, 0.8, 0.6), strict=True):
        equity = 10000 * (1 - scale) + scale * account["equity"]
        cash = 10000 * (1 - scale) + scale * account["cash"]
        allocation = pd.DataFrame(0.0, index=account.index, columns=rules.ASSETS)
        for source, symbol in (("A", "SPY"), ("B", "GLD")):
            allocation[symbol] = scale * account["equity"] * weights[source] / equity
        book = pd.DataFrame(
            {
                "equity": equity,
                "cash": cash,
                "gross_exposure": allocation.sum(axis=1),
                "cost": scale * account["cost"],
                "orders": account["orders"],
            },
            index=account.index,
        )
        curve = [10000, *equity]
        cached = max(1 - value / max(curve[: i + 1]) for i, value in enumerate(curve[1:]))
        books[role] = rules.StoredBook(book, allocation, cached)
    return books, op, cl


def test_three_whole_books_all_own_clocks_groups_offsets_and_contrasts():
    books, op, cl = examples()
    daily, result = rules.compare(books, op, cl)
    assert set(daily) == set(result["own_books"]) == set(rules.ROLES)
    assert len(result["common_clock_episode_views"]) == 3
    assert len(result["adaptive_minus_control_comparisons"]) == 6
    for summary in result["own_books"].values():
        episode = summary["own_maximum_drawdown_episode"]
        assert episode["asset_group_mark_PnL_USD"]["equity_ETFs"] < 0
        assert episode["asset_group_mark_PnL_USD"]["gold_ETF"] > 0
        assert episode["asset_group_is_strategy_expert"] is False
        assert episode["mechanical_PnL_identity_is_expert_causal_alpha"] is False
        assert episode["maximum_whole_interval_identity_error_USD"] <= 1e-6
    for row in result["common_clock_episode_views"]:
        views = row["all_three_stored_accounts_at_identical_clocks"]
        assert set(views) == set(rules.ROLES)
        assert len({(v["start"], v["end"]) for v in views.values()}) == 1
        assert row["different_start_NAVs_are_equal_capital_counterfactuals"] is False


def test_original_accounting_function_objects_and_money_limits_are_reused():
    assert rules.marks is rules.prior.marks
    assert rules.grouped is rules.prior.grouped
    assert rules.StoredBook is rules.prior.StoredBook
    assert rules.marks.MONEY_TOLERANCE == 1e-6
    assert rules.marks.METRIC_TOLERANCE == 1e-10
    rules.validate(read_json(rules.POLICY))


@pytest.mark.parametrize(
    "kind", ["missing_role", "date", "asset", "cash", "equity", "cost", "cached_DD"]
)
def test_missing_or_changed_saved_source_evidence_refuses_diagnosis(kind):
    books, op, cl = examples()
    book = books["adaptive"]
    if kind == "missing_role":
        del books["risk_units"]
    elif kind == "date":
        altered = book.account.copy()
        altered.index = altered.index + pd.Timedelta(days=1)
        books["adaptive"] = replace(book, account=altered)
    elif kind == "asset":
        cl = cl.iloc[:, ::-1]
    elif kind == "cached_DD":
        books["adaptive"] = replace(book, cached_max_drawdown=book.cached_max_drawdown + 1e-8)
    else:
        altered = book.account.copy()
        altered.loc[altered.index[1], kind] += 0.01
        books["adaptive"] = replace(book, account=altered)
    with pytest.raises(QuantError):
        rules.compare(books, op, cl)


def test_initial_10000_peak_equal_peak_ties_and_first_maximum_trough():
    dates = pd.bdate_range("2020-01-02", periods=4)
    op = pd.DataFrame(100.0, index=dates, columns=rules.ASSETS)
    cl = op.copy()
    cl["SPY"] = [100.0, 100.0, 90.0, 90.0]
    equity = np.array([10000.0, 10000.0, 9500.0, 9500.0])
    weights = cl * 0
    weights["SPY"] = cl["SPY"] * 50 / equity
    account = pd.DataFrame(
        {
            "equity": equity,
            "cash": 5000.0,
            "gross_exposure": weights.sum(axis=1),
            "cost": 0.0,
            "orders": [1, 0, 0, 0],
        },
        index=dates,
    )
    book = rules.StoredBook(account, weights, 0.05)
    _, result = rules.compare({role: book for role in rules.ROLES}, op, cl)
    for summary in result["own_books"].values():
        episode = summary["own_maximum_drawdown_episode"]
        assert episode["start_position"] == 1 and episode["end_position"] == 2
    initial = account.iloc[:1].copy()
    initial.loc[:, "equity"] = initial.loc[:, "cash"] = 9990.0
    initial.loc[:, "gross_exposure"] = 0.0
    initial.loc[:, "cost"] = 10.0
    initial.loc[:, "orders"] = 0
    book = rules.StoredBook(initial, weights.iloc[:1] * 0, 0.001)
    _, result = rules.compare({role: book for role in rules.ROLES}, op.iloc[:1], cl.iloc[:1])
    assert result["own_books"]["adaptive"]["own_maximum_drawdown_episode"]["start_position"] == -1


def test_no_observed_drawdown_is_unavailable_not_safe_or_synthetic_episode():
    books, op, cl = examples()
    index = op.index
    weights = pd.DataFrame(0.0, index=index, columns=rules.ASSETS)
    account = pd.DataFrame(
        {
            "equity": 10000.0,
            "cash": 10000.0,
            "gross_exposure": 0.0,
            "cost": 0.0,
            "orders": 0,
        },
        index=index,
    )
    book = rules.StoredBook(account, weights, 0.0)
    _, result = rules.compare({role: book for role in rules.ROLES}, op, cl)
    assert all(v["own_maximum_drawdown_episode"] is None for v in result["own_books"].values())
    assert all(
        v["all_three_stored_accounts_at_identical_clocks"] is None
        for v in result["common_clock_episode_views"]
    )
    assert all(
        v["adaptive_minus_control"] is None for v in result["adaptive_minus_control_comparisons"]
    )


def test_primitive_and_canonical_prefixes_are_bit_exact_when_future_rows_change():
    books, op, cl = examples()
    daily, _ = rules.compare(books, op, cl)
    for role in rules.ROLES:
        book = books[role]
        prefix, _ = rules.marks.decompose(
            book.account.iloc[:3], book.weights.iloc[:3], op.iloc[:3], cl.iloc[:3]
        )
        report = rules.check_prefix(daily[role], prefix)
        assert report["primitive_and_canonical_uint64_prefix_exact_including_signed_zero"]
        future = book.account.copy()
        future.iloc[3:, future.columns.get_loc("equity")] = np.nan
        prefix_after, _ = rules.marks.decompose(
            future.iloc[:3], book.weights.iloc[:3], op.iloc[:3], cl.iloc[:3]
        )
        assert rules.canonical.canonical_prefix_view(prefix).equals(
            rules.canonical.canonical_prefix_view(prefix_after)
        )


def test_signed_zero_primitive_change_is_not_hidden_by_dataframe_equals():
    books, op, cl = examples()
    daily, _ = rules.compare(books, op, cl)
    full = daily["adaptive"]
    changed = full.iloc[:3].copy()
    assert changed["overnight::IEF"].iloc[0] == 0
    changed.iloc[0, changed.columns.get_loc("overnight::IEF")] = -0.0
    assert changed.equals(full.iloc[:3])
    with pytest.raises(QuantError):
        rules.check_prefix(full, changed)


def test_auxiliary_vector_is_retained_and_disclosed_not_used_to_waive_primitive_equality():
    books, op, cl = examples()
    daily, _ = rules.compare(books, op, cl)
    full = daily["adaptive"]
    changed = full.iloc[:3].copy()
    name = rules.canonical.AUXILIARY
    changed.iloc[0, changed.columns.get_loc(name)] = np.nextafter(changed[name].iloc[0], np.inf)
    report = rules.check_prefix(full, changed)
    assert report["auxiliary_NumPy_vector_bit_different_values"] == 1
    assert report["original_V1_all_column_prefix_comparison"] is False
    assert report["failed_V1_all_column_contract_repaired_or_waived"] is False
    assert changed[name].iloc[0] != full[name].iloc[0]


@pytest.mark.parametrize(
    "key,value",
    [
        ("stored_full_books", 8),
        ("monetary_reconciliation_tolerance_USD", 1e-3),
        ("cached_drawdown_agreement_tolerance", 1e-6),
        ("initial_capital_USD", 9990),
        ("all_three_own_episode_common_clocks_required", False),
        ("primitive_and_canonical_prefix_uint64_including_signed_zero", False),
        ("failed_V1_all_column_contract_repaired_or_waived", True),
        ("source_RF_calendar_or_price_coordinates_changed", True),
        ("source_CAGR_Sharpe_or_qualification_recomputed", True),
        ("asset_groups_are_strategy_experts", True),
        ("different_start_NAVs_are_equal_capital_counterfactuals", True),
        ("fee_free_or_expert_causal_counterfactual_claimed", True),
    ],
)
def test_no_scope_money_prefix_clock_source_or_interpretation_rescue(key, value):
    policy = deepcopy(read_json(rules.POLICY))
    with pytest.raises(QuantError):
        rules.validate({**policy, key: value})
