from dataclasses import replace

import numpy as np
import pytest
from test_quintet_learning_loss_accounting_v1 import examples as prior_examples

from us_quant import equal_role_loss_accounting_v1 as rules
from us_quant.config import QuantError
from us_quant.storage import read_json


def examples():
    old, opened, closed = prior_examples()
    return {"equal_role": old["adaptive"], "flat": old["elapsed"]}, opened, closed


def test_two_books_all_own_and_common_clocks_groups_offsets_and_contrasts():
    books, opened, closed = examples()
    daily, summary = rules.compare(books, opened, closed)
    assert set(daily) == set(summary["own_books"]) == set(rules.ROLES)
    assert len(summary["common_clock_episode_views"]) == 2
    assert len(summary["equal_role_minus_flat_comparisons"]) == 2
    for item in summary["own_books"].values():
        episode = item["own_maximum_drawdown_episode"]
        assert episode["asset_group_mark_PnL_USD"]["equity_ETFs"] < 0
        assert episode["asset_group_mark_PnL_USD"]["gold_ETF"] > 0
        assert episode["asset_group_is_strategy_expert"] is False
        assert episode["maximum_whole_interval_identity_error_USD"] <= 1e-6
    for row in summary["common_clock_episode_views"]:
        views = row["both_stored_accounts_at_identical_clocks"]
        assert len({(v["start"], v["end"]) for v in views.values()}) == 1
        assert row["different_start_NAVs_are_equal_capital_counterfactuals"] is False


def test_original_function_objects_and_limits_reused_without_old_pipeline():
    assert rules.marks is rules.prior.marks
    assert rules.grouped is rules.prior.grouped
    assert rules.StoredBook is rules.prior.StoredBook
    assert rules.check_prefix is rules.canonical.check_prefix
    assert rules.marks.MONEY_TOLERANCE == 1e-6
    assert rules.marks.METRIC_TOLERANCE == 1e-10
    rules.validate(read_json(rules.POLICY))


@pytest.mark.parametrize("kind", ["missing", "date", "asset", "cash", "equity", "cost", "DD"])
def test_changed_book_or_coordinate_is_not_repaired(kind):
    books, opened, closed = examples()
    book = books["equal_role"]
    if kind == "missing":
        del books["flat"]
    elif kind == "date":
        books["equal_role"] = replace(book, account=book.account.iloc[:-1])
    elif kind == "asset":
        closed = closed.iloc[:, ::-1]
    elif kind == "DD":
        books["equal_role"] = replace(book, cached_max_drawdown=book.cached_max_drawdown + 1e-8)
    else:
        altered = book.account.copy()
        altered.loc[altered.index[1], kind] += 0.01
        books["equal_role"] = replace(book, account=altered)
    with pytest.raises(QuantError):
        rules.compare(books, opened, closed)


def test_primitive_and_scalar_prefixes_including_signed_zero():
    books, opened, closed = examples()
    daily, _ = rules.compare(books, opened, closed)
    for role, book in books.items():
        for cut in (1, 2, len(opened)):
            prefix, _ = rules.marks.decompose(
                book.account.iloc[:cut],
                book.weights.iloc[:cut],
                opened.iloc[:cut],
                closed.iloc[:cut],
            )
            report = rules.check_prefix(daily[role], prefix)
            assert report["primitive_and_canonical_uint64_prefix_exact_including_signed_zero"]
        mutated = daily[role].copy()
        col = next(c for c in mutated if c.startswith("overnight::") and mutated[c].iloc[0] == 0)
        mutated.loc[mutated.index[0], col] = np.copysign(0.0, -mutated[col].iloc[0])
        with pytest.raises(QuantError):
            rules.check_prefix(daily[role], mutated)


@pytest.mark.parametrize(
    "key,value",
    [
        ("money_tolerance_USD", 1e-3),
        ("cached_drawdown_tolerance", 1e-6),
        ("new_strategy_target_account_BT_or_financial_trials", True),
    ],
)
def test_frozen_numeric_and_nonfinancial_scope(key, value):
    policy = read_json(rules.POLICY)
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)
