import copy
import math
from dataclasses import replace
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_saved_main_book_decomposition_v1 import examples

from us_quant import quartet_trio_loss_accounting_v1 as rules
from us_quant import saved_main_book_decomposition_v1 as original
from us_quant import ssd_quartet_trio_loss_accounting_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json


def sources():
    account, weights, opened, closed, _ = examples()
    mapping = {"A": "SPY", "B": "GLD"}
    opened = opened.rename(columns=mapping)
    closed = closed.rename(columns=mapping)
    weights = weights.rename(columns=mapping)
    for symbol in rules.ASSETS:
        if symbol not in weights:
            weights[symbol] = 0.0
            opened[symbol] = closed[symbol] = 100.0
    opened, closed, weights = [f.loc[:, list(rules.ASSETS)] for f in (opened, closed, weights)]
    nav = np.r_[10000.0, account["equity"].to_numpy()]
    cached = float(max(1 - nav / np.maximum.accumulate(nav)))
    flat = pd.DataFrame(
        {"equity": 10000.0, "cash": 10000.0, "gross_exposure": 0.0, "cost": 0.0, "orders": 0},
        index=account.index,
    )
    books = {
        "quartet": rules.StoredBook(account, weights, cached),
        "cached_trio": rules.StoredBook(flat, weights * 0.0, 0.0),
    }
    return books, opened, closed


def test_paired_interface_reuses_the_unchanged_original_accounting_module():
    books, opened, closed = sources()
    assert rules.marks.decompose is original.decompose
    assert rules.marks.interval is original.interval
    daily, report = rules.compare(books, opened, closed)
    assert set(daily) == set(rules.ROLES)
    assert len(report["common_clock_episode_views"]) == 2
    assert report["new_strategy_target_account_BT_or_financial_trials"] == 0
    for role, book in books.items():
        expected, _ = original.decompose(book.account, book.weights, opened, closed)
        pd.testing.assert_frame_equal(daily[role], expected)
    assert report["own_books"]["quartet"]["cached_drawdown_agreement_error"] <= 1e-10
    assert report["own_books"]["cached_trio"]["own_maximum_drawdown_episode"] is None


def test_same_episode_clocks_do_not_reset_different_existing_wealth():
    books, opened, closed = sources()
    _, report = rules.compare(books, opened, closed)
    view = report["common_clock_episode_views"][0]
    new, old = [view["both_stored_accounts_at_identical_clocks"][r] for r in rules.ROLES]
    assert new["start"] == old["start"] and new["end"] == old["end"]
    assert new["start_NAV_USD"] != old["start_NAV_USD"]
    assert old["actual_NAV_change_USD"] == 0
    assert old["recorded_fees_fraction_of_actual_net_loss"] is None
    assert not view["new_capital_or_matched_window_financial_gate"]
    assert not view["different_start_NAVs_are_equal_capital_counterfactuals"]
    assert (
        report["common_clock_episode_views"][1]["both_stored_accounts_at_identical_clocks"] is None
    )


def test_all_assets_and_grouped_marks_fees_reconcile_without_losing_offsets():
    books, opened, closed = sources()
    _, report = rules.compare(books, opened, closed)
    episode = report["own_books"]["quartet"]["own_maximum_drawdown_episode"]
    assert set(episode["asset_mark_PnL_USD"]) == set(rules.ASSETS)
    assert set(episode["asset_group_mark_PnL_USD"]) == set(rules.GROUPS)
    assert episode["asset_mark_PnL_USD"]["BIL"] == 0
    assert episode["asset_mark_PnL_USD"]["GLD"] > 0
    assert math.fsum(episode["asset_group_mark_PnL_USD"].values()) - episode[
        "recorded_fees_USD"
    ] == pytest.approx(episode["actual_NAV_change_USD"], abs=1e-6)
    assert math.fsum(episode["asset_group_mark_fraction_of_own_start_NAV"].values()) - episode[
        "recorded_fees_USD"
    ] / episode["start_NAV_USD"] == pytest.approx(
        episode["actual_stored_interval_change"], abs=1e-10
    )
    assert not episode["asset_group_is_strategy_expert"]
    assert not episode["mechanical_PnL_identity_is_expert_causal_alpha"]


def test_daily_prefix_causality_does_not_claim_full_window_episode_invariance():
    books, opened, closed = sources()
    full, _ = rules.compare(books, opened, closed)
    early_books = {}
    for role, book in books.items():
        account = book.account.iloc[:3]
        nav = np.r_[10000.0, account["equity"].to_numpy()]
        cached = float(max(1 - nav / np.maximum.accumulate(nav)))
        early_books[role] = rules.StoredBook(account, book.weights.iloc[:3], cached)
    early, _ = rules.compare(early_books, opened.iloc[:3], closed.iloc[:3])
    for role in rules.ROLES:
        pd.testing.assert_frame_equal(early[role], full[role].iloc[:3])


def test_callers_frames_and_saved_costs_are_never_mutated_or_removed():
    books, opened, closed = sources()
    before = copy.deepcopy(books)
    _, report = rules.compare(books, opened, closed)
    for role in rules.ROLES:
        pd.testing.assert_frame_equal(books[role].account, before[role].account)
        pd.testing.assert_frame_equal(books[role].weights, before[role].weights)
    assert report["own_books"]["quartet"]["own_maximum_drawdown_episode"]["recorded_fees_USD"] > 0
    assert not report["fee_free_or_expert_causal_counterfactual_claimed"]
    assert not report["source_CAGR_Sharpe_or_qualification_recomputed"]


@pytest.mark.parametrize(
    "damage",
    ["role", "calendar", "asset_order", "cash", "NAV", "price", "cached_DD", "bool_DD"],
)
def test_coordinate_or_cached_money_evidence_failure_cannot_be_laundered(damage):
    books, opened, closed = sources()
    book = books["quartet"]
    if damage == "role":
        books["foreign"] = books.pop("cached_trio")
    elif damage == "calendar":
        books["cached_trio"] = replace(
            books["cached_trio"], account=books["cached_trio"].account.iloc[:-1]
        )
    elif damage == "asset_order":
        closed = closed.loc[:, list(reversed(rules.ASSETS))]
    elif damage == "cash":
        book.account.iloc[1, book.account.columns.get_loc("cash")] += 0.01
    elif damage == "NAV":
        book.account.iloc[1, book.account.columns.get_loc("equity")] += 1
    elif damage == "price":
        opened.iloc[1, 0] = np.nan
    elif damage == "cached_DD":
        books["quartet"] = replace(book, cached_max_drawdown=book.cached_max_drawdown + 0.01)
    else:
        books["quartet"] = replace(book, cached_max_drawdown=True)
    with pytest.raises(QuantError):
        rules.compare(books, opened, closed)


@pytest.mark.parametrize(
    "key,value",
    [
        ("monetary_reconciliation_tolerance_USD", 0.01),
        ("cached_drawdown_agreement_tolerance", 0.001),
        ("peak_convention", "first_observed_NAV"),
        ("existing_full_books", 8),
        ("new_strategy_target_account_BT_or_financial_trials", 1),
        ("source_CAGR_Sharpe_or_qualification_recomputed", True),
        ("fee_free_or_expert_causal_counterfactual_claimed", True),
    ],
)
def test_no_tolerance_clock_trial_or_causal_interpretation_rescue(key, value):
    with pytest.raises(QuantError):
        rules.validate({**read_json(rules.POLICY), key: value})


def test_second_actual_diagnosis_is_terminal_before_reading_any_source(tmp_path):
    (tmp_path / "analysis-attempt-v1.json").write_text("{}")
    real = FunctionType(
        study.analyze.__code__,
        {
            **study.analyze.__globals__,
            "FOLDER": tmp_path,
            "RESULTS": tmp_path / "results.json",
            "lifecycle": SimpleNamespace(runtime_guard=lambda: None),
        },
    )
    with pytest.raises(QuantError):
        real()


def test_source_manifest_has_all_paired_controls_and_unchanged_original_artifacts():
    inputs = read_json(study.base.checked_reference(read_json(rules.POLICY)["source_inputs"]))
    assert inputs["paired_cases"] == 8 and inputs["existing_full_books"] == 16
    assert tuple(inputs["cases"]) == tuple(source["id"] for source in rules.quartet.CANDIDATES)
    for new_id, old_id in zip(
        (c["id"] for c in rules.quartet.CANDIDATES),
        (c["id"] for c in rules.controls.CANDIDATES),
        strict=True,
    ):
        assert set(inputs["cases"][new_id]) == set(rules.PATHS)
        for case in inputs["cases"][new_id].values():
            assert case["quartet"]["candidate_id"] == new_id
            assert case["cached_trio"]["candidate_id"] == old_id
            assert all(len(v["all_original_artifacts"]) == 5 for v in case.values())
    paths = {r["path"] for r in study.dependencies()}
    assert set(study.OWN).issubset(paths)
    assert "src/us_quant/saved_main_book_decomposition_v1.py" in paths
