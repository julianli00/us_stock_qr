import math
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_quartet_trio_loss_accounting_v1 import sources

from us_quant import quartet_trio_loss_accounting_v1 as original
from us_quant import quartet_trio_loss_accounting_v2 as rules
from us_quant import ssd_quartet_trio_loss_accounting_v2 as study
from us_quant.config import QuantError
from us_quant.storage import read_json


def reduction_example():
    dates = pd.bdate_range("2020-01-02", periods=3)
    values = np.full((3, len(rules.ASSETS)), 0.1, order="F")
    scalar = np.array([math.fsum(float(v) for v in row) for row in values])
    fields = {
        "equity": 10000.0 + np.cumsum(scalar),
        "cash": np.full(3, 5000.0),
        "cost": np.zeros(3),
        "orders": np.zeros(3),
        "observed_net_change": scalar,
        rules.AUXILIARY: values.sum(axis=1),
    }
    for i, symbol in enumerate(rules.ASSETS):
        fields["overnight::" + symbol] = values[:, i]
        fields["intraday::" + symbol] = np.zeros(3)
    full = pd.DataFrame(fields, index=dates)
    prefix = full.iloc[:1].copy()
    prefix[rules.AUXILIARY] = values[:1].copy(order="F").sum(axis=1)
    return full, prefix


def test_synthetic_shape_reduction_reproduces_failed_V1_without_private_source_values():
    full, prefix = reduction_example()
    original_full, original_prefix = full.copy(), prefix.copy()
    assert (
        float(full[rules.AUXILIARY].iloc[0]).hex() != float(prefix[rules.AUXILIARY].iloc[0]).hex()
    )
    assert not prefix.equals(full.iloc[:1])
    result = rules.check_prefix(full, prefix)
    assert result["primitive_and_canonical_scalar_prefix_bit_exact"]
    assert result["auxiliary_NumPy_vector_bit_different_values"] == 1
    assert result["maximum_auxiliary_NumPy_prefix_difference_USD"] > 0
    assert result["original_V1_all_column_prefix_comparison"] is False
    assert not result["original_monetary_and_cached_DD_limits_changed"]
    pd.testing.assert_frame_equal(full, original_full)
    pd.testing.assert_frame_equal(prefix, original_prefix)


def test_canonical_view_retains_every_primitive_and_only_classifies_the_auxiliary_as_noncanonical():
    full, _ = reduction_example()
    view = rules.canonical_prefix_view(full)
    assert set(view) == (set(full) - {rules.AUXILIARY}) | {rules.CANONICAL}
    for name in set(full) - {rules.AUXILIARY}:
        pd.testing.assert_series_equal(view[name], full[name])
    assert view[rules.CANONICAL].iloc[0] == math.fsum([0.1] * 14)


@pytest.mark.parametrize(
    "column",
    ["equity", "cash", "cost", "orders", "observed_net_change", "overnight::SPY", "intraday::GLD"],
)
def test_even_one_ULP_primitive_change_is_not_accepted_under_original_money_tolerance(column):
    full, prefix = reduction_example()
    prefix.loc[prefix.index[0], column] = np.nextafter(float(prefix[column].iloc[0]), np.inf)
    with pytest.raises(QuantError):
        rules.check_prefix(full, prefix)


@pytest.mark.parametrize(
    "damage", ["large_auxiliary", "large_observed", "nonfinite", "missing", "date"]
)
def test_bad_identity_or_missing_fields_cannot_be_hidden_by_the_new_Interface(damage):
    full, prefix = reduction_example()
    if damage == "large_auxiliary":
        prefix.loc[prefix.index[0], rules.AUXILIARY] += 0.01
    elif damage == "large_observed":
        prefix.loc[prefix.index[0], "observed_net_change"] += 0.01
    elif damage == "nonfinite":
        prefix.loc[prefix.index[0], "cash"] = np.nan
    elif damage == "missing":
        prefix = prefix.drop(columns="intraday::GLD")
    else:
        prefix.index = prefix.index + pd.Timedelta(days=1)
    with pytest.raises(QuantError):
        rules.check_prefix(full, prefix)


def test_original_ledger_episode_and_group_Implementation_are_identical_objects():
    assert rules.compare is original.compare
    assert rules.StoredBook is original.StoredBook
    assert rules.marks.decompose is original.marks.decompose
    assert rules.marks.MONEY_TOLERANCE == 1e-6
    assert rules.marks.METRIC_TOLERANCE == 1e-10
    books, opened, closed = sources()
    daily, report = rules.compare(books, opened, closed)
    for role, book in books.items():
        for stop in (1, 3, 5):
            early, _ = rules.marks.decompose(
                book.account.iloc[:stop],
                book.weights.iloc[:stop],
                opened.iloc[:stop],
                closed.iloc[:stop],
            )
            assert rules.check_prefix(daily[role], early)[
                "primitive_and_canonical_scalar_prefix_bit_exact"
            ]
    assert not report["source_CAGR_Sharpe_or_qualification_recomputed"]


def test_projection_future_rows_do_not_change_any_previous_canonical_value():
    full, _ = reduction_example()
    before = rules.canonical_prefix_view(full)
    changed = full.copy()
    changed.loc[changed.index[-1], "overnight::SPY"] += 0.001
    changed.loc[changed.index[-1], "observed_net_change"] += 0.001
    changed.loc[changed.index[-1], rules.AUXILIARY] += 0.001
    after = rules.canonical_prefix_view(changed)
    pd.testing.assert_frame_equal(before.iloc[:-1], after.iloc[:-1])


@pytest.mark.parametrize(
    "key,value",
    [
        ("only_noncanonical_auxiliary_column", "cash"),
        ("original_V1_all_column_prefix_contract_remains_failed", False),
        ("original_monetary_cash_and_cached_DD_limits_unchanged", False),
        ("completed_V1_case_arithmetic_reused_without_reexecution", False),
        ("monetary_reconciliation_tolerance_USD", 0.01),
        ("cached_drawdown_agreement_tolerance", 0.001),
        ("new_strategy_target_account_BT_or_financial_trials", 1),
    ],
)
def test_canonical_contract_cannot_remove_economic_fields_or_change_original_limits(key, value):
    with pytest.raises(QuantError):
        rules.validate({**read_json(rules.POLICY), key: value})


def test_two_completed_V1_cases_and_stronger_proof_are_explicitly_reused_not_reexecuted():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    continuation = read_json(study.base.checked_reference(policy["continuation_inputs"]))
    assert continuation["original_V1_all_column_prefix_contract_remains_failed"]
    assert continuation["completed_V1_paired_cases_reused"] == 2
    assert continuation["completed_V1_book_diagnostics_reused"] == 4
    assert continuation["remaining_paired_cases_for_full_arithmetic"] == 6
    reused = [
        case
        for paths in continuation["V1_completed_case_reuse"].values()
        for case in paths.values()
    ]
    assert len(reused) == 2
    assert all(case["original_stronger_prefix_pass_reused_not_reexecuted"] for case in reused)
    assert all(set(case["daily"]) == set(rules.ROLES) for case in reused)


def test_second_actual_V2_attempt_is_terminal_before_any_source_read(tmp_path):
    (tmp_path / "analysis-attempt-v2.json").write_text("{}")
    real = FunctionType(
        study.analyze.__code__,
        {
            **study.analyze.__globals__,
            "FOLDER": tmp_path,
            "RESULTS": tmp_path / "results-v2.json",
            "lifecycle": SimpleNamespace(runtime_guard=lambda: None),
        },
    )
    with pytest.raises(QuantError):
        real()
