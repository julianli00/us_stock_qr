import numpy as np
import pandas as pd
import pytest

from us_quant import quintet_return_cost_bridge_v1 as rules
from us_quant import ssd_quintet_return_cost_bridge_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json


def book(length=200, fee=2.0):
    index = pd.bdate_range("2016-10-06", periods=length)
    nav = 10000 + np.arange(length) * 7.0 + np.sin(np.arange(length)) * 30
    previous = np.r_[10000.0, nav[:-1]]
    return pd.DataFrame(
        {
            "equity": nav,
            "return": nav / previous - 1,
            "cash": nav * 0.1,
            "gross_exposure": 0.9,
            "turnover": 0.01,
            "cost": fee,
            "orders": 1,
            "risk_free": 0.0001,
        },
        index=index,
    )


def test_original_zero_cost_identity_and_initial_capital():
    account = book(fee=0.0)
    daily, summary = rules.decompose(account)
    assert summary["initial_capital_USD"] == 10000
    assert summary["recorded_paid_costs_USD"] == 0
    assert (daily["recorded_current_cost_log_drag"] == 0).all()
    assert summary["net_log_component"] == summary["recorded_path_gross_log_component"]
    assert summary["fee_free_funded_policy_counterfactual_claimed"] is False
    assert "cagr" not in summary and "sharpe" not in summary


def test_recorded_positive_fees_and_both_paired_identities():
    account = book()
    daily, report = rules.compare(
        {
            "opportunity": account,
            "cached_volatility": book(fee=1.0),
        }
    )
    assert len(daily) == 2
    delta = report["opportunity_minus_cached_volatility"]
    assert delta["recorded_paid_costs_USD"] == len(account)
    assert delta["net_log_component"] == 0
    assert (
        abs(delta["recorded_path_gross_log_component"] - delta["recorded_current_cost_log_drag"])
        <= rules.TOLERANCE
    )
    assert report["fee_or_expert_causal_alpha_attribution_claimed"] is False


def test_exact_prefixes_and_changed_future_do_not_change_prior_rows():
    original = book()
    full, _ = rules.decompose(original)
    changed = original.copy()
    changed.iloc[126:, changed.columns.get_loc("cost")] = 30
    updated, _ = rules.decompose(changed)
    for stop in (1, 21, 126, 100, 200):
        early, _ = rules.decompose(original.iloc[:stop])
        assert rules.check_prefix(full, early)["every_primitive_and_scalar_prefix_bit_exact"]
    pd.testing.assert_frame_equal(full.iloc[:126], updated.iloc[:126])
    damaged = full.iloc[:21].copy()
    damaged.iloc[0, damaged.columns.get_loc("cash")] += 1e-8
    with pytest.raises(QuantError):
        rules.check_prefix(full, damaged)


def test_signed_zero_is_part_of_exact_primitive_prefix_contract():
    full, _ = rules.decompose(book(fee=0.0))
    changed = full.iloc[:21].copy()
    changed.iloc[0, changed.columns.get_loc("cost")] = -0.0
    assert changed.equals(full.iloc[:21])
    with pytest.raises(QuantError):
        rules.check_prefix(full, changed)


@pytest.mark.parametrize(
    "column,value",
    [
        ("equity", 0.0),
        ("return", -1.0),
        ("cash", -1.0),
        ("gross_exposure", 1.1),
        ("turnover", -0.01),
        ("cost", -0.01),
        ("orders", 0.5),
        ("risk_free", np.nan),
        ("return", 0.3),
    ],
)
def test_invalid_recorded_money_or_return_is_not_replaced(column, value):
    source = book()
    source.iloc[0, source.columns.get_loc(column)] = value
    with pytest.raises(QuantError):
        rules.decompose(source)


def test_missing_columns_dates_or_source_RF_are_errors():
    source = book()
    with pytest.raises(QuantError):
        rules.decompose(source.drop(columns="cost"))
    with pytest.raises(QuantError):
        rules.decompose(source.iloc[::-1])
    with pytest.raises(QuantError):
        rules.compare({"opportunity": source, "cached_volatility": source.iloc[1:]})
    study.validate_loaded_book(source, source.index, source["risk_free"])
    with pytest.raises(QuantError):
        study.validate_loaded_book(source, source.index[1:], source["risk_free"])
    with pytest.raises(QuantError):
        study.validate_loaded_book(source, source.index, source["risk_free"] + 1e-12)


@pytest.mark.parametrize(
    "key,value",
    [
        ("initial_capital_usd", 9999),
        ("existing_books", 8),
        ("paired_cases", 4),
        ("return_and_log_identity_tolerance", 1e-8),
        ("money_identity_tolerance_USD", 0.01),
        ("source_CAGR_Sharpe_or_qualification_recomputed", True),
        ("fee_free_funded_policy_counterfactual_claimed", True),
        ("fee_or_expert_causal_alpha_attribution_claimed", True),
        ("new_strategy_targets_accounts_BT_or_financial_trials", 1),
    ],
)
def test_frozen_bridge_scope_cannot_become_strategy_or_fee_counterfactual(key, value):
    with pytest.raises(QuantError):
        rules.validate({**read_json(rules.POLICY), key: value})
