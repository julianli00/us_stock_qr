from copy import deepcopy
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from us_quant import saved_main_book_decomposition_v1 as law
from us_quant import ssd_mixed_six_loss_diagnosis_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json


def examples():
    dates = pd.bdate_range("2020-01-02", periods=5)
    opened = pd.DataFrame(
        [[100, 50], [105, 49], [99, 51], [95, 50], [110, 53]],
        index=dates,
        columns=["A", "B"],
        dtype=float,
    )
    closed = pd.DataFrame(
        [[104, 49], [100, 50], [94, 52], [108, 52], [111, 54]],
        index=dates,
        columns=opened.columns,
        dtype=float,
    )
    units = np.array([[30, 40], [20, 50], [20, 50], [35, 20], [35, 20]], dtype=float)
    fees = [3.0, 4.0, 0.0, 5.0, 0.0]
    previous, cash = [Fraction(0), Fraction(0)], Fraction(10000)
    equities, cashes = [], []
    for i, holdings in enumerate(units):
        cash -= sum(
            (Fraction(float(holdings[j])) - previous[j]) * Fraction(float(opened.iloc[i, j]))
            for j in range(2)
        ) + Fraction(fees[i])
        equity = cash + sum(
            Fraction(float(holdings[j])) * Fraction(float(closed.iloc[i, j])) for j in range(2)
        )
        equities.append(float(equity))
        cashes.append(float(cash))
        previous = [Fraction(float(v)) for v in holdings]
    weights = pd.DataFrame(
        units * closed.to_numpy() / np.array(equities)[:, None],
        index=dates,
        columns=opened.columns,
    )
    account = pd.DataFrame(
        {
            "equity": equities,
            "cash": cashes,
            "gross_exposure": weights.sum(axis=1),
            "cost": fees,
            "orders": [2, 2, 0, 2, 0],
        },
        index=dates,
    )
    return account, weights, opened, closed, units


def test_full_saved_book_matches_independent_fraction_cash_and_daily_pnl():
    account, weights, opened, closed, units = examples()
    daily, summary = law.decompose(account, weights, opened, closed)
    previous = np.vstack([np.zeros((1, 2)), units[:-1]])
    prior_close = np.vstack([opened.iloc[0].to_numpy(), closed.iloc[:-1].to_numpy()])
    for i in range(len(account)):
        expected = sum(
            Fraction(float(previous[i, j]))
            * (Fraction(float(opened.iloc[i, j])) - Fraction(float(prior_close[i, j])))
            + Fraction(float(units[i, j]))
            * (Fraction(float(closed.iloc[i, j])) - Fraction(float(opened.iloc[i, j])))
            for j in range(2)
        ) - Fraction(float(account["cost"].iloc[i]))
        assert abs(float(expected) - daily["reconstructed_net_change"].iloc[i]) < 1e-9
    assert summary["maximum_daily_NAV_identity_error_USD"] < 1e-9
    assert summary["maximum_daily_cash_flow_identity_error_USD"] < 1e-9
    episode = summary["own_maximum_drawdown_episode"]
    assert episode["actual_NAV_change_USD"] < 0
    assert episode["maximum_whole_interval_identity_error_USD"] < 1e-9
    assert set(episode["asset_mark_PnL_USD"]) == {"A", "B"}


@pytest.mark.parametrize(
    "kind",
    [
        "date",
        "duplicate_date",
        "asset",
        "NaN",
        "negative_weight",
        "negative_price",
        "cash",
        "cost",
        "gross",
        "fractional_order",
        "equity",
    ],
)
def test_invalid_or_tampered_saved_coordinate_or_monetary_evidence_fails(kind):
    account, weights, opened, closed, _ = examples()
    if kind == "date":
        weights.index = weights.index + pd.Timedelta(days=1)
    elif kind == "duplicate_date":
        account.index = pd.DatetimeIndex([account.index[0], *account.index[:-1]])
    elif kind == "asset":
        closed = closed.iloc[:, ::-1]
    elif kind == "NaN":
        weights.iloc[1, 0] = np.nan
    elif kind == "negative_weight":
        weights.iloc[0, 0] = -0.01
    elif kind == "negative_price":
        opened.iloc[1, 0] = -1
    elif kind == "cash":
        account.loc[account.index[1], "cash"] += 0.01
    elif kind == "cost":
        account.loc[account.index[1], "cost"] += 0.01
    elif kind == "gross":
        account.loc[account.index[1], "gross_exposure"] += 0.01
    elif kind == "fractional_order":
        account["orders"] = account["orders"].astype(float)
        account.loc[account.index[1], "orders"] = 0.5
    else:
        account.loc[account.index[1], "equity"] += 1
    with pytest.raises(QuantError):
        law.decompose(account, weights, opened, closed)


def test_initial_cost_loss_keeps_original_10000_peak_not_first_observed_nav():
    account, weights, opened, closed, _ = examples()
    account = account.iloc[:1].copy()
    account.loc[:, "equity"] = 9990.0
    account.loc[:, "cash"] = 9990.0
    account.loc[:, "gross_exposure"] = 0.0
    account.loc[:, "cost"] = 10.0
    weights = weights.iloc[:1].copy() * 0
    _, summary = law.decompose(account, weights, opened.iloc[:1], closed.iloc[:1])
    assert summary["initial_peak_USD"] == 10000
    assert summary["maximum_observed_drawdown"] == pytest.approx(0.001)
    episode = summary["own_maximum_drawdown_episode"]
    assert episode["start_position"] == -1
    assert episode["recorded_fees_USD"] == 10
    assert episode["recorded_fees_fraction_of_actual_net_loss"] == 1


def test_exact_equal_peaks_use_latest_preceding_peak_and_earliest_maximum_trough():
    dates = pd.bdate_range("2020-01-02", periods=4)
    opened = pd.DataFrame({"A": [100.0] * 4}, index=dates)
    closed = pd.DataFrame({"A": [100.0, 100.0, 90.0, 90.0]}, index=dates)
    equity = np.array([10000.0, 10000.0, 9500.0, 9500.0])
    weights = closed.mul(50).div(equity, axis=0)
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
    daily, summary = law.decompose(account, weights, opened, closed)
    episode = summary["own_maximum_drawdown_episode"]
    assert episode["start_position"] == 1 and episode["end_position"] == 2
    assert episode["actual_NAV_change_USD"] == -500
    assert law.interval(daily, 1, 3)["actual_NAV_change_USD"] == -500


def test_zero_drawdown_and_nonloss_fee_ratio_are_explicit_not_successful_imputations():
    dates = pd.bdate_range("2020-01-02", periods=2)
    price = pd.DataFrame({"A": [100.0, 100.0]}, index=dates)
    weights = price * 0
    account = pd.DataFrame(
        {"equity": 10000.0, "cash": 10000.0, "gross_exposure": 0.0, "cost": 0.0, "orders": 0},
        index=dates,
    )
    daily, summary = law.decompose(account, weights, price, price)
    assert summary["episode_status"] == "no_observed_drawdown"
    assert summary["own_maximum_drawdown_episode"] is None
    assert law.interval(daily, -1, 1)["recorded_fees_fraction_of_actual_net_loss"] is None


def test_future_saved_rows_do_not_change_prior_daily_reconstruction():
    account, weights, opened, closed, _ = examples()
    daily, _ = law.decompose(account, weights, opened, closed)
    early, _ = law.decompose(account.iloc[:3], weights.iloc[:3], opened.iloc[:3], closed.iloc[:3])
    assert early.equals(daily.iloc[:3])


@pytest.mark.parametrize("start,stop", [(-2, 2), (2, 2), (3, 1), (-1, 10), (0.5, 2), (False, 2)])
def test_matched_date_intervals_never_clip_fill_or_reinitialize(start, stop):
    account, weights, opened, closed, _ = examples()
    daily, _ = law.decompose(account, weights, opened, closed)
    with pytest.raises(QuantError):
        law.interval(daily, start, stop)


@pytest.mark.parametrize(
    "key,value",
    [
        ("initial_capital_USD", 9990),
        ("new_financial_evaluations", 1),
        ("monetary_reconciliation_tolerance_USD", 0.01),
        ("financial_criteria_or_global_pool_changed", True),
        ("source_selectors_targets_accounts_or_BT_replayed", True),
    ],
)
def test_no_capital_tolerance_criteria_or_replay_rescue(key, value):
    policy = deepcopy(read_json(study.POLICY))
    policy[key] = value
    with pytest.raises(QuantError):
        study.validate(policy)


def test_canonical_price_reader_and_complete_tested_inputs_are_in_dependency_closure():
    paths = {r["path"] for r in study.dependencies()}
    assert set(study.OWN).issubset(paths)
    assert "src/us_quant/research_program.py" in paths
    assert "src/us_quant/data.py" in paths
    assert "tests/conftest.py" in paths
