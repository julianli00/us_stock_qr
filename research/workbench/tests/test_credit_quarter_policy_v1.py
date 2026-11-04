import copy

import numpy as np
import pandas as pd
import pytest

from us_quant import credit_quarter_policy_v1 as law
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.storage import read_json


def market():
    index = sessions("2015-08-10", "2017-01-31")
    close = pd.DataFrame(100.0, index=index, columns=law.UNIVERSE)
    credit = pd.DataFrame(
        {
            "relative_total_return": [np.nan] * 64
            + [0.1 if i % 2 else -0.1 for i in range(len(index) - 64)],
            "history_mature": np.arange(len(index)) >= 64,
            "source_start_session": [pd.NaT] * 64 + list(index[:-64]),
            "source_end_session": [pd.NaT] + list(index[:-1]),
        },
        index=index,
    )
    return close, credit


def test_actual_declaration_is_one_fixed_complete_policy_not_pool_or_asset_expansion():
    policy = read_json(law.POLICY)
    law.validate(policy)
    assert policy["candidates"] == [law.CANDIDATE]
    assert policy["new_traded_assets"] == 0
    assert policy["methodology"]["existing_pool_changed"] is False


@pytest.mark.parametrize("value", [np.nextafter(0.0, 1.0), 0.1, 10.0])
def test_strictly_positive_credit_gives_all_four_equal_core_exposures(value):
    row, selected = law.decision(value)
    assert selected == list(law.CORE)
    assert row.loc[list(law.CORE)].tolist() == [0.245] * 4
    assert row.drop(list(law.CORE)).eq(0).all()
    assert abs(row.sum() - 0.98) < 1e-12


@pytest.mark.parametrize("value", [0.0, -0.0, np.nextafter(0.0, -1.0), -0.5])
def test_zero_and_negative_credit_are_the_declared_gold_bill_state(value):
    row, selected = law.decision(value)
    assert selected == ["GLD", "BIL"]
    assert row.loc[selected].tolist() == [0.49, 0.49]
    assert row.drop(selected).eq(0).all()


@pytest.mark.parametrize("value", [np.nan, np.inf, None, True, "positive"])
def test_invalid_credit_is_not_a_success_shaped_defensive_default(value):
    with pytest.raises(QuantError, match="missing credit"):
        law.decision(value)


def test_monthly_native_history_has_only_real_monthends_and_explicit_cold_start():
    close, credit = market()
    records = []
    targets, audits = law.build_targets(close, credit, read_json(law.POLICY), retain=records.append)
    active = targets.dropna(how="all")
    assert active.index[0] == pd.Timestamp("2015-11-30")
    assert all(law.is_month_end(day) for day in active.index)
    assert len(records) == 2 * len(audits) == 2 * len(active)
    assert targets.iloc[:64].isna().all().all()
    assert not (active < 0).any().any()
    assert np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
    assert {"HYG", "LQD"}.isdisjoint(targets.columns)


def test_prefix_and_future_credit_changes_cannot_change_past_policy_targets():
    close, credit = market()
    policy = read_json(law.POLICY)
    targets, audits = law.build_targets(close, credit, policy)
    day = pd.Timestamp("2016-09-30")
    earlier, early_audits = law.build_targets(close.loc[:day], credit.loc[:day], policy)
    assert targets.loc[:day].equals(earlier)
    assert early_audits == [a for a in audits if a["decision_session"] <= str(day.date())]
    changed = credit.copy()
    changed.loc[changed.index > day, "relative_total_return"] *= -1
    later, _ = law.build_targets(close, changed, policy)
    assert targets.loc[:day].equals(later.loc[:day])


@pytest.mark.parametrize("field", ["credit_lookback_intervals", "actual_weight_trade_band"])
def test_declared_failed_knob_cannot_change(field):
    policy = copy.deepcopy(read_json(law.POLICY))
    policy[field] += 1
    with pytest.raises(QuantError, match="single complete"):
        law.validate(policy)


@pytest.mark.parametrize("failure", ["missing", "cold", "future"])
def test_missing_or_retimed_credit_is_rejected(failure):
    close, credit = market()
    if failure == "missing":
        credit = credit.drop(credit.index[-1])
    elif failure == "cold":
        credit.loc[credit.index[100], "history_mature"] = False
    else:
        credit.loc[credit.index[100], "source_end_session"] = credit.index[100]
    with pytest.raises(QuantError):
        law.build_targets(close, credit, read_json(law.POLICY))
