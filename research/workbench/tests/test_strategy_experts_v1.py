from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.strategy_experts_v1 import (
    ArchivedDenseExpert,
    ArchivedMonthlyExpert,
    ExpertIdentity,
)

IDENTITY = ExpertIdentity("complete_policy", "synthetic_mechanism", "2026-10-29T13:25:29Z")
UNIVERSE = ("growth", "defense", "other")


def data():
    index = sessions("2020-01-02", "2020-05-29")
    target = pd.DataFrame({"growth": 0.6, "defense": 0.38}, index=index)
    returns = np.sin(np.arange(len(index)) / 3) * 0.002
    equity = 10000 * np.cumprod(1 + returns)
    shadow = pd.DataFrame(
        {
            "equity": equity,
            "return": returns,
            "risk_free": 0.0001,
            "cash": equity * 0.02,
            "gross_exposure": 0.98,
            "cost": 0.0,
            "orders": 0,
        },
        index=index,
    )
    return target, shadow


def test_complete_expert_observation_is_separate_from_main_capital():
    targets, shadow = data()
    expert = ArchivedDenseExpert(IDENTITY, targets, shadow, UNIVERSE)
    observation = expert.observe(targets.index[20], 10)
    assert observation.identity == IDENTITY
    assert observation.historical_counterfactual
    assert observation.target.sum() == pytest.approx(0.98)
    assert observation.target["other"] == 0
    assert observation.score_mature and len(observation.net_returns) == 10
    assert observation.net_returns.index[-1] == targets.index[20]
    observation.target["growth"] = 0
    assert expert.observe(targets.index[20], 10).target["growth"] == 0.6


def test_unmatured_scores_are_not_zero_padded_or_marked_successful():
    targets, shadow = data()
    observation = ArchivedDenseExpert(IDENTITY, targets, shadow, UNIVERSE).observe(
        targets.index[3], 10
    )
    assert observation.target is not None
    assert not observation.score_mature
    assert len(observation.net_returns) == 4


def test_monthly_adapter_carries_only_known_recommendations_and_records_actual_introduction():
    targets, shadow = data()
    targets.loc[[not is_month_end(d) for d in targets.index]] = np.nan
    expert = ArchivedMonthlyExpert(IDENTITY, targets, shadow, UNIVERSE)
    earlier = expert.observe(pd.Timestamp("2020-01-15"), 10)
    assert earlier.target is None and earlier.target_known_at is None
    known = expert.observe(pd.Timestamp("2020-02-14"), 10)
    assert known.target_known_at == pd.Timestamp("2020-01-31")
    assert known.identity.introduced_at.startswith("2026")
    assert known.historical_counterfactual


@pytest.mark.parametrize("kind", ["dense", "monthly"])
def test_missing_post_warmup_recommendation_is_an_error_not_safe_carry(kind):
    targets, shadow = data()
    if kind == "monthly":
        targets.loc[[not is_month_end(d) for d in targets.index]] = np.nan
        targets.loc[pd.Timestamp("2020-03-31")] = np.nan
        adapter = ArchivedMonthlyExpert
    else:
        targets.iloc[20] = np.nan
        adapter = ArchivedDenseExpert
    with pytest.raises(QuantError):
        adapter(IDENTITY, targets, shadow, UNIVERSE)


def test_missing_shadow_session_and_nonnet_accounting_are_not_accepted():
    targets, shadow = data()
    with pytest.raises(QuantError, match="complete"):
        ArchivedDenseExpert(IDENTITY, targets, shadow.drop(shadow.index[10]), UNIVERSE)
    shadow.loc[shadow.index[10], "return"] += 0.01
    with pytest.raises(QuantError, match="reconcile"):
        ArchivedDenseExpert(IDENTITY, targets, shadow, UNIVERSE)


@pytest.mark.parametrize("damage", ["negative", "leveraged", "partial", "unsupported"])
def test_invalid_exposure_cannot_be_laundered_into_an_expert(damage):
    targets, shadow = data()
    if damage == "negative":
        targets.iloc[3] = [-0.1, 1.08]
    elif damage == "leveraged":
        shadow.iloc[3, shadow.columns.get_loc("gross_exposure")] = 1.01
    elif damage == "partial":
        targets.iloc[3, 0] = np.nan
    else:
        targets["unadmitted_asset"] = 0.0
    with pytest.raises(QuantError):
        ArchivedDenseExpert(IDENTITY, targets, shadow, UNIVERSE)


def test_prefix_and_valid_future_changes_do_not_change_an_earlier_observation():
    targets, shadow = data()
    end = pd.Timestamp("2020-03-31")
    full = ArchivedDenseExpert(IDENTITY, targets, shadow, UNIVERSE).observe(end, 20)
    prefix = ArchivedDenseExpert(IDENTITY, targets.loc[:end], shadow.loc[:end], UNIVERSE).observe(
        end, 20
    )
    targets.loc[targets.index > end] = [0.2, 0.78]
    changed = ArchivedDenseExpert(IDENTITY, targets, shadow, UNIVERSE).observe(end, 20)
    for other in (prefix, changed):
        pd.testing.assert_series_equal(full.target, other.target)
        pd.testing.assert_series_equal(full.net_returns, other.net_returns)
        pd.testing.assert_series_equal(full.risk_free, other.risk_free)


def test_requests_outside_covered_history_do_not_receive_a_live_default():
    targets, shadow = data()
    expert = ArchivedDenseExpert(IDENTITY, targets, shadow, UNIVERSE)
    with pytest.raises(QuantError, match="covered"):
        expert.observe(pd.Timestamp("2026-10-29"), 20)
