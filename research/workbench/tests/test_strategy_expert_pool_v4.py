from __future__ import annotations

import runpy

import numpy as np
import pandas as pd
import pytest

from us_quant import strategy_expert_pool_v3 as prior
from us_quant import strategy_expert_pool_v4 as rules
from us_quant.calendar import is_month_end
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import ArchivedMonthlyExpert, ExpertIdentity


def archived_experts(changed_after=None, missing_defensive_initial=False):
    helper = runpy.run_path(str(rules.ROOT / "tests/test_strategy_expert_pool_v3.py"))
    existing, calendar, assets = helper["archived_experts"](changed_after=changed_after)
    dates = calendar[calendar >= "2016-10-06"]
    values = 0.0002 + 0.003 * np.sin(np.arange(len(dates)) / 7)
    if changed_after is not None:
        values[dates > changed_after] -= 0.01
    nav = 10000 * np.cumprod(1 + values)
    shadow = pd.DataFrame(
        {
            "equity": nav,
            "return": values,
            "risk_free": 0.0001,
            "cash": nav * 0.02,
            "gross_exposure": 0.98,
            "cost": 0.0,
            "orders": 0,
        },
        index=dates,
    )
    targets = pd.DataFrame(np.nan, index=calendar, columns=assets)
    active = pd.DatetimeIndex([d for d in calendar if is_month_end(d)])
    if missing_defensive_initial:
        active = active[active >= "2016-10-31"]
    row = np.array([0.196, 0.196, 0.294, 0.294])
    targets.loc[active] = row
    if changed_after is not None:
        targets.loc[active[active > changed_after]] = np.roll(row, 1)
    defensive = ArchivedMonthlyExpert(
        ExpertIdentity(rules.EXPERT_IDS[5], rules.MECHANISMS[5], "2026-10-30T03:30:18Z"),
        targets,
        shadow,
        assets,
    )
    return (*existing, defensive), calendar, assets


def test_unchanged_pair_law_accepts_six_without_retuning():
    experts, _, _ = archived_experts()
    observed = [e.observe(pd.Timestamp("2017-04-28"), 126) for e in experts]
    weights, audit = rules.choose_pair(observed, np.full(6, 1 / 6))
    expected, old = prior.choose_pair(observed, np.full(6, 1 / 6))
    assert rules.choose_pair is prior.choose_pair and rules.combine is prior.combine
    np.testing.assert_array_equal(weights, expected)
    assert audit == old


def test_six_experts_use_fifteen_pairs_and_one_capital_target():
    experts, _, _ = archived_experts()
    observed = [e.observe(pd.Timestamp("2017-04-28"), 126) for e in experts]
    weights, audit = rules.choose_pair(observed, np.full(6, 1 / 6))
    assert len(audit["pairs"]) == 15 and (weights > 0).sum() == 2
    assert audit["train_end"] < audit["validation_start"] <= audit["validation_end"]
    for pair in audit["pairs"]:
        nonzero = [v for v in pair["training_marginal_variance_contributions"] if v]
        assert nonzero[0] == pytest.approx(nonzero[1], rel=1e-12)
        assert pair["cost_proxy_is_actual_main_account_cost"] is False
    target = rules.combine(observed, np.full(6, 1 / 6))
    assert target.sum() == pytest.approx(0.98)
    assert target.sum() * 10000 == pytest.approx(9800)
    with pytest.raises(QuantError, match="unit-sum"):
        rules.combine(observed, np.ones(6))


def test_actual_interface_prefixes_keep_initial_and_maturity():
    experts, calendar, assets = archived_experts()
    full, audits = rules.build_targets(experts, calendar, assets, read_json(rules.POLICY))
    assert audits[0]["decision_session"] == "2016-10-05"
    assert audits[0]["decision_role"] == "initial_capital"
    assert audits[0]["expert_target_known_at"][5] == "2016-09-30"
    assert sum(a["selection"]["reason"] == "unscored_initial_equal_weights" for a in audits) == 7
    assert (
        next(a["decision_session"] for a in audits if a["selection"]["scores_available"])
        == "2017-04-28"
    )
    assert all(t.ffill().loc["2016-10-05":].notna().all().all() for t in full.values())
    for end in ("2016-10-05", "2017-04-28", "2017-10-18"):
        early, earlier = rules.build_targets(
            experts, calendar[calendar <= end], assets, read_json(rules.POLICY)
        )
        for name in full:
            pd.testing.assert_frame_equal(early[name], full[name].loc[:end])
        assert earlier == [a for a in audits if a["decision_session"] <= end]


def test_future_target_and_shadow_changes_do_not_change_past():
    experts, calendar, assets = archived_experts()
    changed, _, _ = archived_experts(changed_after=pd.Timestamp("2017-04-28"))
    before, audits = rules.build_targets(experts, calendar, assets, read_json(rules.POLICY))
    after, changed_audits = rules.build_targets(changed, calendar, assets, read_json(rules.POLICY))
    for name in before:
        pd.testing.assert_frame_equal(
            before[name].loc[:"2017-04-28"], after[name].loc[:"2017-04-28"]
        )
    assert [a for a in audits if a["decision_session"] <= "2017-04-28"] == [
        a for a in changed_audits if a["decision_session"] <= "2017-04-28"
    ]


def test_missing_initial_defensive_recommendation_is_not_fabricated_or_dropped():
    experts, calendar, assets = archived_experts(missing_defensive_initial=True)
    with pytest.raises(QuantError, match="missing post-start"):
        rules.build_targets(experts, calendar, assets, read_json(rules.POLICY))


@pytest.mark.parametrize(
    "key,value",
    [
        ("lookback_sessions", 252),
        ("training_sessions", 63),
        ("validation_sessions", 21),
        ("selected_experts", 3),
        ("switch_cost_bps_per_side", 0.0),
        ("actual_weight_trade_band", 0.01),
        ("initial_decision_session", "2016-09-30"),
    ],
)
def test_controlled_inclusion_cannot_change_existing_parameters(key, value):
    policy = read_json(rules.POLICY)
    policy[key] = value
    with pytest.raises(QuantError, match="unchanged original"):
        rules.validate(policy)
