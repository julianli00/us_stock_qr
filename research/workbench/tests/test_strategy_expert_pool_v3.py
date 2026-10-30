from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_quant import strategy_expert_pool_v2 as prior
from us_quant import strategy_expert_pool_v3 as rules
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import (
    ArchivedDenseExpert,
    ArchivedMonthlyExpert,
    ExpertIdentity,
)


def archived_experts(changed_after=None, missing_macro_initial=False):
    calendar = sessions("2016-09-01", "2017-12-29")
    dates = calendar[calendar >= "2016-10-06"]
    assets = tuple("abcd")
    experts = []
    for i, identifier in enumerate(rules.EXPERT_IDS):
        phase = np.arange(len(dates))
        values = 0.0004 + (0.004 + i * 0.002) * np.sin(phase / (2 + i))
        if changed_after is not None:
            values[dates > changed_after] -= 0.02
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
        row = np.zeros(4)
        row[i % 4] = 0.98
        if i == 4:
            row = np.array([0.28, 0.2, 0.3, 0.2])
        if i in (2, 4):
            active = pd.DatetimeIndex([d for d in calendar if is_month_end(d)])
            if i == 4 and missing_macro_initial:
                active = active[active >= "2016-10-31"]
            adapter = ArchivedMonthlyExpert
        else:
            active = calendar[calendar >= "2016-10-05"] if i == 3 else calendar
            adapter = ArchivedDenseExpert
        targets.loc[active] = row
        if changed_after is not None:
            targets.loc[active[active > changed_after]] = np.roll(row, 1)
        experts.append(
            adapter(
                ExpertIdentity(identifier, rules.MECHANISMS[i], "2026-10-30T00:24:25Z"),
                targets,
                shadow,
                assets,
            )
        )
    return tuple(experts), calendar, assets


def test_unchanged_pair_law_accepts_five_without_retuning():
    experts, _, _ = archived_experts()
    observed = [e.observe(pd.Timestamp("2017-04-28"), 126) for e in experts]
    weights, audit = rules.choose_pair(observed, np.full(5, 0.2))
    expected, old = prior.choose_pair(observed, np.full(5, 0.2))
    assert rules.choose_pair is prior.choose_pair and rules.combine is prior.combine
    np.testing.assert_array_equal(weights, expected)
    assert audit == old


def test_five_experts_use_ten_pairs_and_keep_one_capital_target():
    experts, _, _ = archived_experts()
    observed = [e.observe(pd.Timestamp("2017-04-28"), 126) for e in experts]
    weights, audit = rules.choose_pair(observed, np.full(5, 0.2))
    assert len(audit["pairs"]) == 10 and (weights > 0).sum() == 2
    assert audit["train_end"] < audit["validation_start"] <= audit["validation_end"]
    for pair in audit["pairs"]:
        contribution = [v for v in pair["training_marginal_variance_contributions"] if v]
        assert contribution[0] == pytest.approx(contribution[1], rel=1e-12)
        assert pair["cost_proxy_is_actual_main_account_cost"] is False
    target = rules.combine(observed, np.full(5, 0.2))
    assert target.sum() == pytest.approx(0.98)
    assert target.sum() * 10000 == pytest.approx(9800)
    with pytest.raises(QuantError, match="unit-sum"):
        rules.combine(observed, np.ones(5))


def test_actual_interface_prefixes_keep_initial_and_maturity():
    experts, calendar, assets = archived_experts()
    full, audits = rules.build_targets(experts, calendar, assets, read_json(rules.POLICY))
    assert audits[0]["decision_session"] == "2016-10-05"
    assert audits[0]["decision_role"] == "initial_capital"
    assert sum(a["selection"]["reason"] == "unscored_initial_equal_weights" for a in audits) == 7
    assert next(a["decision_session"] for a in audits if a["selection"]["scores_available"]) == (
        "2017-04-28"
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


def test_missing_initial_macro_recommendation_is_not_fabricated_or_dropped():
    experts, calendar, assets = archived_experts(missing_macro_initial=True)
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
        ("initial_decision_session", "2016-09-30"),
    ],
)
def test_controlled_extension_cannot_change_existing_parameters(key, value):
    policy = read_json(rules.POLICY)
    policy[key] = value
    with pytest.raises(QuantError, match="unchanged original"):
        rules.validate(policy)
