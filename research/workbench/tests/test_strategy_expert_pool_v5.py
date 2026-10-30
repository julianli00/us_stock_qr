from __future__ import annotations

import runpy

import numpy as np
import pandas as pd
import pytest

from us_quant import strategy_expert_pool_v4 as prior
from us_quant import strategy_expert_pool_v5 as rules
from us_quant.calendar import is_month_end
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import ArchivedMonthlyExpert, ExpertIdentity


def archived_experts(changed_after=None, missing_initial=False):
    helper = runpy.run_path(str(rules.ROOT / "tests/test_strategy_expert_pool_v4.py"))
    existing, calendar, assets = helper["archived_experts"](changed_after=changed_after)
    dates = calendar[calendar >= "2016-10-06"]
    values = 0.00015 + 0.0035 * np.cos(np.arange(len(dates)) / 11)
    if changed_after is not None:
        values[dates > changed_after] += 0.015
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
    target = pd.DataFrame(np.nan, index=calendar, columns=assets)
    active = pd.DatetimeIndex([day for day in calendar if is_month_end(day)])
    if missing_initial:
        active = active[active >= "2016-10-31"]
    row = np.array([0.392, 0.294, 0.196, 0.098])
    target.loc[active] = row
    if changed_after is not None:
        target.loc[active[active > changed_after]] = np.roll(row, 2)
    option = ArchivedMonthlyExpert(
        ExpertIdentity(rules.EXPERT_IDS[6], rules.MECHANISMS[6], "2026-10-28T23:58:17.764688Z"),
        target,
        shadow,
        assets,
    )
    return (*existing, option), calendar, assets


def test_seven_inclusion_reuses_exact_unchanged_pair_law():
    experts, _, _ = archived_experts()
    observed = [e.observe(pd.Timestamp("2017-04-28"), 126) for e in experts]
    weights, audit = rules.choose_pair(observed, np.full(7, 1 / 7))
    expected, old = prior.choose_pair(observed, np.full(7, 1 / 7))
    assert rules.choose_pair is prior.choose_pair and rules.combine is prior.combine
    np.testing.assert_array_equal(weights, expected)
    assert audit == old


def test_all_twenty_one_pairs_and_one_main_target():
    experts, _, _ = archived_experts()
    observed = [e.observe(pd.Timestamp("2017-04-28"), 126) for e in experts]
    weights, audit = rules.choose_pair(observed, np.full(7, 1 / 7))
    assert len(audit["pairs"]) == 21 and (weights > 0).sum() == 2
    assert audit["train_end"] < audit["validation_start"] <= audit["validation_end"]
    for pair in audit["pairs"]:
        nonzero = [v for v in pair["training_marginal_variance_contributions"] if v]
        assert nonzero[0] == pytest.approx(nonzero[1], rel=1e-12)
        assert pair["cost_proxy_is_actual_main_account_cost"] is False
    target = rules.combine(observed, np.full(7, 1 / 7))
    assert target.sum() == pytest.approx(0.98)
    with pytest.raises(QuantError, match="unit-sum"):
        rules.combine(observed, np.ones(7))


def test_initial_maturity_and_prefix_causality():
    experts, calendar, assets = archived_experts()
    full, audits = rules.build_targets(experts, calendar, assets, read_json(rules.POLICY))
    assert audits[0]["decision_session"] == "2016-10-05"
    assert audits[0]["expert_target_known_at"][6] == "2016-09-30"
    assert audits[0]["decision_information_clock"] == rules.CLOCK
    assert sum(a["selection"]["reason"] == "unscored_initial_equal_weights" for a in audits) == 7
    assert (
        next(a["decision_session"] for a in audits if a["selection"]["scores_available"])
        == "2017-04-28"
    )
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
    after, later = rules.build_targets(changed, calendar, assets, read_json(rules.POLICY))
    for name in before:
        pd.testing.assert_frame_equal(
            before[name].loc[:"2017-04-28"], after[name].loc[:"2017-04-28"]
        )
    assert [a for a in audits if a["decision_session"] <= "2017-04-28"] == [
        a for a in later if a["decision_session"] <= "2017-04-28"
    ]


def test_actual_inputs_survive_a_failing_pair_decision(monkeypatch):
    experts, calendar, assets = archived_experts()
    records = []

    def blocked(*args):
        raise QuantError("Synthetic pair failure.")

    monkeypatch.setattr(rules, "choose_pair", blocked)
    with pytest.raises(QuantError, match="Synthetic"):
        rules.build_targets(
            experts, calendar, assets, read_json(rules.POLICY), retain=records.append
        )
    assert len(records) == 1 and records[0]["record"] == "decision_input"
    assert len(records[0]["observations"]) == 7
    assert records[0]["previous_weights_binary64_hex"] == [float(1 / 7).hex()] * 7
    assert all(not record["NET_returns"] for record in records[0]["observations"])
    assert records[0]["binary64_values_truncated_to_fixed_decimals"] is False


def test_missing_option_initial_recommendation_is_not_fabricated():
    experts, calendar, assets = archived_experts(missing_initial=True)
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
        ("decision_information_clock", "at_16_00_ET"),
    ],
)
def test_controlled_inclusion_cannot_retune_existing_parameters_or_clock(key, value):
    policy = read_json(rules.POLICY)
    policy[key] = value
    with pytest.raises(QuantError, match="unchanged pair law"):
        rules.validate(policy)
