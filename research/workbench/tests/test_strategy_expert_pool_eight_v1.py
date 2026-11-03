from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from test_strategy_expert_pool_v5 import archived_experts as original

from us_quant import strategy_expert_pool_eight_v1 as rules
from us_quant.calendar import is_month_end
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import ArchivedMonthlyExpert, ExpertIdentity


def archived_experts(changed_after=None, missing_initial=False):
    existing, calendar, assets = original(changed_after=changed_after)
    dates = calendar[calendar >= "2016-10-06"]
    values = 0.0002 + 0.002 * np.cos(np.arange(len(dates)) / 17)
    if changed_after is not None:
        values[dates > changed_after] += 0.01
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
    active = pd.DatetimeIndex([day for day in calendar if is_month_end(day)])
    if missing_initial:
        active = active[active >= "2016-10-31"]
    targets.loc[active] = [0.196, 0.196, 0.294, 0.294]
    if changed_after is not None:
        targets.loc[active[active > changed_after]] = [0.294, 0.294, 0.196, 0.196]
    new = ArchivedMonthlyExpert(
        ExpertIdentity(rules.EXPERT_IDS[-1], rules.MECHANISMS[-1], "2026-11-03T13:27:35.710425Z"),
        targets,
        shadow,
        assets,
    )
    return (*existing, new), calendar, assets


def test_eight_inclusion_reuses_exact_unchanged_original_pair_and_combine_functions():
    experts, _, _ = archived_experts()
    observed = [e.observe(pd.Timestamp("2017-04-28"), 126) for e in experts]
    assert rules.choose_pair is rules.prior.choose_pair
    assert rules.combine is rules.prior.combine
    new, audit = rules.choose_pair(observed, np.full(8, 1 / 8))
    old, expected = rules.prior.choose_pair(observed, np.full(8, 1 / 8))
    np.testing.assert_array_equal(new, old)
    assert audit == expected and len(audit["pairs"]) == 28


def test_all_twenty_eight_pairs_keep_risk_cost_records_and_one_underlying_target():
    experts, _, _ = archived_experts()
    observed = [e.observe(pd.Timestamp("2017-04-28"), 126) for e in experts]
    selected, audit = rules.choose_pair(observed, np.full(8, 1 / 8))
    assert len(audit["pairs"]) == 28 and (selected > 0).sum() == 2
    assert audit["train_end"] < audit["validation_start"] <= audit["validation_end"]
    assert rules.combine(observed, np.full(8, 1 / 8)).sum() == pytest.approx(0.98)
    assert all(
        row["cost_proxy_is_actual_main_account_cost"] is False
        for row in audit["pairs"]
        if "selection_score_key" in row
    )


def test_unmatured_states_and_prefix_causality_preserve_every_formal_decision():
    experts, calendar, assets = archived_experts()
    full, audits = rules.build_targets(experts, calendar, assets, read_json(rules.POLICY))
    assert audits[0]["decision_session"] == "2016-10-05"
    assert audits[0]["equal_expert_weights"] == [1 / 8] * 8
    assert sum(a["selection"]["reason"] == "unscored_initial_equal_weights" for a in audits) == 7
    assert (
        next(a["decision_session"] for a in audits if a["selection"]["scores_available"])
        == "2017-04-28"
    )
    early, first = rules.build_targets(
        experts, calendar[calendar <= "2017-04-28"], assets, read_json(rules.POLICY)
    )
    for name in full:
        pd.testing.assert_frame_equal(early[name], full[name].loc[:"2017-04-28"])
    assert first == [a for a in audits if a["decision_session"] <= "2017-04-28"]


def test_future_target_and_shadow_changes_do_not_change_eight_expert_past():
    before, calendar, assets = archived_experts()
    after, _, _ = archived_experts(changed_after=pd.Timestamp("2017-04-28"))
    first, a = rules.build_targets(before, calendar, assets, read_json(rules.POLICY))
    later, b = rules.build_targets(after, calendar, assets, read_json(rules.POLICY))
    for name in first:
        pd.testing.assert_frame_equal(
            first[name].loc[:"2017-04-28"], later[name].loc[:"2017-04-28"]
        )
    assert [r for r in a if r["decision_session"] <= "2017-04-28"] == [
        r for r in b if r["decision_session"] <= "2017-04-28"
    ]


def test_actual_inputs_survive_a_failed_new_pair_decision(monkeypatch):
    experts, calendar, assets = archived_experts()
    records = []

    def failed(*args):
        raise QuantError("Synthetic fixed pair preparation failure")

    monkeypatch.setattr(rules, "choose_pair", failed)
    with pytest.raises(QuantError, match="Synthetic"):
        rules.build_targets(
            experts, calendar, assets, read_json(rules.POLICY), retain=records.append
        )
    assert len(records) == 1 and len(records[0]["observations"]) == 8
    assert records[0]["previous_weights_binary64_hex"] == [float(1 / 8).hex()] * 8


def test_missing_new_initial_recommendation_is_never_fabricated():
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
        ("decision_information_clock", "at_16_00_ET"),
    ],
)
def test_no_window_cost_band_clock_or_selection_rescue(key, value):
    policy = copy.deepcopy(read_json(rules.POLICY))
    policy[key] = value
    with pytest.raises(QuantError, match="Only declared eight-source"):
        rules.validate(policy)
