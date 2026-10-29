from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from us_quant import strategy_expert_meta_v1 as original
from us_quant import strategy_expert_pool_v2 as rules
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import (
    ArchivedDenseExpert,
    ArchivedMonthlyExpert,
    ExpertIdentity,
    ExpertObservation,
)


def observations(n=4, mature=True):
    dates = sessions("2016-10-06", "2017-04-06")[-126:]
    selected = dates if mature else dates[:20]
    phase = np.arange(126)
    values = [
        0.0007 + 0.012 * np.sin(phase / 2),
        0.0003 + 0.003 * np.cos(phase / 3),
        0.0009 + 0.018 * np.sin(phase / 2 + 0.1),
        0.00035 + 0.002 * np.cos(phase / 7),
    ]
    return [
        ExpertObservation(
            ExpertIdentity(identifier, rules.MECHANISMS[i], "2026-10-29T17:21:25Z"),
            selected[-1],
            pd.Series([0.98 if k == i else 0.0 for k in range(n)], index=list("abcd")[:n]),
            selected[-1],
            pd.Series(values[i][: len(selected)], index=selected),
            pd.Series(0.0001, index=selected),
            mature,
        )
        for i, identifier in enumerate(rules.EXPERT_IDS[:n])
    ]


@pytest.mark.parametrize("mature", [False, True])
def test_old_three_expert_scoring_is_unchanged_on_synthetic_observations(mature):
    observed = observations(3, mature)
    before_weights, before = original.choose_pair(observed, np.full(3, 1 / 3))
    after_weights, after = rules.choose_pair(observed, np.full(3, 1 / 3))
    np.testing.assert_array_equal(before_weights, after_weights)
    assert before == {key: after[key] for key in before}


def test_four_experts_produce_six_pairs_with_the_same_risk_and_score_law():
    observed = observations()
    weights, audit = rules.choose_pair(observed, np.full(4, 0.25))
    assert len(audit["pairs"]) == 6 and (weights > 0).sum() == 2
    assert audit["train_end"] < audit["validation_start"] <= audit["validation_end"]
    for pair in audit["pairs"]:
        positive = [v for v in pair["training_marginal_variance_contributions"] if v]
        assert positive[0] == pytest.approx(positive[1], rel=1e-12)
        assert pair["selection_score"] == min(
            pair["train_net_risk_score"], pair["validation_net_risk_score"]
        )
        assert pair["selection_score_key"] == round(pair["selection_score"], 12)
        assert pair["cost_proxy_is_actual_main_account_cost"] is False


def test_unchanged_meta_weights_have_zero_incremental_switch_proxy():
    observed = observations()
    weights, audit = rules.choose_pair(observed, np.full(4, 0.25))
    _, again = rules.choose_pair(observed, weights)
    selected = next(p for p in again["pairs"] if p["pair"] == audit["selected_pair"])
    assert selected["estimated_incremental_switch_cost_fraction"] == 0
    assert selected["estimated_incremental_changed_assets"] == 0


def test_one_funded_target_is_not_four_shadow_capital_accounts():
    target = rules.combine(observations(), np.full(4, 0.25))
    np.testing.assert_allclose(target, np.full(4, 0.245))
    assert target.sum() == pytest.approx(0.98)
    assert target.sum() * 10000 == pytest.approx(9800)
    with pytest.raises(QuantError, match="unit-sum"):
        rules.combine(observations(), np.ones(4))


def test_unmatured_scores_are_explicitly_unscored_not_safe():
    weights, audit = rules.choose_pair(observations(mature=False), np.array([1.0, 0, 0, 0]))
    np.testing.assert_array_equal(weights, np.full(4, 0.25))
    assert audit["reason"] == "unscored_initial_equal_weights"
    assert audit["scores_available"] is False and audit["safety_or_success_claimed"] is False


def test_negative_scores_still_rank_without_a_new_cash_cutoff():
    observed = observations()
    observed = [replace(o, net_returns=o.net_returns - 0.03) for o in observed]
    weights, audit = rules.choose_pair(observed, np.full(4, 0.25))
    assert max(p["selection_score"] for p in audit["pairs"]) < 0
    assert (weights > 0).sum() == 2 and audit["scores_available"] is True


def test_score_ties_use_lexical_ids_not_archive_vector_order():
    observed = observations()
    observed = [replace(o, net_returns=observed[0].net_returns.copy()) for o in observed]
    _, audit = rules.choose_pair(observed, np.full(4, 0.25))
    assert len({p["selection_score_key"] for p in audit["pairs"]}) == 1
    assert audit["selected_pair"] == sorted(rules.EXPERT_IDS)[:2]


def test_related_mechanism_pair_is_not_an_additional_independent_choice():
    observed = observations()
    observed[1] = replace(
        observed[1],
        identity=replace(
            observed[1].identity, mechanism_group=observed[0].identity.mechanism_group
        ),
    )
    _, audit = rules.choose_pair(observed, np.full(4, 0.25))
    pair = next(p for p in audit["pairs"] if p["pair"] == sorted(rules.EXPERT_IDS[:2]))
    assert pair["unscorable_reason"] == "same_mechanism_group"


def test_degenerate_risk_remains_explicitly_unscored():
    observed = observations()
    observed = [replace(o, net_returns=o.risk_free.copy()) for o in observed]
    weights, audit = rules.choose_pair(observed, np.full(4, 0.25))
    np.testing.assert_array_equal(weights, np.full(4, 0.25))
    assert audit["reason"] == "unscored_degenerate_pair_equal_weights"
    assert all(p["unscorable_reason"] == "zero_training_variance" for p in audit["pairs"])


@pytest.mark.parametrize("bad", ["missing_date", "stale", "future_target", "nonfinite"])
def test_bad_shadow_or_target_clock_cannot_be_a_warmup_default(bad):
    observed = observations(mature=False)
    o = observed[0]
    if bad == "missing_date":
        o = replace(
            o,
            net_returns=o.net_returns.drop(o.net_returns.index[3]),
            risk_free=o.risk_free.drop(o.risk_free.index[3]),
        )
    elif bad == "stale":
        o = replace(o, as_of=o.as_of + pd.Timedelta(days=1))
    elif bad == "future_target":
        o = replace(o, target_known_at=o.as_of + pd.Timedelta(days=1))
    else:
        values = o.net_returns.copy()
        values.iloc[0] = np.nan
        o = replace(o, net_returns=values)
    observed[0] = o
    with pytest.raises(QuantError):
        rules.choose_pair(observed, np.full(4, 0.25))


def archived_experts(late_initial=False):
    calendar = sessions("2016-09-01", "2017-12-29")
    dates = calendar[calendar >= "2016-10-06"]
    assets = tuple("abcd")
    result = []
    for i, identifier in enumerate(rules.EXPERT_IDS):
        values = 0.0004 + (0.004 + i * 0.002) * np.sin(np.arange(len(dates)) / (2 + i))
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
        row = np.zeros(4)
        row[i] = 0.98
        if i == 2:
            target.loc[[d for d in calendar if is_month_end(d)]] = row
            adapter = ArchivedMonthlyExpert
        else:
            start = (
                pd.Timestamp("2016-10-06" if late_initial else "2016-10-05")
                if i == 3
                else calendar[0]
            )
            target.loc[start:] = row
            adapter = ArchivedDenseExpert
        result.append(
            adapter(
                ExpertIdentity(identifier, rules.MECHANISMS[i], "2026-10-29T17:21:25Z"),
                target,
                shadow,
                assets,
            )
        )
    return tuple(result), calendar, assets


def test_real_initial_anchor_and_prefix_causality_preserve_all_formal_days():
    experts, calendar, assets = archived_experts()
    policy = read_json(rules.POLICY)
    full, audits = rules.build_targets(experts, calendar, assets, policy)
    assert audits[0]["decision_session"] == "2016-10-05"
    assert audits[0]["decision_role"] == "initial_capital"
    assert audits[0]["selection"]["scores_available"] is False
    assert all(a["decision_role"] == "completed_exchange_month_end" for a in audits[1:])
    assert sum(a["selection"]["reason"] == "unscored_initial_equal_weights" for a in audits) == 7
    assert (
        next(a["decision_session"] for a in audits if a["selection"]["scores_available"])
        == "2017-04-28"
    )
    assert all(frame.loc[:"2016-10-04"].isna().all().all() for frame in full.values())
    assert all(not frame.ffill().loc["2016-10-05":].isna().any().any() for frame in full.values())
    for end in (pd.Timestamp("2016-10-05"), pd.Timestamp("2017-04-28"), pd.Timestamp("2017-10-18")):
        early, earlier = rules.build_targets(experts, calendar[calendar <= end], assets, policy)
        for name in full:
            pd.testing.assert_frame_equal(early[name], full[name].loc[:end])
        assert earlier == [a for a in audits if pd.Timestamp(a["decision_session"]) <= end]


def test_missing_initial_specialist_is_not_fabricated_or_skipped():
    experts, calendar, assets = archived_experts(late_initial=True)
    with pytest.raises(QuantError, match="missing post-start"):
        rules.build_targets(experts, calendar, assets, read_json(rules.POLICY))


@pytest.mark.parametrize(
    "key,value",
    [
        ("lookback_sessions", 252),
        ("training_sessions", 63),
        ("validation_sessions", 21),
        ("selected_experts", 1),
        ("switch_cost_bps_per_side", 0.0),
        ("initial_decision_session", "2016-09-30"),
    ],
)
def test_fixed_scoring_and_initial_availability_cannot_be_retuned(key, value):
    policy = read_json(rules.POLICY)
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)
