from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from us_quant import strategy_expert_meta_v1 as rules
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import ArchivedDenseExpert, ExpertIdentity, ExpertObservation


def observations(mature=True):
    dates = sessions("2020-01-02", "2020-09-30")[-126:]
    result = []
    phase = np.arange(126)
    values = [
        0.0007 + 0.012 * np.sin(phase / 2),
        0.0003 + 0.003 * np.cos(phase / 3),
        0.0009 + 0.018 * np.sin(phase / 2 + 0.1),
    ]
    for i, identifier in enumerate(rules.EXPERT_IDS):
        selected = dates if mature else dates[:20]
        result.append(
            ExpertObservation(
                ExpertIdentity(identifier, f"mechanism_{i}", "2026-10-29T13:25:29Z"),
                dates[-1],
                pd.Series([0.98 if k == i else 0.0 for k in range(3)], index=["a", "b", "c"]),
                dates[-1],
                pd.Series(values[i][: len(selected)], index=selected),
                pd.Series(0.0001, index=selected),
                mature,
            )
        )
    return result


def test_unmatured_assessment_is_labeled_and_never_a_successful_score():
    weights, audit = rules.choose_pair(observations(False), np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(weights, [1 / 3] * 3)
    assert audit["reason"] == "unscored_initial_equal_weights"
    assert audit["scores_available"] is False
    assert audit["safety_or_success_claimed"] is False


def test_rolling_pair_uses_joint_risk_and_equal_marginal_contributions():
    observed = observations()
    weights, audit = rules.choose_pair(observed, np.full(3, 1 / 3))
    assert (weights > 0).sum() == 2 and weights.sum() == pytest.approx(1.0)
    assert audit["train_end"] < audit["validation_start"] <= audit["validation_end"]
    assert audit["validation_end"] <= str(observed[0].as_of.date())
    assert len(audit["pairs"]) == 3
    for row in audit["pairs"]:
        positive = [v for v in row["training_marginal_variance_contributions"] if v != 0]
        assert positive[0] == pytest.approx(positive[1], rel=1e-12)
        assert row["selection_score"] == min(
            row["train_net_risk_score"], row["validation_net_risk_score"]
        )
        assert row["cost_proxy_is_actual_main_account_cost"] is False
    assert (
        audit["selected_pair"]
        == sorted(audit["pairs"], key=lambda row: (-row["selection_score_key"], row["pair"]))[0][
            "pair"
        ]
    )


def test_unified_asset_target_is_not_sum_of_separately_funded_accounts():
    combined = rules.combine(observations(), np.full(3, 1 / 3))
    assert combined.sum() == pytest.approx(0.98)
    np.testing.assert_allclose(combined, [0.98 / 3] * 3)
    assert 10000 * combined.sum() == pytest.approx(9800)
    with pytest.raises(QuantError, match="unit-sum"):
        rules.combine(observations(), np.ones(3))


def test_meta_switching_cost_disappears_when_expert_weights_are_unchanged():
    observed = observations()
    weights, audit = rules.choose_pair(observed, np.full(3, 1 / 3))
    _, repeat = rules.choose_pair(observed, weights)
    chosen = next(row for row in repeat["pairs"] if row["pair"] == audit["selected_pair"])
    assert chosen["estimated_incremental_switch_cost_fraction"] == 0
    assert chosen["estimated_incremental_changed_assets"] == 0


def test_future_shadow_observation_is_rejected():
    observed = observations()
    bad = copy.copy(observed[0])
    object.__setattr__(bad, "as_of", observed[0].net_returns.index[-2])
    with pytest.raises(QuantError):
        rules.choose_pair([bad, *observed[1:]], np.full(3, 1 / 3))


def test_highest_full_sample_growth_is_not_the_selection_rule():
    observed = observations()
    phase = np.arange(126)
    returns = [
        np.r_[np.full(84, 0.01), np.full(42, -0.005)] + 0.035 * np.sin(phase),
        0.0007 + 0.005 * np.sin(phase),
        0.0005 + 0.004 * np.cos(phase),
    ]
    for item, values in zip(observed, returns, strict=True):
        object.__setattr__(item, "net_returns", pd.Series(values, index=item.net_returns.index))
    assert np.prod(1 + returns[0]) > max(np.prod(1 + r) for r in returns[1:])
    weights, audit = rules.choose_pair(observed, np.full(3, 1 / 3))
    assert weights[0] == 0
    assert audit["selected_pair"] == sorted(rules.EXPERT_IDS[1:])


def test_composition_prefix_preserves_all_earlier_targets_and_decisions():
    dates = sessions("2020-01-02", "2021-06-30")
    experts = []
    for i, identifier in enumerate(rules.EXPERT_IDS):
        returns = 0.0004 + (0.004 + i * 0.002) * np.sin(np.arange(len(dates)) / (2 + i))
        nav = 10000 * np.cumprod(1 + returns)
        shadow = pd.DataFrame(
            {
                "equity": nav,
                "return": returns,
                "risk_free": 0.0001,
                "cash": nav * 0.02,
                "gross_exposure": 0.98,
                "cost": 0.0,
                "orders": 0,
            },
            index=dates,
        )
        target = pd.DataFrame(0.0, index=dates, columns=["a", "b", "c"])
        target.iloc[:, i] = 0.98
        experts.append(
            ArchivedDenseExpert(
                ExpertIdentity(identifier, f"mechanism_{i}", "2026-10-29T13:25:29Z"),
                target,
                shadow,
                ("a", "b", "c"),
            )
        )
    policy = read_json(rules.POLICY)
    full, audits = rules.build_targets(tuple(experts), dates, ("a", "b", "c"), policy)
    cutoff = pd.Timestamp("2020-11-18")
    prefix, earlier = rules.build_targets(
        tuple(experts), dates[dates <= cutoff], ("a", "b", "c"), policy
    )
    for identifier in full:
        pd.testing.assert_frame_equal(prefix[identifier], full[identifier].loc[:cutoff])
    assert earlier == [a for a in audits if pd.Timestamp(a["decision_session"]) <= cutoff]
    assert any(a["selection"]["scores_available"] for a in audits)
    assert any(not a["selection"]["scores_available"] for a in audits)
    assert all(a["shadow_wealth_added_to_main_capital"] is False for a in audits)


@pytest.mark.parametrize(
    "key,value",
    [
        ("lookback_sessions", 252),
        ("selected_experts", 1),
        ("switch_cost_bps_per_side", 0.0),
        ("training_sessions", 63),
    ],
)
def test_scoring_and_cost_rules_cannot_be_retuned(key, value):
    policy = read_json(rules.POLICY)
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)
