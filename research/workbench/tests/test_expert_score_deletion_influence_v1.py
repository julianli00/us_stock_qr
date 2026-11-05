import copy
import itertools
import math
from contextlib import nullcontext
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from us_quant import expert_score_deletion_influence_v1 as law
from us_quant import ssd_expert_score_deletion_influence_v1 as driver
from us_quant.config import QuantError


def synthetic(*, cold=False, all_equal=False, unscorable=False):
    dates = pd.bdate_range(end="2020-07-31", periods=126).strftime("%Y-%m-%d").tolist()
    values = np.asarray(
        [
            [0.0002 * (i + 1) + 0.008 * math.sin(t * (0.17 + i / 31)) for t in range(126)]
            for i in range(9)
        ]
    )
    if unscorable:
        values[:] = 0.0
    observations = []
    for i, name in enumerate(law.source.EXPERT_IDS):
        target = np.zeros(14)
        target[0 if all_equal else i] = 0.98
        observations.append(
            {
                "id": name,
                "mechanism_group": law.source.MECHANISMS[i],
                "target_known_at": "2020-07-30",
                "asset_order": list(law.source.UNIVERSE),
                "target_binary64_hex": [float(x).hex() for x in target],
                "return_dates": [] if cold else dates,
                "NET_returns_binary64_hex": [] if cold else [float(x).hex() for x in values[i]],
                "RF_binary64_hex": [] if cold else [float(0).hex()] * 126,
                "score_mature": not cold,
            }
        )
    pairs = []
    if not cold:
        for i, j in itertools.combinations(range(9), 2):
            names = sorted((law.source.EXPERT_IDS[i], law.source.EXPERT_IDS[j]))
            pair = {
                "pair": names,
                "eligible_by_current_opinion": not all_equal,
                "exact_same_current_opinion": all_equal,
            }
            if unscorable:
                pair["unscorable_reason"] = "zero_training_variance"
            else:
                weights = np.zeros(9)
                weights[i] = weights[j] = 0.5
                portfolio = (values[i] + values[j]) / 2
                cost = 0.0003

                def score(x, fixed_cost=cost):
                    return float(np.sqrt(252) * (np.mean(x) - fixed_cost / 21) / np.std(x, ddof=1))

                train, valid = score(portfolio[:84]), score(portfolio[84:])
                pair.update(
                    {
                        "expert_weights": weights.tolist(),
                        "estimated_incremental_switch_cost_fraction": cost,
                        "cost_proxy_is_actual_main_account_cost": False,
                        "train_net_risk_score": train,
                        "validation_net_risk_score": valid,
                        "selection_score": min(train, valid),
                        "selection_score_key": round(min(train, valid), 12),
                    }
                )
            pairs.append(pair)
    eligible = [p for p in pairs if p["eligible_by_current_opinion"] and "selection_score_key" in p]
    selected = (
        min(eligible, key=lambda p: (-p["selection_score_key"], p["pair"]))["pair"]
        if eligible
        else None
    )
    before = {
        "record": "decision_input",
        "decision_index": 0,
        "as_of": "2020-07-31",
        "information_clock": law.source.CLOCK,
        "observations": observations,
        "previous_weights_binary64_hex": [float(1 / 9).hex()] * 9,
    }
    after = {
        "record": "decision_resolved",
        "decision_index": 0,
        "asset_order": list(law.source.UNIVERSE),
        "audit": {
            "decision_session": before["as_of"],
            "decision_information_clock": law.source.CLOCK,
            "previous_rolling_expert_weights": [1 / 9] * 9,
            "selection": {
                "NET_history_mature": not cold,
                "pairs": pairs,
                "selected_pair": selected,
                "scores_available": bool(eligible),
                "reason": "synthetic_frozen_catalogue",
            },
        },
    }
    return before, after


def test_real_interface_keeps_every_pair_and_deletion_and_independent_moments():
    before, after = synthetic()
    result = law.diagnose(before, after, copy.deepcopy(law.EXPECTED))
    assert len(result["profiles"]) == 36
    assert len(result["deletions"]) == 126
    assert result["oracle_sample_score_checks"] == 36 * 128
    assert result["maximum_independent_score_gap"] < 1e-8
    assert result["maximum_source_baseline_score_gap"] < 1e-8
    assert before == synthetic()[0]


def test_rational_scalar_oracle_for_a_deleted_training_sample():
    before, after = synthetic()
    result = law.diagnose(before, after, law.EXPECTED)
    pair, profile = after["audit"]["selection"]["pairs"][0], result["profiles"][0]
    samples = [
        sum(
            Fraction.from_float(w)
            * Fraction.from_float(float.fromhex(o["NET_returns_binary64_hex"][t]))
            for w, o in zip(pair["expert_weights"], before["observations"], strict=True)
        )
        for t in range(84)
        if t != 8
    ]
    mean = sum(samples) / len(samples)
    variance = sum((x - mean) ** 2 for x in samples) / (len(samples) - 1)
    cost = Fraction.from_float(pair["estimated_incremental_switch_cost_fraction"]) / 21
    expected = math.sqrt(252) * float(mean - cost) / math.sqrt(float(variance))
    assert abs(profile["training_one_return_deleted_scores"][8] - expected) < 1e-8


@pytest.mark.parametrize(
    "kind",
    [
        "clock",
        "future",
        "RF",
        "target",
        "identity",
        "previous",
        "maturity",
        "pair_drop",
        "pair_flag",
        "cost",
        "score",
        "key",
        "winner",
    ],
)
def test_source_record_tampering_is_not_silently_normalized(kind):
    before, after = synthetic()
    pair = after["audit"]["selection"]["pairs"][0]
    if kind == "clock":
        before["information_clock"] = "before_close"
    elif kind == "future":
        before["observations"][0]["target_known_at"] = "2021-01-01"
    elif kind == "RF":
        before["observations"][0]["RF_binary64_hex"][0] = (0.1).hex()
    elif kind == "target":
        before["observations"][0]["target_binary64_hex"] = []
    elif kind == "identity":
        before["observations"].pop()
    elif kind == "previous":
        before["previous_weights_binary64_hex"][0] = (0.9).hex()
    elif kind == "maturity":
        after["audit"]["selection"]["NET_history_mature"] = False
    elif kind == "pair_drop":
        after["audit"]["selection"]["pairs"].pop()
    elif kind == "pair_flag":
        pair["eligible_by_current_opinion"] = False
    elif kind == "cost":
        pair["estimated_incremental_switch_cost_fraction"] = -0.5
    elif kind == "score":
        pair["train_net_risk_score"] += 0.2
    elif kind == "key":
        pair["selection_score_key"] += 0.2
    else:
        after["audit"]["selection"]["selected_pair"] = ["not", "source"]
    with pytest.raises(QuantError):
        law.diagnose(before, after, law.EXPECTED)


def test_cold_origins_remain_unscored_not_zero_influence_success():
    before, after = synthetic(cold=True)
    result = law.diagnose(before, after, law.EXPECTED)
    assert result["state"] == "source_unscored" and result["deletions"] == []
    assert result["source_key_bucket_differences"] is None


@pytest.mark.parametrize("kind", ["all_equal", "unscorable"])
def test_mature_no_winner_never_becomes_a_safe_allocation(kind):
    before, after = synthetic(**{kind: True})
    result = law.diagnose(before, after, law.EXPECTED)
    assert result["conditional_full_sample_winner"] is None
    assert all(row["no_scorable_conditional_pair"] for row in result["deletions"])
    assert driver._case(result)["changed_vs_kernel_baseline"] is None


def test_fixed_numerical_score_has_explicit_constant_classification():
    assert law._score([0.001] * 41, 0.0003) is None
    assert law._oracle([0.001] * 41, 0.0003) is None
    with pytest.raises(QuantError):
        law._score([0.001], 0)


def test_a_deletion_can_remove_every_scorable_pair_without_a_safe_fallback():
    before, after = synthetic()
    values = np.zeros((9, 126))
    values[0, 10] = 0.01
    values[0, 84:] = np.sin(np.arange(42)) * 0.01
    for i, observation in enumerate(before["observations"]):
        observation["NET_returns_binary64_hex"] = [float(v).hex() for v in values[i]]
    for pair in after["audit"]["selection"]["pairs"]:
        positions = [law.source.EXPERT_IDS.index(n) for n in pair["pair"]]
        if 0 not in positions:
            pair.clear()
            pair.update(
                {
                    "pair": sorted(law.source.EXPERT_IDS[i] for i in positions),
                    "eligible_by_current_opinion": True,
                    "exact_same_current_opinion": False,
                    "unscorable_reason": "zero_training_variance",
                }
            )
        else:
            portfolio = values[0] / 2
            scores = [
                float(np.sqrt(252) * (np.mean(x) - 0.0003 / 21) / np.std(x, ddof=1))
                for x in (portfolio[:84], portfolio[84:])
            ]
            pair.update(
                {
                    "train_net_risk_score": scores[0],
                    "validation_net_risk_score": scores[1],
                    "selection_score": min(scores),
                    "selection_score_key": round(min(scores), 12),
                }
            )
    usable = [p for p in after["audit"]["selection"]["pairs"] if "selection_score_key" in p]
    after["audit"]["selection"]["selected_pair"] = min(
        usable, key=lambda p: (-p["selection_score_key"], p["pair"])
    )["pair"]
    result = law.diagnose(before, after, law.EXPECTED)
    assert result["deletions"][10]["conditional_winner"] is None
    assert result["deletions"][10]["no_scorable_conditional_pair"] is True
    assert len(result["deletions"][10]["eligible_pairs_with_degenerate_deleted_sample"]) == 8
    assert result["new_portfolio_targets_accounts_or_financial_acceptance"] is False


def test_partial_diagnostic_cannot_be_reset_or_replayed(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "FOLDER", tmp_path)
    monkeypatch.setattr(driver, "RESULTS", tmp_path / "results.json")
    monkeypatch.setattr(driver.lifecycle, "runtime_guard", lambda: None)
    monkeypatch.setattr(driver.lifecycle, "research_job", nullcontext)
    marker = tmp_path / "actual-calculation-started-v1.json"
    marker.write_text('{"partial":true}')
    with pytest.raises(QuantError, match="Preserve"):
        driver.run()
    assert marker.read_text() == '{"partial":true}'


@pytest.mark.parametrize(
    "key,value",
    [
        ("train_observations", 63),
        ("current_opinion_eligibility_fixed", False),
        ("new_trading_policy_or_acceptance_rule", True),
        ("expert_count", True),
    ],
)
def test_no_parameter_or_trading_rule_change(key, value):
    policy = copy.deepcopy(law.EXPECTED)
    policy[key] = value
    with pytest.raises(QuantError):
        law.validate(policy)


def test_only_new_exact_dependency_closure_is_bound():
    paths = {r["path"] for r in driver.dependencies()}
    assert set(driver.OWN).issubset(paths)
    assert "src/us_quant/expert_distinct_opinion_pair_v1.py" in paths
