import copy
import json
from dataclasses import replace
from decimal import Decimal
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import test_quintet_anytime_hedge_v1 as fence_source
from test_growth_macro_defense_hedge_v1 import MappedSyntheticExpert
from test_strategy_expert_pool_eight_v1 import archived_experts

from us_quant import protective_six_adaptive_hedge_v1 as rules
from us_quant import ssd_protective_six_adaptive_hedge_v1 as study
from us_quant.config import QuantError
from us_quant.retained_pair_equal_allocation_v1 import ASSETS
from us_quant.storage import read_json


def sources(changed_after=None):
    eight, calendar, toy_assets = archived_experts(changed_after=changed_after)
    mapping = dict(zip(toy_assets, ASSETS[: len(toy_assets)], strict=True))
    selected = tuple(MappedSyntheticExpert(eight[i], mapping) for i in rules.SLOTS)
    return selected, calendar, ASSETS


def test_six_real_identities_not_padded_five_and_original_math_code_is_reused():
    experts, _, _ = sources()
    assert tuple(e.identity.identifier for e in experts) == rules.EXPERT_IDS
    assert rules.EXPERT_IDS[:-1] == rules.prior.EXPERT_IDS
    assert rules.EXPERT_IDS[-1] == "semiconductor_ratchet_protection"
    assert rules.MECHANISMS[-1] == "wealth_feedback_protection"
    assert rules.AdaHedge.update.__code__ is rules.prior.AdaHedge.update.__code__
    assert rules.mix.__code__ is rules.prior.mix.__code__
    assert rules.independent_trace.__code__ is rules.prior.independent_trace.__code__
    assert rules.prior.N == 5 and rules.prior.PRECISION == 512
    assert rules.N == 6 and rules.PRECISION == 768
    assert rules.AdaHedge().rate() is None
    np.testing.assert_array_equal(rules.AdaHedge().weights(), np.full(6, 1 / 6))


def test_original_gap_rule_uses_log6_and_neutral_month_does_not_cool():
    state = rules.AdaHedge()
    weights, audit = state.update([0, 0.2, 0.4, 0.6, 0.8, 1])
    assert audit["learning_rate_before_is_infinite"] is True
    assert len(audit["input_loss_labels"]) == 6
    assert float(state.gap) == pytest.approx(0.5, abs=1e-14)
    assert float(state.rate()) == pytest.approx(2 * np.log(6), abs=1e-14)
    before, rate = weights.copy(), state.rate()
    weights, update = state.update([0] * 6)
    np.testing.assert_array_equal(weights, before)
    assert state.rate() == rate and state.periods == 2
    assert Decimal(update["mixability_gap_increment_decimal"]) == 0
    assert rules.prior.N == 5


def test_119_six_coordinate_updates_match_independent_oracle_and_source_bound():
    state, history, actual = rules.AdaHedge(), [], []
    labels = np.random.default_rng(42).uniform(0, 1, (119, 6))
    labels[:, 0], labels[:, 1] = 0, 1
    for row in labels:
        weights, audit = state.update(row)
        history.append(row.tolist())
        actual.append((weights, audit))
        assert Decimal(audit["normalized_source_regret_decimal"]) <= (
            Decimal(audit["normalized_source_regret_reference_decimal"]) + Decimal("1e-12")
        )
    independent = rules.independent_trace(history)
    for (weights, audit), row in zip(actual, independent, strict=True):
        np.testing.assert_allclose(weights, row["weights_after"], rtol=0, atol=1e-12)
        assert len(audit["state_after"]["cumulative_loss_fractions"]) == 6
        assert audit["state_after"]["arithmetic_precision"] == 768
    with pytest.raises(QuantError):
        state.update([0, 0, 0, 0, 0, 1])


def test_dimension_safe_precision_keeps_tiny_statistical_mass_without_floor():
    state = rules.AdaHedge()
    for _ in range(119):
        state.update([0, 0, 0, 0, 0, 1])
    precise, _ = rules.mix(state.losses, state.rate())
    assert precise[-1] > 0 and state.weights()[-1] == 0
    assert state.gap > 0


@pytest.mark.parametrize(
    "bad",
    [
        [0, 0.25, 0.5, 0.75, 1],
        [0.2] * 6,
        [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        [0, 0, 0, -0.1, 0, 1],
        [0, 0, 0, np.nan, 0, 1],
        [0, 0, 0, np.inf, 0, 1],
    ],
)
def test_missing_sixth_or_invalid_losses_cannot_be_padded_or_filled(bad):
    with pytest.raises(QuantError):
        rules.AdaHedge().update(bad)


def test_sixth_source_future_does_not_change_prefix_or_reset_initial_maturity():
    experts, calendar, assets = sources()
    cutoff = pd.Timestamp("2017-04-28")
    future, _, _ = sources(changed_after=cutoff)
    only_sixth_future = (*experts[:-1], future[-1])
    first, a, p = rules.build_targets(experts, calendar, assets, read_json(rules.POLICY))
    changed, b, q = rules.build_targets(
        only_sixth_future, calendar, assets, read_json(rules.POLICY)
    )
    early, c, r = rules.build_targets(
        experts, calendar[calendar <= cutoff], assets, read_json(rules.POLICY)
    )
    pd.testing.assert_frame_equal(first.loc[:cutoff], changed.loc[:cutoff])
    pd.testing.assert_frame_equal(first.loc[:cutoff], early)
    assert c == [row for row in a if row["decision_session"] <= str(cutoff.date())]
    assert c == [row for row in b if row["decision_session"] <= str(cutoff.date())]
    assert r == [row for row in p if row["end"] <= str(cutoff.date())]
    assert r == [row for row in q if row["end"] <= str(cutoff.date())]
    assert [row["score_mature"] for row in a[:3]] == [False, False, True]
    assert len(p[0]["input_loss_labels"]) == 6


def test_independent_six_state_risk_cost_targets_and_retained_NET_RF_binary64(tmp_path):
    experts, calendar, assets = sources()
    records = []
    target, audits, updates = rules.build_targets(
        experts, calendar, assets, read_json(rules.POLICY), retain=records.append
    )
    data = SimpleNamespace(close=pd.DataFrame(index=calendar, columns=assets))
    result = study.verify_targets(data, experts, target, audits, updates)
    assert result["real_source_coordinates"] == 6
    assert result["maximum_independent_expert_weight_difference"] <= 1e-12
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    assert study.verify_retained_records(path, experts, target, audits)[
        "actual_input_and_resolved_record_count"
    ] == 2 * len(audits)
    damaged = copy.deepcopy(audits)
    damaged[-1]["expert_weights"] = damaged[-1]["expert_weights"][:5]
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, target, damaged, updates)
    damaged = copy.deepcopy(audits)
    damaged[-1]["risk_and_cost"]["marginal_variance_contributions"][5] += 0.01
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, target, damaged, updates)
    records[-2]["canonical_RF_hex"][0] = (0.9).hex()
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(QuantError):
        study.verify_retained_records(path, experts, target, audits)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "RF", "date", "opinion"])
def test_unavailable_or_fabricated_sixth_source_is_not_silent_five_source_fallback(kind):
    experts, calendar, assets = sources()
    if kind == "missing":
        experts = experts[:-1]
    elif kind == "duplicate":
        experts = (*experts[:-1], experts[0])
    else:
        last = experts[-1]

        def observe(day, count):
            original = last.observe(day, count)
            if kind == "RF":
                return replace(original, risk_free=original.risk_free + 0.001)
            if kind == "date":
                return replace(original, net_returns=original.net_returns.iloc[1:])
            return replace(original, target=None, target_known_at=None)

        experts = (*experts[:-1], SimpleNamespace(identity=last.identity, observe=observe))
    with pytest.raises(QuantError):
        rules.build_targets(experts, calendar, assets, read_json(rules.POLICY))


@pytest.mark.parametrize(
    "key,value",
    [
        ("original_source_slots", [4, 5, 7, 0, 1, 2]),
        ("global_pool_changed", True),
        ("new_expert_admissions", 1),
        ("source_wealth_guarantee_transferred_to_main", True),
        ("learning_rate", "2_sqrt_log6_over_t"),
        ("negative_Jensen_gap_clipped", True),
        ("mixability_gap_floor", 0.001),
        ("rate_or_window_sweep", True),
        ("new_source_RF_views", 1),
        ("actual_weight_trade_band", 0.05),
        ("independent_numeric_tolerance", 1e-8),
        ("learning_resets_for_five_year_account", True),
    ],
)
def test_no_source_rate_floor_guard_clock_or_financial_rescue(key, value):
    with pytest.raises(QuantError):
        rules.validate({**read_json(rules.POLICY), key: value})


def test_actual_singleton_registration_precedes_six_source_labels_and_learning(monkeypatch):
    function = fence_source.test_actual_singleton_registration_precedes_all_new_learning
    FunctionType(
        function.__code__,
        {**function.__globals__, "study": study, "rules": rules},
        argdefs=function.__defaults__,
    )(monkeypatch)
