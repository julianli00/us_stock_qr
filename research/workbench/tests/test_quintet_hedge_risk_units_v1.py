import copy
import json
from dataclasses import replace
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import test_quintet_anytime_hedge_v1 as prior_tests
from test_growth_macro_defense_hedge_v1 import MappedSyntheticExpert
from test_strategy_expert_pool_eight_v1 import archived_experts

from us_quant import quintet_hedge_risk_units_v1 as rules
from us_quant import ssd_quintet_hedge_risk_units_v1 as study
from us_quant.config import QuantError
from us_quant.retained_pair_equal_allocation_v1 import ASSETS
from us_quant.storage import read_json


def sources(changed_after=None):
    eight, calendar, toy_assets = archived_experts(changed_after=changed_after)
    mapping = dict(zip(toy_assets, ASSETS[: len(toy_assets)], strict=True))
    experts = tuple(MappedSyntheticExpert(eight[i], mapping) for i in rules.prior.pool.SLOTS)
    dates = rules.decision_sessions(calendar, read_json(rules.POLICY))
    cached = [
        {
            "decision_session": str(day.date()),
            "expert_weights": [0.2] * 5 if i < 2 else [0.1, 0.1, 0.1, 0.6, 0.1],
        }
        for i, day in enumerate(dates)
    ]
    return experts, cached, calendar, ASSETS


def test_current_pool_and_exact_original_covariance_function():
    experts, _, _, _ = sources()
    assert tuple(e.identity.identifier for e in experts) == rules.EXPERT_IDS
    assert rules.EXPERT_IDS == study.prior.rules.EXPERT_IDS
    assert rules.sample_covariance is rules.covariance_source.sample_covariance
    assert rules.CANDIDATES[0]["id"] == rules.ID


def test_inverse_volatility_units_are_belief_scaled_not_full_covariance_erc():
    experts, _, _, _ = sources()
    observed = rules.observations(experts, pd.Timestamp("2017-04-28"))
    common = np.sin(np.arange(126) / 8) * 0.001
    scales = [1, 2, 3, 4, 5]
    changed = [
        replace(
            o,
            net_returns=pd.Series(
                o.risk_free.to_numpy() + common * scale, index=o.net_returns.index
            ),
        )
        for o, scale in zip(observed, scales, strict=True)
    ]
    belief = np.array([0.1, 0.2, 0.3, 0.1, 0.3])
    weights, audit = rules.capital_weights(changed, belief)
    expected = belief / scales
    expected /= expected.sum()
    np.testing.assert_allclose(weights, expected, rtol=0, atol=1e-12)
    assert audit["risk_sizing_available"] is True
    assert audit["full_covariance_ERC_or_variance_target_claimed"] is False
    assert audit["main_drawdown_guarantee"] is False


def test_unmatured_and_valid_constant_risk_are_explicit_belief_passthrough():
    experts, _, _, _ = sources()
    belief = np.array([0.1, 0.1, 0.2, 0.5, 0.1])
    warm = rules.observations(experts, pd.Timestamp("2016-10-05"))
    weights, audit = rules.capital_weights(warm, belief)
    np.testing.assert_array_equal(weights, belief)
    assert audit["risk_sizing_available"] is False
    mature = rules.observations(experts, pd.Timestamp("2017-04-28"))
    constant = [replace(o, net_returns=o.risk_free.copy()) for o in mature]
    weights, audit = rules.capital_weights(constant, belief)
    np.testing.assert_array_equal(weights, belief)
    assert audit["reason"] == "unscored_nonpositive_individual_variance_passthrough_beliefs"
    assert audit["safety_or_success_claimed"] is False


def test_equal_source_risk_retains_beliefs_without_claiming_equal_covariance_risk():
    experts, _, _, _ = sources()
    observed = rules.observations(experts, pd.Timestamp("2017-04-28"))
    common = np.sin(np.arange(126) / 8) * 0.001
    equal_risk = [
        replace(
            o, net_returns=pd.Series(o.risk_free.to_numpy() + common, index=o.net_returns.index)
        )
        for o in observed
    ]
    belief = np.array([0.1, 0.1, 0.2, 0.5, 0.1])
    weights, audit = rules.capital_weights(equal_risk, belief)
    np.testing.assert_allclose(weights, belief, rtol=0, atol=1e-12)
    assert audit["risk_sizing_available"] is True
    assert audit["full_covariance_ERC_or_variance_target_claimed"] is False


@pytest.mark.parametrize(
    "belief",
    [
        [0.2] * 4,
        [0.1] * 5,
        [0.2, 0.2, 0.2, -0.1, 0.5],
        [0.2, 0.2, 0.2, np.nan, 0.2],
    ],
)
def test_invalid_beliefs_cannot_be_filled(belief):
    experts, _, _, _ = sources()
    observed = rules.observations(experts, pd.Timestamp("2017-04-28"))
    with pytest.raises(QuantError):
        rules.capital_weights(observed, belief)


def test_bad_covariance_is_error_not_regularized_or_belief_fallback(monkeypatch):
    experts, _, _, _ = sources()
    observed = rules.observations(experts, pd.Timestamp("2017-04-28"))
    covariance = np.eye(5)
    covariance[0, 1] = covariance[1, 0] = 2
    monkeypatch.setattr(rules, "sample_covariance", lambda values: covariance)
    with pytest.raises(QuantError):
        rules.capital_weights(observed, np.full(5, 0.2))


@pytest.mark.parametrize(
    "kind", ["negative", "nonfinite", "zero_trace_nonzero", "zero_diagonal_indefinite"]
)
def test_invalid_covariance_cannot_be_disguised_as_valid_degenerate_risk(monkeypatch, kind):
    experts, _, _, _ = sources()
    observed = rules.observations(experts, pd.Timestamp("2017-04-28"))
    covariance = np.eye(5)
    if kind == "negative":
        covariance[0, 0] = -1
    elif kind == "nonfinite":
        covariance[0, 0] = np.inf
    elif kind == "zero_trace_nonzero":
        covariance = np.zeros((5, 5))
        covariance[0, 1] = covariance[1, 0] = 1
    else:
        covariance[0, 0] = 0
        covariance[0, 1] = covariance[1, 0] = 1
    monkeypatch.setattr(rules, "sample_covariance", lambda values: covariance)
    with pytest.raises(QuantError):
        rules.capital_weights(observed, np.full(5, 0.2))


@pytest.mark.parametrize("kind", ["RF", "calendar", "nonfinite", "maturity"])
def test_invalid_source_state_stops_before_any_risk_passthrough(kind):
    experts, _, _, _ = sources()
    day = pd.Timestamp("2017-04-28")
    observed = rules.observations(experts, day)
    if kind == "RF":
        broken = replace(observed[0], risk_free=observed[0].risk_free + 0.001)
    elif kind == "calendar":
        broken = replace(observed[0], net_returns=observed[0].net_returns.iloc[1:])
    elif kind == "nonfinite":
        returns = observed[0].net_returns.copy()
        returns.iloc[-1] = np.nan
        broken = replace(observed[0], net_returns=returns)
    else:
        broken = replace(observed[0], score_mature=False)
    with pytest.raises(QuantError):
        rules.capital_weights([broken, *observed[1:]], np.full(5, 0.2))


def test_cached_belief_and_source_future_changes_preserve_every_prefix():
    experts, cached, calendar, assets = sources()
    cutoff = pd.Timestamp("2017-04-28")
    future, _, _, _ = sources(changed_after=cutoff)
    changed = copy.deepcopy(cached)
    for row in changed:
        if row["decision_session"] > str(cutoff.date()):
            row["expert_weights"] = [0.1, 0.1, 0.1, 0.2, 0.5]
    first, a = rules.build_targets(experts, cached, calendar, assets, read_json(rules.POLICY))
    second, b = rules.build_targets(future, changed, calendar, assets, read_json(rules.POLICY))
    early, c = rules.build_targets(
        experts, cached, calendar[calendar <= cutoff], assets, read_json(rules.POLICY)
    )
    pd.testing.assert_frame_equal(first.loc[:cutoff], second.loc[:cutoff])
    pd.testing.assert_frame_equal(first.loc[:cutoff], early)
    assert c == [row for row in a if row["decision_session"] <= str(cutoff.date())]
    assert c == [row for row in b if row["decision_session"] <= str(cutoff.date())]
    assert all(row["cached_beliefs"] == cached[i]["expert_weights"] for i, row in enumerate(a))


def test_independent_risk_sizing_target_and_retained_source_binary_values(tmp_path):
    experts, cached, calendar, assets = sources()
    rows = []
    target, audits = rules.build_targets(
        experts, cached, calendar, assets, read_json(rules.POLICY), retain=rows.append
    )
    data = SimpleNamespace(close=pd.DataFrame(index=calendar, columns=assets))
    check = study.verify_targets(data, experts, cached, target, audits)
    assert check["maximum_independent_expert_weight_difference"] <= 1e-12
    assert check["maximum_independent_source_std_difference"] <= 1e-12
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert study.verify_retained_records(path, experts, target, audits)[
        "actual_input_and_resolved_record_count"
    ] == 2 * len(audits)
    damaged = copy.deepcopy(audits)
    damaged[-1]["cached_beliefs"][0] += 0.01
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, cached, target, damaged)
    for field in (
        "source_daily_NET_RF_std",
        "raw_belief_marginal_variance_contributions",
        "capital_marginal_variance_contributions",
    ):
        damaged = copy.deepcopy(audits)
        damaged[-1]["risk_sizing"][field][0] += 0.01
        with pytest.raises(QuantError):
            study.verify_targets(data, experts, cached, target, damaged)
    rows[-2]["source_NET_returns_hex"][0][0] = (0.9).hex()
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(QuantError):
        study.verify_retained_records(path, experts, target, audits)


def test_missing_cached_date_or_source_opinion_is_error_not_regeneration():
    experts, cached, calendar, assets = sources()
    with pytest.raises(QuantError):
        rules.build_targets(experts, cached[1:], calendar, assets, read_json(rules.POLICY))
    last = experts[-1]
    missing = SimpleNamespace(
        identity=last.identity,
        observe=lambda day, n: replace(last.observe(day, n), target=None, target_known_at=None),
    )
    with pytest.raises(QuantError):
        rules.build_targets(
            (*experts[:-1], missing), cached, calendar, assets, read_json(rules.POLICY)
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("risk_history_sessions", 84),
        ("sample_covariance_ddof", 0),
        ("learning_rate_or_history_changed", True),
        ("cached_learning_state_reexecuted", True),
        ("covariance_regularization_or_variance_floor", True),
        ("full_covariance_ERC_or_global_solver_invoked", True),
        ("main_drawdown_guarantee", True),
        ("normalized_source_regret_transfers_to_sized_capital", True),
        ("new_source_RF_views", 1),
        ("actual_weight_trade_band", 0.05),
        ("independent_numeric_tolerance", 1e-8),
    ],
)
def test_no_window_rate_risk_floor_source_or_financial_gate_rescue(key, value):
    with pytest.raises(QuantError):
        rules.validate({**read_json(rules.POLICY), key: value})


def test_actual_singleton_registration_precedes_new_risk_sizing(monkeypatch):
    function = prior_tests.test_actual_singleton_registration_precedes_all_new_learning
    FunctionType(
        function.__code__,
        {**function.__globals__, "study": study, "rules": rules},
        argdefs=function.__defaults__,
    )(monkeypatch)
