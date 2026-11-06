import copy
import json
from dataclasses import replace
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import test_expert_hierarchical_risk_v1 as old_tests
import test_median_equity_budget_v1 as singleton_tests
from test_conditional_volatility_quintet_v1 import sources

from us_quant import expert_hierarchical_risk_v1 as old
from us_quant import five_hierarchical_risk_v1 as rules
from us_quant import ssd_five_hierarchical_risk_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json


def test_exact_fixed_five_policy_and_original_risk_parameters():
    rules.validate(read_json(rules.POLICY))
    assert len(rules.EXPERT_IDS) == 5 and len(study.IDS) == 1
    assert read_json(rules.POLICY)["lookback_sessions"] == 126
    assert read_json(rules.POLICY)["risk_fit_is_heldout_validation"] is False
    assert read_json(rules.POLICY)["median_equity_budget_promoted_or_retuned"] is False


@pytest.mark.parametrize("case", ["random", "orthogonal", "singular"])
def test_generic_six_coordinate_kernel_matches_preserved_original(case):
    if case == "random":
        values = np.random.default_rng(184317).normal(size=(126, 6))
    elif case == "orthogonal":
        values = old_tests.orthogonal() * np.arange(1, 7)
    else:
        values = np.arange(126, dtype=float)[:, None] * np.ones((1, 6))
    weights, audit = rules.hierarchical_weights(values)
    previous, original_audit = old.hierarchical_weights(values)
    assert np.array_equal(weights, previous) and audit == original_audit
    assert np.array_equal(rules.sample_covariance(values), old.sample_covariance(values))


def test_five_independent_risks_allocate_inverse_variance_without_global_solver():
    values = old_tests.orthogonal()[:, :5] * np.arange(1, 6)
    weights, audit = rules.hierarchical_weights(values)
    expected = 1 / np.arange(1, 6, dtype=float) ** 2
    expected /= expected.sum()
    assert np.max(abs(weights - expected)) <= 1e-12
    assert len(audit["canonical_merges"]) == len(audit["recursive_capital_splits"]) == 4
    assert not audit["variance_floor_or_regularization_applied"]
    assert not audit["global_optimizer_or_previous_failed_solver_used"]
    scaled, _ = rules.hierarchical_weights(values * 17)
    assert np.max(abs(scaled - weights)) <= 1e-12


def test_five_singular_correlations_keep_deterministic_coordinate_ties():
    values = np.arange(126, dtype=float)[:, None] * np.ones((1, 5))
    first, a = rules.hierarchical_weights(values)
    second, b = rules.hierarchical_weights(values)
    assert np.array_equal(first, second) and a == b
    assert a["canonical_merges"][0]["left_members"] == [0]
    assert a["canonical_merges"][0]["right_members"] == [1]
    assert sorted(a["seriated_expert_coordinates"]) == list(range(5))
    assert np.max(abs(first - np.array([0.25, 0.25, 0.25, 0.125, 0.125]))) <= 1e-12
    assert not a["risk_clusters_are_independent_factor_families"]


def test_original_exact_zero_child_behavior_is_dimension_preserved(monkeypatch):
    f = old_tests.test_zero_cluster_risk_is_explicit_not_a_safety_claim
    FunctionType(f.__code__, {**f.__globals__, "rules": rules}, argdefs=f.__defaults__)(monkeypatch)


def test_one_five_coordinate_zero_risk_child_receives_parent():
    values = old_tests.orthogonal()[:, 0]
    weights, audit = rules.hierarchical_weights(np.column_stack([values] * 4 + [-values]))
    zero = [s for s in audit["recursive_capital_splits"] if s["exactly_one_child_risk_zero"]]
    assert zero and any(s["right_IVP_variance"] == 0 for s in zero)
    assert np.isfinite(weights).all() and (weights >= 0).all()


@pytest.mark.parametrize("case", ["shape", "asymmetric", "nonPSD", "nonfinite", "zero"])
def test_invalid_five_covariance_is_not_repaired(case):
    covariance = np.eye(5)
    if case == "shape":
        covariance = np.ones((5, 4))
    elif case == "asymmetric":
        covariance[0, 1] = 0.5
    elif case == "nonPSD":
        covariance[0, 1] = covariance[1, 0] = 2
    elif case == "nonfinite":
        covariance[0, 0] = np.nan
    else:
        covariance[0, 0] = 0
    with pytest.raises(QuantError):
        rules.cluster_order(covariance)


def test_unmatured_and_zero_individual_risk_remain_explicitly_unscored():
    experts, calendar, assets = sources()
    targets, audits = rules.build_targets(experts, calendar, assets, read_json(rules.POLICY))
    assert targets[rules.ID].index.equals(calendar)
    assert audits[0]["decision_session"] == "2016-10-05"
    assert sum(a["allocation"]["reason"] == "unscored_initial_equal_weights" for a in audits) == 7
    observed = [e.observe(calendar[-1], 126) for e in experts]
    observed[0] = replace(observed[0], net_returns=observed[0].risk_free.copy())
    weights, audit = rules.choose_allocation(observed, np.full(5, 0.2))
    assert np.array_equal(weights, np.full(5, 0.2))
    assert audit["reason"] == "unscored_nonpositive_individual_variance_equal_weights"
    assert audit["risk_estimate_available"] is False and audit["safety_or_success_claimed"] is False


@pytest.mark.parametrize("damage", ["identity", "future", "RF", "date", "nonfinite"])
def test_inadmissible_source_clock_or_exact_RF_is_refused(damage):
    experts, calendar, _ = sources()
    observed = [e.observe(calendar[-1], 126) for e in experts]
    if damage == "identity":
        observed.reverse()
    elif damage == "future":
        observed[0] = replace(observed[0], target_known_at=calendar[-1] + pd.Timedelta(days=1))
    elif damage == "RF":
        observed[0] = replace(observed[0], risk_free=observed[0].risk_free + 1e-8)
    elif damage == "date":
        observed[0] = replace(observed[0], net_returns=observed[0].net_returns.iloc[1:])
    else:
        net = observed[0].net_returns.copy()
        net.iloc[0] = np.nan
        observed[0] = replace(observed[0], net_returns=net)
    with pytest.raises(QuantError):
        rules.choose_allocation(observed, np.full(5, 0.2))


def test_future_source_changes_and_prefix_do_not_change_prior_allocations():
    cutoff = pd.Timestamp("2017-04-28")
    experts, calendar, assets = sources()
    changed, _, _ = sources(changed_after=cutoff)
    policy = read_json(rules.POLICY)
    full, a = rules.build_targets(experts, calendar, assets, policy)
    different, b = rules.build_targets(changed, calendar, assets, policy)
    early, c = rules.build_targets(experts, calendar[calendar <= cutoff], assets, policy)
    assert full[rules.ID].loc[:cutoff].equals(different[rules.ID].loc[:cutoff])
    assert full[rules.ID].loc[:cutoff].equals(early[rules.ID])
    assert c == [r for r in a if r["decision_session"] <= str(cutoff.date())]
    assert c == [r for r in b if r["decision_session"] <= str(cutoff.date())]


@pytest.mark.parametrize("damage", ["covariance", "merge", "split", "heldout"])
def test_full_independent_hierarchy_and_retained_input_checks(tmp_path, damage):
    experts, calendar, assets = sources()
    rows = []
    targets, audits = rules.build_targets(
        experts, calendar, assets, read_json(rules.POLICY), retain=rows.append
    )
    data = SimpleNamespace(close=pd.DataFrame(index=calendar, columns=assets))
    checked = study.verify_targets(data, experts, targets, audits)
    assert checked["maximum_independent_target_difference"] <= 1e-12
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert study.verify_retained_records(path, experts, targets, audits)[
        "actual_input_and_resolved_record_count"
    ] == 2 * len(audits)
    changed = copy.deepcopy(audits)
    row = next(a["allocation"] for a in changed if a["allocation"]["risk_estimate_available"])
    if damage == "covariance":
        row["covariance"][0][0] += 0.0001
    elif damage == "merge":
        row["canonical_merges"][0]["distance_key"] += 0.01
    elif damage == "split":
        row["recursive_capital_splits"][0]["both_child_risks_zero"] = not (
            row["recursive_capital_splits"][0]["both_child_risks_zero"]
        )
    else:
        row["risk_fit_is_heldout_validation"] = True
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, targets, changed)
    rows[0]["observations"][0]["target_known_at"] = "2099-01-01"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(QuantError):
        study.verify_retained_records(path, experts, targets, audits)


@pytest.mark.parametrize(
    "key,value",
    [
        ("lookback_sessions", 252),
        ("sample_covariance_ddof", 0),
        ("linkage", "complete"),
        ("distance_key_decimal_places", 10),
        ("correlation_roundoff_tolerance", 1e-10),
        ("risk_fit_is_heldout_validation", True),
        ("return_score_or_rank_used", True),
        ("covariance_regularization_or_variance_floor", True),
        ("median_equity_budget_promoted_or_retuned", True),
        ("actual_weight_trade_band", 0.05),
        ("new_source_RF_views", 1),
        ("global_pool_changed", True),
    ],
)
def test_no_risk_window_source_solver_or_financial_waiver(key, value):
    with pytest.raises(QuantError):
        rules.validate({**read_json(rules.POLICY), key: value})


def test_real_single_registration_precedes_new_risk_target(monkeypatch):
    f = singleton_tests.test_single_registration_precedes_budget_and_cannot_be_replayed
    FunctionType(
        f.__code__, {**f.__globals__, "rules": rules, "study": study}, argdefs=f.__defaults__
    )(monkeypatch)


@pytest.mark.parametrize("cohort", [[rules.ID], [], ["foreign"]])
def test_actual_complete_singleton_is_checked_before_publication(monkeypatch, cohort):
    f = singleton_tests.test_complete_actual_singleton_packet_precedes_publication
    FunctionType(
        f.__code__, {**f.__globals__, "rules": rules, "study": study}, argdefs=f.__defaults__
    )(monkeypatch, cohort)
