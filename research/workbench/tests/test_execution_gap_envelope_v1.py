import copy
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from us_quant import execution_gap_envelope_v1 as rules
from us_quant.cash_funded_accounting_v2 import rebalance
from us_quant.config import QuantError
from us_quant.storage import digest_json, read_json

DAY = pd.Timestamp("2020-03-10")


def nominal(bil=False):
    values = np.zeros(14)
    values[rules.BIL if bil else 0] = 0.98
    return values


def state(nav=10000.0, peak=10000.0, gross=0.0, day=DAY):
    values = np.zeros(14)
    values[0] = nav * gross
    return rules.MainMark(day, values, nav - values[0], nav, peak)


def paths(prefill=(1.0,), opening=(1.0,), delivery=(0.8,), bil=(1.0,)):
    n = len(prefill)
    dates = rules.sessions("2020-01-02", "2020-03-10")
    before, opened, marked = [np.ones((n, 14)) for _ in range(3)]
    before[:, 0], opened[:, 0] = prefill, opening
    marked[:, 0], marked[:, rules.BIL] = delivery, bil
    return rules.StressSample(
        DAY, dates[:n], dates[1 : n + 1], dates[2 : n + 2], before, opened, marked
    )


def test_exact_affine_intersection_retains_both_endpoint_witnesses():
    result = rules.affine_interval(np.array([-0.25, 0.75]), np.array([1.0, -1.0]))
    assert result["feasible"] is True
    assert rules.decode(result["lower"]) == Fraction(1, 4)
    assert rules.decode(result["upper"]) == Fraction(3, 4)
    assert result["lower_witness"] == 0 and result["upper_witness"] == 1
    assert result["largest_certified_nominal_fraction"] == 0.75
    assert result["display_fraction_is_issued_target_or_applied_allocation"] is False


@pytest.mark.parametrize(
    "a,b",
    [
        ([-1.0], [0.0]),
        ([-0.8, 0.2], [1.0, -1.0]),
        ([-2.0], [1.0]),
        ([-0.2], [-1.0]),
    ],
)
def test_empty_interval_never_selects_zero_BIL_or_another_success_default(a, b):
    result = rules.affine_interval(np.array(a), np.array(b))
    assert result["feasible"] is False
    assert result["largest_certified_nominal_fraction"] is None


def test_fresh_capital_finite_interval_matches_independent_scalar_certificate():
    mark, sample, target = state(), paths(), nominal()
    result = rules.resolve(target, mark, sample)
    assert result["status"] == "finite_interval_certified"
    assert 0 < result["interval"]["largest_certified_nominal_fraction"] < 1
    checked = rules.verify(target, mark, sample, result)
    assert checked["maximum_independent_scalar_money_error_USD"] <= 1e-6
    assert checked["exact_represented_interval_and_witness_verified"] is True
    assert checked["future_or_full_month_guarantee_verified"] is False


def test_prefill_holdings_can_breach_before_any_new_order_can_change_them():
    mark = state(gross=0.98)
    sample = paths(prefill=(0.8,), opening=(1.5,), delivery=(2.0,))
    result = rules.resolve(nominal(), mark, sample)
    assert result["status"] == "unavoidable_prefill_reference_breach"
    assert result["interval"] is None
    assert result["prefill_violation_witness"] == 0
    rules.verify(nominal(), mark, sample, result)


def test_current_saved_breach_is_not_hidden_by_a_later_recovery_scenario():
    mark = state(nav=8400.0)
    result = rules.resolve(nominal(), mark, paths(delivery=(2.0,)))
    assert result["status"] == "outside_at_current_mark"
    assert result["interval"] is None


def test_scenario_open_cost_bound_is_not_current_close_allowance():
    mark = state(gross=1.0)
    sample = paths(opening=(0.00000001,))
    result = rules.resolve(nominal(), mark, sample)
    assert result["status"] == "uncertified_conservative_cost_bound"
    assert result["interval"] is None
    rules.verify(nominal(), mark, sample, result)


def test_positive_lower_bound_can_be_necessary_and_BIL_is_not_automatically_safe():
    mark, sample = state(nav=8500.0), paths(delivery=(1.2,))
    result = rules.resolve(nominal(), mark, sample)
    assert result["status"] == "finite_interval_certified"
    assert rules.decode(result["interval"]["lower"]) > 0
    assert result["interval"]["largest_certified_nominal_fraction"] == 1
    rules.verify(nominal(), mark, sample, result)
    bil_sample = paths(delivery=(1.0,), bil=(0.8,))
    result = rules.resolve(nominal(bil=True), state(), bil_sample)
    assert result["status"] == "no_finite_sufficient_interval"
    assert result["interval"]["zero_slope_violation_witness"] == 0
    assert result["interval"]["largest_certified_nominal_fraction"] is None
    rules.verify(nominal(bil=True), state(), bil_sample, result)


@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_conservative_fee_and_noop_lower_mark_bound_against_synthetic_cash_engine(fraction):
    mark = state(gross=0.98)
    sample = paths(prefill=(1.0,), opening=(1.03,), delivery=(0.9,), bil=(0.999,))
    target = nominal()
    actual_target = target * fraction
    actual_target[rules.BIL] += 0.98 * (1 - fraction)
    a, b, pre, _, allowance = rules.coefficients(target, mark, sample)
    opening_values = mark.asset_values * sample.opening[0]
    values, cash, fee, _, orders = rebalance(opening_values, mark.cash, actual_target, 20.0, 1.0)
    delivered_NAV = float(np.sum(values * sample.fill_mark[0]) + cash)
    lower = float(a[0] + b[0] * fraction + 0.85 * max(mark.peak, pre[0]))
    assert 0 <= orders <= 14 and fee <= allowance[0] + 1e-6 and cash >= 0
    assert delivered_NAV + 1e-6 >= lower


def test_original_noop_does_not_invent_fee_or_break_conservative_mark_slack():
    mark, target = state(gross=0.98), nominal()
    sample = paths(delivery=(0.99,))
    values, cash, fee, turnover, orders = rebalance(mark.asset_values, mark.cash, target, 20.0, 1.0)
    assert fee == turnover == orders == 0
    a, b, pre, _, _ = rules.coefficients(target, mark, sample)
    lower = float(a[0] + b[0] + 0.85 * max(mark.peak, pre[0]))
    assert float(values @ sample.fill_mark[0] + cash) + 1e-6 >= lower


def test_tied_exact_witnesses_are_mathematically_valid_not_index_disagreements():
    mark, target = state(), nominal()
    sample = paths(prefill=(1, 1), opening=(1, 1), delivery=(0.8, 0.8), bil=(1, 1))
    result = rules.resolve(target, mark, sample)
    assert result["interval"]["upper_witness"] == 0
    result["interval"]["upper_witness"] = 1
    rules.verify(target, mark, sample, result)


@pytest.mark.parametrize("kind", ["endpoint", "display", "witness", "status", "claim", "clock"])
def test_corrupt_certificate_cannot_be_published_as_feasible(kind):
    mark, sample, target = state(), paths(), nominal()
    result = copy.deepcopy(rules.resolve(target, mark, sample))
    if kind == "endpoint":
        result["interval"]["upper"] = rules.encode(Fraction(1))
    elif kind == "display":
        result["interval"]["largest_certified_nominal_fraction"] = 1.0
    elif kind == "witness":
        result["interval"]["upper_witness"] = 999
    elif kind == "status":
        result["status"] = "success_unknown_future"
    elif kind == "claim":
        result["finite_reference_is_unknown_future_support_or_drawdown_guarantee"] = True
    else:
        result["as_of"] = "2021-01-01"
    with pytest.raises(QuantError):
        rules.verify(target, mark, sample, result)


def test_known_price_adapter_and_resolution_ignore_future_suffix():
    calendar = rules.sessions("2019-10-01", "2020-03-10")
    close = pd.DataFrame(
        100 + np.arange(len(calendar))[:, None] * np.linspace(0.01, 0.03, 14),
        index=calendar,
        columns=rules.ASSETS,
    )
    opening = close * 0.999
    cutoff = calendar[60]
    archive = rules.KnownPriceArchive(opening, close)
    original = archive.observe(cutoff)
    future_open, future_close = opening.copy(), close.copy()
    future_open.loc[future_open.index > cutoff] *= 100
    future_close.loc[future_close.index > cutoff] *= 0.01
    changed = rules.KnownPriceArchive(future_open, future_close).observe(cutoff)
    prefix = rules.KnownPriceArchive(opening.loc[:cutoff], close.loc[:cutoff]).observe(cutoff)
    for name in ("prefill", "opening", "fill_mark"):
        np.testing.assert_array_equal(getattr(original, name), getattr(changed, name))
        np.testing.assert_array_equal(getattr(original, name), getattr(prefix, name))
    mark = state(day=cutoff)
    assert digest_json(rules.resolve(nominal(), mark, original)) == digest_json(
        rules.resolve(nominal(), mark, prefix)
    )


@pytest.mark.parametrize("kind", ["future", "negative", "missing_asset", "wrong_cash"])
def test_invalid_state_or_path_is_an_error_not_an_infeasible_standby(kind):
    mark, sample, target = state(), paths(), nominal()
    if kind == "future":
        sample = replace(sample, ends=sample.ends + pd.Timedelta(days=365))
    elif kind == "negative":
        sample = replace(sample, fill_mark=np.full_like(sample.fill_mark, -1))
    elif kind == "missing_asset":
        target = target[:-1]
    else:
        mark = replace(mark, cash=9999.0)
    with pytest.raises(QuantError):
        rules.resolve(target, mark, sample)


@pytest.mark.parametrize(
    "field,value",
    [
        ("planning_delay_sessions", 1),
        ("planning_cost_bps_per_side", 5.0),
        ("prior_headroom_V1_repaired_retried_or_executed", True),
        ("certificate_covers_full_month_or_unknown_future_support", True),
        ("empty_interval_has_selected_fraction", True),
        ("new_financial_trials", 1),
    ],
)
def test_frozen_reference_scope_cannot_reopen_failed_policy_or_claim_future_support(field, value):
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    policy[field] = value
    with pytest.raises(QuantError):
        rules.validate(policy)
