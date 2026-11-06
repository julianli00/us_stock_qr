import copy

import numpy as np
import pytest

from us_quant import equal_role_projection_v1 as rules
from us_quant.config import QuantError
from us_quant.storage import read_json


def test_uniform_source_preserves_initial_sixths():
    actual, diagnostic = rules.project([1 / 6] * 6)
    np.testing.assert_array_equal(actual, np.full(6, 1 / 6))
    assert diagnostic["projected_role_masses"] == [0.5, 0.5]


@pytest.mark.parametrize(
    "values",
    [
        [0.3, 0.1, 0.1, 0.2, 0.2, 0.1],
        [0.9, 0.02, 0.02, 0.03, 0.02, 0.01],
        [0.0, 0.0, 0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 1e-300, 0.25, 0.25, 1e-300],
    ],
)
def test_half_budgets_ratios_and_independent_target(values):
    actual, _ = rules.project(values)
    opinions = np.zeros((6, 14))
    for i in range(6):
        opinions[i, i] = 0.98
    checked = rules.independent(values, actual, opinions, actual @ opinions)
    assert checked["exact_half_role_masses"] is True
    for group in rules.GROUPS:
        assert abs(sum(actual[list(group)]) - 0.5) <= 1e-12
        for i in group:
            for j in group:
                assert abs(actual[i] * values[j] - actual[j] * values[i]) <= 1e-12


@pytest.mark.parametrize(
    "values",
    [
        [0.4, 0.0, 0.0, 0.3, 0.3, 0.0],
        [0.0, 0.3, 0.3, 0.0, 0.0, 0.4],
        [0.2] * 5,
        [0.1] * 6,
        [float("nan"), 0.2, 0.2, 0.2, 0.1, 0.1],
        [float("inf"), 0.2, 0.2, 0.2, 0.1, 0.1],
        [-0.1, 0.3, 0.2, 0.2, 0.2, 0.2],
        [True, 0.0, 0.0, 0.0, 0.0, 0.0],
        ["0.2", 0.2, 0.2, 0.2, 0.1, 0.1],
    ],
)
def test_invalid_or_zero_role_mass_stops_without_fallback(values):
    with pytest.raises(QuantError):
        rules.project(values)


def test_independent_verifier_rejects_bad_projection():
    values = [0.3, 0.1, 0.1, 0.2, 0.2, 0.1]
    weights, _ = rules.project(values)
    opinions = np.zeros((6, 14))
    opinions[:, 0] = 0.98
    weights[0] += 0.01
    with pytest.raises(QuantError):
        rules.independent(values, weights, opinions, weights @ opinions)


def test_prefix_is_independent_of_future_record_changes():
    records = [[1 / 6] * 6, [0.3, 0.1, 0.1, 0.2, 0.2, 0.1]]
    expected = [rules.project(r)[0].tolist() for r in records]
    changed = copy.deepcopy(records) + [[0.4, 0, 0, 0.3, 0.3, 0]]
    assert [rules.project(r)[0].tolist() for r in changed[:2]] == expected


@pytest.mark.parametrize(
    "key,value",
    [
        ("role_budgets", [0.4, 0.6]),
        ("growth_positions", [0, 1, 2]),
        ("source_learning_state_reexecuted", True),
        ("independent_weight_and_target_tolerance", 1e-9),
    ],
)
def test_frozen_role_and_source_rules_are_not_tunable(key, value):
    policy = read_json(rules.POLICY)
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)
