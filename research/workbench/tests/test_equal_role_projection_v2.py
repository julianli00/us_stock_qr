import inspect

import pytest

from us_quant import equal_role_recovery_policy_v2 as rules
from us_quant import ssd_equal_role_projection_v2 as program
from us_quant.config import QuantError
from us_quant.storage import read_json


def test_new_identity_and_original_numeric_rule():
    rules.validate(read_json(rules.POLICY))
    assert rules.ID != program.failed.rules.ID
    assert rules.CANDIDATES[0]["weighting"] == program.failed.rules.CANDIDATES[0]["weighting"]
    assert program.FOLDER != program.failed.FOLDER
    assert program.PACKETS[rules.ID] != program.failed.PACKETS[program.failed.rules.ID]
    assert list(inspect.signature(program.proof.target_proof).parameters) == ["frozen", "fields"]
    assert list(inspect.signature(program.proof.verify_cached).parameters) == [
        "targets",
        "records",
        "frozen",
    ]


@pytest.mark.parametrize(
    "key,value",
    [
        ("economic_rule_or_tolerance_changed", True),
        ("failed_V1_reopened", True),
        ("cached_targets_verified_not_regenerated", False),
    ],
)
def test_no_source_or_criterion_rescue(key, value):
    policy = read_json(rules.POLICY)
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)
