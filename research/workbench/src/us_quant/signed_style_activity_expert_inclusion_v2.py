from __future__ import annotations

from pathlib import Path
from types import FunctionType

from us_quant import style_activity_expert_inclusion_v1 as preserved
from us_quant.config import QuantError
from us_quant.storage import read_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/signed-style-activity-expert-inclusion-v2.json"
PRIOR_HEAD = preserved.PRIOR_HEAD
prior = preserved.prior
EXPERT_IDS, MECHANISMS, CLOCK = preserved.EXPERT_IDS, preserved.MECHANISMS, preserved.CLOCK
CANDIDATES = (
    {"id": "expert_pool_signed_activity_nine_equal_v2", "weighting": "equal"},
    {
        "id": "expert_pool_signed_activity_nine_rolling_pair_v2",
        "weighting": "train_erc_validate_net",
    },
)
choose_pair, combine = preserved.choose_pair, preserved.combine


def validate(policy):
    original = read_json(preserved.POLICY)
    preserved.validate(original)
    expected = {
        **original,
        "schema_version": 2,
        "study_id": "ssd_signed_style_activity_expert_inclusion_20261105_v2",
        "candidates": list(CANDIDATES),
        "namespace_revision_only": True,
        "preserved_unregistered_collision": {
            "path": "evidence/style_activity_inclusion_identity_collision_20261105_v1.json",
            "sha256": "365475d644a35351dac3d3017783333518c0496e8f0cc6eac8016605f139bca1",
        },
    }
    if (
        not isinstance(policy, dict)
        or policy != expected
        or any(type(policy[k]) is not type(v) for k, v in expected.items())
    ):
        raise QuantError(
            "Only version the unused identity namespace; preserve every scientific law."
        )


def _bound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


def decision_sessions(calendar, policy):
    return _bound(preserved.decision_sessions)(calendar, policy)


def build_targets(experts, calendar, universe, policy, before_decision=None, retain=None):
    return _bound(preserved.build_targets)(
        experts, calendar, universe, policy, before_decision, retain
    )
