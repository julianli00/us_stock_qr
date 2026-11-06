from __future__ import annotations

from pathlib import Path
from types import FunctionType

from us_quant import conditional_volatility_quintet_v1 as prior
from us_quant import strategy_expert_pool_eight_v1 as admitted
from us_quant import strategy_expert_pool_v5 as calendar_source
from us_quant.config import QuantError
from us_quant.storage import read_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/benchmark-opportunity-quintet-v1.json"
SLOTS = (4, 5, 7, 0, 1)
EXPERT_IDS = tuple(admitted.EXPERT_IDS[i] for i in SLOTS)
MECHANISMS = tuple(admitted.MECHANISMS[i] for i in SLOTS)
CLOCK = admitted.CLOCK
PRIOR_HEAD = "3d13b71e0935bdc60678c7eddcc57721c02b4d226168cde314104bdd7997e3e9"
CANDIDATES = (
    {"id": "expert_pool_benchmark_opportunity_quintet_equal_v1", "weighting": "equal"},
    {
        "id": "expert_pool_benchmark_opportunity_quintet_rolling_pair_v1",
        "weighting": "train_erc_validate_net",
    },
)
choose_pair, combine = admitted.choose_pair, admitted.combine


def expected_policy():
    inherited = {k: v for k, v in read_json(prior.POLICY).items() if k != "volatility_source_slot"}
    return {
        **inherited,
        "study_id": "ssd_benchmark_opportunity_quintet_20261106_v1",
        "actual_native_event_seq": 185410,
        "prior_journal_sequence": 1393,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_completed_configurations": 295,
        "source_catalogue": {
            "path": (
                "reports/ssd-continuous-research-v1/benchmark-opportunity-quintet-20261106-v1/"
                "source-input-catalogue-v1.json"
            ),
            "sha256": "5b70de3f1318a3e42fe7ddc6b8ac440528be33e925238e6f7565fe644b6be733",
        },
        "original_source_slots": list(SLOTS),
        "joint_tail_source_slot": 1,
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "candidates": list(CANDIDATES),
    }


def validate(policy):
    expected = expected_policy()
    if policy != expected or any(type(policy[k]) is not type(v) for k, v in expected.items()):
        raise QuantError("Freeze the declared joint-tail role and unchanged ten-pair allocation.")


def bound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


def decision_sessions(calendar, policy):
    return bound(calendar_source.decision_sessions)(calendar, policy)


def build_targets(experts, calendar, universe, policy, before_decision=None, retain=None):
    validate(policy)
    if len(experts) != 5 or len(universe) != len(set(universe)):
        raise QuantError("Use exactly five complete admitted policies and unique coordinates.")
    return bound(calendar_source.build_targets)(
        experts, calendar, universe, policy, before_decision, retain
    )
