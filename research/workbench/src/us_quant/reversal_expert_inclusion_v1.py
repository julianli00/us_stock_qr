from __future__ import annotations

from pathlib import Path
from types import FunctionType

from us_quant import calendar_expert_inclusion_v1 as contract
from us_quant import strategy_expert_pool_eight_v1 as prior
from us_quant.config import QuantError
from us_quant.storage import read_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/reversal-expert-inclusion-v1.json"
PRIOR_HEAD = "c95cc89d7ebaa0997359bfe35703ed76895bb044c669ea8c77b2548ab829f37f"
EXPERT_IDS = (*prior.EXPERT_IDS, "monthly_style_reversal_v1")
MECHANISMS = (*prior.MECHANISMS, "monthly_cross_sectional_style_reversal")
CLOCK = prior.CLOCK
CANDIDATES = (
    {"id": "expert_pool_reversal_equal_v1", "weighting": "equal"},
    {"id": "expert_pool_reversal_rolling_pair_v1", "weighting": "train_erc_validate_net"},
)
choose_pair, combine = prior.choose_pair, prior.combine


def validate(policy):
    original = read_json(contract.POLICY)
    contract.validate(original)
    expected = {
        **{
            key: value
            for key, value in original.items()
            if key not in {"calendar_source", "calendar_financial_rejection_preserved"}
        },
        "study_id": "ssd_reversal_expert_inclusion_20261105_v1",
        "prior_journal_sequence": 969,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_completed_configurations": 259,
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "candidates": list(CANDIDATES),
        "reversal_financial_rejection_preserved": True,
        "reversal_source": {
            "schema_version": 1,
            "id": EXPERT_IDS[-1],
            "mechanism_group": MECHANISMS[-1],
            "introduced_at": "2026-11-05T06:13:35.953079+00:00",
            "registration": {
                "path": "evidence/monthly_style_reversal_financial_20261105_registration_v1.json",
                "sha256": "cc97de80a0dd2e41d11e0c54fa9053ea01fcf8031f2c86b6d57c44495529a75b",
            },
            "results": {
                "path": (
                    "reports/ssd-continuous-research-v1/"
                    "monthly-style-reversal-financial-20261105-v1/results.json"
                ),
                "sha256": "abc868d9ab9f765ad8b9b2b6a318d94e04f2ec9ac7c0a1293bacfb5fdff2d5aa",
            },
            "admission": {
                "path": (
                    "reports/ssd-continuous-research-v1/"
                    "monthly-style-reversal-admission-20261105-v1/technical-admission-v1.json"
                ),
                "sha256": "0d233fb7d7bcb363587be4caf398f4aec81d77cffa353f60ef63830f6a750c9c",
            },
            "completion": {
                "path": (
                    "reports/ssd-continuous-research-v1/"
                    "monthly-style-reversal-admission-20261105-v1/completion-v1.json"
                ),
                "sha256": "f3838f2d34e7ebfb06398953d79eb22421e524f74e9408198600627e34588b0a",
            },
        },
        "pool_scope": {
            "representatives": (
                "Preserve the exact original eight representatives; the sole addition in this "
                "study is the admitted monthly style-reversal policy."
            ),
            "admissible_nonselected": (
                "Other technically admitted calendar, credit and scoped macro policies are not "
                "part of this independently defined reversal-inclusion question. Related GJR "
                "and other existing nonselected variants retain their inherited status, not "
                "financial-performance exclusion."
            ),
            "unreviewed_archives": (
                "Inherited unreviewed archive snapshots remain unreviewed, not automatically "
                "enabled or declared invalid."
            ),
            "invalid_or_unadmitted_archives": (
                "Previously demonstrated technical-invalid, leveraged or source-unadmitted "
                "versions remain excluded; Term/RF/FXY/VNQ and terminal provider stops are "
                "not repaired."
            ),
            "selection_bias": (
                "259 prior completions are exposed. A mechanism hypothesis is not evidence "
                "of independent risk, future gain or untouched out-of-sample validation."
            ),
        },
    }
    if (
        not isinstance(policy, dict)
        or policy != expected
        or any(type(policy[key]) is not type(value) for key, value in expected.items())
    ):
        raise QuantError("Freeze one controlled reversal inclusion without changing an old law.")


def _bound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


def decision_sessions(calendar, policy):
    return _bound(prior.prior.decision_sessions)(calendar, policy)


def build_targets(experts, calendar, universe, policy, before_decision=None, retain=None):
    validate(policy)
    if (
        tuple(expert.identity.identifier for expert in experts) != EXPERT_IDS
        or tuple(expert.identity.mechanism_group for expert in experts) != MECHANISMS
        or len(universe) != len(set(universe))
    ):
        raise QuantError("Keep exactly the original eight plus the admitted reversal policy.")
    return _bound(prior.prior.build_targets)(
        experts, calendar, universe, policy, before_decision, retain
    )
