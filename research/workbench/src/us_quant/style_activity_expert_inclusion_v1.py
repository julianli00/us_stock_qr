from __future__ import annotations

from pathlib import Path
from types import FunctionType

from us_quant import lagged_spy_price_impact_expert_inclusion_v1 as contract
from us_quant.config import QuantError
from us_quant.storage import read_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/style-activity-expert-inclusion-v1.json"
PRIOR_HEAD = "6173e5423b7aa42b8c6b61c1a2c8d80ed6a488a03e129666582a1db9155ef3ea"
prior = contract.prior
EXPERT_IDS = (*prior.EXPERT_IDS, "signed_style_activity_participation_v1")
MECHANISMS = (*prior.MECHANISMS, "cross_sectional_reported_activity_participation")
CLOCK = prior.CLOCK
CANDIDATES = (
    {"id": "expert_pool_style_activity_equal_v1", "weighting": "equal"},
    {"id": "expert_pool_style_activity_rolling_pair_v1", "weighting": "train_erc_validate_net"},
)
choose_pair, combine = prior.choose_pair, prior.combine


def validate(policy):
    original = read_json(contract.POLICY)
    contract.validate(original)
    expected = {
        **{
            key: value
            for key, value in original.items()
            if key
            not in {
                "lagged_price_impact_source",
                "lagged_price_impact_financial_rejection_preserved",
            }
        },
        "study_id": "ssd_style_activity_expert_inclusion_20261105_v1",
        "prior_journal_sequence": 1121,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_completed_configurations": 272,
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "candidates": list(CANDIDATES),
        "signed_style_activity_financial_rejection_preserved": True,
        "style_activity_source": {
            "schema_version": 1,
            "id": EXPERT_IDS[-1],
            "mechanism_group": MECHANISMS[-1],
            "introduced_at": "2026-11-05T19:08:36.964729+00:00",
            "registration": {
                "path": "evidence/style_activity_financial_20261105_registration_v1.json",
                "sha256": "b3514e1fab0c4768ddef2ebb21bc0a180b4f9c8478a78d1f26319f7b32cdc27b",
            },
            "results": {
                "path": (
                    "reports/ssd-continuous-research-v1/"
                    "style-activity-participation-financial-20261105-v1/results.json"
                ),
                "sha256": "095096f76d37564b333678399c826e7284e6b90d9260d03be5312a19b39f0822",
            },
            "admission": {
                "path": "evidence/style_activity_financial_20261105_admission_v1.json",
                "sha256": "20f7d1274d896de0dbc7e4cf0845155f908fe00bf7aabffea7d024dfc1d197bc",
            },
            "completion": {
                "path": (
                    "reports/ssd-continuous-research-v1/"
                    "style-activity-admission-20261105-v1/completion-v1.json"
                ),
                "sha256": "344c6dca83e66a5647519f54a64cc362915cfe024460f512f2b72f7367a783d6",
            },
        },
        "pool_scope": {
            "representatives": (
                "Preserve the exact original eight representatives; the sole addition in this "
                "study is the admitted signed-style-activity complete participation policy."
            ),
            "admissible_nonselected": (
                "Other technically admitted lagged SPY impact, fixed rebalancing, reversal, "
                "calendar, credit and scoped macro policies are not part of this separately "
                "defined signed-activity question. Related GJR and other nonselected variants "
                "retain their inherited status, not financial-performance exclusion."
            ),
            "unreviewed_archives": original["pool_scope"]["unreviewed_archives"],
            "invalid_or_unadmitted_archives": original["pool_scope"][
                "invalid_or_unadmitted_archives"
            ],
            "selection_bias": (
                "272 prior completions and this source's financial rejection are exposed. "
                "Its participation role was defined before its standalone outcomes; this "
                "later controlled roster choice is not independent risk, future gain or "
                "untouched out-of-sample validation."
            ),
        },
    }
    if (
        not isinstance(policy, dict)
        or policy != expected
        or any(type(policy[key]) is not type(value) for key, value in expected.items())
    ):
        raise QuantError("Freeze one controlled signed-activity inclusion without old-law changes.")


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
        raise QuantError("Use exactly original eight plus the admitted signed-activity source.")
    return _bound(prior.prior.build_targets)(
        experts, calendar, universe, policy, before_decision, retain
    )
