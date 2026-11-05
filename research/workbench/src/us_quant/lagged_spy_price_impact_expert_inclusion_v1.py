from __future__ import annotations

from pathlib import Path
from types import FunctionType

from us_quant import fixed_four_sleeve_expert_inclusion_v1 as contract
from us_quant import strategy_expert_pool_eight_v1 as prior
from us_quant.config import QuantError
from us_quant.storage import read_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/lagged-spy-price-impact-expert-inclusion-v1.json"
PRIOR_HEAD = "badd700446385a406f10e6b2bf7aec74a812ed437d64fcd607191ab662fb68c6"
EXPERT_IDS = (*prior.EXPERT_IDS, "lagged_spy_price_impact_participation_v1")
MECHANISMS = (*prior.MECHANISMS, "lagged_market_price_impact_participation")
CLOCK = prior.CLOCK
CANDIDATES = (
    {"id": "expert_pool_lagged_spy_impact_equal_v1", "weighting": "equal"},
    {"id": "expert_pool_lagged_spy_impact_rolling_pair_v1", "weighting": "train_erc_validate_net"},
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
            not in {"fixed_four_sleeve_source", "fixed_four_sleeve_financial_rejection_preserved"}
        },
        "study_id": "ssd_lagged_spy_price_impact_expert_inclusion_20261105_v1",
        "prior_journal_sequence": 1039,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_completed_configurations": 265,
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "candidates": list(CANDIDATES),
        "lagged_price_impact_financial_rejection_preserved": True,
        "lagged_price_impact_source": {
            "schema_version": 1,
            "id": EXPERT_IDS[-1],
            "mechanism_group": MECHANISMS[-1],
            "introduced_at": "2026-11-05T10:00:18.384153+00:00",
            "registration": {
                "path": "evidence/lagged_spy_price_impact_financial_20261105_registration_v1.json",
                "sha256": "1617eb30797fbb393a1616a408a2108c332cb7ddaddfec804db20cbc9ed1fd56",
            },
            "results": {
                "path": (
                    "reports/ssd-continuous-research-v1/"
                    "lagged-spy-price-impact-financial-20261105-v1/results.json"
                ),
                "sha256": "553d23999a45de885e6b8fc8eecea7c5a8f094c8b79a3cf01d5723366175cee2",
            },
            "admission": {
                "path": (
                    "reports/ssd-continuous-research-v1/"
                    "lagged-spy-price-impact-admission-20261105-v1/technical-admission-v1.json"
                ),
                "sha256": "41a2faebbc27590d40cc933c91c77d7c00fc66e23733d290d0a4ccd44e861e8e",
            },
            "completion": {
                "path": (
                    "reports/ssd-continuous-research-v1/"
                    "lagged-spy-price-impact-admission-20261105-v1/completion-v1.json"
                ),
                "sha256": "21eb98fb351ae6d6231e66ec35dcab739372bffc4523f39f590ac978770c781b",
            },
        },
        "pool_scope": {
            "representatives": (
                "Preserve the exact original eight representatives; the sole addition in this "
                "study is the admitted lagged SPY price-impact participation policy."
            ),
            "admissible_nonselected": (
                "Other technically admitted fixed rebalancing, reversal, calendar, credit and "
                "scoped macro policies are not part of this independently defined price-impact "
                "question. Related GJR and other existing nonselected variants retain their "
                "inherited status, not financial-performance exclusion."
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
                "265 prior completions are exposed. A mechanism hypothesis is not evidence "
                "of independent risk, future gain or untouched out-of-sample validation."
            ),
        },
    }
    if (
        not isinstance(policy, dict)
        or policy != expected
        or any(type(policy[key]) is not type(value) for key, value in expected.items())
    ):
        raise QuantError("Freeze one controlled price-impact inclusion without old-law changes.")


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
        raise QuantError("Keep exactly the original eight plus the admitted price-impact source.")
    return _bound(prior.prior.build_targets)(
        experts, calendar, universe, policy, before_decision, retain
    )
