from __future__ import annotations

from pathlib import Path
from types import FunctionType

from us_quant import reversal_expert_inclusion_v1 as contract
from us_quant import strategy_expert_pool_eight_v1 as prior
from us_quant.config import QuantError
from us_quant.storage import read_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/fixed-four-sleeve-expert-inclusion-v1.json"
PRIOR_HEAD = "0c3e4dad66f3ba04914e34da028cd935f683e226478764813427270409a13187"
EXPERT_IDS = (*prior.EXPERT_IDS, "fixed_four_sleeve_rebalance_v1")
MECHANISMS = (*prior.MECHANISMS, "fixed_cross_asset_capital_rebalancing")
CLOCK = prior.CLOCK
CANDIDATES = (
    {"id": "expert_pool_fixed_four_sleeve_equal_v1", "weighting": "equal"},
    {"id": "expert_pool_fixed_four_sleeve_rolling_pair_v1", "weighting": "train_erc_validate_net"},
)
choose_pair, combine = prior.choose_pair, prior.combine


def validate(policy):
    original = read_json(contract.POLICY)
    contract.validate(original)
    expected = {
        **{
            key: value
            for key, value in original.items()
            if key not in {"reversal_source", "reversal_financial_rejection_preserved"}
        },
        "study_id": "ssd_fixed_four_sleeve_expert_inclusion_20261105_v1",
        "prior_journal_sequence": 1004,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "prior_completed_configurations": 262,
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "candidates": list(CANDIDATES),
        "fixed_four_sleeve_financial_rejection_preserved": True,
        "fixed_four_sleeve_source": {
            "schema_version": 1,
            "id": EXPERT_IDS[-1],
            "mechanism_group": MECHANISMS[-1],
            "introduced_at": "2026-11-05T08:15:18.055890+00:00",
            "registration": {
                "path": "evidence/fixed_four_sleeve_financial_20261105_registration_v1.json",
                "sha256": "fc2d6f314a07d51781019e5f4b9d9ea04c253493bb7ee0b6447ef3db1754ef33",
            },
            "results": {
                "path": (
                    "reports/ssd-continuous-research-v1/"
                    "fixed-four-sleeve-financial-20261105-v1/results.json"
                ),
                "sha256": "f292827e4015d8c63156d4e10cdcc2857db430e76bd4c80b34161d50086a1049",
            },
            "admission": {
                "path": (
                    "reports/ssd-continuous-research-v1/"
                    "fixed-four-sleeve-admission-20261105-v1/technical-admission-v1.json"
                ),
                "sha256": "520f909e40df673ced413c675ef30996249d3d6e0843a63515f951632171701b",
            },
            "completion": {
                "path": (
                    "reports/ssd-continuous-research-v1/"
                    "fixed-four-sleeve-admission-20261105-v1/completion-v1.json"
                ),
                "sha256": "7223855fe00644e95e4e6c0406ae4b9cad1c43a771cd5b0dd836e661d5ea09f1",
            },
        },
        "pool_scope": {
            "representatives": (
                "Preserve the exact original eight representatives; the sole addition in this "
                "study is the admitted forecast-free fixed four-bucket capital-rebalancing policy."
            ),
            "admissible_nonselected": (
                "Other technically admitted reversal, calendar, credit and scoped macro policies "
                "are not part of this independently defined fixed-rebalancer question. Related "
                "GJR and other existing nonselected variants retain their inherited status, not "
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
                "262 prior completions are exposed. A mechanism hypothesis is not evidence "
                "of independent risk, future gain or untouched out-of-sample validation."
            ),
        },
    }
    if (
        not isinstance(policy, dict)
        or policy != expected
        or any(type(policy[key]) is not type(value) for key, value in expected.items())
    ):
        raise QuantError(
            "Freeze one controlled fixed-rebalancer inclusion without old-law changes."
        )


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
        raise QuantError("Keep exactly the original eight plus the admitted fixed rebalancer.")
    return _bound(prior.prior.build_targets)(
        experts, calendar, universe, policy, before_decision, retain
    )
