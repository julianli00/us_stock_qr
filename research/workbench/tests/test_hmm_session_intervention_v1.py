import copy

import numpy as np
import pandas as pd
import pytest

from us_quant.config import QuantError
from us_quant.hmm_session_intervention_v1 import inspect_intervention

ASSETS = [
    "SPY",
    "IEF",
    "GLD",
    "BIL",
    "MTUM",
    "VLUE",
    "QUAL",
    "USMV",
    "TLT",
    "QQQ",
    "XLK",
    "SOXX",
    "IJR",
    "PKW",
]


def case():
    old_ids = [f"expert{i}" for i in range(8)]
    new_ids = old_ids[:]
    new_ids[2] = "session_expert"
    spec = {
        "base": {"expert_ids": old_ids, "candidate_ids": ["equal0", "rolling0"]},
        "current": {"expert_ids": new_ids, "candidate_ids": ["equal1", "rolling1"]},
        "assets": ASSETS,
        "slot": 2,
        "core_assets": ASSETS[4:8],
    }

    def pair(ids, style, weights):
        opinions = np.zeros((8, 14))
        opinions[:, 0] = 0.98
        opinions[2] = 0
        opinions[2, style] = 0.98
        before = {
            "record": "decision_input",
            "decision_index": 0,
            "as_of": "2020-01-31",
            "information_clock": "declared_after_close",
            "binary64_values_truncated_to_fixed_decimals": False,
            "previous_weights": [0.125] * 8,
            "observations": [
                {
                    "id": ids[i],
                    "as_of": "2020-01-31",
                    "target_known_at": "2020-01-31",
                    "asset_order": ASSETS,
                    "target": opinions[i].tolist(),
                    "mechanism_group": "conditional_variance_growth" if i == 2 else "other",
                    "RF": [0.0] * 126,
                    "return_dates": pd.bdate_range(end="2020-01-31", periods=126)
                    .strftime("%Y-%m-%d")
                    .tolist(),
                    "NET_returns": [0.01] * 126,
                    "score_mature": True,
                }
                for i in range(8)
            ],
        }
        names = spec["base" if ids == old_ids else "current"]["candidate_ids"]
        after = {
            "record": "decision_resolved",
            "decision_index": 0,
            "asset_order": ASSETS,
            "audit": {
                "decision_session": "2020-01-31",
                "decision_information_clock": "declared_after_close",
                "previous_rolling_expert_weights": [0.125] * 8,
                "equal_expert_weights": [0.125] * 8,
                "rolling_expert_weights": weights,
                "selection": {"scores_available": True, "selected_pair": [ids[2], ids[3]]},
                "shadow_wealth_added_to_main_capital": False,
                "source_feedback_floor_transferred_to_main_account": False,
            },
            "combined_targets": {
                names[0]: (np.full(8, 0.125) @ opinions).tolist(),
                names[1]: (np.asarray(weights) @ opinions).tolist(),
            },
        }
        return before, after

    return (
        pair(old_ids, 4, [0, 0, 0.5, 0.5, 0, 0, 0, 0]),
        pair(new_ids, 5, [0, 0, 0.25, 0.75, 0, 0, 0, 0]),
        spec,
    )


def test_exact_three_channel_identity_and_noncore_routing():
    result = inspect_intervention(*case())
    equal, rolling = result["laws"]["equal"], result["laws"]["rolling"]
    np.testing.assert_allclose(
        equal["channels"]["direct_opinion"]["vector"][4:6], [-0.1225, 0.1225], rtol=0, atol=1e-15
    )
    assert equal["channels"]["allocation_revision"]["L1"] == 0
    assert equal["channels"]["interaction"]["L1"] == 0
    np.testing.assert_allclose(
        rolling["channels"]["direct_opinion"]["vector"][4:6], [-0.49, 0.49], rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(
        rolling["channels"]["interaction"]["vector"][4:6], [0.245, -0.245], rtol=0, atol=1e-15
    )
    assert rolling["channels"]["allocation_revision"]["noncore_L1"] > 0
    assert rolling["channels"]["direct_opinion"]["noncore_L1"] == 0
    assert rolling["channels"]["interaction"]["noncore_L1"] == 0
    assert rolling["exact_rational_identity_passed"]
    assert result["mapped_selected_pairs_equal"]
    assert not rolling["statistical_weights_exactly_equal"]
    assert not result["causal_alpha_or_main_PnL_attributed"]


@pytest.mark.parametrize(
    "kind",
    [
        "source_id",
        "other_NET",
        "source_RF",
        "known_at",
        "asset_order",
        "noncore",
        "combined",
        "weights",
        "pair",
        "maturity",
        "previous",
        "capital",
        "truncated",
        "index",
        "date",
        "equal_weights",
        "missing_rule",
        "nonfinite",
        "negative",
        "bool",
        "clock",
        "return_dates",
    ],
)
def test_record_tampering_is_rejected(kind):
    old, new, spec = copy.deepcopy(case())
    before, after = new
    if kind == "source_id":
        before["observations"][0]["id"] = "foreign"
    elif kind == "other_NET":
        before["observations"][0]["NET_returns"][0] += 0.01
    elif kind == "source_RF":
        before["observations"][2]["RF"][0] = 0.001
    elif kind == "known_at":
        before["observations"][2]["target_known_at"] = "2020-02-01"
    elif kind == "asset_order":
        after["asset_order"] = list(reversed(ASSETS))
    elif kind == "noncore":
        before["observations"][2]["target"][0] = 0.01
        before["observations"][2]["target"][5] -= 0.01
    elif kind == "combined":
        after["combined_targets"]["equal1"][0] -= 0.01
        after["combined_targets"]["equal1"][5] += 0.01
    elif kind == "weights":
        after["audit"]["rolling_expert_weights"][2] = 0.3
    elif kind == "pair":
        after["audit"]["selection"]["selected_pair"] = ["expert0", "expert1"]
    elif kind == "maturity":
        before["observations"][2]["score_mature"] = False
    elif kind == "previous":
        before["previous_weights"][0] = 0.2
    elif kind == "capital":
        after["audit"]["shadow_wealth_added_to_main_capital"] = True
    elif kind == "truncated":
        before["binary64_values_truncated_to_fixed_decimals"] = True
    elif kind == "index":
        after["decision_index"] = 1
    elif kind == "date":
        before["as_of"] = "2020-01-30"
    elif kind == "equal_weights":
        after["audit"]["equal_expert_weights"][0] = 0.2
    elif kind == "missing_rule":
        after["combined_targets"].pop("equal1")
    elif kind == "nonfinite":
        before["observations"][2]["target"][5] = float("nan")
    elif kind == "negative":
        before["observations"][2]["target"][4] = -0.01
    elif kind == "bool":
        before["observations"][2]["target"] = [True] * 14
    elif kind == "clock":
        after["audit"]["decision_information_clock"] = "earlier_clock"
    else:
        before["observations"][2]["return_dates"].pop()
    with pytest.raises(QuantError):
        inspect_intervention(old, new, spec)


def test_unscored_equal_state_is_not_a_successful_choice():
    old, new, spec = case()
    for before, after in (old, new):
        after["audit"]["rolling_expert_weights"] = [0.125] * 8
        after["audit"]["selection"] = {"scores_available": False, "selected_pair": None}
        for observation in before["observations"]:
            observation["score_mature"] = False
        candidates = spec["base" if before is old[0] else "current"]["candidate_ids"]
        after["combined_targets"][candidates[1]] = after["combined_targets"][candidates[0]][:]
    result = inspect_intervention(old, new, spec)
    assert not result["old_scored"] and not result["new_scored"]
    assert result["laws"]["rolling"]["channels"]["allocation_revision"]["L1"] == 0


def test_later_record_changes_cannot_change_an_earlier_decomposition():
    earlier = case()
    expected = inspect_intervention(*earlier)
    later = copy.deepcopy(case())
    later[1][0]["observations"][2]["target"][5] -= 0.01
    assert inspect_intervention(*earlier) == expected


def test_no_source_opinion_change_leaves_only_allocation_channel():
    old, new, spec = case()
    new[0]["observations"][2]["target"] = old[0]["observations"][2]["target"][:]
    opinions = np.asarray([o["target"] for o in new[0]["observations"]])
    new[1]["combined_targets"]["equal1"] = (np.full(8, 0.125) @ opinions).tolist()
    new[1]["combined_targets"]["rolling1"] = (
        np.asarray(new[1]["audit"]["rolling_expert_weights"]) @ opinions
    ).tolist()
    result = inspect_intervention(old, new, spec)
    for law in result["laws"].values():
        assert law["channels"]["direct_opinion"]["L1"] == 0
        assert law["channels"]["interaction"]["L1"] == 0


def test_frozen_policy_cannot_enable_financial_runs_or_widen_tolerance():
    from us_quant import ssd_hmm_session_intervention_v1 as study
    from us_quant.storage import read_json

    policy = read_json(study.POLICY)
    study.validate(policy)
    for key, value in (
        ("new_financial_evaluations", 1),
        ("target_reconciliation_tolerance", 1e-8),
        ("global_pool_changed", True),
    ):
        changed = {**policy, key: value}
        with pytest.raises(QuantError):
            study.validate(changed)


def test_diagnostic_dependency_closure_binds_all_new_scientific_inputs():
    from us_quant import ssd_hmm_session_intervention_v1 as study

    paths = {row["path"] for row in study.dependencies()}
    assert set(study.OWN).issubset(paths)
    assert "src/us_quant/saved_inclusion_execution_v1.py" in paths
