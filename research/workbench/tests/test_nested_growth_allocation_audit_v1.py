import copy
import json
from types import FunctionType, SimpleNamespace

import numpy as np
import pytest

from us_quant import nested_growth_allocation_audit_v1 as rules
from us_quant import ssd_nested_growth_allocation_audit_v1 as study
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.storage import file_digest, read_json


def fixture(index=7, child_pair=(0, 5), direct=1, use_growth=True):
    calendar = sessions("2016-10-05", "2017-05-31")
    origins = [calendar[0], *[d for d in calendar if d > calendar[0] and is_month_end(d)]]
    day = str(origins[index].date())
    mature = index >= 7
    observed_dates = calendar[(calendar >= "2016-10-06") & (calendar <= origins[index])][-126:]
    children = []
    for i, identifier in enumerate(rules.CANONICAL_IDS):
        target = np.zeros(len(rules.ASSETS))
        target[i] = 0.98
        children.append(
            {
                "id": identifier,
                "mechanism_group": rules.canonical.MECHANISMS[i],
                "introduced_at": "2026-11-06T00:00:00Z",
                "as_of": day,
                "target_known_at": day,
                "target": target.tolist(),
                "asset_order": list(rules.ASSETS),
                "return_dates": observed_dates.strftime("%Y-%m-%d").tolist(),
                "NET_returns": [0.0002 + i * 0.00001] * len(observed_dates),
                "RF": [0.00003] * len(observed_dates),
                "score_mature": mature,
            }
        )
    child_weights = np.zeros(8) if mature else np.full(8, 0.125)
    if mature:
        child_weights[list(child_pair)] = [0.4, 0.6]
    child_target = child_weights @ np.array([o["target"] for o in children])
    parents = [copy.deepcopy(children[i]) for i in rules.DIRECT_SLOTS]
    parents.append(
        {
            **copy.deepcopy(children[0]),
            "id": rules.OUTER_IDS[-1],
            "mechanism_group": rules.outer.admitted.GROUP,
            "target": child_target.tolist(),
            "NET_returns": [0.0007] * len(observed_dates),
        }
    )
    rolling = np.full(4, 0.25) if not mature else np.zeros(4)
    if mature and use_growth:
        rolling[direct], rolling[-1] = 0.3, 0.7
    elif mature:
        rolling[0], rolling[1] = 0.6, 0.4
    selected = (
        sorted(rules.OUTER_IDS[i] for i, v in enumerate(rolling) if v > 0) if mature else None
    )
    child_selected = sorted(rules.CANONICAL_IDS[i] for i in child_pair) if mature else None
    parent_targets = np.array([o["target"] for o in parents])
    return {
        "outer_before": {
            "record": "decision_input",
            "as_of": day,
            "decision_index": index,
            "information_clock": rules.CLOCK,
            "observations": parents,
            "previous_weights": [0.25] * 4,
            "previous_weights_binary64_hex": [0.25.hex()] * 4,
            "binary64_values_truncated_to_fixed_decimals": False,
        },
        "outer_after": {
            "record": "decision_resolved",
            "decision_index": index,
            "asset_order": list(rules.ASSETS),
            "audit": {
                "decision_session": day,
                "decision_information_clock": rules.CLOCK,
                "previous_rolling_expert_weights": [0.25] * 4,
                "equal_expert_weights": [0.25] * 4,
                "rolling_expert_weights": rolling.tolist(),
                "selection": {"scores_available": mature, "selected_pair": selected},
                "shadow_wealth_added_to_main_capital": False,
                "source_feedback_floor_transferred_to_main_account": False,
            },
            "combined_targets": {
                rules.IDS[0]: (np.full(4, 0.25) @ parent_targets).tolist(),
                rules.IDS[1]: (rolling @ parent_targets).tolist(),
            },
        },
        "child_before": {
            "record": "saved_forecast_allocation_input",
            "source_before": {
                "as_of": day,
                "decision_index": index,
                "information_clock": rules.CLOCK,
                "observations": children,
                "binary64_values_truncated_to_fixed_decimals": False,
            },
        },
        "child_after": {
            "record": "saved_forecast_allocation_resolved",
            "audit": {
                "decision_session": day,
                "decision_index": index,
                "decision_information_clock": rules.CLOCK,
                "source_forecaster_refitted": False,
                "source_risk_weights_retrained": False,
                "source_shadow_wealth_or_floor_transferred": False,
                "candidates": {
                    rules.CHILD_ID: {
                        "expert_weights": child_weights.tolist(),
                        "scores_available": mature,
                        "selected_pair": child_selected,
                    },
                },
            },
            "targets": {rules.CHILD_ID: child_target.tolist()},
        },
    }


@pytest.mark.parametrize("pair,support,overlap", [((0, 5), 2, 1), ((0, 1), 3, 0), ((4, 5), 2, 1)])
def test_nested_pair_mass_identity_and_shared_channels(pair, support, overlap):
    row = fixture(child_pair=pair)
    original = copy.deepcopy(row)
    policy = read_json(rules.POLICY)
    result = rules.decompose(row, policy)
    independent = rules.verify(row, result, policy)
    rolling = result["candidates"][rules.IDS[1]]
    assert rolling["strict_positive_support"] == support
    assert rolling["outer_strict_positive_support"] == 2
    assert rolling["canonical_members_with_both_channels_positive"] == overlap
    assert sum(rolling["canonical_allocation_mass"]) == pytest.approx(1.0)
    assert independent["exact_symbolic_nested_flat_identity"]
    assert independent["maximum_independent_underlying_target_difference"] <= 1e-12
    assert row == original
    assert not result["main_PNL_or_causal_drawdown_attribution"]


def test_equal_four_labels_can_have_three_five_or_eight_canonical_members():
    policy = read_json(rules.POLICY)
    inside = rules.decompose(fixture(child_pair=(4, 5)), policy)["candidates"][rules.IDS[0]]
    outside = rules.decompose(fixture(child_pair=(0, 1)), policy)["candidates"][rules.IDS[0]]
    initial = rules.decompose(fixture(index=0), policy)
    warm = initial["candidates"][rules.IDS[0]]
    assert inside["strict_positive_support"] == 3
    assert outside["strict_positive_support"] == 5
    assert warm["strict_positive_support"] == 8
    assert warm["maximum_canonical_fraction"] == 0.28125
    assert warm["allocation_HHI"] == 0.2421875
    assert not initial["outer_scores_available"] and not initial["child_scores_available"]
    assert not initial["effective_number_is_independent_risk_count"]
    rules.verify(fixture(index=0), initial, policy)


def test_absent_growth_branch_preserves_only_the_direct_two_sources():
    row = fixture(use_growth=False)
    result = rules.decompose(row, read_json(rules.POLICY))
    rules.verify(row, result, read_json(rules.POLICY))
    rolling = result["candidates"][rules.IDS[1]]
    assert rolling["growth_branch_fraction"] == 0
    assert rolling["strict_positive_support"] == 2
    assert rolling["shared_channel_minimum_mass"] == 0


@pytest.mark.parametrize(
    "damage",
    [
        "future",
        "index",
        "foreign",
        "direct_RF",
        "direct_target",
        "growth_target_ulp",
        "child_alias",
        "strings",
        "boolean",
        "negative",
        "nonfinite",
        "source_replay",
    ],
)
def test_bad_source_clock_opinion_identity_or_mass_is_not_a_successful_fallback(damage):
    row = fixture()
    child = row["child_after"]["audit"]["candidates"][rules.CHILD_ID]
    if damage == "future":
        row["child_before"]["source_before"]["as_of"] = "2026-10-05"
    elif damage == "index":
        row["child_after"]["audit"]["decision_index"] += 1
    elif damage == "foreign":
        row["outer_before"]["observations"][0]["id"] = "foreign"
    elif damage == "direct_RF":
        row["outer_before"]["observations"][0]["RF"][0] += 0.01
    elif damage == "direct_target":
        row["outer_before"]["observations"][0]["target"][0] = 0.01
    elif damage == "growth_target_ulp":
        values = row["outer_before"]["observations"][-1]["target"]
        values[0] = np.nextafter(values[0], np.inf)
    elif damage == "child_alias":
        row["child_after"]["targets"] = {"foreign": row["child_after"]["targets"][rules.CHILD_ID]}
    elif damage == "strings":
        child["expert_weights"] = [str(v) for v in child["expert_weights"]]
    elif damage == "boolean":
        child["expert_weights"] = [True] + [False] * 7
    elif damage == "negative":
        child["expert_weights"][0] = -0.1
    elif damage == "nonfinite":
        child["expert_weights"][0] = np.nan
    else:
        row["child_after"]["audit"]["source_forecaster_refitted"] = True
    with pytest.raises(QuantError):
        rules.decompose(row, read_json(rules.POLICY))


@pytest.mark.parametrize(
    "field",
    [
        "canonical_allocation_mass",
        "direct_canonical_mass",
        "growth_branch_canonical_mass",
        "allocation_HHI",
        "effective_allocation_number",
        "maximum_canonical_fraction",
        "shared_channel_minimum_mass",
        "growth_branch_mass_on_trio_members",
        "growth_branch_fraction",
        "strict_positive_support",
    ],
)
def test_independent_fraction_checker_rejects_mass_and_concentration_tampering(field):
    row = fixture()
    result = rules.decompose(row, read_json(rules.POLICY))
    changed = result["candidates"][rules.IDS[1]]
    if field == "canonical_allocation_mass":
        changed[field][0] += 0.01
        changed[field][5] -= 0.01
    elif isinstance(changed[field], list):
        changed[field][0] += 0.01
    else:
        changed[field] += 1 if field == "strict_positive_support" else 0.01
    with pytest.raises(QuantError):
        rules.verify(row, result, read_json(rules.POLICY))


def test_source_future_records_and_truncation_do_not_change_previous_decompositions():
    records = [fixture(index=i) for i in range(9)]
    changed = copy.deepcopy(records)
    changed[-1] = fixture(index=8, child_pair=(4, 5), direct=0)
    policy = read_json(rules.POLICY)
    before = [rules.decompose(r, policy) for r in records[:8]]
    assert before == [rules.decompose(r, policy) for r in changed[:8]]
    assert before[:4] == [rules.decompose(r, policy) for r in records[:4]]


def test_summary_exports_aggregate_bookkeeping_not_individual_positions_or_returns():
    policy = read_json(rules.POLICY)
    inputs = [fixture(index=0), fixture(child_pair=(0, 1)), fixture(child_pair=(4, 5))]
    results = [rules.decompose(row, policy) for row in inputs]
    checks = [
        rules.verify(row, result, policy) for row, result in zip(inputs, results, strict=True)
    ]
    summary = rules.summarize(results, checks)
    assert summary["matched_decisions"] == 3
    assert summary["mature_outer_decisions"] == 2
    assert summary["exact_direct_source_observation_matches"] == 9
    assert summary["alternative_rankings_targets_or_financial_trials"] == 0
    assert not summary["effective_number_is_independent_risk_count"]
    assert "canonical_allocation_mass" not in json.dumps(summary)
    assert "NET_returns" not in json.dumps(summary)


def test_partial_saved_source_pair_is_an_error_not_an_empty_match(tmp_path, monkeypatch):
    monkeypatch.setattr(study.base, "ROOT", tmp_path)
    path = tmp_path / "partial.jsonl"
    path.write_text('{"record":"decision_input"}\n')
    with pytest.raises(QuantError):
        list(study.pairs({"path": path.name, "sha256": file_digest(path)}))


def test_second_actual_audit_attempt_is_terminal_before_any_source_read(tmp_path):
    (tmp_path / "audit-attempt-v1.json").write_text("{}")
    actual = FunctionType(
        study.audit.__code__,
        {
            **study.audit.__globals__,
            "FOLDER": tmp_path,
            "RESULTS": tmp_path / "results.json",
            "lifecycle": SimpleNamespace(runtime_guard=lambda: None),
        },
    )
    with pytest.raises(QuantError):
        actual()


@pytest.mark.parametrize(
    "key,value",
    [
        ("unit_mass_and_target_tolerance", 1e-8),
        ("canonical_direct_slots", [3, 5, 7]),
        ("effective_number_is_independent_risk_count", True),
        ("new_rankings_targets_shadow_accounts_or_financial_trials", 1),
        ("source_algorithms_or_main_returns_recomputed", True),
    ],
)
def test_no_numeric_source_scope_or_return_attribution_rescue(key, value):
    with pytest.raises(QuantError):
        rules.validate({**read_json(rules.POLICY), key: value})
