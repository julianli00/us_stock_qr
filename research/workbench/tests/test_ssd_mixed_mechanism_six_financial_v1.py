import json
from copy import deepcopy
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import test_macro_expert_inclusion_v1 as existing
from test_mixed_mechanism_six_primary_v1 import examples

from us_quant import mixed_mechanism_six_primary_v1 as primary
from us_quant import ssd_mixed_mechanism_six_financial_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json


def prepared():
    experts, calendar, _, _ = examples()
    rows = []
    targets, audits = study.build_targets(
        experts, calendar, primary.UNIVERSE, read_json(study.POLICY), retain=rows.append
    )
    return (
        SimpleNamespace(close=pd.DataFrame(index=calendar, columns=primary.UNIVERSE)),
        experts,
        targets,
        audits,
        rows,
    )


def bind_test(function):
    rebound = FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )
    rebound.__dict__.update(function.__dict__)
    return rebound


for name in (
    "test_both_real_registrations_precede_first_new_derivation",
    "test_real_packet_constructor_and_complete_cohort_metadata",
    "test_missing_other_packet_blocks_every_funded_candidate",
):
    globals()[name] = bind_test(getattr(existing, name))


def test_six_source_financial_adapter_preserves_both_frozen_primary_outputs():
    experts, calendar, _, _ = examples()
    original, old = primary.build_targets(
        experts, calendar, primary.UNIVERSE, read_json(primary.POLICY)
    )
    actual, audits = study.build_targets(
        experts, calendar, primary.UNIVERSE, read_json(study.POLICY)
    )
    assert audits == old
    assert all(actual[name].equals(original[name]) for name in study.IDS)
    assert all(a["equal_expert_weights"] == [1 / 6] * 6 for a in audits)
    assert all(
        len(a["selection"]["pairs"]) == 15 for a in audits if a["selection"]["NET_history_mature"]
    )


def test_all_fifteen_scalar_oracles_rational_targets_and_actual_hex_records(tmp_path):
    data, experts, targets, audits, rows = prepared()
    proof = study.verify_targets(data, experts, targets, audits)
    assert proof["maximum_independent_expert_weight_difference"] < 1e-10
    assert proof["maximum_independent_net_score_difference"] < 1e-8
    assert proof["maximum_rational_target_difference"] < 1e-12
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    checked = study.verify_retained_records(path, experts, targets, audits)
    assert checked["actual_input_and_resolved_record_count"] == 2 * len(audits)
    assert checked["actual_input_resolved_records_match_source_and_targets"] is True
    assert set(rows[1]["complete_targets_binary64_hex"]) == set(study.IDS)


@pytest.mark.parametrize(
    "kind", ["correlation", "cost", "marginal", "pair_count", "NaN", "dimension"]
)
def test_nonselected_pair_missing_fifteenth_or_nonfinite_risk_cannot_hide(kind):
    data, experts, targets, audits, _ = prepared()
    audits = deepcopy(audits)
    active = next(a for a in audits if a["selection"]["scores_available"])
    row = next(
        p
        for p in active["selection"]["pairs"]
        if p["pair"] != active["selection"]["selected_pair"] and "selection_score_key" in p
    )
    if kind == "correlation":
        row["training_correlation"] += 0.01
    elif kind == "cost":
        row["estimated_incremental_switch_cost_fraction"] += 0.01
    elif kind == "marginal":
        row["training_marginal_variance_contributions"][0] += 0.01
    elif kind == "pair_count":
        active["selection"]["pairs"].pop()
    elif kind == "NaN":
        row["training_marginal_variance_contributions"][0] = np.nan
    else:
        row["expert_weights"].append(0.0)
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, targets, audits)


@pytest.mark.parametrize(
    "kind", ["source_RF", "introduced", "known_at", "previous", "equal_target", "rolling_target"]
)
def test_source_and_both_target_binary64_records_are_independently_bound(kind, tmp_path):
    _, experts, targets, audits, rows = prepared()
    rows = deepcopy(rows)
    if kind == "source_RF":
        next(r for r in rows if r["record"] == "decision_input" and r["decision_index"] > 0)[
            "observations"
        ][0]["RF_binary64_hex"][0] = (0.5).hex()
    elif kind == "introduced":
        rows[0]["observations"][0]["introduced_at"] = "2016-10-05T00:00:00Z"
    elif kind == "known_at":
        rows[0]["observations"][0]["target_known_at"] = "2016-10-06"
    elif kind == "previous":
        rows[0]["previous_weights_binary64_hex"][0] = (0.5).hex()
    else:
        name = study.IDS[0 if kind == "equal_target" else 1]
        rows[1]["complete_targets_binary64_hex"][name][0] = (0.5).hex()
    path = tmp_path / "tampered.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(QuantError):
        study.verify_retained_records(path, experts, targets, audits)


@pytest.mark.parametrize(
    "key,value",
    [
        ("candidate_count", 1),
        ("expert_count", 9),
        ("mature_pair_count", 36),
        ("prior_completed_configurations", 268),
        ("both_registrations_before_either_meta_derivation", False),
        ("complete_cohort_publication_before_any_funds", False),
        ("static_target_packet_schema", 3),
        ("global_pool_changed", True),
    ],
)
def test_no_roster_registration_publication_prior_or_packet_rescue(key, value):
    policy = deepcopy(read_json(study.POLICY))
    policy[key] = value
    with pytest.raises(QuantError):
        study.validate(policy)


def test_separate_financial_namespace_and_unqualified_fixture_dependencies_are_complete():
    old = study.financial.ID, study.financial.FOLDER, study.financial.PACKET
    scoped = study.scoped(study.financial.evaluate, study.CANDIDATES[1])
    assert scoped.__globals__["ID"] == study.IDS[1]
    assert scoped.__globals__["publication"] is study.cohort_publication
    assert old == (study.financial.ID, study.financial.FOLDER, study.financial.PACKET)
    paths = {r["path"] for r in study.dependencies()}
    assert set(study.OWN).issubset(paths)
    assert {
        "tests/test_expert_distinct_opinion_pair_v1.py",
        "tests/test_strategy_expert_pool_eight_v1.py",
        "tests/test_macro_expert_inclusion_v1.py",
        "tests/test_mixed_mechanism_six_primary_v1.py",
        "tests/conftest.py",
    } <= paths
