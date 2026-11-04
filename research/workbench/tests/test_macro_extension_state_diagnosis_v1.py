import copy
import json
import math

import pytest
from test_macro_expert_inclusion_v1 import examples

from us_quant import macro_extension_state_diagnosis_v1 as diagnosis
from us_quant.config import QuantError
from us_quant.storage import read_json


@pytest.fixture(scope="module")
def saved_steps():
    experts, calendar, assets = examples()
    rows8, rows9 = [], []
    diagnosis.saved.rules.prior.build_targets(
        experts[:8],
        calendar,
        assets,
        read_json(diagnosis.saved.rules.prior.POLICY),
        retain=rows8.append,
    )
    diagnosis.saved.rules.build_targets(
        experts, calendar, assets, read_json(diagnosis.saved.rules.POLICY), retain=rows9.append
    )
    rows8, rows9 = json.loads(json.dumps(rows8)), json.loads(json.dumps(rows9))
    return [((rows8[i], rows8[i + 1]), (rows9[i], rows9[i + 1])) for i in range(0, len(rows8), 2)]


def mature(saved_steps):
    return copy.deepcopy(
        next(step for step in saved_steps if step[0][1]["audit"]["selection"]["scores_available"])
    )


def test_complete_shared_inputs_all_pairs_and_immature_states_preserved(saved_steps):
    results = [diagnosis.compare_step(*step) for step in saved_steps]
    summary = diagnosis.aggregate(results)
    assert summary["decisions"] == len(saved_steps)
    assert summary["choice_classes"]["original_immature_no_score"] == 7
    assert summary["common_pair_records"] == 28 * (len(saved_steps) - 7)
    assert summary["extension_only_pair_records_retained"] == 8 * (len(saved_steps) - 7)
    assert summary["segments"]["training"]["reconciliation_residual"]["maximum_absolute"] < 1e-8
    assert all(r["shared_observations_binary64_identical"] for r in results)


@pytest.mark.parametrize("field", ["NET_returns", "RF", "target"])
def test_one_ulp_or_signed_zero_shared_input_change_is_not_tolerance_matched(saved_steps, field):
    control, extension = mature(saved_steps)
    values = extension[0]["observations"][0][field]
    values[0] = math.nextafter(values[0], math.inf)
    with pytest.raises(QuantError, match="binary64-identical"):
        diagnosis.compare_step(control, extension)
    assert diagnosis.exact([0.0]) != diagnosis.exact([-0.0])


@pytest.mark.parametrize("field", ["introduced_at", "target_known_at", "return_dates"])
def test_shared_provenance_or_clock_cannot_change(saved_steps, field):
    control, extension = mature(saved_steps)
    observation = extension[0]["observations"][0]
    if field == "return_dates":
        observation[field][0] = "2016-01-04"
    else:
        observation[field] = "2016-01-04"
    with pytest.raises(QuantError, match="binary64-identical"):
        diagnosis.compare_step(control, extension)


def test_recorded_score_gap_components_are_algebra_not_new_ranking(saved_steps, monkeypatch):
    control, extension = mature(saved_steps)

    def forbidden(*args, **kwargs):
        raise AssertionError("A diagnostic must not call a closed strategy or account pipeline.")

    monkeypatch.setattr(diagnosis.saved.rules, "choose_pair", forbidden)
    monkeypatch.setattr(diagnosis.saved.rules, "build_targets", forbidden)
    monkeypatch.setattr(diagnosis.saved, "evaluate", forbidden)
    result = diagnosis.compare_step(control, extension)
    for row in result["common_pair_rows"]:
        if row["status"] != "recorded_scores_reconciled":
            continue
        for segment in row["segments"].values():
            assert (
                abs(
                    segment["recorded_score_difference"]
                    - segment["source_NET_reference_component"]
                    - segment["incremental_switch_proxy_component"]
                    - segment["reconciliation_residual"]
                )
                < 1e-14
            )


@pytest.mark.parametrize("kind", ["score", "missing_pair", "extra_source", "paid_fee_claim"])
def test_corrupt_nonselected_pairs_are_not_hidden(saved_steps, kind):
    control, extension = mature(saved_steps)
    selection = extension[1]["audit"]["selection"]
    row = next(
        p
        for p in selection["pairs"]
        if p["pair"] != selection["selected_pair"]
        and diagnosis.IDS[-1] not in p["pair"]
        and "train_net_risk_score" in p
    )
    if kind == "score":
        row["train_net_risk_score"] += 0.01
    elif kind == "missing_pair":
        selection["pairs"].remove(row)
    elif kind == "extra_source":
        row["expert_weights"][-1] = 0.01
        index = max(range(8), key=lambda i: row["expert_weights"][i])
        row["expert_weights"][index] -= 0.01
    else:
        row["cost_proxy_is_actual_main_account_cost"] = True
    with pytest.raises(QuantError):
        diagnosis.compare_step(control, extension)


def test_recorded_weight_numerical_nonidentity_is_retained_not_normalized(saved_steps):
    control, extension = mature(saved_steps)
    result = diagnosis.compare_step(control, extension)
    assert result["preceding_macro_weight"] == extension[0]["previous_weights"][-1]
    assert result["current_macro_weight"] == extension[1]["audit"]["rolling_expert_weights"][-1]
    for row in result["common_pair_rows"]:
        if row["status"] == "recorded_scores_reconciled":
            assert row["maximum_recorded_pair_weight_difference"] >= 0
            assert type(row["pair_weights_binary64_identical"]) is bool


@pytest.mark.parametrize("prior", ["same", "macro_mass", "different_original_pair"])
def test_zero_current_macro_does_not_erase_recorded_preceding_state(saved_steps, prior):
    control, extension = mature(saved_steps)
    audit = extension[1]["audit"]
    row = next(
        p
        for p in audit["selection"]["pairs"]
        if diagnosis.IDS[-1] not in p["pair"]
        and p["pair"] != control[1]["audit"]["selection"]["selected_pair"]
        and "train_net_risk_score" in p
    )
    audit["selection"]["selected_pair"] = row["pair"]
    audit["rolling_expert_weights"] = row["expert_weights"]
    previous = control[0]["previous_weights"] + [0.0]
    if prior == "macro_mass":
        previous = [0.0] * 8 + [1.0]
    elif prior == "different_original_pair":
        previous = [0.0] * 9
        previous[0] = 1.0
    extension[0]["previous_weights"] = previous
    extension[0]["previous_weights_binary64_hex"] = [v.hex() for v in previous]
    audit["previous_rolling_expert_weights"] = previous
    result = diagnosis.compare_step(control, extension)
    assert result["current_macro_weight"] == 0
    assert result["choice_class"] == "different_original_pair_current_macro_zero"
    expected = {
        "same": "exact_equal",
        "macro_mass": "macro_mass_present",
        "different_original_pair": "macro_zero_but_original_source_allocation_differs",
    }
    assert result["preceding_state"] == expected[prior]


def test_constant_scored_pair_is_undefined_not_a_variance_floor(saved_steps):
    control, extension = mature(saved_steps)
    for before, _ in (control, extension):
        for observation in before["observations"]:
            observation["NET_returns"] = [0.0012345] * 126
            observation["RF"] = [0.0001] * 126
    with pytest.raises(QuantError, match="exactly constant"):
        diagnosis.compare_step(control, extension)


def test_nonfinite_extension_source_is_not_ignored_because_common_pair_has_zero_weight(saved_steps):
    control, extension = mature(saved_steps)
    extension[0]["observations"][-1]["NET_returns"][0] = float("nan")
    with pytest.raises(QuantError, match="exact assessment windows"):
        diagnosis.compare_step(control, extension)


@pytest.mark.parametrize("kind", ["truncate", "duplicate", "reorder"])
def test_trace_requires_full_ordered_input_resolved_pairs(saved_steps, tmp_path, kind):
    path = tmp_path / "records.jsonl"
    records = list(saved_steps[0][0]) + list(saved_steps[1][0])
    if kind == "truncate":
        records.pop()
    elif kind == "duplicate":
        records[2:] = records[:2]
    else:
        records[:2] = reversed(records[:2])
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(QuantError):
        list(diagnosis.read_records(path))


def test_public_summary_does_not_export_monthly_inputs_targets_or_source_weights(saved_steps):
    result = {
        "completed_at": "2026-11-04T18:12:44Z",
        "analysis_intent": {"sha256": "synthetic"},
        "aggregates": diagnosis.aggregate([diagnosis.compare_step(*s) for s in saved_steps]),
        "financial_counts_unchanged": {"combined": 249, "qualified": 0},
    }
    public = diagnosis.public_report(result)
    text = json.dumps(public)
    assert public["new_financial_trials"] == 0
    assert not public["investment_objective_verified"]
    assert all(
        key not in text
        for key in (
            "NET_returns",
            "previous_weights",
            "expert_weights",
            "combined_targets",
            "decision_session",
            'extension_only_pair_records"',
        )
    )


def test_second_complete_or_partial_actual_attempt_is_terminal(monkeypatch, tmp_path):
    monkeypatch.setattr(diagnosis.lifecycle, "runtime_guard", lambda: None)
    monkeypatch.setattr(diagnosis, "FOLDER", tmp_path)
    monkeypatch.setattr(diagnosis, "RESULT", tmp_path / "result.json")
    (tmp_path / "attempt-v1.json").write_text("{}")
    with pytest.raises(QuantError, match="never retry"):
        diagnosis.run()
