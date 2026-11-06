import copy

import numpy as np
import pytest

from us_quant import protective_source_mass_audit_v1 as audit
from us_quant.config import QuantError
from us_quant.storage import read_json


def fixture(index=2, old=None, new=None, opinions=None):
    day = audit.decision_dates()[index]
    mature = index >= 2
    old = [0.2] * 5 if old is None else old
    new = [0.25, 0.2, 0.1, 0.15, 0.2, 0.1] if new is None else new
    if not mature:
        old, new = [0.2] * 5, [1 / 6] * 6
    if opinions is None:
        opinions = np.zeros((6, len(audit.ASSETS)))
        for i in range(5):
            opinions[i, i] = 0.98
        opinions[5, audit.ASSETS.index("BIL")] = 0.98
    periods = (
        []
        if not mature
        else [str(d.date()) for d in audit.sessions(audit.decision_dates()[index - 1], day)[1:]]
    )
    record = {"decision_index": index}
    for name, size, weights in (("five", 5, old), ("six", 6, new)):
        before = {
            "phase": "before_decision",
            "decision_session": day,
            "state": {"complete_intervals": max(0, index - 2)},
            "previous_weights_hex": [float(1 / size).hex()] * size,
            "known_target_dates": [day] * size,
            "known_targets_hex": [[float(v).hex() for v in row] for row in opinions[:size]],
            "matured_period_dates": periods.copy(),
            "matured_NET_returns_hex": [[float(0).hex()] * len(periods) for _ in range(size)],
            "candidate_accounts_or_metrics_computed": False,
        }
        logs = None if not mature else [float(-i / 100).hex() for i in range(size)]
        labels = None if not mature else [float(i / (size - 1)).hex() for i in range(size)]
        if name == "five":
            before["cached_canonical_log_relatives_hex"] = logs
            before["cached_loss_labels_hex"] = labels
        else:
            before["canonical_log_relatives_hex"] = logs
            before["normalized_loss_labels_hex"] = labels
            before["canonical_RF_hex"] = [float(0).hex()] * len(periods)
        after = {
            "phase": "after_decision",
            "decision_session": day,
            "audit": {
                "decision_session": day,
                "expert_weights": weights.copy(),
                "previous_expert_weights": [1 / size] * size,
                "complete_learning_intervals": max(0, index - 1),
                "score_mature": mature,
                "adaptive_state": {"complete_intervals": max(0, index - 1)},
                "period_update": None
                if not mature
                else {"start_after": audit.decision_dates()[index - 1], "end": day},
                "known_target_dates": [day] * size,
            },
            "unified_target_hex": [float(v).hex() for v in np.asarray(weights) @ opinions[:size]],
        }
        record[name + "_before"], record[name + "_after"] = before, after
    return record


def resolve(record):
    policy = read_json(audit.POLICY)
    result = audit.decompose(record, policy)
    checked = audit.verify(record, result, policy)
    assert checked["exact_signed_algebra_identity"] is True
    assert checked["maximum_independent_term_difference"] <= 1e-12
    return result, checked


def test_signed_identity_and_exact_independent_arithmetic():
    record = fixture()
    result, _ = resolve(record)
    assert any(v < 0 for v in result["direct_added_source_term"])
    assert any(v > 0 for v in result["direct_added_source_term"])
    np.testing.assert_allclose(
        np.array(result["direct_added_source_term"])
        + result["retained_source_redistribution_term"],
        result["stored_target_change"],
        rtol=0,
        atol=1e-12,
    )
    assert result["main_PNL_or_causal_drawdown_attribution"] is False


def test_zero_projected_added_mass_does_not_remove_retained_redistribution():
    result, _ = resolve(fixture(new=[0.4, 0.15, 0.15, 0.15, 0.15, 0.0]))
    assert result["added_source_recorded_binary64_mass"] == 0
    assert result["direct_added_source_L1"] == 0
    assert result["retained_redistribution_L1"] > 0


def test_proportional_retained_weights_make_direct_only_change():
    old = [0.25, 0.25, 0.25, 0.125, 0.125]
    result, _ = resolve(fixture(old=old, new=[v / 2 for v in old] + [0.5]))
    assert result["retained_redistribution_L1"] == 0
    assert result["direct_added_source_L1"] > 0


def test_opposing_signed_channels_cancel_without_valid_percentage_attribution():
    opinions = np.zeros((6, len(audit.ASSETS)))
    opinions[:, 0] = 0.98
    opinions[[0, 5], 0] = 0
    opinions[[0, 5], 3] = 0.98
    result, _ = resolve(fixture(new=[0.0, 0.2, 0.2, 0.2, 0.2, 0.2], opinions=opinions))
    assert result["stored_target_change_L1"] < 1e-12
    assert result["direct_added_source_L1"] > 0.1
    assert result["retained_redistribution_L1"] > 0.1
    assert result["absolute_components_minus_net_L1"] > 0.2


def test_added_opinion_equal_to_old_mixture_has_no_direct_effect():
    opinions = np.zeros((6, len(audit.ASSETS)))
    for i in range(5):
        opinions[i, i] = 0.98
    opinions[-1] = np.mean(opinions[:5], axis=0)
    result, _ = resolve(fixture(new=[0.16] * 5 + [0.2], opinions=opinions))
    assert result["direct_added_source_L1"] <= 1e-12


def test_all_mass_on_added_source_is_not_summed_shadow_capital():
    result, _ = resolve(fixture(new=[0.0] * 5 + [1.0]))
    assert result["retained_redistribution_L1"] == 0
    assert result["added_source_recorded_binary64_mass"] == 1


def test_tiny_positive_binary64_mass_is_not_floored_to_zero():
    result, checked = resolve(fixture(new=[0.2] * 5 + [1e-300]))
    assert result["added_source_recorded_binary64_mass"] == 1e-300
    assert result["direct_added_source_L1"] > 0
    summary = audit.summarize([result], [checked])
    assert summary["all_origins"]["added_source_binary64_zero_origins"] == 0


@pytest.mark.parametrize("index", [0, 1])
def test_both_original_unscored_origins_remain_uniform(index):
    result, _ = resolve(fixture(index))
    assert result["score_mature"] is False
    assert result["retained_source_loss_labels_changed"] == 0
    record = fixture(index)
    record["six_after"]["audit"]["expert_weights"] = [0.2] * 5 + [0.0]
    with pytest.raises(QuantError):
        resolve(record)


def test_original_net_and_logs_unchanged_while_label_values_can_differ():
    result, _ = resolve(fixture())
    assert result["retained_source_loss_labels_changed"] == 4
    assert result["retained_source_NET_values_bit_exact"] > 0


@pytest.mark.parametrize("field", ["opinion", "NET", "log"])
def test_changed_retained_source_is_not_a_controlled_addition(field):
    record = fixture()
    before = record["five_before"]
    if field == "opinion":
        before["known_targets_hex"][0][0] = (0.97).hex()
        before["known_targets_hex"][0][3] = (0.01).hex()
    elif field == "NET":
        before["matured_NET_returns_hex"][0][0] = (-0.02).hex()
    else:
        before["cached_canonical_log_relatives_hex"][0] = (0.03).hex()
    with pytest.raises(QuantError):
        resolve(record)


def test_source_opinion_future_date_is_rejected():
    record = fixture()
    record["six_before"]["known_target_dates"][5] = "2016-12-01"
    record["six_after"]["audit"]["known_target_dates"][5] = "2016-12-01"
    with pytest.raises(QuantError):
        resolve(record)


@pytest.mark.parametrize(
    "weights",
    [
        [-0.1, 0.3, 0.2, 0.2, 0.2, 0.2],
        [float("nan"), 0.2, 0.2, 0.2, 0.1, 0.1],
        [float("inf"), 0.2, 0.2, 0.2, 0.1, 0.1],
        [True, 0.0, 0.0, 0.0, 0.0, 0.0],
        ["0.2", 0.2, 0.2, 0.2, 0.1, 0.1],
        [0.2] * 5,
        [0.1] * 6,
    ],
)
def test_invalid_recorded_capital_fractions_stop(weights):
    record = fixture()
    record["six_after"]["audit"]["expert_weights"] = weights
    with pytest.raises(QuantError):
        resolve(record)


@pytest.mark.parametrize("value", ["nan", "inf", "0.98", "not-a-number"])
def test_nonfinite_or_noncanonical_hex_is_not_silently_coerced(value):
    record = fixture()
    record["six_before"]["known_targets_hex"][5][3] = value
    with pytest.raises(QuantError):
        resolve(record)


@pytest.mark.parametrize("field", ["period", "maturity", "state", "clock", "RF"])
def test_missing_or_changed_original_information_stops(field):
    record = fixture()
    if field == "period":
        record["six_before"]["matured_period_dates"].pop()
    elif field == "maturity":
        record["six_after"]["audit"]["score_mature"] = False
    elif field == "state":
        record["six_before"]["state"]["complete_intervals"] = 1
    elif field == "clock":
        record["five_after"]["decision_session"] = "2016-11-29"
    else:
        record["six_before"]["canonical_RF_hex"].pop()
    with pytest.raises(QuantError):
        resolve(record)


@pytest.mark.parametrize("index", [True, -1, 121])
def test_wrong_decision_origin_stops(index):
    record = fixture()
    record["decision_index"] = index
    with pytest.raises(QuantError):
        resolve(record)


def test_independent_verifier_rejects_changed_signed_term():
    record = fixture()
    result, _ = resolve(record)
    result["retained_source_redistribution_term"][0] += 0.001
    with pytest.raises(QuantError):
        audit.verify(record, result, read_json(audit.POLICY))


def test_prefix_and_later_input_perturbation_preserve_earlier_results():
    records = [fixture(i) for i in range(13)]
    expected = [resolve(record)[0] for record in records]
    changed = copy.deepcopy(records)
    changed[-1]["six_before"]["known_target_dates"][-1] = "2030-01-01"
    for stop in (1, 3, 7, 12):
        assert [resolve(record)[0] for record in changed[:stop]] == expected[:stop]


@pytest.mark.parametrize(
    "key,value",
    [
        ("unit_and_target_tolerance", 1e-9),
        ("signed_components_are_executable_targets", True),
        ("new_learning_targets_shadow_accounts_or_financial_trials", True),
    ],
)
def test_frozen_interpretation_and_numeric_limits_cannot_change(key, value):
    policy = read_json(audit.POLICY)
    policy[key] = value
    with pytest.raises(QuantError):
        audit.validate(policy)


def test_partial_cached_source_pair_stays_failed(tmp_path, monkeypatch):
    from us_quant import ssd_protective_source_mass_audit_v1 as program

    path = tmp_path / "records.jsonl"
    path.write_text('{"phase":"before_decision"}\n')
    monkeypatch.setattr(program.base, "checked_reference", lambda _: path)
    with pytest.raises(QuantError):
        list(program.pairs({"path": "synthetic-only"}))
