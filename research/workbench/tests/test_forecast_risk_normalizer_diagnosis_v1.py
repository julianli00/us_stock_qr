import copy
import itertools
import json
import math

import pytest

from us_quant import forecast_risk_normalizer_diagnosis_v1 as rules
from us_quant import ssd_forecast_risk_normalizer_diagnosis_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json


def pair(names=None, mean=0.001, other=None, training=0.01, validation=0.02, cost=0.0):
    names = sorted(rules.pool.EXPERT_IDS[:2]) if names is None else list(names)
    means = {"SPY": mean, "QQQ": mean if other is None else other}
    scores = {
        b: {
            "training_risk_reference": math.sqrt(252) * (v - cost / 21) / training,
            "validation_risk_reference": math.sqrt(252) * (v - cost / 21) / validation,
        }
        for b, v in means.items()
    }
    score = min(v for row in scores.values() for v in row.values())
    return {
        "pair": names,
        "forecast_mean_daily_active": means,
        "forecast_risk_scores": scores,
        "selection_score": score,
        "selection_score_key": round(score, 12),
        "training_RF_excess_pair_risk": training,
        "validation_RF_excess_pair_risk": validation,
        "estimated_incremental_switch_cost_fraction": cost,
        "cost_proxy_is_actual_main_account_cost": False,
    }


def audits():
    dates = [
        "2016-10-05",
        "2016-10-31",
        "2016-11-30",
        "2016-12-30",
        "2017-01-31",
        "2017-02-28",
        "2017-03-31",
        "2017-04-28",
        "2017-05-31",
    ]
    names = sorted(tuple(sorted(p)) for p in itertools.combinations(rules.pool.EXPERT_IDS, 2))
    result = []
    for i, day in enumerate(dates):
        candidates = {}
        for identifier, model in zip(rules.IDS, rules.MODELS, strict=True):
            mature = i >= 7
            candidates[identifier] = {
                "model": model,
                "safety_or_success_claimed": False,
                "scores_available": mature,
                "selected_pair": list(names[0]) if mature else None,
                "reason": "synthetic_mature" if mature else "synthetic_unscored",
                "pairs": [pair(n, mean=0.001 if i == 7 else -0.001) for n in names]
                if mature
                else [],
            }
        result.append(
            {
                "decision_index": i,
                "decision_session": day,
                "candidates": candidates,
                "decision_information_clock": rules.pool.CLOCK,
                "source_forecaster_refitted": False,
                "source_risk_weights_retrained": False,
                "source_shadow_wealth_or_floor_transferred": False,
            }
        )
    return result


def test_positive_numerator_penalizes_risk_and_binds_larger_window():
    original = rules.case(pair(), True)
    higher = rules.case(pair(training=0.02, validation=0.04), True)
    assert original["numerator_sign"] == "positive"
    assert original["binding_risk_window_set"] == ["validation"]
    assert original["continuous_score_risk_slope"] == "decreasing"
    assert higher["recorded_score"] < original["recorded_score"]


def test_negative_numerator_rewards_risk_and_binds_smaller_window():
    original = rules.case(pair(mean=-0.001), True)
    higher = rules.case(pair(mean=-0.001, training=0.02, validation=0.04), True)
    assert original["numerator_sign"] == "negative"
    assert original["binding_risk_window_set"] == ["training"]
    assert original["continuous_score_risk_slope"] == "increasing"
    assert higher["recorded_score"] > original["recorded_score"]
    assert not original["derivative_is_actual_discrete_ranking_or_causal_drawdown_effect"]


def test_actual_binary64_zero_and_adjacent_values_are_not_a_neutral_tolerance_band():
    cost = 0.0001
    boundary = cost / 21
    for mean, sign in (
        (math.nextafter(boundary, math.inf), "positive"),
        (boundary, "zero"),
        (math.nextafter(boundary, -math.inf), "negative"),
    ):
        result = rules.case(pair(mean=mean, cost=cost), False)
        assert result["numerator_sign"] == sign
    zero = rules.case(pair(mean=0), False)
    assert zero["binding_risk_window_set"] == ["training", "validation"]
    assert zero["continuous_score_risk_slope"] == "flat"


def test_benchmark_and_equal_risk_ties_are_sets_not_a_fitted_choice():
    result = rules.case(pair(training=0.02, validation=0.02), False)
    assert result["worst_benchmark_set"] == ["SPY", "QQQ"]
    assert result["binding_risk_window_set"] == ["training", "validation"]
    worse = rules.case(pair(mean=0.002, other=-0.001), False)
    assert worse["worst_benchmark_set"] == ["QQQ"]
    assert worse["numerator_sign"] == "negative"
    assert worse["independent_decimal_score_gap"] < 1e-10


@pytest.mark.parametrize("damage", ["nan", "zero_risk", "negative_cost", "rounding", "component"])
def test_nonselected_bad_components_or_missing_risk_are_not_success(damage):
    record = pair()
    if damage == "nan":
        record["forecast_mean_daily_active"]["SPY"] = float("nan")
    elif damage == "zero_risk":
        record["training_RF_excess_pair_risk"] = 0
    elif damage == "negative_cost":
        record["estimated_incremental_switch_cost_fraction"] = -0.001
    elif damage == "rounding":
        record["selection_score_key"] += 0.01
    else:
        record["forecast_risk_scores"]["SPY"]["training_risk_reference"] += 0.01
    with pytest.raises(QuantError):
        rules.case(record, False)


def test_all_models_all_pairs_and_null_origins_are_kept():
    summary, decisions = rules.diagnose(audits())
    assert len(decisions) == 9
    for model in rules.MODELS:
        value = summary["models"][model]
        assert value["unscored_origins"] == 7 and value["scored_origins"] == 2
        assert value["evaluable_pair_rows"] == 56
        assert value["all_pairs_numerator_sign_counts"] == {
            "positive": 28,
            "negative": 28,
            "zero": 0,
        }
        assert value["selected_numerator_sign_counts"] == {"positive": 1, "negative": 1, "zero": 0}
        assert value["all_eligible_numerators_negative_origins"] == 1
        assert value["selected_negative_with_any_nonnegative_menu_origins"] == 0
    assert summary["new_targets_accounts_or_financial_trials"] == 0
    assert not summary["main_drawdown_or_causal_loss_attribution"]


def test_source_unscorable_pair_is_explicit_and_never_rescued():
    source = audits()
    first = source[7]["candidates"][rules.IDS[0]]
    excluded = first["pairs"][-1]
    first["pairs"][-1] = {"pair": excluded["pair"], "unscorable_reason": "original_zero_risk"}
    summary, _ = rules.diagnose(source)
    assert summary["models"][rules.MODELS[0]]["original_unscorable_pair_rows"] == 1
    assert summary["models"][rules.MODELS[0]]["evaluable_pair_rows"] == 55


def test_future_saved_values_do_not_change_prior_classifications_or_prefix():
    source = audits()
    changed = copy.deepcopy(source)
    for item in changed[-1]["candidates"].values():
        item["pairs"] = [pair(p["pair"], mean=0.003) for p in item["pairs"]]
    _, original = rules.diagnose(source)
    _, later = rules.diagnose(changed)
    _, prefix = rules.diagnose(source[:-1])
    assert original[:-1] == later[:-1] == prefix


def test_actual_input_is_saved_before_a_failing_identity(monkeypatch):
    records = []

    def failed(*args):
        raise QuantError("Synthetic identity failure")

    monkeypatch.setattr(rules, "case", failed)
    with pytest.raises(QuantError, match="Synthetic"):
        rules.diagnose(audits(), retain=records.append)
    assert records[-1]["record"] == "normalizer_input"
    assert records[-1]["index"] == 7


def test_no_source_selector_ranker_target_or_account_is_called(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A completed source selector may not be rerun.")

    monkeypatch.setattr(rules.pool, "choose_pair", forbidden)
    summary, _ = rules.diagnose(audits())
    assert not summary["alternative_pair_ranking_or_selection_generated"]
    assert not any(hasattr(study, n) for n in ("register", "evaluate", "prepare"))


def test_incomplete_or_false_unscored_menu_cannot_be_filled():
    source = audits()
    source[7]["candidates"][rules.IDS[0]]["pairs"].pop()
    with pytest.raises(QuantError):
        rules.diagnose(source)
    source = audits()
    source[0]["candidates"][rules.IDS[0]]["selected_pair"] = ["foreign", "unknown"]
    with pytest.raises(QuantError):
        rules.diagnose(source)


@pytest.mark.parametrize(
    "key,value",
    [
        ("independent_score_tolerance", 1e-8),
        ("cost_amortization_sessions", 5),
        ("numerator_sign_basis", "tolerance_band"),
        ("alternative_pair_ranking_or_selection_generated", True),
        ("main_drawdown_loss_or_causal_attribution_generated", True),
        ("new_forecast_fit_risk_fit_targets_or_accounts", 1),
    ],
)
def test_no_law_tolerance_or_interpretation_rescue(key, value):
    with pytest.raises(QuantError):
        rules.validate({**read_json(rules.POLICY), key: value})


def test_saved_source_reader_checks_real_envelope_before_diagnosing(tmp_path, monkeypatch):
    source = audits()[:1]
    tape = tmp_path / "source.jsonl"
    records = [
        {
            "record": "saved_forecast_allocation_input",
            "source_before": {"decision_index": 0, "as_of": source[0]["decision_session"]},
        },
        {"record": "saved_forecast_allocation_resolved", "audit": source[0]},
    ]
    tape.write_text("".join(json.dumps(r) + "\n" for r in records))
    monkeypatch.setattr(study, "read_json", lambda path: {"source_allocation_tape": {}})
    monkeypatch.setattr(study.base, "checked_reference", lambda ref: tape)
    assert list(study.source_audits()) == source
    tape.write_text(json.dumps(records[0]) + "\n")
    with pytest.raises(QuantError):
        list(study.source_audits())


def test_complete_dependency_closure_binds_actual_identity_oracle_and_reader():
    paths = {r["path"] for r in study.dependencies()}
    assert set(study.OWN).issubset(paths)
    assert "src/us_quant/forecast_risk_normalizer_diagnosis_v1.py" in paths
