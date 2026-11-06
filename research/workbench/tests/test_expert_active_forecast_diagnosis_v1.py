import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from us_quant import expert_active_forecast_diagnosis_v1 as rules
from us_quant import ssd_expert_active_forecast_diagnosis_v1 as study
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import ExpertIdentity


def synthetic_inputs(end="2017-08-31"):
    calendar = sessions("2015-01-02", end)
    close = pd.DataFrame(
        {
            s: 100 * np.exp(np.arange(len(calendar)) * rate)
            for s, rate in (("SPY", 0.0003), ("IEF", -0.0001), ("BIL", 0.00005))
        },
        index=calendar,
    )
    dates = calendar[calendar >= "2016-10-06"]
    benchmark = pd.DataFrame(
        {
            b: 0.0001 + 0.002 * np.sin(np.arange(len(dates)) / (13 + j))
            for j, b in enumerate(rules.BENCHMARKS)
        },
        index=dates,
    )

    class SyntheticExpert:
        def __init__(self, number):
            self.identity = ExpertIdentity(
                rules.pool.EXPERT_IDS[number],
                rules.pool.MECHANISMS[number],
                "2026-11-03T13:00:00Z",
            )
            self.number = number

        def observe(self, day, lookback):
            known = dates[dates <= day][-lookback:]
            positions = dates.get_indexer(known)
            return SimpleNamespace(
                identity=self.identity,
                as_of=day,
                target_known_at=pd.Timestamp("2016-09-30"),
                target=pd.Series({"SPY": 0.98}),
                score_mature=len(known) == 126,
                net_returns=pd.Series(
                    0.0002 + 0.002 * np.cos(positions / (11 + self.number)), index=known
                ),
                risk_free=pd.Series(0.00005, index=known),
            )

    experts = tuple(SyntheticExpert(i) for i in range(8))
    return list(study.decision_inputs(SimpleNamespace(close=close), experts, benchmark))


def test_frozen_question_is_not_contest_credit_or_a_funded_allocation():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    assert policy["ridge_penalty"] == 1 and policy["new_financial_candidates"] == 0
    assert policy["source_count"] == 8
    assert policy["canonical_source_policy"]["path"] == "config/strategy-expert-pool-eight-v1.json"


@pytest.mark.parametrize(
    "key,value",
    [
        ("ridge_penalty", 0),
        ("intercept_penalized", False),
        ("lookback_sessions", 252),
        ("maximum_complete_learning_intervals", 120),
        ("unmatured_forecast", 0.0),
        ("source_count", 3),
        ("new_allocations", 1),
        ("new_RF_views", 1),
        ("new_funded_paths", 1),
        ("new_financial_candidates", 1),
        ("select_best_expert_or_model_from_complete_results", True),
        ("independent_numeric_tolerance", 1e-8),
        ("independent_forward_validation", True),
    ],
)
def test_no_penalty_window_source_or_qualification_rescue(key, value):
    with pytest.raises(QuantError):
        rules.validate({**read_json(rules.POLICY), key: value})


def test_exact_continuous_context_and_zero_relative_price_tie():
    inputs = synthetic_inputs()
    context = inputs[0]["context"]
    actual = rules.context(context)
    assert actual[0] == 1 and actual[1] > 0 and actual[2] < 0
    tied = copy.deepcopy(context)
    tied["current_prices"] = {s: 2 * v for s, v in tied["earlier_prices"].items()}
    assert rules.context(tied) == [1, 0, 0]
    tied["current_prices"]["IEF"] = 0
    with pytest.raises(QuantError):
        rules.context(tied)


def test_exact_unit_ridge_and_constant_features_need_no_numeric_rescue():
    rows = [([1.0, 0.0, 0.0], [0.004] * 16)] * 3
    expected = [0.003] * 16
    assert rules.ridge(rows, [1.0, 0.0, 0.0], 1) == expected
    assert rules.ridge(rows, [1.0, 0.0, 0.0], 3) == expected
    assert rules.ridge([], [1.0, 0.0, 0.0], 3) == [0] * 16
    with pytest.raises(QuantError):
        rules.ridge([([1, 0, 0], [float("nan")] * 16)], [1, 0, 0], 3)


def test_grade_previous_forecast_before_learning_and_keep_warmup_null():
    inputs = synthetic_inputs()
    summary, audits = rules.analyze(inputs)
    assert audits[0]["assimilated_period"] == [] and audits[1]["assimilated_period"] == []
    assert audits[2]["complete_learning_months"] == 1
    assert audits[2]["assimilated_period"][0] == "2016-11-01"
    first = next(i for i, a in enumerate(audits) if a["forecast_available"])
    assert audits[first]["as_of"] == "2017-04-28"
    assert all(a["predictions"] is None for a in audits[:first])
    grade = audits[first + 1]["graded_prior_forecast"]
    assert grade["predictions"] == audits[first]["predictions"]
    assert grade["issued_at"] == "2017-04-28"
    assert grade["period"][0] == "2017-05-01"
    assert summary["graded_months"] == 4 and summary["complete_learning_months"] == 10
    assert summary["new_financial_trials"] == 0


def test_both_predictors_all_sources_match_independent_batch_oracle():
    inputs = synthetic_inputs()
    summary, audits = rules.analyze(inputs)
    verified = rules.verify(inputs, audits, summary)
    assert verified["maximum_independent_numeric_difference"] < 1e-12
    for model in rules.MODELS:
        for benchmark in rules.BENCHMARKS:
            assert set(summary["models"][model][benchmark]["experts"]) == set(rules.pool.EXPERT_IDS)
    assert not summary["pooled_expert_errors_are_independent_observations"]


def test_future_prices_source_and_benchmark_changes_preserve_prior_forecasts_and_grades():
    inputs = synthetic_inputs()
    changed = copy.deepcopy(inputs)
    for row in changed:
        if row["as_of"] > "2017-05-31":
            row["context"]["current_prices"]["SPY"] *= 1.1
            for o in row["observations"]:
                o["NET_returns"] = [v + 0.001 for v in o["NET_returns"]]
            row["benchmarks"]["QQQ"] = [v - 0.001 for v in row["benchmarks"]["QQQ"]]
    _, original = rules.analyze(inputs)
    _, later = rules.analyze(changed)
    cutoff = next(i for i, row in enumerate(inputs) if row["as_of"] == "2017-05-31")
    assert original[: cutoff + 1] == later[: cutoff + 1]
    _, truncated = rules.analyze(inputs[: cutoff + 1])
    assert truncated == original[: cutoff + 1]


def test_actual_inputs_are_retained_before_a_failing_forecast(monkeypatch):
    inputs = synthetic_inputs()
    records = []

    def failed(*args):
        raise QuantError("Synthetic predictor failure")

    monkeypatch.setattr(rules, "ridge", failed)
    with pytest.raises(QuantError, match="Synthetic"):
        rules.analyze(inputs, retain=records.append)
    assert records[-1]["record"] == "forecast_input"
    assert records[-1]["as_of"] == "2017-04-28"


@pytest.mark.parametrize("damage", ["future", "missing", "nonfinite", "unavailable", "RF", "month"])
def test_invalid_observations_are_not_filled_or_relabelled_success(damage):
    inputs = synthetic_inputs()
    if damage == "future":
        inputs[2]["observations"][7]["target_known_at"] = "2026-01-01"
    elif damage == "missing":
        inputs[2]["benchmark_dates"].pop()
    elif damage == "nonfinite":
        inputs[2]["benchmarks"]["SPY"][-1] = float("nan")
    elif damage == "unavailable":
        inputs[2]["observations"][7]["target_available"] = False
    elif damage == "RF":
        inputs[2]["observations"][7]["RF"][-1] = 0.001
    else:
        inputs.pop(2)
        for i, row in enumerate(inputs):
            row["index"] = i
    with pytest.raises(QuantError):
        rules.analyze(inputs)


@pytest.mark.parametrize("damage", ["context", "label", "forecast", "unscored", "summary"])
def test_oracle_rejects_nonselected_source_and_evidence_tampering(damage):
    inputs = synthetic_inputs()
    summary, audits = rules.analyze(inputs)
    first = next(a for a in audits if a["forecast_available"])
    if damage == "context":
        first["context_features"][1] += 0.01
    elif damage == "label":
        first["assimilated_label"][-1] += 0.01
    elif damage == "forecast":
        first["predictions"][rules.MODELS[1]][-1] += 0.01
    elif damage == "unscored":
        audits[0]["predictions"] = {name: [0] * 16 for name in rules.MODELS}
    else:
        summary["models"][rules.MODELS[1]]["QQQ"]["experts"][rules.pool.EXPERT_IDS[-1]]["mae"] += 1
    with pytest.raises(QuantError):
        rules.verify(inputs, audits, summary)


def test_zero_reference_and_no_graded_months_are_explicitly_undefined():
    assert rules.metrics([], [])["rmse"] is None
    result = rules.metrics([0, 0], [0, 0])
    assert result["rmse"] == 0 and result["direction_accuracy"] == 1
    assert result["skill_vs_zero_active"] is None and not result["skill_defined"]
    summary, audits = rules.analyze(synthetic_inputs()[:1])
    assert summary["graded_months"] == 0 and audits[0]["predictions"] is None


def test_retention_roundtrips_and_no_capital_or_closed_pipeline_interface(tmp_path):
    inputs = synthetic_inputs()
    records = []
    summary, audits = rules.analyze(inputs, retain=records.append)
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    restored = [json.loads(line) for line in path.read_text().splitlines()]
    assert restored[::2] == inputs
    assert [r["audit"] for r in restored[1::2]] == audits
    assert summary["new_allocations"] == 0
    assert not any(hasattr(study, name) for name in ("register", "evaluate", "prepare"))


def test_dependency_closure_binds_actual_predictor_oracle_and_cached_source_adapter():
    refs = study.dependencies()
    paths = {r["path"] for r in refs}
    assert set(study.OWN).issubset(paths)
    assert "src/us_quant/ssd_eight_expert_meta_v1.py" in paths
    assert "src/us_quant/expert_contextual_contest_v1.py" in paths
