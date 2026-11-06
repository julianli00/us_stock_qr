import builtins
import copy
import json
import math

import numpy as np
import pytest
from test_forecast_risk_normalizer_diagnosis_v1 import audits, pair

from us_quant import forecast_risk_normalizer_diagnosis_v2 as rules
from us_quant import ssd_forecast_risk_normalizer_diagnosis_v2 as study
from us_quant.config import QuantError
from us_quant.storage import read_json


def source_records():
    result = audits()
    for origin in result:
        for candidate in origin["candidates"].values():
            for record in candidate["pairs"]:
                record["selection_score_key"] = rules.source_key(record["selection_score"], 12)
    return result


def test_known_binary64_half_boundary_reconstructs_original_NumPy_not_JSON_Python():
    score = -3.0259045725795
    decoded = json.loads(json.dumps(float(np.float64(score))))
    assert type(decoded) is float
    assert builtins.round(decoded, 12) == -3.025904572579
    assert rules.source_key(decoded, 12) == -3.02590457258
    assert rules.source_key(decoded, 12).hex() == float(builtins.round(np.float64(score), 12)).hex()


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -0.0,
        1e-18,
        -1e-18,
        0.5e-12,
        -0.5e-12,
        1.5e-12,
        -1.5e-12,
        3.0259045725795,
        -3.0259045725795,
    ],
)
def test_source_and_independent_scaled_codec_are_bit_exact_including_signed_zero(value):
    actual = rules.source_key(value, 12)
    expected = float(builtins.round(np.float64(value), 12))
    assert actual.hex() == expected.hex()
    if actual == 0:
        assert math.copysign(1, actual) == math.copysign(1, value)


def test_key_decoder_does_not_normalize_overflow_missing_values_or_other_precision():
    for value, digits in ((float("nan"), 12), (1e308, 12), (1.0, 6), ("1.0", 12)):
        with pytest.raises(QuantError):
            rules.source_key(value, digits)


def test_strict_source_key_check_rejects_one_ULP_even_below_score_tolerance():
    record = pair()
    record["selection_score_key"] = rules.source_key(record["selection_score"], 12)
    valid = rules.case(record, False)
    assert valid["source_NumPy_and_scalar_scaled_key_bit_exact"]
    damaged = copy.deepcopy(record)
    damaged["selection_score_key"] = math.nextafter(record["selection_score_key"], math.inf)
    assert abs(damaged["selection_score_key"] - record["selection_score_key"]) < rules.TOLERANCE
    with pytest.raises(QuantError, match="bit-exactly"):
        rules.case(damaged, False)


def test_key_adapter_keeps_frozen_sign_identity_and_does_not_mutate_globals_or_input():
    record = pair(mean=-0.001)
    record["selection_score_key"] = rules.source_key(record["selection_score"], 12)
    before = copy.deepcopy(record)
    old_round_present = "round" in rules.original.case.__globals__
    old_round = rules.original.case.__globals__.get("round")
    old = rules.original.case(record, True)
    new = rules.case(record, True)
    assert all(new[k] == v for k, v in old.items())
    assert record == before
    assert ("round" in rules.original.case.__globals__) == old_round_present
    assert rules.original.case.__globals__.get("round") is old_round
    assert builtins.round is round
    assert not new["recorded_source_score_or_key_rewritten"]


def test_all_models_menus_prefixes_and_continuous_interpretation_remain():
    source = source_records()
    summary, full = rules.diagnose(source)
    _, early = rules.diagnose(source[:-1])
    assert early == full[:-1]
    for model in rules.MODELS:
        value = summary["models"][model]
        assert value["source_key_codec_bit_exact_pair_rows"] == value["evaluable_pair_rows"] == 56
        assert value["selected_numerator_sign_counts"] == {"positive": 1, "negative": 1, "zero": 0}
    assert summary["original_sign_identity_Decimal_tolerance_unchanged"]
    assert not summary["source_scores_keys_or_target_records_rewritten"]
    assert summary["new_targets_accounts_or_financial_trials"] == 0


def test_future_source_values_do_not_change_prior_classifications_or_use_another_winner():
    source = source_records()
    changed = copy.deepcopy(source)
    for candidate in changed[-1]["candidates"].values():
        candidate["pairs"] = [pair(p["pair"], mean=0.003) for p in candidate["pairs"]]
        for record in candidate["pairs"]:
            record["selection_score_key"] = rules.source_key(record["selection_score"], 12)
    _, before = rules.diagnose(source)
    _, after = rules.diagnose(changed)
    assert before[:-1] == after[:-1]
    assert not any(hasattr(study, n) for n in ("register", "evaluate", "prepare"))


@pytest.mark.parametrize(
    "key,value",
    [
        ("independent_scalar_scaled_codec_bit_exact_required", False),
        ("source_signed_zero_preserved", False),
        ("Python_round_after_JSON_float_coercion_is_source_rule", True),
        ("frozen_V1_sign_identity_Decimal_tolerance_or_aggregation_changed", True),
        ("source_scores_keys_or_target_records_rewritten", True),
        ("old_partial_diagnostic_patched_or_retried", True),
        ("new_alternative_rankings_targets_or_financial_trials", 1),
    ],
)
def test_no_key_or_numeric_tolerance_waiver_is_configurable(key, value):
    with pytest.raises(QuantError):
        rules.validate({**read_json(rules.POLICY), key: value})


def test_dependency_closure_binds_frozen_V1_and_both_actual_scalar_implementations():
    paths = {r["path"] for r in study.dependencies()}
    assert set(study.OWN).issubset(paths)
    assert "src/us_quant/forecast_risk_normalizer_diagnosis_v1.py" in paths
    assert "tests/test_forecast_risk_normalizer_diagnosis_v1.py" in paths
