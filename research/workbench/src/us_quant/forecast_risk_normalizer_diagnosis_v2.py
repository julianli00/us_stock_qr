from __future__ import annotations

import builtins
import math
from types import FunctionType

import numpy as np

from us_quant import forecast_risk_normalizer_diagnosis_v1 as original
from us_quant.config import QuantError
from us_quant.storage import read_json

ROOT = original.ROOT
POLICY = ROOT / "config/forecast-risk-normalizer-diagnosis-v2.json"
IDS, MODELS, BENCHMARKS = original.IDS, original.MODELS, original.BENCHMARKS
TOLERANCE = original.TOLERANCE


def validate(policy):
    original.validate(read_json(original.POLICY))
    expected = {
        "schema_version": 2,
        "study_id": "ssd_forecast_risk_normalizer_diagnosis_20261106_v2",
        "actual_native_event_seq": 176855,
        "prior_journal_sequence": 1274,
        "prior_journal_head_sha256": (
            "2056f5365d92c8da4d89d19ae71cc53cd326a1a2fa559f05853a3fd3868b3254"
        ),
        "prior_combined_completed": 285,
        "source_scalar_type": "numpy.float64",
        "source_key_codec": "numpy_float64_round12_scaled_binary64_nearest_even",
        "independent_scalar_scaled_codec_bit_exact_required": True,
        "source_signed_zero_preserved": True,
        "Python_round_after_JSON_float_coercion_is_source_rule": False,
        "frozen_V1_sign_identity_Decimal_tolerance_or_aggregation_changed": False,
        "source_scores_keys_or_target_records_rewritten": False,
        "old_partial_diagnostic_patched_or_retried": False,
        "new_alternative_rankings_targets_or_financial_trials": 0,
        "new_source_requests": 0,
        "new_source_RF_views": 0,
        "independent_economic_research_questions_added_by_codec": 0,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if not isinstance(policy, dict) or any(
        type(policy.get(k)) is not type(v) or policy.get(k) != v for k, v in expected.items()
    ):
        raise QuantError("Restore the exact original scalar codec, not another tolerance or key.")
    if (
        set(policy)
        != set(expected)
        | {
            "source_semantic_policy",
            "source_failed_diagnostic_freeze",
            "source_terminal_failure",
            "source_public_failure",
            "source_allocation_tape",
        }
        or policy["source_allocation_tape"] != read_json(original.POLICY)["source_allocation_tape"]
    ):
        raise QuantError("The decoder cannot change the frozen source records or semantic law.")


def source_key(value, decimal_places):
    value = original.number(value)
    if type(decimal_places) is not int or decimal_places != 12:
        raise QuantError("Use the original twelve-decimal source scalar protocol.")
    scaled = value * 10**12
    if not math.isfinite(scaled):
        raise QuantError("A source-key overflow cannot be approximated or silently repaired.")
    numpy_key = float(builtins.round(np.float64(value), 12))
    integer = builtins.round(scaled)
    scalar_key = float(integer) / 10**12
    if integer == 0:
        scalar_key = math.copysign(scalar_key, value)
    if numpy_key.hex() != scalar_key.hex():
        raise QuantError("The source NumPy codec and independent scalar binary64 codec disagree.")
    return numpy_key


def case(pair, selected):
    if "unscorable_reason" not in pair:
        expected = source_key(pair["selection_score"], 12)
        if original.number(pair["selection_score_key"]).hex() != expected.hex():
            raise QuantError(
                "The recorded source key must match its actual original codec bit-exactly."
            )
    checked = FunctionType(
        original.case.__code__,
        {**original.case.__globals__, "round": source_key},
        argdefs=original.case.__defaults__,
    )(pair, selected)
    if checked["status"] == "recorded_score_identity_verified":
        checked.update(
            source_NumPy_and_scalar_scaled_key_bit_exact=True,
            Python_JSON_float_round_differs_from_source=(
                float(builtins.round(float(pair["selection_score"]), 12)).hex()
                != float(pair["selection_score_key"]).hex()
            ),
            recorded_source_score_or_key_rewritten=False,
        )
    return checked


def aggregate(decisions):
    summary = original.aggregate(decisions)
    for model in MODELS:
        rows = [
            row
            for decision in decisions
            for row in decision["models"][model]["pairs"]
            if row["status"] == "recorded_score_identity_verified"
        ]
        summary["models"][model].update(
            source_key_codec_bit_exact_pair_rows=sum(
                r["source_NumPy_and_scalar_scaled_key_bit_exact"] for r in rows
            ),
            Python_JSON_float_round_differs_all_pair_rows=sum(
                r["Python_JSON_float_round_differs_from_source"] for r in rows
            ),
            Python_JSON_float_round_differs_selected_pair_rows=sum(
                r["Python_JSON_float_round_differs_from_source"]
                for r in rows
                if r["was_recorded_selected"]
            ),
        )
    summary.update(
        source_key_codec="numpy_float64_round12_scaled_binary64_nearest_even",
        source_scores_keys_or_target_records_rewritten=False,
        original_sign_identity_Decimal_tolerance_unchanged=True,
        failed_V1_partial_diagnostic_preserved=True,
        independent_economic_research_questions_added_by_codec=0,
    )
    return summary


def diagnose(audits, retain=None, before_case=None):
    return FunctionType(
        original.diagnose.__code__,
        {**original.diagnose.__globals__, "case": case, "aggregate": aggregate},
        argdefs=original.diagnose.__defaults__,
    )(audits, retain, before_case)
