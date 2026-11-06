from __future__ import annotations

import math
from datetime import date
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np

from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.expert_pair_menu_extension_v1 import _stats
from us_quant.nested_growth_allocation_audit_v1 import _weights, require
from us_quant.saved_inclusion_execution_v1 import _target

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/protective-source-mass-audit-v1.json"
ASSETS = (
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
)
EQUITY = tuple(i for i, name in enumerate(ASSETS) if name not in ("IEF", "TLT", "GLD", "BIL"))
TOLERANCE = 1e-12


def validate(policy):
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_protective_source_mass_audit_20261106_v1",
        "actual_native_event_seq": 191045,
        "prior_journal_sequence": 1464,
        "prior_journal_head_sha256": (
            "0c971a26bbfebacf207e68e598b01c86af7d38680b30c21c2fc05647efb2d555"
        ),
        "source_inputs": {
            "path": (
                "reports/ssd-continuous-research-v1/protective-source-mass-audit-20261106-v1/"
                "source-inputs-v1.json"
            ),
            "sha256": "bfaf654457acab53bfa1bc75c6747fee860c882bbaef4068dfe1574b1e03b512",
        },
        "same_clock_origins": 121,
        "initial_unscored_origins": 2,
        "complete_learning_intervals": 119,
        "added_source_position": 5,
        "invested_target_budget": 0.98,
        "unit_and_target_tolerance": TOLERANCE,
        "original_source_opinions_NET_and_logs_bit_exact": True,
        "source_labels_compared_not_regenerated": True,
        "direct_term": "a_times_added_opinion_minus_exact_five_source_linear_opinion",
        "redistribution_term": (
            "sum_retained_w6_minus_one_minus_a_times_w5_times_retained_opinion"
        ),
        "independent_arithmetic": "exact_Fraction_of_original_binary64_inputs",
        "signed_components_are_executable_targets": False,
        "allocation_mass_is_main_PNL_or_causal_drawdown_attribution": False,
        "loss_normalization_isolated_counterfactual": False,
        "binary64_zero_is_analytic_zero_or_expert_exclusion": False,
        "new_learning_targets_shadow_accounts_or_financial_trials": 0,
        "source_algorithms_or_main_returns_recomputed": False,
        "new_source_requests": 0,
        "new_source_RF_views": 0,
        "global_pool_changed": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    require(
        policy == fixed and all(type(policy[k]) is type(v) for k, v in fixed.items()),
        "Freeze one source-scoped non-capital identity and the original numeric limits.",
    )


@lru_cache(maxsize=1)
def decision_dates():
    dates = sessions("2016-10-05", "2026-10-05")
    return (str(dates[0].date()), *[str(d.date()) for d in dates[1:] if is_month_end(d)])


def _hex(values, size):
    require(
        isinstance(values, list) and len(values) == size and all(type(v) is str for v in values),
        "Retain the complete original binary64 hex vector.",
    )
    try:
        result = np.array([float.fromhex(v) for v in values])
    except ValueError as exc:
        raise QuantError("Invalid recorded binary64 hex value.") from exc
    require(
        np.isfinite(result).all() and [float(v).hex() for v in result] == values,
        "No nonfinite or noncanonical numeric conversion is allowed.",
    )
    return result


def _fractions(values, size):
    require(
        isinstance(values, list) and all(type(v) in (int, float) for v in values),
        "Recorded allocation fractions cannot include booleans or numeric strings.",
    )
    return _weights(values, size)


def _inputs(record):
    index = record["decision_index"]
    require(
        type(index) is int and 0 <= index < len(decision_dates()),
        "Keep one original initial/month-end source origin.",
    )
    day, mature = decision_dates()[index], index >= 2
    matrices, weights, saved, labels = [], [], [], []
    for name, size in (("five", 5), ("six", 6)):
        before, after = record[name + "_before"], record[name + "_after"]
        audit = after["audit"]
        require(
            before["phase"] == "before_decision"
            and after["phase"] == "after_decision"
            and before["decision_session"]
            == after["decision_session"]
            == audit["decision_session"]
            == day
            and before["candidate_accounts_or_metrics_computed"] is False
            and type(audit["score_mature"]) is bool
            and audit["score_mature"] is mature
            and audit["complete_learning_intervals"] == max(0, index - 1)
            and before["state"]["complete_intervals"] == max(0, index - 2)
            and audit["adaptive_state"]["complete_intervals"] == max(0, index - 1)
            and audit["known_target_dates"] == before["known_target_dates"]
            and len(before["known_target_dates"]) == size
            and len(before["known_targets_hex"]) == size,
            "Join the exact before/after state and declared score maturity,not another clock.",
        )
        for known in before["known_target_dates"]:
            require(type(known) is str, "Source opinion dates must remain explicit.")
            try:
                parsed = date.fromisoformat(known)
            except ValueError as exc:
                raise QuantError("Invalid recorded source opinion date.") from exc
            require(
                str(parsed) == known and known <= day,
                "A future or malformed opinion cannot enter the audit.",
            )
        previous = _fractions(audit["previous_expert_weights"], size)
        require(
            [float(v).hex() for v in previous] == before["previous_weights_hex"],
            "Retain the exact previous allocation state.",
        )
        current = _fractions(audit["expert_weights"], size)
        if not mature:
            require(
                np.array_equal(previous, np.full(size, 1 / size))
                and np.array_equal(current, np.full(size, 1 / size))
                and audit["period_update"] is None
                and before["matured_period_dates"] == [],
                "The two unscored origins are actual uniform states,not successful scores.",
            )
        else:
            anchor = decision_dates()[index - 1]
            dates = [str(d.date()) for d in sessions(anchor, day)[1:]]
            require(
                before["matured_period_dates"] == dates
                and audit["period_update"]["start_after"] == anchor
                and audit["period_update"]["end"] == day,
                "A complete source month cannot be cropped,filled or shifted.",
            )
        count = len(before["matured_period_dates"])
        require(
            len(before["matured_NET_returns_hex"]) == size, "Preserve every source NET coordinate."
        )
        for values in before["matured_NET_returns_hex"]:
            require((_hex(values, count) > -1).all(), "Source NET rows must remain solvent.")
        key = "cached_loss_labels_hex" if name == "five" else "normalized_loss_labels_hex"
        raw_labels = before[key]
        if mature:
            vector = _hex(raw_labels, size)
            require(
                ((vector >= 0) & (vector <= 1)).all(),
                "Compare original bounded loss labels without regenerating them.",
            )
        else:
            require(raw_labels is None, "Unmatured labels cannot be filled with zero.")
        matrices.append(
            np.stack(
                [_target(_hex(v, len(ASSETS)), len(ASSETS)) for v in before["known_targets_hex"]]
            )
        )
        weights.append(current)
        saved.append(_target(_hex(after["unified_target_hex"], len(ASSETS)), len(ASSETS)))
        labels.append(raw_labels)
    old, new = record["five_before"], record["six_before"]
    require(
        old["known_targets_hex"] == new["known_targets_hex"][:5]
        and old["known_target_dates"] == new["known_target_dates"][:5]
        and old["matured_period_dates"] == new["matured_period_dates"]
        and old["matured_NET_returns_hex"] == new["matured_NET_returns_hex"][:5],
        "All five retained opinions,clocks and original NET values must be bit-exact.",
    )
    old_logs, new_logs = (
        old["cached_canonical_log_relatives_hex"],
        new["canonical_log_relatives_hex"],
    )
    if mature:
        require(old_logs == new_logs[:5], "Retained source logs cannot change across rosters.")
        _hex(old_logs, 5)
        _hex(new_logs, 6)
    else:
        require(old_logs is None and new_logs is None, "No invented partial-month logs.")
    _hex(new["canonical_RF_hex"], len(new["matured_period_dates"]))
    return matrices[1], weights, saved, labels, mature


def decompose(record, policy):
    validate(policy)
    opinions, (old, new), (saved_old, saved_new), labels, mature = _inputs(record)
    baseline = np.array(
        [
            math.fsum(float(old[i]) * float(opinions[i, j]) for i in range(5))
            for j in range(len(ASSETS))
        ]
    )
    extended = np.array(
        [
            math.fsum(float(new[i]) * float(opinions[i, j]) for i in range(6))
            for j in range(len(ASSETS))
        ]
    )
    added_mass = float(new[-1])
    coefficients = new[:5] - (1 - added_mass) * old
    direct = added_mass * (opinions[-1] - baseline)
    redistributed = np.array(
        [
            math.fsum(float(coefficients[i]) * float(opinions[i, j]) for i in range(5))
            for j in range(len(ASSETS))
        ]
    )
    change = saved_new - saved_old
    residual = float(max(abs(change - direct - redistributed)))
    reconstruction = max(
        float(max(abs(baseline - saved_old))), float(max(abs(extended - saved_new)))
    )
    require(
        residual <= TOLERANCE and reconstruction <= TOLERANCE,
        "The signed identity must agree with both unchanged stored targets.",
    )
    old_l1 = math.fsum(abs(float(v)) for v in change)
    direct_l1 = math.fsum(abs(float(v)) for v in direct)
    retained_l1 = math.fsum(abs(float(v)) for v in redistributed)
    return {
        "decision_index": record["decision_index"],
        "decision_session": decision_dates()[record["decision_index"]],
        "score_mature": mature,
        "added_source_recorded_binary64_mass": added_mass,
        "added_source_opinion_BIL_weight": float(opinions[-1, ASSETS.index("BIL")]),
        "added_source_opinion_equity_weight": math.fsum(opinions[-1, list(EQUITY)]),
        "retained_source_redistribution_coefficients": coefficients.tolist(),
        "direct_added_source_term": direct.tolist(),
        "retained_source_redistribution_term": redistributed.tolist(),
        "stored_target_change": change.tolist(),
        "direct_added_source_L1": direct_l1,
        "retained_redistribution_L1": retained_l1,
        "stored_target_change_L1": old_l1,
        "absolute_components_minus_net_L1": math.fsum((direct_l1, retained_l1, -old_l1)),
        "retained_source_loss_labels_changed": 0
        if not mature
        else sum(a != b for a, b in zip(labels[0], labels[1][:5], strict=True)),
        "retained_source_NET_values_bit_exact": 5
        * len(record["five_before"]["matured_period_dates"]),
        "maximum_stored_target_reconstruction_residual": reconstruction,
        "maximum_signed_identity_residual": residual,
        "signed_terms_are_executable_targets": False,
        "main_PNL_or_causal_drawdown_attribution": False,
        "loss_normalization_isolated_counterfactual": False,
        "new_learning_targets_or_financial_trials": 0,
    }


def verify(record, result, policy):
    validate(policy)
    opinions, (old, new), (saved_old, saved_new), _, _ = _inputs(record)
    require(
        result["decision_index"] == record["decision_index"]
        and result["decision_session"] == decision_dates()[record["decision_index"]]
        and result["added_source_recorded_binary64_mass"] == float(new[-1])
        and result["signed_terms_are_executable_targets"] is False
        and result["main_PNL_or_causal_drawdown_attribution"] is False
        and result["loss_normalization_isolated_counterfactual"] is False
        and result["new_learning_targets_or_financial_trials"] == 0,
        "Keep source allocation influence separate from new capital or causal performance.",
    )
    f = Fraction.from_float
    a = f(float(new[-1]))
    coefficients = [f(float(new[i])) - (1 - a) * f(float(old[i])) for i in range(5)]
    maximum_term = maximum_target = 0.0
    for j in range(len(ASSETS)):
        five = sum(f(float(old[i])) * f(float(opinions[i, j])) for i in range(5))
        six = sum(f(float(new[i])) * f(float(opinions[i, j])) for i in range(6))
        direct = a * (f(float(opinions[-1, j])) - five)
        retained = sum(coefficients[i] * f(float(opinions[i, j])) for i in range(5))
        require(direct + retained == six - five, "Exact signed Fraction identity disagrees.")
        maximum_term = max(
            maximum_term,
            abs(float(direct) - result["direct_added_source_term"][j]),
            abs(float(retained) - result["retained_source_redistribution_term"][j]),
        )
        maximum_target = max(
            maximum_target,
            abs(float(five) - float(saved_old[j])),
            abs(float(six) - float(saved_new[j])),
        )
    maximum_coefficient = max(
        abs(float(v) - value)
        for v, value in zip(
            coefficients, result["retained_source_redistribution_coefficients"], strict=True
        )
    )
    require(
        maximum_term <= TOLERANCE
        and maximum_target <= TOLERANCE
        and maximum_coefficient <= TOLERANCE,
        "Independent exact-input arithmetic exceeds the unchanged unit/target limit.",
    )
    return {
        "exact_signed_algebra_identity": True,
        "maximum_independent_term_difference": maximum_term,
        "maximum_independent_coefficient_difference": maximum_coefficient,
        "maximum_independent_stored_target_difference": maximum_target,
    }


def summarize(rows, checks):
    def summary(selected):
        require(bool(selected), "An absent cohort is not a successful diagnostic result.")
        return {
            "origins": len(selected),
            "added_source_recorded_binary64_mass": _stats(
                [r["added_source_recorded_binary64_mass"] for r in selected]
            ),
            "added_source_binary64_zero_origins": sum(
                r["added_source_recorded_binary64_mass"] == 0 for r in selected
            ),
            "added_source_strict_above_initial_sixth_origins": sum(
                r["added_source_recorded_binary64_mass"] > 1 / 6 for r in selected
            ),
            "retained_term_strict_larger_L1_origins": sum(
                r["retained_redistribution_L1"] > r["direct_added_source_L1"] for r in selected
            ),
            "direct_term_strict_larger_L1_origins": sum(
                r["direct_added_source_L1"] > r["retained_redistribution_L1"] for r in selected
            ),
            **{
                key: _stats([r[key] for r in selected])
                for key in (
                    "direct_added_source_L1",
                    "retained_redistribution_L1",
                    "stored_target_change_L1",
                    "absolute_components_minus_net_L1",
                    "added_source_opinion_BIL_weight",
                    "added_source_opinion_equity_weight",
                )
            },
            "per_asset_signed_terms": {
                name: {
                    key: _stats([r[key][j] for r in selected])
                    for key in (
                        "direct_added_source_term",
                        "retained_source_redistribution_term",
                        "stored_target_change",
                    )
                }
                for j, name in enumerate(ASSETS)
            },
        }

    mature = [r for r in rows if r["score_mature"]]
    return {
        "matched_decisions": len(rows),
        "mature_decisions": len(mature),
        "initial_unscored_decisions": len(rows) - len(mature),
        "all_origins": summary(rows),
        "mature_origins": summary(mature),
        "retained_source_NET_values_bit_exact": sum(
            r["retained_source_NET_values_bit_exact"] for r in rows
        ),
        "mature_origins_with_retained_loss_label_changes": sum(
            r["retained_source_loss_labels_changed"] > 0 for r in mature
        ),
        "retained_source_loss_label_values_changed": sum(
            r["retained_source_loss_labels_changed"] for r in mature
        ),
        "maximum_stored_target_reconstruction_residual": max(
            r["maximum_stored_target_reconstruction_residual"] for r in rows
        ),
        "maximum_signed_identity_residual": max(
            r["maximum_signed_identity_residual"] for r in rows
        ),
        **{
            key: max(c[key] for c in checks)
            for key in (
                "maximum_independent_term_difference",
                "maximum_independent_coefficient_difference",
                "maximum_independent_stored_target_difference",
            )
        },
        "all_exact_signed_algebra_identities": all(
            c["exact_signed_algebra_identity"] for c in checks
        ),
        "signed_terms_are_executable_targets": False,
        "allocation_mass_is_main_PNL_or_causal_drawdown_attribution": False,
        "loss_normalization_isolated_counterfactual": False,
        "binary64_zero_is_analytic_zero_or_expert_exclusion": False,
        "new_learning_targets_shadow_accounts_or_financial_trials": 0,
    }
