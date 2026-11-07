from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import pandas as pd

from us_quant.calendar import is_month_end, sessions
from us_quant.confidence_regret_scope_v1 import decimal_fraction, encode
from us_quant.config import QuantError
from us_quant.retained_pair_equal_allocation_v1 import ASSETS
from us_quant.storage import digest_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/confidence-decision-factorial-v1.json"
STUDY = "ssd_confidence_decision_factorial_20261107_v1"
PATHS = ("10y-base", "10y-stress", "5y-base", "5y-stress")
AXES = ("learner_mass_and_rates", "participation", "source_opinion_vintage")
N, A, PRECISION, TOLERANCE = 6, len(ASSETS), 160, Fraction.from_float(1e-12)


def require(condition, message):
    if not condition:
        raise QuantError(message)


def validate(policy):
    expected = {
        "schema_version": 1,
        "study_id": STUDY,
        "actual_native_event_seq": 198798,
        "prior_journal_sequence": 1518,
        "prior_combined_completed": 305,
        "source_inputs": {
            "path": "reports/ssd-continuous-research-v1/"
            "confidence-decision-factorial-20261107-v1/source-inputs-v1.json",
            "sha256": "d63d1081f9c06235c81cc151c5940ec348b64a6f9026e72a262dcadc8f0a0d35",
        },
        "axes": list(AXES),
        "views_per_pair": 8,
        "equal_weight_shapley_orders": 6,
        "source_experts": 6,
        "underlying_assets": list(ASSETS),
        "original_global_snapshots": 121,
        "original_global_trace_rows": 242,
        "saved_pairs": 364,
        "expected_decisions": {"10y-base": 121, "10y-stress": 121, "5y-base": 61, "5y-stress": 61},
        "expected_main_input_resolution_rows": 728,
        "old_master_selection": "last_known_saved_master_not_new_source_refresh",
        "recent_master_selection": "actual_logged_main_decision",
        "arithmetic": "exact_Fractions_of_saved_decimal_and_binary64_operands",
        "independent_decimal_precision": PRECISION,
        "unchanged_source_target_coherence": 1e-12,
        "prefix_positions": [1, 2, 12, "half", "full"],
        "prefix_inspections": 20,
        "state_is_joint_raw_mass_and_rate_not_period_count": True,
        "raw_source_mass_rewritten_or_renormalized": False,
        "all_crossed_inputs_were_known_by_capital_review": True,
        "timestamp_change_implies_numeric_opinion_change": False,
        "groupings": ["overall", "initial_capital", "regular_month_end"],
        "projection_is_executable_policy_or_causal_wealth_attribution": False,
        "old_learner_weights_update_generator_account_BT_or_diagnosis_replayed": False,
        "source_CAGR_Sharpe_drawdown_or_qualification_recomputed": False,
        "new_financial_trials": 0,
        "new_executable_recommendations_or_accounts": 0,
        "new_source_or_RF_or_event_Adapter": False,
        "new_source_requests": 0,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    require(
        policy == expected and all(type(policy[k]) is type(v) for k, v in expected.items()),
        "Freeze saved factor coordinates,all comparisons and descriptive-only interpretation.",
    )


def trading_day(value):
    require(type(value) is str, "Preserve explicit saved source-clock dates.")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise QuantError("Invalid saved clock cannot be filled or shifted.") from error
    require(str(parsed) == value, "Use canonical dates without implicit timezone conversion.")
    day = pd.Timestamp(value)
    require(len(sessions(day, day)) == 1, "A saved review must be an original exchange session.")
    return day


def binary(values, length):
    require(
        isinstance(values, tuple | list)
        and len(values) == length
        and all(type(v) is str for v in values),
        "Keep complete saved binary64 coordinates rather than silently casting input.",
    )
    result = []
    for text in values:
        try:
            value = float.fromhex(text)
        except (ValueError, OverflowError) as error:
            raise QuantError("Malformed saved binary64 operand.") from error
        require(math.isfinite(value) and value.hex() == text, "Use canonical finite saved hex.")
        result.append(Fraction.from_float(value))
    return tuple(result)


@dataclass(frozen=True)
class SavedSnapshot:
    as_of: str
    raw_mass_decimal: tuple[str, ...]
    rates_decimal: tuple[str, ...]
    complete_intervals: int
    confidence_hex: tuple[str, ...]
    source_targets_hex: tuple[tuple[str, ...], ...]
    native_target_known_at: tuple[str, ...]
    expert_weights_hex: tuple[str, ...]
    target_hex: tuple[str, ...]


@dataclass(frozen=True)
class SavedPair:
    path_key: str
    signal_session: str
    initial_capital: bool
    original: SavedSnapshot
    recent: SavedSnapshot
    original_reason: str
    recent_reason: str
    original_master_csv_hex: tuple[str, ...]
    recent_desired_csv_hex: tuple[str, ...]
    original_issued_hex: tuple[str, ...] | None
    recent_issued_hex: tuple[str, ...] | None


@dataclass(frozen=True)
class _Operands:
    masses: tuple[Fraction, ...]
    rates: tuple[Fraction, ...]
    confidence: tuple[Fraction, ...]
    opinions: tuple[tuple[Fraction, ...], ...]
    weights: tuple[Fraction, ...]
    target: tuple[Fraction, ...]


def operands(saved):
    day = trading_day(saved.as_of)
    require(
        len(saved.raw_mass_decimal) == len(saved.rates_decimal) == N
        and type(saved.complete_intervals) is int
        and 0 <= saved.complete_intervals <= 119,
        "Keep the complete joint raw learner state and its recorded maturity.",
    )
    masses = tuple(decimal_fraction(v) for v in saved.raw_mass_decimal)
    rates = tuple(decimal_fraction(v) for v in saved.rates_decimal)
    confidence = binary(saved.confidence_hex, N)
    weights, target = binary(saved.expert_weights_hex, N), binary(saved.target_hex, A)
    require(
        all(v > 0 for v in masses)
        and all(0 < v <= Fraction(1, 2) for v in rates)
        and all(0 <= v <= 1 for v in confidence)
        and confidence[:5] == (Fraction(1),) * 5
        and all(v >= 0 for v in weights)
        and abs(sum(weights) - 1) <= TOLERANCE
        and all(v >= 0 for v in target)
        and abs(sum(target) - Fraction("0.98")) <= TOLERANCE,
        "Invalid state,participation or source endpoint cannot become a neutral view.",
    )
    require(
        len(saved.source_targets_hex) == len(saved.native_target_known_at) == N,
        "Every complete source policy needs its own original opinion and clock.",
    )
    opinions = tuple(binary(v, A) for v in saved.source_targets_hex)
    require(
        all(trading_day(d) <= day for d in saved.native_target_known_at)
        and all(
            all(v >= 0 for v in row) and abs(sum(row) - Fraction("0.98")) <= TOLERANCE
            for row in opinions
        ),
        "Missing or future native opinions cannot be repaired by carrying a different source.",
    )
    return _Operands(masses, rates, confidence, opinions, weights, target)


def projection(state, participation, opinion):
    mass = tuple(
        m * e * c
        for m, e, c in zip(
            state.masses,
            state.rates,
            participation.confidence,
            strict=True,
        )
    )
    denominator = sum(mass)
    require(denominator > 0, "The algebraic confidence projection has no active mass.")
    weights = tuple(v / denominator for v in mass)
    target = tuple(sum(weights[i] * opinion.opinions[i][j] for i in range(N)) for j in range(A))
    return weights, target


def independent_projection(state, participation, opinion):
    with localcontext() as context:
        context.prec = PRECISION
        raw = [
            Decimal(m) * Decimal(e) * Decimal.from_float(float.fromhex(c))
            for m, e, c in zip(
                state.raw_mass_decimal,
                state.rates_decimal,
                participation.confidence_hex,
                strict=True,
            )
        ]
        weights = [v / sum(raw) for v in raw]
        target = [
            sum(
                weights[i] * Decimal.from_float(float.fromhex(opinion.source_targets_hex[i][j]))
                for i in range(N)
            )
            for j in range(A)
        ]
    return tuple(Fraction(v) for v in weights), tuple(Fraction(v) for v in target)


def inspect_pair(case):
    day = trading_day(case.signal_session)
    require(
        case.path_key in PATHS
        and type(case.initial_capital) is bool
        and trading_day(case.original.as_of) <= day == trading_day(case.recent.as_of)
        and (case.initial_capital or is_month_end(day))
        and (case.initial_capital or case.original.as_of == case.signal_session),
        "Keep original initial/monthly capital clocks and both then-known master snapshots.",
    )
    for reason, issued in (
        (case.original_reason, case.original_issued_hex),
        (case.recent_reason, case.recent_issued_hex),
    ):
        require(
            type(reason) is str
            and bool(reason)
            and ((reason == "hold") is (issued is None))
            and ((reason == "initial_capital") is case.initial_capital),
            "An unissued hold is not a request,and capital initialization is not a source refresh.",
        )
    old, new = operands(case.original), operands(case.recent)
    parsed, saved = (old, new), (case.original, case.recent)
    views, oracle_error = {}, Fraction(0)
    for mask in range(8):
        indexes = tuple((mask >> axis) & 1 for axis in range(3))
        chosen = [parsed[i] for i in indexes]
        weights, target = projection(*chosen)
        independent_weights, independent_target = independent_projection(
            *[saved[i] for i in indexes]
        )
        oracle_error = max(
            oracle_error,
            *[abs(a - b) for a, b in zip(weights, independent_weights, strict=True)],
            *[abs(a - b) for a, b in zip(target, independent_target, strict=True)],
        )
        views[mask] = (weights, target)
    endpoint_error = max(
        *[abs(a - b) for a, b in zip(views[0][0], old.weights, strict=True)],
        *[abs(a - b) for a, b in zip(views[7][0], new.weights, strict=True)],
        *[abs(a - b) for a, b in zip(views[0][1], old.target, strict=True)],
        *[abs(a - b) for a, b in zip(views[7][1], new.target, strict=True)],
        *[
            abs(a - b)
            for a, b in zip(
                binary(case.original_master_csv_hex, A),
                old.target,
                strict=True,
            )
        ],
        *[
            abs(a - b)
            for a, b in zip(
                binary(case.recent_desired_csv_hex, A),
                new.target,
                strict=True,
            )
        ],
    )
    for issued, endpoint in (
        (case.original_issued_hex, old.target),
        (case.recent_issued_hex, new.target),
    ):
        if issued is not None:
            endpoint_error = max(
                endpoint_error,
                *[abs(a - b) for a, b in zip(binary(issued, A), endpoint, strict=True)],
            )
    require(
        max(endpoint_error, oracle_error) <= TOLERANCE,
        "Saved endpoints,issued provenance or independent projection coherence failed.",
    )
    contributions = {}
    for axis, name in enumerate(AXES):
        value = [Fraction(0)] * A
        for mask in range(8):
            if mask & (1 << axis):
                continue
            weight = Fraction(1, 3) if mask.bit_count() in (0, 2) else Fraction(1, 6)
            left, right = views[mask][1], views[mask | (1 << axis)][1]
            for j in range(A):
                value[j] += weight * (right[j] - left[j])
        contributions[name] = tuple(value)
    difference = tuple(b - a for a, b in zip(views[0][1], views[7][1], strict=True))
    require(
        all(sum(contributions[name][j] for name in AXES) == difference[j] for j in range(A)),
        "The equal-order exact factorial allocation must add without a residual.",
    )
    axis_l1 = {name: sum(abs(v) for v in values) for name, values in contributions.items()}
    total_l1 = sum(abs(v) for v in difference)
    cancellation = sum(axis_l1.values()) - total_l1
    require(cancellation >= 0, "Signed contributions may cancel,not fabricate displacement.")
    return {
        "path_key": case.path_key,
        "signal_session": case.signal_session,
        "initial_capital": case.initial_capital,
        "original_master_known_at": case.original.as_of,
        "recent_master_known_at": case.recent.as_of,
        "original_complete_intervals": case.original.complete_intervals,
        "recent_complete_intervals": case.recent.complete_intervals,
        "joint_mass_rate_state_changed": (old.masses, old.rates) != (new.masses, new.rates),
        "participation_changed": old.confidence != new.confidence,
        "numeric_source_opinions_changed": old.opinions != new.opinions,
        "source_snapshot_date_changed": case.original.as_of != case.recent.as_of,
        "native_opinion_clock_changed": (
            case.original.native_target_known_at != case.recent.native_target_known_at
        ),
        "original_request_issued": case.original_reason != "hold",
        "recent_request_issued": case.recent_reason != "hold",
        "eight_algebraic_views": [
            {
                "mask": mask,
                "expert_fractions_exact": [encode(v) for v in weights],
                "target_exact": [encode(v) for v in target],
            }
            for mask, (weights, target) in views.items()
        ],
        "axis_target_contributions_exact": {
            name: [encode(v) for v in values] for name, values in contributions.items()
        },
        "axis_l1_exact": {name: encode(v) for name, v in axis_l1.items()},
        "net_target_displacement_l1_exact": encode(total_l1),
        "cancelling_l1_exact": encode(cancellation),
        "exact_shapley_additivity_zero_residual": True,
        "maximum_saved_endpoint_difference": float(endpoint_error),
        "maximum_independent_projection_difference": float(oracle_error),
        "projection_is_executable_policy_or_causal_wealth_attribution": False,
    }


def _decode(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def _summary(rows):
    count = len(rows)
    totals = {
        name: sum((_decode(r["axis_l1_exact"][name]) for r in rows), Fraction(0)) for name in AXES
    }
    return {
        "cases": count,
        "joint_mass_rate_state_changed": sum(r["joint_mass_rate_state_changed"] for r in rows),
        "participation_changed": sum(r["participation_changed"] for r in rows),
        "numeric_source_opinions_changed": sum(r["numeric_source_opinions_changed"] for r in rows),
        "source_snapshot_date_changed": sum(r["source_snapshot_date_changed"] for r in rows),
        "native_opinion_clock_changed": sum(r["native_opinion_clock_changed"] for r in rows),
        "both_requested": sum(
            r["original_request_issued"] and r["recent_request_issued"] for r in rows
        ),
        "both_held": sum(
            not r["original_request_issued"] and not r["recent_request_issued"] for r in rows
        ),
        "only_original_requested": sum(
            r["original_request_issued"] and not r["recent_request_issued"] for r in rows
        ),
        "only_recent_requested": sum(
            not r["original_request_issued"] and r["recent_request_issued"] for r in rows
        ),
        "mean_axis_target_L1": {
            name: float(value / count) if count else None for name, value in totals.items()
        },
        "mean_net_target_displacement_L1": float(
            sum(
                (_decode(r["net_target_displacement_l1_exact"]) for r in rows),
                Fraction(0),
            )
            / count
        )
        if count
        else None,
        "mean_cancelling_L1": float(
            sum(
                (_decode(r["cancelling_l1_exact"]) for r in rows),
                Fraction(0),
            )
            / count
        )
        if count
        else None,
    }


def inspect(
    adapter: Iterable[SavedPair],
    *,
    retain: Callable[[dict], None],
):
    require(
        callable(retain), "Retain exact input operands before inspection,not just successful rows."
    )
    rows, hashes, previous = [], [], {}
    for case in adapter:
        raw = asdict(case)
        retain({"phase": "input", "case": raw})
        require(
            case.path_key not in previous or case.signal_session > previous[case.path_key],
            "A repeated or unordered saved capital review cannot be discarded.",
        )
        result = inspect_pair(case)
        retain({"phase": "resolved", "result": result})
        hashes.append(
            {
                "path_key": case.path_key,
                "signal_session": case.signal_session,
                "input_sha256": digest_json(raw),
                "resolved_sha256": digest_json(result),
            }
        )
        previous[case.path_key] = case.signal_session
        rows.append(
            {
                k: v
                for k, v in result.items()
                if k not in ("eight_algebraic_views", "axis_target_contributions_exact")
            }
        )
    require(bool(rows), "No saved decision pairs were available for the frozen comparison.")
    return {
        "pairs": len(rows),
        "algebraic_views": 8 * len(rows),
        "all_exact_additivity_zero_residual": True,
        "maximum_saved_endpoint_difference": max(
            r["maximum_saved_endpoint_difference"] for r in rows
        ),
        "maximum_independent_projection_difference": max(
            r["maximum_independent_projection_difference"] for r in rows
        ),
        "paths": {
            key: {
                "overall": _summary([r for r in rows if r["path_key"] == key]),
                "initial_capital": _summary(
                    [r for r in rows if r["path_key"] == key and r["initial_capital"]]
                ),
                "regular_month_end": _summary(
                    [r for r in rows if r["path_key"] == key and not r["initial_capital"]]
                ),
            }
            for key in PATHS
            if any(r["path_key"] == key for r in rows)
        },
        "retained_row_hashes": hashes,
        "new_financial_trials": 0,
        "new_executable_recommendations_or_accounts": 0,
        "independent_forward_validation": False,
        "order_authority": False,
    }
