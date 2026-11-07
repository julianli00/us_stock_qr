from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Protocol

from us_quant.calendar import is_month_end
from us_quant.confidence_decision_factorial_v1 import binary, trading_day
from us_quant.confidence_regret_scope_v1 import encode as fraction_encode
from us_quant.config import QuantError
from us_quant.retained_pair_equal_allocation_v1 import ASSETS
from us_quant.storage import digest_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/source-utility-geometry-v1.json"
STUDY = "ssd_source_utility_geometry_20261107_v1"
N, A, PRECISION = 6, len(ASSETS), 160
TOLERANCE = Fraction.from_float(1e-12)
EXPERT_IDS = (
    "semiconductor_option_dollar_profit_share_confirmation",
    "four_factor_defense_diversified",
    "conditional_covariance_hedge_expert_v1",
    "equity_first_month_end_commitment_v1",
    "benchmark_opportunity_joint_tail_v1",
    "semiconductor_ratchet_protection",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QuantError(message)


def encode(value):
    if isinstance(value, Fraction):
        return fraction_encode(value)
    if isinstance(value, tuple | list):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        require(all(type(k) is str for k in value), "Retained record keys must be strings.")
        return {k: encode(v) for k, v in value.items()}
    require(
        value is None or type(value) in (str, bool, int, float),
        "Unsupported retained operand type cannot be silently cast.",
    )
    require(type(value) is not float or math.isfinite(value), "Nonfinite retained scalar.")
    return value


def _decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "study_id": STUDY,
        "actual_native_event_seq": 201138,
        "prior_journal_sequence": 1542,
        "prior_combined_completed": 307,
        "source_inputs": {
            "path": "reports/ssd-continuous-research-v1/"
            "source-utility-geometry-20261107-v1/source-inputs-v1.json",
            "sha256": "01b4fa5b6fd539aa6f25b60846d1a41b5d50a9643e2fa04f2a4b930efea66437",
        },
        "source_experts": N,
        "source_origins": 121,
        "complete_source_months": 119,
        "past_month_utility_views": 7140,
        "source_records_per_game": 242,
        "expected_retained_input_resolution_rows": 14522,
        "utility_origin": "each_stored_current_rate_after_its_matured_source_month",
        "natural_utility": "original_canonical_NET_log_relative",
        "bounded_utility": "negative_original_sealed_binary64_loss_label",
        "centering": "equal_six_coordinate_mean_common_offset_quotient",
        "projection": "exact_rational_parallel_and_orthogonal_to_natural_log_utility",
        "energy_identity": "parallel_squared_norm_plus_orthogonal_squared_norm",
        "past_month_shares": "L1_centered_utility_magnitude_not_posterior_importance",
        "zero_magnitude_share_status": "explicit_undefined_not_zero_or_uniform",
        "analytic_infinite_rate": "constant_loss_history_zero_centered_utility_limit",
        "independent_decimal_precision": PRECISION,
        "independent_numeric_tolerance": 1e-12,
        "affine_ray_diagnostic_tolerance": 1e-12,
        "nonaffine_or_ordinal_disagreement_is_a_failure_gate": False,
        "prefix_cutoffs": [
            "2017-04-06",
            "2018-10-05",
            "2020-10-06",
            "2022-10-05",
            "2024-10-07",
        ],
        "source_rate_gap_forecast_or_target_regenerated": False,
        "source_posterior_is_rate_invariant": False,
        "utility_energy_or_magnitude_is_main_risk_or_wealth_attribution": False,
        "new_source_or_account_or_BT_or_old_diagnosis_runs": 0,
        "new_financial_trials": 0,
        "new_source_requests": 0,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    require(
        policy == expected and all(type(policy[k]) is type(v) for k, v in expected.items()),
        "Freeze the exact source utility comparison and descriptive-only interpretation.",
    )


@dataclass(frozen=True)
class UtilityMonth:
    start_after: str
    end: str
    log_hex: tuple[str, ...]
    loss_hex: tuple[str, ...]


@dataclass(frozen=True)
class StoredEndpoint:
    source_opinions_hex: tuple[tuple[str, ...], ...]
    source_known_at: tuple[str, ...]
    weight_hex: tuple[str, ...]
    target_hex: tuple[str, ...]


@dataclass(frozen=True)
class UtilitySnapshot:
    origin: str
    months: tuple[UtilityMonth, ...]
    rate_decimal: str
    original: StoredEndpoint
    adaptive: StoredEndpoint


class UtilityFeed(Protocol):
    origins: tuple[str, ...]

    def observe(self, origin: str) -> UtilitySnapshot: ...


@dataclass(frozen=True)
class _Projection:
    natural: tuple[Fraction, ...]
    bounded: tuple[Fraction, ...]
    slope: Fraction | None
    residual: tuple[Fraction, ...]
    parallel: tuple[Fraction, ...]
    orthogonal: tuple[Fraction, ...]
    displacement: tuple[Fraction, ...]
    natural_magnitude: Fraction
    bounded_magnitude: Fraction
    scaled_magnitude: Fraction
    parallel_energy: Fraction
    orthogonal_energy: Fraction
    total_energy: Fraction
    independent_error: str
    independent_operands: dict


def _center(values: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    mean = sum(values, Fraction(0)) / N
    return tuple(v - mean for v in values)


def _dot(left: tuple[Fraction, ...], right: tuple[Fraction, ...]) -> Fraction:
    return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))


def _magnitude(values: tuple[Fraction, ...]) -> Fraction:
    return sum(map(abs, values), Fraction(0))


def _rate(value: str) -> Fraction | None:
    require(type(value) is str, "A stored current rate must remain an explicit string.")
    if value == "analytic_infinity":
        return None
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise QuantError("Malformed stored current rate; do not replace it.") from error
    require(number.is_finite() and number > 0, "A finite stored current rate must be positive.")
    return Fraction(number)


def _month(month: UtilityMonth, origin: str) -> tuple[tuple[Fraction, ...], ...]:
    require(isinstance(month, UtilityMonth), "Use complete typed source-month operands.")
    start, end, day = map(trading_day, (month.start_after, month.end, origin))
    require(
        start < end <= day and is_month_end(start) and is_month_end(end),
        "Only an original complete, matured month can enter a utility view.",
    )
    logs, losses = binary(month.log_hex, N), binary(month.loss_hex, N)
    require(all(0 <= v <= 1 for v in losses), "Keep the original bounded loss labels.")
    return logs, losses


def _endpoint(endpoint: StoredEndpoint, origin: str) -> tuple[Fraction, Fraction]:
    require(
        isinstance(endpoint, StoredEndpoint)
        and len(endpoint.source_opinions_hex) == len(endpoint.source_known_at) == N,
        "Retain all six complete source opinions and known-at dates.",
    )
    day = trading_day(origin)
    require(
        all(trading_day(d) <= day for d in endpoint.source_known_at),
        "A source opinion cannot be refreshed from a future date.",
    )
    opinions = tuple(binary(row, A) for row in endpoint.source_opinions_hex)
    weights, target = binary(endpoint.weight_hex, N), binary(endpoint.target_hex, A)
    require(
        all(v >= 0 for row in opinions for v in row)
        and all(v >= 0 for v in (*weights, *target))
        and abs(sum(weights, Fraction(0)) - 1) <= TOLERANCE,
        "Preserve the stored complete-source mixture, not a different allocation.",
    )
    error = max(
        abs(sum(weights[i] * opinions[i][j] for i in range(N)) - target[j]) for j in range(A)
    )
    require(error <= TOLERANCE, "Stored target operands fail their unchanged mixture identity.")
    return error, _magnitude(weights)


def _independent(
    month: UtilityMonth, rate: Fraction | None, value: _Projection
) -> tuple[str, dict]:
    with localcontext() as context:
        context.prec = PRECISION
        logs = [Decimal.from_float(float.fromhex(v)) for v in month.log_hex]
        losses = [-Decimal.from_float(float.fromhex(v)) for v in month.loss_hex]
        u = [v - sum(logs) / N for v in logs]
        z = [v - sum(losses) / N for v in losses]
        norm = sum(v * v for v in u)
        slope = None if norm == 0 else sum(a * b for a, b in zip(u, z, strict=True)) / norm
        remainder = z if slope is None else [b - slope * a for a, b in zip(u, z, strict=True)]
        if rate is None:
            parallel, orthogonal = [-v for v in u], [Decimal(0)] * N
            displacement = parallel
        else:
            eta = _decimal(rate)
            parallel = [Decimal(0)] * N if slope is None else [(eta * slope - 1) * v for v in u]
            orthogonal = [eta * v for v in remainder]
            displacement = [eta * b - a for a, b in zip(u, z, strict=True)]
        errors = []
        for actual, expected in (
            (u, value.natural),
            (z, value.bounded),
            (remainder, value.residual),
            (parallel, value.parallel),
            (orthogonal, value.orthogonal),
            (displacement, value.displacement),
        ):
            errors.extend(abs(a - _decimal(b)) for a, b in zip(actual, expected, strict=True))
        flat_status_matches = (slope is None) == (value.slope is None)
        if slope is not None and value.slope is not None:
            errors.append(abs(slope - _decimal(value.slope)))
        for vector, energy in (
            (parallel, value.parallel_energy),
            (orthogonal, value.orthogonal_energy),
            (displacement, value.total_energy),
        ):
            errors.append(abs(sum(v * v for v in vector) - _decimal(energy)))
        maximum = max(errors)
        return str(maximum), {
            "natural_centered_decimal": [str(v) for v in u],
            "bounded_centered_decimal": [str(v) for v in z],
            "slope_decimal": None if slope is None else str(slope),
            "residual_decimal": [str(v) for v in remainder],
            "parallel_displacement_decimal": [str(v) for v in parallel],
            "orthogonal_displacement_decimal": [str(v) for v in orthogonal],
            "displacement_decimal": [str(v) for v in displacement],
            "flat_status_matches": flat_status_matches,
        }


def _project(month: UtilityMonth, origin: str, rate: Fraction | None) -> _Projection:
    logs, losses = _month(month, origin)
    natural, bounded = _center(logs), _center(tuple(-v for v in losses))
    norm = _dot(natural, natural)
    slope = None if norm == 0 else _dot(natural, bounded) / norm
    residual = (
        bounded
        if slope is None
        else tuple(b - slope * a for a, b in zip(natural, bounded, strict=True))
    )
    if rate is None:
        require(not any(bounded), "Infinite rate requires analytic constant-loss history.")
        # Constant losses have zero centered utility at every finite rate.
        parallel = tuple(-v for v in natural)
        orthogonal = (Fraction(0),) * N
        displacement = parallel
        scaled_magnitude = Fraction(0)
    else:
        parallel = (
            (Fraction(0),) * N if slope is None else tuple((rate * slope - 1) * v for v in natural)
        )
        orthogonal = tuple(rate * v for v in residual)
        displacement = tuple(rate * b - a for a, b in zip(natural, bounded, strict=True))
        scaled_magnitude = rate * _magnitude(bounded)
    energies = tuple(_dot(v, v) for v in (parallel, orthogonal, displacement))
    value = _Projection(
        natural,
        bounded,
        slope,
        residual,
        parallel,
        orthogonal,
        displacement,
        _magnitude(natural),
        _magnitude(bounded),
        scaled_magnitude,
        *energies,
        "",
        {},
    )
    error, operands = _independent(month, rate, value)
    return _Projection(
        **{**asdict(value), "independent_error": error, "independent_operands": operands}
    )


def _stats(values: list[Fraction]) -> dict:
    if not values:
        return {"observations": 0, "minimum": None, "median": None, "mean": None, "maximum": None}
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {
        "observations": len(values),
        "minimum": float(ordered[0]),
        "median": float(median),
        "mean": float(sum(values, Fraction(0)) / len(values)),
        "maximum": float(ordered[-1]),
    }


def inspect(
    feed: UtilityFeed,
    *,
    retain: Callable[[dict], None] | None = None,
    before_origin: Callable[[], None] | None = None,
) -> dict:
    origins = feed.origins
    require(
        type(origins) is tuple
        and origins
        and all(type(d) is str for d in origins)
        and len(set(origins)) == len(origins)
        and list(origins) == sorted(origins),
        "Preserve all ordered, unique source origins.",
    )
    for origin in origins:
        trading_day(origin)
    origin_rows, cell_rows, monthly = [], [], {}
    weight_deltas, target_deltas, share_distances = [], [], []
    effective_scales, energies = [], [Fraction(0)] * 3
    untrained = infinite = same_opinions = same_dates = cancellations = undefined = 0
    maximum_decimal = Decimal(0)
    maximum_target_error = Fraction(0)
    previous_months: tuple[UtilityMonth, ...] = ()
    for origin in origins:
        if before_origin is not None:
            before_origin()
        snapshot = feed.observe(origin)
        require(isinstance(snapshot, UtilitySnapshot), "The Adapter returned an invalid snapshot.")
        if retain is not None:
            retain({"phase": "before_origin", "origin": origin, "snapshot": asdict(snapshot)})
        require(
            snapshot.origin == origin,
            "The Adapter returned a different source origin.",
        )
        months = snapshot.months
        require(
            type(months) is tuple
            and len(months) <= 119
            and all(isinstance(m, UtilityMonth) for m in months)
            and months[: len(previous_months)] == previous_months
            and len({m.end for m in months}) == len(months),
            "Stored source history cannot regress, change prior labels or duplicate a month.",
        )
        for a, b in zip(months, months[1:], strict=False):
            require(a.end == b.start_after, "The original source-month chain has a gap.")
        rate = _rate(snapshot.rate_decimal)
        old_error, _ = _endpoint(snapshot.original, origin)
        new_error, _ = _endpoint(snapshot.adaptive, origin)
        maximum_target_error = max(maximum_target_error, old_error, new_error)
        source_equal = (
            snapshot.original.source_opinions_hex == snapshot.adaptive.source_opinions_hex
        )
        dates_equal = snapshot.original.source_known_at == snapshot.adaptive.source_known_at
        same_opinions += source_equal
        same_dates += dates_equal
        old_weights, new_weights = (
            binary(e.weight_hex, N) for e in (snapshot.original, snapshot.adaptive)
        )
        old_target, new_target = (
            binary(e.target_hex, A) for e in (snapshot.original, snapshot.adaptive)
        )
        weight_delta = _magnitude(
            tuple(a - b for a, b in zip(old_weights, new_weights, strict=True))
        )
        target_delta = _magnitude(tuple(a - b for a, b in zip(old_target, new_target, strict=True)))
        weight_deltas.append(weight_delta)
        target_deltas.append(target_delta)
        natural_sizes, loss_sizes, scaled_sizes = [], [], []
        for month in months:
            if retain is not None:
                retain(
                    {
                        "phase": "before_cell",
                        "origin": origin,
                        "month": asdict(month),
                        "current_rate_decimal": snapshot.rate_decimal,
                    }
                )
            value = _project(month, origin, rate)
            exact = (
                tuple(a + b for a, b in zip(value.parallel, value.orthogonal, strict=True))
                == value.displacement
                and _dot(value.natural, value.residual) == 0
                and value.parallel_energy + value.orthogonal_energy == value.total_energy
            )
            record = {
                "phase": "after_cell",
                "origin": origin,
                "month_end": month.end,
                "current_rate_decimal": snapshot.rate_decimal,
                "projection": asdict(value),
                "energy_and_reconstruction_exact": exact,
            }
            if retain is not None:
                retain(record)
            error = Decimal(value.independent_error)
            require(
                exact
                and value.independent_operands["flat_status_matches"]
                and error.is_finite()
                and Fraction(error) <= TOLERANCE,
                "Retained independent source-utility comparison or exact identity failed.",
            )
            cell_rows.append(
                {"origin": origin, "month_end": month.end, "sha256": digest_json(encode(record))}
            )
            natural_sizes.append(value.natural_magnitude)
            loss_sizes.append(value.bounded_magnitude)
            scaled_sizes.append(value.scaled_magnitude)
            for i, energy in enumerate(
                (value.parallel_energy, value.orthogonal_energy, value.total_energy)
            ):
                energies[i] += energy
            maximum_decimal = max(maximum_decimal, Decimal(value.independent_error))
            if rate is not None and value.slope is not None:
                effective_scales.append(rate * value.slope)
            if month.end not in monthly:
                reversals = created_ties = broken_ties = 0
                for i in range(N):
                    for j in range(i):
                        a, b = (
                            value.natural[i] - value.natural[j],
                            value.bounded[i] - value.bounded[j],
                        )
                        reversals += a * b < 0
                        created_ties += a != 0 and b == 0
                        broken_ties += a == 0 and b != 0
                monthly[month.end] = {
                    "natural_flat": not any(value.natural),
                    "bounded_flat": not any(value.bounded),
                    "slope": value.slope,
                    "ray_residual_linf": max(map(abs, value.residual)),
                    "strict_rank_reversals": reversals,
                    "created_ties": created_ties,
                    "broken_ties": broken_ties,
                }
        raw_total, loss_total, scaled_total = (
            sum(values, Fraction(0)) for values in (natural_sizes, loss_sizes, scaled_sizes)
        )
        raw_shares = None if raw_total == 0 else [v / raw_total for v in natural_sizes]
        loss_shares = None if loss_total == 0 else [v / loss_total for v in loss_sizes]
        scaled_shares = None if scaled_total == 0 else [v / scaled_total for v in scaled_sizes]
        cancelled = rate is not None and loss_shares is not None
        if cancelled:
            require(
                scaled_shares == loss_shares,
                "A common positive rate failed exact share cancellation.",
            )
            cancellations += 1
        distance = (
            None
            if raw_shares is None or scaled_shares is None
            else sum(
                (abs(a - b) for a, b in zip(raw_shares, scaled_shares, strict=True)), Fraction(0)
            )
        )
        if distance is None:
            undefined += 1
        else:
            share_distances.append(distance)
        untrained += not months
        infinite += rate is None
        record = {
            "phase": "after_origin",
            "origin": origin,
            "matured_months": len(months),
            "rate_status": "analytic_infinite_constant_loss" if rate is None else "stored_finite",
            "same_source_opinions": source_equal,
            "same_source_known_dates": dates_equal,
            "stored_weight_L1": weight_delta,
            "stored_target_L1": target_delta,
            "natural_history_magnitude_shares": raw_shares,
            "bounded_history_magnitude_shares": scaled_shares,
            "history_share_L1_distance": distance,
            "history_share_status": "undefined_zero_magnitude"
            if distance is None
            else "comparable",
            "common_rate_share_cancellation_exact": cancelled,
            "shares_are_source_posterior_importance": False,
        }
        if retain is not None:
            retain(record)
        origin_rows.append({"origin": origin, "sha256": digest_json(encode(record))})
        previous_months = months
    require(energies[0] + energies[1] == energies[2], "Aggregate utility energy is not exact.")
    return {
        "source_origins": len(origins),
        "complete_source_months": len(monthly),
        "past_month_utility_views": len(cell_rows),
        "untrained_origins": untrained,
        "analytic_infinite_origins": infinite,
        "same_source_opinion_pairs": same_opinions,
        "same_source_known_date_pairs": same_dates,
        "cached_source_weight_L1": _stats(weight_deltas),
        "cached_underlying_target_L1": _stats(target_deltas),
        "monthly_geometry": {
            "natural_flat_months": sum(v["natural_flat"] for v in monthly.values()),
            "bounded_flat_months": sum(v["bounded_flat"] for v in monthly.values()),
            "affine_ray_within_original_tolerance_months": sum(
                v["ray_residual_linf"] <= TOLERANCE for v in monthly.values()
            ),
            "maximum_unscaled_ray_residual": float(
                max((v["ray_residual_linf"] for v in monthly.values()), default=Fraction(0))
            ),
            "unscaled_ray_slopes": _stats(
                [v["slope"] for v in monthly.values() if v["slope"] is not None]
            ),
            "strict_rank_reversals": sum(v["strict_rank_reversals"] for v in monthly.values()),
            "created_ties": sum(v["created_ties"] for v in monthly.values()),
            "broken_ties": sum(v["broken_ties"] for v in monthly.values()),
        },
        "current_rate_effective_ray_scales": _stats(effective_scales),
        "history_utility_magnitude_shares": {
            "comparable_origins": len(share_distances),
            "undefined_zero_magnitude_origins": undefined,
            "common_positive_rate_cancellations_exact": cancellations,
            "natural_versus_bounded_share_L1": _stats(share_distances),
            "source_posterior_is_rate_invariant": False,
        },
        "utility_energy": {
            "parallel": float(energies[0]),
            "orthogonal": float(energies[1]),
            "total": float(energies[2]),
            "orthogonal_energy_fraction": None
            if energies[2] == 0
            else float(energies[1] / energies[2]),
            "additivity_exact": True,
            "is_main_risk_or_wealth_attribution": False,
        },
        "maximum_independent_decimal_error": str(maximum_decimal),
        "maximum_cached_target_mixture_error": float(maximum_target_error),
        "origin_row_hashes": origin_rows,
        "cell_row_hashes": cell_rows,
        "new_financial_trials": 0,
        "old_learner_source_target_or_account_replayed": False,
        "source_rate_or_forecast_regenerated": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
