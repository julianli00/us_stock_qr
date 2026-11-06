from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import sessions
from us_quant.cash_funded_accounting_v2 import ORDER_NOTIONAL_THRESHOLD
from us_quant.config import QuantError
from us_quant.retained_pair_equal_allocation_v1 import ASSETS
from us_quant.saved_inclusion_execution_v1 import MONEY_TOLERANCE, TARGET_TOLERANCE

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/execution-gap-envelope-v1.json"
N, BIL = len(ASSETS), ASSETS.index("BIL")
PATHS = ("10y-base", "10y-stress", "5y-base", "5y-stress")


def require(condition, message):
    if not condition:
        raise QuantError(message)


def validate(policy):
    expected = {
        "schema_version": 1,
        "study_id": "ssd_execution_gap_envelope_20261106_v1",
        "actual_native_event_seq": 195083,
        "prior_journal_sequence": 1490,
        "source_inputs": {
            "path": "reports/ssd-continuous-research-v1/execution-gap-envelope-20261106-v1/"
            "source-inputs-v1.json",
            "sha256": "afb0d60ba12ff6be2b6ba3967bcd34d9629babe54121b5a56859818eff840b7f",
        },
        "planning_delay_sessions": 2,
        "planning_cost_bps_per_side": 20.0,
        "planning_commission_per_changed_asset": 1.0,
        "maximum_asset_tickets": N,
        "reference_peak_fraction": 0.85,
        "nominal_ETF_budget": 0.98,
        "nominal_budget_tolerance": TARGET_TOLERANCE,
        "money_coherence_USD": MONEY_TOLERANCE,
        "notional_noop_threshold_USD": ORDER_NOTIONAL_THRESHOLD,
        "scenario_selection": "all_expanding_fully_known_two_transition_price_windows",
        "scenario_marks": ["first_wait_close", "second_open", "second_close"],
        "cost_bound": (
            "scenario_open_gross_plus_nominal_budget_times_open_NAV_times_rate_plus14tickets"
        ),
        "postfill_noop_slack": "noop_threshold_times_sum_abs_fill_mark_minus1_plus_money_coherence",
        "interval_arithmetic": "exact_Fractions_of_represented_affine_coefficients",
        "empty_interval_has_selected_fraction": False,
        "known_main_contexts": {"10y-base": 121, "10y-stress": 121, "5y-base": 61, "5y-stress": 61},
        "total_saved_contexts": 364,
        "current_book_state_is_replaced_by_new_wealth": False,
        "hypothetical_fill_is_actual_original_request_or_band_decision": False,
        "certificate_covers_full_month_or_unknown_future_support": False,
        "certificate_is_maximal_actual_trade_or_drawdown_guarantee": False,
        "prior_headroom_V1_repaired_retried_or_executed": False,
        "source_learner_or_target_or_account_or_prior_diagnosis_replayed": False,
        "new_target_vectors_or_accounts": 0,
        "new_financial_trials": 0,
        "new_source_or_RF_or_event_Adapter": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    require(
        policy == expected and all(type(policy[k]) is type(v) for k, v in expected.items()),
        "Freeze the finite execution-gap reference without changing financial acceptance.",
    )


@dataclass(frozen=True)
class MainMark:
    as_of: pd.Timestamp
    asset_values: np.ndarray
    cash: float
    nav: float
    peak: float

    def validate(self):
        require(
            isinstance(self.as_of, pd.Timestamp)
            and self.as_of.tzinfo is None
            and self.asset_values.shape == (N,)
            and self.asset_values.dtype.kind in "fi"
            and np.isfinite(self.asset_values).all()
            and (self.asset_values >= 0).all()
            and np.isfinite([self.cash, self.nav, self.peak]).all()
            and self.cash >= 0
            and 0 < self.nav <= self.peak
            and self.peak >= 10000
            and abs(math.fsum(self.asset_values) + self.cash - self.nav) <= MONEY_TOLERANCE,
            "Use the actual known main holdings,cash,NAV and fresh-capital closing peak.",
        )


@dataclass(frozen=True)
class StressSample:
    as_of: pd.Timestamp
    starts: pd.DatetimeIndex
    wait_closes: pd.DatetimeIndex
    ends: pd.DatetimeIndex
    prefill: np.ndarray
    opening: np.ndarray
    fill_mark: np.ndarray

    def validate(self):
        require(
            isinstance(self.as_of, pd.Timestamp)
            and self.as_of.tzinfo is None
            and len(self.ends) > 0
            and len(self.starts) == len(self.wait_closes) == len(self.ends)
            and all(
                index.tz is None and index.is_unique and index.is_monotonic_increasing
                for index in (self.starts, self.wait_closes, self.ends)
            )
            and (self.starts < self.wait_closes).all()
            and (self.wait_closes < self.ends).all()
            and (self.ends <= self.as_of).all()
            and all(
                row.shape == (len(self.ends), N)
                and row.dtype.kind in "fi"
                and np.isfinite(row).all()
                and (row > 0).all()
                for row in (self.prefill, self.opening, self.fill_mark)
            ),
            "A finite stress sample needs complete positive known-path marks,not future prices.",
        )


class KnownPriceArchive:
    def __init__(self, opening, closing):
        require(
            isinstance(opening, pd.DataFrame)
            and isinstance(closing, pd.DataFrame)
            and opening.index.equals(closing.index)
            and tuple(opening.columns) == tuple(closing.columns) == tuple(ASSETS)
            and len(closing) >= 3
            and closing.index.equals(sessions(closing.index[0], closing.index[-1]))
            and np.isfinite(opening.to_numpy()).all()
            and np.isfinite(closing.to_numpy()).all()
            and (opening > 0).all().all()
            and (closing > 0).all().all(),
            "Use only the original complete admitted adjusted-open/close lattice.",
        )
        self.opening, self.closing = opening.copy(), closing.copy()

    def observe(self, as_of):
        day = pd.Timestamp(as_of)
        require(day in self.closing.index, "A missing information date cannot be filled.")
        close = self.closing.loc[:day]
        opened = self.opening.loc[close.index]
        require(len(close) >= 3, "At least one fully known two-transition scenario is required.")
        values = close.to_numpy()
        return StressSample(
            day,
            close.index[:-2],
            close.index[1:-1],
            close.index[2:],
            values[1:-1] / values[:-2],
            opened.to_numpy()[2:] / values[:-2],
            values[2:] / opened.to_numpy()[2:],
        )


def fraction(value):
    return Fraction.from_float(float(value))


def encode(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def decode(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def affine_interval(intercepts, slopes):
    require(
        intercepts.ndim == slopes.ndim == 1
        and intercepts.shape == slopes.shape
        and intercepts.dtype.kind in "fi"
        and slopes.dtype.kind in "fi"
        and len(intercepts) > 0
        and np.isfinite(intercepts).all()
        and np.isfinite(slopes).all(),
        "Keep all finite represented affine reference inequalities.",
    )
    lower, upper = Fraction(0), Fraction(1)
    low_witness = high_witness = zero_witness = None
    for index, (a, b) in enumerate(zip(intercepts, slopes, strict=True)):
        a, b = fraction(a), fraction(b)
        if b == 0:
            if a < 0 and zero_witness is None:
                zero_witness = index
        elif b > 0 and -a / b > lower:
            lower, low_witness = -a / b, index
        elif b < 0 and -a / b < upper:
            upper, high_witness = -a / b, index
    feasible = zero_witness is None and lower <= upper
    if feasible:
        require(
            all(
                fraction(a) + fraction(b) * upper >= 0
                for a, b in zip(intercepts, slopes, strict=True)
            ),
            "The exact maximum reference fraction must satisfy every represented inequality.",
        )
    return {
        "feasible": feasible,
        "lower": encode(lower),
        "upper": encode(upper),
        "lower_witness": low_witness,
        "upper_witness": high_witness,
        "zero_slope_violation_witness": zero_witness,
        "largest_certified_nominal_fraction": float(upper) if feasible else None,
        "display_fraction_is_issued_target_or_applied_allocation": False,
    }


def coefficients(nominal, state, sample):
    state.validate()
    sample.validate()
    require(
        sample.as_of == state.as_of
        and nominal.shape == (N,)
        and nominal.dtype.kind in "fi"
        and np.isfinite(nominal).all()
        and (nominal >= 0).all()
        and abs(math.fsum(nominal) - 0.98) <= TARGET_TOLERANCE,
        "Use one unchanged fully known nominal expert composition and matching main state.",
    )
    budget = math.fsum(nominal)
    prefill_nav = state.cash + sample.prefill @ state.asset_values
    opening_gross = sample.opening @ state.asset_values
    opening_nav = opening_gross + state.cash
    allowance = 0.002 * (opening_gross + budget * opening_nav) + N
    lower_nav = opening_nav - allowance
    slack = ORDER_NOTIONAL_THRESHOLD * abs(sample.fill_mark - 1).sum(axis=1) + MONEY_TOLERANCE
    constant_mark = (1 - budget) + budget * sample.fill_mark[:, BIL]
    slope_mark = ((sample.fill_mark - sample.fill_mark[:, BIL, None]) * nominal).sum(axis=1)
    intercepts = lower_nav * constant_mark - slack - 0.85 * np.maximum(state.peak, prefill_nav)
    slopes = lower_nav * slope_mark
    require(
        all(
            np.isfinite(row).all()
            for row in (intercepts, slopes, prefill_nav, lower_nav, allowance)
        ),
        "Nonfinite reference wealth cannot become a feasible risk interval.",
    )
    return intercepts, slopes, prefill_nav, lower_nav, allowance


def resolve(nominal, state, sample):
    a, b, prefill, after_cost, allowance = coefficients(nominal, state, sample)
    witness = int(np.argmin(prefill - 0.85 * state.peak))
    common = {
        "as_of": str(state.as_of.date()),
        "known_historical_scenarios": len(sample.ends),
        "last_scenario_end": str(sample.ends[-1].date()),
        "minimum_prefill_headroom_USD": float((prefill - 0.85 * state.peak).min()),
        "maximum_scenario_open_cost_bound_USD": float(allowance.max()),
        "original_main_band_and_requests_reexecuted": False,
        "finite_reference_is_unknown_future_support_or_drawdown_guarantee": False,
        "new_underlying_target_or_account_generated": False,
    }
    if state.nav < 0.85 * state.peak:
        status, interval, reason = (
            "outside_at_current_mark",
            None,
            "reference_already_breached_in_saved_state",
        )
    elif prefill[witness] < 0.85 * state.peak:
        status, interval, reason = (
            "unavoidable_prefill_reference_breach",
            None,
            "a_new_delayed_fill_cannot_change_existing_waiting_holdings",
        )
    elif (after_cost <= 0).any():
        status, interval, reason = (
            "uncertified_conservative_cost_bound",
            None,
            "fee_upper_bound_exhausts_at_least_one_scenario_NAV",
        )
    else:
        interval = affine_interval(a, b)
        status = (
            "finite_interval_certified" if interval["feasible"] else "no_finite_sufficient_interval"
        )
        reason = "conditional_first_delivery_close_reference_only"
    return {
        **common,
        "status": status,
        "reason": reason,
        "interval": interval,
        "prefill_violation_witness": witness
        if status == "unavoidable_prefill_reference_breach"
        else None,
    }


def verify(nominal, state, sample, saved):
    a, b, prefill, after_cost, allowance = coefficients(nominal, state, sample)
    require(
        saved["status"]
        in {
            "outside_at_current_mark",
            "unavoidable_prefill_reference_breach",
            "uncertified_conservative_cost_bound",
            "finite_interval_certified",
            "no_finite_sufficient_interval",
        }
        and saved["as_of"] == str(state.as_of.date())
        and saved["known_historical_scenarios"] == len(sample.ends)
        and saved["original_main_band_and_requests_reexecuted"] is False
        and saved["finite_reference_is_unknown_future_support_or_drawdown_guarantee"] is False
        and saved["new_underlying_target_or_account_generated"] is False,
        "A reference result cannot change its status, clock or claim an executed policy.",
    )
    maximum_error = 0.0
    budget = math.fsum(float(v) for v in nominal)
    for index in range(len(sample.ends)):
        pre = state.cash + math.fsum(
            float(x) * float(r)
            for x, r in zip(state.asset_values, sample.prefill[index], strict=True)
        )
        gross = math.fsum(
            float(x) * float(r)
            for x, r in zip(state.asset_values, sample.opening[index], strict=True)
        )
        nav = gross + state.cash
        cost_bound = 0.002 * (gross + budget * nav) + 14
        marks = sample.fill_mark[index]
        base = (1 - budget) + budget * float(marks[BIL])
        slope = math.fsum(
            float(w) * (float(r) - float(marks[BIL])) for w, r in zip(nominal, marks, strict=True)
        )
        slack = (
            ORDER_NOTIONAL_THRESHOLD * math.fsum(abs(float(v) - 1) for v in marks) + MONEY_TOLERANCE
        )
        intercept = (nav - cost_bound) * base - slack - 0.85 * max(state.peak, pre)
        errors = (
            abs(intercept - float(a[index])),
            abs((nav - cost_bound) * slope - float(b[index])),
            abs(pre - float(prefill[index])),
            abs(cost_bound - float(allowance[index])),
        )
        maximum_error = max(maximum_error, *errors)
    require(maximum_error <= MONEY_TOLERANCE, "Independent scalar scenario construction disagrees.")
    early = (
        "outside_at_current_mark"
        if state.nav < 0.85 * state.peak
        else "unavoidable_prefill_reference_breach"
        if (prefill < 0.85 * state.peak).any()
        else "uncertified_conservative_cost_bound"
        if (after_cost <= 0).any()
        else None
    )
    require(
        saved["status"] == early if early is not None else saved["interval"] is not None,
        "The reference classification cannot reorder existing-state,prefill and cost failures.",
    )
    if saved["status"] == "outside_at_current_mark":
        require(
            state.nav < 0.85 * state.peak and saved["interval"] is None,
            "The saved current-mark failure must be real and have no selected fraction.",
        )
    elif saved["status"] == "unavoidable_prefill_reference_breach":
        index = saved["prefill_violation_witness"]
        require(
            type(index) is int
            and 0 <= index < len(prefill)
            and prefill[index] < 0.85 * state.peak
            and saved["interval"] is None,
            "A prefill contradiction needs an actual held-position scenario witness.",
        )
    elif saved["status"] == "uncertified_conservative_cost_bound":
        require(
            (after_cost <= 0).any() and saved["interval"] is None,
            "A conservative fee-bound failure cannot be hidden as funded feasibility.",
        )
    else:
        certificate = saved["interval"]
        lower = [(Fraction(0), None)] + [
            (-fraction(x) / fraction(y), i)
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            if y > 0
        ]
        upper = [(Fraction(1), None)] + [
            (-fraction(x) / fraction(y), i)
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            if y < 0
        ]
        lo, hi = max(v for v, _ in lower), min(v for v, _ in upper)
        zero = [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if y == 0 and x < 0]
        feasible = not zero and lo <= hi
        require(
            decode(certificate["lower"]) == lo
            and decode(certificate["upper"]) == hi
            and certificate["feasible"] is feasible
            and (saved["status"] == "finite_interval_certified") is feasible,
            "Independent exact endpoint intersection or feasibility differs.",
        )
        for field, candidates, expected in (
            ("lower_witness", lower, lo),
            ("upper_witness", upper, hi),
        ):
            witness = certificate[field]
            require(
                any(value == expected and index == witness for value, index in candidates),
                "A binding reference endpoint needs a valid exact witness;ties are not errors.",
            )
        require(
            certificate["zero_slope_violation_witness"] in zero
            if zero
            else certificate["zero_slope_violation_witness"] is None,
            "Preserve the constant-constraint contradiction witness.",
        )
        require(
            (certificate["largest_certified_nominal_fraction"] is None) is (not feasible),
            "An empty reference interval cannot select a success-shaped fallback fraction.",
        )
        if feasible:
            display = certificate["largest_certified_nominal_fraction"]
            require(
                math.isfinite(display)
                and abs(fraction(display) - hi) <= fraction(1e-10)
                and certificate["display_fraction_is_issued_target_or_applied_allocation"] is False,
                "The display approximation cannot change the exact endpoint or become an order.",
            )
    return {
        "maximum_independent_scalar_money_error_USD": maximum_error,
        "exact_represented_interval_and_witness_verified": True,
        "future_or_full_month_guarantee_verified": False,
    }
