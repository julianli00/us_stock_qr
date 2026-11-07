from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import next_session
from us_quant.cash_funded_accounting_v2 import ORDER_NOTIONAL_THRESHOLD
from us_quant.config import QuantError
from us_quant.retained_pair_equal_allocation_v1 import ASSETS
from us_quant.saved_inclusion_execution_v1 import MONEY_TOLERANCE, TARGET_TOLERANCE

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/shield-ticket-anatomy-v1.json"
PATHS = ("10y-base", "10y-stress", "5y-base", "5y-stress")
N = len(ASSETS)
BUCKETS = ("below_1", "1_to_10", "10_to_100", "100_plus")
GROUPS = (
    "initial_funded_request",
    "same_issued_target_rebalance",
    "same_nominal_changed_fraction_target",
    "changed_nominal_and_target",
)


def require(condition, message):
    if not condition:
        raise QuantError(message)


def validate(policy):
    expected = {
        "schema_version": 1,
        "study_id": "ssd_shield_ticket_anatomy_20261107_v1",
        "actual_native_event_seq": 197248,
        "prior_journal_sequence": 1506,
        "prior_combined_completed": 304,
        "source_inputs": {
            "path": "reports/ssd-continuous-research-v1/shield-ticket-anatomy-20261107-v1/"
            "source-inputs-v1.json",
            "sha256": "d67bc2788ecea37516ba7bae3db498bbca7c0dd64b322e108085f7c3d5da7872",
        },
        "original_formal_rows": {
            "10y-base": 2512,
            "10y-stress": 2512,
            "5y-base": 1254,
            "5y-stress": 1254,
        },
        "original_model_tickets": {
            "10y-base": 2795,
            "10y-stress": 2395,
            "5y-base": 2130,
            "5y-stress": 1844,
        },
        "ticket_notional_threshold_USD": ORDER_NOTIONAL_THRESHOLD,
        "fixed_model_commission_USD": 1.0,
        "notional_bucket_boundaries_USD": [1.0, 10.0, 100.0],
        "bucket_lower_endpoint_included": True,
        "money_coherence_USD": MONEY_TOLERANCE,
        "target_and_return_coherence": TARGET_TOLERANCE,
        "source_RF_exact_original_review_market": True,
        "prefix_positions": [1, 21, 126, "half", "full"],
        "provenance_compared_to_previous": "issued_instruction_not_unissued_review",
        "fee_base_crossing": "prefee_gap_at_most_original_noop_but_executed_ticket_above",
        "fee_base_component": "minus_known_issued_weight_times_actual_paid_fee",
        "boundary_or_aggregate_tolerance_relaxed": False,
        "strategy_account_solver_BT_or_old_diagnosis_replayed": False,
        "source_CAGR_Sharpe_drawdown_or_qualification_recomputed": False,
        "fee_free_or_fee_removed_policy_counterfactual_claimed": False,
        "causal_fee_or_expert_alpha_attribution_claimed": False,
        "new_financial_trials": 0,
        "new_source_requests": 0,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    require(
        policy == expected and all(type(policy[k]) is type(v) for k, v in expected.items()),
        "Freeze saved ticket arithmetic,original thresholds,coverage and noncausal interpretation.",
    )


def vector(value, *, target=False):
    row = np.asarray(value)
    require(
        row.shape == (N,)
        and row.dtype.kind in "fiu"
        and np.isfinite(row).all()
        and (row >= 0).all(),
        "Use finite nonnegative saved numeric asset coordinates,not coercion or replacement.",
    )
    if target:
        require(abs(math.fsum(row) - 0.98) <= TARGET_TOLERANCE, "Keep saved98%targets.")
    return row.astype(float, copy=False)


@dataclass(frozen=True)
class SavedIntent:
    signal: pd.Timestamp
    execution: pd.Timestamp
    source_known_at: pd.Timestamp
    target: np.ndarray
    nominal: np.ndarray
    fraction: float
    previous_target: np.ndarray | None
    previous_nominal: np.ndarray | None
    previous_fraction: float | None

    def label(self, delay):
        require(
            all(
                isinstance(d, pd.Timestamp) and d.tzinfo is None
                for d in (self.signal, self.execution, self.source_known_at)
            )
            and self.source_known_at <= self.signal < self.execution
            and type(delay) is int
            and delay in (1, 2)
            and not isinstance(self.fraction, bool | np.bool_)
            and math.isfinite(self.fraction)
            and 0 <= self.fraction <= 1,
            "Keep source,value,information and original delayed execution clocks separate.",
        )
        execution = self.signal
        for _ in range(delay):
            execution = next_session(execution)
        require(execution == self.execution, "An actual fill cannot be retimed.")
        target, nominal = vector(self.target, target=True), vector(self.nominal, target=True)
        reconstructed = nominal * self.fraction
        bil = ASSETS.index("BIL")
        reconstructed[bil] = 0.98 - math.fsum(reconstructed[j] for j in range(N) if j != bil)
        require(
            np.max(abs(reconstructed - target)) <= TARGET_TOLERANCE,
            "Verify the existing saved target algebra;do not derive a replacement policy.",
        )
        if self.previous_target is None:
            require(
                self.previous_nominal is None and self.previous_fraction is None,
                "Only real initialization lacks a previous issued instruction.",
            )
            return GROUPS[0]
        previous_target = vector(self.previous_target, target=True)
        previous_nominal = vector(self.previous_nominal, target=True)
        require(
            self.previous_fraction is not None
            and math.isfinite(self.previous_fraction)
            and 0 <= self.previous_fraction <= 1,
            "A carried prior instruction needs its actual saved risk fraction.",
        )
        if np.array_equal(target, previous_target):
            return GROUPS[1]
        if np.array_equal(nominal, previous_nominal):
            require(self.fraction != self.previous_fraction, "Changed target needs changed input.")
            return GROUPS[2]
        return GROUPS[3]


@dataclass(frozen=True)
class SavedStep:
    day: pd.Timestamp
    opening: np.ndarray
    closing: np.ndarray
    previous_units: np.ndarray
    units: np.ndarray
    previous_cash: float
    cash: float
    previous_nav: float
    nav: float
    recorded_return: float
    paid_cost: float
    turnover: float
    model_tickets: int
    cost_bps: float
    delay: int
    intent: SavedIntent | None


def bucket(notional):
    require(
        math.isfinite(notional) and notional > ORDER_NOTIONAL_THRESHOLD,
        "Only an original strict-threshold model ticket enters notional buckets.",
    )
    return (
        BUCKETS[0]
        if notional < 1
        else BUCKETS[1]
        if notional < 10
        else (BUCKETS[2] if notional < 100 else BUCKETS[3])
    )


def inspect(step: SavedStep):
    require(
        isinstance(step.day, pd.Timestamp)
        and step.day.tzinfo is None
        and type(step.delay) is int
        and (step.cost_bps, step.delay) in ((5.0, 1), (20.0, 2))
        and type(step.model_tickets) is int
        and 0 <= step.model_tickets <= N
        and all(
            math.isfinite(v)
            for v in (
                step.previous_cash,
                step.cash,
                step.previous_nav,
                step.nav,
                step.recorded_return,
                step.paid_cost,
                step.turnover,
            )
        )
        and min(step.previous_cash, step.cash, step.paid_cost, step.turnover) >= 0
        and min(step.previous_nav, step.nav) > 0,
        "Use a complete original saved cash-funded step,including explicit zero-fill days.",
    )
    opening, closing = vector(step.opening), vector(step.closing)
    before, after = vector(step.previous_units), vector(step.units)
    require((opening > 0).all() and (closing > 0).all(), "Admitted prices must be positive.")
    old_values, new_values = before * opening, after * opening
    pre_nav = float(old_values.sum() + step.previous_cash)
    post_nav = float(new_values.sum() + step.cash)
    signed = (after - before) * opening
    notionals = abs(signed)
    active = notionals > ORDER_NOTIONAL_THRESHOLD
    total = math.fsum(notionals)
    money_errors = {
        "opening_wealth": abs(pre_nav - post_nav - step.paid_cost),
        "cash_flow": abs(
            math.fsum((step.previous_cash, -step.cash, -step.paid_cost, -math.fsum(signed)))
        ),
        "closing_wealth": abs(math.fsum(after * closing) + step.cash - step.nav),
        "paid_fees": abs(total * step.cost_bps / 10000 + step.model_tickets - step.paid_cost),
        "turnover_dollars": abs(step.turnover * pre_nav - total),
    }
    return_error = abs(step.nav / step.previous_nav - 1 - step.recorded_return)
    require(
        pre_nav > 0
        and post_nav > 0
        and int(active.sum()) == step.model_tickets
        and max(money_errors.values()) <= MONEY_TOLERANCE
        and return_error <= TARGET_TOLERANCE,
        "Saved units,strict ticket count,cash,fee,turnover or return identities disagree.",
    )
    group, prefee, fee_base, residual = "no_scheduled_fill", None, None, None
    if step.intent is None:
        require(
            np.array_equal(before, after)
            and step.previous_cash == step.cash
            and step.model_tickets == 0
            and step.paid_cost == step.turnover == 0,
            "A hidden trade without a causally issued instruction cannot be ignored.",
        )
    else:
        require(step.intent.execution == step.day, "Use the actual scheduled fill day.")
        group = step.intent.label(step.delay)
        prefee = step.intent.target * pre_nav - old_values
        fee_base = -step.intent.target * step.paid_cost
        residual = float(np.max(abs(signed - prefee - fee_base)))
        require(residual <= MONEY_TOLERANCE, "The retained fee-base dollar identity disagrees.")
    tickets = []
    for i in np.flatnonzero(active):
        tickets.append(
            {
                "asset": ASSETS[i],
                "signed_notional_USD": float(signed[i]),
                "notional_USD": float(notionals[i]),
                "bucket": bucket(float(notionals[i])),
                "fixed_commission_USD": 1.0,
                "proportional_fee_USD": float(notionals[i] * step.cost_bps / 10000),
                "prefee_target_gap_USD": float(prefee[i]),
                "actual_fee_base_component_USD": float(fee_base[i]),
                "prefee_noop_to_executed_ticket": bool(abs(prefee[i]) <= ORDER_NOTIONAL_THRESHOLD),
            }
        )
    return {
        "session": str(step.day.date()),
        "provenance": group,
        "scheduled_fill": step.intent is not None,
        "signal_session": str(step.intent.signal.date()) if step.intent is not None else None,
        "actual_model_tickets": step.model_tickets,
        "actual_paid_cost_USD": step.paid_cost,
        "actual_notional_USD": total,
        "unticketed_subthreshold_notional_USD": math.fsum(notionals[~active]),
        "maximum_money_identity_error_USD": max(money_errors.values()),
        "maximum_return_identity_error": return_error,
        "fee_base_identity_error_USD": residual,
        "tickets": tickets,
        "actual_target_or_account_generated": False,
        "fee_base_crossing_is_causal_wealth_attribution": False,
    }


def aggregate(records):
    require(bool(records), "Every formal saved book must retain all days.")
    buckets = {
        key: {"tickets": 0, "notional_USD": 0.0, "fixed_commission_USD": 0.0} for key in BUCKETS
    }
    provenance = {key: {"scheduled_fills": 0, "tickets": 0, "paid_fees_USD": 0.0} for key in GROUPS}
    fee_crossings = 0
    for row in records:
        if row["scheduled_fill"]:
            group = provenance[row["provenance"]]
            group["scheduled_fills"] += 1
            group["tickets"] += row["actual_model_tickets"]
            group["paid_fees_USD"] += row["actual_paid_cost_USD"]
        for ticket in row["tickets"]:
            cell = buckets[ticket["bucket"]]
            cell["tickets"] += 1
            cell["notional_USD"] += ticket["notional_USD"]
            cell["fixed_commission_USD"] += 1
            fee_crossings += ticket["prefee_noop_to_executed_ticket"]
    total_tickets = sum(r["actual_model_tickets"] for r in records)
    require(
        sum(r["tickets"] for r in buckets.values()) == total_tickets
        and sum(r["tickets"] for r in provenance.values()) == total_tickets,
        "Frozen buckets and request-provenance partitions must retain every actual ticket.",
    )
    return {
        "formal_rows": len(records),
        "model_tickets": total_tickets,
        "paid_fees_USD": math.fsum(r["actual_paid_cost_USD"] for r in records),
        "actual_notional_USD": math.fsum(r["actual_notional_USD"] for r in records),
        "unticketed_subthreshold_notional_USD": math.fsum(
            r["unticketed_subthreshold_notional_USD"] for r in records
        ),
        "notional_buckets": buckets,
        "filled_instruction_provenance": provenance,
        "prefee_noop_to_executed_ticket_count": int(fee_crossings),
        "maximum_money_identity_error_USD": max(
            r["maximum_money_identity_error_USD"] for r in records
        ),
        "maximum_return_identity_error": max(r["maximum_return_identity_error"] for r in records),
        "maximum_fee_base_identity_error_USD": max(
            (
                r["fee_base_identity_error_USD"]
                for r in records
                if r["fee_base_identity_error_USD"] is not None
            ),
            default=None,
        ),
        "fee_base_identity_applicable_days": sum(
            r["fee_base_identity_error_USD"] is not None for r in records
        ),
        "scheduled_zero_ticket_fills": sum(
            r["scheduled_fill"] and r["actual_model_tickets"] == 0 for r in records
        ),
        "new_financial_trials": 0,
        "causal_fee_or_expert_alpha_attribution": False,
    }
