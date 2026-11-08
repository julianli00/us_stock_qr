from __future__ import annotations

import math
from collections.abc import Callable
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import next_session, previous_session, sessions
from us_quant.config import QuantError
from us_quant.retained_pair_equal_allocation_v1 import ASSETS

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/rebalance-ticket-spillover-v1.json"
STUDY = "ssd_rebalance_ticket_spillover_20261108_v1"
MONEY, TOLERANCE, BAND, THRESHOLD = 1e-6, 1e-12, 0.025, 1e-6
GROUPS = ("initial", "direct_band_coordinate", "below_band_spillover", "band_boundary")
CUTOFFS = ("2017-04-06", "2018-10-05", "2020-10-06", "2022-10-05", "2024-10-07")


def require(condition, message):
    if not condition:
        raise QuantError(message)


def validate(policy):
    expected = {
        "schema_version": 1,
        "study_id": STUDY,
        "actual_native_event_seq": 213000,
        "prior_journal_sequence": 1570,
        "prior_combined_completed": 309,
        "source_inputs": {
            "path": "reports/ssd-continuous-research-v1/rebalance-ticket-spillover-20261108-v1/"
            "source-inputs-v1.json",
            "sha256": "dab42c6a6db5604b309311cb9cba5166306d98f50a53d56b98eb7f255a62a1fb",
        },
        "source_rules": ["dyadic", "adanormalhedge"],
        "full_original_book_count": 8,
        "full_original_close_count": 15064,
        "original_ticket_notional_threshold_usd": THRESHOLD,
        "original_signal_weight_band": BAND,
        "original_target_numeric_tolerance": TOLERANCE,
        "fixed_commission_usd": 1.0,
        "commission_notional_crossover_usd": {"base": 2000.0, "stress": 500.0},
        "spillover": "executed_ticket_strictly_below_original_signal_band_on_noninitial_trigger",
        "initial_requests_are_spillovers": False,
        "primary_delta": (
            "actual_closing_asset_value_times_open_over_close_minus_recorded_opening_value"
        ),
        "independent_delta": "Decimal64_exact_binary64_operands",
        "money_coherence_usd": MONEY,
        "diagnostic_prefix_cutoffs": list(CUTOFFS),
        "prefixes_before_funding": "explicit_not_applicable_not_passes",
        "source_CAGR_Sharpe_DD_recomputed": False,
        "account_solver_BT_old_diagnosis_or_learner_replayed": False,
        "fee_free_or_selective_rebalance_counterfactual_evaluated": False,
        "crossover_is_user_minimum_trade_or_turnover_cap": False,
        "causal_fee_or_expert_alpha_attribution": False,
        "new_financial_trials": 0,
        "new_source_requests": 0,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    require(
        policy == expected and all(type(policy[k]) is type(v) for k, v in expected.items()),
        "Freeze original ticket/band arithmetic and noncausal paired coverage.",
    )


def vector(values, *, positive=False):
    array = np.asarray(values)
    require(
        array.shape == (len(ASSETS),)
        and array.dtype.kind in "iuf"
        and np.isfinite(array).all()
        and (array >= 0).all()
        and not any(isinstance(v, bool | np.bool_) for v in np.asarray(values, dtype=object).flat)
        and (not positive or (array > 0).all()),
        "Use exact finite saved numeric asset coordinates without boolean or textual casting.",
    )
    return array.astype(float, copy=True)


def number(value):
    require(
        isinstance(value, int | float | np.integer | np.floating)
        and not isinstance(value, bool | np.bool_)
        and math.isfinite(value),
        "A saved amount is not a boolean, text, default or nonfinite number.",
    )
    return float(value)


def day(value):
    require(
        isinstance(value, str | pd.Timestamp) and not isinstance(value, bool | np.bool_),
        "A recorded exchange date cannot be a boolean or an implicit numeric epoch.",
    )
    result = pd.Timestamp(value)
    require(
        pd.notna(result) and result.tzinfo is None and result == result.normalize(),
        "Keep genuine date-only exchange-session coordinates.",
    )
    return result


def signal(row, delay):
    values, desired = vector(row["main_asset_values_at_decision"]), vector(row["desired_target"])
    cash, nav = number(row["main_cash_at_decision"]), number(row["main_equity_at_decision"])
    initial = row["initial_capital"]
    require(
        type(initial) is bool
        and cash >= 0
        and nav > 0
        and abs(math.fsum(values) + cash - nav) <= MONEY
        and abs(math.fsum(desired) - 0.98) <= TOLERANCE,
        "Saved decision capital and complete funded target are inconsistent.",
    )
    gaps = abs(desired - values / nav)
    maximum = float(gaps.max())
    expected = (
        "initial_capital"
        if initial
        else ("observed_weight_gap" if maximum + TOLERANCE >= BAND else "hold")
    )
    require(
        row["reason"] == expected
        and abs(number(row["maximum_observed_target_gap"]) - maximum) <= TOLERANCE
        and (not initial or ((values == 0).all() and cash == nav == 10000)),
        "The original max-coordinate holding band or initial cash condition changed.",
    )
    origin, execution = day(row["signal_session"]), day(row["execution_session"])
    expected_day = origin
    for _ in range(delay):
        expected_day = next_session(expected_day)
    require(execution == expected_day, "Never retime a recorded delayed fill.")
    return {
        "origin": origin,
        "execution": execution,
        "target": desired,
        "values": values,
        "cash": cash,
        "nav": nav,
        "gaps": gaps,
        "initial": initial,
        "reason": expected,
    }


def inspect_execution(intent, opening_input, close_row, opening, closing, cost_bps):
    before = vector(opening_input["opening_asset_values"])
    target = vector(opening_input["target"])
    close_values = vector(close_row["asset_values"])
    opening, closing = vector(opening, positive=True), vector(closing, positive=True)
    old_cash, cash = number(opening_input["cash"]), number(close_row["cash"])
    cost, turnover = number(close_row["cost"]), number(close_row["turnover"])
    tickets = close_row["model_tickets"]
    require(
        type(tickets) is int
        and 0 <= tickets <= len(ASSETS)
        and min(old_cash, cash, cost, turnover) >= 0
        and np.max(abs(target - intent["target"])) <= TOLERANCE,
        "Keep the original issued instruction, cash, cost and model-ticket count.",
    )
    post_open = close_values * (opening / closing)
    delta = post_open - before
    with localcontext() as ctx:
        ctx.prec = 64
        exact = [
            Decimal.from_float(float(v))
            * Decimal.from_float(float(o))
            / Decimal.from_float(float(c))
            - Decimal.from_float(float(b))
            for v, o, c, b in zip(close_values, opening, closing, before, strict=True)
        ]
    independent = np.array([float(v) for v in exact])
    notional, oracle_notional = float(abs(delta).sum()), math.fsum(abs(v) for v in independent)
    pre_nav, post_nav = math.fsum(before) + old_cash, math.fsum(post_open) + cash
    observed_tickets = int((abs(delta) > THRESHOLD).sum())
    errors = {
        "independent_asset_delta_usd": float(abs(delta - independent).max()),
        "independent_total_notional_usd": abs(notional - oracle_notional),
        "paid_cost_identity_usd": abs(notional * cost_bps / 10000 + tickets - cost),
        "turnover_dollar_identity_usd": abs(notional - turnover * pre_nav),
        "self_financing_opening_nav_usd": abs(post_nav - (pre_nav - cost)),
    }
    require(
        max(errors.values()) <= MONEY and observed_tickets == tickets,
        "Recorded ticket/notional/cash/cost identities or independent marking disagree.",
    )
    if tickets:
        require(
            np.max(abs(post_open / post_nav - target)) <= TOLERANCE,
            "The recorded fill did not fund its complete target.",
        )
    else:
        require(cost == turnover == 0, "A no-op cannot invent fees or turnover.")
    crossover = 10000.0 / cost_bps
    rows = []
    for index in np.flatnonzero(abs(delta) > THRESHOLD):
        amount = float(abs(delta[index]))
        gap = float(intent["gaps"][index])
        group = (
            "initial"
            if intent["initial"]
            else (
                "below_band_spillover"
                if gap < BAND - TOLERANCE
                else ("direct_band_coordinate" if gap >= BAND else "band_boundary")
            )
        )
        rows.append(
            {
                "asset": ASSETS[index],
                "signed_notional_usd": float(delta[index]),
                "notional_usd": amount,
                "signal_coordinate_gap": gap,
                "group": group,
                "fixed_commission_usd": 1.0,
                "proportional_fee_usd": amount * cost_bps / 10000,
                "fixed_fee_exceeds_proportional": bool(amount < crossover),
            }
        )
    return {
        "model_tickets": tickets,
        "actual_notional_usd": notional,
        "paid_cost_usd": cost,
        "fixed_commission_usd": float(tickets),
        "proportional_cost_usd": notional * cost_bps / 10000,
        "crossover_usd": crossover,
        "tickets": rows,
        "coherence_errors_usd": errors,
    }


class MemoryFeed:
    def __init__(self, events):
        self.events = events

    def read(self, cutoff):
        for row in self.events:
            observed = day(row["signal_session"] if row["kind"] == "decision" else row["session"])
            if observed > cutoff:
                break
            yield row


def scan(
    opening: pd.DataFrame,
    closing: pd.DataFrame,
    feed,
    start,
    end,
    cost_bps,
    delay,
    *,
    retain: Callable[[dict], None] | None = None,
    before: Callable[[], None] | None = None,
):
    require(
        (cost_bps, delay) in ((5.0, 1), (20.0, 2))
        and type(delay) is int
        and tuple(opening.columns) == tuple(closing.columns) == ASSETS
        and opening.index.equals(closing.index),
        "Keep exact original price, asset, fee and delay coordinates.",
    )
    dates = sessions(start, end)
    require(len(dates) >= 2 and dates.isin(closing.index).all(), "No missing formal close dates.")
    anchor, cutoff = previous_session(dates[0]), dates[-1]
    pending = execution = None
    previous_values, previous_cash, previous_nav = np.zeros(len(ASSETS)), 10000.0, 10000.0
    previous_day = anchor
    closes = decisions = initial = 0
    groups = {g: {"tickets": 0, "notional_usd": 0.0} for g in GROUPS}
    by_asset = {a: {"tickets": 0, "notional_usd": 0.0} for a in ASSETS}
    sums = {
        "model_tickets": 0,
        "actual_notional_usd": 0.0,
        "paid_cost_usd": 0.0,
        "fixed_commission_usd": 0.0,
        "proportional_cost_usd": 0.0,
    }
    maximum = fixed_dominant = fills = 0
    amounts = []
    for row in feed.read(cutoff):
        if before is not None:
            before()
        kind = row["kind"]
        observed = day(row["signal_session"] if kind == "decision" else row["session"])
        if kind == "decision":
            intent = signal(row, delay)
            require(
                observed == previous_day
                and intent["initial"] == (decisions == 0)
                and np.max(abs(intent["values"] - previous_values)) <= MONEY
                and abs(intent["cash"] - previous_cash) <= MONEY,
                "A recorded signal cannot use future or unrelated main capital.",
            )
            decisions += 1
            initial += int(intent["initial"])
            if intent["reason"] != "hold":
                require(pending is None, "A recorded pending request was overwritten.")
                pending = intent
            if retain is not None:
                retain(
                    {
                        "kind": "signal",
                        "session": str(observed.date()),
                        "initial": intent["initial"],
                        "reason": intent["reason"],
                        "target_active_assets": int((intent["target"] > 0).sum()),
                        "maximum_signal_gap": float(intent["gaps"].max()),
                    }
                )
        elif kind == "modeled_execution_input":
            require(
                pending is not None and execution is None and observed == pending["execution"],
                "A fill needs its original unique pending instruction and delay.",
            )
            expected_before = previous_values * (
                vector(opening.loc[observed].to_numpy(), positive=True)
                / vector(closing.loc[previous_day].to_numpy(), positive=True)
            )
            require(
                np.max(abs(vector(row["opening_asset_values"]) - expected_before)) <= MONEY
                and abs(number(row["cash"]) - previous_cash) <= MONEY,
                "Opening marking differs from the preceding actual inventory.",
            )
            execution = row
        elif kind == "main_close":
            require(closes < len(dates) and observed == dates[closes], "Keep every formal close.")
            values, cash, nav = (
                vector(row["asset_values"]),
                number(row["cash"]),
                number(row["equity"]),
            )
            require(
                cash >= 0
                and nav > 0
                and abs(math.fsum(values) + cash - nav) <= MONEY
                and abs(number(row["return"]) - (nav / previous_nav - 1)) <= TOLERANCE,
                "Saved closing wealth or return identity changed; do not recalculate a strategy.",
            )
            detail = None
            if execution is not None:
                detail = inspect_execution(
                    pending,
                    execution,
                    row,
                    opening.loc[observed].to_numpy(),
                    closing.loc[observed].to_numpy(),
                    cost_bps,
                )
                fills += 1
                maximum = max(maximum, max(detail["coherence_errors_usd"].values()))
                for name in sums:
                    sums[name] += detail[name]
                for ticket in detail["tickets"]:
                    group, asset = groups[ticket["group"]], by_asset[ticket["asset"]]
                    for cell in (group, asset):
                        cell["tickets"] += 1
                        cell["notional_usd"] += ticket["notional_usd"]
                    amounts.append(ticket["notional_usd"])
                    fixed_dominant += int(ticket["fixed_fee_exceeds_proportional"])
                pending = execution = None
            else:
                require(
                    row["model_tickets"] == 0 and row["cost"] == row["turnover"] == 0,
                    "Unissued main closes cannot acquire model tickets or fees.",
                )
                expected = previous_values * (
                    closing.loc[observed].to_numpy() / closing.loc[previous_day].to_numpy()
                )
                require(
                    np.max(abs(values - expected)) <= MONEY and abs(cash - previous_cash) <= MONEY,
                    "A held book changed without a fill.",
                )
            closes += 1
            previous_values, previous_cash, previous_nav, previous_day = values, cash, nav, observed
            if retain is not None:
                retain({"kind": "close", "session": str(observed.date()), "execution": detail})
        else:
            raise QuantError(f"Unsupported retained event kind: {kind}")
    require(
        closes == len(dates)
        and initial == 1
        and execution is None
        and sum(v["tickets"] for v in groups.values()) == sums["model_tickets"]
        and sum(v["tickets"] for v in by_asset.values()) == sums["model_tickets"],
        "Retain all closes, initial cash and complete ticket partitions.",
    )
    sorted_amounts = sorted(amounts)
    return {
        "formal_closes": closes,
        "decision_records": decisions,
        "recorded_fills": fills,
        **sums,
        "ticket_groups": groups,
        "asset_ticket_groups": by_asset,
        "fixed_fee_dominant_ticket_count": fixed_dominant,
        "commission_notional_crossover_usd": 10000.0 / cost_bps,
        "average_notional_per_ticket_usd": None
        if not amounts
        else sums["actual_notional_usd"] / len(amounts),
        "median_notional_per_ticket_usd": None if not amounts else float(np.median(sorted_amounts)),
        "maximum_coherence_error_usd": maximum,
        "pending_after_cutoff": pending is not None,
        "initial_request_is_band_spillover": False,
        "crossover_is_user_trade_cap": False,
        "new_financial_trials": 0,
    }
