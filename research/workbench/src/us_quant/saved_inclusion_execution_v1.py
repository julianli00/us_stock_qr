from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pandas as pd

from us_quant.config import QuantError

MONEY_TOLERANCE = 1e-6
TARGET_TOLERANCE = 1e-12


def _target(value, size):
    original = np.asarray(value)
    if original.dtype.kind not in "fiu":
        raise QuantError("Saved numeric opinions cannot be coerced from strings or booleans.")
    row = original.astype(float, copy=False)
    if (
        row.shape != (size,)
        or not np.isfinite(row).all()
        or (row < 0).any()
        or abs(math.fsum(row) - 0.98) > TARGET_TOLERANCE
    ):
        raise QuantError("Only complete finite saved 98-percent opinions are admissible.")
    return row


def _choice(audit, ids):
    weights = np.asarray(audit["rolling_expert_weights"], dtype=float)
    selection = audit["selection"]
    if (
        weights.shape != (len(ids),)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or abs(math.fsum(weights) - 1) > TARGET_TOLERANCE
        or type(selection["scores_available"]) is not bool
    ):
        raise QuantError("Keep complete recorded statistical weights and explicit score state.")
    pair = selection["selected_pair"]
    if selection["scores_available"]:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or len(set(pair)) != 2
            or not set(pair).issubset(ids)
            or {ids[i] for i, weight in enumerate(weights) if weight > 0} != set(pair)
        ):
            raise QuantError("Recorded pair names must match actual positive expert weights.")
    elif pair is not None or not np.allclose(weights, 1 / len(ids), rtol=0, atol=1e-12):
        raise QuantError("Unscored is explicit equal allocation, not an invented selected pair.")
    return weights, pair, selection["scores_available"]


def inspect_meta(old, new, identities):
    before8, after8 = old
    before9, after9 = new
    ids = tuple(identities["experts"])
    assets = after9["asset_order"]
    day = before9["as_of"]
    if (
        len(ids) != 9
        or len(set(ids)) != 9
        or len(set(assets)) != len(assets)
        or before8["record"] != before9["record"]
        or before9["record"] != "decision_input"
        or after8["record"] != after9["record"]
        or after9["record"] != "decision_resolved"
        or before8["as_of"] != day
        or before8["decision_index"] != before9["decision_index"]
        or after8["decision_index"] != after9["decision_index"]
        or after9["decision_index"] != before9["decision_index"]
        or after8["asset_order"] != assets
        or before8["observations"] != before9["observations"][:8]
        or tuple(o["id"] for o in before8["observations"]) != ids[:8]
        or tuple(o["id"] for o in before9["observations"]) != ids
        or after8["audit"]["decision_session"] != day
        or after9["audit"]["decision_session"] != day
        or any(
            o["as_of"] != day
            or o["asset_order"] != assets
            or o["target_known_at"] is None
            or pd.Timestamp(o["target_known_at"]) > pd.Timestamp(day)
            for o in before9["observations"]
        )
    ):
        raise QuantError("Compare exact common-clock original-eight source observations only.")
    old_targets, new_targets = after8["combined_targets"], after9["combined_targets"]
    if set(old_targets) != set(identities["old8"]) or set(new_targets) != set(identities["new9"]):
        raise QuantError("Keep both recorded equal/rolling rules, not a chosen outcome.")
    extra = _target(before9["observations"][-1]["target"], len(assets))
    eight = _target(old_targets[identities["old8"][0]], len(assets))
    nine = _target(new_targets[identities["new9"][0]], len(assets))
    residual = float(np.max(np.abs(nine - (8 * eight + extra) / 9)))
    scalar = [
        float((8 * Fraction.from_float(float(a)) + Fraction.from_float(float(b))) / 9)
        for a, b in zip(eight, extra, strict=True)
    ]
    scalar_gap = float(np.max(np.abs(nine - scalar)))
    if max(residual, scalar_gap) > TARGET_TOLERANCE:
        raise QuantError("Saved equal-nine must retain its affine equal-eight inclusion identity.")
    w8, pair8, scored8 = _choice(after8["audit"], ids[:8])
    w9, pair9, scored9 = _choice(after9["audit"], ids)
    roll8 = _target(old_targets[identities["old8"][1]], len(assets))
    roll9 = _target(new_targets[identities["new9"][1]], len(assets))
    cash = assets.index("BIL")
    risk_off = extra[cash] == 0.98 and np.count_nonzero(extra) == 1
    return {
        "decision_session": day,
        "decision_index": before9["decision_index"],
        "old_scored": scored8,
        "new_scored": scored9,
        "old_selected_pair": pair8,
        "new_selected_pair": pair9,
        "selected_pairs_exactly_equal": pair8 == pair9,
        "new_selected_pair_contains_extra": scored9 and ids[-1] in pair9,
        "new_extra_statistical_weight": float(w9[-1]),
        "new_extra_known_at": before9["observations"][-1]["target_known_at"],
        "new_extra_opinion_is_actual_BIL_only": bool(risk_off),
        "equal_inclusion_max_residual": residual,
        "independent_fraction_equal_inclusion_max_residual": scalar_gap,
        "rolling_saved_target_L1_distance": math.fsum(abs(roll9 - roll8)),
        "rolling_saved_targets_exactly_equal": bool(np.array_equal(roll8, roll9)),
        "old_statistical_weights": w8.tolist(),
        "new_statistical_weights": w9.tolist(),
        "same_original_eight_observations": True,
        "target_L1_is_executed_turnover": False,
    }


def _book(frame, decisions):
    if (
        not isinstance(frame.index, pd.DatetimeIndex)
        or frame.empty
        or frame.index.tz is not None
        or not frame.index.is_unique
        or not frame.index.is_monotonic_increasing
        or not {"cost", "orders"}.issubset(frame.columns)
        or not np.isfinite(frame[["cost", "orders"]].to_numpy()).all()
        or (frame[["cost", "orders"]] < 0).any().any()
        or not np.array_equal(frame["orders"], np.floor(frame["orders"]))
        or not decisions
    ):
        raise QuantError("Actual saved cost/ticket rows need one complete ordered account.")
    signals, executions, issued = [], [], set()
    for i, row in enumerate(decisions):
        signal, execution = (
            pd.Timestamp(row["signal_session"]),
            pd.Timestamp(row["execution_session"]),
        )
        if (
            pd.isna(signal)
            or pd.isna(execution)
            or signal.tzinfo is not None
            or execution.tzinfo is not None
            or execution <= signal
            or row["initial_capital"] is not (i == 0)
            or row["reason"] not in {"initial_capital", "observed_weight_gap", "hold"}
            or (row["reason"] == "initial_capital") != (i == 0)
            or not math.isfinite(row["maximum_observed_target_gap"])
            or not 0 <= row["maximum_observed_target_gap"] <= 1
            or execution < frame.index[0]
        ):
            raise QuantError(
                "Preserve actual request clocks, seed roles and original issue reasons."
            )
        signals.append(signal)
        executions.append(execution)
        if row["reason"] != "hold" and execution in frame.index:
            issued.add(execution)
    if (
        signals != sorted(set(signals))
        or executions != sorted(set(executions))
        or not set(frame.index[(frame["cost"] != 0) | (frame["orders"] != 0)]).issubset(issued)
    ):
        raise QuantError("No overlapping, hidden, unmatched or reordered actual execution.")


def compare_execution(old_frame, new_frame, old_decisions, new_decisions, ticket_usd=1.0):
    if ticket_usd != 1.0 or not old_frame.index.equals(new_frame.index):
        raise QuantError("Keep exact common original calendars and unchanged USD1 model tickets.")
    _book(old_frame, old_decisions)
    _book(new_frame, new_decisions)
    if len(old_decisions) != len(new_decisions):
        raise QuantError("No capital/month-end decision may disappear from a saved comparison.")
    rows = []
    for left, right in zip(old_decisions, new_decisions, strict=True):
        if any(
            left[k] != right[k] for k in ("signal_session", "execution_session", "initial_capital")
        ):
            raise QuantError(
                "Match actual identical source clocks, not retimed comparison windows."
            )
        execution = pd.Timestamp(left["execution_session"])
        old_issue, new_issue = left["reason"] != "hold", right["reason"] != "hold"
        group = {
            (True, True): "both_issued",
            (True, False): "only_old_issued",
            (False, True): "only_new_issued",
            (False, False): "neither_issued",
        }[old_issue, new_issue]
        result = {
            "signal_session": left["signal_session"],
            "execution_session": left["execution_session"],
            "initial_capital": left["initial_capital"],
            "issue_group": group,
            "old_reason": left["reason"],
            "new_reason": right["reason"],
            "old_maximum_observed_target_gap": left["maximum_observed_target_gap"],
            "new_maximum_observed_target_gap": right["maximum_observed_target_gap"],
        }
        if execution not in old_frame.index:
            if execution <= old_frame.index[-1]:
                raise QuantError("A missing in-window execution date cannot be censored.")
            rows.append({**result, "status": "outside_original_window_censored"})
            continue
        costs, orders = [
            [float(frame.at[execution, key]) for frame in (old_frame, new_frame)]
            for key in ("cost", "orders")
        ]
        if (
            any(cost < count * ticket_usd for cost, count in zip(costs, orders, strict=True))
            or not old_issue
            and (costs[0] != 0 or orders[0] != 0)
            or not new_issue
            and (costs[1] != 0 or orders[1] != 0)
        ):
            raise QuantError(
                "A hold has no own fill; actual fees cannot be less than model tickets."
            )
        rows.append(
            {
                **result,
                "status": "actual_execution_session_observed",
                "old_cost_USD": costs[0],
                "new_cost_USD": costs[1],
                "old_model_tickets": int(orders[0]),
                "new_model_tickets": int(orders[1]),
                "old_issued_numeric_noop": old_issue and orders[0] == 0,
                "new_issued_numeric_noop": new_issue and orders[1] == 0,
                "cost_difference_USD": costs[1] - costs[0],
                "fixed_ticket_difference_USD": (orders[1] - orders[0]) * ticket_usd,
                "proportional_fee_difference_USD": (costs[1] - orders[1] * ticket_usd)
                - (costs[0] - orders[0] * ticket_usd),
            }
        )
    observed = [r for r in rows if r["status"] == "actual_execution_session_observed"]
    for label, frame in (("old", old_frame), ("new", new_frame)):
        if abs(
            math.fsum(r[label + "_cost_USD"] for r in observed) - math.fsum(frame["cost"])
        ) > MONEY_TOLERANCE or sum(r[label + "_model_tickets"] for r in observed) != int(
            frame["orders"].sum()
        ):
            raise QuantError(
                "Preserve every actual paid fee and model ticket, including numeric noops."
            )
    return rows, {
        "matched_decisions": len(rows),
        "actual_execution_cases": len(observed),
        "censored_cases": len(rows) - len(observed),
        "old_total_cost_USD": math.fsum(old_frame["cost"]),
        "new_total_cost_USD": math.fsum(new_frame["cost"]),
        "total_cost_difference_USD": math.fsum(r["cost_difference_USD"] for r in observed),
        "fixed_ticket_difference_USD": math.fsum(
            r["fixed_ticket_difference_USD"] for r in observed
        ),
        "proportional_fee_difference_USD": math.fsum(
            r["proportional_fee_difference_USD"] for r in observed
        ),
        "old_model_tickets": int(old_frame["orders"].sum()),
        "new_model_tickets": int(new_frame["orders"].sum()),
        "group_counts": {
            group: sum(r["issue_group"] == group for r in rows)
            for group in ("both_issued", "only_old_issued", "only_new_issued", "neither_issued")
        },
        "group_cost_difference_USD": {
            group: math.fsum(
                r["cost_difference_USD"] for r in observed if r["issue_group"] == group
            )
            for group in ("both_issued", "only_old_issued", "only_new_issued", "neither_issued")
        },
        "new_issued_numeric_noops": sum(r["new_issued_numeric_noop"] for r in observed),
        "old_issued_numeric_noops": sum(r["old_issued_numeric_noop"] for r in observed),
        "counterfactual_common_capital_or_causal_attribution": False,
    }
