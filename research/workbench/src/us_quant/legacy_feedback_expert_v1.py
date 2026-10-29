from __future__ import annotations

import numpy as np
import pandas as pd

from us_quant.calendar import next_session, previous_session, sessions
from us_quant.config import QuantError
from us_quant.semiconductor_portfolio_protection import protected_target


def feedback_observations(
    monthly: pd.DataFrame,
    shadow: pd.DataFrame,
    weights: pd.DataFrame,
    issued: pd.DataFrame,
    states: list[dict],
    delay: int,
) -> tuple[pd.DataFrame, dict]:
    """Derive known recommendations from saved own-book states, without running a book."""
    if (
        type(delay) is not int
        or delay not in (1, 2)
        or shadow.empty
        or not shadow.index.equals(weights.index)
        or not shadow.index.equals(sessions(shadow.index[0], shadow.index[-1]))
        or not monthly.index.equals(issued.index)
        or not monthly.index.equals(sessions(monthly.index[0], monthly.index[-1]))
        or not monthly.index.is_unique
        or not monthly.columns.is_unique
        or not monthly.columns.equals(issued.columns)
        or not monthly.columns.equals(weights.columns)
        or not shadow.index.isin(monthly.index).all()
        or "BIL" not in monthly
        or len(states) != len(shadow)
        or [s["session"] for s in states] != shadow.index.strftime("%Y-%m-%d").tolist()
        or not np.isfinite(shadow["equity"]).all()
        or (shadow["equity"] <= 0).any()
        or not np.isfinite(weights).all().all()
        or (weights < 0).any().any()
        or (weights.sum(axis=1) > 1 + 1e-12).any()
    ):
        raise QuantError("Feedback observations require complete aligned saved states and clocks.")
    for frame in (monthly, issued):
        active = frame.dropna(how="all")
        if (
            active.empty
            or (frame.isna().any(axis=1) & ~frame.isna().all(axis=1)).any()
            or not np.isfinite(active).all().all()
            or (active < 0).any().any()
            or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
        ):
            raise QuantError("Saved baseline and issued intents must be complete funded targets.")
    anchor = previous_session(shadow.index[0])
    known = monthly.loc[:anchor].dropna(how="all")
    if anchor not in monthly.index or known.empty:
        raise QuantError("A real known baseline must precede the initial shadow-capital decision.")
    prefix_issued = issued.loc[: shadow.index[-1]].dropna(how="all")
    if prefix_issued.index[0] != anchor:
        raise QuantError("Do not invent an expert recommendation before its actual initial anchor.")
    current = known.iloc[-1].copy()
    peak = 10000.0
    requested = False
    pending: dict[pd.Timestamp, bool] = {}
    dense = pd.DataFrame(np.nan, index=monthly.index, columns=monthly.columns)
    maximum_target_error = maximum_state_error = 0.0
    request_count = 0

    def check_request(day: pd.Timestamp, desired: pd.Series) -> None:
        nonlocal maximum_target_error, request_count
        if pending or issued.loc[day].isna().any():
            raise QuantError("An actual request cannot be missing, overlapped or overwritten.")
        error = float(abs(issued.loc[day] - desired).max())
        if error > 1e-12:
            raise QuantError("The archived request differs from the original known protection law.")
        maximum_target_error = max(maximum_target_error, error)
        execution = day
        for _ in range(delay):
            execution = next_session(execution)
        pending[execution] = True
        request_count += 1

    dense.loc[anchor] = protected_target(current, 10000.0, peak)
    check_request(anchor, dense.loc[anchor])
    for day, state in zip(shadow.index, states, strict=True):
        pending.pop(day, None)
        nav = float(shadow.loc[day, "equity"])
        peak = max(peak, nav)
        if not monthly.loc[day].isna().all():
            current = monthly.loc[day].copy()
            requested = True
        desired = protected_target(current, nav, peak)
        dense.loc[day] = desired
        scale = float(np.clip((nav - 0.85 * peak) / (0.15 * nav), 0, 1))
        observed = (
            (state["nav_usd"], nav),
            (state["peak_usd"], peak),
            (state["floor_usd"], 0.85 * peak),
            (state["scale"], scale),
            (state["source_bil_weight"], float(current["BIL"])),
            (state["desired_bil_weight"], float(desired["BIL"])),
        )
        for actual, expected in observed:
            error = abs(float(actual) - expected)
            maximum_state_error = max(maximum_state_error, error)
            if not np.isfinite(actual) or error > 1e-9:
                raise QuantError("A stored feedback state is not its own known account prefix.")
        should_request = not pending and (
            requested or float(abs(desired - weights.loc[day]).max()) >= 0.05
        )
        if should_request:
            check_request(day, desired)
            requested = False
        elif not issued.loc[day].isna().all():
            raise QuantError("A hold or pending request cannot acquire a new archived intent.")
        if state["monthly_requested"] is not requested or state["pending_executions"] != [
            str(d.date()) for d in pending
        ]:
            raise QuantError(
                "Preserve queued month-end updates and every original delayed request."
            )
    if request_count != len(prefix_issued):
        raise QuantError("Every stored request must occur on its actual observed decision clock.")
    active = dense.dropna(how="all")
    expected_index = monthly.loc[anchor : shadow.index[-1]].index
    if (
        not active.index.equals(expected_index)
        or not np.isfinite(active).all().all()
        or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
    ):
        raise QuantError(
            "Known daily recommendations cannot omit formal sessions or invent capital."
        )
    return dense, {
        "first_known_recommendation": str(anchor.date()),
        "last_known_recommendation": str(shadow.index[-1].date()),
        "known_recommendation_rows": len(active),
        "real_shadow_return_observations": len(shadow),
        "archived_requests_checked": request_count,
        "maximum_issued_target_difference": maximum_target_error,
        "maximum_known_state_difference": maximum_state_error,
        "source_native_delay_sessions": delay,
        "native_feedback_uses_own_shadow_wealth_not_main_wealth": True,
        "unissued_daily_recommendations_are_not_native_executions": True,
        "shadow_accounts_replayed": False,
        "main_drawdown_guarantee": False,
        "independent_forward_validation": False,
    }
