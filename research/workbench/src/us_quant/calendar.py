from __future__ import annotations

from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd

from us_quant.config import QuantError


@lru_cache(maxsize=1)
def market_calendar():
    return xcals.get_calendar("XNYS", start="2000-01-01", end="2035-12-31")


def sessions(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    return market_calendar().sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))


def next_session(day: str | pd.Timestamp) -> pd.Timestamp:
    return market_calendar().next_session(pd.Timestamp(day))


def previous_session(day: str | pd.Timestamp) -> pd.Timestamp:
    return market_calendar().previous_session(pd.Timestamp(day))


def is_month_end(day: pd.Timestamp) -> bool:
    following = next_session(day)
    return (day.year, day.month) != (following.year, following.month)


def completed_session(now: pd.Timestamp | None = None) -> pd.Timestamp:
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        raise QuantError("An explicit timezone is required for market-data freshness.")
    local_date = current.tz_convert("America/New_York").date()
    cal = market_calendar()
    day = cal.date_to_session(pd.Timestamp(local_date), direction="previous")
    if current < cal.session_close(day) + pd.Timedelta(minutes=30):
        day = cal.previous_session(day)
    return day


def require_execution_window(
    signal_date: str, now: pd.Timestamp, window_minutes: int
) -> pd.Timestamp:
    if now.tzinfo is None or not 1 <= window_minutes <= 60:
        raise QuantError("Execution requires timezone-aware time and a 1-60 minute window.")
    day = pd.Timestamp(signal_date)
    if not market_calendar().is_session(day) or not is_month_end(day):
        raise QuantError("Only completed month-end signals are executable.")
    execution_day = next_session(day)
    opening = market_calendar().session_open(execution_day)
    if not opening <= now <= opening + pd.Timedelta(minutes=window_minutes):
        raise QuantError("Outside the next-session regular-market execution window.")
    return execution_day
