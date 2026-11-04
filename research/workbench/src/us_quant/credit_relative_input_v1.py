from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from us_quant.calendar import sessions
from us_quant.config import QuantError

SYMBOLS = ("HYG", "LQD")
LOOKBACK_SESSIONS = 63
LAG_SESSIONS = 1


@dataclass(frozen=True)
class CreditInputObservation:
    as_of_session: pd.Timestamp
    source_start_session: pd.Timestamp | None
    source_end_session: pd.Timestamp | None
    relative_total_return: float | None
    history_mature: bool
    predates_consumer_introduction: bool


class LaggedQuarterCreditInput:
    """A date-causal archival input, not a quote-arrival or strategy Interface."""

    def __init__(self, adjusted_close: pd.DataFrame, introduced_at: str):
        try:
            introduced = pd.Timestamp(introduced_at)
        except (TypeError, ValueError) as exc:
            raise QuantError("Credit consumer introduction needs a real aware timestamp.") from exc
        if pd.isna(introduced) or introduced.tzinfo is None:
            raise QuantError("Credit consumer introduction needs a real aware timestamp.")
        if (
            not isinstance(adjusted_close, pd.DataFrame)
            or not isinstance(adjusted_close.index, pd.DatetimeIndex)
            or adjusted_close.empty
            or tuple(adjusted_close.columns) != SYMBOLS
            or adjusted_close.index.tz is not None
            or not adjusted_close.index.is_unique
            or not adjusted_close.index.is_monotonic_increasing
            or not adjusted_close.index.equals(
                sessions(adjusted_close.index[0], adjusted_close.index[-1])
            )
            or any(dtype != np.dtype("float64") for dtype in adjusted_close.dtypes)
        ):
            raise QuantError("Credit inputs require their complete ordered float64 NYSE lattice.")
        values = adjusted_close.to_numpy(copy=True)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise QuantError("Credit input prices must be finite and strictly positive.")
        values.setflags(write=False)
        self._values = values
        self._index = adjusted_close.index.copy()
        self._introduced_day = (
            introduced.tz_convert("America/New_York").normalize().tz_localize(None)
        )

    def observe(self, as_of_session: pd.Timestamp) -> CreditInputObservation:
        try:
            day = pd.Timestamp(as_of_session)
        except (TypeError, ValueError) as exc:
            raise QuantError("Observe an actual covered NYSE session date.") from exc
        if pd.isna(day) or day.tzinfo is not None or day not in self._index:
            raise QuantError("Observe an actual covered NYSE session date.")
        position = int(self._index.get_loc(day))
        end = position - LAG_SESSIONS
        start = end - LOOKBACK_SESSIONS
        source_end = self._index[end] if end >= 0 else None
        if start < 0:
            return CreditInputObservation(
                day, None, source_end, None, False, day <= self._introduced_day
            )
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                gross = self._values[end] / self._values[start]
                relative = float(gross[0] / gross[1] - 1.0)
        except FloatingPointError as exc:
            raise QuantError("The fixed credit input arithmetic is not finite.") from exc
        if not np.isfinite(relative):
            raise QuantError("The fixed credit input arithmetic is not finite.")
        return CreditInputObservation(
            day,
            self._index[start],
            source_end,
            relative,
            True,
            day <= self._introduced_day,
        )
