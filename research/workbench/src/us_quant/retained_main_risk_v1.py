from __future__ import annotations

import bisect
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end, previous_session, sessions
from us_quant.config import QuantError
from us_quant.storage import digest_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/retained-main-risk-v1.json"
STUDY = "ssd_retained_main_risk_20261108_v1"
ANCHOR = pd.Timestamp("2016-10-05")
TOLERANCE = 1e-12
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
EXPERTS = (
    "semiconductor_option_dollar_profit_share_confirmation",
    "four_factor_defense_diversified",
    "conditional_covariance_hedge_expert_v1",
    "equity_first_month_end_commitment_v1",
    "benchmark_opportunity_joint_tail_v1",
    "semiconductor_ratchet_protection",
)
BOOK_COLUMNS = (
    "equity",
    "return",
    "cash",
    "gross_exposure",
    "turnover",
    "cost",
    "orders",
    "risk_free",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QuantError(message)


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "study_id": STUDY,
        "actual_native_event_seq": 209368,
        "prior_journal_sequence": 1554,
        "prior_combined_completed": 308,
        "closed_financial_cycle": {
            "path": "reports/ssd-continuous-research-v1/expert-adanormalhedge-20261107-v1/"
            "cycle-closure-v1.json",
            "sha256": "2d5017e34dbb3c01ee7e0a8a81b1461fdd2831dc4296043e695b3639eeb7b46a",
        },
        "price_history_anchor": "2016-10-05",
        "covariance": "expanding_unbiased_adjusted_asset_return_sample_at_original_close",
        "primary_moments": "symmetric_Welford_scatter",
        "independent_moments": "raw_second_moment_minus_mean_outer_product",
        "minimum_known_returns": 2,
        "actual_main_risk": "actual_closing_asset_weights_transpose_covariance_weights",
        "source_contribution": (
            "probability_times_source_target_transpose_covariance_saved_mixture"
        ),
        "source_contribution_is_realized_main_PnL_attribution": False,
        "signed_hedging_contributions_clipped": False,
        "zero_variance_shares": "explicit_undefined_not_uniform",
        "insufficient_history": "explicit_unavailable_no_imputed_returns",
        "initial_drawdown_peak_usd": 10000.0,
        "drawdown_contexts": [
            "full_window",
            "first_strictly_above_15pct",
            "worst_drawdown_first_peak_and_trough",
        ],
        "numeric_tolerance": TOLERANCE,
        "diagnostic_prefix_cutoffs": [
            "2017-04-06",
            "2018-10-05",
            "2020-10-06",
            "2022-10-05",
            "2024-10-07",
        ],
        "prefixes_before_funding": "explicit_not_applicable_not_a_pass",
        "new_source_requests": 0,
        "old_learner_account_BT_or_prefix_runs": 0,
        "new_financial_trials": 0,
        "counterfactual_policy_evaluated": False,
        "financial_gates_changed": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    require(
        policy == expected and all(type(policy[k]) is type(v) for k, v in expected.items()),
        "Freeze the descriptive main-risk question, not an outcome-selected risk policy.",
    )


def _numeric(values, shape) -> np.ndarray:
    array = np.asarray(values)
    require(
        array.shape == shape
        and array.dtype.kind in "iuf"
        and np.isfinite(array).all()
        and not any(isinstance(v, bool | np.bool_) for v in np.asarray(values, dtype=object).flat),
        "Retained coordinates must be complete finite numbers, not boolean or coerced labels.",
    )
    return array.astype(float, copy=True)


@dataclass(frozen=True)
class SourcePoint:
    origin: pd.Timestamp
    probabilities: tuple[float, ...]
    targets: tuple[tuple[float, ...], ...]
    mixed_target: tuple[float, ...]
    known_at: tuple[pd.Timestamp, ...]
    complete_intervals: int

    def checked(self):
        p = _numeric(self.probabilities, (6,))
        targets = _numeric(self.targets, (6, len(ASSETS)))
        mixture = _numeric(self.mixed_target, (len(ASSETS),))
        require(
            isinstance(self.origin, pd.Timestamp)
            and pd.notna(self.origin)
            and self.origin.tzinfo is None
            and len(self.known_at) == 6
            and all(
                isinstance(day, pd.Timestamp)
                and pd.notna(day)
                and day.tzinfo is None
                and day <= self.origin
                for day in self.known_at
            )
            and type(self.complete_intervals) is int
            and self.complete_intervals >= 0
            and (p >= 0).all()
            and abs(math.fsum(p) - 1) <= TOLERANCE
            and (targets >= 0).all()
            and (mixture >= 0).all()
            and np.max(abs(targets.sum(axis=1) - 0.98)) <= TOLERANCE,
            "Keep all six admitted, then-known source recommendations and original budgets.",
        )
        scalar = np.array(
            [math.fsum(p[i] * targets[i, j] for i in range(6)) for j in range(len(ASSETS))]
        )
        error = float(max(abs(p @ targets - mixture).max(), abs(scalar - mixture).max()))
        require(error <= TOLERANCE, "The recorded mixture differs from its recorded operands.")
        return p, targets, mixture, error


class SourceFeed(Protocol):
    origins: tuple[pd.Timestamp, ...]

    def latest(self, day: pd.Timestamp) -> SourcePoint: ...


class MemoryFeed:
    def __init__(self, points: tuple[SourcePoint, ...]):
        self.points = points
        self.origins = tuple(point.origin for point in points)
        require(
            bool(points)
            and len(set(self.origins)) == len(points)
            and self.origins == tuple(sorted(self.origins)),
            "Source origins must remain complete, unique and chronological.",
        )

    def latest(self, day: pd.Timestamp) -> SourcePoint:
        position = bisect.bisect_right(self.origins, day) - 1
        require(position >= 0, "No future or neutral opinion may seed missing source history.")
        return self.points[position]


def _observed(feed: SourceFeed, day: pd.Timestamp) -> SourcePoint:
    position = bisect.bisect_right(feed.origins, day) - 1
    require(position >= 0, "Missing then-known source history is not a neutral recommendation.")
    point = feed.latest(day)
    require(
        isinstance(point, SourcePoint)
        and point.origin == feed.origins[position]
        and point.origin <= day
        and point.complete_intervals == max(0, position - 1),
        "The Adapter must return the latest then-known original source, even for equal targets.",
    )
    return point


@dataclass(frozen=True)
class PathEvidence:
    key: str
    account: pd.DataFrame
    held_weights: pd.DataFrame
    desired: pd.DataFrame
    close: pd.DataFrame


class _Moments:
    def __init__(self):
        self.n = 0
        self.mean = np.zeros(len(ASSETS))
        self.scatter = np.zeros((len(ASSETS), len(ASSETS)))
        self.raw_sum = self.mean.copy()
        self.raw_second = self.scatter.copy()

    def add(self, value):
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.scatter += np.outer(delta, delta) * ((self.n - 1) / self.n)
        self.raw_sum += value
        self.raw_second += np.outer(value, value)

    def covariance(self):
        if self.n < 2:
            return None, None
        mean = self.raw_sum / self.n
        return (
            self.scatter / (self.n - 1),
            (self.raw_second - self.n * np.outer(mean, mean)) / (self.n - 1),
        )


def _bilinear(left, covariance, right) -> float:
    return math.fsum(
        float(left[i]) * float(covariance[i, j]) * float(right[j])
        for i in range(len(ASSETS))
        for j in range(len(ASSETS))
    )


def _risks(p, targets, mixture, actual, covariance, independent):
    actual_variance = float(actual @ covariance @ actual)
    proposed_variance = float(mixture @ covariance @ mixture)
    standalone = np.einsum("ij,jk,ik->i", targets, covariance, targets)
    contributions = p * (targets @ covariance @ mixture)
    oracle = [p[i] * _bilinear(targets[i], independent, mixture) for i in range(6)]
    error = max(
        float(abs(covariance - independent).max()),
        abs(actual_variance - _bilinear(actual, independent, actual)),
        abs(proposed_variance - _bilinear(mixture, independent, mixture)),
        max(abs(float(a) - b) for a, b in zip(contributions, oracle, strict=True)),
        abs(math.fsum(contributions) - proposed_variance),
    )
    require(
        error <= TOLERANCE
        and actual_variance >= 0
        and proposed_variance >= 0
        and (standalone >= 0).all(),
        "Independent sample risk or Euler conservation differs; never clip the result.",
    )
    shares = None if proposed_variance == 0 else (contributions / proposed_variance).tolist()
    return {
        "actual_variance": actual_variance,
        "proposed_variance": proposed_variance,
        "standalone_variances": standalone.tolist(),
        "signed_source_contributions": contributions.tolist(),
        "source_contribution_shares": shares,
        "share_status": "undefined_zero_variance" if shares is None else "defined_signed",
        "independent_max_error": float(error),
    }


def audit_path(
    evidence: PathEvidence,
    feed: SourceFeed,
    *,
    retain: Callable[[dict], None] | None = None,
) -> dict:
    def record(value):
        if retain is not None:
            retain(value)

    book, held, desired = evidence.account, evidence.held_weights, evidence.desired
    record({"phase": "before_input_validation", "path": evidence.key, "rows": len(book)})
    dates = book.index
    require(
        isinstance(dates, pd.DatetimeIndex)
        and len(dates) > 0
        and dates.tz is None
        and dates.is_unique
        and dates.is_monotonic_increasing
        and dates.equals(sessions(dates[0], dates[-1])),
        "Every original funded return date must remain present and ordered.",
    )
    require(
        tuple(book.columns) == BOOK_COLUMNS
        and tuple(held.columns) == ASSETS
        and held.index.equals(dates)
        and tuple(desired.columns) == ASSETS
        and desired.index.is_unique
        and desired.index.is_monotonic_increasing,
        "Keep canonical saved account and actual held-weight coordinates.",
    )
    require(
        all(
            pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)
            for frame in (book, held, desired, evidence.close)
            for dtype in frame.dtypes
        ),
        "Boolean or textual saved financial columns cannot be silently cast.",
    )
    account = _numeric(book.to_numpy(), (len(dates), len(BOOK_COLUMNS)))
    weights = _numeric(held.to_numpy(), (len(dates), len(ASSETS)))
    require(
        (account[:, 0] > 0).all()
        and (account[:, 2:7] >= 0).all()
        and (weights >= 0).all()
        and (account[:, 3] <= 1).all()
        and np.equal(account[:, 6], np.floor(account[:, 6])).all()
        and np.max(abs(weights.sum(axis=1) - account[:, 3])) <= TOLERANCE
        and np.max(abs(account[:, 2] / account[:, 0] + weights.sum(axis=1) - 1)) <= TOLERANCE,
        "Actual main weights, cash, gross, costs or model tickets disagree.",
    )
    previous_nav = np.r_[10000.0, account[:-1, 0]]
    require(
        np.max(abs(account[:, 1] - (account[:, 0] / previous_nav - 1))) <= TOLERANCE,
        "Retained NET returns must match actual NAV and the original fresh capital.",
    )
    anchor = previous_session(dates[0])
    proposals = desired.loc[anchor : dates[-1]].dropna(how="all")
    expected = pd.DatetimeIndex([anchor, *[d for d in dates if is_month_end(d)]])
    require(proposals.index.equals(expected), "Initial and monthly saved proposals are missing.")
    _numeric(proposals.to_numpy(), (len(expected), len(ASSETS)))
    maximum_mixture_error = 0.0
    for day, proposal in proposals.iterrows():
        point = _observed(feed, day)
        _, _, mixture, error = point.checked()
        error = max(error, float(abs(proposal.to_numpy() - mixture).max()))
        require(error <= TOLERANCE, "A saved main proposal uses a different or future mixture.")
        maximum_mixture_error = max(maximum_mixture_error, error)
    prices = evidence.close.loc[ANCHOR : dates[-1]]
    require(
        tuple(prices.columns) == ASSETS and prices.index.equals(sessions(ANCHOR, dates[-1])),
        "The shared original source-clock price history is missing; do not fill or crop dates.",
    )
    values = _numeric(prices.to_numpy(), (len(prices), len(ASSETS)))
    require((values > 0).all(), "Adjusted prices must be positive; no imputed observations.")
    source_days = tuple([ANCHOR, *[d for d in prices.index[1:] if is_month_end(d)]])
    require(
        tuple(d for d in feed.origins if d <= dates[-1]) == source_days,
        "Keep every original source origin, not a conveniently carried missing month.",
    )
    returns = values[1:] / values[:-1] - 1
    require(np.isfinite(returns).all(), "Adjusted return arithmetic overflowed; do not impute it.")
    moments, cursor = _Moments(), 0
    peak, peak_day = 10000.0, anchor
    peak_record = {"date": str(anchor.date()), "initial_capital_peak": True, "equity": peak}
    worst, first_breach = None, None
    maximum_drawdown = 0.0
    cost_sum = turnover_sum = 0.0
    tickets_sum = 0
    peak_totals = (0.0, 0.0, 0)
    probability_sum = np.zeros(6)
    share_sum = np.zeros(6)
    negative_days = np.zeros(6, dtype=int)
    leaders = np.zeros(6, dtype=int)
    actual_vol_sum = source_hhi_sum = gap_sum = 0.0
    actual_hhi_sum = 0.0
    defined = shares_defined = actual_hhi_defined = 0
    risk_error = 0.0
    daily = {}
    for index, day in enumerate(dates):
        while cursor < len(returns) and prices.index[cursor + 1] <= day:
            moments.add(returns[cursor])
            cursor += 1
        covariance, independent = moments.covariance()
        point = _observed(feed, day)
        p, targets, mixture, error = point.checked()
        row, actual = account[index], weights[index]
        record(
            {
                "phase": "before_risk_validation",
                "path": evidence.key,
                "date": str(day.date()),
                "source_origin": str(point.origin.date()),
                "known_returns": moments.n,
                "actual_main_weights": actual.tolist(),
                "source_probabilities": p.tolist(),
                "primary_covariance": None if covariance is None else covariance.tolist(),
                "independent_covariance": None if independent is None else independent.tolist(),
            }
        )
        risk = (
            None
            if covariance is None
            else _risks(p, targets, mixture, actual, covariance, independent)
        )
        probability_sum += p
        leaders[int(np.argmax(p))] += 1
        source_hhi = float(p @ p)
        source_hhi_sum += source_hhi
        gap = float(abs(actual - mixture).sum())
        gap_sum += gap
        gross = float(row[3])
        actual_hhi = None if gross == 0 else float((actual / gross) @ (actual / gross))
        if actual_hhi is not None:
            actual_hhi_sum += actual_hhi
            actual_hhi_defined += 1
        cost_sum += float(row[5])
        turnover_sum += float(row[4])
        tickets_sum += int(row[6])
        if float(row[0]) > peak:
            peak, peak_day = float(row[0]), day
        drawdown = 1 - float(row[0]) / peak
        resolved = {
            "date": str(day.date()),
            "equity": float(row[0]),
            "cash": float(row[2]),
            "drawdown": drawdown,
            "gross": gross,
            "source_origin": str(point.origin.date()),
            "source_complete_intervals": point.complete_intervals,
            "known_returns": moments.n,
            "source_probability_hhi": source_hhi,
            "actual_invested_asset_hhi": actual_hhi,
            "actual_to_saved_mixture_L1": gap,
            "source_probability_leader": EXPERTS[int(np.argmax(p))],
            "risk_status": "insufficient_known_returns" if risk is None else "defined",
            "risk": risk,
            "cumulative_model_cost": cost_sum,
            "cumulative_turnover": turnover_sum,
            "cumulative_model_tickets": tickets_sum,
        }
        if day == peak_day:
            peak_record = resolved
            peak_totals = (cost_sum, turnover_sum, tickets_sum)
        if risk is not None:
            defined += 1
            actual_vol_sum += math.sqrt(252 * risk["actual_variance"])
            risk_error = max(risk_error, risk["independent_max_error"])
            if risk["source_contribution_shares"] is not None:
                shares_defined += 1
                share_sum += risk["source_contribution_shares"]
                negative_days += np.asarray(risk["source_contribution_shares"]) < 0
        if drawdown > 0.15 and first_breach is None:
            first_breach = resolved
        if drawdown > maximum_drawdown:
            maximum_drawdown = drawdown
            worst = {
                "peak": peak_record,
                "trough": resolved,
                "model_cost_since_peak": cost_sum - peak_totals[0],
                "turnover_since_peak": turnover_sum - peak_totals[1],
                "model_tickets_since_peak": tickets_sum - peak_totals[2],
            }
        maximum_mixture_error = max(maximum_mixture_error, error)
        daily[str(day.date())] = digest_json(resolved)
        record({"phase": "after_risk_validation", "path": evidence.key, "result": resolved})
    profiles = {
        expert: {
            "mean_source_probability": float(probability_sum[i] / len(dates)),
            "probability_leader_days": int(leaders[i]),
            "mean_defined_signed_variance_share": None
            if not shares_defined
            else float(share_sum[i] / shares_defined),
            "negative_variance_contribution_days": int(negative_days[i]),
        }
        for i, expert in enumerate(EXPERTS)
    }
    return {
        "summary": {
            "rows": len(dates),
            "defined_covariance_days": defined,
            "undefined_covariance_days": len(dates) - defined,
            "defined_source_variance_share_days": shares_defined,
            "last_known_return_count": moments.n,
            "mean_source_probability_hhi": source_hhi_sum / len(dates),
            "mean_actual_invested_asset_hhi": None
            if not actual_hhi_defined
            else actual_hhi_sum / actual_hhi_defined,
            "mean_actual_to_saved_mixture_L1": gap_sum / len(dates),
            "mean_defined_actual_sample_annualized_volatility": None
            if not defined
            else actual_vol_sum / defined,
            "maximum_drawdown_from_saved_NAV": maximum_drawdown,
            "source_profiles": profiles,
            "actual_model_cost_sum": cost_sum,
            "actual_model_ticket_sum": tickets_sum,
            "actual_turnover_sum": turnover_sum,
        },
        "contexts": {"first_strict_15pct_breach": first_breach, "worst_drawdown": worst},
        "daily_digests": daily,
        "maximum_independent_risk_error": risk_error,
        "maximum_recorded_mixture_error": maximum_mixture_error,
        "source_game_is_main_wealth_or_risk_bound": False,
        "new_learner_or_account_or_BT_runs": 0,
    }
