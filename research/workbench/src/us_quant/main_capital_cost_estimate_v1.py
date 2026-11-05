from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from us_quant import strategy_expert_pool_v2 as original
from us_quant.cash_funded_accounting_v2 import rebalance
from us_quant.config import QuantError
from us_quant.strategy_experts_v1 import ExpertObservation

BASE_COST_BPS = 5.0
COMMISSION_USD = 1.0
REFERENCE_CAPITAL_USD = 10000.0
ACTUAL_WEIGHT_BAND = 0.025
TARGET_BUDGET = 0.98
AMORTIZATION_SESSIONS = 21


@dataclass(frozen=True)
class CostEstimate:
    model: str
    capital_usd: float
    cost_usd: float
    cost_fraction: float
    turnover_fraction: float
    model_tickets: int
    reason: str

    def __post_init__(self) -> None:
        numeric = (
            self.capital_usd,
            self.cost_usd,
            self.cost_fraction,
            self.turnover_fraction,
        )
        if (
            not isinstance(self.model, str)
            or not self.model
            or not isinstance(self.reason, str)
            or not self.reason
            or any(
                isinstance(value, bool)
                or not isinstance(value, int | float | np.integer | np.floating)
                for value in numeric
            )
            or not np.isfinite(numeric).all()
            or self.capital_usd <= 0
            or min(numeric[1:]) < 0
            or self.cost_fraction >= 1
            or type(self.model_tickets) is not int
            or self.model_tickets < 0
            or not math.isclose(
                self.cost_usd,
                self.cost_fraction * self.capital_usd,
                rel_tol=1e-12,
                abs_tol=1e-10,
            )
        ):
            raise QuantError("A cost estimate must be finite, funded and explicitly identified.")


class SwitchCostEstimator(Protocol):
    def estimate(self, target: pd.Series) -> CostEstimate: ...


def _vector(values: pd.Series, *, target: bool) -> np.ndarray:
    if (
        not isinstance(values, pd.Series)
        or values.empty
        or not values.index.is_unique
        or any(not isinstance(name, str) or not name for name in values.index)
        or not pd.api.types.is_numeric_dtype(values.dtype)
        or pd.api.types.is_bool_dtype(values.dtype)
        or pd.api.types.is_complex_dtype(values.dtype)
        or values.isna().any()
    ):
        raise QuantError("Use a complete numeric Series on unique named asset coordinates.")
    array = values.to_numpy(dtype=float, copy=True)
    if (
        not np.isfinite(array).all()
        or (array < 0).any()
        or target
        and abs(float(array.sum()) - TARGET_BUDGET) > 1e-12
    ):
        raise QuantError("Keep finite nonnegative values and the original 98% target budget.")
    return array


class TargetProxyCost:
    def __init__(self, reference_target: pd.Series):
        self._reference = _vector(reference_target, target=True)
        self._assets = reference_target.index.copy(deep=True)

    def estimate(self, target: pd.Series) -> CostEstimate:
        desired = _vector(target, target=True)
        if not target.index.equals(self._assets):
            raise QuantError("A target proxy requires exactly ordered matching asset coordinates.")
        change = abs(desired - self._reference)
        fraction = float(change.sum() * 5 / 10000 + (change > 1e-12).sum() / 10000)
        return CostEstimate(
            "original_target_proxy",
            REFERENCE_CAPITAL_USD,
            fraction * REFERENCE_CAPITAL_USD,
            fraction,
            float(change.sum()),
            int((change > 1e-12).sum()),
            "nominal_target_change_not_actual_holdings",
        )


class CloseMarkedCashCost:
    def __init__(
        self, asset_values: pd.Series, cash: float, *, initial_capital: bool = False
    ):
        values = _vector(asset_values, target=False)
        if (
            isinstance(cash, bool)
            or not isinstance(cash, int | float | np.integer | np.floating)
            or not np.isfinite(cash)
            or cash < 0
            or type(initial_capital) is not bool
            or initial_capital
            and (values != 0).any()
        ):
            raise QuantError("Use known nonnegative cash; an initialization seed starts all cash.")
        self._capital = float(values.sum() + cash)
        if not math.isfinite(self._capital) or self._capital <= 0:
            raise QuantError("A close-marked cost estimate requires positive finite capital.")
        self._values = values
        self._cash = float(cash)
        self._assets = asset_values.index.copy(deep=True)
        self._initial = initial_capital

    def estimate(self, target: pd.Series) -> CostEstimate:
        desired = _vector(target, target=True)
        if not target.index.equals(self._assets):
            raise QuantError(
                "A capital estimate requires exactly ordered matching asset coordinates."
            )
        gap = float(abs(self._values / self._capital - desired).max())
        if not self._initial and gap + 1e-12 < ACTUAL_WEIGHT_BAND:
            return CostEstimate(
                "close_marked_cash_funded", self._capital, 0.0, 0.0, 0.0, 0,
                "hold_inside_original_actual_weight_band",
            )
        values, cash, cost, turnover, tickets = rebalance(
            self._values, self._cash, desired, BASE_COST_BPS, COMMISSION_USD
        )
        post_nav = float(values.sum() + cash)
        if (
            not math.isfinite(post_nav)
            or post_nav <= 0
            or not math.isclose(post_nav, self._capital - cost, rel_tol=1e-12, abs_tol=1e-10)
            or not np.allclose(values / post_nav, desired, rtol=0, atol=1e-12)
        ):
            raise QuantError("The unchanged solver cannot fund the complete close-marked target.")
        return CostEstimate(
            "close_marked_cash_funded",
            self._capital,
            cost,
            cost / self._capital,
            turnover,
            tickets,
            "initial_capital" if self._initial else "observed_weight_gap",
        )


def choose_pair_with_cost(
    observed: list[ExpertObservation],
    previous: np.ndarray,
    estimator: SwitchCostEstimator,
) -> tuple[np.ndarray, dict]:
    weights, original_assessment = original.choose_pair(observed, previous)
    assessment = deepcopy(original_assessment)
    assessment["cost_estimation_is_actual_future_execution"] = False
    if not assessment["scores_available"]:
        return weights.copy(), assessment
    excess = np.column_stack(
        [observation.net_returns.to_numpy() - observation.risk_free.to_numpy()
         for observation in observed]
    )
    usable = []
    for row in assessment["pairs"]:
        if "unscorable_reason" in row:
            continue
        allocation = np.asarray(row["expert_weights"])
        estimate = estimator.estimate(original.combine(observed, allocation))
        if not isinstance(estimate, CostEstimate):
            raise QuantError("The cost-estimation Interface must return a validated CostEstimate.")
        scores = []
        for sample in (excess[:84], excess[84:]):
            portfolio = sample @ allocation
            sigma = float(np.std(portfolio, ddof=1))
            if not math.isfinite(sigma) or sigma <= 0:
                raise QuantError("A previously scorable pair lost its unchanged native risk.")
            scores.append(
                float(math.sqrt(252) *
                      (np.mean(portfolio) - estimate.cost_fraction / AMORTIZATION_SESSIONS) /
                      sigma)
            )
        if not np.isfinite(scores).all():
            raise QuantError("A new cost estimate cannot create a nonfinite successful score.")
        row["original_target_proxy_assessment"] = {
            key: row.pop(key)
            for key in (
                "estimated_incremental_switch_cost_fraction",
                "estimated_incremental_changed_assets",
                "cost_proxy_is_actual_main_account_cost",
                "train_net_risk_score",
                "validation_net_risk_score",
                "selection_score",
                "selection_score_key",
            )
        }
        row.update(
            cost_estimate=asdict(estimate),
            cost_estimation_is_actual_future_execution=False,
            train_net_risk_score=scores[0],
            validation_net_risk_score=scores[1],
            selection_score=min(scores),
            selection_score_key=round(min(scores), 12),
        )
        usable.append(row)
    if not usable:
        raise QuantError("A scorable original catalogue unexpectedly lost every candidate pair.")
    selected = sorted(usable, key=lambda row: (-row["selection_score_key"], row["pair"]))[0]
    assessment["selected_pair"] = selected["pair"]
    assessment["reason"] = "original_pair_catalogue_with_explicit_cost_estimator"
    return np.asarray(selected["expert_weights"]), assessment
