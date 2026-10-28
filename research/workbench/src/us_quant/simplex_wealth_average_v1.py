from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import daily_variance_disagreement_v1 as donor
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.data import MarketData

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/simplex-wealth-average-v1.json"
CORE, STOCKS, UNIVERSE = donor.CORE, donor.STOCKS, donor.UNIVERSE
SLEEVES = ("four_style_core", "SOXX", "GLD")
PRIOR_HEAD = "1be6c08f1f72d7a90b1e37343d5f05bee401d7a8622002ecb32135bf7badd160"
CANDIDATES = (
    {
        "id": "dollar_guarded_simplex_wealth_average_v1",
        "construction": "exact_uniform_simplex_gross_score_average",
    },
)


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "study_id": "ssd_simplex_wealth_average_20261028_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 193,
        "prior_journal_sequence": 178,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "source_candidate": "semiconductor_option_proxy_dollar_observed_band025",
        "sleeves": list(SLEEVES),
        "dirichlet_prior": [1, 1, 1],
        "learning_clock": "completed_XNYS_month_end_close_to_close",
        "learning_window_reset": False,
        "forgetting_or_temperature_parameter": None,
        "monthly_decisions": 131,
        "core_symbols": list(CORE),
        "factor_ids": list(donor.FACTOR_IDS),
        "new_economic_factor_definitions": 0,
        "cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "candidate_count_in_this_study": 1,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
    }
    if any(
        key not in policy or policy[key] != value or type(policy[key]) is not type(value)
        for key, value in expected.items()
    ) or any(
        policy.get("methodology", {}).get(key) is not False
        for key in (
            "gross_scores_are_funded_net_performance",
            "net_wealth_identity_or_theorem_claimed",
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("Keep the fixed continuous-prior learner and unchanged acceptance scope.")


class SimplexMoments:
    def __init__(self) -> None:
        self.degree = 0
        self.mass = np.ones((1, 1))

    def weights(self) -> np.ndarray:
        first, second = np.indices(self.mass.shape)
        third = self.degree - first - second
        result = np.array(
            [
                float(np.sum(self.mass * (count + 1))) / (self.degree + 3)
                for count in (first, second, third)
            ]
        )
        if (
            not np.isfinite(result).all()
            or (result <= 0).any()
            or not np.isclose(result.sum(), 1, rtol=0, atol=1e-12)
        ):
            raise QuantError("The integrated simplex allocation is invalid.")
        return result

    def update(self, relative: np.ndarray) -> None:
        relative = np.asarray(relative, dtype=float)
        if relative.shape != (3,) or not np.isfinite(relative).all() or (relative <= 0).any():
            raise QuantError("Require three actual positive completed-period gross relatives.")
        first, second = np.indices(self.mass.shape)
        third = self.degree - first - second
        new = np.zeros((self.degree + 2, self.degree + 2))
        # Integrated monomial masses update by E[b^(k+e_j)]/E[b^k]=(k_j+1)/(n+3).
        weighted = self.mass / (self.degree + 3)
        new[1:, :-1] += weighted * (first + 1) * relative[0]
        new[:-1, 1:] += weighted * (second + 1) * relative[1]
        new[:-1, :-1] += weighted * np.maximum(third + 1, 0) * relative[2]
        total = float(new.sum())
        if not np.isfinite(new).all() or (new < 0).any() or not np.isfinite(total) or total <= 0:
            raise QuantError("Invalid moment normalization cannot become a neutral allocation.")
        self.mass = new / total
        self.degree += 1
        self.weights()


def independent_weights(relatives: np.ndarray) -> np.ndarray:
    values = np.asarray(relatives, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or not np.isfinite(values).all()
        or (values <= 0).any()
    ):
        raise QuantError("Independent integration requires complete positive gross observations.")
    order = (len(values) + 4) // 2
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes, weights = (nodes + 1) / 2, weights / 2
    u, v = np.meshgrid(nodes, nodes, indexing="ij")
    points = np.column_stack((u.ravel(), ((1 - u) * v).ravel(), ((1 - u) * (1 - v)).ravel()))
    prior = (2 * (1 - u) * weights[:, None] * weights[None, :]).ravel()
    log_score = np.zeros(len(points))
    for relative in values:
        log_score += np.log(points @ relative)
    scaled = prior * np.exp(log_score - log_score.max())
    result = scaled @ points / scaled.sum()
    if not np.isfinite(result).all() or (result <= 0).any():
        raise QuantError("Independent simplex integration failed.")
    return result


def month_relatives(close: pd.DataFrame) -> pd.DataFrame:
    dates = [day for day in close.index if is_month_end(day)]
    if len(dates) < 2:
        raise QuantError("At least two actual completed month ends are required.")
    monthly = close.loc[dates, [*CORE, "SOXX", "GLD"]]
    ratios = monthly / monthly.shift(1)
    rows = pd.DataFrame(
        {
            "four_style_core": ratios[list(CORE)].mean(axis=1, skipna=False),
            "SOXX": ratios["SOXX"],
            "GLD": ratios["GLD"],
        }
    ).iloc[1:]
    if not np.isfinite(rows).all().all() or (rows <= 0).any().any():
        raise QuantError("A missing monthly price cannot be filled or omitted.")
    return rows


def source_budget(row: pd.Series) -> float:
    if (
        tuple(row.index) != UNIVERSE
        or not np.isfinite(row).all()
        or (row < 0).any()
        or not np.isclose(row.sum(), 0.98, rtol=0, atol=1e-12)
        or (row.loc[list(donor.UNUSED)].abs() > 1e-12).any()
    ):
        raise QuantError("Require the original complete guarded donor allocation.")
    budget = float(row.loc[[*STOCKS, "GLD"]].sum())
    if not any(
        np.isclose(budget, value, rtol=0, atol=1e-12) for value in (0.49, 0.98)
    ) or not np.isclose(row["BIL"], 0.98 - budget, rtol=0, atol=1e-12):
        raise QuantError("Do not change the original donor dollar-guard budget.")
    return budget


def allocate(row: pd.Series, fractions: np.ndarray) -> pd.Series:
    if (
        fractions.shape != (3,)
        or not np.isfinite(fractions).all()
        or (fractions <= 0).any()
        or not np.isclose(fractions.sum(), 1, rtol=0, atol=1e-12)
    ):
        raise QuantError("All three integrated sleeve fractions must be positive and sum to one.")
    budget = source_budget(row)
    target = row.copy()
    target.loc[list(CORE)] = budget * fractions[0] / 4
    target["SOXX"], target["GLD"] = budget * fractions[1], budget * fractions[2]
    if (
        (target < 0).any()
        or target["BIL"] != row["BIL"]
        or not np.isclose(target.sum(), 0.98, rtol=0, atol=1e-12)
    ):
        raise QuantError("The learner changed funding or the saved BIL target.")
    return target


def build_targets(data: MarketData, desired: pd.DataFrame, policy: dict) -> tuple[dict, list[dict]]:
    validate(policy)
    data.validate()
    if (
        tuple(data.close.columns) != UNIVERSE
        or not data.close.index.equals(sessions(data.close.index[0], data.close.index[-1]))
        or not desired.index.equals(data.close.index)
        or not desired.columns.equals(data.close.columns)
        or (desired.isna().any(axis=1) & ~desired.isna().all(axis=1)).any()
    ):
        raise QuantError("Keep the original complete market calendar and sparse source targets.")
    active = desired.dropna(how="all")
    relatives = month_relatives(data.close)
    if (
        active.empty
        or not active.index.isin(relatives.index).all()
        or not all(is_month_end(day) for day in active.index)
    ):
        raise QuantError("Every source decision needs at least one matured monthly interval.")
    result = pd.DataFrame(np.nan, index=desired.index, columns=desired.columns)
    state, audits = SimplexMoments(), []
    for day, relative in relatives.iterrows():
        state.update(relative.to_numpy())
        if day not in active.index:
            continue
        fractions = state.weights()
        row = active.loc[day]
        result.loc[day] = allocate(row, fractions)
        audits.append(
            {
                "decision_session": str(day.date()),
                "first_complete_learning_interval_end": str(relatives.index[0].date()),
                "last_observed_score_close": str(day.date()),
                "completed_learning_periods": state.degree,
                "sleeve_fractions": dict(zip(SLEEVES, map(float, fractions), strict=True)),
                "source_non_BIL_budget_preserved": source_budget(row),
                "source_BIL_target_preserved": float(row["BIL"]),
                "algebraic_monomial_states": (state.degree + 1) * (state.degree + 2) // 2,
                "gross_score_is_funded_net_wealth": False,
                "expert_grid_or_historical_best_weight_selected": False,
            }
        )
    return {CANDIDATES[0]["id"]: result}, audits


def independent_row(close: pd.DataFrame, row: pd.Series, day: pd.Timestamp) -> pd.Series:
    months = [value for value in close.index if value <= day and is_month_end(value)]
    records = []
    for earlier, later in zip(months[:-1], months[1:], strict=True):
        core = sum(float(close.loc[later, name] / close.loc[earlier, name]) for name in CORE) / 4
        records.append(
            [
                core,
                float(close.loc[later, "SOXX"] / close.loc[earlier, "SOXX"]),
                float(close.loc[later, "GLD"] / close.loc[earlier, "GLD"]),
            ]
        )
    if not records or months[-1] != day:
        raise QuantError("Independent integration lacks completed same-clock observations.")
    fractions = independent_weights(np.asarray(records))
    budget = float(sum(row[name] for name in (*STOCKS, "GLD")))
    target = row.copy()
    for name in CORE:
        target[name] = budget * fractions[0] / 4
    target["SOXX"], target["GLD"] = budget * fractions[1], budget * fractions[2]
    return target
