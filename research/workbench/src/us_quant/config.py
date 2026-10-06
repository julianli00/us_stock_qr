from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path


class QuantError(Exception):
    """An actionable input, data, research, or execution failure."""


@dataclass(frozen=True)
class Candidate:
    id: str
    universe: tuple[str, ...]
    kind: str
    top_k: int
    max_weight: float
    target_volatility: float

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9_]+", self.id):
            raise QuantError(
                "Candidate IDs must contain only lowercase letters, digits, underscores."
            )
        if self.kind not in {"trend", "momentum"}:
            raise QuantError(f"Unknown strategy kind: {self.kind}")
        if not self.universe or len(set(self.universe)) != len(self.universe):
            raise QuantError(f"Invalid universe for {self.id}.")
        if type(self.top_k) is not int or not 1 <= self.top_k <= len(self.universe):
            raise QuantError(f"Invalid top_k for {self.id}.")
        for name, value in (
            ("max_weight", self.max_weight),
            ("target_volatility", self.target_volatility),
        ):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise QuantError(f"{self.id}: {name} must be finite and in (0, 1].")


@dataclass(frozen=True)
class Selection:
    training_years: int
    test_years: int
    first_test_year: int
    max_training_drawdown: float
    minimum_training_cagr: float
    ranking: tuple[str, ...]
    no_eligible_candidate: str
    final_training_start: str


@dataclass(frozen=True)
class Targets:
    cagr_strictly_above: float
    sharpe_strictly_above: float
    max_drawdown_at_most: float
    beat_primary_benchmark: bool
    minimum_holdout_sessions: int
    minimum_forward_paper_sessions: int


@dataclass(frozen=True)
class Stress:
    cost_bps_per_side: float
    extra_execution_delay_sessions: int
    bootstrap_samples: int
    bootstrap_block_sessions: int
    bootstrap_seed: int


@dataclass(frozen=True)
class ResearchConfig:
    protocol_version: int
    registered_on: str
    purpose: str
    data_start: str
    simulation_start: str
    development_end: str
    holdout_start: str
    as_of: str
    symbols: tuple[str, ...]
    risk_free_symbol: str
    primary_benchmark: str
    secondary_benchmark: str
    initial_capital: float
    cash_interest: float
    cost_bps_per_side: float
    commission_per_order: float
    execution_delay_sessions: int
    cash_reserve: float
    momentum_lookbacks: tuple[int, ...]
    trend_lookback: int
    volatility_lookback: int
    selection: Selection
    targets: Targets
    stress: Stress
    candidates: tuple[Candidate, ...]

    def __post_init__(self) -> None:
        from datetime import date

        for value in (
            self.registered_on,
            self.data_start,
            self.simulation_start,
            self.development_end,
            self.holdout_start,
            self.as_of,
            self.selection.final_training_start,
        ):
            date.fromisoformat(value)
        if not (
            self.data_start
            < self.simulation_start
            <= self.selection.final_training_start
            <= self.development_end
            < self.holdout_start
            <= self.as_of
        ):
            raise QuantError("Research dates are not chronologically consistent.")
        if self.protocol_version != 1:
            raise QuantError("Unsupported protocol version.")
        numeric_settings = (
            self.cash_reserve,
            self.cost_bps_per_side,
            self.commission_per_order,
            self.selection.max_training_drawdown,
            self.selection.minimum_training_cagr,
            self.targets.cagr_strictly_above,
            self.targets.sharpe_strictly_above,
            self.targets.max_drawdown_at_most,
            self.stress.cost_bps_per_side,
        )
        if not all(math.isfinite(value) for value in numeric_settings):
            raise QuantError("Research numeric settings must be finite.")
        integer_settings = (
            *self.momentum_lookbacks,
            self.trend_lookback,
            self.volatility_lookback,
            self.execution_delay_sessions,
            self.selection.training_years,
            self.selection.test_years,
            self.selection.first_test_year,
            self.targets.minimum_holdout_sessions,
            self.targets.minimum_forward_paper_sessions,
            self.stress.extra_execution_delay_sessions,
            self.stress.bootstrap_samples,
            self.stress.bootstrap_block_sessions,
            self.stress.bootstrap_seed,
        )
        if any(type(value) is not int for value in integer_settings):
            raise QuantError("Research windows, counts, and seed must be integers.")
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise QuantError("Research symbols must be nonempty and unique.")
        if any(not re.fullmatch(r"[A-Z]{1,6}", symbol) for symbol in self.symbols):
            raise QuantError("Only plain US ETF symbols are supported.")
        if self.risk_free_symbol != "^IRX":
            raise QuantError("This protocol implements only the explicitly disclosed ^IRX proxy.")
        if not {self.primary_benchmark, self.secondary_benchmark} <= set(self.symbols):
            raise QuantError("Benchmarks must be in the data universe.")
        if not math.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise QuantError("initial_capital must be positive.")
        if self.cash_interest != 0:
            raise QuantError("Do not invent IBKR cash interest: this implementation assumes zero.")
        if not 0 <= self.cash_reserve < 1:
            raise QuantError("cash_reserve must be in [0, 1).")
        if not 0 <= self.cost_bps_per_side < 100 or self.commission_per_order < 0:
            raise QuantError("Invalid transaction costs.")
        if self.execution_delay_sessions < 1:
            raise QuantError("Same-close fills are forbidden.")
        if (
            not self.momentum_lookbacks
            or min(self.momentum_lookbacks) < 2
            or self.trend_lookback < 2
            or self.volatility_lookback < 2
        ):
            raise QuantError("Signal windows must be at least two sessions.")
        if not self.candidates or len({item.id for item in self.candidates}) != len(
            self.candidates
        ):
            raise QuantError("Candidate IDs must be unique.")
        if any(not set(item.universe) <= set(self.symbols) for item in self.candidates):
            raise QuantError("Candidate contains an unfetched symbol.")
        if self.selection.ranking != ("sharpe", "cagr", "candidate_id"):
            raise QuantError("Unsupported selection ranking.")
        if self.selection.no_eligible_candidate != "cash":
            raise QuantError("An ineligible candidate may not be silently promoted.")
        if self.selection.training_years < 1 or self.selection.test_years < 1:
            raise QuantError("Training and testing windows must be positive.")
        if not (
            date.fromisoformat(self.simulation_start).year + self.selection.training_years
            <= self.selection.first_test_year
            <= date.fromisoformat(self.development_end).year
        ):
            raise QuantError("The first walk-forward fold lacks a complete training window.")
        if not 0 < self.selection.max_training_drawdown < 1:
            raise QuantError("Invalid training drawdown limit.")
        if not 0 < self.targets.max_drawdown_at_most < 1:
            raise QuantError("Invalid acceptance drawdown limit.")
        if self.targets.minimum_holdout_sessions < 2:
            raise QuantError("The holdout must have at least two observations.")
        if self.targets.minimum_forward_paper_sessions < 1:
            raise QuantError("Forward paper evidence is required.")
        if not self.cost_bps_per_side <= self.stress.cost_bps_per_side < 100:
            raise QuantError("Stress costs must be higher than base costs and below 100 bps.")
        if self.stress.extra_execution_delay_sessions < 1:
            raise QuantError("Execution stress must add at least one session.")
        if self.stress.bootstrap_samples < 100 or self.stress.bootstrap_block_sessions < 2:
            raise QuantError("Bootstrap settings are too small.")

    def to_dict(self) -> dict:
        return asdict(self)

    def candidate(self, candidate_id: str | None) -> Candidate | None:
        if candidate_id is None:
            return None
        for item in self.candidates:
            if item.id == candidate_id:
                return item
        raise QuantError(f"Unknown frozen candidate: {candidate_id}")


def load_config(path: str | Path) -> ResearchConfig:
    try:
        raw = json.loads(Path(path).read_text())
        raw["symbols"] = tuple(raw["symbols"])
        raw["momentum_lookbacks"] = tuple(raw["momentum_lookbacks"])
        raw["selection"]["ranking"] = tuple(raw["selection"]["ranking"])
        raw["selection"] = Selection(**raw["selection"])
        raw["targets"] = Targets(**raw["targets"])
        raw["stress"] = Stress(**raw["stress"])
        raw["candidates"] = tuple(
            Candidate(**{**item, "universe": tuple(item["universe"])}) for item in raw["candidates"]
        )
        return ResearchConfig(**raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise QuantError(f"Invalid research configuration {path}: {exc}") from exc
