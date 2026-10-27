from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.equity_first_regimes_v1 import COMPOSITION, ROOT, UNIVERSE

POLICY = ROOT / "config/commitment-duration-v1.json"
CANDIDATES = (
    {"id": "equity_first_month_end_commitment_v1", "commitment": "completed_month_end"},
    {"id": "equity_first_21slot_commitment_v1", "commitment": "equal_21_rotating_intents"},
)
SOURCE_CANDIDATE = "equity_first_trend200_band025_v1"
SLOTS = 21
PRIOR_JOURNAL_HEAD = "29acd47dcb2ec03232925b71400533fe167d51b0af403fd1b3d748f341d73e3b"


def validate(policy: dict) -> None:
    if (
        policy.get("schema_version") != 1
        or policy.get("study_id") != "ssd_commitment_duration_20261027_v1"
        or policy.get("protocol_id") != "ssd_dual_benchmark_net_cagr_v1"
        or policy.get("source_candidate_id") != SOURCE_CANDIDATE
        or policy.get("prior_completed_configurations") != 180
        or policy.get("prior_new_lane_registered_candidates") != 2
        or policy.get("prior_new_lane_completed_configurations") != 2
        or policy.get("prior_journal_sequence") != 24
        or policy.get("prior_journal_head_sha256") != PRIOR_JOURNAL_HEAD
        or policy.get("new_economic_factor_definitions") != 0
        or policy.get("minimum_economic_factor_families") != 3
        or type(policy.get("common_source_intent_warmup")) is not int
        or policy["common_source_intent_warmup"] != SLOTS
        or policy.get("candidate_count_in_this_study") != 2
        or policy.get("study_size_is_a_total_research_quota") is not False
        or policy.get("candidates") != list(CANDIDATES)
        or any(
            policy.get("methodology", {}).get(key) is not False
            for key in (
                "independent_forward_validation",
                "investment_objective_verified",
                "order_authority",
                "automatic_live_deployment",
            )
        )
    ):
        raise QuantError("The fixed commitment rules, source or research boundary changed.")


def validate_source(source: pd.DataFrame) -> pd.Series:
    if (
        not isinstance(source, pd.DataFrame)
        or source.empty
        or not isinstance(source.index, pd.DatetimeIndex)
        or source.index.tz is not None
        or not source.index.is_unique
        or not source.index.is_monotonic_increasing
        or not source.columns.is_unique
        or set(source.columns) != UNIVERSE
        or not source.index.equals(sessions(source.index[0], source.index[-1]))
        or any(not pd.api.types.is_float_dtype(dtype) for dtype in source.dtypes)
    ):
        raise QuantError("Require the complete original-calendar, twelve-asset source lattice.")
    valid = source.notna().all(axis=1)
    if (source.isna().any(axis=1) & ~source.isna().all(axis=1)).any() or (
        (valid.cumsum() > 0) & ~valid
    ).any():
        raise QuantError("Missing source intents cannot become cash or carried safe defaults.")
    expected = np.zeros((3, len(source.columns)))
    for symbol, share in COMPOSITION.items():
        expected[0, source.columns.get_loc(symbol)] = 0.98 * share
    expected[1, source.columns.get_loc("GLD")] = 0.98
    expected[2, source.columns.get_loc("BIL")] = 0.98
    defined = source.loc[valid].to_numpy()
    if not np.isfinite(defined).all() or (
        len(defined)
        and not np.isclose(defined[:, None, :], expected, rtol=0, atol=1e-12)
        .all(axis=2)
        .any(axis=1)
        .all()
    ):
        raise QuantError("Use only the unchanged uncapped equity-first source intents.")
    return valid


def build_targets(source: pd.DataFrame, policy: dict) -> dict[str, pd.DataFrame]:
    validate(policy)
    valid = validate_source(source)
    month_ends = np.array([is_month_end(day) for day in source.index])
    monthly = source.loc[month_ends].reindex(source.index).ffill()
    monthly.loc[valid.cumsum() < SLOTS] = np.nan
    staggered = source.rolling(SLOTS, min_periods=SLOTS).mean()
    return {CANDIDATES[0]["id"]: monthly, CANDIDATES[1]["id"]: staggered}


def independent_targets(source: pd.DataFrame, policy: dict) -> dict[str, pd.DataFrame]:
    validate(policy)
    valid = validate_source(source)
    monthly = np.full(source.shape, np.nan)
    staggered = np.full(source.shape, np.nan)
    commitment = None
    slots: deque[list[float]] = deque(maxlen=SLOTS)
    for position, day in enumerate(source.index):
        if not valid.iloc[position]:
            continue
        row = source.iloc[position].tolist()
        slots.append(row)
        if is_month_end(day):
            commitment = row
        if len(slots) == SLOTS:
            if commitment is not None:
                monthly[position] = commitment
            staggered[position] = [
                sum(slot[column] for slot in slots) / SLOTS for column in range(source.shape[1])
            ]
    return {
        candidate["id"]: pd.DataFrame(values, index=source.index, columns=source.columns)
        for candidate, values in zip(CANDIDATES, (monthly, staggered), strict=True)
    }
