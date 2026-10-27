from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end, next_session
from us_quant.commitment_duration_v1 import validate_source
from us_quant.config import QuantError

POLICY = Path(__file__).resolve().parents[2] / "config/option-exit-latch-v1.json"
CANDIDATES = (
    {"id": "equity_month_end_option_exit_latch_v1", "reentry": "noninverted_completed_month_end"},
)
SOURCE_CANDIDATE = "equity_first_month_end_commitment_v1"
PRIOR_JOURNAL_HEAD = "b8bccf79142256bb4b3f7afcf4e9ec86525adead1ed2f721e1f1547232d7a22a"


def validate(policy: dict) -> None:
    required = {
        "schema_version": 1,
        "study_id": "ssd_option_exit_latch_20261027_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 184,
        "prior_new_lane_registered_candidates": 6,
        "prior_new_lane_completed_configurations": 6,
        "prior_journal_sequence": 70,
        "prior_journal_head_sha256": PRIOR_JOURNAL_HEAD,
        "source_candidate_id": SOURCE_CANDIDATE,
        "new_economic_factor_definitions": 0,
        "minimum_economic_factor_families": 3,
        "option_ratio_risk_threshold": 1.0,
        "risk_off_investment": "BIL",
        "cash_reserve": 0.02,
        "candidate_count_in_this_study": 1,
        "candidates": list(CANDIDATES),
    }
    if (
        any(policy.get(key) != value for key, value in required.items())
        or policy.get("initial_risk_off") is not True
        or policy.get("study_size_is_a_total_research_quota") is not False
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
        raise QuantError("Keep the fixed option exit/reentry law and actual prior trial history.")


def validate_inputs(source: pd.DataFrame, terms: pd.DataFrame) -> pd.Series:
    valid = validate_source(source)
    if (
        not terms.index.equals(source.index)
        or list(terms.columns) != ["VIX", "VIX3M"]
        or not np.isfinite(terms.to_numpy()).all()
        or (terms <= 0).any().any()
    ):
        raise QuantError("Require both genuine, complete, positive option-term close series.")
    changed = source.diff().abs().max(axis=1).gt(1e-12)
    if any(not is_month_end(day) for day in source.index[changed]):
        raise QuantError("The frozen source may change only on completed exchange month ends.")
    return valid


def build_targets(
    source: pd.DataFrame, terms: pd.DataFrame, policy: dict
) -> dict[str, pd.DataFrame]:
    validate(policy)
    valid = validate_inputs(source, terms)
    inverted = terms["VIX"].ge(terms["VIX3M"])
    month_ends = pd.Series([is_month_end(day) for day in source.index], index=source.index)
    events = pd.Series(np.nan, index=source.index)
    events.loc[inverted] = 1.0
    events.loc[~inverted & month_ends] = 0.0
    # This propagates explicit state events, not missing market observations.
    risk_off = events.ffill().fillna(1.0).eq(1)
    result = source.copy()
    blocked = valid & risk_off
    result.loc[blocked, :] = 0.0
    result.loc[blocked, "BIL"] = 0.98
    return {CANDIDATES[0]["id"]: result}


def independent_targets(
    source: pd.DataFrame, terms: pd.DataFrame, policy: dict
) -> dict[str, pd.DataFrame]:
    validate(policy)
    valid = validate_inputs(source, terms)
    result = np.full(source.shape, np.nan)
    bills = source.columns.get_loc("BIL")
    off = True
    for position, day in enumerate(source.index):
        near, far = terms.iloc[position].tolist()
        if near >= far:
            off = True
        elif next_session(day).month != day.month:
            off = False
        if not valid.iloc[position]:
            continue
        if off:
            row = np.zeros(len(source.columns))
            row[bills] = 0.98
        else:
            row = source.iloc[position].to_numpy(copy=True)
        result[position] = row
    return {CANDIDATES[0]["id"]: pd.DataFrame(result, index=source.index, columns=source.columns)}
