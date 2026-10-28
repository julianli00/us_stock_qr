from __future__ import annotations

import numpy as np
import pandas as pd

from us_quant.commitment_duration_v1 import validate_source
from us_quant.config import QuantError
from us_quant.equity_first_regimes_v1 import ROOT
from us_quant.multifactor_stability import FACTORS
from us_quant.ssd_factor_catalog_v1 import ADDED, EXISTING

POLICY = ROOT / "config/family-core-expansion-v1.json"
FAMILIES = (*FACTORS, *ADDED)
UNIVERSE = (*EXISTING, *ADDED)
FACTOR_IDS = (
    "price_momentum",
    "value_exposure",
    "quality_exposure",
    "low_volatility_exposure",
    "size_exposure_ijr_v1",
    "net_buyback_exposure_pkw_v1",
)
CANDIDATES = (
    {
        "id": "equity_month_end_six_family_core_v1",
        "core_composition": "equal_six_registered_families",
    },
)
SOURCE_CANDIDATE = "equity_first_month_end_commitment_v1"
SOURCE_TARGET_SHA = "58d38d198b383c91f8db938c60d4967e5fe23f8dde1ef06d93157d5bff286eb4"
SOURCE_PROOF_SHA = "bc2b57bede44d4fdeaf18495ad55795982bf0463651d747c5d1d2c42712ff4f7"
CATALOG_COMMIT = "83514dd1186ff4fe612c5078a7787509919f80cb"
PRIOR_JOURNAL_HEAD = "d7944ac455326ecc8a6c6f6f859ce47ed05dad0899874bbf4d2f8c69620be771"


def validate(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "study_id": "ssd_family_core_expansion_20261028_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "source_candidate_id": SOURCE_CANDIDATE,
        "source_targets_sha256": SOURCE_TARGET_SHA,
        "source_proof_sha256": SOURCE_PROOF_SHA,
        "catalog_publication_commit": CATALOG_COMMIT,
        "prior_completed_configurations": 185,
        "prior_new_lane_registered_candidates": 7,
        "prior_new_lane_completed_configurations": 7,
        "prior_journal_sequence": 82,
        "prior_journal_head_sha256": PRIOR_JOURNAL_HEAD,
        "new_economic_factor_definitions_in_this_study": 0,
        "required_previously_registered_SSD_definitions": 2,
        "minimum_economic_factor_families": 3,
        "used_economic_factor_families": 6,
        "factor_symbols": list(FAMILIES),
        "factor_ids": list(FACTOR_IDS),
        "source_stock_share_preserved_SOXX": 0.5,
        "equal_core_family_count": 6,
        "cash_reserve": 0.02,
        "weight_band": 0.025,
        "candidate_count_in_this_study": 1,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
    }
    if any(
        policy.get(key) != value or type(policy.get(key)) is not type(value)
        for key, value in expected.items()
    ) or any(
        policy.get("methodology", {}).get(key) is not False
        for key in (
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("The fixed six-family composition, source or execution boundary changed.")


def source_mask(source: pd.DataFrame) -> pd.Series:
    valid = validate_source(source)
    if (source.loc[valid] < 0).any().any():
        raise QuantError("A negative saved source target cannot become a safe default.")
    return valid


def check_targets(source: pd.DataFrame, target: pd.DataFrame) -> None:
    valid = source_mask(source)
    unchanged = [symbol for symbol in EXISTING if symbol not in FACTORS]
    if (
        not target.index.equals(source.index)
        or tuple(target.columns) != UNIVERSE
        or not target.isna().all(axis=1).equals(~valid)
        or not target.loc[:, unchanged].equals(source.loc[:, unchanged])
    ):
        raise QuantError("Preserve the complete source clock, warmup and non-core targets.")
    defined = target.loc[valid]
    if (
        not np.isfinite(defined.to_numpy()).all()
        or (defined < 0).any().any()
        or not np.allclose(defined.sum(axis=1), 0.98, rtol=0, atol=1e-12)
        or not np.allclose(
            defined.loc[:, list(FAMILIES)].to_numpy(),
            (source.loc[valid, list(FACTORS)].sum(axis=1) / 6).to_numpy()[:, None],
            rtol=0,
            atol=1e-12,
        )
    ):
        raise QuantError("The new factor core changed its actual funded budget.")


def build_targets(source: pd.DataFrame, policy: dict) -> dict[str, pd.DataFrame]:
    validate(policy)
    source_mask(source)
    target = source.reindex(columns=UNIVERSE).copy()
    budget = source.loc[:, list(FACTORS)].sum(axis=1, min_count=len(FACTORS))
    for symbol in FAMILIES:
        target[symbol] = budget / len(FAMILIES)
    check_targets(source, target)
    return {CANDIDATES[0]["id"]: target}


def independent_targets(source: pd.DataFrame, policy: dict) -> dict[str, pd.DataFrame]:
    validate(policy)
    valid = source_mask(source)
    rows = np.full((len(source), len(UNIVERSE)), np.nan)
    for position, day in enumerate(source.index):
        if not valid.iloc[position]:
            continue
        row = source.loc[day]
        for column, symbol in enumerate(UNIVERSE):
            # The validated original composition puts equal budgets in SOXX and the core.
            rows[position, column] = row["SOXX"] / 6 if symbol in FAMILIES else row[symbol]
    target = pd.DataFrame(rows, index=source.index, columns=UNIVERSE)
    check_targets(source, target)
    return {CANDIDATES[0]["id"]: target}
