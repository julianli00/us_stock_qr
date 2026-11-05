from __future__ import annotations

import math

import numpy as np

from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError

CORE = ("MTUM", "VLUE", "QUAL", "USMV")


def check_composition(parent, child, audits, proof, method):
    if (
        len(parent.columns) != 12
        or not parent.index.equals(child.index)
        or not parent.columns.equals(child.columns)
        or not parent.isna().equals(child.isna())
        or not set(CORE).issubset(parent.columns)
        or not parent.index.equals(sessions("2015-08-10", "2026-10-05"))
        or proof["monthly_decisions"] != 122
        or proof["original_calendar_opportunities"] != 131
        or proof["source_target_months"] != 122
        or proof["undefined_prehistory_months"] != 9
        or proof["additional_source_months_dropped"] != 0
        or proof["first_eligible_source_month"] != "2016-08-31"
        or proof["exact_noncore_and_combined_core_budget_preserved"] is not True
        or proof["hmm_refitted_or_states_recomputed"] is not False
        or proof["original_source_signal_recomputed"] is not False
        or proof["candidate_accounts_or_metrics_computed"] is not False
        or len(proof["prefix_truncations"]) != 5
    ):
        raise QuantError("Keep the full original HMM carrier, source dates and cached proofs.")
    for key in (
        "maximum_independent_target_difference",
        "maximum_independent_unrounded_score_difference",
        "maximum_log_split_identity_error",
    ):
        value = proof[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1e-12:
            raise QuantError("The original independent session score/target proof changed.")
    active = child.dropna(how="all")
    noncore = child.columns.difference(CORE)
    if (
        len(active) != 122
        or str(active.index[0].date()) != "2016-08-31"
        or not all(is_month_end(day) for day in active.index)
        or not np.isfinite(active.to_numpy()).all()
        or (active < 0).any().any()
        or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
        or not parent[noncore].equals(child[noncore])
        or len(audits) != 122
        or [a["decision_session"] for a in audits] != active.index.strftime("%Y-%m-%d").tolist()
    ):
        raise QuantError("No native warmup, undefined month or noncore exposure may be rewritten.")
    gaps = []
    for day, audit in zip(active.index, audits, strict=True):
        position = int(parent.index.get_loc(day))
        old_budget = math.fsum(parent.loc[day, list(CORE)])
        budget = math.fsum(child.loc[day, list(CORE)])
        gaps.append(abs(old_budget - budget))
        if (
            gaps[-1] > 1e-12
            or abs(audit["source_core_budget"] - old_budget) > 1e-12
            or audit["method"] != method
            or audit["first_prior_close"] != str(parent.index[position - 126].date())
            or audit["last_return_session"] != str(parent.index[position - 21].date())
            or audit["return_sessions"] != 105
            or set(audit["ranking"]) != set(CORE)
            or len(audit["ranking"]) != 4
            or audit["all_noncore_targets_preserved"] is not True
            or audit["hmm_refitted_or_states_recomputed"] is not False
            or audit["partial_session_trading_account_claimed"] is not False
        ):
            raise QuantError("Preserve recorded lagged score clocks and HMM/core-budget meaning.")
    return {
        "genuine_native_monthly_opinions": 122,
        "undefined_prehistory_months_retained": 9,
        "maximum_core_budget_conservation_difference": max(gaps),
        "all_noncore_cells_exact": True,
        "old_prefix_certificates_read": 5,
        "HMM_fits_states_session_scores_ranks_or_targets_recomputed": False,
    }


def check_book(account, weights, expected, canonical_rf):
    required = ("equity", "return", "risk_free", "cash", "gross_exposure", "cost", "orders")
    if (
        not account.index.equals(expected)
        or not weights.index.equals(expected)
        or not set(required).issubset(account.columns)
        or not np.isfinite(account.loc[:, list(required)].to_numpy()).all()
        or not np.isfinite(weights.to_numpy()).all()
        or (weights < 0).any().any()
        or (weights.sum(axis=1) > 1).any()
        or (account["cash"] < 0).any()
        or (account["equity"] <= 0).any()
        or (account["return"] <= -1).any()
        or (account["gross_exposure"] < 0).any()
        or (account["gross_exposure"] > 1).any()
        or (account["cost"] < 0).any()
        or (account["orders"] < 0).any()
    ):
        raise QuantError("Only complete finite unleveraged cash-funded saved books are admissible.")
    styles = [name for name in CORE if (weights[name] > 1e-10).any()]
    if len(styles) < 3:
        raise QuantError("Require actual original style exposure, not nominal source names.")
    native, common = account["risk_free"], canonical_rf.loc[expected]
    mismatch = native.to_numpy() != common.to_numpy()
    return {
        "native_RF_exact_common": native.equals(common),
        "native_RF_mismatched_days": int(mismatch.sum()),
        "native_RF_first_mismatch": str(expected[np.flatnonzero(mismatch)[0]].date())
        if mismatch.any()
        else None,
        "native_RF_maximum_absolute_difference": float(
            np.max(np.abs(native.to_numpy() - common.to_numpy()))
        ),
        "actual_core_styles": styles,
        "RF_rounding_normalization_or_view_applied": False,
    }
