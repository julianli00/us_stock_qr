from __future__ import annotations

import numpy as np
import pandas as pd

from us_quant import legacy_term_risk_eligibility_v1 as original
from us_quant.calendar import is_month_end, previous_session, sessions
from us_quant.config import QuantError

CANDIDATE, GROUP, CLOCK, ASSETS, PATHS = (
    original.CANDIDATE,
    original.GROUP,
    original.CLOCK,
    original.ASSETS,
    original.PATHS,
)
check_identity = original.check_identity


def check(snapshot, bundle, frames, common_rf):
    if set(frames) != set(PATHS):
        raise QuantError("Preserve all four original cached native paths.")
    if (
        not isinstance(common_rf, pd.Series)
        or common_rf.empty
        or not isinstance(common_rf.index, pd.DatetimeIndex)
        or not common_rf.index.is_unique
        or not common_rf.index.is_monotonic_increasing
        or not common_rf.index.equals(sessions(common_rf.index[0], common_rf.index[-1]))
        or not np.isfinite(common_rf).all()
    ):
        raise QuantError("Keep the unchanged active common RF reader and source calendar.")
    rows, blockers = [], []
    native = {(p["years"], p["scenario"]): p for p in bundle["paths"]}
    reviews = {(p["years"], p["scenario"]): p for p in snapshot["review"]["paths"]}
    required = (
        "equity",
        "return",
        "cash",
        "gross_exposure",
        "turnover",
        "cost",
        "orders",
        "risk_free",
    )
    for years, scenario in PATHS:
        record, data = native[years, scenario], frames[years, scenario]
        book, independent, weights, targets = (
            data[key] for key in ("strategy", "strategy_bt", "weights", "targets")
        )
        start = "2016-10-06" if years == 10 else "2021-10-06"
        end, anchor = "2026-10-05", previous_session(start)
        expected = sessions(start, end)
        if (
            record["capital_usd"] != 10000
            or record["commission_per_order"] != 1
            or record["cost_bps"] != (5.0 if scenario == "base" else 20.0)
            or record["delay_sessions"] != (1 if scenario == "base" else 2)
            or not book.index.equals(expected)
            or not independent.index.equals(expected)
            or not weights.index.equals(expected)
            or list(weights.columns) != list(ASSETS)
            or not set(required).issubset(book.columns)
            or not np.isfinite(book.loc[:, list(required)].to_numpy()).all()
            or not np.isfinite(independent["equity"]).all()
            or not np.isfinite(weights.to_numpy()).all()
            or (weights < 0).any().any()
            or (book["equity"] <= 0).any()
            or (book["return"] <= -1).any()
            or (book["cost"] < 0).any()
            or (book["orders"] < 0).any()
            or targets.empty
            or not isinstance(targets.index, pd.DatetimeIndex)
            or not targets.index.is_unique
            or not targets.index.is_monotonic_increasing
            or not targets.index.equals(sessions(targets.index[0], targets.index[-1]))
            or list(targets.columns) != list(ASSETS)
            or not book.index.isin(targets.index).all()
            or (targets.isna().any(axis=1) & ~targets.isna().all(axis=1)).any()
            or not expected.isin(common_rf.index).all()
        ):
            raise QuantError("Cached native books or full source target calendar are incomplete.")
        active = targets.dropna(how="all")
        if (
            active.empty
            or anchor not in active.index
            or active.index[0] > anchor
            or not np.isfinite(active.to_numpy()).all()
            or (active < 0).any().any()
            or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
        ):
            raise QuantError(
                "Keep complete full-source requests and a defined funded formal anchor row."
            )
        native_first = active.index[0]
        prior_nav = book["equity"].shift(1)
        prior_nav.iloc[0] = record["capital_usd"]
        nav_gap = float(max(abs(book["equity"] / prior_nav - 1 - book["return"])))
        initial_gap = abs(
            float(book["equity"].iloc[0]) / record["capital_usd"]
            - 1
            - float(book["return"].iloc[0])
        )
        weight_gap = float(max(abs(weights.sum(axis=1) - book["gross_exposure"])))
        bt_gap = float(max(abs(book["equity"] - independent["equity"])))
        cached = reviews[years, scenario]
        if (
            cached["market_and_target_replay_passed"] is not True
            or cached["independent_bt_reexecuted"] is not True
            or cached["max_independent_equity_error_usd"] > 1e-6
        ):
            raise QuantError("The retained source technical verification was not complete.")
        funding = (
            float(book["cash"].min()) >= 0
            and float(book["gross_exposure"].max()) <= 1
            and float(book["gross_exposure"].min()) >= 0
            and nav_gap <= 1e-12
            and initial_gap <= 1e-12
            and weight_gap <= 1e-12
            and bt_gap <= 1e-6
        )
        comparable_rf = common_rf.loc[book.index]
        mismatches = book["risk_free"].to_numpy() != comparable_rf.to_numpy()
        rf_exact = not bool(mismatches.any())
        full_source_monthends = pd.DatetimeIndex(
            [day for day in targets.loc[native_first:].index if is_month_end(day)]
        )
        formal_monthends = pd.DatetimeIndex(
            [day for day in targets.loc[anchor:end].index if is_month_end(day)]
        )
        formal_decisions = pd.DatetimeIndex([anchor, *formal_monthends])
        missing = formal_decisions[~formal_decisions.isin(active.index)]
        dense = active.index.equals(targets.loc[native_first:].index)
        monthly = active.index.equals(full_source_monthends)
        for passed, reason in (
            (funding, "native_cash_funding_or_accounting"),
            (rf_exact, "native_RF_not_exact_active_common"),
            (
                dense or monthly,
                "native_event_requests_do_not_fit_existing_dense_or_monthly_Adapter",
            ),
            (not len(missing), "missing_original_meta_decision"),
        ):
            if not passed:
                blockers.append({"path": f"{years}y-{scenario}", "reason": reason})
        rows.append(
            {
                "path_key": f"{years}y-{scenario}",
                "formal_return_dates": len(book),
                "first_return_date": start,
                "last_return_date": end,
                "actual_formal_anchor": str(anchor.date()),
                "formal_anchor_row_is_defined": True,
                "formal_anchor_complete_target_budget": float(targets.loc[anchor].sum()),
                "formal_source_initial_capital_usd": record["capital_usd"],
                "formal_first_return_initial_capital_reconciliation_difference": initial_gap,
                "native_source_calendar_start": str(targets.index[0].date()),
                "native_source_calendar_end": str(targets.index[-1].date()),
                "native_first_active_request": str(native_first.date()),
                "native_requests_before_formal_anchor_preserved": int(
                    (active.index < anchor).sum()
                ),
                "native_request_rows": len(active),
                "native_full_source_month_end_request_rows": int(
                    active.index.isin(full_source_monthends).sum()
                ),
                "native_full_source_non_month_end_request_rows": int(
                    (~active.index.isin(full_source_monthends)).sum()
                ),
                "formal_meta_decisions": len(formal_decisions),
                "missing_formal_meta_decision_dates": missing.strftime("%Y-%m-%d").tolist(),
                "existing_dense_full_native_Adapter_calendar_compatible": dense,
                "existing_monthly_full_native_Adapter_calendar_compatible": monthly,
                "minimum_native_cash_usd": float(book["cash"].min()),
                "maximum_native_gross_exposure": float(book["gross_exposure"].max()),
                "maximum_book_NAV_return_difference": nav_gap,
                "maximum_stored_weight_gross_difference": weight_gap,
                "maximum_cached_independent_BT_NAV_difference_usd": bt_gap,
                "native_cash_funding_and_accounting_checks_pass": funding,
                "native_RF_exact_active_common_reader": rf_exact,
                "RF_mismatch_count": int(mismatches.sum()),
                "first_RF_mismatch_date": (
                    str(book.index[np.flatnonzero(mismatches)[0]].date()) if not rf_exact else None
                ),
                "maximum_native_common_RF_difference": float(
                    max(abs(book["risk_free"] - comparable_rf))
                ),
                "full_source_targets_cropped_filled_rebuilt_or_retimed": False,
                "formal_first_active_equals_funding_anchor_required": False,
            }
        )
    compatible = not blockers
    return {
        "candidate_id": CANDIDATE,
        "mechanism_group": GROUP,
        "four_native_cached_paths_checked": 4,
        "paths": rows,
        "status": "eligible_not_admitted" if compatible else "not_admitted_current_Interface",
        "new_time_range_contract_checked_without_source_crop": True,
        "current_expert_pipeline_eligible_without_new_Adapter_or_RF_view": compatible,
        "blockers": blockers,
        "old_V1_first_active_assumption_or_failure_rewritten": False,
        "source_policy_technically_invalid_merely_for_prehistory_or_sparse_requests": False,
        "source_pipeline_or_net_books_replayed": False,
        "native_RF_redefined_or_new_common_RF_view_created": False,
        "native_targets_filled_retimed_or_rebuilt": False,
        "new_Adapter_or_recommendation_view_created": False,
        "new_expert_admitted": False,
        "active_pool_changed": False,
        "new_financial_configuration_selected": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
    }
