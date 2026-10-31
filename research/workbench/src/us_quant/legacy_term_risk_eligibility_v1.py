from __future__ import annotations

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end, previous_session, sessions
from us_quant.config import QuantError
from us_quant.storage import digest_json

CANDIDATE = "factor_gold_term_half_risk"
GROUP = "option_implied_insurance_horizon_pressure"
CLOCK = "after_16_15_ET_index_close_before_later_open_execution"
ASSETS = ("SPY", "IEF", "GLD", "BIL", "MTUM", "VLUE", "QUAL", "USMV")
PATHS = ((10, "base"), (10, "stress"), (5, "base"), (5, "stress"))


def check_identity(snapshot, registration, results, bundle, policy):
    spec, review = snapshot["specification"], snapshot["review"]
    matching = [row for row in registration["candidates"] if row["spec"]["id"] == CANDIDATE]
    if (
        snapshot["candidate_id"] != CANDIDATE
        or snapshot["technical_failures"] != 0
        or [row["sequence"] for row in snapshot["events"]] != [17, 19]
        or len(matching) != 1
        or matching[0]["spec"] != spec
        or matching[0]["registration"]["spec_sha256"] != snapshot["spec_sha256"]
        or matching[0]["registration"]["registered_at"] != snapshot["registered_at"]
        or digest_json(spec) != snapshot["spec_sha256"]
        or digest_json(bundle) != snapshot["evidence_sha256"]
        or results["candidates"][CANDIDATE] != review
        or review["status"] != "rejected_historical"
        or review["order_authority"] is not False
        or review["independent_forward_validation"] is not False
        or bundle["candidate_spec_sha256"] != snapshot["spec_sha256"]
        or bundle["market"] != spec["market"]
        or bundle["leveraged_products_allowed"] is not False
        or bundle["order_authority"] is not False
        or spec["leveraged_products_allowed"] is not False
        or spec["order_authority"] is not False
        or set(spec["asset_leverage"]) != set(ASSETS)
        or any(value != 1.0 for value in spec["asset_leverage"].values())
        or spec["configuration"] != {"id": CANDIDATE, "inverted_equity_scale": 0.5}
        or spec["configuration"] != policy["candidates"][0]
        or policy["risk_indicator"] != "VIX/VIX3M"
        or policy["risk_off_threshold"] != 1.0
        or policy["cash_reserve"] != 0.02
        or set((row["years"], row["scenario"]) for row in bundle["paths"]) != set(PATHS)
        or len(bundle["paths"]) != 4
        or set((row["years"], row["scenario"]) for row in review["paths"]) != set(PATHS)
        or len(review["paths"]) != 4
        or not pd.Timestamp(snapshot["registered_at"])
        <= pd.Timestamp(bundle["completed_at"])
        <= pd.Timestamp(snapshot["reviewed_at"])
    ):
        raise QuantError("The complete original term policy,source identity or review changed.")
    return {
        "original_candidate_event_sequence": 17,
        "original_review_event_sequence": 19,
        "original_standalone_verdict_preserved": "rejected_historical",
        "nominal_asset_leverage_one": True,
        "source_identity_and_four_path_completeness_verified": True,
    }


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
        raise QuantError("Use the unchanged active common RF reader on its actual calendar.")
    rows, blockers = [], []
    native_records = {(row["years"], row["scenario"]): row for row in bundle["paths"]}
    reviews = {(row["years"], row["scenario"]): row for row in snapshot["review"]["paths"]}
    for years, scenario in PATHS:
        record, data = native_records[years, scenario], frames[years, scenario]
        book, independent, weights, targets = (
            data[key] for key in ("strategy", "strategy_bt", "weights", "targets")
        )
        start = "2016-10-06" if years == 10 else "2021-10-06"
        end = "2026-10-05"
        expected = sessions(start, end)
        anchor = previous_session(start)
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
            or not isinstance(targets.index, pd.DatetimeIndex)
            or targets.empty
            or not targets.index.is_unique
            or not targets.index.is_monotonic_increasing
            or not targets.index.equals(sessions(targets.index[0], targets.index[-1]))
            or list(targets.columns) != list(ASSETS)
            or not book.index.isin(targets.index).all()
            or (targets.isna().any(axis=1) & ~targets.isna().all(axis=1)).any()
            or not expected.isin(common_rf.index).all()
        ):
            raise QuantError(
                "Cached native book,prices-free weights or request calendar is incomplete."
            )
        active = targets.dropna(how="all")
        if (
            active.empty
            or active.index[0] != anchor
            or not np.isfinite(active.to_numpy()).all()
            or (active < 0).any().any()
            or not np.allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
        ):
            raise QuantError(
                "Retain complete native funded requests and the actual initial anchor."
            )
        prior_nav = book["equity"].shift(1)
        prior_nav.iloc[0] = 10000
        nav_gap = float(max(abs(book["equity"] / prior_nav - 1 - book["return"])))
        weight_gap = float(max(abs(weights.sum(axis=1) - book["gross_exposure"])))
        bt_gap = float(max(abs(book["equity"] - independent["equity"])))
        review = reviews[years, scenario]
        if (
            review["market_and_target_replay_passed"] is not True
            or review["independent_bt_reexecuted"] is not True
            or review["max_independent_equity_error_usd"] > 1e-6
        ):
            raise QuantError("The original cached technical verification was not complete.")
        funding = (
            float(book["cash"].min()) >= 0
            and float(book["gross_exposure"].max()) <= 1
            and float(book["gross_exposure"].min()) >= 0
            and nav_gap <= 1e-12
            and weight_gap <= 1e-12
            and bt_gap <= 1e-6
        )
        comparable_rf = common_rf.loc[book.index]
        mismatches = book["risk_free"].to_numpy() != comparable_rf.to_numpy()
        rf_exact = not bool(mismatches.any())
        monthends = pd.DatetimeIndex(
            [day for day in targets.loc[anchor:end].index if is_month_end(day)]
        )
        current_decisions = pd.DatetimeIndex([anchor, *monthends])
        missing_decisions = current_decisions[~current_decisions.isin(active.index)]
        dense = active.index.equals(targets.loc[anchor:].index)
        monthly = active.index.equals(monthends)
        if not funding:
            blockers.append(
                {"path": f"{years}y-{scenario}", "reason": "native_cash_funding_or_accounting"}
            )
        if not rf_exact:
            blockers.append(
                {"path": f"{years}y-{scenario}", "reason": "native_RF_not_exact_active_common"}
            )
        if not dense and not monthly:
            blockers.append(
                {
                    "path": f"{years}y-{scenario}",
                    "reason": "native_event_requests_do_not_fit_existing_dense_or_monthly_Adapter",
                }
            )
        if len(missing_decisions):
            blockers.append(
                {"path": f"{years}y-{scenario}", "reason": "missing_original_meta_decision"}
            )
        rows.append(
            {
                "path_key": f"{years}y-{scenario}",
                "formal_return_dates": len(book),
                "first_return_date": start,
                "last_return_date": end,
                "actual_initial_anchor": str(anchor.date()),
                "first_native_request": str(active.index[0].date()),
                "native_request_rows": len(active),
                "native_month_end_request_rows": int(active.index.isin(monthends).sum()),
                "native_non_month_end_request_rows": int((~active.index.isin(monthends)).sum()),
                "original_meta_decisions": len(current_decisions),
                "missing_meta_decision_dates": missing_decisions.strftime("%Y-%m-%d").tolist(),
                "native_initial_anchor_is_month_end": is_month_end(anchor),
                "existing_dense_Adapter_calendar_compatible": dense,
                "existing_monthly_Adapter_calendar_compatible": monthly,
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
                "native_sparse_calendar_claimed_source_technically_invalid": False,
            }
        )
    compatible = not blockers
    return {
        "candidate_id": CANDIDATE,
        "mechanism_group": GROUP,
        "four_native_cached_paths_checked": 4,
        "paths": rows,
        "current_expert_pipeline_eligible_without_new_Adapter_or_RF_view": compatible,
        "status": "eligible_not_admitted" if compatible else "not_admitted_current_Interface",
        "blockers": blockers,
        "review_is_actual_expert_admission": False,
        "source_policy_technically_invalid_merely_for_sparse_requests": False,
        "source_half_exit_variants_ranked_by_historical_returns": False,
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
