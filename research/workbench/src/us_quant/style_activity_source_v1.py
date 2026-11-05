from __future__ import annotations

import numpy as np
import pandas as pd

from us_quant import core_activity_inputs_v1 as admitted
from us_quant import ssd_research_v1 as base
from us_quant import style_activity_participation_v1 as law
from us_quant.calendar import previous_session, sessions
from us_quant.config import QuantError
from us_quant.storage import read_json

ROOT = law.ROOT


def unique_refs(rows):
    result = {}
    for row in rows:
        if row["path"] in result and row != result[row["path"]]:
            raise QuantError("One immutable input path cannot have conflicting source hashes.")
        result[row["path"]] = row
    return [result[k] for k in sorted(result)]


def source_references():
    policy = read_json(law.POLICY)
    law.validate(policy)
    initial = [
        policy[k]
        for k in (
            "source_market_freeze",
            "source_market_completion",
            "activity_admission",
            "activity_manifest",
        )
    ]
    for row in initial:
        base.checked_reference(row)
    ready = read_json(base.checked_reference(policy["activity_admission"]))
    manifest = read_json(base.checked_reference(policy["activity_manifest"]))
    current = read_json(base.checked_reference(policy["source_market_freeze"]))["frozen"]["market"]
    if (
        tuple(admitted.CORE) != law.CORE
        or ready["source_admitted"] is not True
        or manifest["source_admitted"] is not True
        or ready["admission_scope"] != "original_four_style_fund_reported_activity_fields_only"
        or manifest["admission_scope"] != ready["admission_scope"]
        or tuple(r["symbol"] for r in ready["funds"]) != law.CORE
        or ready["funds"] != manifest["funds"]
        or ready["private_manifest"] != policy["activity_manifest"]
        or ready["source_quote_requests"] != 0
        or ready["old_prices_Rf_volumes_replaced"] is not False
        or any(
            r["reported_share_volume_native_CSV_registered_exact"] is not True
            or r["reported_split_events"] != 0
            or r["original_price_identity_rtol"] != admitted.RTOL
            or r["original_price_identity_atol"] != admitted.ATOL
            or r["source_gaps_filled_or_dates_deleted"] is not False
            for r in ready["funds"]
        )
    ):
        raise QuantError("Consume only the unchanged four-fund admitted activity scope.")
    validation = read_json(base.checked_reference(ready["validation_intent"]))
    refs = unique_refs(
        [
            *initial,
            *ready["original_market"].values(),
            *current.values(),
            ready["source_intent"],
            ready["validation_intent"],
            *validation["inputs"],
        ]
    )
    for row in refs:
        base.checked_reference(row)
    return current, ready, refs


def compare_fields(old: dict, data) -> dict:
    if set(old) != {"open", "close", "raw_close", "volume"}:
        raise QuantError("Reuse only four admitted price/activity fields, not an old RF view.")
    checks = {}
    for key, frame in old.items():
        current = getattr(data, key)
        if (
            not frame.index.equals(current.index)
            or not frame.index.is_unique
            or not set(law.CORE).issubset(frame.columns)
            or not set(law.CORE).issubset(current.columns)
        ):
            raise QuantError("No activity calendar or admitted style may be replaced or cropped.")
        left, right = [p.loc[:, list(law.CORE)].to_numpy(dtype=float) for p in (frame, current)]
        if (
            not np.isfinite(left).all()
            or not np.isfinite(right).all()
            or (left <= 0).any()
            or (right <= 0).any()
        ):
            raise QuantError("Missing or nonpositive activity cannot receive a source waiver.")
        equal = (
            np.array_equal(left, right)
            if key == "volume"
            else np.allclose(left, right, rtol=admitted.RTOL, atol=admitted.ATOL)
        )
        if not equal or (key == "volume" and not np.array_equal(right, np.floor(right))):
            raise QuantError("Current inputs differ from the unchanged four-fund source admission.")
        checks[key] = float(np.max(np.abs(left - right)))
    return {
        "source_admission_scope": "original_four_style_fund_reported_activity_fields_only",
        "matched_price_volume_values_checked": len(data.close) * len(law.CORE) * 4,
        "maximum_compatibility_difference_by_field": checks,
        "original_price_identity_rtol": admitted.RTOL,
        "original_price_identity_atol": admitted.ATOL,
        "reported_share_volume_exact_and_positive_integer": True,
        "old_activity_source_RF_imported_or_normalized": False,
        "new_policy_RF_is_unchanged_current_canonical_native_RF": True,
        "old_policy_native_RF_claimed_equal_to_current_native_RF": False,
        "source_scope_extended_to_IJR_PKW_or_other_funds": False,
        "source_verification_pipeline_replayed": False,
    }


def readiness():
    market, ready, refs = source_references()
    data = base.review_market({"market": market}, ROOT)
    old = {
        key: pd.read_csv(
            base.checked_reference(ready["original_market"][key]),
            index_col=0,
            parse_dates=True,
        )
        for key in ("open", "close", "raw_close", "volume")
    }
    checked = compare_fields(old, data)
    dates = law.eligible_sessions(data)
    anchors = []
    for start in ("2016-10-06", "2021-10-06"):
        formal = sessions(start, "2026-10-05")
        anchor = previous_session(start)
        known = dates[dates <= anchor]
        if (
            not formal.isin(data.close.index).all()
            or len(known) == 0
            or not np.isfinite(data.risk_free.loc[formal]).all()
        ):
            raise QuantError(
                "Keep full original returns and a possible fully informed native seed."
            )
        anchors.append(
            {
                "formal_start": start,
                "formal_return_sessions": len(formal),
                "capital_anchor": str(anchor.date()),
                "planned_prior_native_opinion_session": str(known[-1].date()),
                "actual_new_policy_target_or_score_computed": False,
            }
        )
    return (
        data,
        refs,
        {
            **checked,
            "market_sessions": len(data.close),
            "asset_count": len(data.close.columns),
            "admitted_activity_symbol_count": len(law.CORE),
            "planned_eligible_native_month_ends": len(dates),
            "planned_first_native_month_end": str(dates[0].date()),
            "planned_last_native_month_end": str(dates[-1].date()),
            "formal_anchor_calendar_readiness": anchors,
            "actual_new_indicators_scores_opinions_targets_accounts_or_metrics_computed": False,
            "expert_technically_admitted": False,
            "global_pool_changed": False,
            "new_source_requests": 0,
            "order_authority": False,
        },
    )
