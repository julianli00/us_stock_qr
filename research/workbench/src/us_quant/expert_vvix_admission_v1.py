from __future__ import annotations

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.daily_variance_disagreement_v1 import UNIVERSE
from us_quant.strategy_experts_v1 import ArchivedMonthlyExpert

SOURCE = "two_segment_vvix_uncertainty_confirmation_v1"
GROUP = "option_market_uncertainty_confirmation"
CLOCK = "after_16_15_ET_index_close_before_later_open_execution"
SCOPE = "fixed original archival daily closes only"
ASSETS = (*UNIVERSE, "IJR", "PKW")


def validate_scope(policy, readiness, manifest, proof):
    fixed = {
        "study_id": "ssd_vvix_uncertainty_confirmation_20261029_v1",
        "vvix_comparison_sessions": 63,
        "source_monthly_opportunities": 131,
        "missing_required_quote_behavior": "stop_version_without_filling_or_defaulting",
        "decision_information_clock": CLOCK,
        "candidate_count_in_this_study": 1,
    }
    if not isinstance(policy, dict) or any(
        policy.get(key) != value or type(policy.get(key)) is not type(value)
        for key, value in fixed.items()
    ):
        raise QuantError("Preserve the complete fixed VVIX policy and after-index-close clock.")
    for source in (readiness, manifest):
        if (
            not isinstance(source, dict)
            or source.get("source_admitted") is not True
            or source.get("admission_scope") != SCOPE
            or source.get("historical_intraday_first_publication_or_complete_method_chain_verified")
            is not False
            or source.get("old_near_term_VVIX_admitted_false_record_unchanged") is not True
        ):
            raise QuantError("Daily-close source admission cannot be widened into intraday PIT.")
        coverage = source.get("coverage", {})
        if (
            coverage.get("full_daily_coverage") is not True
            or coverage.get("all_original_monthly_queries_covered") is not True
            or coverage.get("values_filled_or_market_dates_dropped") is not False
        ):
            raise QuantError("Complete original source coverage cannot be defaulted or filled.")
        for name in ("VVIX", "VIX"):
            record = coverage.get("series", {}).get(name, {})
            if (
                record.get("context_sessions") != 2805
                or record.get("observed_daily_values") != 2805
                or record.get("original_monthly_queries") != 131
                or record.get("observed_monthly_values") != 131
                or record.get("missing_daily_dates") != []
                or record.get("missing_monthly_dates") != []
            ):
                raise QuantError(
                    "Retain every admitted native daily and monthly source observation."
                )
    if (
        readiness.get("private_manifest") != policy["vvix_manifest"]
        or readiness.get("aligned_closes") != manifest.get("aligned_closes")
        or proof.get("monthly_decisions") != 131
        or proof.get("independent_native_quote_queries_checked") != 262
        or proof.get("VVIX_comparison_sessions") != 63
        or proof.get("decision_after_index_close_before_original_delayed_opens") is not True
        or proof.get("intraday_first_publication_or_method_history_authenticated") is not False
        or proof.get("original_source_signal_recomputed") is not False
        or proof.get("candidate_accounts_or_metrics_computed") is not False
        or proof.get("VVIX_index_return_or_variance_premium_claimed") is not False
        or proof.get("source_monthly_dates_and_relative_composition_preserved") is not True
    ):
        raise QuantError("The original source scope,target proof or publication clock changed.")
    return {
        "source_admitted": True,
        "admission_scope": SCOPE,
        "required_decision_information_clock": CLOCK,
        "historical_intraday_first_publication_or_method_chain_authenticated": False,
        "original_near_term_source_nonadmission_preserved": True,
        "native_daily_context_sessions": 2805,
        "native_monthly_source_opportunities": 131,
        "original_independent_native_quote_queries": 262,
        "new_source_requests": 0,
    }


def check_cached_observations(expert, targets, shadow, common_rf, retain=None):
    if (
        not isinstance(targets, pd.DataFrame)
        or targets.empty
        or tuple(targets.columns) != UNIVERSE
        or (targets.isna().any(axis=1) & ~targets.isna().all(axis=1)).any()
        or not targets.index.equals(sessions(targets.index[0], targets.index[-1]))
        or not isinstance(shadow, pd.DataFrame)
        or not shadow.index.equals(sessions("2016-10-06", "2026-10-05"))
        or not isinstance(common_rf, pd.Series)
        or not shadow.index.isin(common_rf.index).all()
        or not np.array_equal(
            shadow["risk_free"].to_numpy(), common_rf.loc[shadow.index].to_numpy()
        )
    ):
        raise QuantError("Use the complete native NET book and unchanged active common RF.")
    active = targets.dropna(how="all")
    if active.empty:
        raise QuantError(
            "An expert archive needs genuine native recommendations,not a filled seed."
        )
    native_dates = pd.DatetimeIndex(
        [day for day in targets.loc[active.index[0] :].index if is_month_end(day)]
    )
    if len(active) != 131 or not active.index.equals(native_dates):
        raise QuantError("The original monthly expert cannot fill or skip a source recommendation.")
    initial = pd.Timestamp("2016-10-05")
    if initial not in targets.index:
        raise QuantError("The genuine initial main signal must be covered by the native archive.")
    dates = pd.DatetimeIndex(
        [initial, *[day for day in targets.index if day > initial and is_month_end(day)]]
    )
    if len(dates) != 121:
        raise QuantError("Preserve all original initial/month-end expert queries.")
    available = mature = 0
    for day in dates:
        observation = expert.observe(day, 126)
        previous = active.loc[:day]
        known = shadow.loc[:day].tail(126)
        expected = previous.iloc[-1].reindex(ASSETS, fill_value=0.0)
        if (
            observation.as_of != day
            or observation.target is None
            or observation.target_known_at != previous.index[-1]
            or observation.target_known_at > day
            or not observation.target.equals(expected)
            or not observation.net_returns.equals(known["return"])
            or not observation.risk_free.equals(known["risk_free"])
            or observation.score_mature is not (len(known) == 126)
            or observation.historical_counterfactual is not True
            or any(observation.net_returns.index > day)
        ):
            raise QuantError(
                f"A cached expert exposed a future or altered observation at{day.date()}."
            )
        available += 1
        mature += observation.score_mature
        if retain is not None:
            retain(
                {
                    "as_of": str(day.date()),
                    "target_known_at": str(observation.target_known_at.date()),
                    "target_available": True,
                    "score_mature": observation.score_mature,
                    "known_NET_observations": len(known),
                    "native_NET_and_RF_unchanged": True,
                    "information_clock_required": CLOCK,
                }
            )
    prefixes = []
    for position in (6, 12, 60, 120):
        cutoff = dates[position]
        before = ArchivedMonthlyExpert(
            expert.identity, targets.loc[:cutoff], shadow.loc[:cutoff], ASSETS
        )
        for day in dates[dates <= cutoff]:
            full, truncated = expert.observe(day, 126), before.observe(day, 126)
            if (
                full.target_known_at != truncated.target_known_at
                or not full.target.equals(truncated.target)
                or not full.net_returns.equals(truncated.net_returns)
                or not full.risk_free.equals(truncated.risk_free)
                or full.score_mature is not truncated.score_mature
            ):
                raise QuantError(
                    "Later cached source records changed an earlier expert observation."
                )
        prefixes.append(str(cutoff.date()))
    if available != 121 or mature != 114:
        raise QuantError("Keep the original availability and seven unscored initial observations.")
    return {
        "native_month_end_targets_unchanged": 131,
        "first_native_target": str(active.index[0].date()),
        "last_native_target": str(active.index[-1].date()),
        "all_main_observation_queries_checked": 121,
        "available_main_observations": available,
        "main_observations_with_mature_126session_assessment": mature,
        "available_but_unmatured_initial_observations": available - mature,
        "initial_empty_shadow_history_checked_without_synthetic_fill": True,
        "nonempty_shadow_prefix_truncations": prefixes,
        "native_NET_and_common_RF_equal_without_new_view": True,
        "native_12_to_admitted_14_assets_are_structural_zero_exposures": True,
        "source_generator_account_or_verifier_pipeline_replayed": False,
        "counterfactual_historical_availability": True,
        "independent_forward_validation": False,
    }
