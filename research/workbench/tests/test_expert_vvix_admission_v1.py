from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import expert_vvix_admission_v1 as rules
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import ArchivedMonthlyExpert, ExpertIdentity


def source_scope():
    policy = read_json(
        Path(rules.__file__).resolve().parents[2] / "config/vvix-uncertainty-confirmation-v1.json"
    )
    coverage = {
        "full_daily_coverage": True,
        "all_original_monthly_queries_covered": True,
        "values_filled_or_market_dates_dropped": False,
        "series": {
            name: {
                "context_sessions": 2805,
                "observed_daily_values": 2805,
                "original_monthly_queries": 131,
                "observed_monthly_values": 131,
                "missing_daily_dates": [],
                "missing_monthly_dates": [],
            }
            for name in ("VIX", "VVIX")
        },
    }
    manifest = {
        "source_admitted": True,
        "admission_scope": rules.SCOPE,
        "historical_intraday_first_publication_or_complete_method_chain_verified": False,
        "old_near_term_VVIX_admitted_false_record_unchanged": True,
        "coverage": coverage,
        "aligned_closes": {"path": "synthetic.csv", "sha256": "a" * 64},
    }
    readiness = deepcopy(manifest)
    readiness["private_manifest"] = policy["vvix_manifest"]
    proof = {
        "monthly_decisions": 131,
        "independent_native_quote_queries_checked": 262,
        "VVIX_comparison_sessions": 63,
        "decision_after_index_close_before_original_delayed_opens": True,
        "intraday_first_publication_or_method_history_authenticated": False,
        "original_source_signal_recomputed": False,
        "candidate_accounts_or_metrics_computed": False,
        "VVIX_index_return_or_variance_premium_claimed": False,
        "source_monthly_dates_and_relative_composition_preserved": True,
    }
    return policy, readiness, manifest, proof


def synthetic():
    calendar = sessions("2015-08-10", "2026-10-05")
    target = pd.DataFrame(np.nan, index=calendar, columns=rules.UNIVERSE)
    for day in calendar:
        if day >= pd.Timestamp("2015-11-30") and is_month_end(day):
            target.loc[day] = 0.0
            target.loc[day, ["MTUM", "VLUE", "QUAL", "USMV"]] = 0.245
    dates = sessions("2016-10-06", "2026-10-05")
    phase = np.arange(len(dates))
    returns = pd.Series(0.0001 + 0.001 * np.sin(phase / 19), index=dates)
    equity = 10000 * (1 + returns).cumprod()
    common = pd.Series(0.00005, index=calendar, name="risk_free")
    shadow = pd.DataFrame(
        {
            "equity": equity,
            "return": returns,
            "risk_free": common.loc[dates],
            "cash": equity * 0.02,
            "gross_exposure": 0.98,
            "cost": 0.0,
            "orders": 0,
        }
    )
    expert = ArchivedMonthlyExpert(
        ExpertIdentity(rules.SOURCE, rules.GROUP, "2026-10-28T23:58:17.764688+00:00"),
        target,
        shadow,
        rules.ASSETS,
    )
    return expert, target, shadow, common


@pytest.mark.parametrize(
    "kind", ["early_clock", "not_admitted", "intraday", "missing_quotes", "old_scope_reset"]
)
def test_archival_scope_and_after_index_close_clock_are_not_widened(kind):
    policy, readiness, manifest, proof = source_scope()
    if kind == "early_clock":
        policy["decision_information_clock"] = "16_00_stock_close"
    elif kind == "not_admitted":
        manifest["source_admitted"] = False
    elif kind == "intraday":
        readiness["historical_intraday_first_publication_or_complete_method_chain_verified"] = True
    elif kind == "missing_quotes":
        manifest["coverage"]["series"]["VVIX"]["missing_monthly_dates"] = ["2020-03-31"]
    else:
        readiness["old_near_term_VVIX_admitted_false_record_unchanged"] = False
    with pytest.raises(QuantError):
        rules.validate_scope(policy, readiness, manifest, proof)


def test_valid_scope_retains_limitations():
    facts = rules.validate_scope(*source_scope())
    assert facts["source_admitted"] is True and facts["admission_scope"] == rules.SCOPE
    assert facts["required_decision_information_clock"] == rules.CLOCK
    assert facts["historical_intraday_first_publication_or_method_chain_authenticated"] is False


def test_cached_observation_availability_and_maturity_are_distinct():
    values = synthetic()
    records = []
    facts = rules.check_cached_observations(*values, retain=records.append)
    assert facts["native_month_end_targets_unchanged"] == 131
    assert facts["available_main_observations"] == 121
    assert facts["main_observations_with_mature_126session_assessment"] == 114
    assert facts["available_but_unmatured_initial_observations"] == 7
    assert facts["nonempty_shadow_prefix_truncations"] == [
        "2017-03-31",
        "2017-09-29",
        "2021-09-30",
        "2026-09-30",
    ]
    assert records[0]["known_NET_observations"] == 0 and records[0]["target_available"] is True
    assert records[0]["score_mature"] is False and len(records) == 121


def test_native_RF_mismatch_is_not_replaced():
    expert, target, shadow, common = synthetic()
    common.iloc[-1] += 1e-10
    with pytest.raises(QuantError, match="common RF"):
        rules.check_cached_observations(expert, target, shadow, common)


def test_native_monthly_gap_is_not_filled():
    expert, target, shadow, common = synthetic()
    target.loc["2022-12-30"] = np.nan
    with pytest.raises(QuantError, match="skip a source recommendation"):
        rules.check_cached_observations(expert, target, shadow, common)


@pytest.mark.parametrize("kind", ["all_empty", "partial_row", "wrong_native_lattice"])
def test_native_recommendation_shape_cannot_be_defaulted(kind):
    expert, target, shadow, common = synthetic()
    if kind == "all_empty":
        target.loc[:] = np.nan
    elif kind == "partial_row":
        target.loc["2022-12-30", "SPY"] = np.nan
    else:
        target["IJR"] = 0.0
    with pytest.raises(QuantError):
        rules.check_cached_observations(expert, target, shadow, common)


def test_no_future_cached_recommendation_is_accepted():
    from dataclasses import replace
    from types import SimpleNamespace

    expert, target, shadow, common = synthetic()

    def observe(day, lookback):
        item = expert.observe(day, lookback)
        return replace(item, target_known_at=day + pd.Timedelta(days=1))

    with pytest.raises(QuantError, match="future or altered"):
        rules.check_cached_observations(SimpleNamespace(observe=observe), target, shadow, common)
