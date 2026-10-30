from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_quant import legacy_profit_expert_v1 as admission
from us_quant.bt_audit import independent_equity
from us_quant.calendar import previous_session, sessions
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import digest_json, file_digest, read_json
from us_quant.strategy_experts_v1 import ArchivedMonthlyExpert, ExpertIdentity


def identity_fixture():
    policy = read_json(admission.POLICY)
    spec = {
        "id": admission.SOURCE,
        "configuration": {"id": admission.SOURCE, "profit_measure": "output_share"},
        "market": {},
        "asset_leverage": {"SOXX": 1.0, "BIL": 1.0},
        "leveraged_products_allowed": False,
        "order_authority": False,
    }
    spec_sha = digest_json(spec)
    engine = {
        "id": "cash_funded_noop_v2",
        "source_sha256": file_digest(admission.ROOT / "src/us_quant/cash_funded_accounting_v2.py"),
    }
    bundle = {
        "candidate_spec_sha256": spec_sha,
        "market": {},
        "accounting_engine": engine,
        "leveraged_products_allowed": False,
        "order_authority": False,
        "completed_at": "2026-10-12T04:15:00+00:00",
    }
    bundle_sha = digest_json(bundle)
    registration = {
        "candidate_id": admission.SOURCE,
        "spec_sha256": spec_sha,
        "registered_at": policy["source_registered_at"],
    }
    review = {
        "candidate_id": admission.SOURCE,
        "evidence_sha256": bundle_sha,
        "status": "rejected_historical",
        "order_authority": False,
        "live_strategy_update": False,
        "accounting_engine": engine,
    }
    ledger = {
        "spec": spec,
        "spec_sha": spec_sha,
        "evidence_sha": bundle_sha,
        "review": review,
        "registered_at": policy["source_registered_at"],
        "reviewed_at": "2026-10-12T04:17:00+00:00",
        "events": {
            219: {"kind": "candidate", "body": registration, "at": policy["source_registered_at"]},
            221: {"kind": "review", "body": review, "at": "2026-10-12T04:17:00+00:00"},
        },
    }
    policy["source_specification_semantic_sha256"] = spec_sha
    policy["source_bundle_semantic_sha256"] = bundle_sha
    top = {"registered_before_strategy_accounts": True, "order_authority": False}
    results = {
        "started_at": "2026-10-12T04:11:00+00:00",
        "published_registration_commit": "a" * 40,
        "order_authority": False,
    }
    publication = {
        "verified_on_remote_at": "2026-10-12T04:10:00+00:00",
        "published_commit": "a" * 40,
        "observed_remote_head": "a" * 40,
        "publication_fence_predates_new_strategy_accounts": True,
        "order_authority": False,
    }
    return (
        policy,
        top,
        {"spec": spec, "registration": registration},
        bundle,
        ledger,
        results,
        publication,
    )


def test_actual_legacy_identity_does_not_require_financial_pass():
    args = identity_fixture()
    admission.check_identity(*args)
    assert args[4]["review"]["status"] == "rejected_historical"


@pytest.mark.parametrize("damage", ["leverage", "false_fence", "wrong_spec", "live", "clock"])
def test_invalid_legacy_identity_is_not_washed_by_expert_admission(damage):
    args = identity_fixture()
    if damage == "leverage":
        args[2]["spec"]["asset_leverage"]["SOXX"] = 2.0
    elif damage == "false_fence":
        args[6]["verified_on_remote_at"] = "2026-10-12T04:12:00+00:00"
    elif damage == "wrong_spec":
        args[4]["spec_sha"] = "b" * 64
    elif damage == "live":
        args[4]["review"]["live_strategy_update"] = True
    else:
        args[4]["events"][219]["at"] = "2026-10-11T00:00:00+00:00"
    with pytest.raises(QuantError, match="identity|permission|sequence"):
        admission.check_identity(*args)


def macro_fixture():
    calendar = sessions("2015-08-10", "2016-05-31")
    pairs = admission.decision_pairs(calendar)
    queries = pd.DatetimeIndex(sorted(set(pairs) | set(pairs.values())))
    rows = []
    for i, day in enumerate(queries):
        cutoff = previous_session(previous_session(day))
        quarter = (cutoff.to_period("Q") - 1).start_time
        completed = quarter + pd.offsets.QuarterEnd(startingMonth=3)
        profit, output = 1000.0 - 2 * i, 10000.0
        rows.append(
            {
                "vintage_cutoff": str(cutoff.date()),
                "observation_quarter": str(quarter.date()),
                "completed_period_end": str(completed.date()),
                "completed_period_age_calendar_days": (day - completed).days,
                "snapshot_available_two_nyse_sessions_before_query": True,
                "quarter_label_claimed_first_release": False,
                "same_quarter_selected_in_both_original_query_snapshots": True,
                "macro_proxy_claimed_company_margin": False,
                "independently_latest_profit_quarter": str(quarter.date()),
                "independently_latest_output_quarter": str(quarter.date()),
                "CPATAX": profit,
                "GDP": output,
                "profit_share": profit / output,
            }
        )
    common = pd.DataFrame(rows, index=queries)
    original = pd.DataFrame(np.nan, index=calendar, columns=["SOXX", "MTUM", "GLD", "BIL"])
    records = []
    for i, (day, earlier) in enumerate(pairs.items()):
        original.loc[day] = [0.49, 0.245, 0.245, 0.0]
        row = {"dollar_change": 0.01 if i % 2 else -0.01}
        for field in ("CPATAX", "profit_share"):
            row[f"{field}_level"] = float(common.loc[day, field])
            row[f"{field}_reference_level"] = float(common.loc[earlier, field])
            row[f"{field}_change"] = row[f"{field}_level"] - row[f"{field}_reference_level"]
        records.append(row)
    comparisons = pd.DataFrame(records, index=pd.DatetimeIndex(list(pairs)))
    desired = original.copy()
    for day, row in comparisons.iterrows():
        if row["dollar_change"] > 0 and row["profit_share_change"] < 0:
            desired.loc[day, ["SOXX", "MTUM", "GLD"]] *= 0.5
            desired.loc[day, "BIL"] = 0.49
    return common, comparisons, original, desired, calendar


def test_saved_common_quarter_rule_uses_two_prior_sessions_and_exact_queries():
    args = macro_fixture()
    result = admission.check_macro_tables(*args)
    assert result["common_query_rows_checked"] == len(args[0])
    assert result["monthly_nominal_rows_checked"] == len(args[1])
    assert result["profit_share_confirmed_dollar_risk_off_months"] > 0
    assert result["new_target_artifact_generated"] is False
    assert result["raw_snapshot_selector_or_original_target_generator_rerun"] is False


@pytest.mark.parametrize(
    "damage", ["cutoff", "quarter", "age", "GDP", "ratio", "comparison", "target"]
)
def test_cached_macro_source_or_rule_error_is_not_a_successful_default(damage):
    common, comparisons, original, desired, calendar = macro_fixture()
    if damage == "cutoff":
        common.iloc[0, common.columns.get_loc("vintage_cutoff")] = str(common.index[0].date())
    elif damage == "quarter":
        common.iloc[0, common.columns.get_loc("observation_quarter")] = "2026-10-01"
    elif damage == "age":
        common.iloc[0, common.columns.get_loc("completed_period_age_calendar_days")] = 0
    elif damage == "GDP":
        common.iloc[0, common.columns.get_loc("GDP")] = 0
    elif damage == "ratio":
        common.iloc[0, common.columns.get_loc("profit_share")] += 0.01
    elif damage == "comparison":
        comparisons.iloc[0, comparisons.columns.get_loc("profit_share_change")] += 0.01
    else:
        desired.loc[desired.dropna().index[0], "SOXX"] += 0.01
    with pytest.raises(QuantError, match="macro|quarter|vintage|policy"):
        admission.check_macro_tables(common, comparisons, original, desired, calendar)


@pytest.fixture(scope="module")
def native_book():
    calendar = sessions("2016-09-01", "2016-10-31")
    phase = np.arange(len(calendar))
    closing = pd.DataFrame(
        {
            "SOXX": 100 * np.exp(0.001 * phase),
            "BIL": 100 * np.exp(0.0001 * phase),
        },
        index=calendar,
    )
    opening = closing.shift(1)
    opening.iloc[0] = closing.iloc[0]
    data = MarketData(
        opening,
        closing,
        closing.copy(),
        pd.DataFrame(1e6, index=calendar, columns=closing.columns),
        pd.Series(0.00005, index=calendar, name="risk_free"),
    )
    nominal = pd.DataFrame(np.nan, index=calendar, columns=closing.columns)
    nominal.loc["2016-09-30"] = [0.49, 0.49]
    nominal.loc["2016-10-31"] = [0.49, 0.49]
    issued = pd.DataFrame(np.nan, index=calendar, columns=closing.columns)
    issued.loc["2016-10-05"] = [0.49, 0.49]
    own = simulate(data, issued, "2016-10-06", "2016-10-31", initial_capital=10000)
    bt = independent_equity(
        data,
        issued,
        "2016-10-06",
        "2016-10-31",
        capital=10000,
        cost_bps=5,
        commission=1,
        delay=1,
    )
    return own.frame, own.weights, bt, issued, nominal, data, own.frame.index


def test_cached_book_matches_existing_BT_without_account_reexecution(native_book):
    result = admission.check_native_book(*native_book)
    assert result["old_account_BT_or_primary_metrics_recomputed"] is False
    assert abs(result["inferred_initial_capital_usd"] - 10000) < 1e-8
    assert result["cached_BT_maximum_NAV_difference_usd"] < 1e-8


@pytest.mark.parametrize("damage", ["cash", "gross", "BT", "RF", "signal"])
def test_invalid_cached_book_cannot_enter_pool(native_book, damage):
    account, weights, bt, issued, nominal, market, dates = native_book
    account, bt, issued = account.copy(), bt.copy(), issued.copy()
    if damage == "cash":
        account.iloc[0, account.columns.get_loc("cash")] = -1
    elif damage == "gross":
        account.iloc[0, account.columns.get_loc("gross_exposure")] = 1.1
    elif damage == "BT":
        bt.iloc[0, bt.columns.get_loc("equity")] += 1
    elif damage == "RF":
        account.iloc[0, account.columns.get_loc("risk_free")] += 0.01
    else:
        issued.loc["2016-10-10"] = [0.49, 0.49]
    with pytest.raises(QuantError, match="cached|native request"):
        admission.check_native_book(account, weights, bt, issued, nominal, market, dates)


def test_existing_monthly_adapter_preserves_known_targets_and_net_prefixes(native_book):
    account, _, _, _, nominal, _, _ = native_book
    expert = ArchivedMonthlyExpert(
        ExpertIdentity(admission.SOURCE, admission.GROUP, "2026-10-30T00:00:00+00:00"),
        nominal,
        account,
        tuple(nominal.columns),
    )
    initial = expert.observe(pd.Timestamp("2016-10-05"), 126)
    assert initial.target_known_at == pd.Timestamp("2016-09-30")
    assert initial.net_returns.empty and initial.score_mature is False
    before = expert.observe(pd.Timestamp("2016-10-12"), 126)
    changed = account.copy()
    changed.loc["2016-10-20":, "return"] += 0.01
    changed["equity"] = 10000 * (1 + changed["return"]).cumprod()
    changed["cash"] = 0.02 * changed["equity"]
    after = ArchivedMonthlyExpert(expert.identity, nominal, changed, tuple(nominal.columns))
    pd.testing.assert_series_equal(before.net_returns, after.observe("2016-10-12", 126).net_returns)
    pd.testing.assert_series_equal(before.target, after.observe("2016-10-12", 126).target)


def test_admission_policy_cannot_claim_company_margin_or_frozen_future_pool():
    policy = read_json(admission.POLICY)
    policy["methodology"]["expanded_pool_or_next_meta_rule_frozen"] = True
    with pytest.raises(QuantError, match="qualification"):
        admission.validate(policy)
