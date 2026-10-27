from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import (
    DEADLINE,
    POLICY_SHA256,
    before_deadline,
    benchmark_account,
    execution_statistics,
    gates,
    issuer_identity,
    research_job,
)
from us_quant.storage import file_digest, read_json


def metrics(**changes):
    return {
        "start": "2021-10-06",
        "end": "2026-10-05",
        "sessions": 1254,
        "cagr": 0.16,
        "sharpe": 1.1,
        "max_drawdown": 0.15,
        **changes,
    }


def test_protocol_matches_the_actual_pre_outcome_freeze():
    path = Path(__file__).parents[1] / "config/dual-benchmark-v1.json"
    assert file_digest(path) == POLICY_SHA256
    policy = read_json(path)
    assert policy["benchmarks"]["symbols"] == ["SPY", "QQQ"]
    assert policy["benchmarks"]["target_weight"] == 1
    assert policy["execution_disclosure"]["numeric_turnover_cap"] is None
    assert policy["hard_gates_per_path"]["net_cagr_strictly_greater_than"] == 0.12


@pytest.mark.parametrize(
    "changes,failed",
    [
        ({"cagr": 0.12}, "cagr_above_12pct"),
        ({"sharpe": 1.0}, "sharpe_above_1"),
        ({"sharpe": None}, "sharpe_above_1"),
        ({"max_drawdown": 0.15000001}, "drawdown_at_most_15pct"),
        ({"cagr": 0.14}, "beats_qqq"),
    ],
)
def test_each_exact_new_threshold_is_enforced(changes, failed):
    checked = gates(
        metrics(**changes),
        metrics(cagr=0.13),
        metrics(cagr=0.14),
        metrics(cagr=0.135),
        "base",
    )
    assert checked[failed] is False


def test_all_thresholds_pass_only_with_both_benchmarks_and_old_stress_gate():
    assert all(
        gates(
            metrics(), metrics(cagr=0.13), metrics(cagr=0.14), metrics(cagr=0.135), "stress"
        ).values()
    )
    checked = gates(
        metrics(cagr=0.155),
        metrics(cagr=0.13),
        metrics(cagr=0.14),
        metrics(cagr=0.155),
        "stress",
    )
    assert checked["beats_spy"] and checked["beats_qqq"]
    assert checked["also_beats_base_cost_spy"] is False


@pytest.mark.parametrize(
    "bad",
    [{}, metrics(end="2026-10-02"), metrics(cagr=float("nan")), metrics(cagr=True)],
)
def test_missing_or_mismatched_benchmark_evidence_cannot_qualify(bad):
    with pytest.raises(QuantError):
        gates(metrics(), metrics(cagr=0.13), bad, metrics(cagr=0.13), "base")


def test_shortened_windows_cannot_pass_even_when_all_labels_match():
    changed = metrics(start="2022-10-06", sessions=1000)
    with pytest.raises(QuantError, match="complete original"):
        gates(changed, changed, changed, changed, "base")


@pytest.mark.parametrize("symbol", ["SPY", "QQQ"])
@pytest.mark.parametrize(
    "scenario",
    [
        {
            "id": "base",
            "cost_bps_per_side": 5.0,
            "commission_usd_per_order": 1.0,
            "delay_sessions": 1,
        },
        {
            "id": "stress",
            "cost_bps_per_side": 20.0,
            "commission_usd_per_order": 1.0,
            "delay_sessions": 2,
        },
    ],
)
def test_full_capital_benchmarks_match_bt_closed_form_and_real_costs(
    market_factory, symbol, scenario
):
    data = market_factory()
    window = {"years": 0, "start": "2020-01-02", "end": "2020-07-31"}
    row, account, independent = benchmark_account(data, symbol, window, scenario, 10000)
    np.testing.assert_allclose(account.frame["equity"], independent["equity"], rtol=0, atol=1e-8)
    assert row["execution"]["model_order_count"] == 1
    assert row["execution"]["rebalance_session_count"] == 1
    assert row["execution"]["total_cost_usd"] > 1
    assert row["execution"]["minimum_cash_usd"] >= -1e-7
    assert row["execution"]["maximum_gross_exposure"] <= 1 + 1e-10
    assert row["execution"]["total_two_sided_turnover"] > 0.99
    if scenario["id"] == "stress":
        assert account.frame.iloc[0]["equity"] == 10000
        assert account.frame.iloc[0]["orders"] == 0


def test_execution_disclosure_rejects_relabelled_fees(market_factory):
    data = market_factory()
    scenario = {
        "id": "base",
        "cost_bps_per_side": 5,
        "commission_usd_per_order": 1,
        "delay_sessions": 1,
    }
    _, account, _ = benchmark_account(
        data, "SPY", {"years": 0, "start": "2020-01-02", "end": "2020-07-31"}, scenario, 10000
    )
    changed = deepcopy(account)
    changed.frame.loc[changed.frame.index[0], "cost"] += 1
    with pytest.raises(QuantError, match="costs"):
        execution_statistics(changed, data, 10000, 5, 1)


def test_issuer_identity_uses_actual_product_paragraph_not_country_splash():
    spy = "SPY S&P 500 price and yield performance Inception Date 01/22/1993"
    identity = issuer_identity(spy, "<p>Invesco QQQ ETF tracks the Nasdaq-100 Index.</p>")
    assert identity["QQQ"]["issuer_reported_inception"] is None
    assert identity["full_historical_constituents_or_methodology_verified"] is False
    for html in (
        "<title>Country Splash</title>",
        "<script>Invesco QQQ ETF tracks the Nasdaq-100 Index</script>",
        "<p><script>Invesco QQQ ETF tracks the Nasdaq-100 Index</script></p>",
    ):
        with pytest.raises(QuantError):
            issuer_identity(spy, html)


def test_deadline_is_exact_and_requires_an_aware_actual_clock():
    before_deadline(DEADLINE - timedelta(microseconds=1))
    for instant in (DEADLINE, DEADLINE + timedelta(seconds=1), datetime(2026, 10, 27)):
        with pytest.raises(QuantError):
            before_deadline(instant)
    before_deadline(datetime(2026, 10, 27, tzinfo=timezone.utc))


def test_one_compute_lock_is_exclusive_and_releases_without_deleting_evidence(tmp_path):
    lock = tmp_path / "research.lock"
    with research_job(lock):
        assert read_json(lock)["state"] == "running"
        with pytest.raises(QuantError, match="Another research"):
            with research_job(lock):
                pytest.fail("A concurrent computation was admitted")
    assert read_json(lock)["state"] == "released"
    with research_job(lock):
        assert read_json(lock)["state"] == "running"
