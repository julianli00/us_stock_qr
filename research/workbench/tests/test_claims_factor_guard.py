from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.bt_audit import independent_equity
from us_quant.calendar import sessions
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.claims_factor_guard import (
    CANDIDATE,
    build_from_inputs,
    check_registration_eligibility,
    four_week_queries,
    validate_policy,
)
from us_quant.config import QuantError
from us_quant.dual_horizon import seed_window
from us_quant.jobless_claims_inputs import needed_vintage_dates
from us_quant.multifactor_stability import FACTORS
from us_quant.research_program import ResearchProgram
from us_quant.sector_growth_balance import build_targets as sector_targets
from us_quant.storage import file_digest, read_json, write_text_atomic
from us_quant.strategy_replay import registered_targets

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/claims-factor-guard.json")


@pytest.fixture
def market(market_factory):
    return market_factory(
        "2015-08-10",
        "2017-06-30",
        ("SPY", "IEF", "TLT", "GLD", "BIL", "QQQ", *FACTORS, "XLK", "SOXX"),
    )


def inputs(market, dollar=1.0, change=1.0):
    baseline = sector_targets(market, read_json(ROOT / "config/sector-growth-balance.json"))[
        "four_factor_semiconductor_sector50"
    ]
    source = pd.DataFrame(
        {
            "dollar_change": dollar,
            "claims_change": change,
            "claims_change_available": True,
        },
        index=baseline.dropna(how="all").index,
    )
    return source, baseline


@pytest.mark.parametrize(
    "dollar,change",
    [
        (1.0, 1.0),
        (0.0, 1.0),
        (-1.0, 1.0),
        (1.0, 0.0),
        (1.0, -1.0),
    ],
)
def test_only_joint_positive_information_halves_all_nonbil_assets(market, policy, dollar, change):
    source, baseline = inputs(market, dollar, change)
    targets = build_from_inputs(market, source, policy)[CANDIDATE["id"]]
    active, original = targets.dropna(how="all"), baseline.dropna(how="all")
    assets = targets.columns.drop("BIL")
    scale = 0.5 if dollar > 0 and change > 0 else 1.0
    np.testing.assert_allclose(active[assets], original[assets] * scale, rtol=0, atol=1e-12)
    removed = original[assets].sum(axis=1) * (1 - scale)
    np.testing.assert_allclose(active["BIL"], original["BIL"] + removed, rtol=0, atol=1e-12)
    np.testing.assert_allclose(active.sum(axis=1), 0.98, rtol=0, atol=1e-12)
    assert (active[[*FACTORS, "SOXX", "GLD"]] > 0).all().all()
    assert targets.index.equals(baseline.index)
    assert targets.isna().all(axis=1).equals(baseline.isna().all(axis=1))


@pytest.mark.parametrize("dollar", [-1.0, 0.0, 1.0])
def test_explicit_missing_confirmation_pauses_even_when_dollar_gate_is_off(market, policy, dollar):
    source, _ = inputs(market, dollar=dollar)
    day = source.index[-1]
    source.loc[day, "claims_change"] = np.nan
    source.loc[day, "claims_change_available"] = False
    target = build_from_inputs(market, source, policy)[CANDIDATE["id"]]
    assert target.loc[day].isna().all()
    assert target.loc[day].isna().sum() == len(target.columns)
    assert target.dropna(how="all").shape[0] == len(source) - 1


@pytest.mark.parametrize(
    "problem",
    [
        "missing_month",
        "extra_month",
        "nan_dollar",
        "inf_change",
        "false_available",
        "true_available",
        "nonboolean_available",
        "columns",
    ],
)
def test_unknown_or_inconsistent_inputs_are_not_absorbed_as_standby(market, policy, problem):
    source, _ = inputs(market)
    if problem == "missing_month":
        source = source.iloc[1:]
    elif problem == "extra_month":
        source.loc[pd.Timestamp("2016-10-12")] = [1.0, 1.0, True]
    elif problem == "nan_dollar":
        source.iloc[0, 0] = np.nan
    elif problem == "inf_change":
        source.iloc[0, 1] = np.inf
    elif problem == "false_available":
        source.iloc[0, 2] = False
    elif problem == "true_available":
        source.iloc[0, 1] = np.nan
    elif problem == "nonboolean_available":
        source["claims_change_available"] = 1
    else:
        source.columns = ["dollar_change", "current_download_change", "available"]
    with pytest.raises(QuantError, match="explicit source availability"):
        build_from_inputs(market, source, policy)


def test_future_macro_and_prices_cannot_rewrite_past_targets(market, policy):
    source, _ = inputs(market)
    before = build_from_inputs(market, source, policy)[CANDIDATE["id"]]
    cutoff = pd.Timestamp("2017-01-31")
    later = source.index > cutoff
    source.loc[later, ["dollar_change", "claims_change"]] *= -1
    close, opening = market.close.copy(), market.open.copy()
    later = close.index > cutoff
    close.loc[later, "SOXX"] *= np.linspace(1, 2, later.sum())
    opening.loc[later, "SOXX"] *= np.linspace(1, 2, later.sum())
    changed = replace(market, close=close, open=opening, raw_close=close.copy())
    after = build_from_inputs(changed, source, policy)[CANDIDATE["id"]]
    pd.testing.assert_frame_equal(before.loc[:cutoff], after.loc[:cutoff])


@pytest.fixture
def claims():
    index = sessions("2015-08-10", "2026-10-05")
    weekly = pd.date_range("2014-01-04", "2026-10-03", freq="W-SAT")
    amounts = pd.Series(200000.0 + 3 * np.arange(len(weekly)), index=weekly)
    first = pd.DataFrame(
        {
            "initial_claims": amounts,
            "first_release_date": weekly + pd.Timedelta(days=4),
        },
        index=weekly,
    )
    vintages = {
        day: amounts.loc[first["first_release_date"] <= pd.Timestamp(day)].copy()
        for day in needed_vintage_dates(index)
    }
    return index, vintages, first


def test_mean_uses_four_consecutive_weeks_from_each_own_earlier_snapshot(claims):
    index, vintages, first = claims
    selected = four_week_queries(index, vintages, first)
    assert len(selected) == 225
    for _, row in selected.iterrows():
        day = pd.Timestamp(row["observation_date"])
        weeks = pd.date_range(day - pd.Timedelta(days=21), day, freq="W-SAT")
        assert row["claims_four_week_mean"] == vintages[row["vintage_cutoff"]].loc[weeks].sum() / 4
    original = selected.loc[:"2020-12-31"].copy()
    for cutoff in vintages:
        if cutoff > "2020-12-31":
            vintages[cutoff] *= 2
    changed = four_week_queries(index, vintages, first)
    pd.testing.assert_frame_equal(original, changed.loc[:"2020-12-31"])


def test_a_missing_week_in_the_average_is_fatal_not_interpolation(claims):
    index, vintages, first = claims
    cutoff = "2026-09-28"
    latest = vintages[cutoff].index[-1]
    vintages[cutoff] = vintages[cutoff].drop(latest - pd.Timedelta(days=7))
    with pytest.raises(QuantError, match="four actual consecutive"):
        four_week_queries(index, vintages, first)


def test_known_stale_query_never_produces_a_claims_mean_or_safe_numeric_state(claims):
    index, vintages, first = claims
    vintages["2026-09-28"] = vintages["2026-09-28"].loc[:"2026-09-05"]
    selected = four_week_queries(index, vintages, first)
    assert not selected.loc["2026-09-30", "source_ready"]
    assert pd.isna(selected.loc["2026-09-30", "claims_four_week_mean"])
    assert selected.loc["2026-09-30", "blocker"] == "observation_age_exceeds_frozen_fourteen_days"


def test_only_exact_predeclared_missing_component_is_recorded_as_unavailable(claims):
    index, vintages, first = claims
    cutoff = "2018-11-21"
    vintages[cutoff] = vintages[cutoff].drop(pd.Timestamp("2018-10-27"))
    selected = four_week_queries(index, vintages, first)
    assert selected.loc["2018-11-26", "source_ready"]
    assert pd.isna(selected.loc["2018-11-26", "claims_four_week_mean"])
    assert selected.loc["2018-11-26", "claims_mean_blocker"] == (
        "recorded_missing_four_week_component_2018_10_27"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("risk_change_sessions", 21),
        ("claims_smoothing_weeks", 5),
        ("claims_change_threshold", 1000.0),
        ("risk_off_investment_scale", 0.75),
        ("cash_reserve", 0),
        ("stale_input_behavior", "cash_fallback"),
        ("eligible_not_before", "2026-10-11T09:00:00+08:00"),
        ("planned_configurations", 2),
        ("new_economic_factor_definitions", 1),
        ("source_age_limit_relaxed", True),
        ("existing_forward_rules_changed", True),
        ("order_authority", True),
    ],
)
def test_no_threshold_smoothing_scale_standby_or_calendar_tuning(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        validate_policy(policy)


def frozen_spec(tmp_path, policy):
    dependencies = (
        "src/us_quant/claims_factor_guard.py",
        "config/claims-factor-guard.json",
        "src/us_quant/dollar_risk_guard.py",
        "config/dollar-risk-guard.json",
        "src/us_quant/sector_growth_balance.py",
        "config/sector-growth-balance.json",
        "src/us_quant/growth_factor_satellite.py",
        "config/growth-factor-satellite.json",
        "src/us_quant/multifactor_stability.py",
        "src/us_quant/macro_factor_tilt.py",
        "src/us_quant/financial_conditions_guard.py",
        "src/us_quant/jobless_claims_inputs.py",
        "src/us_quant/inflation_inputs.py",
        "config/jobless-claims-inputs.json",
    )
    frozen = {}
    for name in dependencies:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        frozen[name] = file_digest(destination)
    for name in (
        "data/dollar-information-source-20261011/verified-manifest.json",
        "data/claims-inputs-20261011/manifest.json",
        "evidence/jobless_claims_input_readiness_20261011.json",
        "data/claims-confirmation-context-20261011/context-verification.json",
        "reports/claims-confirmation-plan-20261011/availability-rejected-policy-v1.json",
        "reports/claims-confirmation-plan-20261011/availability-rejected-implementation-v1.py",
        "reports/claims-confirmation-plan-20261011/source-window-failure-v1.json",
        "data/macro-rate-access-20261010/verified-manifest.json",
        "data/macro-rate-access-20261010/DFII10-curl-response.csv",
    ):
        write_text_atomic(
            tmp_path / name, "synthetic seal,not source observations or portfolio evidence"
        )
        frozen[name] = file_digest(tmp_path / name)
    return {
        "id": CANDIDATE["id"],
        "configuration": CANDIDATE,
        "frozen_files": frozen,
        "eligible_not_before": policy["eligible_not_before"],
    }


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_seeded_replay_and_synthetic_accounting_retain_standby_sessions(
    market,
    policy,
    tmp_path,
    monkeypatch,
    cost,
    delay,
):
    source, _ = inputs(market)
    source.loc[pd.Timestamp("2016-10-31"), "claims_change"] = np.nan
    source.loc[pd.Timestamp("2016-10-31"), "claims_change_available"] = False
    monkeypatch.setattr("us_quant.claims_factor_guard.load_inputs", lambda data, p: (source, {}))
    spec = frozen_spec(tmp_path, policy)
    start, end = "2016-10-06", "2016-11-04"
    targets = registered_targets(spec, market, start, end, cost, delay, tmp_path, {})
    expected = seed_window(build_from_inputs(market, source, policy)[CANDIDATE["id"]], start)
    pd.testing.assert_frame_equal(targets, expected)
    assert targets.loc["2016-10-31"].isna().all()
    funded = simulate(
        market,
        targets,
        start,
        end,
        initial_capital=10000,
        cost_bps=cost,
        commission=1,
        delay=delay,
    )
    independent = independent_equity(
        market,
        targets,
        start,
        end,
        capital=10000,
        cost_bps=cost,
        commission=1,
        delay=delay,
    )
    assert funded.frame.index.equals(sessions(start, end))
    assert funded.frame["cash"].min() >= 0
    np.testing.assert_allclose(funded.frame["equity"], independent["equity"], rtol=0, atol=1e-8)
    incomplete = {
        **spec,
        "frozen_files": {
            name: digest
            for name, digest in spec["frozen_files"].items()
            if name != "data/claims-inputs-20261011/manifest.json"
        },
    }
    with pytest.raises(QuantError, match="helper dependency"):
        registered_targets(incomplete, market, start, end, cost, delay, tmp_path, {})


def test_actual_time_gate_cannot_be_bypassed_by_wrapper_or_generic_registration(
    policy, tmp_path, monkeypatch
):
    spec = frozen_spec(tmp_path, policy)
    now = pd.Timestamp("2026-10-11T16:00:00+08:00")
    with pytest.raises(QuantError, match="not eligible"):
        check_registration_eligibility(spec, tmp_path, now)
    monkeypatch.setattr("us_quant.research_program.readiness_blockers", lambda *args, **kwargs: [])
    program = ResearchProgram(
        tmp_path / "ledger.sqlite3",
        read_json(ROOT / "config/research-program.json"),
        root=tmp_path,
        create=True,
    )
    try:
        before = program.status()
        complete = {
            **spec,
            "data_scope": "factor_etf_portfolio",
            "factor_ids": [
                "price_momentum",
                "value_exposure",
                "quality_exposure",
                "low_volatility_exposure",
            ],
            "order_authority": False,
            "leveraged_products_allowed": False,
            "history_status": "exposed_history_not_independent_holdout",
        }
        with pytest.raises(QuantError, match="not eligible"):
            program.register_candidate(complete, {"data_scope": "factor_etf_portfolio"}, now=now)
        assert program.status() == before
    finally:
        program.close()


@pytest.mark.parametrize("malformed", [None, [], "frozen", {}])
def test_malformed_frozen_manifest_is_an_explicit_error_before_any_registration(
    tmp_path, malformed, monkeypatch
):
    monkeypatch.setattr("us_quant.research_program.readiness_blockers", lambda *args, **kwargs: [])
    program = ResearchProgram(
        tmp_path / "ledger.sqlite3",
        read_json(ROOT / "config/research-program.json"),
        root=tmp_path,
        create=True,
    )
    try:
        before = program.status()
        spec = {
            "id": CANDIDATE["id"],
            "frozen_files": malformed,
            "factor_ids": ["price_momentum", "value_exposure", "quality_exposure"],
            "order_authority": False,
            "leveraged_products_allowed": False,
            "history_status": "exposed_history_not_independent_holdout",
        }
        with pytest.raises(QuantError, match="frozen"):
            program.register_candidate(spec, {}, now=pd.Timestamp("2026-10-11T16:00:00+08:00"))
        assert program.status() == before
    finally:
        program.close()
