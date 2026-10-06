from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.backtest import simulate
from us_quant.bt_audit import independent_equity
from us_quant.calendar import is_month_end, previous_session, sessions
from us_quant.config import QuantError
from us_quant.dual_horizon import (
    build_signals,
    gates,
    load_protocol,
    month_end_weights,
    seed_window,
    verify_registration,
    verify_sources,
)
from us_quant.storage import digest_json, file_digest, write_json, write_text_atomic


@pytest.fixture
def protocol():
    return load_protocol(Path(__file__).parents[1] / "config/dual-horizon.json")


@pytest.fixture
def month_prices(protocol):
    dates = sessions("2020-01-01", "2021-01-31")
    ends = pd.DatetimeIndex([day for day in dates if is_month_end(day)])
    return pd.DataFrame(100.0, index=ends, columns=protocol.symbols)


def select_rule(protocol, name):
    return next(rule for rule in protocol.candidates if rule.id == name)


def marked_last(frame, values):
    result = frame.copy()
    for symbol, value in values.items():
        result.loc[result.index[-1], symbol] = value
    return result


def test_same_endpoint_exact_ten_and_five_years(protocol):
    windows = protocol.windows()
    assert [window["years"] for window in windows] == [10, 5]
    assert [window["first_return_session"] for window in windows] == ["2016-10-06", "2021-10-06"]
    assert [window["sessions"] for window in windows] == [2512, 1254]
    assert {window["last_session"] for window in windows} == {"2026-10-05"}


@pytest.mark.parametrize(
    "change",
    [
        {"horizons_years": (5,)},
        {"cagr_strictly_above": 0.10},
        {"drawdown_at_most": 0.30},
        {"cash_reserve": 0},
        {"capital_usd": 1000000},
        {"primary_benchmark": "SHY"},
        {"prior_disclosed_trials": 0},
        {"cost_bps_per_side": 0},
        {"stress_additional_delay_sessions": 0},
    ],
)
def test_dual_horizon_protocol_cannot_relax_the_objective(protocol, change):
    with pytest.raises(QuantError):
        replace(protocol, **change).validate()


def test_vaa_switches_on_any_nonpositive_canary(protocol, month_prices):
    rule = select_rule(protocol, "vaa_g4")
    prices = marked_last(
        month_prices,
        {
            "SPY": 140,
            "EFA": 130,
            "EEM": 120,
            "AGG": 110,
            "SHY": 101,
            "IEF": 103,
            "LQD": 102,
        },
    )
    assert month_end_weights(prices, rule, protocol)["SPY"] == pytest.approx(0.98)
    prices.iloc[-1, prices.columns.get_loc("AGG")] = 100
    weights = month_end_weights(prices, rule, protocol)
    assert weights["IEF"] == pytest.approx(0.98) and weights["SPY"] == 0


def test_daa_canary_breadth_controls_half_and_full_defense(protocol, month_prices):
    rule = select_rule(protocol, "daa_g12")
    prices = marked_last(
        month_prices,
        {
            "SPY": 140,
            "QQQ": 150,
            "EEM": 90,
            "AGG": 110,
            "SHY": 160,
            "GLD": 120,
            "IWM": 130,
            "VGK": 125,
            "VNQ": 115,
        },
    )
    weights = month_end_weights(prices, rule, protocol)
    assert weights["SHY"] == pytest.approx(0.49)
    assert weights.sum() == pytest.approx(0.98)
    assert (weights > 0).sum() == 7
    prices.iloc[-1, prices.columns.get_loc("AGG")] = 90
    weights = month_end_weights(prices, rule, protocol)
    assert weights["SHY"] == pytest.approx(0.98) and (weights > 0).sum() == 1


def test_baa_defensive_assets_below_bills_are_replaced_not_borrowed(protocol, month_prices):
    rule = select_rule(protocol, "baa_aggressive")
    prices = marked_last(
        month_prices,
        {
            "SPY": 90,
            "EFA": 105,
            "EEM": 110,
            "AGG": 95,
            "BIL": 130,
            "TIP": 120,
            "DBC": 110,
        },
    )
    weights = month_end_weights(prices, rule, protocol)
    assert weights["BIL"] == pytest.approx(0.98)
    assert (weights > 0).sum() == 1


def test_baa_aggressive_and_balanced_use_same_canary_different_fixed_counts(protocol, month_prices):
    prices = marked_last(
        month_prices,
        {
            "SPY": 120,
            "EFA": 121,
            "EEM": 125,
            "AGG": 110,
            "QQQ": 150,
            "IWM": 130,
            "GLD": 132,
            "VNQ": 129,
            "VGK": 128,
        },
    )
    aggressive = month_end_weights(prices, select_rule(protocol, "baa_aggressive"), protocol)
    balanced = month_end_weights(prices, select_rule(protocol, "baa_balanced"), protocol)
    assert aggressive["QQQ"] == pytest.approx(0.98)
    assert (balanced > 0).sum() == 6
    assert balanced.max() == pytest.approx(0.98 / 6)


def test_haa_tip_canary_and_negative_selection_replacement(protocol, month_prices):
    rule = select_rule(protocol, "haa")
    prices = marked_last(month_prices, {"TIP": 90, "IEF": 110, "BIL": 102})
    assert month_end_weights(prices, rule, protocol)["IEF"] == pytest.approx(0.98)
    prices = marked_last(month_prices, {"TIP": 110, "BIL": 101})
    assert month_end_weights(prices, rule, protocol)["BIL"] == pytest.approx(0.98)


def test_gem_tests_us_excess_momentum_before_picking_foreign_winner(protocol, month_prices):
    rule = select_rule(protocol, "gem_developed_proxy")
    prices = marked_last(month_prices, {"SPY": 99, "EFA": 200, "BIL": 101})
    assert month_end_weights(prices, rule, protocol)["AGG"] == pytest.approx(0.98)
    prices.iloc[-1, prices.columns.get_loc("SPY")] = 120
    assert month_end_weights(prices, rule, protocol)["EFA"] == pytest.approx(0.98)


def test_adm_uses_original_long_treasury_defense_not_a_hindsight_tip_choice(protocol, month_prices):
    rule = select_rule(protocol, "adm_original_etf_proxy")
    prices = marked_last(month_prices, {"SPY": 90, "SCZ": 80, "TIP": 200, "TLT": 80})
    assert month_end_weights(prices, rule, protocol)["TLT"] == pytest.approx(0.98)
    prices.iloc[-1, prices.columns.get_loc("SCZ")] = 120
    assert month_end_weights(prices, rule, protocol)["SCZ"] == pytest.approx(0.98)


def test_monthly_rules_are_causal_actual_calendar_and_constant_under_price_rescaling(
    protocol, market_factory
):
    prices = market_factory("2014-01-02", "2017-12-29", protocol.symbols).close
    full = build_signals(prices, protocol)
    prefix_end = pd.Timestamp("2016-07-14")
    prefix = build_signals(prices.loc[:prefix_end], protocol)
    changed = prices.copy()
    later = changed.index > prefix_end
    changed.loc[later, "QQQ"] *= np.linspace(1, 1.4, later.sum())
    perturbed = build_signals(changed, protocol)
    scaled = build_signals(prices * np.arange(1, len(prices.columns) + 1), protocol)
    for rule in protocol.candidates:
        expected = full[rule.id]
        pd.testing.assert_frame_equal(expected.loc[:prefix_end], prefix[rule.id])
        pd.testing.assert_frame_equal(
            expected.loc[:prefix_end], perturbed[rule.id].loc[:prefix_end]
        )
        pd.testing.assert_frame_equal(expected, scaled[rule.id], atol=1e-12, rtol=1e-12)
        assert expected.loc["2016-07-14"].isna().all()
        assert all(is_month_end(day) for day in expected.dropna(how="all").index)
        assert (expected.dropna().sum(axis=1).round(12) == 0.98).all()
    assert full["vaa_g4"].loc["2017-12-29"].notna().all()


def test_ensemble_is_fixed_equal_weight_not_selected_after_results(protocol, market_factory):
    prices = market_factory("2014-01-02", "2016-12-30", protocol.symbols).close
    results = build_signals(prices, protocol)
    for day in results["fixed_seven_model_ensemble"].dropna().index:
        expected = sum(results[rule.id].loc[day] for rule in protocol.candidates[:-1]) / 7
        pd.testing.assert_series_equal(expected, results["fixed_seven_model_ensemble"].loc[day])


def test_midmonth_fresh_account_uses_known_previous_month_signal(protocol, market_factory):
    prices = market_factory("2014-01-02", "2017-12-29", protocol.symbols).close
    signals = build_signals(prices, protocol)["vaa_g4"]
    seeded = seed_window(signals, "2016-10-06")
    pd.testing.assert_series_equal(
        seeded.loc["2016-10-05"], signals.loc["2016-09-30"], check_names=False
    )
    pd.testing.assert_frame_equal(seeded.loc["2016-10-06":], signals.loc["2016-10-06":])
    assert previous_session(pd.Timestamp("2016-10-06")) == pd.Timestamp("2016-10-05")


@pytest.mark.parametrize("delay", [1, 2])
@pytest.mark.parametrize("cost,commission", [(0, 0), (5, 1), (20, 1)])
def test_independent_bt_accounts_for_gaps_costs_rotation_and_cash(
    market_factory, delay, cost, commission
):
    data = market_factory("2021-01-04", "2021-05-28", ("SPY", "QQQ", "IEF"))
    signal = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    signal.iloc[0] = [0.98, 0, 0]
    signal.iloc[10] = [0, 0.98, 0]
    signal.iloc[20] = [0.3, 0.3, 0.3]
    signal.iloc[35] = [0, 0, 0]
    signal.iloc[50] = [0, 0, 1]
    first, end = str(data.close.index[0].date()), str(data.close.index[-1].date())
    own = simulate(
        data,
        signal,
        first,
        end,
        initial_capital=10000,
        cost_bps=cost,
        commission=commission,
        delay=delay,
    )
    external = independent_equity(
        data, signal, first, end, capital=10000, cost_bps=cost, commission=commission, delay=delay
    )
    np.testing.assert_allclose(external.equity, own.frame.equity, rtol=1e-10, atol=1e-6)
    np.testing.assert_allclose(external["return"], own.frame["return"], rtol=1e-9, atol=1e-12)


def test_acceptance_does_not_use_one_good_window_to_hide_the_other(protocol):
    benchmark = {"start": "2021-10-06", "end": "2026-10-05", "sessions": 1254, "cagr": 0.10}
    passing = {**benchmark, "cagr": 0.25, "sharpe": 1.3, "max_drawdown": 0.10}
    assert all(gates(passing, benchmark, protocol).values())
    failing = {**passing, "cagr": 0.20, "sharpe": 1, "max_drawdown": 0.16}
    assert gates(failing, benchmark, protocol) == {
        "cagr_above_20pct": False,
        "sharpe_above_1": False,
        "max_drawdown_at_most_15pct": False,
        "beats_spy": True,
    }
    with pytest.raises(QuantError, match="different market intervals"):
        gates(passing, {**benchmark, "start": "2020-01-01"}, protocol)


def test_preregistration_detects_changed_candidate_windows_and_trial_counts(protocol, tmp_path):
    path = tmp_path / "registration.json"
    record = {
        "protocol_sha256": digest_json(asdict(protocol)),
        "candidate_ids": [item.id for item in protocol.candidates],
        "windows": protocol.windows(),
        "same_rule_required_in_both_windows": True,
        "prior_disclosed_trials": 34,
        "new_trials": 8,
        "cumulative_trials_after_round": 42,
    }
    write_json(path, record)
    verify_registration(protocol, path)
    write_json(path, {**record, "cumulative_trials_after_round": 8})
    with pytest.raises(QuantError, match="trial count changed"):
        verify_registration(protocol, path)


def test_upstream_license_and_code_evidence_cannot_be_substituted(tmp_path):
    license_path = tmp_path / "LICENSE"
    write_text_atomic(license_path, "MIT License\nSynthetic attribution fixture\n")
    manifest = {
        "items": [
            {"license": "MIT", "files": [{"path": "LICENSE", "sha256": file_digest(license_path)}]}
        ],
    }
    path = tmp_path / "manifest.json"
    write_json(path, manifest)
    registration = {"source_manifest_sha256": file_digest(path)}
    verify_sources(tmp_path, registration)
    write_text_atomic(license_path, "Changed license")
    with pytest.raises(QuantError, match="license notice changed"):
        verify_sources(tmp_path, registration)
