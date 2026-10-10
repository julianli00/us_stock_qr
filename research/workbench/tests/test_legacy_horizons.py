from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from us_quant.config import QuantError, load_config
from us_quant.dual_horizon import load_protocol
from us_quant.evolution import seed_genome
from us_quant.legacy_horizons import (
    INPUTS,
    cases,
    legacy_signals,
    register_comparison,
    subset,
    verify_comparison,
)
from us_quant.storage import digest_json, read_json, write_json


@pytest.fixture
def old_project(tmp_path):
    root = Path(__file__).parents[1]
    for path in INPUTS:
        if path.startswith("config/"):
            write_json(tmp_path / path, read_json(root / path))
    base = load_config(tmp_path / "config/research.json")
    policy = read_json(tmp_path / "config/evolution.json")
    seed = seed_genome(base, policy)
    registered = []
    for item in policy["mutations"]:
        genome = {**seed, item["field"]: item["value"]}
        registered.append({"id": "evo_" + digest_json(genome)[:12], "genome": genome})
    write_json(tmp_path / INPUTS[-1], {"candidates": registered})
    return tmp_path


@pytest.fixture
def protocol():
    return load_protocol(Path(__file__).parents[1] / "config/dual-horizon.json")


def test_all_original_configurations_are_present_without_new_hypotheses(
    old_project, protocol, tmp_path
):
    original = cases(old_project)
    assert len(original) == 34 and len({item.id for item in original}) == 34
    assert sum(item.family == "expanded_family" for item in original) == 8
    assert sum(item.family == "daily_regime" for item in original) == 6
    assert sum(item.family == "evolution_monthly" for item in original) == 8
    path = tmp_path / "comparison.json"
    record = register_comparison(old_project, protocol, path)
    assert record["new_hypotheses"] == 0
    assert len(record["required_supplemental_symbols"]) == 11
    assert {"QLD", "SSO", "XLK"} <= set(record["required_supplemental_symbols"])
    verify_comparison(old_project, protocol, path)
    with pytest.raises(QuantError, match="overwrite"):
        register_comparison(old_project, protocol, path)


def test_previously_rejected_rules_may_not_be_removed_from_comparison(
    old_project, protocol, tmp_path
):
    path = tmp_path / "comparison.json"
    record = register_comparison(old_project, protocol, path)
    record["case_ids"].pop()
    write_json(path, record)
    with pytest.raises(QuantError, match="comparison plan changed"):
        verify_comparison(old_project, protocol, path)


def test_source_parameter_edits_invalidate_comparison_registration(old_project, protocol, tmp_path):
    path = tmp_path / "comparison.json"
    register_comparison(old_project, protocol, path)
    raw = read_json(old_project / "config/research.json")
    raw["trend_lookback"] = 150
    write_json(old_project / "config/research.json", raw)
    with pytest.raises(QuantError, match="changed"):
        verify_comparison(old_project, protocol, path)


def test_legacy_signal_adapters_preserve_universes_and_rules(old_project, protocol, market_factory):
    original = cases(old_project)
    symbols = tuple(
        sorted(
            set(protocol.symbols) | {symbol for item in original for symbol in item.config.symbols}
        )
    )
    data = market_factory("2014-01-02", "2016-12-30", symbols)
    signals = legacy_signals(old_project, data)
    assert set(signals) == {item.id for item in original}
    for item in original:
        market, signal = signals[item.id]
        assert tuple(market.close.columns) == item.config.symbols
        assert signal.columns.equals(market.close.columns)
        assert signal.index.equals(data.close.index)
        known = signal.dropna(how="all")
        assert not known.empty and (known >= 0).all().all()
        assert (known.sum(axis=1) <= 0.98 + 1e-10).all()
    control = (
        "evo_"
        + digest_json(
            seed_genome(
                load_config(old_project / "config/research.json"),
                read_json(old_project / "config/evolution.json"),
            )
        )[:12]
    )
    pd.testing.assert_frame_equal(signals[control][1], signals["cross_asset_top3_12"][1])
    trimmed = subset(data, ("SPY", "QQQ"))
    assert trimmed.risk_free.equals(data.risk_free)
