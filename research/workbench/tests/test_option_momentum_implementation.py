from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import implied_risk_execution as parent
from us_quant import option_momentum_implementation as study
from us_quant.bt_audit import independent_equity
from us_quant.config import QuantError
from us_quant.research_program import verified_etf_market
from us_quant.sector_growth_balance import build_targets as sector_targets
from us_quant.storage import file_digest, read_json
from us_quant.strategy_replay import registered_generator

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/option-momentum-implementation.json")


@pytest.fixture
def market(market_factory):
    return market_factory(
        "2015-08-10", "2026-10-05",
        ("SPY", "IEF", "GLD", "BIL", *study.FACTORS, "TLT", "QQQ", "XLK", "SOXX", "PDP"),
    )


def original(data):
    prior = study.MarketData(
        data.open.drop(columns="PDP"), data.close.drop(columns="PDP"),
        data.raw_close.drop(columns="PDP"), data.volume.drop(columns="PDP"), data.risk_free,
    )
    frame = sector_targets(prior, read_json(ROOT / "config/sector-growth-balance.json"))["four_factor_semiconductor_sector50"]
    frame["PDP"] = frame["MTUM"] * 0
    return frame.reindex(columns=data.close.columns)


def test_only_original_momentum_sleeve_moves_without_risk_budget_or_calendar_change(market, policy):
    baseline = original(market)
    outputs = study.build_from_original(market, baseline, policy)
    preserved = baseline.columns.drop(["MTUM", "PDP"])
    for candidate in study.CANDIDATES:
        frame = outputs[candidate["id"]]
        pd.testing.assert_frame_equal(frame[preserved], baseline[preserved])
        assert frame.isna().equals(baseline.isna()) and len(frame.dropna(how="all")) == 131
        np.testing.assert_allclose(frame["PDP"], baseline["MTUM"] * candidate["pdp_share_of_momentum"], equal_nan=True)
        np.testing.assert_allclose(frame["MTUM"], baseline["MTUM"] * (1 - candidate["pdp_share_of_momentum"]), equal_nan=True)
        np.testing.assert_allclose(frame.dropna(how="all").sum(axis=1), 0.98, rtol=0, atol=1e-12)


@pytest.mark.parametrize("problem", ["missing_month", "partial", "nonfinite", "negative", "already_pdp", "wrong_columns"])
def test_unknown_or_invalid_original_requests_are_fatal_not_filled_or_shortened(market, policy, problem):
    frame = original(market)
    day = frame.dropna(how="all").index[0]
    if problem == "missing_month":
        frame.loc[day] = np.nan
    elif problem == "partial":
        frame.loc[day, "PDP"] = np.nan
    elif problem == "nonfinite":
        frame.loc[day, "PDP"] = np.inf
    elif problem == "negative":
        frame.loc[day, "PDP"] = -1
    elif problem == "already_pdp":
        frame.loc[day, "PDP"] = 0.1
    else:
        frame = frame.drop(columns="PDP")
    with pytest.raises(QuantError):
        study.build_from_original(market, frame, policy)


@pytest.mark.parametrize("cost,delay", [(5.0, 1), (20.0, 2)])
def test_original_actual_holding_band_and_funded_bt_execution_are_reused(market, policy, cost, delay):
    desired = study.build_from_original(market, original(market), policy)[study.CANDIDATES[0]["id"]]
    result, issued, decisions = study.run(
        market, desired, study.CANDIDATES[0], policy, "2021-10-06", "2026-10-05", cost, delay,
    )
    old, previous_issued, previous_decisions = parent.run(
        market, desired, parent.CANDIDATES[1], read_json(ROOT / "config/implied-risk-execution.json"),
        "2021-10-06", "2026-10-05", cost, delay,
    )
    pd.testing.assert_frame_equal(result.frame, old.frame)
    pd.testing.assert_frame_equal(issued, previous_issued)
    assert decisions == previous_decisions and len(result.frame) == 1254
    independent = independent_equity(market, issued, "2021-10-06", "2026-10-05",
                                    capital=10000, cost_bps=cost, commission=1, delay=delay)
    np.testing.assert_allclose(result.frame["equity"], independent["equity"], rtol=0, atol=1e-8)


@pytest.mark.parametrize("field,value", [
    ("approved_round_configuration_cap", 12), ("approved_round_batch_cap", 6),
    ("new_configurations", 6), ("new_economic_factor_definitions", 2),
    ("actual_weight_trade_band", 0.05), ("replacement_symbol", "SPHQ"),
    ("data_start", "2016-07-01"),
])
def test_user_budget_and_source_scope_cannot_be_silently_relabelled(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        study.validate(policy)


def test_arbitrary_new_momentum_mix_is_not_a_parameter_grid(policy):
    policy["candidates"][1]["pdp_share_of_momentum"] = 0.75
    with pytest.raises(QuantError):
        study.validate(policy)


def test_unique_generator_requires_complete_original_allocation_execution_and_pdp_graph():
    candidate = study.CANDIDATES[0]
    spec = {
        "id": candidate["id"], "configuration": candidate,
        "frozen_files": {"src/us_quant/option_momentum_implementation.py": "0" * 64,
                         "config/option-momentum-implementation.json": "0" * 64},
    }
    with pytest.raises(QuantError, match="frozen helper dependency"):
        registered_generator(spec, ROOT)


def test_adapter_must_bind_exact_momentum_manifest_before_loader(policy, monkeypatch):
    invoked = False

    def load(_policy, _root):
        nonlocal invoked
        invoked = True
        return object()

    monkeypatch.setattr(study, "verified_market", load)
    readiness = {
        "verified_etf_source": {
            "adapter": "option_momentum_implementation_20261013",
            "policy": "config/option-momentum-implementation.json",
            "policy_sha256": file_digest(ROOT / "config/option-momentum-implementation.json"),
            "factor_manifest": policy["momentum_manifest"]["path"],
            "factor_manifest_sha256": "0" * 64,
        },
    }
    with pytest.raises(QuantError, match="exact original actual-fund"):
        verified_etf_market(readiness, ROOT)
    assert not invoked
    readiness["verified_etf_source"]["factor_manifest_sha256"] = policy["momentum_manifest"]["sha256"]
    monkeypatch.setattr("us_quant.research_program.safe_file", lambda root, relative, expected=None: root / relative)
    assert verified_etf_market(readiness, ROOT) is not None and invoked
