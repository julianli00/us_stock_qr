from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from us_quant import lagged_spy_price_impact_admission_v1 as review
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.storage import digest_json, read_json


@pytest.fixture
def case(monkeypatch):
    policy, protocol = read_json(review.POLICY), read_json(review.base.PROTOCOL)
    dates = sessions("2015-08-10", "2026-10-05")
    close, risk = (
        pd.DataFrame(100.0, dates, review.law.UNIVERSE),
        pd.Series(0.0, dates, name="risk_free"),
    )
    targets = pd.DataFrame(np.nan, dates, review.law.UNIVERSE)
    number = 0
    for position, day in enumerate(dates):
        if position >= 274 and is_month_end(day):
            targets.loc[day] = 0.0
            selected = list(review.law.CORE) if number % 2 == 0 else ["BIL"]
            targets.loc[day, selected] = 0.98 / len(selected)
            number += 1
    active, weights = targets.dropna(how="all"), targets.ffill()
    frozen = {
        "study_id": "synthetic_source",
        "order_authority": False,
        "decision_information_clock": review.law.CLOCK,
    }
    reg = {
        "sequence": 1,
        "at": "2026-11-05T10:00:00Z",
        "sha256": "a" * 64,
        "body": {"candidate": review.law.CANDIDATE, "frozen": frozen},
    }
    publication = {
        "sequence": 2,
        "at": "2026-11-05T10:01:00Z",
        "sha256": "b" * 64,
        "body": {"verification": {"registration_packet": {"path": "packet", "sha256": "c" * 64}}},
    }

    def reference(name):
        return {"path": name, "sha256": "d" * 64}

    native = {"targets": reference("targets"), "proof": reference("proof")}
    paths, receipts, frames, starts, ends = {}, {}, {}, [], []
    for window in protocol["windows"]:
        days = sessions(window["start"], window["end"])
        for scenario in ("base", "stress"):
            key = f"{window['years']}y-{scenario}"
            paths[key] = reference(key)
            receipts[key] = {
                "execution": {"minimum_cash_usd": 200.0, "maximum_gross_exposure": 0.98},
                "independent_bt_max_equity_difference_usd": 0.0,
                "broker_orders": 0,
                "artifacts": [reference(key + "/account.csv"), reference(key + "/weights.csv")],
            }
            frames[key + "/account.csv"] = pd.DataFrame(
                {
                    "equity": 10000.0,
                    "return": 0.0,
                    "risk_free": 0.0,
                    "cash": 200.0,
                    "gross_exposure": 0.98,
                    "cost": 0.0,
                    "orders": 0.0,
                },
                index=days,
            )
            frames[key + "/weights.csv"] = weights.loc[days].copy()
            number = len(starts)
            starts.append(
                {
                    "sequence": 3 + 2 * number,
                    "at": f"2026-11-05T10:02:{number*2:02d}Z",
                    "body": {"path_key": key},
                }
            )
            ends.append(
                {
                    "sequence": 4 + 2 * number,
                    "at": f"2026-11-05T10:02:{number*2+1:02d}Z",
                    "body": {"path_key": key, "receipt": paths[key]},
                }
            )
    result = {
        "frozen": frozen,
        "publication": publication["body"]["verification"],
        "review": {
            "status": "rejected",
            "all_four_paths_same_candidate_and_frozen_parameters": True,
            "order_authority": False,
            "paths": paths,
        },
        "paths": receipts,
        "native_source_proof": native,
    }
    proof_event = {
        "sequence": 11,
        "at": "2026-11-05T10:03:00Z",
        "body": {"native_source_proof": native},
    }
    review_event = {"sequence": 12, "at": "2026-11-05T10:04:00Z", "body": result["review"]}
    proof = {
        "frozen_specification_sha256": digest_json(frozen),
        "publication_event_sha256": publication["sha256"],
        "source_start": reference("started"),
        "checked_at": "2026-11-05T10:01:02Z",
        "real_native_indicators_first_generated_after_actual_publication": True,
        "real_native_targets_first_generated_after_actual_publication": True,
        "all_independent_lagged_price_impact_targets_exact": True,
        "full_MarketData_volume_Interface_preserved": True,
        "decision_day_or_future_input_used": False,
        "source_price_volume_binary64_inputs_checked": len(active) * 820,
        "whole_native_prehistory_retained": True,
        "synthetic_initial_recommendation_added": False,
        "native_month_end_opinions": len(active),
        "prefix_truncations": [
            str(active.index[p].date())
            for p in sorted({0, 6, 12, len(active) // 2, len(active) - 1})
        ],
        "retained_input_resolved_trace": reference("trace"),
    }
    objects = {
        Path("results"): result,
        Path("proof"): proof,
        Path("started"): {
            "publication_event_sha256": publication["sha256"],
            "started_at": "2026-11-05T10:01:01Z",
        },
        review.base.PROTOCOL: protocol,
    }
    events = {
        "registered": [reg],
        "review": [review_event],
        "target_proof": [proof_event],
        "path_started": starts,
        "path_completed": ends,
    }
    fake = SimpleNamespace(
        inputs=lambda: (SimpleNamespace(close=close, risk_free=risk), frozen, {}),
        checked_packet=lambda f: {
            "entries": [{"registered_at": reg["at"], "registered_event_sha256": reg["sha256"]}]
        },
        RESULTS=Path("results"),
        PACKET=Path("packet"),
        events=lambda kind: events[kind],
        publication_fence=lambda: publication,
        reference=lambda p: publication["body"]["verification"]["registration_packet"]
        if p == Path("packet")
        else reference("results"),
        source_frame=lambda value: targets.copy(),
    )
    fake_base = SimpleNamespace(
        PATH_KEYS=set(paths),
        PROTOCOL=review.base.PROTOCOL,
        checked_reference=lambda ref: Path(ref["path"]),
        verify_path=lambda ref, *args: receipts[ref["path"]],
    )
    monkeypatch.setattr(review, "source", fake)
    monkeypatch.setattr(review, "base", fake_base)
    monkeypatch.setattr(review, "read_json", lambda p: objects[p])
    monkeypatch.setattr(review, "_frame", lambda ref: frames[ref["path"]].copy())
    return SimpleNamespace(
        policy=policy,
        result=result,
        proof=proof,
        frames=frames,
        objects=objects,
        risk=risk,
        starts=starts,
        ends=ends,
        events=events,
        targets=targets,
    )


def test_rejected_source_has_real_styles_BIL_and_separate_availability_maturity(case):
    expert, evidence = review.inspect_source(case.policy)
    assert evidence["standalone_financial_verdict_preserved"] == "rejected"
    assert all(len(value) == 6 for value in evidence["actual_style_families_by_path"].values())
    assert all(value == ["BIL"] for value in evidence["actual_defensive_assets_by_path"].values())
    assert [row["NET_observations"] for row in evidence["NET_maturity_boundary_observations"]] == [
        0,
        125,
        126,
    ]
    assert expert.identity.mechanism_group == "lagged_market_price_impact_participation"
    assert expert.observe(case.targets.index[273], 126).target is None
    assert evidence["current_global_pool_or_roster_changed"] is False


@pytest.mark.parametrize(
    "failure",
    ["cash", "gross", "missing_day", "RF", "families", "NaN", "stray_asset", "missing_defense"],
)
def test_invalid_funding_dates_RF_or_actual_asset_coverage_fails(case, failure):
    key = "5y-stress"
    account, weights = case.frames[key + "/account.csv"], case.frames[key + "/weights.csv"]
    if failure == "cash":
        account.iloc[0, account.columns.get_loc("cash")] = -1
    elif failure == "gross":
        weights.iloc[0] = 0
        weights.iloc[0, 0] = 1.01
    elif failure == "missing_day":
        case.frames[key + "/account.csv"] = account.iloc[1:]
    elif failure == "RF":
        account.iloc[0, account.columns.get_loc("risk_free")] = 1e-10
    elif failure == "families":
        weights.loc[:, list(review.law.CORE)] = 0
        weights.loc[:, list(review.law.CORE[:2])] = 0.1
    elif failure == "stray_asset":
        weights.loc[:, "SPY"] = 0.01
    elif failure == "missing_defense":
        weights.loc[:, "BIL"] = 0
    else:
        account.iloc[0, account.columns.get_loc("cash")] = np.nan
    with pytest.raises(QuantError):
        review.inspect_source(case.policy)


@pytest.mark.parametrize(
    "failure",
    ["target_before_publication", "path_before_fence", "missing_path", "missing_opinion", "warmup"],
)
def test_chronology_native_warmup_and_completeness_are_not_relabelled(case, failure):
    if failure == "target_before_publication":
        case.objects[Path("started")]["started_at"] = "2026-11-05T10:00:00Z"
    elif failure == "path_before_fence":
        case.starts[0]["sequence"] = 1
    elif failure == "missing_path":
        case.ends.pop()
    elif failure == "missing_opinion":
        case.targets.loc[case.targets.dropna(how="all").index[20]] = np.nan
    else:
        case.targets.iloc[20] = case.targets.dropna(how="all").iloc[0]
    with pytest.raises(QuantError):
        review.inspect_source(case.policy)


@pytest.mark.parametrize(
    "key,value",
    [
        ("full_MarketData_volume_Interface_preserved", False),
        ("decision_day_or_future_input_used", True),
        ("source_price_volume_binary64_inputs_checked", True),
        ("source_price_volume_binary64_inputs_checked", 0),
        ("real_native_indicators_first_generated_after_actual_publication", False),
    ],
)
def test_saved_full_OHLCV_and_lagged_input_proof_cannot_be_weakened(case, key, value):
    case.proof[key] = value
    with pytest.raises(QuantError):
        review.inspect_source(case.policy)


@pytest.mark.parametrize(
    "key,value",
    [
        ("minimum_actual_style_families_per_path", 2),
        ("existing_independent_BT_tolerance_usd", 0.01),
        ("financial_performance_is_technical_admission_gate", True),
        ("source_packet_schema", True),
        ("global_pool_promotion", True),
        ("required_actual_defensive_assets", []),
        ("source_price_warmup_position", 0),
    ],
)
def test_original_technical_standards_and_source_scope_cannot_be_softened(case, key, value):
    bad = deepcopy(case.policy)
    bad[key] = value
    with pytest.raises(QuantError):
        review.validate(bad)


def admission(case):
    _, evidence = review.inspect_source(case.policy)
    case.objects[Path("admission")] = {
        "technical_expert_admitted": True,
        "global_pool_changed": False,
        "new_strategy_evaluations": 0,
        "order_authority": False,
        "review_implementation_freeze": {"path": "freeze"},
        "evidence": evidence,
        "admitted_at": "2026-11-05T11:00:00Z",
    }
    inputs = [{"path": name} for name in review.OWN]
    case.objects[Path("freeze")] = {
        "created_at": "2026-11-05T10:06:00Z",
        "implementation_and_test_inputs": inputs,
        "complete_dependency_inputs": inputs,
    }
    case.objects[review.POLICY] = case.policy
    return {"path": "admission"}


def test_consumer_checks_all_NET_RF_and_keeps_frozen_helper_namespace(case):
    reference = admission(case)
    original_globals = review._consume_saved.__globals__
    old_id = original_globals["law"].ID
    expert, evidence = review.load_admitted_expert(reference, review.law.UNIVERSE, case.risk)
    assert expert.identity.identifier == review.law.ID
    assert evidence["actual_expert_admitted_at"] == "2026-11-05T11:00:00Z"
    assert original_globals["law"].ID == old_id == "fixed_four_sleeve_rebalance_v1"
    changed = case.risk.copy()
    changed.loc["2017-04-06"] = 1e-10
    with pytest.raises(QuantError, match="consumer assessment RF"):
        review.load_admitted_expert(reference, review.law.UNIVERSE, changed)
    with pytest.raises(QuantError):
        review.load_admitted_expert(reference, tuple(reversed(review.law.UNIVERSE)), case.risk)


@pytest.mark.parametrize(
    "key,value",
    [
        ("technical_expert_admitted", False),
        ("global_pool_changed", True),
        ("new_strategy_evaluations", False),
        ("order_authority", True),
    ],
)
def test_consumer_rejects_laundered_admission_or_trading_authority(case, key, value):
    reference = admission(case)
    case.objects[Path("admission")][key] = value
    with pytest.raises(QuantError):
        review.load_admitted_expert(reference, review.law.UNIVERSE, case.risk)


@pytest.mark.parametrize("failure", ["backdated", "before_freeze", "unfrozen", "RF_crop"])
def test_consumer_rejects_backdating_unfrozen_code_and_cropped_RF(case, failure):
    reference = admission(case)
    risk = case.risk
    if failure == "backdated":
        case.objects[Path("admission")]["admitted_at"] = "2016-01-01T00:00:00Z"
    elif failure == "before_freeze":
        case.objects[Path("admission")]["admitted_at"] = "2026-11-05T10:05:00Z"
    elif failure == "unfrozen":
        case.objects[Path("freeze")]["complete_dependency_inputs"] = []
    else:
        risk = risk.loc["2021-01-01":]
    with pytest.raises(QuantError):
        review.load_admitted_expert(reference, review.law.UNIVERSE, risk)
