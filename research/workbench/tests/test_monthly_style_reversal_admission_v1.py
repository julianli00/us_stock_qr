from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from us_quant import monthly_style_reversal_admission_v1 as review
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.storage import digest_json, read_json


@pytest.fixture
def case(monkeypatch):
    policy = read_json(review.POLICY)
    protocol = read_json(review.base.PROTOCOL)
    dates = sessions("2015-08-10", "2026-10-05")
    close = pd.DataFrame(100.0, dates, review.law.UNIVERSE)
    risk = pd.Series(0.0, dates, name="risk_free")
    targets = pd.DataFrame(np.nan, dates, review.law.UNIVERSE)
    for position, day in enumerate(dates):
        if position >= 21 and is_month_end(day):
            targets.loc[day] = 0.0
            selected = [review.law.CORE[day.month % 6], review.law.CORE[(day.month + 1) % 6]]
            targets.loc[day, selected] = 0.49
    active = targets.dropna(how="all")
    weights = targets.ffill()
    frozen = {
        "study_id": "synthetic_source",
        "order_authority": False,
        "decision_information_clock": review.law.CLOCK,
    }
    reg = {
        "sequence": 1,
        "at": "2026-11-05T06:00:00Z",
        "sha256": "a" * 64,
        "body": {"candidate": review.law.CANDIDATE, "frozen": frozen},
    }
    publication = {
        "sequence": 2,
        "at": "2026-11-05T06:01:00Z",
        "sha256": "b" * 64,
        "body": {"verification": {"registration_packet": {"path": "packet", "sha256": "c" * 64}}},
    }

    def reference(name):
        return {"path": name, "sha256": "d" * 64}

    source = {"targets": reference("targets"), "proof": reference("proof")}
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
                    "at": f"2026-11-05T06:02:{number*2:02d}Z",
                    "body": {"path_key": key},
                }
            )
            ends.append(
                {
                    "sequence": 4 + 2 * number,
                    "at": f"2026-11-05T06:02:{number*2+1:02d}Z",
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
        "native_source_proof": source,
    }
    proof_event = {
        "sequence": 11,
        "at": "2026-11-05T06:03:00Z",
        "body": {"native_source_proof": source},
    }
    review_event = {"sequence": 12, "at": "2026-11-05T06:04:00Z", "body": result["review"]}
    proof = {
        "frozen_specification_sha256": digest_json(frozen),
        "publication_event_sha256": publication["sha256"],
        "source_start": reference("started"),
        "checked_at": "2026-11-05T06:01:02Z",
        "real_native_targets_first_generated_after_actual_publication": True,
        "all_independent_reversal_targets_exact": True,
        "whole_native_prehistory_retained": True,
        "synthetic_initial_recommendation_added": False,
        "native_month_end_opinions": len(active),
    }
    objects = {
        Path("results"): result,
        Path("proof"): proof,
        Path("started"): {
            "publication_event_sha256": publication["sha256"],
            "started_at": "2026-11-05T06:01:01Z",
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
        source_frame=lambda native: targets.copy(),
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


def test_financial_rejection_does_not_prevent_complete_technical_inspection(case):
    expert, evidence = review.inspect_source(case.policy)
    assert evidence["standalone_financial_verdict_preserved"] == "rejected"
    assert evidence["native_RF_exact_common"]
    assert all(len(value) == 6 for value in evidence["actual_style_families_by_path"].values())
    assert [row["NET_observations"] for row in evidence["NET_maturity_boundary_observations"]] == [
        0,
        125,
        126,
    ]
    assert expert.identity.mechanism_group == "monthly_cross_sectional_style_reversal"
    assert evidence["current_global_pool_or_roster_changed"] is False


@pytest.mark.parametrize("failure", ["cash", "gross", "missing_day", "RF", "families", "NaN"])
def test_bad_funding_dates_RF_or_real_family_coverage_is_not_laundered(case, failure):
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
        weights.loc[:, :] = 0.0
        weights.loc[:, list(review.law.CORE[:2])] = 0.49
    else:
        account.iloc[0, account.columns.get_loc("cash")] = np.nan
    with pytest.raises(QuantError):
        review.inspect_source(case.policy)


@pytest.mark.parametrize(
    "failure", ["target_before_publication", "path_before_fence", "missing_path", "missing_opinion"]
)
def test_source_and_account_chronology_or_completeness_cannot_be_relabelled(case, failure):
    if failure == "target_before_publication":
        case.objects[Path("started")]["started_at"] = "2026-11-05T06:00:00Z"
    elif failure == "path_before_fence":
        case.starts[0]["sequence"] = 1
    elif failure == "missing_path":
        case.ends.pop()
    else:
        day = case.targets.dropna(how="all").index[20]
        case.targets.loc[day] = np.nan
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
    ],
)
def test_unchanged_technical_standards_cannot_be_softened_or_tied_to_profit(case, key, value):
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
        "admitted_at": "2026-11-05T07:00:00Z",
    }
    case.objects[Path("freeze")] = {
        "implementation_and_test_inputs": [
            {"path": name}
            for name in (
                "config/monthly-style-reversal-admission-v1.json",
                "src/us_quant/monthly_style_reversal_admission_v1.py",
                "tests/test_monthly_style_reversal_admission_v1.py",
                "MONTHLY_STYLE_REVERSAL_ADMISSION_V1.md",
            )
        ]
    }
    case.objects[review.POLICY] = case.policy
    return {"path": "admission"}


def test_admitted_consumer_checks_the_entire_NET_RF_history_not_only_last126(case):
    reference = admission(case)
    expert, evidence = review.load_admitted_expert(reference, review.law.UNIVERSE, case.risk)
    assert expert.identity.identifier == review.law.ID
    assert evidence["actual_expert_admitted_at"] == "2026-11-05T07:00:00Z"
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
def test_no_admission_or_qualification_trading_laundering_in_consumer(case, key, value):
    reference = admission(case)
    case.objects[Path("admission")][key] = value
    with pytest.raises(QuantError):
        review.load_admitted_expert(reference, review.law.UNIVERSE, case.risk)


@pytest.mark.parametrize("failure", ["backdated", "unfrozen", "RF_crop"])
def test_consumer_cannot_backdate_admission_drop_freeze_or_crop_RF(case, failure):
    reference = admission(case)
    risk = case.risk
    if failure == "backdated":
        case.objects[Path("admission")]["admitted_at"] = "2016-01-01T00:00:00Z"
    elif failure == "unfrozen":
        case.objects[Path("freeze")]["implementation_and_test_inputs"] = []
    else:
        risk = risk.loc["2021-01-01":]
    with pytest.raises(QuantError):
        review.load_admitted_expert(reference, review.law.UNIVERSE, risk)
