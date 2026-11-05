import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from us_quant import style_activity_admission_v1 as review
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.storage import digest_json, read_json


@pytest.fixture
def case(monkeypatch):
    policy, protocol = read_json(review.POLICY), read_json(review.base.PROTOCOL)
    dates = sessions("2015-08-10", "2026-10-05")
    close = pd.DataFrame(100.0, dates, review.law.UNIVERSE)
    risk = pd.Series(0.0, dates, name="risk_free")
    targets = pd.DataFrame(np.nan, dates, review.law.UNIVERSE)
    number = 0
    for position, day in enumerate(dates):
        if position >= 64 and is_month_end(day):
            targets.loc[day] = 0.0
            selected = (
                list(review.law.CORE[:3] if number % 4 == 0 else review.law.CORE[1:])
                if number % 2 == 0
                else ["BIL"]
            )
            targets.loc[day, selected] = 0.98 / len(selected)
            number += 1
    active, held = targets.dropna(how="all"), targets.ffill()
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

    def reference(name):
        return {"path": name, "sha256": "d" * 64}

    publication = {
        "sequence": 2,
        "at": "2026-11-05T10:01:00Z",
        "sha256": "b" * 64,
        "body": {"verification": {"registration_packet": reference("packet")}},
    }
    native = {"targets": reference("targets"), "proof": reference("proof")}
    prefix_ref = reference("prefixes")
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
            frames[key + "/weights.csv"] = held.loc[days].copy()
            n = len(starts)
            starts.append(
                {
                    "sequence": 3 + 2 * n,
                    "at": f"2026-11-05T10:02:{2*n:02d}Z",
                    "body": {"path_key": key},
                }
            )
            ends.append(
                {
                    "sequence": 4 + 2 * n,
                    "at": f"2026-11-05T10:02:{2*n+1:02d}Z",
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
        "real_main_capital_prefix_audit": prefix_ref,
    }
    proved = {
        "sequence": 11,
        "at": "2026-11-05T10:03:00Z",
        "body": {"native_source_proof": native, "real_main_capital_prefix_audit": prefix_ref},
    }
    reviewed = {"sequence": 12, "at": "2026-11-05T10:04:00Z", "body": result["review"]}
    proof = {
        "frozen_specification_sha256": digest_json(frozen),
        "publication_event_sha256": publication["sha256"],
        "source_start": reference("started"),
        "checked_at": "2026-11-05T10:01:02Z",
        "real_native_indicators_first_generated_after_actual_publication": True,
        "real_native_targets_first_generated_after_actual_publication": True,
        "all_independent_signed_activity_targets_exact": True,
        "full_MarketData_four_fund_activity_scope_preserved": True,
        "old_activity_source_RF_imported_or_normalized": False,
        "decision_day_or_future_input_used": False,
        "source_price_volume_binary64_inputs_checked": len(active) * 760,
        "independent_four_style_balance_checks": len(active) * 4,
        "maximum_independent_scalar_numpy_balance_gap": 0.0,
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
        Path("prefixes"): {
            "auxiliary_main_book_audit_runs": 5,
            "extra_economic_rules_or_qualifying_windows": 0,
            "prefixes": [
                {"formal_return_position": p, "capital_and_decisions_exact": True, "artifacts": []}
                for p in (125, 503, 1006, 1509, 2012)
            ],
        },
        review.base.PROTOCOL: protocol,
    }
    event_map = {
        "registered": [reg],
        "review": [reviewed],
        "target_proof": [proved],
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
        events=lambda kind: event_map[kind],
        publication_fence=lambda: publication,
        reference=lambda p: publication["body"]["verification"]["registration_packet"]
        if p == Path("packet")
        else reference("results"),
        source_frame=lambda value: targets.copy(),
    )
    fake_base = SimpleNamespace(
        PATH_KEYS=set(paths),
        PROTOCOL=review.base.PROTOCOL,
        checked_reference=lambda r: Path(r["path"]),
        verify_path=lambda r, *args: receipts[r["path"]],
    )
    monkeypatch.setattr(review, "source", fake)
    monkeypatch.setattr(review, "base", fake_base)
    monkeypatch.setattr(review, "read_json", lambda p: objects[p])
    monkeypatch.setattr(review, "_frame", lambda r: frames[r["path"]].copy())
    monkeypatch.setattr(review, "check_trace", lambda *args: len(active) * 760)
    return SimpleNamespace(
        policy=policy,
        result=result,
        proof=proof,
        frames=frames,
        objects=objects,
        risk=risk,
        starts=starts,
        ends=ends,
        events=event_map,
        targets=targets,
    )


def test_financial_rejection_preserves_real_complete_expert_and_separate_score_maturity(case):
    expert, evidence = review.inspect_source(case.policy)
    assert evidence["standalone_financial_verdict_preserved"] == "rejected"
    assert all(len(v) == 4 for v in evidence["actual_style_families_by_path"].values())
    assert all(v == ["BIL"] for v in evidence["actual_defensive_assets_by_path"].values())
    assert [r["NET_observations"] for r in evidence["NET_maturity_boundary_observations"]] == [
        0,
        125,
        126,
    ]
    assert expert.identity.mechanism_group == "cross_sectional_reported_activity_participation"
    assert expert.observe(case.targets.index[63], 126).target is None
    assert evidence["current_global_pool_or_roster_changed"] is False


@pytest.mark.parametrize(
    "failure",
    ["cash", "gross", "missing_day", "RF", "families", "NaN", "stray_asset", "missing_defense"],
)
def test_bad_funding_dates_native_RF_or_actual_asset_coverage_cannot_be_admitted(case, failure):
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
    [
        "target_before_pub",
        "path_before_fence",
        "missing_path",
        "missing_opinion",
        "warmup",
        "prefix",
    ],
)
def test_saved_chronology_warmup_and_prefix_completeness_are_not_relabelled(case, failure):
    if failure == "target_before_pub":
        case.objects[Path("started")]["started_at"] = "2026-11-05T10:00:00Z"
    elif failure == "path_before_fence":
        case.starts[0]["sequence"] = 1
    elif failure == "missing_path":
        case.ends.pop()
    elif failure == "missing_opinion":
        case.targets.loc[case.targets.dropna(how="all").index[20]] = np.nan
    elif failure == "warmup":
        case.targets.iloc[20] = case.targets.dropna(how="all").iloc[0]
    else:
        case.objects[Path("prefixes")]["prefixes"][0]["capital_and_decisions_exact"] = False
    with pytest.raises(QuantError):
        review.inspect_source(case.policy)


@pytest.mark.parametrize(
    "key,value",
    [
        ("full_MarketData_four_fund_activity_scope_preserved", False),
        ("decision_day_or_future_input_used", True),
        ("source_price_volume_binary64_inputs_checked", True),
        ("source_price_volume_binary64_inputs_checked", 0),
        ("independent_four_style_balance_checks", 0),
        ("real_native_indicators_first_generated_after_actual_publication", False),
        ("old_activity_source_RF_imported_or_normalized", True),
        ("maximum_independent_scalar_numpy_balance_gap", -1.0),
        ("maximum_independent_scalar_numpy_balance_gap", float("nan")),
    ],
)
def test_frozen_source_scope_or_independent_evidence_cannot_be_softened(case, key, value):
    case.proof[key] = value
    with pytest.raises(QuantError):
        review.inspect_source(case.policy)


def test_nonfinite_or_negative_saved_BT_disagreement_cannot_pass(case):
    for value in (float("nan"), -1.0):
        case.result["paths"]["10y-base"]["independent_bt_max_equity_difference_usd"] = value
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
def test_original_technical_standards_and_financial_rejection_not_tuned(case, key, value):
    bad = deepcopy(case.policy)
    bad[key] = value
    with pytest.raises(QuantError):
        review.validate(bad)


def test_trace_contract_checks_saved_binary64_inputs_without_regenerating_source(
    market_factory, tmp_path, monkeypatch
):
    from us_quant import style_activity_participation_v1 as law

    data = market_factory(start="2015-08-10", end="2017-12-29", symbols=law.UNIVERSE)
    records = []
    targets, _ = law.build_targets(data, read_json(law.POLICY), retain=records.append)
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    monkeypatch.setattr(review.base, "checked_reference", lambda r: path)
    assert review.check_trace(data, targets, {}) == 760 * len(targets.dropna())
    records[0]["inputs_binary64_hex"]["MTUM"]["reported_share_volumes"][0] = (1.0).hex()
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(QuantError):
        review.check_trace(data, targets, {})


@pytest.fixture
def admitted_case(case, monkeypatch):
    expert, evidence = review.inspect_source(case.policy)
    freeze = {
        "created_at": "2026-11-05T11:00:00Z",
        "implementation_and_test_inputs": [{"path": p, "sha256": "e" * 64} for p in review.OWN],
        "complete_dependency_inputs": [{"path": p, "sha256": "e" * 64} for p in review.OWN],
    }
    admission = {
        "technical_expert_admitted": True,
        "global_pool_changed": False,
        "new_strategy_evaluations": 0,
        "order_authority": False,
        "review_implementation_freeze": {"path": "review-freeze", "sha256": "f" * 64},
        "admitted_at": "2026-11-05T12:00:00Z",
        "evidence": evidence,
    }
    case.objects[Path("admission")] = admission
    case.objects[Path("review-freeze")] = freeze
    case.objects[review.POLICY] = case.policy
    monkeypatch.setattr(review, "inspect_source", lambda policy: (expert, evidence))
    return SimpleNamespace(
        expert=expert,
        evidence=evidence,
        admission=admission,
        freeze=freeze,
        reference={"path": "admission", "sha256": "f" * 64},
        risk=case.risk,
    )


def test_saved_source_consumer_accepts_technically_admitted_financially_rejected_policy(
    admitted_case,
):
    expert, evidence = review.load_admitted_expert(
        admitted_case.reference, review.law.UNIVERSE, admitted_case.risk
    )
    assert expert is admitted_case.expert
    assert evidence["standalone_financial_verdict_preserved"] == "rejected"
    assert evidence["actual_expert_admitted_at"] == "2026-11-05T12:00:00Z"
    assert expert.observe(pd.Timestamp("2016-10-05"), 126).net_returns.empty


@pytest.mark.parametrize(
    "failure", ["RF", "crop", "universe", "not_admitted", "trading", "review_code", "backdated"]
)
def test_consumer_cannot_launder_wrong_RF_universe_code_or_admission(admitted_case, failure):
    universe, rf = review.law.UNIVERSE, admitted_case.risk.copy()
    if failure == "RF":
        rf.iloc[-1] += 1e-10
    elif failure == "crop":
        rf = rf.drop(pd.Timestamp("2016-10-06"))
    elif failure == "universe":
        universe = universe[::-1]
    elif failure == "not_admitted":
        admitted_case.admission["technical_expert_admitted"] = False
    elif failure == "trading":
        admitted_case.admission["order_authority"] = True
    elif failure == "review_code":
        admitted_case.freeze["complete_dependency_inputs"].pop()
    else:
        admitted_case.admission["admitted_at"] = "2026-11-05T10:00:00Z"
    with pytest.raises(QuantError):
        review.load_admitted_expert(admitted_case.reference, universe, rf)
