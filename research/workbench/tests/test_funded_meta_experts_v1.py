import copy
import json
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from us_quant import funded_meta_experts_v1 as rules
from us_quant import ssd_commitment_research_v1 as canonical
from us_quant.calendar import is_month_end, next_session, sessions
from us_quant.config import QuantError
from us_quant.storage import digest_json, file_digest, read_json, utc_now
from us_quant.strategy_experts_v1 import ArchivedMonthlyExpert, ExpertIdentity


@pytest.fixture(scope="module")
def native():
    index = sessions("2016-10-04", "2026-10-05")
    targets = pd.DataFrame(np.nan, index=index, columns=rules.ASSETS)
    opinions = [rules.INITIAL, *[d for d in index if d > rules.INITIAL and is_month_end(d)]]
    for i, day in enumerate(opinions):
        targets.loc[day] = 0.0
        targets.loc[day, "MTUM" if i % 2 == 0 else "VLUE"] = 0.98
    dates = sessions("2016-10-06", "2026-10-05")
    shadow = pd.DataFrame(
        {
            "equity": 10000.0,
            "return": 0.0,
            "risk_free": 0.0001,
            "cash": 200.0,
            "gross_exposure": 0.98,
            "turnover": 0.0,
            "cost": 0.0,
            "orders": 0,
        },
        index=dates,
    )
    return targets, shadow


def expert(native, identifier=None):
    targets, shadow = native
    return rules.ArchivedInitialMonthlyExpert(
        ExpertIdentity(identifier or rules.IDS[0], rules.GROUP, "2026-11-06T00:00:00Z"),
        targets,
        shadow,
        rules.ASSETS,
    )


def record():
    return {
        "id": rules.IDS[0],
        "mechanism_group": rules.GROUP,
        "risk_behavior": "always_invested",
        "model_variant": "unconditional",
        "initial_policy_opinion_session": "2016-10-05",
        "native_information_clock": rules.CLOCK,
        "tape_input_kind": "saved_forecast_allocation_input",
        "tape_resolved_kind": "saved_forecast_allocation_resolved",
        "tape_candidate_id": rules.IDS[0].removesuffix("_v2") + "_v1",
        "target_proof_decision_count": 121,
        "counterfactual_historical_availability": True,
        "new_global_pool_membership": False,
    }


@pytest.mark.parametrize("identifier", rules.IDS)
def test_all_four_complete_programs_cross_same_observation_interface(native, identifier):
    source = expert(native, identifier)
    initial = source.observe(rules.INITIAL, 126)
    assert initial.target_known_at == rules.INITIAL and initial.target is not None
    assert initial.net_returns.empty and not initial.score_mature
    assert source.preservation()["native_policy_opinions"] == 121
    assert source.preservation()["ordinary_month_end_opinions"] == 120
    assert not source.preservation()["source_target_rows_removed_retimed_or_backfilled"]
    with pytest.raises(QuantError):
        ArchivedMonthlyExpert(source.identity, *native, rules.ASSETS)


def test_initial_announcement_and_later_capital_seed_have_different_roles(native):
    source = expert(native)
    initial = source.event_context("2016-10-05")
    assert initial["declared_initial_policy_opinion_today"]
    assert initial["native_policy_opinion_refreshed_today"]
    assert not initial["ordinary_month_end_opinion_today"]
    assert set(initial["formal_capital_anchor_paths"]) == {"10y-base", "10y-stress"}
    later = source.event_context("2021-10-05")
    assert later["policy_opinion_known_at"] == "2021-09-30"
    assert not later["native_policy_opinion_refreshed_today"]
    assert set(later["formal_capital_anchor_paths"]) == {"5y-base", "5y-stress"}
    assert not later["capital_initialization_itself_is_policy_refresh"]
    assert not later["observation_is_a_new_execution_request"]
    carried = source.event_context("2016-10-06")
    assert carried["policy_opinion_known_at"] == "2016-10-05"
    assert not carried["native_policy_opinion_refreshed_today"]


def test_missing_history_maturity_and_returned_copies_are_explicit(native):
    source = expert(native)
    absent = source.observe(pd.Timestamp("2016-10-04"), 126)
    assert absent.target is None and absent.target_known_at is None
    assert absent.net_returns.empty and not absent.score_mature
    assert not source.observe(native[1].index[124], 126).score_mature
    mature = source.observe(native[1].index[125], 126)
    assert mature.score_mature and len(mature.net_returns) == 126
    pd.testing.assert_series_equal(mature.risk_free, native[1]["risk_free"].iloc[:126])
    mature.target.iloc[0] = 9.0
    mature.net_returns.iloc[0] = 1.0
    assert source.observe(native[1].index[125], 126).target.sum() == 0.98
    assert source.observe(native[1].index[125], 126).net_returns.iloc[0] == 0.0


def test_exact_truncated_prefixes_keep_every_previous_observation(native):
    full = expert(native)
    targets, shadow = native
    for cutoff in (
        shadow.index[0],
        pd.Timestamp("2016-10-31"),
        shadow.index[125],
        pd.Timestamp("2021-10-05"),
        shadow.index[-1],
    ):
        early = expert((targets.loc[:cutoff], shadow.loc[:cutoff]))
        for day in targets.dropna().index.union(pd.DatetimeIndex([cutoff, rules.INITIAL])):
            if day > cutoff:
                continue
            left, right = full.observe(day, 126), early.observe(day, 126)
            assert left.identity == right.identity and left.target_known_at == right.target_known_at
            assert left.score_mature is right.score_mature
            pd.testing.assert_series_equal(left.target, right.target)
            pd.testing.assert_series_equal(left.net_returns, right.net_returns)
            pd.testing.assert_series_equal(left.risk_free, right.risk_free)
            assert full.event_context(day) == early.event_context(day)


@pytest.mark.parametrize(
    "damage",
    [
        "missing_initial",
        "missing_month",
        "extra_day",
        "negative",
        "borrowed",
        "leveraged",
        "missing_return",
        "foreign",
        "coordinates",
    ],
)
def test_invalid_history_or_funding_is_not_a_safe_default(native, damage):
    targets, shadow = (v.copy() for v in native)
    identifier = rules.IDS[0]
    if damage == "missing_initial":
        targets.loc[rules.INITIAL] = np.nan
    elif damage == "missing_month":
        targets.loc["2016-10-31"] = np.nan
    elif damage == "extra_day":
        targets.loc["2016-10-06"] = targets.loc[rules.INITIAL]
    elif damage == "negative":
        targets.loc[rules.INITIAL, "SPY"] = -0.01
    elif damage == "borrowed":
        shadow.iloc[0, shadow.columns.get_loc("cash")] = -1.0
    elif damage == "leveraged":
        shadow.iloc[0, shadow.columns.get_loc("gross_exposure")] = 1.01
    elif damage == "missing_return":
        shadow = shadow.drop(shadow.index[10])
    elif damage == "foreign":
        identifier = "expert_pool_eight_unconditional_forecast_pair_v1"
    else:
        targets = targets.loc[:, list(reversed(rules.ASSETS))]
    with pytest.raises(QuantError):
        expert((targets, shadow), identifier)


@pytest.mark.parametrize("lookback", [1, True, 2.0])
def test_observation_window_is_an_explicit_integer(native, lookback):
    with pytest.raises(QuantError):
        expert(native).observe(rules.INITIAL, lookback)


@pytest.mark.parametrize(
    "key,value",
    [
        ("id", "term_event_policy"),
        ("new_global_pool_membership", True),
        ("tape_candidate_id", rules.IDS[0]),
        ("native_information_clock", "future"),
        ("initial_policy_opinion_session", "2016-09-30"),
    ],
)
def test_foreign_source_seed_clock_and_promotion_are_rejected(key, value):
    with pytest.raises(QuantError):
        rules.check_record({**record(), key: value})


def retained_tape(tmp_path, monkeypatch, native, damage=None):
    monkeypatch.setattr(rules.base, "ROOT", tmp_path)
    r = record()
    rows = []
    for i, (day, target) in enumerate(native[0].dropna().iterrows()):
        audit = {
            "decision_index": i,
            "decision_session": str(day.date()),
            "decision_information_clock": rules.CLOCK,
        }
        values = target.tolist()
        if i == 0 and damage == "one_ulp":
            values[4] = np.nextafter(values[4], np.inf)
        if i == 0 and damage == "future":
            audit["decision_session"] = "2026-10-05"
        rows.extend(
            [
                {"record": r["tape_input_kind"]},
                {
                    "record": r["tape_resolved_kind"],
                    "audit": audit,
                    "targets": {r["tape_candidate_id"]: values},
                },
            ]
        )
    if damage == "partial":
        rows.pop()
    path = tmp_path / "source.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    r["resolved_opinion_tape"] = {"path": path.name, "sha256": file_digest(path)}
    return r


@pytest.mark.parametrize("damage", [None, "one_ulp", "future", "partial"])
def test_retained_initial_and_monthly_source_values_are_bit_exact(
    tmp_path,
    monkeypatch,
    native,
    damage,
):
    r = retained_tape(tmp_path, monkeypatch, native, damage)
    if damage is None:
        assert (
            rules.verify_retained_opinions(r, native[0])["retained_policy_opinions_bit_exact"]
            == 121
        )
    else:
        with pytest.raises(QuantError):
            rules.verify_retained_opinions(r, native[0])


def funding_files(tmp_path, monkeypatch, native, key):
    monkeypatch.setattr(rules.base, "ROOT", tmp_path)
    targets, original = native
    first = "2016-10-06" if key.startswith("10y") else "2021-10-06"
    book = original.loc[first:].copy()
    weights = pd.DataFrame(0.0, index=book.index, columns=rules.ASSETS)
    weights["SPY"] = 0.98
    issued = pd.DataFrame(np.nan, index=targets.index, columns=targets.columns)
    days = [rules.ANCHORS[key], *[d for d in book.index if is_month_end(d)]]
    decisions = []
    for i, day in enumerate(days):
        execution = day
        for _ in range(1 if key.endswith("base") else 2):
            execution = next_session(execution)
        issued.loc[day] = targets.dropna().loc[:day].iloc[-1]
        decisions.append(
            {
                "signal_session": str(day.date()),
                "execution_session": str(execution.date()),
                "initial_capital": i == 0,
                "reason": "initial_capital" if i == 0 else "weight_gap",
            }
        )
    values = {
        "account.csv": book,
        "weights.csv": weights,
        "issued.csv": issued,
        "independent-bt.csv": book[["equity"]],
    }
    for name, value in values.items():
        value.to_csv(tmp_path / name, float_format="%.17g")
    (tmp_path / "decisions.json").write_text(json.dumps({"decisions": decisions}))
    receipt = {
        "path_key": key,
        "artifacts": [
            {"path": n, "sha256": file_digest(tmp_path / n)} for n in (*values, "decisions.json")
        ],
        "independent_bt_max_equity_difference_usd": 0.0,
        "broker_orders": 0,
        "independent_monthly_feedback": {"all_actual_gaps_targets_and_delays_checked": True},
    }
    return receipt, targets, original["risk_free"]


@pytest.mark.parametrize("key", ["10y-base", "10y-stress", "5y-base", "5y-stress"])
def test_all_original_funded_calendars_delays_and_capital_roles(tmp_path, monkeypatch, native, key):
    receipt, targets, rf = funding_files(tmp_path, monkeypatch, native, key)
    _, checked = rules._funding(receipt, targets, rf)
    assert checked["return_dates"] == (2512 if key.startswith("10y") else 1254)
    assert checked["native_RF_exact_canonical"]
    assert checked["original_policy_opinion_known_at_capital_initialization"] == (
        "2016-10-05" if key.startswith("10y") else "2021-09-30"
    )
    assert not checked["capital_initialization_creates_new_policy_opinion"]


@pytest.mark.parametrize("damage", ["RF", "delay", "seed_target", "borrow", "broker"])
def test_native_RF_timing_seed_cash_and_broker_failures_are_not_laundered(
    tmp_path,
    monkeypatch,
    native,
    damage,
):
    receipt, targets, rf = funding_files(tmp_path, monkeypatch, native, "5y-stress")
    if damage == "broker":
        receipt["broker_orders"] = 1
    elif damage == "delay":
        path = tmp_path / "decisions.json"
        rows = read_json(path)
        rows["decisions"][0]["execution_session"] = "2021-10-06"
        path.write_text(json.dumps(rows))
    else:
        name = "issued.csv" if damage == "seed_target" else "account.csv"
        path = tmp_path / name
        values = pd.read_csv(path, index_col=0, parse_dates=True, float_precision="round_trip")
        if damage == "seed_target":
            values.loc["2021-10-05", "SPY"] = 0.01
        elif damage == "borrow":
            values.iloc[0, values.columns.get_loc("cash")] = -1.0
        else:
            values.iloc[0, values.columns.get_loc("risk_free")] = np.nextafter(0.0001, np.inf)
        values.to_csv(path, float_format="%.17g")
    for ref in receipt["artifacts"]:
        ref["sha256"] = file_digest(tmp_path / ref["path"])
    with pytest.raises(QuantError):
        rules._funding(receipt, targets, rf)


def test_existing_real_packet_entry_contract_retains_actual_attestation_chronology():
    candidate, frozen = {"id": rules.IDS[0]}, {"study_id": "synthetic_source"}
    registration = {
        "at": "2026-11-06T00:00:00Z",
        "sha256": "registered",
        "body": {"candidate": candidate, "frozen": frozen},
    }
    proof_event = {
        "at": utc_now(),
        "sha256": "proved",
        "body": {
            "registered_event_sha256": "registered",
            "targets": {"path": "target"},
            "proof": {"path": "proof"},
        },
    }
    proof = {
        "checked_at": registration["at"],
        "frozen_specification_sha256": digest_json(frozen),
        "candidate_accounts_or_metrics_computed": False,
        "original_source_signal_recomputed": False,
        "source_original_target_derivation_precedes_V2_registration": True,
    }
    reader = SimpleNamespace(
        matching_events=lambda kind, identifier: [registration]
        if kind == "registered"
        else [proof_event],
        checked_reference=lambda ref: ref["path"],
    )
    real = FunctionType(
        canonical.packet_entry.__code__,
        {**canonical.packet_entry.__globals__, "base": reader, "read_json": lambda path: proof},
    )
    assert real(candidate, frozen)["registered_at"] == registration["at"]
    bad = copy.deepcopy(proof)
    bad["original_source_signal_recomputed"] = True
    rejected = FunctionType(real.__code__, {**real.__globals__, "read_json": lambda path: bad})
    with pytest.raises(QuantError):
        rejected(candidate, frozen)


@pytest.mark.parametrize(
    "key,value",
    [
        ("source_family_count", 4),
        ("new_financial_trials", 1),
        ("global_pool_changed", True),
        ("source_native_RF_must_equal_canonical_default_reader", False),
    ],
)
def test_review_does_not_inflate_risks_relax_RF_or_promote_to_a_portfolio(key, value):
    with pytest.raises(QuantError):
        rules.validate({**read_json(rules.POLICY), key: value})
