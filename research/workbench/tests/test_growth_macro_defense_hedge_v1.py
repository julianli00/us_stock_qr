import copy
import json
from contextlib import nullcontext
from dataclasses import replace
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_strategy_expert_pool_eight_v1 import archived_experts

from us_quant import funded_meta_experts_v1 as admitted
from us_quant import growth_macro_defense_hedge_v1 as rules
from us_quant import ssd_commitment_research_v1 as contract
from us_quant import ssd_growth_macro_defense_hedge_v1 as study
from us_quant.calendar import is_month_end
from us_quant.config import QuantError
from us_quant.storage import read_json, utc_now
from us_quant.strategy_experts_v1 import ExpertIdentity


class MappedSyntheticExpert:
    def __init__(self, source, mapping):
        self.identity, self.source, self.mapping = source.identity, source, mapping

    def observe(self, day, lookback):
        observed = self.source.observe(day, lookback)
        target = (
            observed.target.rename(index=self.mapping).reindex(admitted.ASSETS, fill_value=0.0)
            if observed.target is not None
            else None
        )
        return replace(observed, target=target)


def sources(changed_after=None, missing_growth=False):
    eight, calendar, toy_assets = archived_experts(changed_after=changed_after)
    mapping = dict(zip(toy_assets, admitted.ASSETS[: len(toy_assets)], strict=True))
    original = study.trio.select_sources(eight)
    selected = tuple(MappedSyntheticExpert(e, mapping) for e in original)
    dates = calendar[calendar >= "2016-10-06"]
    values = 0.0007 + 0.006 * np.sin(np.arange(len(dates)) / 11)
    if changed_after is not None:
        values[dates > changed_after] += 0.01
    nav = 10000 * np.cumprod(1 + values)
    shadow = pd.DataFrame(
        {
            "equity": nav,
            "return": values,
            "risk_free": 0.0001,
            "cash": nav * 0.02,
            "gross_exposure": 0.98,
            "cost": 0.0,
            "orders": 0,
        },
        index=dates,
    )
    targets = pd.DataFrame(np.nan, index=calendar, columns=admitted.ASSETS)
    active = pd.DatetimeIndex(
        [
            admitted.INITIAL,
            *[d for d in calendar if d > admitted.INITIAL and is_month_end(d)],
        ]
    )
    targets.loc[active] = 0.0
    targets.loc[active, ["MTUM", "VLUE", "QUAL", "USMV"]] = 0.245
    if changed_after is not None:
        later = active[active > changed_after]
        targets.loc[later] = 0.0
        targets.loc[later, ["SPY", "MTUM"]] = 0.49
    growth = admitted.ArchivedInitialMonthlyExpert(
        ExpertIdentity(rules.GROWTH_ID, admitted.GROUP, "2026-11-06T01:47:37.296678Z"),
        targets,
        shadow,
        admitted.ASSETS,
    )
    if missing_growth:
        original_observe = growth.observe
        growth = SimpleNamespace(
            identity=growth.identity,
            observe=lambda day, lookback: replace(
                original_observe(day, lookback), target=None, target_known_at=None
            ),
        )
    return (*selected, growth), calendar, admitted.ASSETS


def test_real_source_adapter_and_exact_unchanged_six_pair_implementations():
    experts, _, assets = sources()
    assert tuple(e.identity.identifier for e in experts) == rules.EXPERT_IDS
    assert tuple(e.identity.mechanism_group for e in experts) == rules.MECHANISMS
    assert len(assets) == 14
    assert isinstance(experts[-1], admitted.ArchivedInitialMonthlyExpert)
    assert rules.choose_pair is study.trio.rules.choose_pair
    assert rules.combine is study.trio.rules.combine
    observed = [e.observe(pd.Timestamp("2017-04-28"), 126) for e in experts]
    weights, audit = rules.choose_pair(observed, np.full(4, 0.25))
    assert len(audit["pairs"]) == 6 and (weights > 0).sum() == 2
    assert all(len(p["expert_weights"]) == 4 for p in audit["pairs"])
    assert all("training_correlation" in p for p in audit["pairs"])
    assert rules.combine(observed, np.full(4, 0.25)).sum() == pytest.approx(0.98)


def test_declared_initial_opinion_full_dates_and_unscored_quarters():
    experts, calendar, assets = sources()
    initial = experts[-1].observe(admitted.INITIAL, 126)
    assert initial.target_known_at == admitted.INITIAL and initial.target is not None
    assert initial.net_returns.empty and not initial.score_mature
    targets, audits = rules.build_targets(experts, calendar, assets, read_json(rules.POLICY))
    assert audits[0]["decision_session"] == "2016-10-05"
    assert audits[0]["equal_expert_weights"] == [0.25] * 4
    assert sum(not a["selection"]["scores_available"] for a in audits) == 7
    assert next(a["decision_session"] for a in audits if a["selection"]["scores_available"]) == (
        "2017-04-28"
    )
    assert all(t.index.equals(calendar) for t in targets.values())
    assert all(len(p["expert_weights"]) == 4 for a in audits for p in a["selection"]["pairs"])


def test_future_source_values_and_opinions_do_not_change_previous_allocations():
    cutoff = pd.Timestamp("2017-04-28")
    before, calendar, assets = sources()
    after, _, _ = sources(changed_after=cutoff)
    original, first = rules.build_targets(before, calendar, assets, read_json(rules.POLICY))
    changed, second = rules.build_targets(after, calendar, assets, read_json(rules.POLICY))
    prefix, prefix_audits = rules.build_targets(
        before, calendar[calendar <= cutoff], assets, read_json(rules.POLICY)
    )
    for name in study.IDS:
        pd.testing.assert_frame_equal(original[name].loc[:cutoff], changed[name].loc[:cutoff])
        pd.testing.assert_frame_equal(original[name].loc[:cutoff], prefix[name])
    assert prefix_audits == [a for a in first if a["decision_session"] <= str(cutoff.date())]
    assert prefix_audits == [a for a in second if a["decision_session"] <= str(cutoff.date())]


def test_all_six_pairs_retained_and_independently_verified_not_only_selected(tmp_path):
    experts, calendar, assets = sources()
    records = []
    targets, audits = rules.build_targets(
        experts, calendar, assets, read_json(rules.POLICY), retain=records.append
    )
    data = SimpleNamespace(close=pd.DataFrame(index=calendar, columns=assets))
    checked = study.verify_targets(data, experts, targets, audits)
    assert checked["maximum_independent_target_difference"] <= 1e-10
    assert checked["maximum_independent_expert_weight_difference"] <= 1e-10
    assert checked["maximum_independent_net_score_difference"] <= 1e-8
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    retained = study.verify_retained_records(path, experts, targets, audits)
    assert retained["actual_input_and_resolved_record_count"] == 2 * len(audits)
    assert all(len(r["observations"]) == 4 for r in records if r["record"] == "decision_input")
    assert records[0]["previous_weights_binary64_hex"] == [0.25.hex()] * 4
    damaged = copy.deepcopy(audits)
    mature = next(a for a in damaged if a["selection"]["scores_available"])
    unselected = next(
        p for p in mature["selection"]["pairs"] if p["pair"] != mature["selection"]["selected_pair"]
    )
    unselected["validation_net_risk_score"] += 0.01
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, targets, damaged)


def test_missing_growth_opinion_is_not_dropped_or_replaced_with_cash():
    experts, calendar, assets = sources(missing_growth=True)
    with pytest.raises(QuantError):
        rules.build_targets(experts, calendar, assets, read_json(rules.POLICY))


@pytest.mark.parametrize(
    "key,value",
    [
        ("growth_expert_id", admitted.IDS[1]),
        ("original_trio_source_slots", [3, 5, 7]),
        ("lookback_sessions", 252),
        ("training_sessions", 63),
        ("validation_sessions", 21),
        ("selected_experts", 3),
        ("mature_pair_count", 3),
        ("switch_cost_bps_per_side", 0.0),
        ("actual_weight_trade_band", 0.01),
        ("decision_information_clock", "at_16_00_ET"),
        ("new_source_RF_views", 1),
        ("all_source_native_RF_claimed_equal_without_view", True),
        ("growth_nested_source_overlap_disclosed", False),
        ("mechanism_labels_are_independent_risks", True),
        ("opportunity_gate_threshold_or_participation_state_tuned", True),
    ],
)
def test_no_growth_sibling_window_cost_view_or_financial_gate_rescue(key, value):
    with pytest.raises(QuantError):
        rules.validate({**read_json(rules.POLICY), key: value})


def test_both_real_registrations_precede_either_new_combination(monkeypatch):
    history, packets = [], []
    frozen = {"study_id": "synthetic_quartet"}

    def events(kind, name):
        return [r for r in history if r["kind"] == kind and r["candidate_id"] == name]

    def append(directory, kind, name, body):
        history.append(
            {"kind": kind, "candidate_id": name, "body": body, "sha256": f"event{len(history)}"}
        )

    def prepare(data, experts, actual):
        assert actual == frozen and all(len(events("registered", n)) == 1 for n in study.IDS)
        return {
            n: {
                "targets": {"path": "target", "sha256": "a"},
                "proof": {"path": "proof", "sha256": "b"},
            }
            for n in study.IDS
        }

    def retain(path, packet, *args):
        assert packet["complete_fixed_cohort_ids"] == list(study.IDS)
        assert len(packet["cohort_registrations"]) == 2
        packets.append(packet)

    common = SimpleNamespace(
        packet_entry=lambda c, f: {"candidate_id": c["id"]}, retain_receipt=retain
    )

    def bind(function, **overrides):
        return FunctionType(
            function.__code__,
            {
                **function.__globals__,
                **study.__dict__,
                "lifecycle": SimpleNamespace(runtime_guard=lambda: None, research_job=nullcontext),
                "inputs": lambda: (None, None, frozen, None),
                "events": events,
                "base": SimpleNamespace(JOURNAL=None, append_event=append),
                "prepare": prepare,
                "seven": SimpleNamespace(common=common),
                "reference": lambda p: {"path": str(p), "sha256": "x"},
                "progress": lambda *a, **k: None,
                **overrides,
            },
            argdefs=function.__defaults__,
        )

    monkeypatch.setattr(study, "bound", bind)
    study.register()
    assert [r["kind"] for r in history][:2] == ["registered"] * 2 and len(packets) == 2


def test_foreign_last_identity_blocks_the_first_append(monkeypatch):
    monkeypatch.setattr(study, "bound", lambda function: lambda: (None, None, {"frozen": 1}, None))
    monkeypatch.setattr(
        study, "events", lambda k, n: [{"body": {"foreign": True}}] if n == study.IDS[1] else []
    )
    with pytest.raises(QuantError):
        study.inputs()


def test_missing_second_complete_packet_blocks_all_funding():
    first, second = study.IDS
    packet = {"complete_fixed_cohort_ids": list(study.IDS), "cohort_registrations": ["r0", "r1"]}

    def events(kind, name):
        return [{"sha256": "r0" if name == first else "r1"}] if kind == "registered" else []

    def read(path):
        return (
            packet
            if path == study.PACKETS[first]
            else {**packet, "complete_fixed_cohort_ids": [second]}
        )

    checked = FunctionType(
        study.template.cohort_publication.__code__,
        {
            **study.template.cohort_publication.__globals__,
            **study.__dict__,
            "base": SimpleNamespace(JOURNAL=None, journal_events=lambda path: []),
            "events": events,
            "read_json": read,
            "scoped": lambda *a: lambda *b: {"ok": True},
        },
    )
    with pytest.raises(QuantError):
        checked({"study_id": "synthetic"}, "remotecommit")


def test_new_generated_proof_passes_the_real_canonical_packet_entry():
    candidate, frozen = rules.CANDIDATES[0], {"study_id": "synthetic"}
    evidence = study.target_proof(frozen, {"all_pairs_per_mature_decision": 6})
    registered = {
        "at": "2026-11-06T00:00:00Z",
        "sha256": "r",
        "body": {"candidate": candidate, "frozen": frozen},
    }
    proved = {
        "at": utc_now(),
        "sha256": "p",
        "body": {"registered_event_sha256": "r", "targets": {"path": "t"}, "proof": {"path": "p"}},
    }
    reader = SimpleNamespace(
        matching_events=lambda k, n: [registered] if k == "registered" else [proved],
        checked_reference=lambda r: r["path"],
    )
    real = FunctionType(
        contract.packet_entry.__code__,
        {**contract.packet_entry.__globals__, "base": reader, "read_json": lambda p: evidence},
    )
    assert real(candidate, frozen)["candidate_id"] == candidate["id"]
    assert evidence["original_source_signal_recomputed"] is False
    with pytest.raises(QuantError):
        study.target_proof(frozen, {"original_source_signal_recomputed": True})


def test_exact_admission_nonselection_nesting_and_fixture_imports_stay_bound():
    catalogue = read_json(study.base.checked_reference(read_json(rules.POLICY)["source_catalogue"]))
    assert catalogue["expert_ids"] == list(rules.EXPERT_IDS)
    assert catalogue["admissible_nonselected_variants"].keys() == set(admitted.IDS[1:])
    assert not catalogue["opportunity_gate_threshold_or_participation_state_tuned"]
    assert catalogue["admitted_legacy_defensive_views_reused_unchanged"]
    assert not catalogue["all_source_native_RF_claimed_equal_without_view"]
    paths = {r["path"] for r in study.dependencies()}
    assert set(study.OWN).issubset(paths)
    assert "src/us_quant/funded_meta_experts_v1.py" in paths
