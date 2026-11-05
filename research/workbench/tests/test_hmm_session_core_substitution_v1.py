import copy
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from test_strategy_expert_pool_eight_v1 import archived_experts as original

from us_quant import hmm_session_core_substitution_v1 as rules
from us_quant import ssd_hmm_session_core_substitution_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json, utc_now
from us_quant.strategy_experts_v1 import ExpertIdentity


def examples(changed_after=None):
    experts, calendar, assets = original(changed_after=changed_after)
    old = experts[2]
    pools = {}
    for variant in rules.VARIANTS:
        replacement = type(old)(
            ExpertIdentity(
                rules.EXPERT_IDS[variant][2], rules.MECHANISMS[2], "2026-10-29T08:18:12Z"
            ),
            old._targets.copy(),
            old._shadow.copy(),
            assets,
        )
        pools[variant] = study.substitute(experts, replacement, variant)
    return pools, calendar, assets


def prepared():
    pools, calendar, assets = examples()
    records = []
    targets, audits = rules.build_targets(
        pools, calendar, assets, read_json(rules.POLICY), retain=records.append
    )
    return (
        SimpleNamespace(close=pd.DataFrame(index=calendar, columns=assets)),
        pools,
        targets,
        audits,
        records,
    )


def test_three_separate_pools_only_slot2_changes_and_no_siblings_coexist():
    pools, _, _ = examples()
    for variant, experts in pools.items():
        assert len(experts) == 8
        assert tuple(e.identity.identifier for e in experts) == rules.EXPERT_IDS[variant]
        assert [
            i for i in range(8) if rules.EXPERT_IDS[variant][i] != rules.prior.EXPERT_IDS[i]
        ] == [2]
        assert len(set(rules.SOURCE_IDS) & {e.identity.identifier for e in experts}) == 1
    assert rules.choose_pair is rules.prior.choose_pair
    assert rules.combine is rules.prior.combine
    assert rules.MECHANISMS == rules.prior.MECHANISMS


def test_unapproved_source_order_or_append_is_not_declared_substitution():
    old, _, _ = original()
    pools, _, _ = examples()
    with pytest.raises(QuantError):
        study.substitute((*old, old[0]), pools["close"][2], "close")
    with pytest.raises(QuantError):
        study.substitute(old, pools["overnight"][2], "intraday")
    with pytest.raises(QuantError):
        study.substitute(tuple(reversed(old)), pools["close"][2], "close")


def test_each_pool_preserves_seven_unscored_states_and_every28_pair():
    _, _, _, audits, _ = prepared()
    for variant in rules.VARIANTS:
        rows = study.variant_audits(audits, variant)
        assert sum(a["selection"]["reason"] == "unscored_initial_equal_weights" for a in rows) == 7
        mature = [a for a in rows if a["selection"]["scores_available"]]
        assert mature[0]["decision_session"] == "2017-04-28"
        assert all(len(a["selection"]["pairs"]) == 28 for a in mature)


def test_prefix_and_future_changes_preserve_all_six_pasts():
    pools, calendar, assets = examples()
    changed, _, _ = examples(changed_after=pd.Timestamp("2017-04-28"))
    full, audits = rules.build_targets(pools, calendar, assets, read_json(rules.POLICY))
    after, future = rules.build_targets(changed, calendar, assets, read_json(rules.POLICY))
    early, first = rules.build_targets(
        pools, calendar[calendar <= "2017-04-28"], assets, read_json(rules.POLICY)
    )
    for name in study.IDS:
        assert early[name].equals(full[name].loc[:"2017-04-28"])
        assert early[name].equals(after[name].loc[:"2017-04-28"])
    assert first == [a for a in audits if a["decision_session"] <= "2017-04-28"]
    assert first == [a for a in future if a["decision_session"] <= "2017-04-28"]


def test_sibling_learning_states_do_not_cross():
    pools, calendar, assets = examples()
    before, _ = rules.build_targets(pools, calendar, assets, read_json(rules.POLICY))
    old = pools["intraday"][2]
    target = old._targets.copy()
    target.loc[~target.isna().all(axis=1)] = [0.098, 0.294, 0.294, 0.294]
    changed = {
        **pools,
        "intraday": (
            *pools["intraday"][:2],
            type(old)(old.identity, target, old._shadow.copy(), assets),
            *pools["intraday"][3:],
        ),
    }
    after, _ = rules.build_targets(changed, calendar, assets, read_json(rules.POLICY))
    assert all(
        before[n].equals(after[n]) for variant in ("close", "overnight") for n in rules.ids(variant)
    )
    assert not before[rules.ids("intraday")[0]].equals(after[rules.ids("intraday")[0]])


def test_all_three_scalar_oracles_and_fullprecision_input_records(tmp_path, monkeypatch):
    data, pools, targets, audits, rows = prepared()
    proof = study.verify_targets(data, pools, targets, audits)
    assert proof["each_variant_all28_pairs_verified"]
    assert set(proof["paired_pool_independent_oracle_checks"]) == set(rules.VARIANTS)
    assert proof["maximum_independent_target_difference"] < 1e-10
    assert proof["maximum_independent_net_score_difference"] < 1e-8
    path = tmp_path / "actual.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(study, "reference", lambda p: {"path": str(p), "sha256": "synthetic"})
    checked = study.verify_retained_records(path, pools, targets, audits)
    assert checked["actual_input_and_resolved_record_count"] == 6 * len(audits)
    assert set(checked["paired_pool_retained_record_checks"]) == set(rules.VARIANTS)


@pytest.mark.parametrize("kind", ["correlation", "cost", "marginal", "pair_count"])
def test_nonselected_pair_and_any_variant_cannot_hide_tampering(kind):
    data, pools, targets, audits, _ = prepared()
    audits = copy.deepcopy(audits)
    active = next(
        a["variants"]["overnight"]
        for a in audits
        if a["variants"]["overnight"]["selection"]["scores_available"]
    )
    row = next(
        p
        for p in active["selection"]["pairs"]
        if p["pair"] != active["selection"]["selected_pair"] and "selection_score_key" in p
    )
    if kind == "correlation":
        row["training_correlation"] += 0.01
    elif kind == "cost":
        row["estimated_incremental_switch_cost_fraction"] += 0.01
    elif kind == "marginal":
        row["training_marginal_variance_contributions"][0] += 0.01
    else:
        active["selection"]["pairs"].pop()
    with pytest.raises(QuantError):
        study.verify_targets(data, pools, targets, audits)


@pytest.mark.parametrize(
    "key,value",
    [
        ("lookback_sessions", 252),
        ("selected_experts", 3),
        ("switch_cost_bps_per_side", 0.0),
        ("actual_weight_trade_band", 0.01),
        ("replaced_slot", 5),
        ("source_variants_are_independent_economic_factors", True),
        ("candidate_count_in_this_study", 4),
    ],
)
def test_no_window_cost_band_role_or_cohort_rescue(key, value):
    policy = read_json(rules.POLICY)
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)


def test_all_six_registrations_precede_first_new_derivation(monkeypatch, tmp_path):
    rows = []
    monkeypatch.setattr(study.lifecycle, "runtime_guard", lambda: None)
    monkeypatch.setattr(study.lifecycle, "research_job", nullcontext)
    monkeypatch.setattr(study, "inputs", lambda: (None, None, {}, None))
    monkeypatch.setattr(study, "FOLDER", tmp_path)
    monkeypatch.setattr(study, "progress", lambda *a, **k: None)
    monkeypatch.setattr(
        study,
        "events",
        lambda kind, name: [r for r in rows if r["kind"] == kind and r["candidate_id"] == name],
    )
    monkeypatch.setattr(
        study.base,
        "append_event",
        lambda directory, kind, name, body: rows.append(
            {"kind": kind, "candidate_id": name, "body": body, "sha256": name}
        ),
    )

    def stop(*args):
        assert [r["candidate_id"] for r in rows if r["kind"] == "registered"] == list(study.IDS)
        raise QuantError("synthetic stop after all six registrations")

    monkeypatch.setattr(study, "prepare", stop)
    with pytest.raises(QuantError, match="all six"):
        study.register()
    assert {r["candidate_id"] for r in rows if r["kind"] == "technical_failure"} == set(study.IDS)


def test_real_packet_constructor_and_all_six_cohort_metadata(monkeypatch, tmp_path):
    rows = []
    frozen = {"study_id": "synthetic"}
    monkeypatch.setattr(study.lifecycle, "runtime_guard", lambda: None)
    monkeypatch.setattr(study.lifecycle, "research_job", nullcontext)
    monkeypatch.setattr(study, "inputs", lambda: (None, None, frozen, None))
    monkeypatch.setattr(study, "FOLDER", tmp_path)
    monkeypatch.setattr(study, "progress", lambda *a, **k: None)
    monkeypatch.setattr(
        study, "PACKETS", {name: tmp_path / f"{i}.json" for i, name in enumerate(study.IDS)}
    )

    def finder(kind, name):
        return [r for r in rows if r["kind"] == kind and r["candidate_id"] == name]

    monkeypatch.setattr(study, "events", finder)
    monkeypatch.setattr(study.base, "matching_events", finder)

    def append(directory, kind, name, body):
        rows.append(
            {
                "kind": kind,
                "candidate_id": name,
                "body": body,
                "at": utc_now(),
                "sha256": name + kind,
            }
        )

    monkeypatch.setattr(study.base, "append_event", append)
    monkeypatch.setattr(study, "reference", lambda p: {"path": str(p), "sha256": "synthetic"})
    monkeypatch.setattr(study.base, "checked_reference", lambda r: Path(r["path"]))
    monkeypatch.setattr(
        study.seven.common,
        "retain_receipt",
        lambda path, body, *args: path.write_text(json.dumps(body)),
    )

    def prepare(*args):
        assert len(rows) == 6
        result = {}
        for i, name in enumerate(study.IDS):
            target, proof = tmp_path / f"{i}.csv", tmp_path / f"{i}-proof.json"
            target.write_text("synthetic target")
            proof.write_text(
                json.dumps(
                    {
                        "checked_at": utc_now(),
                        "candidate_accounts_or_metrics_computed": False,
                        "original_source_signal_recomputed": False,
                    }
                )
            )
            result[name] = {"targets": study.reference(target), "proof": study.reference(proof)}
        return result

    monkeypatch.setattr(study, "prepare", prepare)
    study.register()
    for name in study.IDS:
        packet = json.loads(study.PACKETS[name].read_text())
        assert packet["entries"][0]["candidate_id"] == name
        assert packet["complete_fixed_cohort_ids"] == list(study.IDS)
        assert len(packet["cohort_registrations"]) == 6


@pytest.mark.parametrize("missing", range(6))
def test_any_missing_packet_blocks_every_funded_candidate(monkeypatch, missing):
    monkeypatch.setattr(study.base, "journal_events", lambda p: [])
    monkeypatch.setattr(
        study, "events", lambda kind, name: [{"sha256": name}] if kind == "registered" else []
    )
    monkeypatch.setattr(
        study,
        "read_json",
        lambda path: {
            "complete_fixed_cohort_ids": list(study.IDS)
            if path != study.PACKETS[study.IDS[missing]]
            else list(study.IDS[:missing]),
            "cohort_registrations": list(study.IDS),
        },
    )
    monkeypatch.setattr(study, "scoped", lambda *args: lambda *args: {"verified": True})
    with pytest.raises(QuantError, match="every declared"):
        study.cohort_publication({"study_id": "synthetic"}, "0" * 40)


def test_foreign_last_id_blocks_first_append_atomically(monkeypatch):
    rows = []
    frozen = {"study_id": "synthetic"}
    original_bound = study.bound
    input_function = study.template.inputs
    monkeypatch.setattr(study.lifecycle, "runtime_guard", lambda: None)
    monkeypatch.setattr(study.lifecycle, "research_job", nullcontext)
    monkeypatch.setattr(
        study,
        "bound",
        lambda fn, **kw: (lambda: (None, None, frozen, None))
        if fn is input_function
        else original_bound(fn, **kw),
    )
    monkeypatch.setattr(
        study,
        "events",
        lambda kind, name: [{"body": {"candidate": {"id": "foreign"}, "frozen": {}}}]
        if kind == "registered" and name == study.IDS[-1]
        else [],
    )
    monkeypatch.setattr(study.base, "append_event", lambda *args: rows.append(args))
    with pytest.raises(QuantError, match="before any new event"):
        study.register()
    assert rows == []


def test_source_and_financial_namespaces_unchanged_and_dependency_closure_complete():
    old = study.financial.ID, study.financial.FOLDER, study.financial.PACKET
    scoped = study.scoped(study.financial.evaluate, study.CANDIDATES[-1])
    assert scoped.__globals__["ID"] == study.IDS[-1]
    assert scoped.__globals__["publication"] is study.cohort_publication
    assert old == (study.financial.ID, study.financial.FOLDER, study.financial.PACKET)
    paths = {r["path"] for r in study.dependencies()}
    assert set(study.OWN).issubset(paths)
    assert "src/us_quant/archived_hmm_session_eligibility_v1.py" in paths
