import copy
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from test_strategy_expert_pool_eight_v1 import archived_experts as original

from us_quant import paired_activity_information_v1 as rules
from us_quant import ssd_paired_activity_information_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json, utc_now
from us_quant.strategy_experts_v1 import ArchivedMonthlyExpert, ExpertIdentity


def examples(changed_after=None):
    experts, calendar, assets = original(changed_after=changed_after)
    old = experts[5]
    pools = {}
    for key in rules.VARIANTS:
        replacement = ArchivedMonthlyExpert(
            ExpertIdentity(rules.EXPERT_IDS[key][5], rules.MECHANISMS[5], "2026-10-29T04:55:54Z"),
            old._targets,
            old._shadow,
            assets,
        )
        pools[key] = study.substitute(experts, replacement, key)
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


def test_two_siblings_replace_only_same_role_and_never_coexist_or_inflate_pool():
    pools, _, _ = examples()
    for key, experts in pools.items():
        assert len(experts) == 8
        assert tuple(e.identity.identifier for e in experts) == rules.EXPERT_IDS[key]
        assert [i for i in range(8) if rules.EXPERT_IDS[key][i] != rules.prior.EXPERT_IDS[i]] == [5]
        assert len(set(rules.SOURCE_IDS) & {e.identity.identifier for e in experts}) == 1
    assert rules.choose_pair is rules.prior.choose_pair
    assert rules.combine is rules.prior.combine
    assert rules.MECHANISMS == rules.prior.MECHANISMS


def test_unapproved_source_order_or_append_is_not_a_paired_ablation():
    old, _, _ = original()
    pools, _, _ = examples()
    with pytest.raises(QuantError):
        study.substitute((*old, old[0]), pools["price"][5], "price")
    with pytest.raises(QuantError):
        study.substitute(old, pools["activity"][5], "price")


def test_identical_synthetic_sources_reproduce_both_original_laws_exactly():
    pools, calendar, assets = examples()
    old, _, _ = original()
    result, _ = rules.build_targets(pools, calendar, assets, read_json(rules.POLICY))
    expected, _ = rules.prior.build_targets(old, calendar, assets, read_json(rules.prior.POLICY))
    for key in rules.VARIANTS:
        for name, old_candidate in zip(rules.ids(key), rules.prior.CANDIDATES, strict=True):
            assert result[name].equals(expected[old_candidate["id"]])


def test_each_pool_preserves_seven_unscored_states_and_every_mature_pair():
    _, _, _, audits, _ = prepared()
    for key in rules.VARIANTS:
        rows = study.variant_audits(audits, key)
        assert sum(a["selection"]["reason"] == "unscored_initial_equal_weights" for a in rows) == 7
        mature = [a for a in rows if a["selection"]["scores_available"]]
        assert mature[0]["decision_session"] == "2017-04-28"
        assert all(len(a["selection"]["pairs"]) == 28 for a in mature)


def test_paired_prefix_and_future_changes_preserve_all_four_pasts():
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
    actual, _ = rules.build_targets(pools, calendar, assets, read_json(rules.POLICY))
    old = pools["activity"][5]
    target = old._targets.copy()
    active = ~target.isna().all(axis=1)
    target.loc[active] = [0.098, 0.294, 0.294, 0.294]
    replacement = ArchivedMonthlyExpert(old.identity, target, old._shadow, assets)
    changed = {**pools, "activity": (*pools["activity"][:5], replacement, *pools["activity"][6:])}
    after, _ = rules.build_targets(changed, calendar, assets, read_json(rules.POLICY))
    assert all(actual[name].equals(after[name]) for name in rules.ids("price"))
    assert not actual[rules.ids("activity")[0]].equals(after[rules.ids("activity")[0]])


def test_both_pool_scalar_oracles_and_full_precision_input_records(tmp_path, monkeypatch):
    data, pools, targets, audits, rows = prepared()
    proof = study.verify_targets(data, pools, targets, audits)
    assert proof["each_variant_all28_pairs_verified"]
    assert proof["maximum_independent_target_difference"] < 1e-10
    assert proof["maximum_independent_net_score_difference"] < 1e-8
    path = tmp_path / "actual.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(study, "reference", lambda p: {"path": str(p), "sha256": "synthetic-only"})
    checked = study.verify_retained_records(path, pools, targets, audits)
    assert checked["actual_input_and_resolved_record_count"] == 4 * len(audits)
    assert set(checked["paired_pool_retained_record_checks"]) == set(rules.VARIANTS)


@pytest.mark.parametrize("kind", ["correlation", "cost", "marginal", "pair_count"])
def test_nonselected_pair_and_either_variant_cannot_hide_tampering(kind):
    data, pools, targets, audits, _ = prepared()
    audits = copy.deepcopy(audits)
    active = next(
        a["variants"]["activity"]
        for a in audits
        if a["variants"]["activity"]["selection"]["scores_available"]
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


def test_actual_first_inputs_survive_a_synthetic_selection_failure(monkeypatch):
    pools, calendar, assets = examples()
    rows = []

    def failed(*args):
        raise QuantError("synthetic score stop")

    monkeypatch.setattr(rules, "choose_pair", failed)
    with pytest.raises(QuantError, match="synthetic score"):
        rules.build_targets(pools, calendar, assets, read_json(rules.POLICY), retain=rows.append)
    assert len(rows) == 1 and rows[0]["source_variant"] == "price"
    assert rows[0]["observations"][5]["id"] == rules.EXPERT_IDS["price"][5]


@pytest.mark.parametrize(
    "key,value",
    [
        ("lookback_sessions", 252),
        ("selected_experts", 3),
        ("switch_cost_bps_per_side", 0.0),
        ("actual_weight_trade_band", 0.01),
        ("replaced_slot", 2),
        ("source_variants_are_independent_economic_factors", True),
    ],
)
def test_no_window_cost_band_role_or_independence_rescue(key, value):
    policy = copy.deepcopy(read_json(rules.POLICY))
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)


def test_all_four_real_registrations_precede_first_new_derivation(monkeypatch, tmp_path):
    rows = []
    monkeypatch.setattr(study.lifecycle, "runtime_guard", lambda: None)
    monkeypatch.setattr(study.lifecycle, "research_job", nullcontext)
    monkeypatch.setattr(study, "inputs", lambda: (None, None, {}, None))
    monkeypatch.setattr(study, "FOLDER", tmp_path)
    monkeypatch.setattr(study, "progress", lambda *args, **kwargs: None)
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
        raise QuantError("synthetic stop after all four registrations")

    monkeypatch.setattr(study, "prepare", stop)
    with pytest.raises(QuantError, match="all four"):
        study.register()
    assert {r["candidate_id"] for r in rows if r["kind"] == "technical_failure"} == set(study.IDS)


def test_real_packet_constructor_and_all_four_cohort_metadata(monkeypatch, tmp_path):
    rows = []
    frozen = {"study_id": "synthetic-only"}
    monkeypatch.setattr(study.lifecycle, "runtime_guard", lambda: None)
    monkeypatch.setattr(study.lifecycle, "research_job", nullcontext)
    monkeypatch.setattr(study, "inputs", lambda: (None, None, frozen, None))
    monkeypatch.setattr(study, "FOLDER", tmp_path)
    monkeypatch.setattr(study, "progress", lambda *args, **kwargs: None)
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

    def prepare(*args):
        assert len(rows) == 4
        result = {}
        for i, name in enumerate(study.IDS):
            target, proof = tmp_path / f"target{i}.csv", tmp_path / f"proof{i}.json"
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

    monkeypatch.setattr(
        study,
        "reference",
        lambda p: {
            "path": str(p),
            "sha256": "synthetic-local-reference",
        },
    )
    monkeypatch.setattr(study.base, "checked_reference", lambda r: Path(r["path"]))
    monkeypatch.setattr(study, "prepare", prepare)
    monkeypatch.setattr(
        study.seven.common,
        "retain_receipt",
        lambda path, body, *args: path.write_text(json.dumps(body)),
    )
    study.register()
    for name in study.IDS:
        packet = json.loads(study.PACKETS[name].read_text())
        assert packet["entries"][0]["candidate_id"] == name
        assert packet["complete_fixed_cohort_ids"] == list(study.IDS)
        assert len(packet["cohort_registrations"]) == 4


def test_missing_any_of_four_packets_blocks_first_funded_account(monkeypatch):
    monkeypatch.setattr(study.base, "journal_events", lambda p: [])
    monkeypatch.setattr(
        study, "events", lambda kind, name: [{"sha256": name}] if kind == "registered" else []
    )
    monkeypatch.setattr(
        study,
        "read_json",
        lambda p: {
            "complete_fixed_cohort_ids": list(study.IDS)
            if p != study.PACKETS[study.IDS[3]]
            else list(study.IDS[:3]),
            "cohort_registrations": list(study.IDS),
        },
    )
    monkeypatch.setattr(study, "scoped", lambda *args: lambda *args: {"verified": True})
    with pytest.raises(QuantError, match="every declared"):
        study.cohort_publication({"study_id": "synthetic"}, "0" * 40)


def test_original_source_and_financial_namespaces_unchanged_and_dependency_closure_complete():
    old = study.financial.ID, study.financial.FOLDER, study.financial.PACKET
    context = study.scoped(study.financial.evaluate, study.CANDIDATES[3])
    assert context.__globals__["ID"] == study.IDS[3]
    assert context.__globals__["publication"] is study.cohort_publication
    assert old == (study.financial.ID, study.financial.FOLDER, study.financial.PACKET)
    paths = {r["path"] for r in study.dependencies()}
    assert set(study.OWN).issubset(paths)
    assert "src/us_quant/cash_funded_accounting_v2.py" in paths
