import copy
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from test_strategy_expert_pool_eight_v1 import archived_experts as prior_experts

from us_quant import hmm_expert_substitution_v1 as rules
from us_quant import ssd_hmm_expert_substitution_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json, utc_now
from us_quant.strategy_experts_v1 import ArchivedMonthlyExpert, ExpertIdentity


def examples(changed_after=None):
    experts, calendar, assets = prior_experts(changed_after=changed_after)
    old = experts[2]
    new = ArchivedMonthlyExpert(
        ExpertIdentity(rules.EXPERT_IDS[2], rules.MECHANISMS[2], "2026-10-29T06:58:47Z"),
        old._targets,
        old._shadow,
        assets,
    )
    return study.substitute(experts, new), calendar, assets


def prepared():
    experts, calendar, assets = examples()
    records = []
    targets, audits = rules.build_targets(
        experts, calendar, assets, read_json(rules.POLICY), retain=records.append
    )
    return (
        SimpleNamespace(close=pd.DataFrame(index=calendar, columns=assets)),
        experts,
        targets,
        audits,
        records,
    )


def test_only_family_slot_changes_and_original_pair_combine_functions_are_identical():
    experts, _, _ = examples()
    assert tuple(e.identity.identifier for e in experts) == rules.EXPERT_IDS
    assert len(experts) == 8
    assert rules.MECHANISMS == rules.prior.MECHANISMS
    assert rules.choose_pair is rules.prior.choose_pair
    assert rules.combine is rules.prior.combine
    assert [i for i in range(8) if rules.EXPERT_IDS[i] != rules.prior.EXPERT_IDS[i]] == [2]


def test_family_substitution_cannot_append_reorder_or_silently_replace_another_source():
    old, _, _ = prior_experts()
    changed, _, _ = examples()
    with pytest.raises(QuantError):
        study.substitute((*old, old[0]), changed[2])
    with pytest.raises(QuantError):
        study.substitute(old, changed[0])


def test_same_original_law_reproduces_control_values_when_only_synthetic_label_changes():
    experts, calendar, assets = examples()
    old, _, _ = prior_experts()
    actual, _ = rules.build_targets(experts, calendar, assets, read_json(rules.POLICY))
    control, _ = rules.prior.build_targets(old, calendar, assets, read_json(rules.prior.POLICY))
    for new_id, old_candidate in zip(study.IDS, rules.prior.CANDIDATES, strict=True):
        assert actual[new_id].equals(control[old_candidate["id"]])


def test_seven_unscored_states_and_every_mature_pair_are_preserved():
    _, _, _, audits, _ = prepared()
    assert sum(a["selection"]["reason"] == "unscored_initial_equal_weights" for a in audits) == 7
    mature = [a for a in audits if a["selection"]["scores_available"]]
    assert mature[0]["decision_session"] == "2017-04-28"
    assert all(len(a["selection"]["pairs"]) == 28 for a in mature)
    assert all(
        "symmetric_garch_growth_bil_v1" not in p["pair"]
        for a in mature
        for p in a["selection"]["pairs"]
    )


def test_prefix_and_future_source_changes_do_not_change_substituted_past():
    experts, calendar, assets = examples()
    changed, _, _ = examples(changed_after=pd.Timestamp("2017-04-28"))
    targets, audits = rules.build_targets(experts, calendar, assets, read_json(rules.POLICY))
    later, after = rules.build_targets(changed, calendar, assets, read_json(rules.POLICY))
    early, first = rules.build_targets(
        experts, calendar[calendar <= "2017-04-28"], assets, read_json(rules.POLICY)
    )
    for name in study.IDS:
        assert early[name].equals(targets[name].loc[:"2017-04-28"])
        assert later[name].loc[:"2017-04-28"].equals(early[name])
    assert first == [a for a in audits if a["decision_session"] <= "2017-04-28"]
    assert [a for a in after if a["decision_session"] <= "2017-04-28"] == first


def test_new_source_full_pair_scalar_oracle_and_retained_records(tmp_path):
    data, experts, targets, audits, records = prepared()
    proof = study.verify_targets(data, experts, targets, audits)
    assert proof["maximum_independent_target_difference"] < 1e-10
    assert proof["maximum_independent_net_score_difference"] < 1e-8
    path = tmp_path / "new-source-inputs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    checked = study.verify_retained_records(path, experts, targets, audits)
    assert checked["actual_input_and_resolved_record_count"] == 2 * len(audits)


@pytest.mark.parametrize("kind", ["correlation", "cost", "marginal", "pair_count"])
def test_nonselected_pair_or_missing_record_is_not_hidden(kind):
    data, experts, targets, audits, _ = prepared()
    audits = copy.deepcopy(audits)
    active = next(a for a in audits if a["selection"]["scores_available"])
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
        study.verify_targets(data, experts, targets, audits)


def test_actual_inputs_survive_a_synthetic_new_selection_error(monkeypatch):
    experts, calendar, assets = examples()
    rows = []

    def failed(*args):
        raise QuantError("synthetic score stop")

    monkeypatch.setattr(rules, "choose_pair", failed)
    with pytest.raises(QuantError, match="synthetic score"):
        rules.build_targets(experts, calendar, assets, read_json(rules.POLICY), retain=rows.append)
    assert len(rows) == 1
    assert rows[0]["observations"][2]["id"] == "causal_two_state_growth_bil_v1"


@pytest.mark.parametrize(
    "key,value",
    [
        ("lookback_sessions", 252),
        ("selected_experts", 3),
        ("switch_cost_bps_per_side", 0.0),
        ("actual_weight_trade_band", 0.01),
        ("source_variants_are_independent_economic_factors", True),
    ],
)
def test_unchanged_source_pool_laws_cannot_be_retuned(key, value):
    policy = copy.deepcopy(read_json(rules.POLICY))
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)


def test_both_real_registrations_precede_first_new_derivation(monkeypatch, tmp_path):
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
        raise QuantError("synthetic stop after both real registrations")

    monkeypatch.setattr(study, "prepare", stop)
    with pytest.raises(QuantError, match="after both"):
        study.register()
    assert {r["candidate_id"] for r in rows if r["kind"] == "technical_failure"} == set(study.IDS)


def test_real_packet_constructor_and_complete_cohort_metadata(monkeypatch, tmp_path):
    rows = []
    frozen = {"study_id": "synthetic-only"}
    monkeypatch.setattr(study.lifecycle, "runtime_guard", lambda: None)
    monkeypatch.setattr(study.lifecycle, "research_job", nullcontext)
    monkeypatch.setattr(study, "inputs", lambda: (None, None, frozen, None))
    monkeypatch.setattr(study, "progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        study, "PACKETS", {name: tmp_path / f"{i}.json" for i, name in enumerate(study.IDS)}
    )
    monkeypatch.setattr(
        study,
        "events",
        lambda kind, name: [r for r in rows if r["kind"] == kind and r["candidate_id"] == name],
    )
    monkeypatch.setattr(
        study.base,
        "matching_events",
        lambda kind, name: [r for r in rows if r["kind"] == kind and r["candidate_id"] == name],
    )

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
        assert len(rows) == 2
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

    def ref(path):
        return {"path": str(path), "sha256": "synthetic-local-reference"}

    monkeypatch.setattr(study, "reference", ref)
    monkeypatch.setattr(study.base, "checked_reference", lambda row: Path(row["path"]))
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
        assert len(packet["cohort_registrations"]) == 2


def test_missing_other_packet_blocks_every_funded_candidate(monkeypatch):
    monkeypatch.setattr(study.base, "journal_events", lambda path: [])
    monkeypatch.setattr(
        study, "events", lambda kind, name: [{"sha256": name}] if kind == "registered" else []
    )
    monkeypatch.setattr(
        study,
        "read_json",
        lambda path: {
            "complete_fixed_cohort_ids": list(study.IDS)
            if path == study.PACKETS[study.IDS[0]]
            else [study.IDS[1]],
            "cohort_registrations": list(study.IDS),
        },
    )
    monkeypatch.setattr(study, "scoped", lambda *args: lambda *args: {"verified": True})
    with pytest.raises(QuantError, match="every declared"):
        study.cohort_publication({"study_id": "synthetic"}, "0" * 40)


def test_source_and_financial_global_namespaces_are_not_mutated():
    original = study.financial.ID, study.financial.FOLDER, study.financial.PACKET
    scoped = study.scoped(study.financial.evaluate, study.CANDIDATES[1])
    assert scoped.__globals__["ID"] == study.IDS[1]
    assert scoped.__globals__["publication"] is study.cohort_publication
    assert original == (study.financial.ID, study.financial.FOLDER, study.financial.PACKET)
    paths = {r["path"] for r in study.dependencies()}
    assert set(study.OWN).issubset(paths)
    assert "src/us_quant/cash_funded_accounting_v2.py" in paths
