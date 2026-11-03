from __future__ import annotations

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pandas as pd
import pytest
from test_strategy_expert_pool_eight_v1 import archived_experts

from us_quant import ssd_eight_expert_meta_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json


def prepared():
    experts, calendar, assets = archived_experts()
    records = []
    targets, audits = study.rules.build_targets(
        experts, calendar, assets, read_json(study.rules.POLICY), retain=records.append
    )
    return (
        SimpleNamespace(close=pd.DataFrame(index=calendar, columns=assets)),
        experts,
        targets,
        audits,
        records,
    )


def test_independent_scalar_oracle_checks_every_one_of_twenty_eight_pairs():
    data, experts, targets, audits, _ = prepared()
    proof = study.verify_targets(data, experts, targets, audits)
    assert proof["maximum_independent_target_difference"] < 1e-10
    assert proof["maximum_independent_net_score_difference"] < 1e-8


@pytest.mark.parametrize("kind", ["correlation", "cost", "marginal", "pair_count"])
def test_nonselected_pair_risk_cost_or_missing_record_is_not_hidden(kind):
    data, experts, targets, audits, _ = prepared()
    audits = copy.deepcopy(audits)
    row = next(a for a in audits if a["selection"]["scores_available"])
    pairs = row["selection"]["pairs"]
    other = next(
        p
        for p in pairs
        if p["pair"] != row["selection"]["selected_pair"] and "selection_score_key" in p
    )
    if kind == "correlation":
        other["training_correlation"] += 0.01
    elif kind == "cost":
        other["estimated_incremental_switch_cost_fraction"] += 0.01
    elif kind == "marginal":
        other["training_marginal_variance_contributions"][0] += 0.01
    else:
        pairs.pop()
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, targets, audits)


def test_retained_full_precision_source_records_match_every_expert_and_both_targets(tmp_path):
    _, experts, targets, audits, records = prepared()
    import json

    path = tmp_path / "retained.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    proof = study.verify_retained_records(path, experts, targets, audits)
    assert proof["actual_input_and_resolved_record_count"] == 2 * len(audits)
    assert proof["actual_input_resolved_records_match_source_and_targets"] is True


def test_new_source_scope_does_not_mutate_original_pure_or_financial_globals():
    before = study.financial.ID, study.financial.FOLDER, study.financial.PACKET
    fn = study.scoped(study.financial.evaluate, study.CANDIDATES[1])
    assert fn.__globals__["ID"] == study.IDS[1]
    assert fn.__globals__["PACKET"] == study.PACKETS[study.IDS[1]]
    assert fn.__globals__["publication"] is study.cohort_publication
    assert before == (study.financial.ID, study.financial.FOLDER, study.financial.PACKET)
    deps = {r["path"] for r in study.dependencies()}
    assert set(study.OWN).issubset(deps)
    assert "src/us_quant/november_research_lifecycle_v1.py" in deps
    assert "src/us_quant/cash_funded_accounting_v2.py" in deps


def test_both_real_registrations_precede_the_first_new_score_or_target(tmp_path, monkeypatch):
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

    def append(directory, kind, name, body):
        row = {"kind": kind, "candidate_id": name, "body": body, "sha256": name}
        rows.append(row)
        return row

    monkeypatch.setattr(study.base, "append_event", append)

    def stop_after_order_check(*args):
        assert [r["candidate_id"] for r in rows if r["kind"] == "registered"] == list(study.IDS)
        raise QuantError("Synthetic stop after both actual registrations")

    monkeypatch.setattr(study, "prepare", stop_after_order_check)
    with pytest.raises(QuantError, match="after both"):
        study.register()
    assert {r["candidate_id"] for r in rows if r["kind"] == "technical_failure"} == set(study.IDS)


def test_one_missing_cohort_packet_prevents_all_funding(monkeypatch):
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
        study.cohort_publication({"study_id": "test"}, "0" * 40)
