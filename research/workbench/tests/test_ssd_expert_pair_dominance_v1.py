import copy
import json
import runpy
from contextlib import nullcontext
from dataclasses import replace
from types import FunctionType, SimpleNamespace

import numpy as np
import pytest
from test_expert_pair_dominance_v1 import archived_experts

from us_quant import ssd_expert_pair_dominance_v1 as study
from us_quant import ssd_expert_pair_menu_extension_v1 as menu
from us_quant.config import QuantError
from us_quant.storage import read_json


@pytest.fixture(scope="module")
def synthetic():
    experts, calendar, assets = archived_experts()
    records = []
    targets, audits = study.rules.build_targets(
        experts, calendar, assets, read_json(study.rules.POLICY), retain=records.append
    )
    return SimpleNamespace(close=targets[study.IDS[0]]), experts, targets, audits, records


def test_scalar_and_boolean_oracle_checks_all_pair_evidence(synthetic):
    checked = study.verify_targets(*synthetic[:4])
    assert checked["maximum_independent_target_difference"] < 1e-12
    assert checked["maximum_independent_expert_weight_difference"] < 1e-12
    assert checked["maximum_independent_net_score_difference"] < 1e-10
    assert checked["independent_strict_dominance_gate_and_pair_state_verified"] is True
    assert all(row["unscored_initial_decisions"] == 7 for row in checked["per_rule"].values())


@pytest.mark.parametrize(
    "field",
    [
        "train_net_risk_score",
        "validation_net_risk_score",
        "training_correlation",
        "estimated_incremental_switch_cost_fraction",
        "expert_weights",
        "training_marginal_variance_contributions",
        "gate_boolean",
        "qualified_challengers",
        "incumbent_pair_before",
        "confidence_claim",
    ],
)
def test_oracle_rejects_nonselected_pair_and_gate_mutations(synthetic, field):
    data, experts, targets, audits, _ = synthetic
    changed = copy.deepcopy(audits)
    rows = changed[study.IDS[0]]
    if field == "incumbent_pair_before":
        rows[8]["switch_evidence"][field] = None
    elif field == "confidence_claim":
        rows[8]["switch_evidence"]["confidence_interval_or_significance_claimed"] = True
    elif field == "qualified_challengers":
        rows[8]["switch_evidence"][field].append(list(study.rules.EXPERT_IDS[:2]))
    elif field == "gate_boolean":
        record = rows[8]["switch_evidence"]["dominance_comparisons"][0]
        record["qualifies"] = not record["qualifies"]
    else:
        evidence = rows[7]["switch_evidence"]["pair_assessments"]
        pair = next(row for row in evidence["pairs"] if row["pair"] != evidence["selected_pair"])
        if isinstance(pair[field], list):
            pair[field][0] = np.nan
        else:
            pair[field] += 0.1
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, targets, changed)


def test_retained_rule_incumbent_inputs_and_resolved_records_are_exact(synthetic, tmp_path):
    _, experts, targets, audits, records = synthetic
    path = tmp_path / "synthetic-dominance-records.jsonl"
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in records))
    checked = study.verify_retained_records(path, experts, targets, audits)
    assert checked["actual_input_and_resolved_record_count"] == 4 * len(audits[study.IDS[0]])
    changed = copy.deepcopy(records)
    changed[0]["incumbent_pair"] = list(study.rules.EXPERT_IDS[:2])
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in changed))
    with pytest.raises(QuantError, match="incumbent"):
        study.verify_retained_records(path, experts, targets, audits)


def test_degenerate_mature_fallback_is_unscored_without_missing_clocks():
    experts, calendar, assets = archived_experts()

    class FlatExpert:
        def __init__(self, original):
            self.original = original

        def observe(self, day, lookback):
            item = self.original.observe(day, lookback)
            return replace(item, net_returns=item.risk_free.copy())

    flat = tuple(FlatExpert(expert) for expert in experts)
    targets, audits = study.rules.build_targets(
        flat, calendar, assets, read_json(study.rules.POLICY)
    )
    checked = study.verify_targets(
        SimpleNamespace(close=targets[study.IDS[0]]), flat, targets, audits
    )
    assert all(row["mature_scored_decisions"] == 0 for row in checked["per_rule"].values())
    assert all(row["mature_degenerate_equal_fallbacks"] > 0 for row in checked["per_rule"].values())
    assert all(
        audit["switch_evidence"]["incumbent_pair_after"] is None
        for rows in audits.values()
        for audit in rows
    )


def test_cached_source_loader_never_replays_original_pipeline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A completed source/control/admission/diagnostic pipeline was replayed.")

    for module, names in (
        (study.source, ("initialize", "prepare", "register", "evaluate")),
        (study.source.rules, ("build_targets",)),
        (menu, ("diagnose", "freeze", "checked_intent")),
        (study.source.admitted, ("admit",)),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    data, experts, _, _, refs, clocks, _, _ = study.source_inputs()
    assert tuple(expert.identity.identifier for expert in experts) == study.rules.EXPERT_IDS
    assert len(refs) >= 1237
    assert all(expert.observe("2016-10-05", 126).target is not None for expert in experts)
    assert all(
        len(expert.observe(data.close.index[-1], 2512).net_returns) == 2512 for expert in experts
    )
    assert clocks[-1]["decision_information_clock"] == study.rules.CLOCK


def test_shared_publication_wrappers_do_not_recurse_or_mutate_source(monkeypatch):
    old_own = study.source.OWN
    old_rules = study.source.rules.POLICY
    refs = study.dependencies()
    assert study.source.OWN is old_own and study.source.rules.POLICY is old_rules
    assert set(study.OWN).issubset({ref["path"] for ref in refs})
    calls = []
    frozen = {"study_id": "test_dominance", "code": [], "policies": []}
    monkeypatch.setattr(study, "published_inputs", lambda f, c: {"published_commit": c})
    monkeypatch.setattr(study, "read_json", lambda path: {"files": []})
    monkeypatch.setattr(study, "verify_publication", lambda *args: {"published_commit": "a" * 40})
    monkeypatch.setattr(
        study, "reference", lambda path: {"path": "packet.json", "sha256": "b" * 64}
    )
    monkeypatch.setattr(study, "matching_events", lambda kind: [])
    monkeypatch.setattr(study, "append_event", lambda *args: calls.append("actual_fence"))
    receipt = study.publication(frozen, "a" * 40)
    assert receipt["published_commit"] == "a" * 40 and calls == ["actual_fence"]


def test_failed_publication_cannot_start_main_account(monkeypatch, tmp_path):
    calls = []

    def blocked(*args):
        calls.append("publication")
        raise QuantError("Publication is not verified.")

    def forbidden(*args, **kwargs):
        pytest.fail("A main account started before actual verified preregistration.")

    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "research_job", lambda path: nullcontext())
    monkeypatch.setattr(study, "inputs", lambda: (None, None, {}, None))
    monkeypatch.setattr(study, "checked_packet", lambda frozen: {})
    monkeypatch.setattr(study, "RESULTS", tmp_path / "no-results.json")
    monkeypatch.setattr(study, "publication", blocked)
    monkeypatch.setattr(study.execution, "evaluate_path", forbidden)
    monkeypatch.setattr(study, "failure", lambda *args: calls.append("retained_failure"))
    with pytest.raises(QuantError, match="not verified"):
        study.evaluate("a" * 40)
    assert calls == ["publication", "retained_failure"]


def test_canonical_export_keeps_every_actual_path_and_no_private_rows(monkeypatch):
    namespace = runpy.run_path(str(study.ROOT / "tests/test_ssd_six_expert_meta_v1.py"))
    original_test = namespace["test_canonical_export_preserves_all_paths_without_daily_rows"]
    FunctionType(original_test.__code__, {**namespace, "study": study})(monkeypatch)


def test_original_monthly_execution_is_preserved():
    assert study.execution is study.source.execution
    assert study.execution.evaluate_path is study.source.execution.evaluate_path
