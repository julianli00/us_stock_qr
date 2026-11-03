from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
from test_strategy_expert_pool_v5 import archived_experts

from us_quant import expert_target_consensus_v1 as rules
from us_quant import ssd_expert_target_consensus_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json, write_json


def synthetic_preparation():
    experts, dates, assets = archived_experts()
    records = []
    targets, audits = rules.build_targets(
        experts, dates, assets, read_json(rules.POLICY), retain=records.append
    )
    data = SimpleNamespace(close=targets[study.ID])
    return data, experts, targets, audits, records


def test_new_orchestrator_keeps_old_guards_and_pure_monthly_accounting_unchanged():
    import us_quant.dual_benchmark_v1 as old
    import us_quant.expert_initial_monthly_execution_v1 as execution

    before = old.runtime_guard
    assert study.execution.run is execution.run
    assert study.lifecycle.runtime_guard is not old.runtime_guard
    assert study.lifecycle.DEADLINE > old.DEADLINE
    with pytest.raises(QuantError, match="expired"):
        old.before_deadline(study.lifecycle.START)
    study.dependencies()
    assert old.runtime_guard is before
    assert execution.runtime_guard is before


def test_import_capture_includes_new_and_old_actual_helpers_without_mutating_sources():
    inputs = study.dependencies()
    paths = {row["path"] for row in inputs}
    assert set(study.OWN).issubset(paths)
    assert "src/us_quant/cash_funded_accounting_v2.py" in paths
    assert "src/us_quant/strategy_experts_v1.py" in paths
    assert "config/dual-benchmark-v1.json" in paths
    assert len(paths) == len(inputs)


def test_independent_real_record_checker_handles_all_current_synthetic_decisions():
    data, experts, targets, audits, records = synthetic_preparation()
    proof = study.verify_preparation(data, experts, targets, audits, records)
    assert proof["decision_count"] == len(audits[study.ID])
    assert proof["retained_input_resolved_records"] == 2 * proof["decision_count"]
    assert proof["maximum_independent_integer_target_difference"] <= 1e-12


@pytest.mark.parametrize(
    "kind", ["input", "hex", "chosen", "weights", "clock", "resolved", "objective"]
)
def test_retention_or_oracle_mismatch_stops_preparation(kind):
    data, experts, targets, audits, records = synthetic_preparation()
    audits, records = copy.deepcopy(audits), copy.deepcopy(records)
    if kind == "input":
        records[0]["observations"][0]["target"][0] += 0.01
    elif kind == "hex":
        records[0]["observations"][0]["target_binary64_hex"][0] = (0.99).hex()
    elif kind == "resolved":
        records[1]["combined_target"][0] += 0.01
    else:
        row = audits[study.ID][0]
        if kind == "chosen":
            row["consensus"]["selected_expert_ids"] = []
        elif kind == "weights":
            row["expert_weights"][0] += 0.01
        elif kind == "objective":
            row["consensus"]["objective_exact_fractions"][0]["numerator"] = "-1"
        else:
            row["decision_information_clock"] = "at_16_00_ET"
        records[1]["audit"] = row
    with pytest.raises(QuantError):
        study.verify_preparation(data, experts, targets, audits, records)


def test_unresolved_preparation_attempt_is_never_replayed(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "FOLDER", tmp_path)
    monkeypatch.setattr(study, "PREPARED", tmp_path / "prepared.json")
    write_json(tmp_path / "preparation-attempt-v1.json", {"partial": True})
    with pytest.raises(QuantError, match="unfinished preparation"):
        study.prepare(None, None, {})


def test_terminal_registration_cannot_start_targets(tmp_path, monkeypatch):
    monkeypatch.setattr(study.lifecycle, "runtime_guard", lambda: None)
    from contextlib import nullcontext

    monkeypatch.setattr(study.lifecycle, "research_job", nullcontext)
    monkeypatch.setattr(study, "inputs", lambda: (None, None, {}, None))
    monkeypatch.setattr(study, "events", lambda kind: [{}] if kind == "technical_failure" else [])
    monkeypatch.setattr(study, "prepare", lambda *args: pytest.fail("A terminal version restarted"))
    with pytest.raises(QuantError, match="terminal"):
        study.register()


def test_failed_publication_never_starts_a_main_account(tmp_path, monkeypatch):
    from contextlib import nullcontext

    monkeypatch.setattr(study.lifecycle, "runtime_guard", lambda: None)
    monkeypatch.setattr(study.lifecycle, "research_job", nullcontext)
    monkeypatch.setattr(study, "inputs", lambda: (None, None, {}, None))
    monkeypatch.setattr(study, "events", lambda kind: [])
    monkeypatch.setattr(study, "RESULTS", tmp_path / "results.json")

    def failed(*args):
        raise QuantError("Synthetic failed publication")

    monkeypatch.setattr(study, "publication", failed)
    monkeypatch.setattr(study, "evaluate_path", lambda *args: pytest.fail("A main account started"))
    with pytest.raises(QuantError, match="failed publication"):
        study.evaluate("0" * 40)


def test_completed_account_recovery_reads_receipt_without_running_any_account(monkeypatch):
    monkeypatch.setattr(
        study,
        "events",
        lambda kind: [
            {"body": {"path_key": "10y-base", "receipt": {"path": "existing", "sha256": "old"}}}
        ]
        if kind == "path_completed"
        else [],
    )
    monkeypatch.setattr(study.base, "verify_path", lambda *args: {"preserved": True})
    monkeypatch.setattr(
        study.execution, "run", lambda *args: pytest.fail("A completed path replayed")
    )
    result = study.evaluate_path(None, None, {}, {"years": 10}, {"id": "base"}, None)
    assert result == {"preserved": True}


def test_november_progress_never_overwrites_october_global_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "PROGRESS", tmp_path / "new-november/progress.json")
    monkeypatch.setattr(study.base, "storage_guard", lambda: None)
    monkeypatch.setattr(study.base, "journal_events", lambda path: [])
    monkeypatch.setattr(study, "current_counts", lambda events: {"actual_count": 675})
    study.progress("test", next_step="pending")
    assert read_json(study.PROGRESS)["actual_count"] == 675
    assert read_json(study.PROGRESS)["authorization_id"] == "november-research-20261103-v1"
