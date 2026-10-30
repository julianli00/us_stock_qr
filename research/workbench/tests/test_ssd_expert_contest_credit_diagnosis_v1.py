import json
from copy import deepcopy

import pytest

from us_quant import ssd_expert_contest_credit_diagnosis_v1 as driver
from us_quant.config import QuantError


def records():
    allocations = {name: {"saved_posterior": name} for name in driver.rules.IDS}
    audit = {"decision_session": "2017-04-28", "allocations": allocations}
    return [
        {"stage": "input", "session": "2017-04-28"},
        *[
            {
                "stage": "allocation_resolved",
                "session": "2017-04-28",
                "candidate_id": name,
                "allocation": allocations[name],
            }
            for name in driver.rules.IDS
        ],
        {"stage": "resolved", "session": "2017-04-28", "audit": audit},
    ]


def save(path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def test_record_adapter_preserves_exact_four_record_groups(tmp_path):
    path = tmp_path / "source.jsonl"
    rows = records()
    save(path, rows)
    assert list(driver.read_decisions(path)) == [(rows[0], rows[3]["audit"])]
    assert path.read_text().count("\n") == 4


@pytest.mark.parametrize("kind", ["truncated", "duplicate", "changed", "bad_json", "not_object"])
def test_record_adapter_rejects_truncation_duplicate_or_altered_allocation(tmp_path, kind):
    path = tmp_path / "source.jsonl"
    rows = deepcopy(records())
    if kind == "truncated":
        rows.pop()
    elif kind == "duplicate":
        rows[2] = deepcopy(rows[1])
    elif kind == "changed":
        rows[2]["allocation"] = {"saved_posterior": "altered"}
    elif kind == "not_object":
        rows[2] = []
    save(path, rows)
    if kind == "bad_json":
        path.write_text("{bad json}\n")
    with pytest.raises(QuantError):
        list(driver.read_decisions(path))


def test_source_parsing_does_not_call_closed_entrypoints(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A saved credit diagnosis called a closed financial entrypoint.")

    for name in ("prepare", "register", "evaluate", "inputs"):
        monkeypatch.setattr(driver.source, name, forbidden)
    monkeypatch.setattr(driver.source.rules, "build_targets", forbidden)
    path = tmp_path / "source.jsonl"
    save(path, records())
    assert len(list(driver.read_decisions(path))) == 1


def test_public_report_does_not_export_monthly_vectors_or_raw_records():
    result = {
        "completed_at": "2026-10-30T00:00:00+00:00",
        "analysis_intent": {"sha256": "a" * 64},
        "source_decisions_preserved": 121,
        "source_trace_records_preserved": 484,
        "complete_next_month_pairs": 119,
        "initial_partial_excluded": [],
        "final_forecast_without_full_outcome": {},
        "aggregates": {},
        "journal_derived_counts_unchanged": {},
        "private_per_month_rows": {"path": "PRIVATE"},
        "raw_source_prices": "PRIVATE",
    }
    public = driver.public_report(result)
    assert "PRIVATE" not in json.dumps(public)
    assert public["new_financial_trials"] == public["new_funded_paths"] == 0
    assert public["credit_forecast_interpretation_introduced_by_this_diagnostic"] is True
    assert public["independent_forward_validation"] is False


def test_second_actual_analysis_attempt_is_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "FOLDER", tmp_path)
    monkeypatch.setattr(driver, "runtime_guard", lambda: None)
    (tmp_path / "analysis-attempt-v1.json").write_text("{}")
    with pytest.raises(QuantError, match="first actual diagnostic attempt"):
        driver.diagnose()
