import json

import pytest

from us_quant import ssd_hmm_session_intervention_v2 as study
from us_quant.config import QuantError
from us_quant.storage import read_json


def test_real_schema_binding_is_exact_and_preserves_terminal_v1():
    catalogue, refs = study.inputs()
    assert set(catalogue["variants"]) == {"close", "overnight", "intraday"}
    assert (
        read_json(study.prior.FOLDER / "completion-v1.json")[
            "numeric_intervention_outcomes_computed"
        ]
        is False
    )
    assert read_json(study.POLICY)["schema_bridge"] in refs


def test_verified_split_adapter_adds_only_provenance(monkeypatch, tmp_path):
    path = tmp_path / "split.jsonl"
    original = [
        {"record": "decision_input", "operand": ["0x1.0p-2"]},
        {"record": "decision_resolved", "operand": [0.25]},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in original))
    monkeypatch.setattr(study, "_bindings", lambda: (tmp_path / "base", {path: "close"}))
    pair = next(study._pairs(path))
    assert [row["source_variant"] for row in pair] == ["close", "close"]
    assert [{k: v for k, v in row.items() if k != "source_variant"} for row in pair] == original
    assert [json.loads(line) for line in path.read_text().splitlines()] == original


def test_adapter_refuses_unknown_path_and_existing_label(monkeypatch, tmp_path):
    path = tmp_path / "split.jsonl"
    path.write_text('{"source_variant":"overnight"}\n{}\n')
    monkeypatch.setattr(study, "_bindings", lambda: (tmp_path / "base", {path: "close"}))
    with pytest.raises(QuantError):
        next(study._pairs(tmp_path / "foreign"))
    with pytest.raises(QuantError):
        next(study._pairs(path))


def test_v2_cannot_change_numerical_law_or_tolerance():
    policy = read_json(study.POLICY)
    study.validate(policy)
    for key, value in (
        ("target_reconciliation_tolerance", 1e-8),
        ("source_slot", 3),
        ("new_financial_evaluations", 1),
    ):
        with pytest.raises(QuantError):
            study.validate({**policy, key: value})


def test_v2_dependency_closure_retains_failed_v1_and_unchanged_kernel():
    paths = {row["path"] for row in study.dependencies()}
    assert set(study.OWN).issubset(paths)
    assert "src/us_quant/hmm_session_intervention_v1.py" in paths
