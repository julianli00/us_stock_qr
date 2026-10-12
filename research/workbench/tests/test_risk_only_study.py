from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from us_quant import risk_only_study as study
from us_quant.config import QuantError
from us_quant.storage import digest_json, file_digest, read_json, write_json

ROOT = Path(__file__).parents[1]


@pytest.fixture
def environment(tmp_path, monkeypatch):
    root = tmp_path / "research/workbench"
    root.mkdir(parents=True)
    prepared = root / "data/prepared"
    monkeypatch.setattr(study, "ROOT", root)
    monkeypatch.setattr(study, "PREPARED", prepared)
    proof = root / "evidence/target-proof.json"
    monkeypatch.setattr(study, "TARGET_PROOF", proof)
    monkeypatch.setattr(study, "fingerprint", lambda: "synthetic_fingerprint")
    monkeypatch.setattr(study, "engine_fingerprint", lambda: "synthetic_engine")
    policy = read_json(ROOT / "config/factor-covariance-risk.json")
    write_json(root / "config/factor-covariance-risk.json", policy)
    write_json(root / "evidence/frozen-source.json", {"synthetic": True})
    write_json(prepared / "readiness.json", {"synthetic": True})
    specimens = {}
    for candidate in study.strategy.CANDIDATES:
        spec = {
            "id": candidate["id"], "configuration": candidate,
            "frozen_files": {
                "evidence/frozen-source.json": file_digest(root / "evidence/frozen-source.json"),
            },
        }
        specimens[candidate["id"]] = spec
        write_json(prepared / f"{candidate['id']}-spec.json", spec)
    write_json(proof, {
        "checked_risk_targets": 244,
        "candidate_returns_or_metrics_computed": False,
        "new_registered_candidates": 0,
        "prepared_spec_sha256": {
            identifier: file_digest(prepared / f"{identifier}-spec.json")
            for identifier in specimens
        },
    })
    registrations = {}

    def register(spec_path, readiness_path, receipt_path, **kwargs):
        spec = read_json(spec_path)
        registration = {
            "candidate_id": spec["id"], "spec_sha256": digest_json(spec),
            "registered_at": "2026-10-11T10:40:00+00:00",
        }
        registrations[spec["id"]] = registration
        write_json(receipt_path, {
            "registration": registration,
            "candidate_spec": {
                "path": spec_path.relative_to(root).as_posix(), "sha256": file_digest(spec_path),
            },
            "readiness": {
                "path": readiness_path.relative_to(root).as_posix(), "sha256": file_digest(readiness_path),
            },
        })
        return {
            "receipt": receipt_path.relative_to(root).as_posix(),
            "receipt_sha256": file_digest(receipt_path),
        }

    monkeypatch.setattr(study, "register_or_recover", register)
    packet_path = root / "evidence/registration.json"
    return root, prepared, proof, packet_path, specimens, registrations


def test_register_uses_actual_receipt_not_the_admission_status_object(environment):
    _, _, _, path, _, registrations = environment
    result = study.register(path)
    assert len(result["candidates"]) == 2
    assert all(
        row["registration"] == registrations[row["spec"]["id"]] for row in result["candidates"]
    )
    assert result["new_strategy_evaluations"] == 0
    before = file_digest(path)
    with pytest.raises(QuantError, match="original metadata"):
        study.register(path)
    assert file_digest(path) == before


def test_target_proof_must_precede_any_candidate_registration(environment, monkeypatch):
    _, _, proof, path, _, _ = environment
    value = read_json(proof)
    value["candidate_returns_or_metrics_computed"] = True
    write_json(proof, value)
    monkeypatch.setattr(
        study, "register_or_recover", lambda *args, **kwargs: pytest.fail("proof first"),
    )
    with pytest.raises(QuantError, match="must precede"):
        study.register(path)


def test_unpublished_packet_blocks_before_program_and_accounts(environment, monkeypatch):
    root, _, _, path, _, _ = environment
    study.register(path)

    def unpublished(*args):
        raise QuantError("not actually published")

    monkeypatch.setattr(study, "publication", unpublished)
    monkeypatch.setattr(study, "program", lambda: pytest.fail("no accounts before publication"))
    with pytest.raises(QuantError, match="not actually published"):
        study.evaluate(path, root / "reports/new", "a" * 40)
    assert not (root / "reports/new").exists()


@pytest.mark.parametrize(
    "change", ["missing_candidate", "retuned_method", "bad_runner", "early_returns", "partial_record"],
)
def test_changed_packet_cannot_run_a_relabelled_or_incomplete_study(environment, monkeypatch, change):
    root, _, _, path, _, _ = environment
    packet = study.register(path)
    if change == "missing_candidate":
        packet["candidates"].pop()
    elif change == "retuned_method":
        packet["candidates"][0]["spec"]["configuration"]["method"] = "fit_after_returns"
    elif change == "bad_runner":
        packet["runner_sha256"] = "another_runner"
    elif change == "early_returns":
        packet["new_strategy_evaluations"] = 2
    else:
        del packet["candidates"][0]["readiness"]
    write_json(path, packet)
    monkeypatch.setattr(study, "publication", lambda *args: pytest.fail("bad packet"))
    with pytest.raises(QuantError):
        study.evaluate(path, root / "reports/new", "a" * 40)


@pytest.fixture
def runnable(environment, monkeypatch):
    root, _, _, path, specs, registrations = environment
    study.register(path)
    evaluated = []

    class Database:
        def execute(self, query, arguments):
            identifier = arguments[0]
            row = {"body": json.dumps(specs[identifier])} if "FROM candidates" in query else None
            return SimpleNamespace(fetchone=lambda: row)

    def evaluate(identifier, output):
        assert (output / "publication.json").exists()
        evaluated.append(identifier)
        return {"candidate_id": identifier, "historical_gates_passed": False}

    program = SimpleNamespace(
        db=Database(), evaluate_registered=evaluate, technical_failures=lambda: {},
        close=lambda: None,
    )
    monkeypatch.setattr(study, "program", lambda: program)
    monkeypatch.setattr(
        study, "recorded_registration",
        lambda _, spec, readiness: {"registration": registrations[spec["id"]]},
    )
    monkeypatch.setattr(study, "publication", lambda *args: {"published_commit": "a" * 40})
    return root, path, program, evaluated


def test_actual_known_candidates_run_only_after_durable_publication_fence(runnable):
    root, packet, _, evaluated = runnable
    result = study.evaluate(packet, root / "reports/new", "a" * 40)
    assert evaluated == [candidate["id"] for candidate in study.strategy.CANDIDATES]
    assert result["new_strategy_evaluations"] == 2
    assert result["investment_objective_verified"] is False


def test_forged_registration_event_cannot_evaluate_known_spec(runnable, monkeypatch):
    root, packet, _, evaluated = runnable
    monkeypatch.setattr(
        study, "recorded_registration", lambda *args: {"registration": {"candidate_id": "wrong"}},
    )
    with pytest.raises(QuantError, match="actual original registration event"):
        study.evaluate(packet, root / "reports/new", "a" * 40)
    assert not evaluated and not (root / "reports/new").exists()


def test_existing_partial_output_is_not_deleted_or_recalculated(runnable):
    root, packet, _, evaluated = runnable
    output = root / "reports/new"
    output.mkdir(parents=True)
    partial = output / "partial.csv"
    partial.write_text("preserved incomplete evidence")
    with pytest.raises(QuantError, match="overwrite evidence"):
        study.evaluate(packet, output, "a" * 40)
    assert not evaluated and partial.read_text() == "preserved incomplete evidence"


def test_publication_checks_new_replay_and_imported_core(environment, monkeypatch):
    root, _, _, _, specs, _ = environment
    write_json(root / "config/equity-budget-attribution.json", {"synthetic": True})
    calls = []
    monkeypatch.setattr(
        study.publication_core, "published_inputs",
        lambda *args: calls.append("imported_core") or {"published_commit": "a" * 40},
    )

    def own(actual_root, reference, required, commit):
        assert actual_root == root
        assert {"src/us_quant/strategy_replay.py", "src/us_quant/risk_only_study.py"}.issubset(required)
        calls.append("new_code")
        return {"published_commit": commit}

    monkeypatch.setattr(study, "verify_publication", own)
    result = study.publication({}, [{"spec": spec} for spec in specs.values()], "a" * 40)
    assert calls == ["imported_core", "new_code"]
    assert result["published_imported_core"]["published_commit"] == "a" * 40
