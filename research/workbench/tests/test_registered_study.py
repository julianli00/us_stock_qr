from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from us_quant import registered_study as study
from us_quant.config import QuantError
from us_quant.storage import digest_json, file_digest, read_json, write_json


@pytest.fixture
def environment(tmp_path, monkeypatch):
    root = tmp_path / "research/workbench"
    root.mkdir(parents=True)
    monkeypatch.setattr(study, "ROOT", root)
    monkeypatch.setattr(study, "fingerprint", lambda: "synthetic_runner")
    monkeypatch.setattr(study, "engine_fingerprint", lambda: "synthetic_engine")
    monkeypatch.setattr(study, "registered_generator", lambda *args: ("synthetic_only", {}))
    policy = {
        "schema_version": 1, "data_scope": "factor_etf_portfolio", "new_configurations": 2,
        "new_economic_factor_definitions": 0,
        "candidates": [{"id": "synthetic_first"}, {"id": "synthetic_second"}],
        "methodology": {"order_authority": False, "automatic_live_deployment": False},
    }
    policy_path = root / "config/policy.json"
    write_json(policy_path, policy)
    prepared = root / "data/prepared"
    write_json(prepared / "readiness.json", {"synthetic_test_fixture": True})
    specs, registrations = {}, {}
    for configuration in policy["candidates"]:
        spec = {
            "id": configuration["id"], "configuration": configuration,
            "frozen_files": {"config/policy.json": file_digest(policy_path)},
        }
        write_json(prepared / f"{configuration['id']}-spec.json", spec)
        specs[spec["id"]] = spec
    proof_path = root / "evidence/proof.json"
    write_json(proof_path, {
        "candidate_returns_or_metrics_computed": False,
        "prepared_spec_sha256": {name: file_digest(prepared / f"{name}-spec.json") for name in specs},
    })

    def admission(spec_path, readiness_path, receipt_path, **kwargs):
        spec = read_json(spec_path)
        registration = {
            "candidate_id": spec["id"], "spec_sha256": digest_json(spec),
            "registered_at": "2026-10-11T11:00:00+00:00",
        }
        registrations[spec["id"]] = registration
        write_json(receipt_path, {
            "registration": registration,
            "candidate_spec": {"path": spec_path.relative_to(root).as_posix(), "sha256": file_digest(spec_path)},
            "readiness": {"path": readiness_path.relative_to(root).as_posix(), "sha256": file_digest(readiness_path)},
        })
        return {"receipt": receipt_path.relative_to(root).as_posix()}

    monkeypatch.setattr(study, "register_or_recover", admission)
    return root, policy_path, prepared, proof_path, root / "evidence/packet.json", specs, registrations


def test_register_uses_two_exact_spec_and_proof_references_without_accounts(environment):
    root, policy, prepared, proof, packet, _, _ = environment
    result = study.register(policy, prepared, proof, packet)
    assert len(result["candidates"]) == 2 and result["new_strategy_evaluations"] == 0
    assert result["registered_before_strategy_accounts"] is True
    assert not (root / "reports/accounts").exists()
    before = file_digest(packet)
    with pytest.raises(QuantError):
        study.register(policy, prepared, proof, packet)
    assert file_digest(packet) == before


@pytest.mark.parametrize("problem", ["count", "authority", "family", "bad_id", "bad_metadata"])
def test_generic_runner_cannot_weaken_a_frozen_study_scope(environment, problem):
    _, path, _, _, _, _, _ = environment
    policy = read_json(path)
    if problem == "count":
        policy["candidates"].pop()
    elif problem == "authority":
        policy["methodology"]["order_authority"] = True
    elif problem == "family":
        policy["new_economic_factor_definitions"] = 2
    elif problem == "bad_id":
        policy["candidates"][0]["id"] = "../outside"
    else:
        policy["methodology"] = None
    write_json(path, policy)
    with pytest.raises(QuantError):
        study.policy_input(path)


@pytest.mark.parametrize("problem", ["outcomes_seen", "bad_proof_map", "changed_spec"])
def test_preaccount_proof_is_checked_before_admission(environment, monkeypatch, problem):
    _, policy, prepared, proof, packet, _, _ = environment
    value = read_json(proof)
    if problem == "outcomes_seen":
        value["candidate_returns_or_metrics_computed"] = True
    elif problem == "bad_proof_map":
        value["prepared_spec_sha256"] = None
    else:
        value["prepared_spec_sha256"]["synthetic_first"] = "different_hash"
    write_json(proof, value)
    monkeypatch.setattr(study, "register_or_recover", lambda *args, **kwargs: pytest.fail("proof first"))
    with pytest.raises(QuantError):
        study.register(policy, prepared, proof, packet)


def test_actual_publication_is_required_before_program_or_account_output(environment, monkeypatch):
    root, policy, prepared, proof, packet, _, _ = environment
    study.register(policy, prepared, proof, packet)

    def unpublished(*args):
        raise QuantError("actual publication not observed")

    monkeypatch.setattr(study, "publication", unpublished)
    monkeypatch.setattr(study, "program", lambda: pytest.fail("publication precedes accounts"))
    with pytest.raises(QuantError, match="not observed"):
        study.evaluate(packet, root / "reports/accounts", "a" * 40)
    assert not (root / "reports/accounts").exists()


@pytest.fixture
def runnable(environment, monkeypatch):
    root, policy, prepared, proof, packet, specimens, registrations = environment
    study.register(policy, prepared, proof, packet)
    evaluated = []

    def evaluate(identifier, output):
        assert (output / "publication.json").exists()
        evaluated.append(identifier)
        return {"candidate_id": identifier, "historical_gates_passed": False}

    program = SimpleNamespace(
        db=SimpleNamespace(execute=lambda *args: SimpleNamespace(fetchone=lambda: None)),
        evaluate_registered=evaluate, close=lambda: None, technical_failures=lambda: {},
    )
    monkeypatch.setattr(study, "program", lambda: program)
    monkeypatch.setattr(
        study, "recorded_registration",
        lambda _, spec, ready: {"registration": registrations[spec["id"]]},
    )
    monkeypatch.setattr(study, "publication", lambda *args: {"published_commit": "a" * 40})
    return root, packet, program, evaluated


def test_generic_runner_reuses_original_four_path_evaluator_after_durable_fence(runnable):
    root, packet, _, evaluated = runnable
    result = study.evaluate(packet, root / "reports/accounts", "a" * 40)
    assert evaluated == ["synthetic_first", "synthetic_second"]
    assert result["new_strategy_evaluations"] == 2
    assert result["investment_objective_verified"] is False


def test_forged_event_cannot_start_a_known_candidate(runnable, monkeypatch):
    root, packet, _, evaluated = runnable
    monkeypatch.setattr(study, "recorded_registration", lambda *args: {"registration": {}})
    with pytest.raises(QuantError, match="original candidate registration event"):
        study.evaluate(packet, root / "reports/accounts", "a" * 40)
    assert not evaluated


def test_existing_partial_evidence_is_never_deleted_or_overwritten(runnable):
    root, packet, _, evaluated = runnable
    directory = root / "reports/accounts"
    directory.mkdir(parents=True)
    partial = directory / "partial.csv"
    partial.write_text("preserve original incomplete account")
    with pytest.raises(QuantError, match="overwrite evidence"):
        study.evaluate(packet, directory, "a" * 40)
    assert not evaluated and partial.read_text() == "preserve original incomplete account"


@pytest.mark.parametrize("problem", ["bad_reference", "bad_entry", "bad_frozen", "bad_code", "new_outcomes"])
def test_changed_registration_is_an_explicit_error_before_publication(environment, monkeypatch, problem):
    root, policy, prepared, proof, packet, _, _ = environment
    value = study.register(policy, prepared, proof, packet)
    if problem == "bad_reference":
        value["policy_reference"]["path"] = None
    elif problem == "bad_entry":
        del value["candidates"][0]["readiness"]
    elif problem == "bad_frozen":
        value["candidates"][0]["spec"]["frozen_files"] = None
    elif problem == "bad_code":
        value["runner_sha256"] = "different_runner"
    else:
        value["new_strategy_evaluations"] = 2
    write_json(packet, value)
    monkeypatch.setattr(study, "publication", lambda *args: pytest.fail("changed receipt"))
    with pytest.raises(QuantError):
        study.evaluate(packet, root / "reports/accounts", "a" * 40)


def test_every_study_publishes_original_source_code_plus_generic_runner_and_proof(environment, monkeypatch):
    root, policy, _, proof, _, _, _ = environment
    write_json(root / "config/equity-budget-attribution.json", {})
    calls = []
    monkeypatch.setattr(
        study.publication_core, "published_inputs",
        lambda *args: calls.append("inherited_core") or {"published_commit": "a" * 40},
    )

    def verify(actual_root, reference, required, commit):
        assert actual_root == root
        assert {"src/us_quant/registered_study.py", "config/policy.json", "evidence/proof.json"}.issubset(required)
        calls.append("runner_policy_proof")
        return {"published_commit": commit}

    monkeypatch.setattr(study, "verify_publication", verify)
    result = study.publication(
        {}, [], {"path": policy.relative_to(root).as_posix()},
        {"path": proof.relative_to(root).as_posix()}, "a" * 40,
    )
    assert calls == ["inherited_core", "runner_policy_proof"]
    assert result["published_imported_core"]["published_commit"] == "a" * 40
