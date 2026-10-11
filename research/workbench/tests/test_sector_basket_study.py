from __future__ import annotations

from pathlib import Path

import pytest

from us_quant import sector_basket_study as study
from us_quant.config import QuantError
from us_quant.storage import digest_json, file_digest, read_json, write_json

ROOT = Path(__file__).parents[1]


@pytest.fixture
def environment(tmp_path, monkeypatch):
    root = tmp_path / "research/workbench"
    prepared = root / "data/prepared"
    root.mkdir(parents=True)
    monkeypatch.setattr(study, "ROOT", root)
    monkeypatch.setattr(study, "PREPARED", prepared)
    proof = root / "evidence/target-proof.json"
    monkeypatch.setattr(study, "TARGET_PROOF", proof)
    monkeypatch.setattr(study, "fingerprint", lambda: "synthetic_fingerprint")
    monkeypatch.setattr(study, "engine_fingerprint", lambda: "synthetic_engine")
    write_json(root / "config/sector-basket-strategy.json", read_json(ROOT / "config/sector-basket-strategy.json"))
    write_json(root / "evidence/frozen.json", {"synthetic": True})
    write_json(prepared / "readiness.json", {"synthetic": True})
    for candidate in study.strategy.CANDIDATES:
        write_json(prepared / f"{candidate['id']}-spec.json", {
            "id": candidate["id"], "configuration": candidate,
            "frozen_files": {"evidence/frozen.json": file_digest(root / "evidence/frozen.json")},
        })
    write_json(proof, {
        "checked_defined_targets": 244, "candidate_returns_or_metrics_computed": False,
        "prepared_spec_sha256": {
            candidate["id"]: file_digest(prepared / f"{candidate['id']}-spec.json")
            for candidate in study.strategy.CANDIDATES
        },
    })

    def admission(spec_path, readiness_path, receipt_path, **kwargs):
        spec = read_json(spec_path)
        write_json(receipt_path, {
            "registration": {"candidate_id": spec["id"], "spec_sha256": digest_json(spec)},
            "readiness": {"path": readiness_path.relative_to(root).as_posix(), "sha256": file_digest(readiness_path)},
            "candidate_spec": {"path": spec_path.relative_to(root).as_posix(), "sha256": file_digest(spec_path)},
        })
        return {"receipt": receipt_path.relative_to(root).as_posix(), "receipt_sha256": file_digest(receipt_path)}

    monkeypatch.setattr(study, "register_or_recover", admission)
    return root, root / "evidence/registration.json"


def test_fixed_pair_registration_does_not_count_performance_and_cannot_overwrite(environment):
    root, packet = environment
    result = study.register(packet)
    assert len(result["candidates"]) == 2 and result["new_strategy_evaluations"] == 0
    assert not (root / "reports/sector-basket-strategy-20261011").exists()
    before = file_digest(packet)
    with pytest.raises(QuantError):
        study.register(packet)
    assert file_digest(packet) == before


def test_unpublished_pair_cannot_create_accounts_or_directory(environment, monkeypatch):
    root, packet = environment
    study.register(packet)

    def unpublished(*args):
        raise QuantError("actual remote publication absent")

    monkeypatch.setattr(study, "publication", unpublished)
    monkeypatch.setattr(study, "program", lambda: pytest.fail("publication must precede account admission"))
    with pytest.raises(QuantError, match="publication absent"):
        study.evaluate(packet, root / "reports/new", "a" * 40)
    assert not (root / "reports/new").exists()


@pytest.mark.parametrize("problem", ["missing", "method", "fingerprint", "outcomes", "receipt"])
def test_a_changed_scope_or_receipt_cannot_be_evaluated(environment, monkeypatch, problem):
    root, path = environment
    packet = study.register(path)
    if problem == "missing":
        packet["candidates"].pop()
    elif problem == "method":
        packet["candidates"][0]["spec"]["configuration"]["dollar_guard"] = True
    elif problem == "fingerprint":
        packet["runner_sha256"] = "changed_runner"
    elif problem == "outcomes":
        packet["new_strategy_evaluations"] = 2
    else:
        del packet["candidates"][0]["readiness"]
    write_json(path, packet)
    monkeypatch.setattr(study, "publication", lambda *args: pytest.fail("changed packet"))
    with pytest.raises(QuantError):
        study.evaluate(path, root / "reports/new", "a" * 40)
