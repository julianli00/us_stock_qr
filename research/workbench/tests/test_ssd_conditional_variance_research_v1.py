from __future__ import annotations

import ast
import inspect

import pandas as pd
import pytest

from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_conditional_variance_research_v1 as study
from us_quant import ssd_research_v1 as base
from us_quant.config import QuantError
from us_quant.storage import read_json, utc_now, write_json


@pytest.fixture
def journal(tmp_path, monkeypatch):
    path = tmp_path / "events"
    monkeypatch.setattr(base, "ROOT", tmp_path)
    monkeypatch.setattr(base, "JOURNAL", path)
    monkeypatch.setattr(base, "storage_guard", lambda: None)
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(common, "progress", lambda *args, **kwargs: None)
    base.append_event(path, "baseline", None, {"legacy": {"synthetic": True}})
    return path


def setup_registration(journal, monkeypatch):
    frozen = {"study_id": "synthetic", "legacy": {"synthetic": True}}
    monkeypatch.setattr(study, "TARGETS", journal.parent / "targets")
    monkeypatch.setattr(study, "PREPARED", journal.parent / "prepared.json")
    monkeypatch.setattr(study, "PACKET", journal.parent / "packet.json")
    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "inputs", lambda: (None, None, None, frozen, None, None))
    monkeypatch.setattr(study, "check_frozen", lambda value: None)
    monkeypatch.setattr(
        common,
        "counts",
        lambda rows: {
            "new_registered_candidates": 28,
            "new_completed_strategy_evaluations": 27,
            "new_technical_failures": 1,
            "new_pending_candidates": 0,
        },
    )
    return frozen


def synthetic_proof():
    return (
        {
            name: pd.DataFrame({"BIL": [0.98]}, index=pd.to_datetime(["2020-01-31"]))
            for name in study.IDS
        },
        {name: [] for name in study.IDS},
        {
            name: {
                "checked_at": utc_now(),
                "candidate_accounts_or_metrics_computed": False,
                "original_source_signal_recomputed": False,
            }
            for name in study.IDS
        },
    )


def test_both_registrations_precede_fitting_and_completed_preparation_never_replays(
    journal,
    monkeypatch,
):
    frozen = setup_registration(journal, monkeypatch)
    calls = []

    def proof(*args):
        assert all(len(base.matching_events("registered", name)) == 1 for name in study.IDS)
        calls.append(True)
        return synthetic_proof()

    monkeypatch.setattr(study, "prove_targets", proof)
    study.register()
    study.register()
    assert calls == [True]
    packet = read_json(study.PACKET)
    assert len(packet["entries"]) == 2
    packet["candidate_accounts_or_metrics_computed"] = True
    write_json(study.PACKET, packet)
    with pytest.raises(QuantError, match="pre-account"):
        study.checked_packet(frozen)


def test_prepared_bundle_is_reused_without_target_or_fit_regeneration(journal, monkeypatch):
    frozen = setup_registration(journal, monkeypatch)
    monkeypatch.setattr(study, "prove_targets", lambda *args: synthetic_proof())
    first = study.prepare_bundle(None, None, None, frozen)
    monkeypatch.setattr(
        study,
        "prove_targets",
        lambda *args: pytest.fail("No repeated completed preparation"),
    )
    assert study.prepare_bundle(None, None, None, frozen) == first


@pytest.mark.parametrize("expired", [False, True])
def test_failure_once_and_expiry_preserves_every_pending_trial(journal, monkeypatch, expired):
    for name in study.IDS:
        base.append_event(journal, "registered", name, {})
    monkeypatch.setattr(
        study,
        "utc_now",
        lambda: "2026-10-31T15:59:00Z" if expired else "2026-10-29T00:00:00Z",
    )
    study.failure("synthetic", study.IDS, QuantError("stop"))
    study.failure("synthetic", study.IDS, QuantError("stop"))
    assert all(
        len(base.matching_events("technical_failure", name)) == int(not expired)
        for name in study.IDS
    )


def test_fit_failure_retains_partials_and_both_trials_without_seed_retry(journal, monkeypatch):
    setup_registration(journal, monkeypatch)
    partial = journal.parent / "partial-fit.json"

    def fail(*args):
        assert all(base.matching_events("registered", name) for name in study.IDS)
        write_json(partial, {"synthetic_failed_fit": True})
        raise QuantError("single initialization failed")

    monkeypatch.setattr(study, "prove_targets", fail)
    with pytest.raises(QuantError, match="initialization"):
        study.register()
    assert read_json(partial) == {"synthetic_failed_fit": True}
    assert all(len(base.matching_events("technical_failure", name)) == 1 for name in study.IDS)
    with pytest.raises(QuantError, match="failed preparation"):
        study.register()


def test_each_fit_requires_both_actual_registered_specs(journal, monkeypatch):
    frozen = setup_registration(journal, monkeypatch)
    first, second = study.rules.CANDIDATES
    base.append_event(journal, "registered", first["id"], {"candidate": first, "frozen": frozen})
    with pytest.raises(QuantError, match="Both actual"):
        study.registration_hashes(frozen)
    base.append_event(journal, "registered", second["id"], {"candidate": second, "frozen": frozen})
    assert set(study.registration_hashes(frozen)) == set(study.IDS)


def test_unresolved_started_fit_cannot_be_retried(journal, monkeypatch):
    frozen = setup_registration(journal, monkeypatch)
    for candidate in study.rules.CANDIDATES:
        base.append_event(
            journal, "registered", candidate["id"], {"candidate": candidate, "frozen": frozen}
        )
    day = pd.Timestamp("2020-01-31")
    path = study.fit_path(study.IDS[0], day, "fit-attempts")
    write_json(path, {"original_unresolved_attempt": True})

    def attempted(data, calendar, policy, *, retain_fit, before_fit, load_fit):
        assert load_fit(study.IDS[0], day) is None
        before_fit(study.IDS[0], day, False)
        pytest.fail("An unresolved fit was retried")

    monkeypatch.setattr(study.rules, "build_targets", attempted)
    with pytest.raises(QuantError, match="unresolved"):
        study.prove_targets(None, None, None, {}, frozen)
    assert read_json(path) == {"original_unresolved_attempt": True}


def test_publication_requires_kernel_and_actual_hmm_reference(journal, monkeypatch):
    calls = []
    frozen = {
        "study_id": "synthetic",
        "daily_proxy_registration": {"path": "prior", "sha256": "a" * 64},
    }
    monkeypatch.setattr(base, "published_inputs", lambda *args: {})
    monkeypatch.setattr(base, "reference", lambda path: {"path": str(path), "sha256": "a" * 64})

    def verify(root, reference, required, commit):
        calls.append(required)
        return {"published_commit": commit, "registration_packet": reference}

    monkeypatch.setattr(study, "verify_publication", verify)
    study.publication(frozen, {}, "b" * 40)
    assert "src/us_quant/conditional_variance_v1.py" in calls[-1]
    assert "tests/test_conditional_variance_v1.py" in calls[-1]
    assert "evidence/hmm_growth_bil_20261029_results_v1.json" in calls[-1]
    with pytest.raises(QuantError, match="original publication"):
        study.publication(frozen, {}, "c" * 40)


def test_completed_evaluation_never_refits_or_replays_accounts(journal, monkeypatch):
    frozen = {"study_id": "synthetic"}
    actual = {name: {"status": "rejected", "paths": {}} for name in study.IDS}
    path = journal.parent / "results.json"
    monkeypatch.setattr(study, "RESULTS", path)
    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "inputs", lambda: (None, None, None, frozen, None, None))
    monkeypatch.setattr(study, "checked_packet", lambda value: {})
    monkeypatch.setattr(study, "states", lambda value: actual)
    monkeypatch.setattr(study, "publication", lambda *args: pytest.fail("No repeated publication"))
    monkeypatch.setattr(study.rules.model, "fit", lambda *args: pytest.fail("No repeated fits"))
    monkeypatch.setattr(
        study.source,
        "evaluate_path",
        lambda *args: pytest.fail("No account replay"),
    )
    write_json(
        path, {"frozen_specification_sha256": study.digest_json(frozen), "candidates": actual}
    )
    assert study.evaluate("unused") == base.reference(path)


def test_account_and_prefix_stages_never_optimize_and_use_original_execution():
    tree = ast.parse(inspect.getsource(study.evaluate))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(
        isinstance(node.func, ast.Attribute) and node.func.attr in {"fit", "build_targets"}
        for node in calls
    )
    funded = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "evaluate_path"
    ]
    assert len(funded) == 1
    assert ast.unparse(funded[0]) == (
        "source.evaluate_path(data, targets, frozen, identifier, window, scenario, benchmarks)"
    )
    preparation = ast.parse(inspect.getsource(study.prove_targets))
    build = [
        node
        for node in ast.walk(preparation)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "build_targets"
    ]
    assert len(build) == 1
    assert {item.arg for item in build[0].keywords} == {"retain_fit", "before_fit", "load_fit"}
