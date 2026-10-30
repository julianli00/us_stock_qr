from __future__ import annotations

import ast
import copy
from contextlib import nullcontext
from pathlib import Path

import pytest

from us_quant import ssd_expert_headroom_stress_v1 as study
from us_quant.config import QuantError


def test_cached_loader_never_invokes_closed_entrypoints(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail(
            "A closed source strategy, target, account or whole-study verifier was replayed."
        )

    for owner in (study.previous, study.previous.previous, study.participation):
        for name in ("initialize", "register", "evaluate", "public_result", "source_inputs"):
            if hasattr(owner, name):
                monkeypatch.setattr(owner, name, forbidden)
        if hasattr(owner, "prepare"):
            monkeypatch.setattr(owner, "prepare", forbidden)
        if hasattr(owner.rules, "build_targets"):
            monkeypatch.setattr(owner.rules, "build_targets", forbidden)
    monkeypatch.setattr(study.rules, "run", forbidden)
    before = len(study.base.journal_events(study.base.JOURNAL))
    data, targets, policy, parent, entries, refs, benchmarks = study.source_inputs()
    assert len(study.base.journal_events(study.base.JOURNAL)) == before
    assert set(targets) == set(study.IDS)
    assert set(entries) == set(study.SOURCE_SPECS)
    assert len(refs) >= 946 and len(data.close.loc["2016-10-06":]) == 2512
    assert parent["study_id"] == "ssd_expert_path_baskets_20261030_v1"
    assert policy["source_registrations"][study.rules.CANDIDATES[1]["source_candidate_id"]]
    assert benchmarks["benchmark_accounts"] == 8
    assert [
        study.reference(
            study.base.checked_reference(policy["nominal_sources"][c["source_candidate_id"]])
        )
        for c in study.rules.CANDIDATES
    ] == [policy["nominal_sources"][c["source_candidate_id"]] for c in study.rules.CANDIDATES]


def test_complete_local_import_closure_is_bound():
    code, policies = study.dependency_inputs()
    bound = {r["path"] for r in code} | set(study.OWN)
    pending = {study.OWN[1], study.OWN[2], "src/us_quant/__init__.py"}
    visited = set()
    while pending:
        relative = pending.pop()
        if relative in visited:
            continue
        visited.add(relative)
        assert relative in bound, f"Unbound local import: {relative}"
        for node in ast.walk(ast.parse((study.ROOT / relative).read_text())):
            names = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
                names.extend(f"{node.module}.{alias.name}" for alias in node.names)
            for name in names:
                if name == "us_quant" or name.startswith("us_quant."):
                    pending.add("src/us_quant/__init__.py")
                    path = Path("src", *name.split(".")).with_suffix(".py")
                    if (study.ROOT / path).is_file():
                        pending.add(path.as_posix())
    assert "src/us_quant/expert_master_protection_v1.py" in visited
    assert len(visited) > 90
    assert "config/expert-master-protection-v1.json" in {r["path"] for r in policies}
    assert study.OWN[0] not in {r["path"] for r in policies}
    for ref in [*code, *policies]:
        study.base.checked_reference(ref)


def proof_fixture(monkeypatch):
    candidate = study.rules.CANDIDATES[0]
    nominal = {"path": "data/retained/nominal.csv", "sha256": "a" * 64}
    proof = {"path": "data/retained/proof.json", "sha256": "b" * 64}
    source = {
        "registered_event_sha256": "c" * 64,
        "target_proof": {"targets": nominal, "proof": proof},
    }
    frozen = {
        "target_proof_role": study.PROOF_ROLE,
        "state_contingent_issued_targets_precomputed": False,
        "source_nominal_target_bindings": {
            candidate["id"]: {
                "source_candidate_id": candidate["source_candidate_id"],
                "targets": nominal,
                "proof": proof,
            }
        },
        "execution_adapter": {"path": study.OWN[1], "sha256": "d" * 64},
        "validation_evidence": [{"path": "reports/tests.xml", "sha256": "e" * 64}],
    }
    checked = []
    monkeypatch.setattr(study.base, "checked_reference", lambda r: checked.append(r))
    return candidate, frozen, source, {"sha256": "f" * 64}, checked


def test_closed_loop_proof_binds_only_old_nominal_targets(monkeypatch):
    candidate, frozen, source, registration, checked = proof_fixture(monkeypatch)
    result = study.transition_binding(candidate, frozen, source, registration)
    assert result["source_nominal_targets"] == source["target_proof"]["targets"]
    assert result["source_nominal_target_proof"] == source["target_proof"]["proof"]
    assert result["controller_transition"] == frozen["execution_adapter"]
    assert result["candidate_accounts_or_metrics_computed"] is False
    assert result["state_contingent_issued_targets_precomputed"] is False
    assert result["original_source_signal_recomputed"] is False
    assert checked == [source["target_proof"]["targets"], source["target_proof"]["proof"]]


@pytest.mark.parametrize("damage", ["future_targets", "different_source", "wrong_role"])
def test_transition_binding_rejects_relabelled_state(damage, monkeypatch):
    candidate, frozen, source, registration, _ = proof_fixture(monkeypatch)
    if damage == "future_targets":
        frozen["state_contingent_issued_targets_precomputed"] = True
    elif damage == "different_source":
        frozen["source_nominal_target_bindings"][candidate["id"]]["source_candidate_id"] = "alien"
    else:
        frozen["target_proof_role"] = "final_targets"
    with pytest.raises(QuantError, match="precomputed"):
        study.transition_binding(candidate, frozen, source, registration)


def test_real_probe_preparation_starts_after_both_registrations(monkeypatch, tmp_path):
    events, calls = [], []
    frozen = {"study_id": "synthetic_generative_stress"}
    source_entries = {
        c["source_candidate_id"]: {"target_proof": {"targets": {}, "proof": {}}}
        for c in study.rules.CANDIDATES
    }

    def append(directory, kind, candidate, body):
        calls.append((kind, candidate))
        event = {
            "kind": kind,
            "candidate_id": candidate,
            "body": body,
            "sha256": f"{len(events):064x}",
        }
        events.append(event)
        return event

    def probes(data, nominal, candidate, specification, source, registration, directory):
        assert {e["candidate_id"] for e in events if e["kind"] == "registered"} == set(study.IDS)
        assert not any(e["kind"] == "path_started" for e in events)
        assert specification is frozen
        calls.append(("synthetic_probes", candidate["id"]))
        return {"actual_main_NAV_peak_or_issued_path_computed": False}

    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "research_job", lambda path: nullcontext())
    monkeypatch.setattr(
        study,
        "inputs",
        lambda: (None, {name: None for name in study.IDS}, frozen, source_entries, {}),
    )
    monkeypatch.setattr(
        study,
        "matching_events",
        lambda kind, name: [e for e in events if e["kind"] == kind and e["candidate_id"] == name],
    )
    monkeypatch.setattr(study, "append_event", append)
    monkeypatch.setattr(study, "transition_proof", probes)
    monkeypatch.setattr(study, "PROOFS", tmp_path / "synthetic-proofs")
    monkeypatch.setattr(study, "PACKET", tmp_path / "synthetic-packet.json")
    monkeypatch.setattr(study, "new_output_directory", lambda p: p.mkdir(parents=True))
    monkeypatch.setattr(study, "reference", lambda p: {"path": p.name, "sha256": "a" * 64})
    monkeypatch.setattr(study.common, "packet_entry", lambda c, f: {"candidate_id": c["id"]})
    monkeypatch.setattr(study.common, "retain_receipt", lambda *args: None)
    monkeypatch.setattr(study, "checked_packet", lambda f: {})
    monkeypatch.setattr(study.current, "progress", lambda *args, **kwargs: None)
    study.register()
    assert calls[:2] == [("registered", name) for name in study.IDS]
    assert sum(kind == "synthetic_probes" for kind, _ in calls) == 2
    assert sum(kind == "target_proof" for kind, _ in calls) == 2


def publication_fixture(monkeypatch, found=None):
    calls = []
    commit, explicit, imported = "a" * 40, study.OWN[1], "src/us_quant/backtest.py"
    frozen = {
        "study_id": "synthetic_stress",
        "code": [{"path": p, "sha256": "b" * 64} for p in (explicit, imported)],
        "policies": [],
    }

    def dependencies(value, sha):
        assert value is frozen and sha == commit
        assert {r["path"] for r in value["code"]} == {explicit, imported}
        calls.append("all_dependencies")
        return {"unchanged_pinned_import_files_verified": 1}

    def packet(root, ref, paths, sha):
        assert root == study.ROOT and paths == {explicit} and sha == commit
        calls.append("packet")
        return {"published_commit": sha}

    monkeypatch.setattr(study, "published_inputs", dependencies)
    monkeypatch.setattr(study, "verify_publication", packet)
    monkeypatch.setattr(study, "reference", lambda p: {"path": "packet.json", "sha256": "c" * 64})
    monkeypatch.setattr(study, "read_json", lambda p: {"files": ["research/workbench/" + explicit]})
    monkeypatch.setattr(study, "matching_events", lambda kind: found or [])
    monkeypatch.setattr(study, "append_event", lambda *args: calls.append("fence"))
    return frozen, commit, calls


def test_complete_dependencies_precede_the_new_publication_fence(monkeypatch):
    frozen, commit, calls = publication_fixture(monkeypatch)
    receipt = study.publication(frozen, commit)
    assert calls == ["all_dependencies", "packet", "fence"]
    assert receipt["complete_dependency_publication"]["unchanged_pinned_import_files_verified"] == 1


def test_import_failure_or_later_commit_cannot_create_a_new_fence(monkeypatch):
    frozen, commit, calls = publication_fixture(monkeypatch)

    def refuse(*args):
        raise QuantError("Unlisted actual dependency")

    monkeypatch.setattr(study, "published_inputs", refuse)
    with pytest.raises(QuantError, match="Unlisted"):
        study.publication(frozen, commit)
    assert calls == []
    found = [
        {"body": {"study_id": "synthetic_stress", "publication": {"published_commit": "d" * 40}}}
    ]
    frozen, commit, calls = publication_fixture(monkeypatch, found)
    with pytest.raises(QuantError, match="original publication"):
        study.publication(frozen, commit)
    assert calls == ["all_dependencies", "packet"]


def test_failed_publication_cannot_start_main_capital(monkeypatch, tmp_path):
    calls = []

    def blocked(*args):
        calls.append("publication")
        raise QuantError("Publication not verified.")

    def forbidden(*args, **kwargs):
        pytest.fail("New main capital started without an actual original publication fence.")

    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "research_job", lambda path: nullcontext())
    monkeypatch.setattr(study, "inputs", lambda: (None, None, {}, None, {}))
    monkeypatch.setattr(study, "checked_packet", lambda f: {})
    monkeypatch.setattr(study, "RESULTS", tmp_path / "absent.json")
    monkeypatch.setattr(study, "publication", blocked)
    monkeypatch.setattr(study.rules, "run", forbidden)
    monkeypatch.setattr(study, "failure", lambda *args: calls.append("failure"))
    with pytest.raises(QuantError, match="Publication not verified"):
        study.evaluate("a" * 40)
    assert calls == ["publication", "failure"]


def test_deadline_failure_preserves_partials_without_new_events(monkeypatch):
    progress, events = [], []
    monkeypatch.setattr(study, "utc_now", lambda: "2026-10-31T16:00:00+00:00")
    monkeypatch.setattr(study, "append_event", lambda *args: events.append(args))
    monkeypatch.setattr(study.current, "progress", lambda *args, **kwargs: progress.append(args))
    study.failure("synthetic_stage", study.IDS, QuantError("deadline"))
    assert events == []
    assert progress == [("expert_headroom_stress_expired",)]


def test_canonical_export_preserves_original_paths_without_private_rows(monkeypatch):
    frozen = {"study_id": "synthetic_stress", "protocol": {"path": "config/original.json"}}
    paths = {}
    benchmarks = []
    for years in (10, 5):
        for scenario in ("base", "stress"):
            key = f"{years}y-{scenario}"
            paths[key] = {
                "path_key": key,
                "metrics": {"cagr": 0.1, "sharpe": 0.5, "max_drawdown": 0.2},
                "execution": {"total_cost_usd": 123.0, "model_order_count": 40},
                "gates": {"sharpe_gt_1": False},
                "all_point_gates_pass": False,
                "excess_net_cagr_vs_spy": -0.02,
                "excess_net_cagr_vs_qqq": -0.03,
                "stress_vs_base_cost_qqq_diagnostic": None,
                "independent_bt_max_equity_difference_usd": 1e-10,
                "independent_main_wealth_feedback": {"main_drawdown_guarantee": False},
            }
            benchmarks.extend(
                {
                    "years": years,
                    "scenario": scenario,
                    "symbol": symbol,
                    "metrics": {"cagr": cagr},
                }
                for symbol, cagr in (("SPY", 0.12), ("QQQ", 0.13))
            )
    actual = {
        name: {
            "candidate_id": name,
            "status": "rejected",
            "paths": copy.deepcopy(paths),
            "actual_factor_participation": {},
            "reviewed_at": "2026-10-30T16:00:00+00:00",
            "all_four_paths_same_candidate_and_frozen_parameters": True,
        }
        for name in study.IDS
    }
    saved = {
        "candidates": copy.deepcopy(actual),
        "finished_at": "2026-10-30T16:00:00+00:00",
        "actual_journal_event_count": 1,
        "actual_journal_head_sha256": "b" * 64,
        "publication": {"published_commit": "c" * 40},
        "global_counts_at_completion": {},
    }
    captured = []
    monkeypatch.setattr(
        study, "inputs", lambda: (None, None, frozen, None, {"benchmarks": benchmarks})
    )
    monkeypatch.setattr(study, "checked_packet", lambda f: None)
    monkeypatch.setattr(study, "read_json", lambda p: saved)
    monkeypatch.setattr(study, "states", lambda f: actual)
    monkeypatch.setattr(study.base, "journal_events", lambda p: [])
    monkeypatch.setattr(
        study.common, "counts", lambda events: {"new_completed_strategy_evaluations": 2}
    )
    monkeypatch.setattr(study.common, "retain_receipt", lambda p, b, v: captured.append(b))
    monkeypatch.setattr(study, "reference", lambda p: {"path": p.name, "sha256": "d" * 64})
    study.public_result()
    body = captured[0]
    assert len(body["reviews"]) == 2
    for review in body["reviews"]:
        assert {p["path_key"] for p in review["paths"]} == set(paths)
        for path in review["paths"]:
            assert path["metrics"] == paths[path["path_key"]]["metrics"]
            assert path["execution"] == paths[path["path_key"]]["execution"]
            assert path["gates"] == paths[path["path_key"]]["gates"]
    assert body["original_records_reclassified"] is False
    assert body["raw_prices_documents_targets_or_daily_accounts_exported"] is False
    assert body["order_authority"] is body["investment_objective_verified"] is False
    assert not any(key in str(body) for key in ("known_main_NAV_USD", "pending_executions"))


def test_duplicate_completions_and_unregistered_failures_are_not_success(monkeypatch):
    duplicate = [{"body": {"path_key": "10y-base", "receipt": {}}}] * 2
    monkeypatch.setattr(
        study, "matching_events", lambda k, n: duplicate if k == "path_completed" else []
    )
    with pytest.raises(QuantError, match="Duplicate actual completion"):
        study.states({})
    events = []
    monkeypatch.setattr(study, "utc_now", lambda: "2026-10-30T17:00:00+00:00")
    monkeypatch.setattr(study, "matching_events", lambda k, n: [])
    monkeypatch.setattr(study, "append_event", lambda *args: events.append(args))
    monkeypatch.setattr(study.current, "progress", lambda *args, **kwargs: None)
    study.failure("unregistered", study.IDS, QuantError("not registered"))
    assert events == []
