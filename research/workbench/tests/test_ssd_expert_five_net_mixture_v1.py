from __future__ import annotations

import copy
from contextlib import nullcontext

import pytest

from us_quant import legacy_profit_expert_v1 as admitted
from us_quant import ssd_expert_five_net_mixture_v1 as study
from us_quant.config import QuantError


def test_cached_five_source_loader_does_not_run_completed_financial_entrypoints(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A completed admission, financial model or old learner was replayed.")

    for name in ("initialize", "register", "evaluate", "public_result"):
        monkeypatch.setattr(study.previous, name, forbidden)
    monkeypatch.setattr(admitted, "admit", forbidden)
    monkeypatch.setattr(admitted, "load_profit_expert", forbidden)
    monkeypatch.setattr(study.rules.original.TetrahedralMoments, "update", forbidden)
    data, experts, policy, parent, refs, _ = study.source_inputs()
    assert tuple(e.identity.identifier for e in experts) == study.rules.EXPERT_IDS
    assert len(experts) == 5
    assert all(e.observe("2016-10-05", 2512).target is not None for e in experts)
    assert all(len(e.observe("2016-10-05", 2512).net_returns) == 0 for e in experts)
    assert all(len(e.observe(data.close.index[-1], 2512).net_returns) == 2512 for e in experts)
    assert parent["policy"] == policy["source_five_policy"]
    assert any(r["path"].endswith("expert_net_mixture_20261029_results_v1.json") for r in refs)


def publication_fixture(monkeypatch, found=None):
    calls = []
    commit, explicit, imported = "a" * 40, study.OWN[1], "src/us_quant/backtest.py"
    frozen = {
        "study_id": "synthetic_five_net",
        "code": [{"path": p, "sha256": "b" * 64} for p in (explicit, imported)],
        "policies": [],
    }

    def dependencies(value, sha):
        assert value is frozen and sha == commit
        assert {r["path"] for r in value["code"]} == {explicit, imported}
        calls.append("full_dependencies")
        return {"published_commit": sha, "unchanged_pinned_import_files_verified": 1}

    def packet(root, ref, required, sha):
        assert root == study.ROOT and sha == commit
        assert ref["path"] == study.PACKET.relative_to(study.ROOT).as_posix()
        assert required == {explicit}
        calls.append("explicit_packet")
        return {"published_commit": sha}

    monkeypatch.setattr(study, "published_inputs", dependencies)
    monkeypatch.setattr(study, "verify_publication", packet)
    monkeypatch.setattr(
        study,
        "reference",
        lambda p: {
            "path": p.relative_to(study.ROOT).as_posix(),
            "sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(study, "read_json", lambda p: {"files": [f"research/workbench/{explicit}"]})
    monkeypatch.setattr(study, "matching_events", lambda kind: found or [])
    monkeypatch.setattr(study, "append_event", lambda *args: calls.append("actual_fence"))
    return frozen, commit, calls


def test_publication_checks_full_dependencies_before_packet_and_fence(monkeypatch):
    frozen, commit, calls = publication_fixture(monkeypatch)
    result = study.publication(frozen, commit)
    assert calls == ["full_dependencies", "explicit_packet", "actual_fence"]
    assert result["complete_dependency_publication"]["unchanged_pinned_import_files_verified"] == 1


def test_unknown_dependency_blocks_packet_and_fence(monkeypatch):
    frozen, commit, calls = publication_fixture(monkeypatch)

    def refuse(*args):
        raise QuantError("Unknown frozen dependency.")

    monkeypatch.setattr(study, "published_inputs", refuse)
    with pytest.raises(QuantError, match="Unknown"):
        study.publication(frozen, commit)
    assert calls == []


def test_publication_cannot_move_to_a_post_outcome_commit(monkeypatch):
    found = [
        {
            "body": {
                "study_id": "synthetic_five_net",
                "publication": {"published_commit": "d" * 40},
            }
        }
    ]
    frozen, commit, calls = publication_fixture(monkeypatch, found)
    with pytest.raises(QuantError, match="original publication"):
        study.publication(frozen, commit)
    assert calls == ["full_dependencies", "explicit_packet"]


def test_failed_publication_cannot_start_a_funded_path(monkeypatch, tmp_path):
    calls = []

    def refuse(*args):
        calls.append("publication")
        raise QuantError("Publication is not verified.")

    def forbidden(*args, **kwargs):
        pytest.fail("A funded account started before verified publication.")

    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "research_job", lambda path: nullcontext())
    monkeypatch.setattr(study, "inputs", lambda: (None, None, {}, None))
    monkeypatch.setattr(study, "checked_packet", lambda f: {})
    monkeypatch.setattr(study, "RESULTS", tmp_path / "not-completed.json")
    monkeypatch.setattr(study, "publication", refuse)
    monkeypatch.setattr(study.execution, "evaluate_path", forbidden)
    monkeypatch.setattr(study, "failure", lambda *a: calls.append("retained_failure"))
    with pytest.raises(QuantError, match="not verified"):
        study.evaluate("a" * 40)
    assert calls == ["publication", "retained_failure"]


def test_duplicate_completed_path_cannot_be_chosen_by_dictionary_overwrite(monkeypatch):
    duplicate = [{"body": {"path_key": "10y-base", "receipt": {}}}] * 2
    monkeypatch.setattr(
        study, "matching_events", lambda kind, name: duplicate if kind == "path_completed" else []
    )
    with pytest.raises(QuantError, match="Duplicate completion"):
        study.states({})


def test_expired_failure_does_not_append_new_research_events(monkeypatch):
    events, progress = [], []
    monkeypatch.setattr(study, "utc_now", lambda: "2026-10-31T16:00:00+00:00")
    monkeypatch.setattr(study, "append_event", lambda *args: events.append(args))
    monkeypatch.setattr(study.current, "progress", lambda *a, **k: progress.append(a))
    study.failure("synthetic_stage", QuantError("expired"))
    assert events == [] and progress == [("five_net_mixture_expired",)]


def test_canonical_export_preserves_original_public_flags_and_every_scalar(monkeypatch):
    frozen = {"study_id": "synthetic_five_net", "protocol": {"path": "config/original.json"}}
    metrics = {"cagr": 0.13, "sharpe": 1.1, "max_drawdown": 0.14}
    metrics.update({f"other_scalar_{i}": i for i in range(13)})
    paths = {}
    for key in ("10y-base", "10y-stress", "5y-base", "5y-stress"):
        paths[key] = {
            "path_key": key,
            "metrics": metrics,
            "execution": {"model_order_count": 123, "total_cost_usd": 456.0},
            "gates": {"example_failed_gate": False},
            "all_point_gates_pass": False,
            "excess_net_cagr_vs_spy": -0.02,
            "excess_net_cagr_vs_qqq": -0.03,
            "stress_vs_base_cost_qqq_diagnostic": None,
            "independent_bt_max_equity_difference_usd": 1e-10,
            "independent_monthly_feedback": {"verified": True},
            "artifacts": [{"path": "private/daily-account.csv", "sha256": "a" * 64}],
        }
    actual = {
        study.IDS[0]: {
            "candidate_id": study.IDS[0],
            "status": "rejected",
            "paths": paths,
            "all_four_paths_same_candidate_and_frozen_parameters": True,
        }
    }
    saved = {
        "candidates": copy.deepcopy(actual),
        "finished_at": "2026-10-30T03:00:00Z",
        "actual_journal_event_count": 467,
        "actual_journal_head_sha256": "b" * 64,
        "publication": {"published_commit": "c" * 40},
        "global_counts_at_completion": {},
    }
    benchmarks = {
        "benchmarks": [
            {"years": year, "scenario": scenario, "symbol": symbol, "metrics": {"cagr": 0.12}}
            for year in (10, 5)
            for scenario in ("base", "stress")
            for symbol in ("SPY", "QQQ")
        ]
    }
    captured = []
    monkeypatch.setattr(study, "inputs", lambda: (None, None, frozen, benchmarks))
    monkeypatch.setattr(study, "checked_packet", lambda f: {})
    monkeypatch.setattr(study, "read_json", lambda p: saved)
    monkeypatch.setattr(study, "states", lambda f: actual)
    monkeypatch.setattr(study.base, "journal_events", lambda p: [])
    monkeypatch.setattr(
        study.common, "counts", lambda events: {"new_completed_strategy_evaluations": 1}
    )
    monkeypatch.setattr(study.common, "retain_receipt", lambda path, body, v: captured.append(body))
    monkeypatch.setattr(study, "reference", lambda p: {"path": p.name, "sha256": "d" * 64})
    study.public_result()
    body = captured[0]
    assert body["original_records_reclassified"] is False
    assert body["raw_prices_documents_targets_or_daily_accounts_exported"] is False
    assert body["order_authority"] is body["investment_objective_verified"] is False
    assert len(body["reviews"]) == 1 and len(body["reviews"][0]["paths"]) == 4
    assert all(p["metrics"] == metrics for p in body["reviews"][0]["paths"])
    assert "private/daily-account.csv" not in str(body)
