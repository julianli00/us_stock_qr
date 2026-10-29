from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from us_quant import ssd_expert_meta_research_v1 as study
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.storage import read_json, utc_now, write_json
from us_quant.strategy_experts_v1 import ArchivedDenseExpert, ExpertIdentity


@pytest.fixture
def journal(tmp_path, monkeypatch):
    base, common = study.base, study.common
    monkeypatch.setattr(base, "ROOT", tmp_path)
    monkeypatch.setattr(base, "JOURNAL", tmp_path / "events")
    monkeypatch.setattr(base, "storage_guard", lambda: None)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "FOLDER", tmp_path / "report")
    monkeypatch.setattr(study, "TARGETS", tmp_path / "targets")
    monkeypatch.setattr(study, "PACKET", tmp_path / "packet.json")
    monkeypatch.setattr(study, "PREPARED", tmp_path / "prepared.json")
    monkeypatch.setattr(study, "RESULTS", tmp_path / "results.json")
    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study.current, "progress", lambda *args, **kwargs: None)
    base.append_event(base.JOURNAL, "baseline", None, {"legacy": {"synthetic": True}})
    return base.JOURNAL


def setup(journal, monkeypatch):
    frozen = {"study_id": "synthetic", "legacy": {"synthetic": True}}
    monkeypatch.setattr(study, "inputs", lambda: (None, None, frozen, None, None))
    monkeypatch.setattr(study, "check_frozen", lambda f: None)
    monkeypatch.setattr(
        study.common,
        "counts",
        lambda events: {
            "new_registered_candidates": 30,
            "new_completed_strategy_evaluations": 29,
            "new_technical_failures": 1,
            "new_pending_candidates": 0,
        },
    )
    return frozen


def test_both_registrations_precede_real_meta_scores_and_completed_preparation_never_replays(
    journal,
    monkeypatch,
):
    frozen = setup(journal, monkeypatch)
    calls = []

    def prepare(*args):
        assert all(len(study.base.matching_events("registered", name)) == 1 for name in study.IDS)
        calls.append(True)
        result = {}
        audit_path = journal.parent / "audit.json"
        write_json(audit_path, {"decisions": []})
        for name in study.IDS:
            target = journal.parent / f"{name}.json"
            proof = journal.parent / f"{name}-proof.json"
            write_json(target, {"synthetic": True})
            write_json(
                proof,
                {
                    "checked_at": utc_now(),
                    "candidate_accounts_or_metrics_computed": False,
                    "original_source_signal_recomputed": False,
                    "information_audits": study.base.reference(audit_path),
                },
            )
            result[name] = {
                "targets": study.base.reference(target),
                "proof": study.base.reference(proof),
            }
        return result

    monkeypatch.setattr(study, "prepare", prepare)
    study.register()
    study.register()
    assert calls == [True]
    packet = read_json(study.PACKET)
    packet["candidate_accounts_or_metrics_computed"] = True
    write_json(study.PACKET, packet)
    with pytest.raises(QuantError, match="pre-account"):
        study.checked_packet(frozen)


def test_failure_retains_both_trials_and_never_uses_an_alternate_rule(journal, monkeypatch):
    setup(journal, monkeypatch)
    partial = journal.parent / "partial.json"

    def stop(*args):
        write_json(partial, {"partial_meta_scores": "synthetic"})
        raise QuantError("synthetic preparation failure")

    monkeypatch.setattr(study, "prepare", stop)
    with pytest.raises(QuantError, match="preparation failure"):
        study.register()
    assert read_json(partial) == {"partial_meta_scores": "synthetic"}
    assert all(
        len(study.base.matching_events("technical_failure", name)) == 1 for name in study.IDS
    )
    with pytest.raises(QuantError, match="terminal"):
        study.register()


def test_expiry_preserves_pending_trial_without_manufacturing_technical_failure(
    journal, monkeypatch
):
    for name in study.IDS:
        study.base.append_event(journal, "registered", name, {})
    monkeypatch.setattr(study, "utc_now", lambda: "2026-10-31T15:59:00Z")
    study.failure("expired", study.IDS, QuantError("cutoff"))
    assert all(not study.base.matching_events("technical_failure", name) for name in study.IDS)


def test_completed_account_evaluation_never_replays_or_reselects(journal, monkeypatch):
    frozen = setup(journal, monkeypatch)
    actual = {name: {"status": "rejected", "paths": {}} for name in study.IDS}
    monkeypatch.setattr(study, "checked_packet", lambda f: None)
    monkeypatch.setattr(study, "states", lambda f: actual)
    monkeypatch.setattr(study, "publication", lambda *args: pytest.fail("No publication replay"))
    monkeypatch.setattr(study.rules, "build_targets", lambda *args: pytest.fail("No target replay"))
    monkeypatch.setattr(
        study.execution, "evaluate_path", lambda *args: pytest.fail("No account replay")
    )
    write_json(
        study.RESULTS,
        {"frozen_specification_sha256": study.digest_json(frozen), "candidates": actual},
    )
    assert study.evaluate("unused") == study.base.reference(study.RESULTS)


def test_unresolved_preparation_attempt_is_not_replayed(journal, monkeypatch):
    frozen = setup(journal, monkeypatch)
    write_json(study.FOLDER / "preparation-attempt.json", {"synthetic": True})
    with pytest.raises(QuantError, match="unresolved"):
        study.prepare(None, None, frozen)


def test_actual_publication_requires_all_expert_code_and_frozen_policies(journal, monkeypatch):
    frozen = {
        "study_id": "synthetic",
        "code": [{"path": "src/us_quant/strategy_experts_v1.py", "sha256": "a" * 64}],
        "policies": [{"path": "config/strategy-expert-meta-v1.json", "sha256": "b" * 64}],
    }
    calls = []
    monkeypatch.setattr(study.base, "published_inputs", lambda *args: {})
    monkeypatch.setattr(study.base, "reference", lambda p: {"path": str(p), "sha256": "c" * 64})

    def verify(root, packet, required, commit):
        calls.append(required)
        return {"published_commit": commit, "registration_packet": packet}

    monkeypatch.setattr(study, "verify_publication", verify)
    study.publication(frozen, {}, "d" * 40)
    assert calls == [{"src/us_quant/strategy_experts_v1.py", "config/strategy-expert-meta-v1.json"}]
    with pytest.raises(QuantError, match="original"):
        study.publication(frozen, {}, "e" * 40)


def test_independent_scalar_meta_proof_matches_real_interface_and_rejects_tampering():
    dates = sessions("2015-08-10", "2026-10-05")
    columns = ("a", "b", "c")
    prices = pd.DataFrame(100.0, index=dates, columns=columns)
    market = MarketData(
        prices.copy(),
        prices.copy(),
        prices.copy(),
        prices * 1000,
        pd.Series(0.0, index=dates),
    )
    shadow_dates = dates[dates >= pd.Timestamp("2016-10-06")]
    experts = []
    for i, identifier in enumerate(study.rules.EXPERT_IDS):
        target = pd.DataFrame(np.nan, index=dates, columns=columns)
        target.loc["2016-09-07":] = [0.98 if k == i else 0.0 for k in range(3)]
        returns = 0.0004 + (0.004 + i * 0.002) * np.sin(np.arange(len(shadow_dates)) / (2 + i))
        nav = 10000 * np.cumprod(1 + returns)
        shadow = pd.DataFrame(
            {
                "equity": nav,
                "return": returns,
                "risk_free": 0.0,
                "cash": nav * 0.02,
                "gross_exposure": 0.98,
                "cost": 0.0,
                "orders": 0,
            },
            index=shadow_dates,
        )
        experts.append(
            ArchivedDenseExpert(
                ExpertIdentity(identifier, f"mechanism_{i}", "2026-10-29T13:25:29Z"),
                target,
                shadow,
                columns,
            )
        )
    targets, audits = study.rules.build_targets(
        tuple(experts), dates, columns, read_json(study.rules.POLICY)
    )
    checked = study.verify_targets(market, tuple(experts), targets, audits)
    assert checked["maximum_independent_target_difference"] < 1e-10
    assert checked["maximum_independent_net_score_difference"] < 1e-8
    changed = copy.deepcopy(audits)
    changed[-1]["expert_shadow_last_observation"][0] = "2026-10-05"
    with pytest.raises(QuantError, match="state"):
        study.verify_targets(market, tuple(experts), targets, changed)
