import copy
from types import FunctionType, SimpleNamespace

import pytest

from us_quant import expert_active_forecast_pair_v2 as rules
from us_quant import ssd_commitment_research_v1 as contract
from us_quant import ssd_expert_active_forecast_pair_v2 as study
from us_quant.config import QuantError
from us_quant.storage import digest_json, read_json, utc_now


def source():
    return {
        "registered_at": "2026-11-06T00:00:00+00:00",
        "registered_event_sha256": "source-registration",
        "registered_frozen_sha256": "old-frozen",
        "target_reference": {"path": "immutable-target.csv", "sha256": "a" * 64},
        "proof_reference": {"path": "immutable-proof.json", "sha256": "b" * 64},
        "proof": {
            "checked_at": "2026-11-06T00:01:00+00:00",
            "frozen_specification_sha256": "old-frozen",
            "candidate_accounts_or_metrics_computed": False,
            "source_models_selectors_forecasters_or_accounts_replayed": False,
            "new_Ridge_fits": 0,
            "new_source_requests": 0,
            "decision_count": 121,
            "all_pairs_per_mature_decision": 28,
            "actual_input_resolved_record_count": 242,
            "source_NET_RF_binary64_values_checked": 236512,
            "independent_forward_validation": False,
            "prefix_truncations": [
                "2016-10-05",
                "2017-03-31",
                "2017-09-29",
                "2021-09-30",
                "2026-09-30",
            ],
            "formal_factor_coverage": [
                {"years": 10, "actual_factor_symbols": ["MTUM", "VLUE", "QUAL"]},
                {"years": 5, "actual_factor_symbols": ["MTUM", "VLUE", "QUAL"]},
            ],
            "maximum_independent_target_difference": 2.8e-17,
            "maximum_independent_weight_difference": 2.3e-16,
            "maximum_independent_risk_difference": 7e-18,
            "maximum_independent_score_difference": 5.7e-14,
        },
    }


def real_packet_entry(proof, registered_at="2026-11-06T00:00:00+00:00"):
    candidate, frozen = rules.CANDIDATES[0], {"study_id": "synthetic_v2"}
    registered = {
        "at": registered_at,
        "sha256": "new-registration",
        "body": {"candidate": candidate, "frozen": frozen},
    }
    target_proof = {
        "at": utc_now(),
        "sha256": "new-proof-event",
        "body": {
            "registered_event_sha256": registered["sha256"],
            "targets": {"path": "target", "sha256": "t"},
            "proof": {"path": "proof", "sha256": "p"},
        },
    }
    reader = SimpleNamespace(
        matching_events=lambda kind, identifier: [registered]
        if kind == "registered"
        else [target_proof],
        checked_reference=lambda value: value["path"],
    )
    checked = FunctionType(
        contract.packet_entry.__code__,
        {**contract.packet_entry.__globals__, "base": reader, "read_json": lambda path: proof},
    )
    return checked(candidate, frozen)


def test_real_packet_entry_reproduces_missing_V1_boolean_without_mocking_contract():
    with pytest.raises(KeyError, match="original_source_signal_recomputed"):
        real_packet_entry(source()["proof"])


def test_metadata_adapter_passes_actual_contract_and_preserves_every_numeric_field():
    original = source()
    before = copy.deepcopy(original)
    frozen = {"study_id": "synthetic_v2"}
    proof = rules.attest(original, digest_json(frozen), utc_now())
    entry = real_packet_entry(proof, "2026-11-06T00:03:00+00:00")
    assert entry["specification"]["frozen"] == frozen
    assert original == before
    for key, value in original["proof"].items():
        if key not in ("checked_at", "frozen_specification_sha256"):
            assert proof[key] == value
    assert proof["source_target_derivation_checked_at"] == original["proof"]["checked_at"]
    assert proof["source_target_proof"] == original["proof_reference"]
    assert proof["reused_target_artifact"] == original["target_reference"]
    assert proof["original_source_signal_recomputed"] is False
    assert not proof["new_V2_target_derivation_performed"]


def test_actual_contract_rejects_attestation_before_new_registration():
    proof = rules.attest(source(), "new-frozen", "2026-11-06T00:02:00+00:00")
    with pytest.raises(QuantError):
        real_packet_entry(proof, "2026-11-06T00:03:00+00:00")


@pytest.mark.parametrize(
    "key,value",
    [
        ("source_models_selectors_forecasters_or_accounts_replayed", True),
        ("original_source_signal_recomputed", True),
        ("new_Ridge_fits", 1),
        ("new_source_requests", 1),
        ("candidate_accounts_or_metrics_computed", True),
        ("decision_count", 120),
        ("all_pairs_per_mature_decision", 27),
        ("source_NET_RF_binary64_values_checked", 236511),
        ("maximum_independent_target_difference", 1e-8),
        ("maximum_independent_weight_difference", 1e-8),
        ("maximum_independent_risk_difference", 1e-8),
        ("maximum_independent_score_difference", 1e-8),
    ],
)
def test_adapter_cannot_waive_source_account_or_numeric_failure(key, value):
    observed = source()
    observed["proof"][key] = value
    with pytest.raises(QuantError):
        rules.attest(observed, "new-frozen", utc_now())


def test_adapter_cannot_fill_missing_original_provenance_or_fake_derivation_time():
    observed = source()
    del observed["proof"]["source_models_selectors_forecasters_or_accounts_replayed"]
    with pytest.raises(KeyError):
        rules.attest(observed, "new-frozen", utc_now())
    observed = source()
    observed["proof"]["checked_at"] = "2026-12-01T00:00:00+00:00"
    with pytest.raises(QuantError):
        rules.attest(observed, "new-frozen", utc_now())


@pytest.mark.parametrize(
    "key,value",
    [
        ("new_target_derivations", 1),
        ("economic_law_or_parameters_changed", True),
        ("source_V1_terminal_failures_preserved", False),
        ("source_V1_target_derivation_precedes_V2_registration", False),
        ("new_source_RF_views", 1),
        ("independent_economic_research_questions_added_by_adapter", 1),
    ],
)
def test_no_numeric_or_trial_history_repair_is_configurable(key, value):
    with pytest.raises(QuantError):
        rules.validate({**read_json(rules.POLICY), key: value})


def test_prepare_uses_old_target_refs_and_never_calls_old_numeric_entrypoints(
    tmp_path, monkeypatch
):
    from us_quant.storage import file_digest

    original = source()
    frozen = {"study_id": "synthetic_v2"}
    prepared = {
        "entries": {
            old_id: {"targets": original["target_reference"], "proof": original["proof_reference"]}
            for old_id in rules.SOURCE_IDS
        }
    }
    monkeypatch.setattr(study, "FOLDER", tmp_path)
    monkeypatch.setattr(study, "PREPARED", tmp_path / "prepared.json")
    monkeypatch.setattr(study.lifecycle, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "events", lambda kind, name: [{}])
    monkeypatch.setattr(study.base, "checked_reference", lambda ref: ref["path"])

    def read(path):
        if path == study.rules.POLICY:
            return {"source_prepared_targets": {"path": "bundle", "sha256": "x"}}
        return prepared if path == "bundle" else original["proof"]

    def forbidden(*args, **kwargs):
        raise AssertionError("A terminal source or allocation must not be rerun.")

    monkeypatch.setattr(study, "read_json", read)
    monkeypatch.setattr(study, "reference", lambda p: {"path": str(p), "sha256": file_digest(p)})
    monkeypatch.setattr(
        study.prior,
        "events",
        lambda kind, name: [
            {
                "at": original["registered_at"],
                "sha256": original["registered_event_sha256"],
                "body": {"frozen": {"source": "synthetic"}},
            }
        ],
    )
    original["proof"]["frozen_specification_sha256"] = digest_json({"source": "synthetic"})
    for name in ("prepare", "register", "evaluate"):
        monkeypatch.setattr(study.prior, name, forbidden)
    monkeypatch.setattr(study.prior.rules, "allocate", forbidden)
    entries = study.prepare(None, None, frozen)
    assert set(entries) == set(rules.IDS)
    for entry in entries.values():
        assert entry["targets"] == original["target_reference"]
        proof = read_json(tmp_path / entry["proof"]["path"])
        assert proof["original_source_signal_recomputed"] is False
        assert not proof["new_V2_target_derivation_performed"]


def test_both_V2_registrations_precede_any_attestation(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "FOLDER", tmp_path)
    monkeypatch.setattr(study, "PREPARED", tmp_path / "prepared.json")
    monkeypatch.setattr(study, "events", lambda kind, n: [{}] if n == rules.IDS[0] else [])
    with pytest.raises(QuantError, match="Both new registrations"):
        study.prepare(None, None, {"study_id": "synthetic"})
    assert not list(tmp_path.iterdir())


def test_source_and_real_contract_dependency_closure_is_bound():
    paths = {r["path"] for r in study.dependencies()}
    assert set(study.OWN).issubset(paths)
    assert "src/us_quant/ssd_commitment_research_v1.py" in paths
    assert "src/us_quant/expert_active_forecast_pair_v1.py" in paths
    assert not hasattr(rules, "allocate")
