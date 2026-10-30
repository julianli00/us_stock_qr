import json
from contextlib import nullcontext
from hashlib import sha256
from types import SimpleNamespace

import pytest
from test_expert_vvix_admission_v1 import source_scope, synthetic

from us_quant import ssd_anfci_expert_eligibility_v1 as old_review
from us_quant import ssd_vvix_expert_admission_v1 as admission
from us_quant import ssd_vvix_uncertainty_research_v1 as original
from us_quant.config import QuantError
from us_quant.storage import read_json


def test_loader_requires_explicit_after_index_close_clock():
    _, _, _, common = synthetic()
    with pytest.raises(TypeError):
        admission.load_admitted(admission.rules.ASSETS, common)
    with pytest.raises(QuantError, match="after16:15ET"):
        admission.load_admitted(
            admission.rules.ASSETS, common, decision_information_clock="16_00_stock_close"
        )


def test_closed_source_generators_accounts_and_E_review_are_not_called(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A cached expert observation replayed a closed research pipeline.")

    for module, names in (
        (original, ("inputs", "register", "evaluate")),
        (admission.prior, ("initialize", "prepare", "register", "evaluate", "inputs")),
        (old_review, ("review",)),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    assert admission.rules.validate_scope(*source_scope())["source_admitted"] is True
    assert (
        admission.rules.check_cached_observations(*synthetic())["available_main_observations"]
        == 121
    )


def test_dependency_capture_does_not_mutate_template():
    before = admission.import_template.OWN
    refs = admission.dependencies()
    assert admission.import_template.OWN is before
    paths = {ref["path"] for ref in refs}
    assert set(admission.OWN).issubset(paths)
    assert "src/us_quant/strategy_experts_v1.py" in paths
    assert "src/us_quant/expert_vvix_admission_v1.py" in paths


def test_active_common_RF_reuses_the_existing_frozen_reader(monkeypatch):
    design = read_json(admission.DESIGN)
    rf = design["active_common_risk_free"]
    frozen = {"market": {"risk_free": rf}}
    monkeypatch.setattr(admission.prior, "PARENT_SPEC", admission.digest_json(frozen))
    monkeypatch.setattr(
        admission,
        "read_json",
        lambda path: {"entries": [{"specification": {"frozen": frozen}}]},
    )
    expected = object()

    def unchanged_reader(bundle, root):
        assert bundle == {"market": frozen["market"]}
        assert root == admission.ROOT
        return SimpleNamespace(risk_free=expected)

    monkeypatch.setattr(admission, "review_market", unchanged_reader)
    assert admission.common_risk_free(design) is expected
    design["active_common_risk_free"] = {"path": "different.csv", "sha256": "a" * 64}
    with pytest.raises(QuantError, match="interpretation"):
        admission.common_risk_free(design)


def test_recorded_scope_survives_unrelated_future_configuration_additions(tmp_path, monkeypatch):
    design = read_json(admission.DESIGN)
    intent = {
        "admission_id": admission.ADMISSION,
        "actual_cached_admission_started": False,
        "maximum_actual_admission_passes": 1,
        "design": admission.prior.reference(admission.DESIGN),
        "implementation_and_test_inputs": [
            admission.prior.reference(admission.ROOT / path) for path in admission.OWN
        ],
        "source_references": design["source_references"],
        "validation_evidence": [],
    }
    path = tmp_path / "intent.json"
    path.write_text(json.dumps(intent))
    monkeypatch.setattr(admission, "INTENT", path)
    monkeypatch.setattr(admission, "OUTPUT", tmp_path / "no-existing-admission.json")

    def forbidden():
        pytest.fail("A future unrelated policy changed the recorded admission scope.")

    monkeypatch.setattr(admission, "dependencies", forbidden)
    monkeypatch.setattr(admission, "source_references", forbidden)
    assert admission.checked_intent()["source_references"] == design["source_references"]


def test_public_admission_excludes_raw_targets_and_return_paths(monkeypatch):
    monkeypatch.setattr(
        admission.prior,
        "reference",
        lambda path: {"path": "private/admission.json", "sha256": "a" * 64},
    )
    value = {
        "admitted_at": "2026-10-30T00:00:00+00:00",
        "source_original_registered_at": "2026-10-28T23:58:17.764688+00:00",
        "expert": {
            "record": {"id": admission.rules.SOURCE},
            "technical": {"source_frozen_specification_sha256": "b" * 64},
        },
        "source_scope": admission.rules.validate_scope(*source_scope()),
        "observation_checks": {"available_main_observations": 121},
        "source_inputs_checked": 1,
        "analysis_import_and_configuration_dependencies": 1,
        "passed_offline_checks": 1,
        "active_common_RF_reader": "us_quant.research_program.review_market",
        "all_current_financial_counts_unchanged": {},
        "raw_targets": "PRIVATE",
        "raw_returns": "PRIVATE",
    }
    public = admission.public_report(value)
    assert "PRIVATE" not in json.dumps(public)
    assert public["source_original_standalone_financial_verdict"] == "rejected"
    assert public["financial_failure_is_not_expert_admission_failure"] is True
    assert public["existing_six_expert_pool_changed"] is False
    assert public["expanded_pool_or_next_meta_rule_frozen"] is False
    assert public["new_strategy_evaluations"] == public["new_shadow_account_runs"] == 0


def test_second_admission_attempt_is_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "FOLDER", tmp_path)
    monkeypatch.setattr(admission, "runtime_guard", lambda: None)
    (tmp_path / "admission-attempt-v1.json").write_text("{}")
    with pytest.raises(QuantError, match="first actual admission"):
        admission.admit()


@pytest.fixture
def synthetic_admission(tmp_path, monkeypatch):
    expert, targets, shadow, common = synthetic()
    design = read_json(admission.DESIGN)
    scope = admission.rules.validate_scope(*source_scope())
    technical = {"source_frozen_specification_sha256": "b" * 64}
    for name, leaf in (
        ("FOLDER", None),
        ("INTENT", "intent.json"),
        ("DESIGN", "design.json"),
        ("OUTPUT", "admission.json"),
        ("PUBLIC", "public.json"),
    ):
        monkeypatch.setattr(admission, name, tmp_path if leaf is None else tmp_path / leaf)
    admission.DESIGN.write_text(json.dumps(design))
    admission.INTENT.write_text("{}")
    intent = {
        "created_at": "2026-10-30T00:00:00+00:00",
        "source_references": [{"path": "synthetic-only.json", "sha256": "c" * 64}],
        "passed_offline_checks": 1,
    }
    counts = {"synthetic_only": True, "registered": 0}
    monkeypatch.setattr(admission, "runtime_guard", lambda: None)
    monkeypatch.setattr(admission, "research_job", lambda path: nullcontext())
    monkeypatch.setattr(admission, "checked_intent", lambda: intent)
    monkeypatch.setattr(admission, "dependencies", lambda: [])
    monkeypatch.setattr(
        admission, "cached_expert", lambda value: (expert, technical, scope, targets, shadow)
    )
    monkeypatch.setattr(
        admission.prior,
        "reference",
        lambda path: {"path": path.name, "sha256": sha256(path.read_bytes()).hexdigest()},
    )
    monkeypatch.setattr(admission.base, "journal_events", lambda path: [])
    monkeypatch.setattr(admission.prior.current, "current_counts", lambda events: counts)
    monkeypatch.setattr(admission.prior.current, "progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(admission, "common_risk_free", lambda design: common)

    def forbidden(*args, **kwargs):
        pytest.fail("A synthetic cached admission crossed a closed financial/source entrypoint.")

    for module, names in (
        (original, ("inputs", "register", "evaluate", "native_quotes", "prove_targets")),
        (admission.prior, ("initialize", "prepare", "register", "evaluate", "inputs")),
        (old_review, ("review",)),
        (admission.base, ("evaluate_path",)),
        (admission.prior.source, ("source_inputs",)),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    return common


def test_admission_interface_uses_only_cached_observations_and_preserves_clocks(
    synthetic_admission,
):
    admission.admit()
    saved = read_json(admission.OUTPUT)
    assert saved["source_original_registered_at"] == "2026-10-28T23:58:17.764688+00:00"
    assert saved["admitted_at"] > "2026-10-30T00:00:00+00:00"
    assert saved["observation_checks"]["available_main_observations"] == 121
    assert saved["current_six_expert_pool_changed"] is False
    assert saved["new_strategy_evaluations"] == 0
    assert (admission.FOLDER / "admission-attempt-v1.json").is_file()
    assert not (admission.FOLDER / "failure-v1.json").exists()


def test_failed_cached_admission_retains_attempt_without_source_rescue(synthetic_admission):
    synthetic_admission.iloc[-1] += 1e-10
    with pytest.raises(QuantError, match="common RF"):
        admission.admit()
    assert (admission.FOLDER / "failure-v1.json").is_file()
    assert not admission.OUTPUT.exists()
    with pytest.raises(QuantError, match="first actual admission"):
        admission.admit()


@pytest.mark.parametrize("mutation", ["none", "backdate", "RF", "scope"])
def test_admitted_loader_preserves_real_clock_scope_and_exact_common_RF(
    synthetic_admission, mutation
):
    admission.admit()
    saved = read_json(admission.OUTPUT)
    if mutation == "backdate":
        saved["admitted_at"] = saved["source_original_registered_at"]
    elif mutation == "scope":
        saved["source_scope"][
            "historical_intraday_first_publication_or_method_chain_authenticated"
        ] = True
    elif mutation == "RF":
        synthetic_admission.iloc[-1] += 1e-10
    admission.OUTPUT.write_text(json.dumps(saved))
    if mutation == "none":
        expert, checked = admission.load_admitted(
            admission.rules.ASSETS,
            synthetic_admission,
            decision_information_clock=admission.rules.CLOCK,
        )
        assert expert.identity.identifier == admission.rules.SOURCE
        assert checked["admitted_at"] == saved["admitted_at"]
    else:
        with pytest.raises(QuantError):
            admission.load_admitted(
                admission.rules.ASSETS,
                synthetic_admission,
                decision_information_clock=admission.rules.CLOCK,
            )
