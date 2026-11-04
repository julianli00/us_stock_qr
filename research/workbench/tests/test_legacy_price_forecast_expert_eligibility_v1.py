import copy

import numpy as np
import pytest
from test_legacy_real_yield_expert_eligibility_v1 import source

from us_quant import legacy_price_forecast_expert_eligibility_v1 as rules
from us_quant.config import QuantError
from us_quant.storage import digest_json, read_json


def identity_source():
    policy = read_json(rules.POLICY)
    spec = {
        "id": rules.CANDIDATE,
        "configuration": {"id": rules.CANDIDATE, "augmented_information": False},
        "asset_leverage": {s: 1.0 for s in rules.ASSETS},
        "market": {"synthetic": "unchanged"},
        "order_authority": False,
        "leveraged_products_allowed": False,
    }
    bundle = {
        "candidate_spec_sha256": digest_json(spec),
        "market": spec["market"],
        "order_authority": False,
        "leveraged_products_allowed": False,
        "completed_at": "2026-10-10T18:30:46Z",
        "paths": [{"years": y, "scenario": s} for y, s in rules.PATHS],
    }
    snapshot = {
        "candidate_id": rules.CANDIDATE,
        "technical_failures": 0,
        "events": [{"sequence": 42}, {"sequence": 44}],
        "specification": spec,
        "spec_sha256": digest_json(spec),
        "evidence_sha256": digest_json(bundle),
        "registered_at": "2026-10-10T18:29:35Z",
        "reviewed_at": "2026-10-10T18:30:47Z",
        "review": {"status": "rejected_historical"},
    }
    registration = {
        "candidates": [
            {
                "spec": spec,
                "registration": {
                    "registered_at": snapshot["registered_at"],
                    "spec_sha256": snapshot["spec_sha256"],
                },
            }
        ],
        "original_invalid_code_versions_retained": True,
        "outcomes_computed": False,
    }
    return (
        snapshot,
        registration,
        {"candidates": {rules.CANDIDATE: snapshot["review"]}},
        bundle,
        policy,
    )


def test_original_price_only_identity_and_financial_rejection_not_a_model_refit():
    result = rules.check_identity(*identity_source())
    assert result["source_events"] == [42, 44]
    assert result["source_standalone_financial_rejection_preserved"]
    assert not result["augmented_sibling_reviewed_or_admitted"]


@pytest.mark.parametrize("kind", ["augmented", "ridge", "history", "registration"])
def test_source_scope_hyperparameter_or_original_identity_change_fails(kind):
    values = copy.deepcopy(identity_source())
    if kind == "augmented":
        values[0]["specification"]["configuration"]["augmented_information"] = True
    elif kind == "ridge":
        values[-1]["model_parameters"]["ridge"]["alpha"] = 1.0
    elif kind == "history":
        values[1]["original_invalid_code_versions_retained"] = False
    else:
        values[0]["events"][0]["sequence"] = 43
    with pytest.raises(QuantError):
        rules.check_identity(*values)


def training_source():
    row = {
        "first_training_decision": "2015-11-30",
        "last_training_decision": "2016-07-29",
        "latest_label_available": "2016-09-01",
        "decision_date": "2016-09-30",
        "unique_training_months": 9,
        "training_rows": 45,
        "hyperparameters_changed": False,
        "clipped_current_feature_values": 1,
        "training_inputs_sha256": "a" * 64,
        "training_labels_sha256": "b" * 64,
        "current_features_sha256": "c" * 64,
    }
    return {
        "source_sha256": "d" * 64,
        "future_labels_used": False,
        "source_bound_targets_verified": True,
        "independent_forward_validation": False,
        "models": {rules.CANDIDATE: [row]},
    }


def test_saved_matured_label_hash_receipts_are_not_fitted_models_or_forward_proof():
    result = rules.check_training(training_source(), "d" * 64)
    assert result["stored_training_receipts"] == 1
    assert result["minimum_stored_mature_months"] == 9
    assert not result["hash_receipts_are_saved_fitted_coefficients"]
    assert not result["new_fit_feature_label_or_forecast_executed"]


@pytest.mark.parametrize("kind", ["future", "unmatured", "rows", "changed", "missing_hash"])
def test_training_clock_maturity_or_hyperparameter_failures_are_explicit(kind):
    audit = training_source()
    row = audit["models"][rules.CANDIDATE][0]
    if kind == "future":
        row["latest_label_available"] = "2016-10-03"
    elif kind == "unmatured":
        row["unique_training_months"] = 8
    elif kind == "rows":
        row["training_rows"] = 9
    elif kind == "changed":
        row["hyperparameters_changed"] = True
    else:
        row["training_inputs_sha256"] = ""
    with pytest.raises(QuantError):
        rules.check_training(audit, "d" * 64)


def test_native_checker_is_original_money_RF_and_calendar_law_with_only_identity_bound():
    result = rules.check_native(*source())
    assert result["candidate_id"] == rules.CANDIDATE
    assert result["mechanism_group"] == rules.GROUP
    assert result["status"] == "eligible_not_admitted"
    assert rules.native.CANDIDATE == "macro_real_yield_factor_tilt"


def test_one_ulp_RF_gap_and_full_native_seed_block_without_a_new_view_or_Adapter():
    result = rules.check_native(*source(funding_seed=True, rf_gap=True))
    assert result["status"] == "not_admitted_current_Interface"
    assert all(r["RF_mismatch_count"] == 1 for r in result["paths"])
    assert all(len(r["full_native_nonmonthend_dates"]) == 1 for r in result["paths"])
    assert not result["new_RF_recommendation_assessment_view_or_Adapter"]
    assert not result["active_pool_changed"]


def test_invalid_cash_not_weak_financial_return_excludes_a_native_source():
    values = source()
    values[2][10, "base"]["strategy"].iloc[0, 2] = -1
    result = rules.check_native(*values)
    assert any(r["reason"] == "native_cash_funding_or_accounting" for r in result["blockers"])
    assert result["old_standalone_financial_rejection_preserved"]


def test_missing_full_return_day_cannot_be_filled_or_cropped():
    values = source()
    values[2][10, "base"]["strategy"] = values[2][10, "base"]["strategy"].iloc[1:]
    with pytest.raises(QuantError):
        rules.check_native(*values)


def test_repeated_actual_review_is_terminal(monkeypatch, tmp_path):
    monkeypatch.setattr(rules.lifecycle, "runtime_guard", lambda: None)
    monkeypatch.setattr(rules, "FOLDER", tmp_path)
    monkeypatch.setattr(rules, "RESULT", tmp_path / "result.json")
    (tmp_path / "review-attempt-v1.json").write_text("{}")
    with pytest.raises(QuantError, match="never repair and retry"):
        rules.review()


def test_native_variance_and_RF_are_not_normalized_by_price_model_reviewer():
    snapshot, bundle, frames, common = source()
    before = frames[10, "base"]["strategy"]["risk_free"].copy()
    result = rules.check_native(snapshot, bundle, frames, common)
    assert np.array_equal(before, frames[10, "base"]["strategy"]["risk_free"])
    assert not result["new_expert_admitted"]


def test_native_review_does_not_call_old_financial_verifiers(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A cached technical review must not rerun an old financial pipeline.")

    monkeypatch.setattr(rules.BASE, "verify_path", forbidden)
    monkeypatch.setattr(rules.BASE, "review_market", forbidden)
    result = rules.check_native(*source(funding_seed=True, rf_gap=True))
    assert result["four_cached_native_paths_checked"] == 4
    assert not result["source_pipeline_or_independent_accounts_replayed"]
