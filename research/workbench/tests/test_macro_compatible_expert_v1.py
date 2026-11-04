from __future__ import annotations

import ast
import copy
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_macro_policy_compatibility_v1 import source_data

from us_quant import macro_compatible_expert_v1 as rules
from us_quant import macro_policy_compatibility_v1 as compatible
from us_quant.config import QuantError


def scope_evidence():
    policy = {
        "publication_delay_sessions": 2,
        "maximum_observation_age_days": 7,
        "real_yield_change_sessions": 63,
        "candidates": [{"id": rules.original.CANDIDATE, "curve_defense": False}],
    }
    rows = []
    for years, count, start in ((10, 2512, "2016-10-06"), (5, 1254, "2021-10-06")):
        rows.append(
            {
                "candidate_id": rules.original.CANDIDATE,
                "years": years,
                "formal_return_start": start,
                "formal_return_end": "2026-10-05",
                "formal_return_sessions": count,
                "seeded_original_and_source_replaced_targets_byte_identical": True,
                "queried_DFII10_cells_only_replaced_other_macro_data_unchanged": True,
                "this_is_not_a_fully_PIT_daily_feature_dataset": True,
                "changed_source_target_sessions_before_formal_window": ["2015-11-30"],
            }
        )
    return policy, {
        "existing_validation_not_replayed": True,
        "formal_target_parity_reference": rows,
    }


def test_original_limited_monthly_scope_keeps_preformal_difference_not_full_daily_PIT():
    policy, evidence = scope_evidence()
    scope = rules.check_scope(policy, evidence)
    assert scope == rules.SCOPE
    assert not scope["full_daily_PIT_or_revision_free_source_admitted"]
    assert scope["earlier_native_context_retained_not_new_daily_macro_admission"]
    assert evidence["formal_target_parity_reference"][0][
        "changed_source_target_sessions_before_formal_window"
    ] == ["2015-11-30"]


@pytest.mark.parametrize(
    "damage",
    [
        "lag",
        "age",
        "curve",
        "parity_false",
        "full_daily_claim",
        "formal_change",
        "replay",
    ],
)
def test_scope_source_false_flags_timing_or_formal_changes_cannot_be_waived(damage):
    policy, evidence = scope_evidence()
    if damage == "lag":
        policy["publication_delay_sessions"] = 0
    elif damage == "age":
        policy["maximum_observation_age_days"] = 8
    elif damage == "curve":
        policy["candidates"][0]["curve_defense"] = True
    elif damage == "parity_false":
        evidence["formal_target_parity_reference"][0][
            "seeded_original_and_source_replaced_targets_byte_identical"
        ] = False
    elif damage == "full_daily_claim":
        evidence["formal_target_parity_reference"][0][
            "this_is_not_a_fully_PIT_daily_feature_dataset"
        ] = False
    elif damage == "formal_change":
        evidence["formal_target_parity_reference"][0][
            "changed_source_target_sessions_before_formal_window"
        ] = ["2018-01-31"]
    else:
        evidence["existing_validation_not_replayed"] = False
    with pytest.raises(QuantError):
        rules.check_scope(policy, evidence)


def test_existing_assessment_roundtrip_preserves_every_nonRF_value_not_native_RF_identity():
    _, _, book, native_rf, common, _ = source_data()
    book = pd.read_csv(
        io.StringIO(book.to_csv(float_format="%.17g")),
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )
    view = compatible.assessment_view(book, native_rf, common)
    stored = pd.read_csv(
        io.StringIO(view.to_csv(float_format="%.17g")),
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )
    proof = rules.verify_existing_assessment(book, stored, native_rf, common)
    assert proof["native_RF_still_not_exact_common"]
    assert proof["common_assessment_RF_exact_frozen_default_reader"]
    assert not proof["future_score_tie_or_financial_effect_absent_claimed"]
    assert stored.drop(columns="risk_free").equals(book.drop(columns="risk_free"))


@pytest.mark.parametrize("column", ["return", "equity", "cash", "cost"])
def test_even_one_native_money_value_cannot_change_in_an_existing_assessment(column):
    _, _, book, native_rf, common, _ = source_data()
    view = compatible.assessment_view(book, native_rf, common)
    view.iloc[0, view.columns.get_loc(column)] = np.nextafter(view.iloc[0][column], np.inf)
    with pytest.raises(QuantError, match="exact native NET"):
        rules.verify_existing_assessment(book, view, native_rf, common)


def test_one_ULP_wrong_common_RF_is_not_rounded_or_tolerated():
    _, _, book, native_rf, common, _ = source_data()
    view = compatible.assessment_view(book, native_rf, common)
    view.iloc[0, view.columns.get_loc("risk_free")] = np.nextafter(
        view.iloc[0]["risk_free"], np.inf
    )
    with pytest.raises(QuantError, match="exact native NET"):
        rules.verify_existing_assessment(book, view, native_rf, common)


def test_old_native_RF_cannot_be_relabelled_exact_common():
    _, _, book, native_rf, _, _ = source_data()
    with pytest.raises(QuantError, match="old unequal native RF"):
        rules.verify_existing_assessment(book, book, native_rf, native_rf)


def admitted_record():
    return {
        "admission_id": rules.ADMISSION_ID,
        "status": rules.ADMISSION_STATUS,
        "scope": copy.deepcopy(rules.SCOPE),
        "selected_representative_or_pool_member": False,
        "contract": {"path": "synthetic-only", "sha256": rules.CONTRACT_SHA},
        "cached_source_checks": {"source": "synthetic-only"},
    }


@pytest.mark.parametrize("damage", ["status", "scope", "selected", "identity"])
def test_future_consumer_rejects_unadmitted_promoted_wrong_source_or_upgraded_scope(
    monkeypatch, damage
):
    record = admitted_record()
    if damage == "status":
        record["status"] = "blocked"
    elif damage == "scope":
        record["scope"]["full_daily_PIT_or_revision_free_source_admitted"] = True
    elif damage == "selected":
        record["selected_representative_or_pool_member"] = True
    else:
        record["admission_id"] = "Term_or_ANFCI"
    monkeypatch.setattr(rules.base, "checked_reference", lambda ref: "synthetic")
    monkeypatch.setattr(rules, "read_json", lambda path: record)
    with pytest.raises(QuantError, match="actual scoped admission record"):
        rules.load_admitted({"path": "synthetic", "sha256": "synthetic"}, rules.UNIVERSE)


def test_future_consumer_binds_exact_cached_source_facts(monkeypatch):
    record = admitted_record()
    expert = object()
    monkeypatch.setattr(rules.base, "checked_reference", lambda ref: "synthetic")
    monkeypatch.setattr(rules, "read_json", lambda path: record)
    monkeypatch.setattr(
        rules, "load_candidate", lambda *args: (expert, record["cached_source_checks"])
    )
    assert rules.load_admitted({"path": "synthetic"}, rules.UNIVERSE) is expert
    monkeypatch.setattr(rules, "load_candidate", lambda *args: (expert, {"source": "changed"}))
    with pytest.raises(QuantError, match="admitted source identity"):
        rules.load_admitted({"path": "synthetic"}, rules.UNIVERSE)


def test_wrong_contract_or_asset_lattice_rejected_before_source_reads():
    with pytest.raises(QuantError, match="exact macro consumer contract"):
        rules.load_candidate({"sha256": "wrong"}, rules.UNIVERSE)
    with pytest.raises(QuantError, match="exact macro consumer contract"):
        rules.load_candidate({"sha256": rules.CONTRACT_SHA}, rules.UNIVERSE[:8])


def test_loader_has_no_old_source_generator_view_or_financial_entrypoint_calls():
    path = Path(rules.__file__)
    tree = ast.parse(path.read_text())
    prohibited = {
        "assessment_view",
        "build_targets",
        "target",
        "prepare",
        "register",
        "evaluate",
        "fullverify",
        "probe",
        "freeze",
        "run_cash_funded",
        "backtest",
    }
    assert not [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in prohibited
    ]
    assert "mode=ro&immutable=1" in path.read_text()
    assert "PRAGMA query_only=ON" in path.read_text()
