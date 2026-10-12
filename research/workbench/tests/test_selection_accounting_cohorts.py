from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import selection_accounting_cohorts as audit
from us_quant.calendar import sessions
from us_quant.cash_funded_accounting_v2 import engine_reference
from us_quant.config import QuantError
from us_quant.storage import digest_json, read_json

ROOT = Path(__file__).parents[1]


@pytest.fixture
def policy():
    return read_json(ROOT / "config/selection-accounting-cohorts.json")


def state():
    reviews = [{"candidate_id":f"candidate_{position:02}", "evidence_sha256":"0"*64}
               for position in range(76)]
    for review in reviews[19:]:
        review["accounting_engine"] = engine_reference()
    value = {"candidate_reviews":reviews, "total_evaluated_configurations":160,
             "registered_candidates":[{} for _ in range(78)], "pending_candidate_ids":[],
             "event_chain_sha256":"1"*64}
    proof = {"reviewed_candidates":76, "regenerated_strategy_paths":304, "ledger_unchanged":True,
             "event_chain_sha256":"1"*64, "reviews":reviews}
    return value, proof, [row["candidate_id"] for row in reviews[:36]]


def test_all_76_original_outcomes_have_exact_accounting_membership_not_performance_selection(policy):
    value, proof, previous = state()
    audit.validate_policy(policy)
    groups = audit.cohort_members(value, proof, previous, policy)
    assert len(groups["legacy_original"]) == 19
    assert len(groups["cash_funded_v2"]) == 57
    assert set(groups["legacy_original"]).isdisjoint(groups["cash_funded_v2"])
    for row in value["candidate_reviews"]:
        row["status"] = "rejected_historical"
    assert audit.cohort_members(value, proof, previous, policy) == groups
    for row in value["candidate_reviews"]:
        row["status"] = "qualified_research"
    assert audit.cohort_members(value, proof, previous, policy) == groups


@pytest.mark.parametrize("problem", ["dropped", "duplicate", "legacy_relabelled", "unknown_engine", "old_prefix",
                                   "pending", "incomplete_audit", "changed_head", "invalid_version_count"])
def test_incomplete_changed_or_performance_filtered_scope_cannot_be_silently_inferred(policy, problem):
    value, proof, previous = state()
    if problem == "dropped":
        value["candidate_reviews"] = value["candidate_reviews"][:-1]
    elif problem == "duplicate":
        value["candidate_reviews"][-1] = value["candidate_reviews"][0]
    elif problem == "legacy_relabelled":
        value["candidate_reviews"][0]["accounting_engine"] = engine_reference()
    elif problem == "unknown_engine":
        value["candidate_reviews"][19]["accounting_engine"] = {"id":"different", "source_sha256":"0"*64}
    elif problem == "old_prefix":
        previous[0], previous[1] = previous[1], previous[0]
    elif problem == "pending":
        value["pending_candidate_ids"] = ["new"]
    elif problem == "incomplete_audit":
        proof["regenerated_strategy_paths"] = 303
    elif problem == "changed_head":
        proof["event_chain_sha256"] = "2"*64
    else:
        value["registered_candidates"].pop()
    with pytest.raises(QuantError):
        audit.cohort_members(value, proof, previous, policy)


@pytest.mark.parametrize("field,value", [
    ("bootstrap_samples", 10000), ("block_sessions", 63), ("seed", 20261012),
    ("familywise_cell_count", 2), ("alpha_per_cohort_horizon_cell", 0.025),
    ("cohort_candidate_counts", {"legacy_original":19, "cash_funded_v2":56}),
    ("earlier_configurations_not_in_joint_inference", 0),
    ("change_research_qualification_policy", True), ("full_search_selection_adjusted_alpha_verified", True),
])
def test_cohort_statistic_alpha_and_scope_cannot_be_refitted(policy, field, value):
    policy[field] = value
    with pytest.raises(QuantError):
        audit.validate_policy(policy)


def test_original_version_only_and_unknown_explicit_tags_are_not_current_funded_accounts():
    assert audit.accounting_cohort({"candidate_id":"old"}) == "legacy_original"
    assert audit.accounting_cohort({"accounting_engine":engine_reference()}) == "cash_funded_v2"
    with pytest.raises(QuantError):
        audit.accounting_cohort({"accounting_engine":None})


def test_all_original_cost_pairs_enter_separate_full_horizon_matrices(policy, monkeypatch):
    value, proof, previous = state()
    cohorts = audit.cohort_members(value, proof, previous, policy)
    bundle = {"accounting_engine":None}
    calls = []
    bindings = {}
    for review in value["candidate_reviews"]:
        actual = {"accounting_engine":review.get("accounting_engine")}
        review["evidence_sha256"] = digest_json(actual)
        bindings[review["candidate_id"]] = {"bundle":{"path":review["candidate_id"], "sha256":"1"*64},
                                           "review_evidence_sha256":review["evidence_sha256"]}

    def original(path):
        identifier = path.name
        review = next(row for row in value["candidate_reviews"] if row["candidate_id"] == identifier)
        return {"accounting_engine":review.get("accounting_engine")}

    def accounts(bundle, review, years, settings):
        calls.append((review["candidate_id"], years))
        index = sessions("2016-10-06" if years == 10 else "2021-10-06", "2026-10-05")
        return {"base":{"strategy":pd.DataFrame({"return":0.002}, index=index),
                        "spy":pd.DataFrame({"return":0.001}, index=index)},
                "stress":{"strategy":pd.DataFrame({"return":0.0015}, index=index),
                          "spy":pd.DataFrame({"return":0.0008}, index=index)}}

    monkeypatch.setattr(audit, "safe_file", lambda root, path, digest: Path(path))
    monkeypatch.setattr(audit, "read_json", original)
    monkeypatch.setattr(audit, "verified_accounts", accounts)
    frames = audit.matrices(value, cohorts, policy, bindings)
    assert len(calls) == 152
    assert frames["legacy_original"]["10"].shape == (2512, 57)
    assert frames["cash_funded_v2"]["10"].shape == (2512, 171)
    assert frames["legacy_original"]["5"].shape == (1254, 57)
    assert frames["cash_funded_v2"]["5"].shape == (1254, 171)
    assert frames["legacy_original"]["10"].columns[-1] == "candidate_18::stress_vs_base"
    assert frames["cash_funded_v2"]["10"].columns[0] == "candidate_19::base_vs_base"


def small_inputs():
    rng = np.random.default_rng(19)
    result = {name:{} for name in audit.COHORTS}
    for years, count in (("10", 137), ("5", 91)):
        frame = pd.DataFrame(rng.normal(0, 0.002, (count, 3)),
                             columns=["a::base_vs_base", "a::stress_vs_stress", "a::stress_vs_base"],
                             index=pd.date_range("2020-01-01", periods=count))
        for cohort in audit.COHORTS:
            result[cohort][years] = frame.copy()
    return result


def test_equal_data_get_equal_common_blocks_and_four_cell_bonferroni_without_cohort_independence(policy):
    settings = deepcopy(policy)
    settings["bootstrap_samples"] = 128
    values = small_inputs()
    result = audit.analyze(values, settings)
    for years in ("10", "5"):
        left, right = (result[name][years] for name in audit.COHORTS)
        assert left == right
        assert left["alpha_allocated_to_horizon"] == 0.0125
        assert left["seed"] == 20261011+(years == "5")
        assert len(left["comparison_results"]) == 3
    assert len([value for group in result.values() for value in group.values()]) == 4
    assert sum(value["alpha_allocated_to_horizon"] for group in result.values() for value in group.values()) == 0.05


@pytest.mark.parametrize("problem", ["missing_cohort", "missing_window", "date_mismatch"])
def test_corrupt_or_misaligned_cells_do_not_claim_common_date_resampling(policy, problem):
    values = small_inputs()
    if problem == "missing_cohort":
        values.pop("legacy_original")
    elif problem == "missing_window":
        values["legacy_original"].pop("5")
    else:
        values["legacy_original"]["10"] = values["legacy_original"]["10"].iloc[1:]
    with pytest.raises(QuantError):
        audit.analyze(values, policy)


def test_publication_uses_frozen_import_guard_and_explicit_new_files_without_duplicate_core_approvals(policy, monkeypatch):
    calls = []
    reference = {"path":"evidence/current.json", "sha256":"1"*64}

    def core(ref, entries, source_policy, commit):
        calls.append(("manifest", ref, entries, source_policy, commit))
        return {"published_import_manifest_verified":True, "verified_imported_core_file_count":7}

    def explicit(root, ref, required, commit):
        calls.append(("explicit", root, ref, required, commit))
        assert not {"src/us_quant/calendar.py", "src/us_quant/metrics.py", "src/us_quant/storage.py"} & required
        assert {"src/us_quant/selection_accounting_cohorts.py", "config/selection-accounting-cohorts.json"} <= required
        assert {policy[key]["path"] for key in audit.REFERENCES} <= required
        return {"published_commit":commit, "registration_packet":ref}

    monkeypatch.setattr(audit.publication_core, "published_inputs", core)
    monkeypatch.setattr(audit, "verify_publication", explicit)
    result = audit.published_inputs(reference, policy, "2"*40)
    assert [call[0] for call in calls] == ["manifest", "explicit"]
    assert result["published_frozen_imported_core"]["published_import_manifest_verified"] is True
    assert result["published_commit"] == "2"*40
