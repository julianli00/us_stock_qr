from contextlib import nullcontext
from types import FunctionType

import pytest
import test_style_activity_expert_inclusion_v1 as preserved_tests

from us_quant import signed_style_activity_expert_inclusion_v2 as rules
from us_quant import ssd_signed_style_activity_expert_inclusion_v2 as study
from us_quant.config import QuantError
from us_quant.storage import read_json


def rebound(function):
    result = FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )
    result.__dict__.update(function.__dict__)
    return result


original = preserved_tests.original
examples = rebound(preserved_tests.examples)
prepared = rebound(preserved_tests.prepared)
for name, function in vars(preserved_tests).items():
    if name.startswith("test_") and callable(function):
        globals()[name] = rebound(function)


def test_namespace_correction_changes_no_source_scoring_or_financial_acceptance():
    old, new = read_json(rules.preserved.POLICY), read_json(rules.POLICY)
    changed = {
        "schema_version",
        "study_id",
        "candidates",
        "namespace_revision_only",
        "preserved_unregistered_collision",
    }
    assert {k: v for k, v in old.items() if k not in changed} == {
        k: v for k, v in new.items() if k not in changed
    }
    assert {c["id"] for c in rules.CANDIDATES}.isdisjoint(
        {c["id"] for c in rules.preserved.CANDIDATES}
    )


def test_foreign_second_identity_blocks_fresh_first_before_any_append(monkeypatch):
    rows = []
    frozen = {"study_id": "synthetic"}
    actual_call = study._call
    monkeypatch.setattr(study.lifecycle, "runtime_guard", lambda: None)
    monkeypatch.setattr(study.lifecycle, "research_job", nullcontext)
    monkeypatch.setattr(
        study,
        "_call",
        lambda name, *args: (None, None, frozen, None)
        if name == "inputs"
        else actual_call(name, *args),
    )
    monkeypatch.setattr(
        study,
        "events",
        lambda kind, name: [{"body": {"candidate": {"id": "foreign"}, "frozen": {}}}]
        if kind == "registered" and name == study.IDS[1]
        else [],
    )
    monkeypatch.setattr(study.base, "append_event", lambda *args: rows.append(args))
    with pytest.raises(QuantError, match="before appending any"):
        study.register()
    assert rows == []


def test_same_registered_cohort_can_pass_identity_check_without_old_ID_reclassification(
    monkeypatch,
):
    frozen = {"study_id": "synthetic"}
    monkeypatch.setattr(
        study,
        "events",
        lambda kind, name: [
            {
                "body": {
                    "candidate": next(c for c in study.CANDIDATES if c["id"] == name),
                    "frozen": frozen,
                }
            }
        ],
    )
    study.validate_namespace(frozen)
