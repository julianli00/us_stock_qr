from contextlib import nullcontext
from copy import deepcopy
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
from test_expert_contextual_contest_v1 import synthetic

from us_quant import ssd_expert_payoff_persistence_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json


@pytest.fixture(scope="module")
def prepared():
    experts, close = synthetic()
    records = []
    targets, audits = study.rules.build_targets(
        experts, close, read_json(study.rules.POLICY), record_learning=records.append
    )
    return SimpleNamespace(close=close), experts, targets, audits, records


def test_independent_integer_mass_matches_exact_rational_path():
    for refresh in (False, True):
        mass = (1,) * 6
        state = study.rules.PayoffMassState.initial()
        for i in range(119):
            relatives = tuple(float(1 + 0.02 * np.sin(i / 7 + j)) for j in range(6))
            state, score = state.advance(relatives, refresh)
            mass, independent_score = study.integer_advance(mass, relatives, refresh)
            assert tuple(Fraction(value, sum(mass)) for value in mass) == state.mass
            assert independent_score == score


def test_complete_independent_saved_input_risk_and_target_proof(prepared):
    data, experts, targets, audits, records = prepared
    proof = study.verify_targets(data, experts, targets, audits, iter(records))
    assert proof["decisions_checked"] == len(audits)
    assert proof["complete_source_months_checked"] == len(audits) - 2
    assert proof["unscored_initial_decisions"] == 7
    assert proof["exact_integer_learning_mass_and_source_scores_match"] is True
    assert proof["maximum_independent_expert_weight_difference"] < 1e-12


@pytest.mark.parametrize("kind", ["mass", "score", "hex", "target", "clock", "risk"])
def test_independent_proof_rejects_mass_score_hex_and_target_tampering(prepared, kind):
    data, experts, targets, audits, records = deepcopy(prepared)
    if kind == "mass":
        row = next(r for r in records if r["stage"] == "allocation_resolved")
        row["allocation"]["learning_mass_after_update"][0] = study.rules.rational_record(
            Fraction(1, 7)
        )
    elif kind == "score":
        row = next(
            r
            for r in records
            if r["stage"] == "allocation_resolved"
            and r["allocation"]["prequential_source_score_relative"] is not None
        )
        row["allocation"]["prequential_source_score_relative"] = study.rules.rational_record(
            Fraction(2)
        )
    elif kind == "hex":
        row = next(
            r
            for r in records
            if r["stage"] == "input" and r["monthly_relative_input_hex"] is not None
        )
        row["monthly_relative_input_hex"][0] = "0x1.0000000000000p+0"
    elif kind == "target":
        targets[study.IDS[0]].iloc[targets[study.IDS[0]].index.get_loc("2016-10-05"), 0] += 0.001
    elif kind == "clock":
        audits[0]["decision_session"] = "2016-10-06"
    else:
        row = next(
            r
            for r in records
            if r["stage"] == "allocation_resolved"
            and r["allocation"]["co_loss_assessment"] is not None
        )
        row["allocation"]["co_loss_assessment"]["risk_tilt_reason"] = "invented"
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, targets, audits, iter(records))


def test_cached_sources_and_verifier_never_call_closed_entrypoints(prepared, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("A frozen source strategy or account pipeline was replayed.")

    for module in (study.source, study.risk_template):
        for name in ("prepare", "register", "evaluate", "inputs"):
            monkeypatch.setattr(module, name, fail)
    data, experts, targets, audits, records = prepared
    assert (
        study.verify_targets(data, experts, targets, audits, iter(records))["decisions_checked"] > 0
    )


def test_complete_import_closure_does_not_mutate_template_globals():
    before = study.import_template.OWN
    code, policies = study.dependency_inputs()
    assert study.import_template.OWN is before
    assert any(ref["path"] == "src/us_quant/expert_contextual_contest_v1.py" for ref in code)
    assert any(ref["path"] == "src/us_quant/expert_initial_monthly_execution_v1.py" for ref in code)
    assert all(ref["path"] != study.OWN[0] for ref in policies)
    assert "src/us_quant/ssd_expert_payoff_persistence_v1.py" not in {ref["path"] for ref in code}


def test_failed_publication_cannot_start_main_capital(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(study, "runtime_guard", lambda: None)
    monkeypatch.setattr(study, "research_job", lambda *args: nullcontext())
    monkeypatch.setattr(study, "inputs", lambda: (object(), (), {"study_id": "synthetic"}, {}))
    monkeypatch.setattr(study, "checked_packet", lambda frozen: {})
    monkeypatch.setattr(study, "RESULTS", tmp_path / "not-started.json")

    def failed(*args):
        raise QuantError("synthetic publication refused")

    monkeypatch.setattr(study, "publication", failed)
    monkeypatch.setattr(study, "failure", lambda *args: events.append(args))
    monkeypatch.setattr(
        study.execution,
        "evaluate_path",
        lambda *args: pytest.fail("Account started before publication."),
    )
    with pytest.raises(QuantError, match="publication refused"):
        study.evaluate("f" * 40)
    assert len(events) == 1 and events[0][0] == "expert_payoff_persistence_evaluation"


def test_original_monthly_execution_is_preserved():
    assert study.execution.__name__ == "us_quant.expert_initial_monthly_execution_v1"
    policy = read_json(study.rules.POLICY)
    assert policy["actual_weight_trade_band"] == 0.025
    assert policy["reference_capital_usd"] == 10000
    assert policy["invested_target_budget"] == 0.98
