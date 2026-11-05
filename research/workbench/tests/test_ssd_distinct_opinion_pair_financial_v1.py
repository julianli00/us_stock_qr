import json
from types import FunctionType, SimpleNamespace

import pandas as pd
import pytest
import test_ssd_expert_instrument_concentration_v1 as existing
from test_expert_distinct_opinion_pair_v1 import examples

from us_quant import ssd_distinct_opinion_pair_financial_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json


def fixture(tmp_path):
    experts, calendar = examples()
    rows = []
    frames, audits = study.rules.build_targets(
        experts, calendar, study.rules.UNIVERSE, read_json(study.rules.POLICY), retain=rows.append
    )
    data = SimpleNamespace(close=pd.DataFrame(100.0, index=calendar, columns=study.rules.UNIVERSE))
    path = tmp_path / "retained.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return data, experts, path, frames[study.ID], audits, rows


def test_independent_original_scores_exact_eligibility_and_single_target_are_verified(tmp_path):
    data, experts, path, target, audits, rows = fixture(tmp_path)
    proof = study.verify_retained(path, data, experts, target, audits)
    assert proof["verified_decisions"] == len(audits)
    assert proof["actual_input_resolved_records"] == len(rows)
    assert proof["all36_mature_catalogues_and_exact_eligibility_verified"] is True
    assert proof["maximum_independent_target_difference"] < 1e-12
    assert proof["maximum_independent_expert_weight_difference"] < 1e-10
    assert proof["maximum_independent_net_score_difference"] < 1e-8


@pytest.mark.parametrize(
    "kind",
    [
        "source_target",
        "source_NET",
        "source_RF",
        "prior",
        "known_at",
        "drop_pair",
        "eligibility",
        "choice",
        "score",
        "target",
    ],
)
def test_retained_operand_catalogue_eligibility_choice_or_target_tampering_fails(tmp_path, kind):
    data, experts, path, target, audits, rows = fixture(tmp_path)
    mature_index = next(i for i, a in enumerate(audits) if a["selection"]["scores_available"])
    before, after = rows[2 * mature_index], rows[2 * mature_index + 1]
    pair = after["audit"]["selection"]["pairs"][0]
    if kind == "source_target":
        before["observations"][0]["target_binary64_hex"][0] = (0.25).hex()
    elif kind == "source_NET":
        before["observations"][0]["NET_returns_binary64_hex"][0] = (0.25).hex()
    elif kind == "source_RF":
        before["observations"][0]["RF_binary64_hex"][0] = (0.25).hex()
    elif kind == "prior":
        before["previous_weights_binary64_hex"][0] = (0.5).hex()
    elif kind == "known_at":
        before["observations"][0]["target_known_at"] = "2026-10-05"
    elif kind == "drop_pair":
        after["audit"]["selection"]["pairs"].pop()
    elif kind == "eligibility":
        pair["eligible_by_current_opinion"] = not pair["eligible_by_current_opinion"]
    elif kind == "choice":
        after["audit"]["selection"]["selected_pair"] = list(study.rules.EXPERT_IDS[:2])
    elif kind == "score":
        scored = next(p for p in after["audit"]["selection"]["pairs"] if "selection_score" in p)
        scored["selection_score"] += 0.5
    else:
        target.iloc[
            target.index.get_loc(pd.Timestamp(audits[mature_index]["decision_session"])), 0
        ] += 0.01
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(QuantError):
        study.verify_retained(path, data, experts, target, audits)


def test_all_equal_mature_opinions_and_unscored_origins_have_explicit_proof(tmp_path):
    experts, calendar = examples()
    for expert in experts:
        expert._targets.loc[expert._active.index] = experts[0]._active.to_numpy()
        expert._active = expert._targets.dropna(how="all")
    rows = []
    frames, audits = study.rules.build_targets(
        experts, calendar, study.rules.UNIVERSE, read_json(study.rules.POLICY), retain=rows.append
    )
    path = tmp_path / "retained.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    data = SimpleNamespace(close=pd.DataFrame(100.0, index=calendar, columns=study.rules.UNIVERSE))
    proof = study.verify_retained(path, data, experts, frames[study.ID], audits)
    assert proof["verified_decisions"] == len(audits)
    assert any(
        a["selection"]["NET_history_mature"] and not a["selection"]["scores_available"]
        for a in audits
    )


def test_distinct_but_zero_variance_menu_fallback_has_independent_proof(tmp_path):
    experts, calendar = examples()
    for expert in experts:
        expert._shadow.loc[:, "return"] = 0.0
        expert._shadow.loc[:, "equity"] = 10000.0
    rows = []
    frames, audits = study.rules.build_targets(
        experts, calendar, study.rules.UNIVERSE, read_json(study.rules.POLICY), retain=rows.append
    )
    path = tmp_path / "retained.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    data = SimpleNamespace(close=pd.DataFrame(100.0, index=calendar, columns=study.rules.UNIVERSE))
    proof = study.verify_retained(path, data, experts, frames[study.ID], audits)
    assert proof["verified_decisions"] == len(audits)
    assert any(
        a["selection"]["reason"] == "no_scorable_distinct_current_opinion_pair_equal_weights"
        for a in audits
    )


def test_extra_records_and_nonmonthly_target_cannot_be_ignored(tmp_path):
    data, experts, path, target, audits, rows = fixture(tmp_path)
    target.loc["2016-10-12"] = target.loc["2016-10-05"]
    with pytest.raises(QuantError):
        study.verify_retained(path, data, experts, target, audits)
    target.loc["2016-10-12"] = float("nan")
    path.write_text(path.read_text() + json.dumps(rows[-1]) + "\n")
    with pytest.raises(QuantError):
        study.verify_retained(path, data, experts, target, audits)


def _bind_test(function):
    result = FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )
    result.__dict__.update(function.__dict__)
    return result


for name in (
    "test_unregistered_opinions_never_generate_current_hhi_or_targets",
    "test_funding_scope_keeps_old_monthly_execution_and_does_not_mutate_template",
    "test_unmodified_real_packet_contract_is_used_after_registration",
):
    globals()[name] = _bind_test(getattr(existing, name))


def test_financial_dependencies_bind_the_unchanged_primary_and_new_program():
    paths = {row["path"] for row in study.dependencies()}
    assert set(study.OWN).issubset(paths)
    assert "src/us_quant/ssd_expert_target_consensus_v1.py" in paths
    assert study.financial.evaluate_path.__globals__["ID"] != study.ID
