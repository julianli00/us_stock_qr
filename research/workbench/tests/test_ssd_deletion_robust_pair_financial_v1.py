import json
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import test_ssd_expert_instrument_concentration_v1 as existing
from test_expert_distinct_opinion_pair_v1 import examples

from us_quant import ssd_deletion_robust_pair_financial_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json


def fixture(tmp_path, *, conditional_degenerate=False, original_unscorable=False):
    experts, calendar = examples()
    if conditional_degenerate or original_unscorable:
        days = experts[0]._shadow.index
        values = np.zeros(len(days))
        if conditional_degenerate:
            first = days[days <= "2017-04-28"][-126:]
            positions = days.get_indexer(first)
            values[positions[10]] = 0.01
            values[positions[84:]] = np.sin(np.arange(42)) * 0.01
            values[days > "2017-04-28"] = np.sin(np.arange(sum(days > "2017-04-28"))) * 0.01
        for expert in experts:
            expert._shadow.loc[:, "return"] = values
            expert._shadow.loc[:, "equity"] = 10000 * np.cumprod(1 + values)
    rows = []
    frames, audits = study.rules.build_targets(
        experts, calendar, study.rules.UNIVERSE, read_json(study.rules.POLICY), retain=rows.append
    )
    path = tmp_path / "retained.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    data = SimpleNamespace(close=pd.DataFrame(100.0, index=calendar, columns=study.rules.UNIVERSE))
    return data, experts, path, frames[study.ID], audits, rows


def test_full_independent_source_risk_cost_profile_and_single_target_proof(tmp_path):
    data, experts, path, target, audits, rows = fixture(tmp_path)
    proof = study.verify_retained(path, data, experts, target, audits)
    assert proof["verified_decisions"] == len(audits)
    assert proof["actual_input_resolved_records"] == len(rows)
    assert proof["all36_original_catalogues_and_all128_conditional_profile_scores_verified"]
    assert proof["independent_conditional_sample_scores_checked"] > 0
    assert proof["maximum_independent_conditional_profile_difference"] < 1e-8


@pytest.mark.parametrize(
    "kind",
    [
        "source_NET",
        "source_RF",
        "source_target",
        "source_clock",
        "prior",
        "drop_pair",
        "pair_weight",
        "point_score",
        "cost",
        "correlation",
        "deleted_score",
        "undefined",
        "lower",
        "key",
        "admissible",
        "selected",
        "target",
    ],
)
def test_independent_proof_rejects_source_catalogue_profile_or_winner_tampering(tmp_path, kind):
    data, experts, path, target, audits, rows = fixture(tmp_path)
    position = next(i for i, a in enumerate(audits) if a["selection"]["scores_available"])
    before, after = rows[2 * position], rows[2 * position + 1]
    selection = after["audit"]["selection"]
    pair = next(p for p in selection["pairs"] if p["conditional_deletion_admissible"])
    profile = pair["conditional_profile"]
    if kind == "source_NET":
        before["observations"][0]["NET_returns_binary64_hex"][0] = (0.25).hex()
    elif kind == "source_RF":
        before["observations"][0]["RF_binary64_hex"][0] = (0.25).hex()
    elif kind == "source_target":
        before["observations"][0]["target_binary64_hex"][0] = (0.25).hex()
    elif kind == "source_clock":
        before["observations"][0]["target_known_at"] = "2026-10-05"
    elif kind == "prior":
        before["previous_weights_binary64_hex"][0] = (0.5).hex()
    elif kind == "drop_pair":
        selection["pairs"].pop()
    elif kind == "pair_weight":
        pair["expert_weights"][0] += 0.2
    elif kind == "point_score":
        pair["train_net_risk_score"] += 0.2
    elif kind == "cost":
        pair["estimated_incremental_switch_cost_fraction"] += 0.2
    elif kind == "correlation":
        pair["training_correlation"] += 0.2
    elif kind == "deleted_score":
        profile["validation_one_return_deleted_scores"][-1] += 0.2
    elif kind == "undefined":
        profile["undefined_sample_positions_full_train_valid_then_train_valid_deletions"] = [127]
    elif kind == "lower":
        profile["conditional_lower_envelope"] += 0.2
    elif kind == "key":
        profile["conditional_lower_envelope_key"] += 0.2
    elif kind == "admissible":
        pair["conditional_deletion_admissible"] = False
    elif kind == "selected":
        selection["selected_pair"] = ["not", "source"]
    else:
        target.loc[audits[position]["decision_session"], target.columns[0]] += 0.01
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(QuantError):
        study.verify_retained(path, data, experts, target, audits)


@pytest.mark.parametrize("kind", ["conditional_degenerate", "original_unscorable"])
def test_mature_fallback_profiles_are_verified_without_safe_or_success_claim(tmp_path, kind):
    data, experts, path, target, audits, _ = fixture(tmp_path, **{kind: True})
    proof = study.verify_retained(path, data, experts, target, audits)
    assert proof["verified_decisions"] == len(audits)
    assert any(
        a["selection"]["NET_history_mature"] and not a["selection"]["scores_available"]
        for a in audits
    )


def test_extra_records_and_nonmonthly_targets_cannot_be_ignored(tmp_path):
    data, experts, path, target, audits, rows = fixture(tmp_path)
    target.loc["2016-10-12"] = target.loc["2016-10-05"]
    with pytest.raises(QuantError):
        study.verify_retained(path, data, experts, target, audits)
    target.loc["2016-10-12"] = float("nan")
    path.write_text(path.read_text() + json.dumps(rows[-1]) + "\n")
    with pytest.raises(QuantError):
        study.verify_retained(path, data, experts, target, audits)


def _bind_test(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


for name in (
    "test_unregistered_opinions_never_generate_current_hhi_or_targets",
    "test_funding_scope_keeps_old_monthly_execution_and_does_not_mutate_template",
    "test_unmodified_real_packet_contract_is_used_after_registration",
):
    globals()[name] = _bind_test(getattr(existing, name))


def test_dependencies_explicitly_pin_all_reused_test_import_sources():
    paths = {r["path"] for r in study.dependencies()}
    assert set(study.OWN).issubset(paths)
    assert "tests/test_ssd_expert_instrument_concentration_v1.py" in paths
    assert "tests/test_expert_native_arrival_equal_v1.py" in paths
    assert "src/us_quant/ssd_expert_target_consensus_v1.py" in paths
    assert study.financial.evaluate_path.__globals__["ID"] != study.ID
