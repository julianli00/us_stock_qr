import copy
import json
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import test_macro_expert_inclusion_v1 as existing
from test_strategy_expert_pool_eight_v1 import archived_experts as original

from us_quant import credit_expert_inclusion_v1 as rules
from us_quant import ssd_credit_expert_inclusion_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import ArchivedMonthlyExpert, ExpertIdentity


def examples(changed_after=None):
    experts, calendar, assets = original(changed_after=changed_after)
    old = experts[-1]
    target = old._targets.copy()
    book = old._shadow.copy()
    credit = ArchivedMonthlyExpert(
        ExpertIdentity(rules.EXPERT_IDS[-1], rules.MECHANISMS[-1], "2026-11-04T21:51:50Z"),
        target,
        book,
        assets,
    )
    return study.include(experts, credit), calendar, assets


def prepared():
    experts, calendar, assets = examples()
    records = []
    targets, audits = rules.build_targets(
        experts, calendar, assets, read_json(rules.POLICY), retain=records.append
    )
    return (
        SimpleNamespace(close=pd.DataFrame(index=calendar, columns=assets)),
        experts,
        targets,
        audits,
        records,
    )


def bind_test(function):
    bound = FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )
    bound.__dict__.update(function.__dict__)
    return bound


for name in (
    "test_nonselected_pair_or_missing36th_record_is_not_hidden",
    "test_both_real_registrations_precede_first_new_derivation",
    "test_real_packet_constructor_and_complete_cohort_metadata",
    "test_missing_other_packet_blocks_every_funded_candidate",
):
    globals()[name] = bind_test(getattr(existing, name))


def test_original_eight_preserved_and_only_credit_appended_with_exact_old_laws():
    experts, _, _ = examples()
    assert tuple(e.identity.identifier for e in experts) == rules.EXPERT_IDS
    assert rules.EXPERT_IDS[:8] == rules.prior.EXPERT_IDS
    assert rules.MECHANISMS[:8] == rules.prior.MECHANISMS
    assert len(experts) == 9
    assert rules.choose_pair is rules.prior.choose_pair
    assert rules.combine is rules.prior.combine


def test_reorder_replace_or_another_source_is_not_the_frozen_inclusion():
    old, _, _ = original()
    experts, _, _ = examples()
    with pytest.raises(QuantError):
        study.include((*old, old[0]), experts[-1])
    with pytest.raises(QuantError):
        study.include(tuple(reversed(old)), experts[-1])
    with pytest.raises(QuantError):
        study.include(old, old[0])


def test_nine_preserves_equal_ninths_seven_unscored_origins_and_all36_pairs():
    _, _, _, audits, _ = prepared()
    assert sum(a["selection"]["reason"] == "unscored_initial_equal_weights" for a in audits) == 7
    mature = [a for a in audits if a["selection"]["scores_available"]]
    assert mature[0]["decision_session"] == "2017-04-28"
    assert all(len(a["selection"]["pairs"]) == 36 for a in mature)
    assert all(a["equal_expert_weights"] == [1 / 9] * 9 for a in audits)


def test_prefix_and_future_source_changes_leave_every_past_target_and_score_unchanged():
    experts, calendar, assets = examples()
    changed, _, _ = examples(changed_after=pd.Timestamp("2017-04-28"))
    full, audits = rules.build_targets(experts, calendar, assets, read_json(rules.POLICY))
    after, future = rules.build_targets(changed, calendar, assets, read_json(rules.POLICY))
    early, first = rules.build_targets(
        experts, calendar[calendar <= "2017-04-28"], assets, read_json(rules.POLICY)
    )
    for name in study.IDS:
        assert early[name].equals(full[name].loc[:"2017-04-28"])
        assert early[name].equals(after[name].loc[:"2017-04-28"])
    assert first == [a for a in audits if a["decision_session"] <= "2017-04-28"]
    assert first == [a for a in future if a["decision_session"] <= "2017-04-28"]


def test_all36_scalar_oracles_and_fullprecision_retained_records(tmp_path):
    data, experts, targets, audits, records = prepared()
    proof = study.verify_targets(data, experts, targets, audits)
    assert proof["maximum_independent_target_difference"] < 1e-10
    assert proof["maximum_independent_net_score_difference"] < 1e-8
    path = tmp_path / "source-inputs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    retained = study.verify_retained_records(path, experts, targets, audits)
    assert retained["actual_input_and_resolved_record_count"] == 2 * len(audits)


def test_complete_before_inputs_survive_a_synthetic_new_selection_error(monkeypatch):
    experts, calendar, assets = examples()
    records = []

    def failed(*args):
        raise QuantError("Synthetic credit selection stop")

    monkeypatch.setattr(rules, "choose_pair", failed)
    with pytest.raises(QuantError, match="selection stop"):
        rules.build_targets(
            experts, calendar, assets, read_json(rules.POLICY), retain=records.append
        )
    assert len(records) == 1
    assert records[0]["observations"][-1]["id"] == rules.EXPERT_IDS[-1]


@pytest.mark.parametrize(
    "key,value",
    [
        ("lookback_sessions", 252),
        ("selected_experts", 3),
        ("actual_weight_trade_band", 0.01),
        ("all_pairs_per_mature_decision", 28),
        ("credit_financial_rejection_preserved", False),
        ("nine_study_sources_claimed_independent_economic_risks", True),
    ],
)
def test_no_window_band_pair_count_source_verdict_or_independence_rescue(key, value):
    policy = copy.deepcopy(read_json(rules.POLICY))
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)


def test_financial_namespaces_unchanged_and_dependencies_include_actual_native_credit():
    old = study.financial.ID, study.financial.FOLDER, study.financial.PACKET
    context = study.scoped(study.financial.evaluate, study.CANDIDATES[1])
    assert context.__globals__["ID"] == study.IDS[1]
    assert context.__globals__["publication"] is study.cohort_publication
    assert old == (study.financial.ID, study.financial.FOLDER, study.financial.PACKET)
    paths = {r["path"] for r in study.dependencies()}
    assert set(study.OWN).issubset(paths)
    assert "src/us_quant/ssd_credit_quarter_policy_v1.py" in paths
    assert "src/us_quant/credit_relative_input_v1.py" in paths


def test_equal_combined_target_remains_one_cash_funded_underlying_portfolio():
    experts, _, assets = examples()
    observed = [e.observe(pd.Timestamp("2017-04-28"), 126) for e in experts]
    combined = rules.combine(observed, np.full(9, 1 / 9))
    assert tuple(combined.index) == assets
    assert combined.sum() == pytest.approx(0.98)
    assert (combined >= 0).all()
