import copy
from types import FunctionType, SimpleNamespace

import pandas as pd
import pytest
import test_hmm_expert_substitution_v1 as existing
from test_hmm_expert_substitution_v1 import examples as mother_examples

from us_quant import ssd_vxn_hmm_composition_substitution_v1 as study
from us_quant import vxn_hmm_composition_substitution_v1 as rules
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import ArchivedMonthlyExpert, ExpertIdentity


def examples(changed_after=None):
    experts, calendar, assets = mother_examples(changed_after=changed_after)
    old = experts[2]
    new = ArchivedMonthlyExpert(
        ExpertIdentity(rules.EXPERT_IDS[2], rules.MECHANISMS[2], "2026-10-29T10:50:07Z"),
        old._targets,
        old._shadow,
        assets,
    )
    return study.substitute(experts, new), calendar, assets


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
    result = FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )
    result.__dict__.update(function.__dict__)
    return result


for name in (
    "test_prefix_and_future_source_changes_do_not_change_substituted_past",
    "test_new_source_full_pair_scalar_oracle_and_retained_records",
    "test_nonselected_pair_or_missing_record_is_not_hidden",
    "test_both_real_registrations_precede_first_new_derivation",
    "test_real_packet_constructor_and_complete_cohort_metadata",
    "test_missing_other_packet_blocks_every_funded_candidate",
    "test_source_and_financial_global_namespaces_are_not_mutated",
):
    globals()[name] = bind_test(getattr(existing, name))


def test_only_HMM_composition_slot_changes_and_laws_are_the_exact_original_functions():
    experts, _, _ = examples()
    assert tuple(e.identity.identifier for e in experts) == rules.EXPERT_IDS
    assert len(experts) == 8
    assert [i for i in range(8) if rules.EXPERT_IDS[i] != rules.control.EXPERT_IDS[i]] == [2]
    assert rules.control.EXPERT_IDS[2] == "causal_two_state_growth_bil_v1"
    assert rules.EXPERT_IDS[2] == "hmm_vxn_relative_uncertainty_v1"
    assert rules.MECHANISMS == rules.control.MECHANISMS
    assert rules.choose_pair is rules.control.choose_pair
    assert rules.combine is rules.control.combine


def test_appending_reordering_or_replacing_another_slot_is_not_this_experiment():
    old, _, _ = mother_examples()
    changed, _, _ = examples()
    assert study.substitute(old, changed[2])[2] is changed[2]
    with pytest.raises(QuantError):
        study.substitute((*old, old[0]), changed[2])
    with pytest.raises(QuantError):
        study.substitute(tuple(reversed(old)), changed[2])
    with pytest.raises(QuantError):
        study.substitute(old, changed[0])


def test_identical_synthetic_source_values_reproduce_cached_law_not_a_new_rank_metric():
    experts, calendar, assets = examples()
    old, _, _ = mother_examples()
    actual, _ = rules.build_targets(experts, calendar, assets, read_json(rules.POLICY))
    control, _ = rules.control.build_targets(old, calendar, assets, read_json(rules.control.POLICY))
    for new_id, old_candidate in zip(study.IDS, rules.control.CANDIDATES, strict=True):
        assert actual[new_id].equals(control[old_candidate["id"]])


def test_seven_unscored_origins_and_all28pairs_remain_explicit():
    _, _, _, audits, _ = prepared()
    assert sum(a["selection"]["reason"] == "unscored_initial_equal_weights" for a in audits) == 7
    mature = [a for a in audits if a["selection"]["scores_available"]]
    assert mature[0]["decision_session"] == "2017-04-28"
    assert all(len(a["selection"]["pairs"]) == 28 for a in mature)
    assert all(
        "causal_two_state_growth_bil_v1" not in p["pair"]
        for a in mature
        for p in a["selection"]["pairs"]
    )


def test_actual_inputs_survive_new_selection_error_without_rerunning_sources(monkeypatch):
    experts, calendar, assets = examples()
    rows = []

    def failed(*args):
        raise QuantError("synthetic score stop")

    monkeypatch.setattr(rules, "choose_pair", failed)
    with pytest.raises(QuantError, match="synthetic score"):
        rules.build_targets(experts, calendar, assets, read_json(rules.POLICY), retain=rows.append)
    assert len(rows) == 1
    assert rows[0]["observations"][2]["id"] == "hmm_vxn_relative_uncertainty_v1"


@pytest.mark.parametrize(
    "key,value",
    [
        ("lookback_sessions", 252),
        ("selected_experts", 3),
        ("switch_cost_bps_per_side", 0.0),
        ("actual_weight_trade_band", 0.01),
        ("source_variants_are_independent_economic_factors", True),
        ("replaced_expert_id", "symmetric_garch_growth_bil_v1"),
        ("vxn_eligibility_completion_sha256", "0" * 64),
    ],
)
def test_no_window_cost_band_independence_or_qualification_rescue(key, value):
    policy = copy.deepcopy(read_json(rules.POLICY))
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)
