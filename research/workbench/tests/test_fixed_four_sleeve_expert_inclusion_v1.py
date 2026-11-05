from copy import deepcopy
from types import FunctionType, SimpleNamespace

import pandas as pd
import pytest
import test_macro_expert_inclusion_v1 as existing
from test_strategy_expert_pool_eight_v1 import archived_experts as original

from us_quant import fixed_four_sleeve_expert_inclusion_v1 as rules
from us_quant import ssd_fixed_four_sleeve_expert_inclusion_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import ArchivedMonthlyExpert, ExpertIdentity


def examples(changed_after=None):
    experts, calendar, assets = original(changed_after=changed_after)
    old = experts[-1]
    constant = old._targets.copy()
    constant.loc[constant.dropna(how="all").index] = [0.245] * len(assets)
    extra = ArchivedMonthlyExpert(
        ExpertIdentity(rules.EXPERT_IDS[-1], rules.MECHANISMS[-1], "2026-11-05T08:15:18Z"),
        constant,
        old._shadow.copy(),
        assets,
    )
    return study.include(experts, extra), calendar, assets


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
    rebound = FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )
    rebound.__dict__.update(function.__dict__)
    return rebound


for name in (
    "test_nonselected_pair_or_missing36th_record_is_not_hidden",
    "test_both_real_registrations_precede_first_new_derivation",
    "test_real_packet_constructor_and_complete_cohort_metadata",
    "test_missing_other_packet_blocks_every_funded_candidate",
):
    globals()[name] = bind_test(getattr(existing, name))


def test_original_eight_and_exact_equal_rolling_laws_preserved():
    experts, _, _ = examples()
    assert tuple(expert.identity.identifier for expert in experts) == rules.EXPERT_IDS
    assert rules.EXPERT_IDS[:8] == rules.prior.EXPERT_IDS
    assert rules.MECHANISMS[:8] == rules.prior.MECHANISMS
    assert rules.choose_pair is rules.prior.choose_pair
    assert rules.combine is rules.prior.combine


def test_no_source_replacement_reordering_or_automatic_other_ninth():
    old, _, _ = original()
    experts, _, _ = examples()
    with pytest.raises(QuantError):
        study.include(tuple(reversed(old)), experts[-1])
    with pytest.raises(QuantError):
        study.include((*old, old[0]), experts[-1])
    with pytest.raises(QuantError):
        study.include(old, old[0])


def test_all36_pairs_equal_ninths_and_explicit_unscored_initial_states():
    _, _, _, audits, _ = prepared()
    assert sum(a["selection"]["reason"] == "unscored_initial_equal_weights" for a in audits) == 7
    assert all(a["equal_expert_weights"] == [1 / 9] * 9 for a in audits)
    mature = [a for a in audits if a["selection"]["scores_available"]]
    assert mature[0]["decision_session"] == "2017-04-28"
    assert all(len(a["selection"]["pairs"]) == 36 for a in mature)


def test_prefix_and_future_changes_preserve_every_past_target_weight_score():
    experts, calendar, assets = examples()
    changed, _, _ = examples(changed_after=pd.Timestamp("2017-04-28"))
    full, audits = rules.build_targets(experts, calendar, assets, read_json(rules.POLICY))
    after, later = rules.build_targets(changed, calendar, assets, read_json(rules.POLICY))
    early, earlier = rules.build_targets(
        experts, calendar[calendar <= "2017-04-28"], assets, read_json(rules.POLICY)
    )
    for name in study.IDS:
        assert early[name].equals(full[name].loc[:"2017-04-28"])
        assert early[name].equals(after[name].loc[:"2017-04-28"])
    assert earlier == [a for a in audits if a["decision_session"] <= "2017-04-28"]
    assert earlier == [a for a in later if a["decision_session"] <= "2017-04-28"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("lookback_sessions", 252),
        ("selected_experts", 3),
        ("all_pairs_per_mature_decision", 28),
        ("actual_weight_trade_band", 0.01),
        ("fixed_four_sleeve_financial_rejection_preserved", False),
        ("nine_study_sources_claimed_independent_economic_risks", True),
        ("prior_completed_configurations", 228),
        ("prior_completed_configurations", 259),
    ],
)
def test_no_scoring_band_paircount_verdict_or_prior_rescue(key, value):
    policy = deepcopy(read_json(rules.POLICY))
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)


def test_source_scope_is_not_global_admission_or_enabling_archives():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    assert policy["original_active_seven_or_selected_eight_rosters_changed"] is False
    assert "unreviewed" in policy["pool_scope"]["unreviewed_archives"]
    assert "not financial-performance exclusion" in policy["pool_scope"]["admissible_nonselected"]
    policy["fixed_four_sleeve_source"]["introduced_at"] = "2015-09-30T00:00:00Z"
    with pytest.raises(QuantError):
        rules.validate(policy)


def test_separate_namespaces_do_not_mutate_source_financial_globals():
    before = study.financial.ID, study.financial.FOLDER, study.financial.PACKET
    scoped = study.scoped(study.financial.evaluate, study.CANDIDATES[1])
    assert scoped.__globals__["ID"] == study.IDS[1]
    assert scoped.__globals__["publication"] is study.cohort_publication
    assert before == (study.financial.ID, study.financial.FOLDER, study.financial.PACKET)
    paths = {row["path"] for row in study.dependencies()}
    assert set(study.OWN).issubset(paths)
    assert "src/us_quant/fixed_four_sleeve_admission_v1.py" in paths
