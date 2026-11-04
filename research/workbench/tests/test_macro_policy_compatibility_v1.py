from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_quant import macro_policy_compatibility_v1 as rules
from us_quant.calendar import is_month_end, previous_session, sessions
from us_quant.config import QuantError
from us_quant.strategy_experts_v1 import ArchivedMonthlyExpert, ExpertIdentity


def source_data():
    dates = sessions("2015-08-10", "2026-10-05")
    cols = ("SPY", "IEF", "GLD", "BIL", "MTUM", "VLUE", "QUAL", "USMV")
    target = pd.DataFrame(np.nan, index=dates, columns=cols)
    target.loc[[is_month_end(d) for d in dates]] = [
        0.0,
        0.0,
        0.294,
        0.0,
        0.1715,
        0.1715,
        0.1715,
        0.1715,
    ]
    variants = {}
    for key in rules.PATHS:
        start = "2016-10-06" if key.startswith("10y") else "2021-10-06"
        anchor = previous_session(start)
        t = target.copy()
        t.loc[anchor] = t.loc[:anchor].dropna(how="all").iloc[-1]
        variants[key] = t
    ret_dates = sessions("2016-10-06", "2026-10-05")
    returns = 0.0002 + 0.001 * np.sin(np.arange(len(ret_dates)) / 11)
    nav = 10000 * np.cumprod(1 + returns)
    book = pd.DataFrame(
        {
            "equity": nav,
            "return": returns,
            "cash": nav * 0.02,
            "gross_exposure": 0.98,
            "cost": 0.0,
            "orders": 0,
            "turnover": 0.0,
            "risk_free": 0.0001,
        },
        index=ret_dates,
    )
    native_rf = pd.Series(0.0001, index=dates, name="risk_free")
    common = native_rf.copy()
    common.iloc[common.index.get_loc(ret_dates[0])] = np.nextafter(0.0001, 1.0)
    identity = ExpertIdentity(rules.CANDIDATE, rules.GROUP, "2026-11-03T17:31:02Z")
    return identity, variants, book, native_rf, common, cols


def test_native_policy_funding_seed_and_nondeclaration_remain_different_roles():
    identity, variants, book, native_rf, common, cols = source_data()
    saved = {k: t.copy() for k, t in variants.items()}
    expert = rules.MacroSeedAwarePolicyOpinion(
        identity, variants, rules.assessment_view(book, native_rf, common), cols
    )
    initial = pd.Timestamp("2016-10-05")
    observation = expert.observe(initial, 126)
    context = expert.event_context(initial)
    assert observation.target_known_at == pd.Timestamp("2016-09-30")
    assert observation.net_returns.empty and not observation.score_mature
    assert not context["native_policy_recommendation_emitted_today"]
    assert (
        context["native_capital_initialization_events"][0]["original_capital_initialization_date"]
        == "2016-10-05"
    )
    assert not context["observation_is_a_new_rebalance_request"]
    non = expert.event_context(pd.Timestamp("2016-10-18"))
    assert (
        non["no_new_native_source_request_today"]
        and not non["observation_is_a_new_rebalance_request"]
    )
    month = expert.event_context(pd.Timestamp("2016-10-31"))
    assert month["native_policy_recommendation_emitted_today"]
    assert expert.observe(pd.Timestamp("2016-10-31"), 126).target_known_at == pd.Timestamp(
        "2016-10-31"
    )
    assert all(expert._native_variants[k].equals(saved[k]) for k in variants)
    assert expert.preservation()["seed_stripped_nominal_view_artifacts"] == 0


def test_explicit_assessment_view_changes_only_exact_reference_not_native_money():
    _, _, book, native_rf, common, _ = source_data()
    before = book.copy()
    view = rules.assessment_view(book, native_rf, common)
    assert book.equals(before)
    assert view.drop(columns="risk_free").equals(book.drop(columns="risk_free"))
    assert view["risk_free"].equals(common.loc[book.index])
    assert not view["risk_free"].equals(book["risk_free"])


def test_prefix_truncations_preserve_known_opinion_event_and_assessment():
    identity, variants, book, native_rf, common, cols = source_data()
    view = rules.assessment_view(book, native_rf, common)
    full = rules.MacroSeedAwarePolicyOpinion(identity, variants, view, cols)
    for day in map(pd.Timestamp, ("2016-10-31", "2017-04-28", "2021-10-05")):
        early = rules.MacroSeedAwarePolicyOpinion(
            identity, {k: t.loc[:day] for k, t in variants.items()}, view.loc[:day], cols
        )
        assert rules.same_observation(early.observe(day, 126), full.observe(day, 126))
        assert early.event_context(day) == full.event_context(day)


@pytest.mark.parametrize("kind", ["unknown_event", "seed_value", "wrong_source", "missing_month"])
def test_term_style_events_stale_source_identity_or_missing_month_are_not_admitted(kind):
    identity, variants, book, native_rf, common, cols = source_data()
    if kind == "unknown_event":
        variants["10y-base"].loc["2017-04-18"] = variants["10y-base"].loc["2017-03-31"]
    elif kind == "seed_value":
        variants["10y-base"].loc["2016-10-05", "GLD"] += 0.01
    elif kind == "wrong_source":
        identity = ExpertIdentity("ANFCI_unadmitted", rules.GROUP, identity.introduced_at)
    else:
        for t in variants.values():
            t.loc["2017-04-28"] = np.nan
    with pytest.raises(QuantError):
        rules.MacroSeedAwarePolicyOpinion(
            identity, variants, rules.assessment_view(book, native_rf, common), cols
        )


def test_existing_approved_monthly_observer_behavior_not_mutated_by_new_adapter():
    identity, variants, book, native_rf, common, cols = source_data()
    target = variants["10y-base"].copy()
    target.loc["2016-10-05"] = np.nan
    old = ArchivedMonthlyExpert(
        ExpertIdentity("existing", "existing_group", identity.introduced_at),
        target,
        rules.assessment_view(book, native_rf, common),
        cols,
    )
    day = pd.Timestamp("2017-04-28")
    before = old.observe(day, 126)
    rules.MacroSeedAwarePolicyOpinion(
        identity, variants, rules.assessment_view(book, native_rf, common), cols
    ).observe(day, 126)
    assert rules.same_observation(before, old.observe(day, 126))


def test_dependency_scope_contains_both_real_precedent_and_unchanged_source_interface():
    paths = {r["path"] for r in rules.dependencies()}
    assert set(rules.OWN).issubset(paths)
    assert "src/us_quant/legacy_defensive_rotation_expert_v1.py" in paths
    assert "src/us_quant/strategy_experts_v1.py" in paths
