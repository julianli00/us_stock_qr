import copy
import json
from dataclasses import replace
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import test_growth_macro_defense_hedge_v1 as prior_tests
from test_strategy_expert_pool_eight_v1 import archived_experts

from us_quant import commitment_macro_defense_hedge_v1 as rules
from us_quant import ssd_commitment_macro_defense_hedge_v1 as study
from us_quant.config import QuantError
from us_quant.retained_pair_equal_allocation_v1 import ASSETS
from us_quant.storage import read_json


def sources(changed_after=None):
    eight, calendar, toy_assets = archived_experts(changed_after=changed_after)
    mapping = dict(zip(toy_assets, ASSETS[: len(toy_assets)], strict=True))
    selected = tuple(
        prior_tests.MappedSyntheticExpert(e, mapping) for e in study.select_sources(eight)
    )
    return selected, calendar, ASSETS


def test_exact_direct_role_and_unchanged_pair_implementation():
    experts, _, _ = sources()
    assert tuple(e.identity.identifier for e in experts) == rules.EXPERT_IDS
    assert tuple(e.identity.mechanism_group for e in experts) == rules.MECHANISMS
    assert rules.SLOTS == (4, 5, 7, 0)
    assert rules.EXPERT_IDS[-1] == "equity_first_month_end_commitment_v1"
    assert rules.choose_pair is study.template.rules.choose_pair
    assert rules.combine is study.template.rules.combine
    observed = [e.observe(pd.Timestamp("2017-04-28"), 126) for e in experts]
    weights, audit = rules.choose_pair(observed, np.full(4, 0.25))
    assert len(audit["pairs"]) == 6 and (weights > 0).sum() == 2
    assert all(len(p["expert_weights"]) == 4 for p in audit["pairs"])


def test_initial_quarters_all_dates_and_unscored_origins():
    experts, calendar, assets = sources()
    targets, audits = rules.build_targets(experts, calendar, assets, read_json(rules.POLICY))
    assert audits[0]["decision_session"] == "2016-10-05"
    assert audits[0]["equal_expert_weights"] == [0.25] * 4
    assert sum(not a["selection"]["scores_available"] for a in audits) == 7
    assert all(t.index.equals(calendar) for t in targets.values())
    assert set(targets) == set(study.IDS)


def test_future_source_values_do_not_change_prior_allocations():
    cutoff = pd.Timestamp("2017-04-28")
    experts, calendar, assets = sources()
    changed, _, _ = sources(changed_after=cutoff)
    first, a = rules.build_targets(experts, calendar, assets, read_json(rules.POLICY))
    second, b = rules.build_targets(changed, calendar, assets, read_json(rules.POLICY))
    early, c = rules.build_targets(
        experts, calendar[calendar <= cutoff], assets, read_json(rules.POLICY)
    )
    for name in study.IDS:
        pd.testing.assert_frame_equal(first[name].loc[:cutoff], second[name].loc[:cutoff])
        pd.testing.assert_frame_equal(first[name].loc[:cutoff], early[name])
    assert c == [x for x in a if x["decision_session"] <= str(cutoff.date())]
    assert c == [x for x in b if x["decision_session"] <= str(cutoff.date())]


def test_all_pair_scores_and_retained_records_checked(tmp_path):
    experts, calendar, assets = sources()
    rows = []
    targets, audits = rules.build_targets(
        experts, calendar, assets, read_json(rules.POLICY), retain=rows.append
    )
    data = SimpleNamespace(close=pd.DataFrame(index=calendar, columns=assets))
    verified = study.verify_targets(data, experts, targets, audits)
    assert verified["maximum_independent_target_difference"] <= 1e-10
    path = tmp_path / "source-records.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    retained = study.verify_retained_records(path, experts, targets, audits)
    assert retained["actual_input_and_resolved_record_count"] == 2 * len(audits)
    damaged = copy.deepcopy(audits)
    scored = next(x for x in damaged if x["selection"]["scores_available"])
    unused = next(
        p for p in scored["selection"]["pairs"] if p["pair"] != scored["selection"]["selected_pair"]
    )
    unused["validation_net_risk_score"] += 0.01
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, targets, damaged)


def test_missing_direct_opinion_is_not_dropped_or_cash_filled():
    experts, calendar, assets = sources()
    direct = experts[-1]
    missing = SimpleNamespace(
        identity=direct.identity,
        observe=lambda day, n: replace(direct.observe(day, n), target=None, target_known_at=None),
    )
    with pytest.raises(QuantError):
        rules.build_targets((*experts[:-1], missing), calendar, assets, read_json(rules.POLICY))


@pytest.mark.parametrize(
    "key,value",
    [
        ("original_source_slots", [4, 5, 7, 1]),
        ("direct_commitment_source_slot", 1),
        ("nested_growth_controller_included", True),
        ("current_opinion_risk_forecast_promoted", True),
        ("ranking_or_weighting_parameters_tuned", True),
        ("lookback_sessions", 252),
        ("training_sessions", 63),
        ("validation_sessions", 21),
        ("selected_experts", 3),
        ("mature_pair_count", 3),
        ("actual_weight_trade_band", 0.05),
        ("switch_cost_bps_per_side", 0.0),
        ("new_source_RF_views", 1),
        ("all_source_native_RF_claimed_equal_without_view", True),
    ],
)
def test_no_source_window_cost_or_risk_forecast_rescue(key, value):
    with pytest.raises(QuantError):
        rules.validate({**read_json(rules.POLICY), key: value})


def test_both_real_registrations_precede_new_targets(monkeypatch):
    function = prior_tests.test_both_real_registrations_precede_either_new_combination
    FunctionType(
        function.__code__,
        {**function.__globals__, "study": study, "rules": rules},
        argdefs=function.__defaults__,
    )(monkeypatch)
