import copy
import json
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_conditional_volatility_quintet_v1 import sources

from us_quant import median_equity_budget_v1 as rules
from us_quant import ssd_median_equity_budget_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import ExpertIdentity, ExpertObservation

DAY = pd.Timestamp("2017-04-28")


def target(equity, day=DAY):
    row = pd.Series(0.0, index=rules.ASSETS, name=day)
    row["SPY"], row["BIL"] = equity, 0.98 - equity
    return row


def opinions(budgets=(0.1, 0.2, 0.3, 0.4, 0.5), day=DAY):
    return [
        ExpertObservation(
            ExpertIdentity(name, mechanism, "2026-11-06T08:25:49Z"),
            day,
            target(budget, day),
            day,
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            False,
        )
        for name, mechanism, budget in zip(rules.EXPERT_IDS, rules.MECHANISMS, budgets, strict=True)
    ]


def synthetic_cache():
    experts, calendar, assets = sources()
    dates = rules.decision_sessions(calendar, read_json(rules.POLICY))
    cached = pd.DataFrame(np.nan, index=calendar, columns=assets)
    cached.loc[dates] = 0.0
    cached.loc[dates, "SOXX"] = 0.98
    return experts, calendar, assets, cached


def test_exact_policy_keeps_source_roster_and_one_new_budget():
    rules.validate(read_json(rules.POLICY))
    assert len(study.IDS) == len(rules.CANDIDATES) == 1
    assert len(rules.EXPERT_IDS) == 5
    assert rules.EQUITY == (
        "SPY",
        "MTUM",
        "VLUE",
        "QUAL",
        "USMV",
        "QQQ",
        "XLK",
        "SOXX",
        "IJR",
        "PKW",
    )
    assert read_json(rules.POLICY)["upstream_selection_state_recomputed_or_changed"] is False


def test_proportional_equity_transfer_preserves_other_targets_and_budget():
    original = target(0.8)
    original["SPY"], original["SOXX"], original["GLD"], original["BIL"] = 0.5, 0.3, 0.08, 0.1
    before = original.copy()
    result, audit = rules.cap_target(original, opinions(), DAY)
    assert result["SPY"] == pytest.approx(0.1875)
    assert result["SOXX"] == pytest.approx(0.1125)
    assert result["GLD"] == original["GLD"] == 0.08
    assert result["BIL"] == pytest.approx(0.6)
    assert result.sum() == pytest.approx(0.98)
    assert audit["median_equity_allocation"] == 0.3 and audit["cap_binding"]
    assert not audit["actual_holdings_or_drawdown_bound_claimed"]
    pd.testing.assert_series_equal(original, before)


def test_nonbinding_budget_keeps_every_binary64_coordinate():
    original = target(0.2)
    result, audit = rules.cap_target(original, opinions(), DAY)
    assert [float(v).hex() for v in result] == [float(v).hex() for v in original]
    assert audit["cap_binding"] is False


def test_zero_equity_and_zero_median_are_explicit_valid_states():
    result, audit = rules.cap_target(target(0), opinions((0, 0, 0, 0.5, 0.98)), DAY)
    assert result.equals(target(0)) and not audit["cap_binding"]
    result, audit = rules.cap_target(target(0.8), opinions((0, 0, 0, 0.5, 0.98)), DAY)
    assert result["SPY"] == 0 and result["BIL"] == 0.98 and audit["cap_binding"]


def test_budget_uses_majority_not_lowest_highest_or_score_maturity():
    observed = opinions((0.1, 0.2, 0.5, 0.9, 0.98))
    result, audit = rules.cap_target(target(0.8), observed, DAY)
    assert result["SPY"] == 0.5 and audit["median_equity_allocation"] == 0.5
    assert audit["source_score_mature"] == [False] * 5
    changed = [
        replace(o, net_returns=pd.Series([0.99]), risk_free=pd.Series([-0.5]), score_mature=True)
        for o in observed
    ]
    assert rules.cap_target(target(0.8), changed, DAY)[0].equals(result)


def test_numeric_equality_tolerance_does_not_create_spurious_target_changes():
    original = target(np.nextafter(0.3, 1))
    result, audit = rules.cap_target(original, opinions(), DAY)
    assert result.equals(original) and not audit["cap_binding"]


@pytest.mark.parametrize(
    "damage", ["missing", "future", "order", "negative", "nan", "budget", "clock"]
)
def test_invalid_or_future_source_cannot_be_dropped_or_filled(damage):
    observed = opinions()
    if damage == "missing":
        observed[0] = replace(observed[0], target=None, target_known_at=None)
    elif damage == "future":
        observed[0] = replace(observed[0], target_known_at=DAY + pd.Timedelta(days=1))
    elif damage == "order":
        observed.reverse()
    elif damage == "clock":
        observed[0] = replace(observed[0], as_of=DAY - pd.Timedelta(days=1))
    else:
        bad = observed[0].target.copy()
        bad["SPY"] = {"negative": -0.1, "nan": np.nan, "budget": 1.1}[damage]
        observed[0] = replace(observed[0], target=bad)
    with pytest.raises(QuantError):
        rules.cap_target(target(0.8), observed, DAY)


def test_foreign_upstream_date_or_coordinates_are_refused():
    with pytest.raises(QuantError):
        rules.cap_target(target(0.8, DAY - pd.Timedelta(days=1)), opinions(), DAY)
    with pytest.raises(QuantError):
        rules.cap_target(target(0.8).iloc[::-1], opinions(), DAY)


def test_independent_rational_transfer_matches_synthetic_finite_portfolios():
    rng = np.random.default_rng(8642)
    for _ in range(32):
        original = pd.Series(rng.dirichlet(np.ones(14)) * 0.98, index=rules.ASSETS, name=DAY)
        observed = [
            replace(
                o, target=pd.Series(rng.dirichlet(np.ones(14)) * 0.98, index=rules.ASSETS, name=DAY)
            )
            for o in opinions()
        ]
        result, audit = rules.cap_target(original, observed, DAY)
        expected, median = rules.independent_target(original, observed, DAY)
        assert np.max(np.abs(result - expected)) <= 1e-10
        assert audit["median_equity_allocation"] == median


def test_full_lattice_prefix_and_future_source_and_cached_target_causality():
    experts, calendar, assets, cached = synthetic_cache()
    policy = read_json(rules.POLICY)
    first, a = rules.build_targets(experts, calendar, assets, policy, cached)
    changed, _, _ = sources(changed_after=DAY)
    future = cached.copy()
    later = future.dropna().index[future.dropna().index > DAY]
    future.loc[later, ["SOXX", "SPY"]] = [0.0, 0.98]
    second, b = rules.build_targets(changed, calendar, assets, policy, future)
    early, c = rules.build_targets(
        experts, calendar[calendar <= DAY], assets, policy, cached.loc[:DAY]
    )
    assert first[rules.ID].index.equals(calendar)
    assert first[rules.ID].loc[:DAY].equals(second[rules.ID].loc[:DAY])
    assert first[rules.ID].loc[:DAY].equals(early[rules.ID])
    assert c == [v for v in a if v["decision_session"] <= str(DAY.date())]
    assert c == [v for v in b if v["decision_session"] <= str(DAY.date())]


def test_all_inputs_resolutions_and_independent_budget_checked(tmp_path):
    experts, calendar, assets, cached = synthetic_cache()
    rows = []
    targets, audits = rules.build_targets(
        experts, calendar, assets, read_json(rules.POLICY), cached, retain=rows.append
    )
    data = SimpleNamespace(close=pd.DataFrame(index=calendar, columns=assets))
    proof = study.verify_targets(data, experts, targets, audits, cached)
    assert proof["maximum_independent_target_difference"] <= 1e-10
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert study.verify_retained_records(path, experts, targets, audits, cached)[
        "actual_input_and_resolved_record_count"
    ] == 2 * len(audits)
    damaged = copy.deepcopy(audits)
    damaged[-1]["median_equity_allocation"] += 0.01
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, targets, damaged, cached)
    rows[0]["observations"][0]["target_known_at"] = "2099-01-01"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(QuantError):
        study.verify_retained_records(path, experts, targets, audits, cached)


@pytest.mark.parametrize(
    "key,value",
    [
        ("budget_operator", "lower_quantile"),
        ("removed_equity_destination", "GLD"),
        ("equity_symbols", ["SPY"]),
        ("upstream_selection_state_recomputed_or_changed", True),
        ("source_score_maturity_required_for_budget", True),
        ("actual_holdings_or_drawdown_bound_claimed", True),
        ("training_sessions", 63),
        ("actual_weight_trade_band", 0.05),
        ("switch_cost_bps_per_side", 0.0),
        ("new_source_RF_views", 1),
        ("global_pool_changed", True),
    ],
)
def test_no_budget_parameter_source_cost_or_execution_waiver(key, value):
    with pytest.raises(QuantError):
        rules.validate({**read_json(rules.POLICY), key: value})


def test_single_registration_precedes_budget_and_cannot_be_replayed(monkeypatch):
    records, packets, prepared = [], [], []

    @contextmanager
    def job():
        yield

    def append(journal, kind, name, body):
        records.append(
            {"kind": kind, "candidate_id": name, "body": body, "sha256": str(len(records) + 1)}
        )

    def prepare(data, experts, frozen):
        assert [r["kind"] for r in records] == ["registered"]
        prepared.append(True)
        return {rules.ID: {"targets": {}, "proof": {}}}

    monkeypatch.setattr(study.lifecycle, "runtime_guard", lambda: None)
    monkeypatch.setattr(study.lifecycle, "research_job", job)
    monkeypatch.setattr(study, "inputs", lambda: (None, None, {"study_id": "toy"}, None))
    monkeypatch.setattr(
        study,
        "events",
        lambda k, n: [r for r in records if r["kind"] == k and r["candidate_id"] == n],
    )
    monkeypatch.setattr(study.base, "append_event", append)
    monkeypatch.setattr(study, "prepare", prepare)
    monkeypatch.setattr(study.seven.common, "packet_entry", lambda c, f: {"candidate_id": c["id"]})
    monkeypatch.setattr(
        study.seven.common, "retain_receipt", lambda p, b, volatile: packets.append(b)
    )
    monkeypatch.setattr(study, "reference", lambda p: {"path": p.name, "sha256": "toy"})
    monkeypatch.setattr(study, "progress", lambda *a, **k: None)
    study.register()
    assert [r["kind"] for r in records] == ["registered", "target_proof"]
    assert packets[0]["complete_fixed_cohort_ids"] == [rules.ID]
    assert packets[0]["cohort_registrations"] == ["1"]
    assert packets[0]["candidate_accounts_or_metrics_computed"] is False
    with pytest.raises(QuantError):
        study.register()
    assert prepared == [True]


@pytest.mark.parametrize("cohort", [[rules.ID], [], ["foreign"]])
def test_complete_actual_singleton_packet_precedes_publication(monkeypatch, cohort):
    verified = []
    monkeypatch.setattr(study.base, "journal_events", lambda p: [])
    monkeypatch.setattr(
        study, "events", lambda k, n: [{"sha256": "registered"}] if k == "registered" else []
    )
    monkeypatch.setattr(
        study,
        "read_json",
        lambda p: {
            "complete_fixed_cohort_ids": cohort,
            "cohort_registrations": ["registered"],
        },
    )
    monkeypatch.setattr(
        study,
        "scoped",
        lambda f, c: (lambda frozen, commit: verified.append(c["id"]) or {"commit": commit}),
    )
    if cohort == [rules.ID]:
        result = study.cohort_publication({"study_id": "toy"}, "commit")
        assert result["all_fixed_cohort_packets_verified_before_accounts"]
        assert verified == [rules.ID]
    else:
        with pytest.raises(QuantError):
            study.cohort_publication({"study_id": "toy"}, "commit")
        assert not verified
