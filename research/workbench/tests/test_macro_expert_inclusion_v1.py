import copy
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_strategy_expert_pool_eight_v1 import archived_experts as original

from us_quant import macro_expert_inclusion_v1 as rules
from us_quant import macro_policy_compatibility_v1 as compatible
from us_quant import ssd_macro_expert_inclusion_v1 as study
from us_quant.calendar import is_month_end, previous_session
from us_quant.config import QuantError
from us_quant.storage import read_json, utc_now
from us_quant.strategy_experts_v1 import ExpertIdentity


def examples(changed_after=None):
    existing, calendar, assets = original(changed_after=changed_after)
    target = pd.DataFrame(np.nan, index=calendar, columns=assets)
    active = pd.DatetimeIndex([day for day in calendar if is_month_end(day)])
    target.loc[active] = [0.147, 0.245, 0.294, 0.294]
    if changed_after is not None:
        target.loc[active[active > changed_after]] = [0.294, 0.294, 0.245, 0.147]
    variants = {}
    for key in compatible.PATHS:
        frame = target.copy()
        anchor = previous_session("2016-10-06" if key.startswith("10") else "2021-10-06")
        if anchor in calendar:
            frame.loc[anchor] = frame.loc[:anchor].dropna(how="all").iloc[-1]
        variants[key] = frame
    dates = calendar[calendar >= "2016-10-06"]
    values = 0.00015 + 0.0015 * np.cos(np.arange(len(dates)) / 19)
    if changed_after is not None:
        values[dates > changed_after] += 0.01
    nav = 10000 * np.cumprod(1 + values)
    book = pd.DataFrame(
        {
            "equity": nav,
            "return": values,
            "risk_free": 0.0001,
            "cash": nav * 0.02,
            "gross_exposure": 0.98,
            "cost": 0.0,
            "orders": 0,
            "turnover": 0.0,
        },
        index=dates,
    )
    macro = compatible.MacroSeedAwarePolicyOpinion(
        ExpertIdentity(rules.EXPERT_IDS[-1], rules.MECHANISMS[-1], "2026-11-04T17:06:58Z"),
        variants,
        book,
        assets,
    )
    return study.include(existing, macro), calendar, assets


def prepared():
    experts, calendar, assets = examples()
    rows = []
    targets, audits = rules.build_targets(
        experts, calendar, assets, read_json(rules.POLICY), retain=rows.append
    )
    return (
        SimpleNamespace(close=pd.DataFrame(index=calendar, columns=assets)),
        experts,
        targets,
        audits,
        rows,
    )


def test_original_eight_preserved_and_only_macro_appended_with_exact_old_laws():
    experts, _, _ = examples()
    assert tuple(e.identity.identifier for e in experts) == rules.EXPERT_IDS
    assert rules.EXPERT_IDS[:8] == rules.prior.EXPERT_IDS
    assert rules.MECHANISMS[:8] == rules.prior.MECHANISMS
    assert len(experts) == 9
    assert rules.choose_pair is rules.prior.choose_pair
    assert rules.combine is rules.prior.combine
    assert type(experts[-1]) is compatible.MacroSeedAwarePolicyOpinion


def test_append_replace_or_outside_macro_scope_is_not_the_declared_study():
    old, _, _ = original()
    experts, _, _ = examples()
    with pytest.raises(QuantError):
        study.include((*old, old[0]), experts[-1])
    with pytest.raises(QuantError):
        study.include(old, old[0])
    with pytest.raises(QuantError, match="within admitted macro"):
        rules.macro_context(experts, pd.Timestamp("2015-09-30"))


def test_nine_sources_preserve_seven_unscored_states_and_every36_pair():
    _, _, _, audits, _ = prepared()
    assert sum(a["selection"]["reason"] == "unscored_initial_equal_weights" for a in audits) == 7
    mature = [a for a in audits if a["selection"]["scores_available"]]
    assert mature[0]["decision_session"] == "2017-04-28"
    assert all(len(a["selection"]["pairs"]) == 36 for a in mature)
    assert all(a["equal_expert_weights"] == [1 / 9] * 9 for a in audits)


def test_prefix_and_future_changes_preserve_macro_inclusion_past():
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


def test_initial_capital_is_not_macro_policy_emission_or_source_rebalance():
    _, experts, _, _, rows = prepared()
    context = rows[0]["macro_policy_event_context"]
    assert context["original_policy_opinion_known_at"] == "2016-09-30"
    assert not context["native_policy_recommendation_emitted_today"]
    assert len(context["native_capital_initialization_events"]) == 2
    assert not context["observation_is_a_new_rebalance_request"]
    assert not context["seed_removed_or_retimed_to_month_end"]
    month = rules.macro_context(experts, pd.Timestamp("2016-10-31"))
    assert month["native_policy_recommendation_emitted_today"]
    assert not month["observation_is_a_new_rebalance_request"]


def test_all36_scalar_oracle_and_fullprecision_inputs_including_macro_context(tmp_path):
    data, experts, targets, audits, rows = prepared()
    proof = study.verify_targets(data, experts, targets, audits)
    assert proof["maximum_independent_target_difference"] < 1e-10
    assert proof["maximum_independent_net_score_difference"] < 1e-8
    path = tmp_path / "source-inputs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    checked = study.verify_retained_records(path, experts, targets, audits)
    assert checked["actual_input_and_resolved_record_count"] == 2 * len(audits)
    assert checked["actual_macro_role_context_checks"] == len(audits)


@pytest.mark.parametrize("kind", ["correlation", "cost", "marginal", "pair_count"])
def test_nonselected_pair_or_missing36th_record_is_not_hidden(kind):
    data, experts, targets, audits, _ = prepared()
    audits = copy.deepcopy(audits)
    active = next(a for a in audits if a["selection"]["scores_available"])
    row = next(
        p
        for p in active["selection"]["pairs"]
        if p["pair"] != active["selection"]["selected_pair"] and "selection_score_key" in p
    )
    if kind == "correlation":
        row["training_correlation"] += 0.01
    elif kind == "cost":
        row["estimated_incremental_switch_cost_fraction"] += 0.01
    elif kind == "marginal":
        row["training_marginal_variance_contributions"][0] += 0.01
    else:
        active["selection"]["pairs"].pop()
    with pytest.raises(QuantError):
        study.verify_targets(data, experts, targets, audits)


def test_macro_context_cannot_be_relabelled_a_source_order(tmp_path):
    _, experts, targets, audits, rows = prepared()
    rows[0]["macro_policy_event_context"]["observation_is_a_new_rebalance_request"] = True
    path = tmp_path / "tampered-role.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(QuantError, match="macro source opinion"):
        study.verify_retained_records(path, experts, targets, audits)


def test_actual_inputs_survive_synthetic_new_selection_error(monkeypatch):
    experts, calendar, assets = examples()
    rows = []

    def failed(*args):
        raise QuantError("synthetic score stop")

    monkeypatch.setattr(rules, "choose_pair", failed)
    with pytest.raises(QuantError, match="synthetic score"):
        rules.build_targets(experts, calendar, assets, read_json(rules.POLICY), retain=rows.append)
    assert len(rows) == 1 and rows[0]["observations"][-1]["id"] == rules.EXPERT_IDS[-1]
    assert "macro_policy_event_context" in rows[0]


@pytest.mark.parametrize(
    "key,value",
    [
        ("lookback_sessions", 252),
        ("selected_experts", 3),
        ("actual_weight_trade_band", 0.01),
        ("all_pairs_per_mature_decision", 28),
        ("macro_information_start", "2015-08-10"),
        ("macro_native_RF_claimed_exact_common", True),
    ],
)
def test_no_window_band_scope_nativeRF_or_paircount_rescue(key, value):
    policy = copy.deepcopy(read_json(rules.POLICY))
    policy[key] = value
    with pytest.raises(QuantError):
        rules.validate(policy)


def test_both_real_registrations_precede_first_new_derivation(monkeypatch, tmp_path):
    rows = []
    monkeypatch.setattr(study.lifecycle, "runtime_guard", lambda: None)
    monkeypatch.setattr(study.lifecycle, "research_job", nullcontext)
    monkeypatch.setattr(study, "inputs", lambda: (None, None, {}, None))
    monkeypatch.setattr(study, "FOLDER", tmp_path)
    monkeypatch.setattr(study, "progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        study,
        "events",
        lambda kind, name: [r for r in rows if r["kind"] == kind and r["candidate_id"] == name],
    )
    monkeypatch.setattr(
        study.base,
        "append_event",
        lambda directory, kind, name, body: rows.append(
            {"kind": kind, "candidate_id": name, "body": body, "sha256": name}
        ),
    )

    def stop(*args):
        assert [r["candidate_id"] for r in rows if r["kind"] == "registered"] == list(study.IDS)
        raise QuantError("synthetic stop after both real registrations")

    monkeypatch.setattr(study, "prepare", stop)
    with pytest.raises(QuantError, match="both real"):
        study.register()
    assert {r["candidate_id"] for r in rows if r["kind"] == "technical_failure"} == set(study.IDS)


def test_real_packet_constructor_and_complete_cohort_metadata(monkeypatch, tmp_path):
    rows = []
    frozen = {"study_id": "synthetic-only"}
    monkeypatch.setattr(study.lifecycle, "runtime_guard", lambda: None)
    monkeypatch.setattr(study.lifecycle, "research_job", nullcontext)
    monkeypatch.setattr(study, "inputs", lambda: (None, None, frozen, None))
    monkeypatch.setattr(study, "FOLDER", tmp_path)
    monkeypatch.setattr(study, "progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        study, "PACKETS", {name: tmp_path / f"{i}.json" for i, name in enumerate(study.IDS)}
    )

    def finder(kind, name):
        return [r for r in rows if r["kind"] == kind and r["candidate_id"] == name]

    monkeypatch.setattr(study, "events", finder)
    monkeypatch.setattr(study.base, "matching_events", finder)

    def append(directory, kind, name, body):
        rows.append(
            {
                "kind": kind,
                "candidate_id": name,
                "body": body,
                "at": utc_now(),
                "sha256": name + kind,
            }
        )

    monkeypatch.setattr(study.base, "append_event", append)

    def prepare(*args):
        assert len(rows) == 2
        result = {}
        for i, name in enumerate(study.IDS):
            target, proof = tmp_path / f"target{i}.csv", tmp_path / f"proof{i}.json"
            target.write_text("synthetic target")
            proof.write_text(
                json.dumps(
                    {
                        "checked_at": utc_now(),
                        "candidate_accounts_or_metrics_computed": False,
                        "original_source_signal_recomputed": False,
                    }
                )
            )
            result[name] = {"targets": study.reference(target), "proof": study.reference(proof)}
        return result

    monkeypatch.setattr(
        study,
        "reference",
        lambda p: {
            "path": str(p),
            "sha256": "synthetic-local-reference",
        },
    )
    monkeypatch.setattr(study.base, "checked_reference", lambda r: Path(r["path"]))
    monkeypatch.setattr(study, "prepare", prepare)
    monkeypatch.setattr(
        study.seven.common,
        "retain_receipt",
        lambda path, body, *args: path.write_text(json.dumps(body)),
    )
    study.register()
    for name in study.IDS:
        packet = json.loads(study.PACKETS[name].read_text())
        assert packet["entries"][0]["candidate_id"] == name
        assert packet["complete_fixed_cohort_ids"] == list(study.IDS)
        assert len(packet["cohort_registrations"]) == 2


def test_missing_other_packet_blocks_every_funded_candidate(monkeypatch):
    monkeypatch.setattr(study.base, "journal_events", lambda p: [])
    monkeypatch.setattr(
        study, "events", lambda kind, name: [{"sha256": name}] if kind == "registered" else []
    )
    monkeypatch.setattr(
        study,
        "read_json",
        lambda p: {
            "complete_fixed_cohort_ids": list(study.IDS)
            if p == study.PACKETS[study.IDS[0]]
            else [study.IDS[1]],
            "cohort_registrations": list(study.IDS),
        },
    )
    monkeypatch.setattr(study, "scoped", lambda *args: lambda *args: {"verified": True})
    with pytest.raises(QuantError, match="every declared"):
        study.cohort_publication({"study_id": "synthetic"}, "0" * 40)


def test_original_financial_namespaces_unchanged_and_dependency_closure_complete():
    old = study.financial.ID, study.financial.FOLDER, study.financial.PACKET
    context = study.scoped(study.financial.evaluate, study.CANDIDATES[1])
    assert context.__globals__["ID"] == study.IDS[1]
    assert context.__globals__["publication"] is study.cohort_publication
    assert old == (study.financial.ID, study.financial.FOLDER, study.financial.PACKET)
    paths = {r["path"] for r in study.dependencies()}
    assert set(study.OWN).issubset(paths)
    assert "src/us_quant/macro_compatible_expert_v1.py" in paths
    assert "src/us_quant/macro_policy_compatibility_v1.py" in paths
