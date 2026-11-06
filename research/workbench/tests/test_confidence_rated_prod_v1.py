import copy
import json
import math
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from us_quant import confidence_rated_prod_v1 as rules
from us_quant import ssd_commitment_research_v1 as packet
from us_quant import ssd_confidence_rated_prod_v1 as program
from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.retained_pair_equal_allocation_v1 import ASSETS
from us_quant.semiconductor_portfolio_protection import protected_target
from us_quant.storage import digest_json, read_json
from us_quant.strategy_experts_v1 import ExpertIdentity, ExpertObservation


class SyntheticExpert:
    def __init__(self, number, calendar):
        self.number, self.calendar = number, calendar
        self.identity = ExpertIdentity(
            rules.EXPERT_IDS[number], rules.MECHANISMS[number], "2026-11-06T00:00:00+00:00"
        )

    def observe(self, day, lookback):
        day = pd.Timestamp(day)
        index = self.calendar[(self.calendar > pd.Timestamp("2016-10-05")) & (self.calendar <= day)]
        index = index[-lookback:]
        positions = self.calendar.get_indexer(index)
        returns = pd.Series(
            0.0001 * self.number + ((positions % 13) - 6) * 0.00001 * (self.number + 1),
            index=index,
            name="return",
        )
        target = pd.Series(0.0, index=ASSETS)
        target.iloc[self.number] = 0.98
        return ExpertObservation(
            self.identity, day, target, day, returns, returns * 0, len(returns) == lookback
        )


def sources(end="2017-03-31"):
    calendar = sessions("2016-10-05", end)
    experts = tuple(SyntheticExpert(i, calendar) for i in range(6))
    dates = rules.decision_sessions(calendar, read_json(rules.POLICY))
    cached = []
    for i, day in enumerate(dates[2:], start=2):
        observed = rules.observations(experts, day)
        logs, _ = rules.prior.period_logs(observed, dates[i - 1], day)
        cached.append(
            {
                "start_after": str(dates[i - 1].date()),
                "end": str(day.date()),
                "canonical_log_relatives": logs.tolist(),
                "input_loss_labels": rules.prior.bounded_losses(logs).tolist(),
            }
        )
    equity = pd.Series(9995.0 - np.arange(len(calendar) - 1) * 0.1, index=calendar[1:])
    return experts, cached, calendar, rules.OwnShadowConfidenceArchive(equity)


def build(experts, cached, calendar, archive, **kwargs):
    return rules.build_targets(
        experts, calendar, ASSETS, read_json(rules.POLICY), cached, archive, **kwargs
    )


def test_uniform_raw_prior_is_normalized_by_confidence_not_a_capital_floor():
    state = rules.ConfidenceProd()
    np.testing.assert_array_equal(state.weights([1] * 6), np.full(6, 1 / 6))
    np.testing.assert_allclose(state.weights([1] * 5 + [0]), [0.2] * 5 + [0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        state.weights([1] * 5 + [0.5]), [2 / 11] * 5 + [1 / 11], rtol=0, atol=1e-12
    )


def test_first_update_uses_prequential_loss_and_paper_positive_factor():
    state = rules.ConfidenceProd()
    old = state.mass.copy()
    audit = state.update([0, 1, 0.25, 0.5, 0.75, 1], [1] * 5 + [0])
    assert float(Decimal(audit["played_loss_decimal"])) == pytest.approx(0.5, abs=1e-12)
    expected = [0.5, -0.5, 0.25, 0, -0.25, 0]
    np.testing.assert_allclose(
        [float(v) for v in audit["confidence_excess_decimal"]], expected, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        [float(v) for v in state.mass],
        [float(w) * (1 + 0.5 * r) for w, r in zip(old, expected, strict=True)],
        rtol=0,
        atol=1e-12,
    )
    assert state.mass[-1] == old[-1] and state.squares[-1] == state.regrets[-1] == 0


def test_neutral_round_advances_maturity_without_cooling_or_roundtrip_mass_change():
    state = rules.ConfidenceProd()
    state.update([0, 1, 0.2, 0.4, 0.6, 0.8], [1] * 6)
    before = state.snapshot()
    weights = state.weights([1] * 6)
    state.update([0] * 6, [1] * 6)
    after = state.snapshot()
    assert {k: v for k, v in before.items() if k != "complete_intervals"} == {
        k: v for k, v in after.items() if k != "complete_intervals"
    }
    assert after["complete_intervals"] == before["complete_intervals"] + 1
    np.testing.assert_array_equal(state.weights([1] * 6), weights)


def test_sleeping_expert_state_is_preserved_and_can_reactivate():
    state = rules.ConfidenceProd()
    original = state.mass[-1]
    for _ in range(20):
        state.update([0, 1, 0.2, 0.4, 0.6, 0.8], [1] * 5 + [0])
    assert state.mass[-1] == original
    assert state.squares[-1] == state.regrets[-1] == 0
    assert state.weights([1] * 5 + [0])[-1] == 0
    assert state.weights([1] * 6)[-1] > 0


def test_all119_updates_match_independent_modified_loss_reduction():
    state, history, actual = rules.ConfidenceProd(), [], []
    for i in range(119):
        labels = ((np.arange(6) + i) % 6) / 5
        confidence = [1] * 5 + [(i % 5) / 4]
        history.append((labels.tolist(), confidence))
        actual.append(state.update(labels, confidence))
    oracle = rules.independent_trace(history)
    for original, independent in zip(actual, oracle, strict=True):
        np.testing.assert_allclose(
            original["prequential_capital_weights_before_assimilation"],
            independent["weights_before"],
            rtol=0,
            atol=1e-12,
        )
        for key, other in (
            ("unnormalized_mass_decimal", "mass_after_decimal"),
            ("squared_confidence_excess_decimal", "squares_after_decimal"),
            ("confidence_regret_decimal", "regrets_after_decimal"),
        ):
            np.testing.assert_allclose(
                [float(v) for v in original["state_after"][key]],
                [float(v) for v in independent[other]],
                rtol=0,
                atol=1e-12,
            )
        assert independent["confidence_reduction_error"] <= 1e-80
    assert not math.isclose(sum(float(v) for v in state.mass), 1, rel_tol=0, abs_tol=1e-12)
    assert min(float(v) for v in state.snapshot()["learning_rates_decimal"]) < 0.5
    with pytest.raises(QuantError):
        state.update([0] * 6, [1] * 6)


@pytest.mark.parametrize(
    "values,confidence",
    [
        ([0, 1], [1] * 6),
        ([True] * 6, [1] * 6),
        ([0, 1, np.nan, 0, 0, 0], [1] * 6),
        ([0, 1, -0.01, 0, 0, 0], [1] * 6),
        ([0, 1, 1.01, 0, 0, 0], [1] * 6),
        ([0] * 6, [0] * 6),
        ([0] * 6, [True] * 6),
        ([0] * 6, [1, 1, 1, 1, 1, np.inf]),
        ([0] * 6, [1, 1, 1, 1, 1, -0.1]),
    ],
)
def test_invalid_losses_or_confidence_are_errors_not_safe_defaults(values, confidence):
    with pytest.raises(QuantError):
        rules.ConfidenceProd().update(values, confidence)


@pytest.mark.parametrize(
    "field,value",
    [
        ("periods", True),
        ("periods", -1),
        ("mass", [Decimal("-1")] * 6),
        ("squares", [Decimal("NaN")] * 6),
        ("squares", [Decimal(1)] * 6),
        ("regrets", [Decimal(1)] * 6),
    ],
)
def test_invalid_state_cannot_reset_to_equal_or_clip(field, value):
    state = rules.ConfidenceProd()
    setattr(state, field, value)
    with pytest.raises(QuantError):
        state.weights([1] * 6)


@pytest.mark.parametrize(
    "nav,peak", [(10000, 10000), (9500, 10000), (8500, 10000), (8000, 10000), (19000, 20000)]
)
def test_confidence_uses_original_expert_scale_not_a_new_main_floor(nav, peak):
    baseline = pd.Series(0.0, index=ASSETS)
    baseline["SPY"] = 0.98
    protected = protected_target(baseline, nav, peak)
    actual = rules.confidences(
        {
            "own_shadow_nav_hex": float(nav).hex(),
            "own_shadow_peak_hex": float(peak).hex(),
            "main_wealth_or_future_shadow_used": False,
        }
    )
    assert actual[-1] == pytest.approx(1 - protected["SPY"] / 0.98, abs=1e-12)
    np.testing.assert_array_equal(actual[:5], [1] * 5)


def test_confidence_archive_uses_only_existing_prefix_and_real_initial_seed():
    _, _, calendar, archive = sources()
    first = archive.observe(calendar[0])
    assert first["initial_source_wealth_seed"] is True and first["past_saved_return_rows"] == 0
    assert rules.confidences(first)[-1] == 0
    cutoff = calendar[45]
    changed = archive.equity.copy()
    changed.loc[changed.index > cutoff] = 999999
    assert rules.OwnShadowConfidenceArchive(changed).observe(cutoff) == archive.observe(cutoff)
    early = rules.OwnShadowConfidenceArchive(archive.equity.loc[:cutoff])
    assert early.observe(cutoff) == archive.observe(cutoff)
    with pytest.raises(QuantError):
        early.observe(calendar[-1])
    with pytest.raises(QuantError):
        rules.OwnShadowConfidenceArchive(archive.equity.iloc[::2])


def test_prior_confidence_not_end_month_confidence_weights_past_loss():
    experts, cached, calendar, archive = sources()
    _, audits, updates = build(experts, cached, calendar, archive)
    assert updates[0]["confidence_held_from_prior_origin"] == audits[1]["current_confidences"]
    assert updates[0]["confidence_held_from_prior_origin"] != audits[2]["current_confidences"]
    assert [a["score_mature"] for a in audits[:3]] == [False, False, True]
    assert updates[0]["start_after"] == "2016-10-31" and updates[0]["end"] == "2016-11-30"


def test_complete_month_builder_retains_operands_and_future_prefixes():
    experts, cached, calendar, archive = sources()
    records = []
    full, audits, updates = build(experts, cached, calendar, archive, retain=records.append)
    assert len(records) == 2 * len(audits)
    cutoff = pd.Timestamp("2017-01-31")
    altered = copy.deepcopy(cached)
    for row in altered:
        if row["end"] > str(cutoff.date()):
            row["input_loss_labels"] = [np.nan] * 6
    changed = archive.equity.copy()
    changed.loc[changed.index > cutoff] *= 100
    early, before, earlier_updates = build(
        experts, altered, calendar[calendar <= cutoff], rules.OwnShadowConfidenceArchive(changed)
    )
    pd.testing.assert_frame_equal(full.loc[:cutoff], early)
    assert before == [a for a in audits if a["decision_session"] <= str(cutoff.date())]
    assert earlier_updates == [a for a in updates if a["end"] <= str(cutoff.date())]
    assert records[2]["confidence_wealth_input"] == archive.observe(pd.Timestamp("2016-10-31"))
    assert len(records[2]["known_targets_hex"]) == 6


def test_full121_synthetic_decisions_exercise_independent_driver_verification():
    experts, cached, calendar, archive = sources("2026-10-05")
    target, audits, updates = build(experts, cached, calendar, archive)
    data = SimpleNamespace(close=pd.DataFrame(index=calendar, columns=ASSETS))
    result = program.verify_targets(data, experts, archive, target, audits, updates)
    assert result["maximum_independent_expert_weight_difference"] <= 1e-12
    assert result["maximum_independent_decimal_state_difference"] <= 1e-12
    assert result["complete_mature_learning_intervals"] == 119
    damaged = copy.deepcopy(audits)
    damaged[-1]["expert_weights"][0] += 0.01
    with pytest.raises(QuantError):
        program.verify_targets(data, experts, archive, target, damaged, updates)


@pytest.mark.parametrize("kind", ["missing", "label", "month", "RF", "opinion"])
def test_missing_or_changed_native_source_inputs_cannot_be_filled(kind):
    experts, cached, calendar, archive = sources()
    damaged = copy.deepcopy(cached)
    if kind == "missing":
        damaged = damaged[1:]
    elif kind == "label":
        damaged[0]["input_loss_labels"][0] += 0.001
    elif kind == "month":
        damaged[0]["start_after"] = "2016-10-05"
    else:
        original = experts[-1]

        def observe(day, lookback):
            old = original.observe(day, lookback)
            return (
                replace(old, risk_free=old.risk_free + 0.001)
                if kind == "RF"
                else replace(old, target=None, target_known_at=None)
            )

        experts = (*experts[:-1], SimpleNamespace(identity=original.identity, observe=observe))
    with pytest.raises(QuantError):
        build(experts, damaged, calendar, archive)


def test_canonical_packet_entry_accepts_new_proof_interface(monkeypatch, tmp_path):
    frozen, candidate = {"study_id": "synthetic_confidence_Prod"}, rules.CANDIDATES[0]
    proof = program.target_proof(frozen, {"decision_count": 121})
    proof["checked_at"] = "2026-11-01T00:00:01+00:00"
    proof_path = tmp_path / "synthetic-proof.json"
    proof_path.write_text(json.dumps(proof))
    target_path = tmp_path / "synthetic-target.csv"
    target_path.write_text("synthetic_only\n")
    target_ref, proof_ref = {"path": "target", "sha256": "1"}, {"path": "proof", "sha256": "2"}
    registered = {
        "sha256": "registration",
        "at": "2026-11-01T00:00:00+00:00",
        "body": {"candidate": candidate, "frozen": frozen},
    }
    event = {
        "sha256": "proof_event",
        "at": "2026-11-01T00:00:02+00:00",
        "body": {
            "registered_event_sha256": "registration",
            "targets": target_ref,
            "proof": proof_ref,
        },
    }
    monkeypatch.setattr(
        packet.base,
        "matching_events",
        lambda kind, _: [registered if kind == "registered" else event],
    )
    monkeypatch.setattr(
        packet.base,
        "checked_reference",
        lambda ref: target_path if ref == target_ref else proof_path,
    )
    saved = packet.packet_entry(candidate, frozen)
    assert saved["candidate_id"] == rules.ID
    assert proof["frozen_specification_sha256"] == digest_json(frozen)
    assert proof["candidate_accounts_or_metrics_computed"] is False
    assert proof["original_source_signal_recomputed"] is False
    assert program.source is program.seven


@pytest.mark.parametrize(
    "field,value",
    [
        ("confidence_clock", "end_month_lookahead"),
        ("raw_mass_renormalized_after_update", True),
        ("actual_weight_trade_band", 0.01),
        ("new_source_RF_views", 1),
        ("learning_resets_for_five_year_account", True),
        ("confidence_is_probability_or_source_admission", True),
        ("candidate_count_in_this_study", True),
    ],
)
def test_frozen_confidence_and_financial_contract_cannot_be_retuned(field, value):
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    policy[field] = value
    with pytest.raises(QuantError):
        rules.validate(policy)
