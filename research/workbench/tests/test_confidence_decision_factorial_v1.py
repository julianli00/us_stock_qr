import hashlib
import itertools
import json
import math
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from us_quant import confidence_decision_factorial_v1 as model
from us_quant import ssd_confidence_decision_factorial_v1 as driver
from us_quant.calendar import is_month_end, next_session, sessions
from us_quant.config import QuantError
from us_quant.storage import read_json, write_json


def _hex(values):
    return tuple(float(v).hex() for v in values)


def _decode(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def snapshot(day="2021-10-29", *, masses=None, rates=None, rating=0.25, shift=0, periods=0):
    masses = tuple(masses or ("1", "2", "3", "4", "5", "6"))
    rates = tuple(rates or ("0.5",) * 6)
    confidence = [1.0] * 5 + [rating]
    opinions = [
        np.roll(np.arange(1, 15, dtype=float), shift + 2 * slot) / 105 * 0.98 for slot in range(6)
    ]
    raw = [float(m) * float(e) * c for m, e, c in zip(masses, rates, confidence, strict=True)]
    weights = [v / math.fsum(raw) for v in raw]
    target = [
        math.fsum(w * opinion[j] for w, opinion in zip(weights, opinions, strict=True))
        for j in range(14)
    ]
    return model.SavedSnapshot(
        day,
        masses,
        rates,
        periods,
        _hex(confidence),
        tuple(_hex(row) for row in opinions),
        (day,) * 6,
        _hex(weights),
        _hex(target),
    )


def pair(original=None, recent=None, *, day="2021-10-29", initial=False):
    original, recent = original or snapshot(day), recent or snapshot(day)
    reason = "initial_capital" if initial else "hold"
    return model.SavedPair(
        "5y-base",
        day,
        initial,
        original,
        recent,
        reason,
        reason,
        original.target_hex,
        recent.target_hex,
        original.target_hex if initial else None,
        recent.target_hex if initial else None,
    )


class MemoryAdapter:
    def __init__(self, cases):
        self.cases = tuple(cases)

    def __iter__(self):
        return iter(self.cases)


def test_frozen_complete_scope_not_a_financial_trial():
    policy = read_json(model.POLICY)
    model.validate(policy)
    assert policy["saved_pairs"] == 364 and policy["new_financial_trials"] == 0
    with pytest.raises(QuantError):
        model.validate({**policy, "source_CAGR_Sharpe_drawdown_or_qualification_recomputed": True})


def test_identical_snapshots_have_no_contribution():
    result = model.inspect_pair(pair())
    assert result["exact_shapley_additivity_zero_residual"]
    assert len(result["eight_algebraic_views"]) == 8
    assert all(
        _decode(v) == 0 for row in result["axis_target_contributions_exact"].values() for v in row
    )


@pytest.mark.parametrize("axis", range(3))
def test_one_axis_change_has_only_that_axis_contribution(axis):
    old = snapshot()
    options = [
        {"masses": ("7", "2", "3", "4", "5", "6")},
        {"rating": 1.0},
        {"shift": 3},
    ]
    recent = snapshot(**options[axis])
    result = model.inspect_pair(pair(old, recent))
    for i, name in enumerate(model.AXES):
        magnitude = sum(abs(_decode(v)) for v in result["axis_target_contributions_exact"][name])
        assert (magnitude > 0) is (i == axis)


def test_all_three_axes_add_exactly_over_all_orders():
    old = snapshot()
    recent = snapshot(
        masses=("9", "2", "3", "4", "5", "6"),
        rates=("0.25", "0.5", "0.375", "0.5", "0.5", "0.4"),
        rating=1.0,
        shift=3,
        periods=59,
    )
    result = model.inspect_pair(pair(old, recent))
    targets = {
        row["mask"]: tuple(_decode(v) for v in row["target_exact"])
        for row in result["eight_algebraic_views"]
    }
    for axis, name in enumerate(model.AXES):
        expected = [Fraction(0)] * 14
        for order in itertools.permutations(range(3)):
            before = 0
            for other in order:
                if other == axis:
                    after = before | (1 << axis)
                    for j in range(14):
                        expected[j] += (targets[after][j] - targets[before][j]) / 6
                    break
                before |= 1 << other
        assert tuple(expected) == tuple(
            _decode(v) for v in result["axis_target_contributions_exact"][name]
        )
    for j in range(14):
        assert (
            sum(_decode(result["axis_target_contributions_exact"][name][j]) for name in model.AXES)
            == targets[7][j] - targets[0][j]
        )
    assert result["maximum_independent_projection_difference"] <= 1e-12


def test_source_raw_mass_scaling_is_not_a_state_effect():
    old = snapshot()
    recent = replace(
        old,
        raw_mass_decimal=tuple(str(7 * int(v)) for v in old.raw_mass_decimal),
        complete_intervals=59,
    )
    result = model.inspect_pair(pair(old, recent))
    assert result["joint_mass_rate_state_changed"]
    assert _decode(result["axis_l1_exact"][model.AXES[0]]) == 0
    assert _decode(result["net_target_displacement_l1_exact"]) == 0
    assert old.raw_mass_decimal == ("1", "2", "3", "4", "5", "6")


def test_sleeping_specialist_mass_and_opinion_do_not_receive_pseudocounts():
    old = snapshot(rating=0)
    recent = snapshot(masses=("1", "2", "3", "4", "5", "99"), rating=0)
    result = model.inspect_pair(pair(old, recent))
    assert _decode(result["net_target_displacement_l1_exact"]) == 0
    assert all(
        _decode(row["expert_fractions_exact"][-1]) == 0 for row in result["eight_algebraic_views"]
    )


def test_new_source_timestamp_is_not_automatically_new_numeric_opinion():
    original = snapshot("2021-09-30")
    recent = snapshot("2021-10-05")
    result = model.inspect_pair(pair(original, recent, day="2021-10-05", initial=True))
    assert result["source_snapshot_date_changed"]
    assert result["native_opinion_clock_changed"]
    assert not result["numeric_source_opinions_changed"]
    assert _decode(result["axis_l1_exact"]["source_opinion_vintage"]) == 0


@pytest.mark.parametrize(
    "case",
    [
        "future_master",
        "future_native",
        "missing_mass",
        "bad_rate",
        "nonfinite_mass",
        "all_sleep",
        "partial_opinion",
        "overfunded_opinion",
    ],
)
def test_invalid_saved_operand_fails_without_neutral_fallback(case):
    old = snapshot()
    if case == "future_master":
        old = replace(old, as_of="2021-11-30")
    elif case == "future_native":
        old = replace(old, native_target_known_at=("2021-11-30",) * 6)
    elif case == "missing_mass":
        old = replace(old, raw_mass_decimal=("1",) * 5)
    elif case == "bad_rate":
        old = replace(old, rates_decimal=("0.6",) * 6)
    elif case == "nonfinite_mass":
        old = replace(old, raw_mass_decimal=("NaN",) * 6)
    elif case == "all_sleep":
        old = replace(old, confidence_hex=_hex([0] * 6))
    elif case == "partial_opinion":
        old = replace(old, source_targets_hex=old.source_targets_hex[:-1])
    else:
        old = replace(old, source_targets_hex=(_hex([0.1] * 14), *old.source_targets_hex[1:]))
    with pytest.raises(QuantError):
        model.inspect_pair(pair(old))


def test_retains_input_before_endpoint_failure():
    original = snapshot()
    wrong = list(original.target_hex)
    wrong[0] = (float.fromhex(wrong[0]) + 0.01).hex()
    case = replace(pair(), original_master_csv_hex=tuple(wrong))
    events = []
    with pytest.raises(QuantError, match="endpoint"):
        model.inspect(MemoryAdapter([case]), retain=events.append)
    assert len(events) == 1 and events[0]["phase"] == "input"
    assert events[0]["case"]["original_master_csv_hex"][0] == wrong[0]


def test_hold_cannot_invent_an_issued_instruction():
    case = replace(pair(), original_issued_hex=snapshot().target_hex)
    with pytest.raises(QuantError, match="unissued hold"):
        model.inspect_pair(case)


def test_duplicate_decision_is_retained_then_rejected():
    events = []
    with pytest.raises(QuantError, match="repeated or unordered"):
        model.inspect(MemoryAdapter([pair(), pair()]), retain=events.append)
    assert [r["phase"] for r in events] == ["input", "resolved", "input"]


def test_only_named_path_has_summary_and_no_executable_target_stream():
    output = model.inspect(MemoryAdapter([pair()]), retain=lambda row: None)
    assert set(output["paths"]) == {"5y-base"}
    assert output["pairs"] == 1 and output["algebraic_views"] == 8
    assert output["paths"]["5y-base"]["initial_capital"]["cases"] == 0
    assert output["paths"]["5y-base"]["initial_capital"]["mean_axis_target_L1"] == {
        key: None for key in model.AXES
    }
    assert (
        output["new_financial_trials"] == output["new_executable_recommendations_or_accounts"] == 0
    )


@pytest.fixture
def archived(tmp_path, monkeypatch):
    dates = sessions("2016-10-05", "2026-10-05")
    master_days = [dates[0], *[d for d in dates if is_month_end(d)]]
    assert len(master_days) == 121
    targets = pd.DataFrame(np.nan, index=dates, columns=model.ASSETS)
    records, audits, saved = [], [], {}

    def audit(row):
        return {
            "decision_session": row.as_of,
            "complete_learning_intervals": row.complete_intervals,
            "current_confidences": [float.fromhex(v) for v in row.confidence_hex],
            "expert_weights": [float.fromhex(v) for v in row.expert_weights_hex],
            "known_target_dates": list(row.native_target_known_at),
            "learner_state": {
                "complete_intervals": row.complete_intervals,
                "unnormalized_mass_decimal": list(row.raw_mass_decimal),
                "learning_rates_decimal": list(row.rates_decimal),
            },
        }

    for i, day in enumerate(master_days):
        value = snapshot(str(day.date()), periods=max(0, i - 1))
        saved[value.as_of] = value
        info = audit(value)
        audits.append(info)
        records.extend(
            [
                {
                    "phase": "before_decision",
                    "decision_session": value.as_of,
                    "known_target_dates": list(value.native_target_known_at),
                    "known_targets_hex": [list(r) for r in value.source_targets_hex],
                },
                {
                    "phase": "after_decision",
                    "decision_session": value.as_of,
                    "audit": info,
                    "unified_target_hex": list(value.target_hex),
                },
            ]
        )
        targets.loc[day] = [float.fromhex(v) for v in value.target_hex]
    (tmp_path / "source.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    write_json(tmp_path / "audits.json", {"decisions": audits, "periods": [{}] * 119})
    targets.to_csv(tmp_path / "targets.csv", float_format="%.17g")
    old_issued, new_issued, desired = (targets.copy() * np.nan for _ in range(3))
    old_rows, new_rows = [], []
    for i, day in enumerate(("2021-10-05", "2021-10-29", "2021-11-30")):
        old = saved[max(d for d in saved if d <= day)]
        recent = snapshot(day, masses=("1",) * 6, rating=0.5, shift=1, periods=max(0, i - 1))
        original_reason = (
            "hold" if i == 2 else ("initial_capital" if i == 0 else "observed_weight_gap")
        )
        recent_reason = "initial_capital" if i == 0 else "observed_weight_gap"
        common = {
            "signal_session": day,
            "execution_session": str(next_session(pd.Timestamp(day)).date()),
            "initial_capital": i == 0,
            "maximum_observed_target_gap": 0.1,
        }
        old_rows.append({**common, "reason": original_reason})
        new_audit = audit(recent)
        new_audit["known_source_targets_hex"] = [list(r) for r in recent.source_targets_hex]
        new_audit["source_target_known_at"] = list(recent.native_target_known_at)
        new_rows.append(
            {
                **common,
                "reason": recent_reason,
                "assessment": new_audit,
                "desired_target": [float.fromhex(v) for v in recent.target_hex],
            }
        )
        if original_reason != "hold":
            old_issued.loc[day] = [float.fromhex(v) for v in old.target_hex]
        new_issued.loc[day] = desired.loc[day] = [float.fromhex(v) for v in recent.target_hex]
    write_json(tmp_path / "old.json", {"decisions": old_rows})
    write_json(tmp_path / "new.json", {"decisions": new_rows})
    for name, frame in (
        ("old-issued.csv", old_issued),
        ("new-issued.csv", new_issued),
        ("desired.csv", desired),
    ):
        frame.to_csv(tmp_path / name, float_format="%.17g")

    def ref(name):
        return {"path": name, "sha256": hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()}

    def resolve(value):
        path = tmp_path / value["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == value["sha256"]
        return path

    monkeypatch.setattr(driver.base, "checked_reference", resolve)
    catalogue = {
        "original_global_audits": ref("audits.json"),
        "original_master_targets": ref("targets.csv"),
        "original_global_records": ref("source.jsonl"),
        "four_saved_pairs": {
            "5y-base": {
                "original_capital_decisions": ref("old.json"),
                "original_issued": ref("old-issued.csv"),
                "recent_capital_decisions": ref("new.json"),
                "recent_desired": ref("desired.csv"),
                "recent_issued": ref("new-issued.csv"),
                "formal_start": "2021-10-06",
                "formal_end": "2021-11-30",
                "formal_return_rows": len(sessions("2021-10-06", "2021-11-30")),
                "expected_capital_decisions": 3,
            }
        },
    }
    return catalogue, tmp_path, ref


def test_original_seed_and_actual_main_Adapters_cross_same_Interface(archived):
    catalogue, _, _ = archived
    old = driver.OriginalSavedAdapter(catalogue)
    recent = driver.ActualMainSavedAdapter(catalogue["four_saved_pairs"]["5y-base"])
    assert old.snapshot_at("2021-10-05").as_of == "2021-09-30"
    assert recent.snapshot_at("2021-10-05").as_of == "2021-10-05"
    cases = list(driver.PairedArchiveAdapter(catalogue, paths=("5y-base",)))
    assert len(cases) == 3
    assert cases[0].original.complete_intervals > cases[0].recent.complete_intervals
    result = model.inspect(MemoryAdapter(cases), retain=lambda row: None)
    actual = model.inspect(
        driver.PairedArchiveAdapter(catalogue, paths=("5y-base",)),
        retain=lambda row: None,
    )
    assert actual == result and actual["all_exact_additivity_zero_residual"]
    assert actual["paths"]["5y-base"]["overall"]["only_recent_requested"] == 1


def test_future_records_do_not_change_prefix(archived):
    catalogue, tmp_path, ref = archived
    before = model.inspect(
        driver.PairedArchiveAdapter(catalogue, paths=("5y-base",), last_signal="2021-10-29"),
        retain=lambda row: None,
    )
    new = read_json(tmp_path / "new.json")
    new["decisions"][-1]["assessment"]["learner_state"]["unnormalized_mass_decimal"] = ["NaN"] * 6
    write_json(tmp_path / "new.json", new)
    catalogue["four_saved_pairs"]["5y-base"]["recent_capital_decisions"] = ref("new.json")
    after = model.inspect(
        driver.PairedArchiveAdapter(catalogue, paths=("5y-base",), last_signal="2021-10-29"),
        retain=lambda row: None,
    )
    assert before == after
    with pytest.raises(QuantError):
        model.inspect(
            driver.PairedArchiveAdapter(catalogue, paths=("5y-base",)),
            retain=lambda row: None,
        )


def test_no_source_before_first_global_record_can_be_invented(archived):
    catalogue, _, _ = archived
    source = driver.OriginalSavedAdapter(catalogue)
    with pytest.raises(QuantError, match="future master opinion"):
        source.snapshot_at("2016-09-30")


def test_original_learning_and_account_entrypoints_never_called(monkeypatch):
    from us_quant import (
        cash_funded_accounting_v2,
        confidence_rated_prod_v1,
        expert_initial_monthly_execution_v1,
        main_risk_confidence_prod_v1,
        ssd_confidence_regret_scope_v1,
        ssd_shield_ticket_anatomy_v1,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("A closed learner,policy,account or diagnosis must not replay.")

    for name in ("weights", "update"):
        monkeypatch.setattr(confidence_rated_prod_v1.ConfidenceProd, name, forbidden)
    monkeypatch.setattr(confidence_rated_prod_v1, "build_targets", forbidden)
    monkeypatch.setattr(main_risk_confidence_prod_v1, "run", forbidden)
    monkeypatch.setattr(cash_funded_accounting_v2, "simulate", forbidden)
    monkeypatch.setattr(expert_initial_monthly_execution_v1, "run", forbidden)
    monkeypatch.setattr(ssd_confidence_regret_scope_v1, "analyze", forbidden)
    monkeypatch.setattr(ssd_shield_ticket_anatomy_v1, "analyze", forbidden)
    assert model.inspect(MemoryAdapter([pair()]), retain=lambda row: None)["pairs"] == 1
