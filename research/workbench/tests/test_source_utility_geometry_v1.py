from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from us_quant import source_utility_geometry_v1 as rules
from us_quant import ssd_source_utility_geometry_v1 as study
from us_quant.config import QuantError
from us_quant.storage import file_digest, read_json

LOGS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625)
LOSSES = (0.625, 0.5, 0.375, 0.25, 0.125, 0.0)
ENDS = ("2016-11-30", "2016-12-30", "2017-01-31")


def _hex(values):
    return tuple(float(v).hex() for v in values)


def _month(index=0, logs=LOGS, losses=LOSSES):
    return rules.UtilityMonth(
        "2016-10-31" if index == 0 else ENDS[index - 1],
        ENDS[index],
        _hex(logs),
        _hex(losses),
    )


def _endpoint(weights=(1 / 6,) * 6):
    opinions = tuple(tuple(0.98 if j == i else 0.0 for j in range(rules.A)) for i in range(rules.N))
    target = tuple(sum(weights[i] * opinions[i][j] for i in range(rules.N)) for j in range(rules.A))
    return rules.StoredEndpoint(
        tuple(_hex(row) for row in opinions),
        ("2016-09-30",) * rules.N,
        _hex(weights),
        _hex(target),
    )


class MemoryAdapter:
    def __init__(self, snapshots):
        self.snapshots = {s.origin: s for s in snapshots}
        self.origins = tuple(self.snapshots)

    def observe(self, origin):
        return self.snapshots[origin]


def _feed(months=None, rates=None):
    months = (_month(),) if months is None else months
    rates = ("2",) * len(months) if rates is None else rates
    old, new = _endpoint(), _endpoint((0.5, 0.1, 0.1, 0.1, 0.1, 0.1))
    snapshots = [
        rules.UtilitySnapshot(d, (), "analytic_infinity", old, new)
        for d in ("2016-10-05", "2016-10-31")
    ]
    snapshots.extend(
        rules.UtilitySnapshot(m.end, months[: i + 1], rates[i], old, new)
        for i, m in enumerate(months)
    )
    return MemoryAdapter(snapshots)


def _inspect(feed):
    records = []
    result = rules.inspect(feed, retain=records.append)
    return result, records


def _cells(records):
    return [r for r in records if r["phase"] == "after_cell"]


def _origins(records):
    return [r for r in records if r["phase"] == "after_origin"]


def _archive(tmp_path, feed):
    snapshots = tuple(feed.snapshots.values())
    months = snapshots[-1].months
    teacher = tmp_path / "teacher.json"
    teacher.write_text(
        json.dumps(
            {
                "decisions": [{"decision_session": s.origin} for s in snapshots],
                "periods": [
                    {
                        "start_after": m.start_after,
                        "end": m.end,
                        "canonical_log_relatives": [float.fromhex(v) for v in m.log_hex],
                        "input_loss_labels": [float.fromhex(v) for v in m.loss_hex],
                    }
                    for m in months
                ],
            }
        )
    )
    files = {}
    for name, attribute, field in (
        ("wealth", "original", "original_log_relatives_hex"),
        ("adaptive_loss", "adaptive", "original_loss_labels_hex"),
    ):
        path = tmp_path / (name + ".jsonl")
        with path.open("w") as stream:
            for s in snapshots:
                endpoint = getattr(s, attribute)
                month = s.months[-1] if s.months else None
                before = {
                    "phase": "before_decision",
                    "decision_session": s.origin,
                    "source_targets_hex": endpoint.source_opinions_hex,
                    "source_native_known_at": endpoint.source_known_at,
                    "matured_source_interval": None
                    if month is None
                    else {
                        "start_after": month.start_after,
                        "end": month.end,
                        field: month.log_hex
                        if field == "original_log_relatives_hex"
                        else month.loss_hex,
                    },
                }
                after = {
                    "phase": "after_decision",
                    "decision_session": s.origin,
                    "unified_target_hex": endpoint.target_hex,
                    "audit": {
                        "decision_session": s.origin,
                        "complete_source_intervals": len(s.months),
                        "source_native_known_at": endpoint.source_known_at,
                        "expert_weights": [float.fromhex(v) for v in endpoint.weight_hex],
                        "state": {"learning_rate_decimal": s.rate_decimal}
                        if attribute == "adaptive"
                        else {"closed_old_mass_is_not_an_input": True},
                    },
                }
                stream.write(json.dumps(before) + "\n" + json.dumps(after) + "\n")
        files[name] = {"path": str(path), "sha256": file_digest(path)}
    catalogue = {
        "source_expert_ids": list(rules.EXPERT_IDS),
        "teacher_periods": {"path": str(teacher), "sha256": file_digest(teacher)},
        "wealth_records": files["wealth"],
        "adaptive_loss_records": files["adaptive_loss"],
    }

    def reader(ref):
        path = Path(ref["path"])
        assert file_digest(path) == ref["sha256"]
        return path

    return catalogue, reader


def test_exact_parallel_orthogonal_and_energy():
    result, records = _inspect(_feed())
    p = _cells(records)[0]["projection"]
    assert p["slope"] == 1 and not any(p["residual"])
    assert p["parallel"] == p["natural"] == p["displacement"]
    assert p["parallel_energy"] + p["orthogonal_energy"] == p["total_energy"]
    assert p["orthogonal_energy"] == 0
    assert result["utility_energy"]["additivity_exact"]
    assert result["past_month_utility_views"] == 1
    assert len(records) == 2 * (3 + 1)
    assert Decimal(result["maximum_independent_decimal_error"]) <= Decimal("1e-12")


def test_nonaffine_direction_is_retained_not_forced_to_pass():
    month = _month(losses=(1.0, 0.5, 0.5, 0.125, 0.0, 0.0))
    result, records = _inspect(_feed((month,)))
    p = _cells(records)[0]["projection"]
    assert p["orthogonal_energy"] > 0
    assert (
        tuple(a + b for a, b in zip(p["parallel"], p["orthogonal"], strict=True))
        == p["displacement"]
    )
    assert p["parallel_energy"] + p["orthogonal_energy"] == p["total_energy"]
    assert result["monthly_geometry"]["affine_ray_within_original_tolerance_months"] == 0
    assert not result["utility_energy"]["is_main_risk_or_wealth_attribution"]


def test_reversed_ranks_are_observed_not_an_acceptance_waiver():
    result, _ = _inspect(_feed((_month(losses=LOGS),)))
    assert result["monthly_geometry"]["strict_rank_reversals"] == 15
    assert result["monthly_geometry"]["unscaled_ray_slopes"]["mean"] == -1
    assert result["new_financial_trials"] == 0


def test_common_offset_is_quotiented_without_rewriting_source():
    _, original = _inspect(_feed())
    shifted = _month(logs=tuple(v + 1 for v in LOGS), losses=tuple(v + 0.125 for v in LOSSES))
    _, recent = _inspect(_feed((shifted,)))
    assert _cells(original)[0]["projection"] == _cells(recent)[0]["projection"]
    original_input = next(r for r in original if r["phase"] == "before_cell")
    recent_input = next(r for r in recent if r["phase"] == "before_cell")
    assert original_input["month"] != recent_input["month"]


def test_expert_coordinate_permutation_preserves_geometric_scalar_identity():
    order = (5, 3, 1, 4, 0, 2)
    old, _ = _inspect(_feed())
    new, _ = _inspect(
        _feed((_month(logs=tuple(LOGS[i] for i in order), losses=tuple(LOSSES[i] for i in order)),))
    )
    assert old["utility_energy"] == new["utility_energy"]
    assert old["monthly_geometry"] == new["monthly_geometry"]


def test_common_rate_cancels_magnitude_not_displacement():
    months = (_month(), _month(1, logs=tuple(v * 2 for v in LOGS)))
    first, a = _inspect(_feed(months, ("2", "3")))
    second, b = _inspect(_feed(months, ("7", "11")))
    assert (
        _origins(a)[-1]["bounded_history_magnitude_shares"]
        == _origins(b)[-1]["bounded_history_magnitude_shares"]
    )
    assert (
        _origins(a)[-1]["history_share_L1_distance"] == _origins(b)[-1]["history_share_L1_distance"]
    )
    assert float(_origins(a)[-1]["history_share_L1_distance"]) == pytest.approx(1 / 3)
    assert (
        first["history_utility_magnitude_shares"]["common_positive_rate_cancellations_exact"] == 2
    )
    assert not first["history_utility_magnitude_shares"]["source_posterior_is_rate_invariant"]
    assert first["utility_energy"] != second["utility_energy"]


def test_tiny_spread_has_no_floor_or_neutral_fallback():
    result, _ = _inspect(_feed((_month(logs=tuple(v * 2**-40 for v in LOGS)),)))
    assert result["monthly_geometry"]["natural_flat_months"] == 0
    assert result["monthly_geometry"]["unscaled_ray_slopes"]["maximum"] == 2**40


def test_analytic_infinite_neutral_history_has_explicit_undefined_shares():
    result, records = _inspect(
        _feed((_month(logs=(0.125,) * 6, losses=(0.0,) * 6),), ("analytic_infinity",))
    )
    assert result["analytic_infinite_origins"] == 3
    assert result["untrained_origins"] == 2
    assert result["utility_energy"]["total"] == 0
    assert result["utility_energy"]["orthogonal_energy_fraction"] is None
    assert _origins(records)[-1]["natural_history_magnitude_shares"] is None
    assert _origins(records)[-1]["history_share_status"] == "undefined_zero_magnitude"


def test_constant_loss_collapse_can_be_described_against_nonflat_natural_utility():
    result, records = _inspect(_feed((_month(losses=(0.0,) * 6),), ("analytic_infinity",)))
    p = _cells(records)[0]["projection"]
    assert p["parallel"] == tuple(-v for v in p["natural"]) and not any(p["orthogonal"])
    assert result["monthly_geometry"]["created_ties"] == 15
    assert _origins(records)[-1]["bounded_history_magnitude_shares"] is None


def test_nonconstant_loss_cannot_use_analytic_infinite_history():
    with pytest.raises(QuantError, match="constant-loss"):
        rules.inspect(_feed(rates=("analytic_infinity",)))


@pytest.mark.parametrize("rate", ("0", "-1", "NaN", "Infinity", "-Infinity", "not-a-rate"))
def test_invalid_rate_never_becomes_a_success_default(rate):
    with pytest.raises(QuantError):
        rules.inspect(_feed(rates=(rate,)))


@pytest.mark.parametrize("tokens", (_hex(LOGS)[:-1], ("not-hex",) * 6, ("inf",) * 6, ("nan",) * 6))
def test_invalid_binary_operands_are_explicit_errors(tokens):
    with pytest.raises(QuantError):
        rules.inspect(_feed((replace(_month(), log_hex=tokens),)))


@pytest.mark.parametrize("value", (-0.125, 1.125))
def test_outside_original_loss_range_is_an_error(value):
    with pytest.raises(QuantError, match="bounded loss"):
        rules.inspect(_feed((_month(losses=(value,) * 6),)))


def test_future_month_is_not_silently_cropped():
    feed = _feed((_month(), _month(1)))
    first = feed.snapshots["2016-11-30"]
    feed.snapshots[first.origin] = replace(first, months=feed.snapshots["2016-12-30"].months)
    with pytest.raises(QuantError, match="matured month"):
        rules.inspect(feed)


def test_past_month_cannot_change_at_a_later_origin():
    feed = _feed((_month(), _month(1)))
    last = feed.snapshots["2016-12-30"]
    feed.snapshots[last.origin] = replace(
        last, months=(replace(_month(), loss_hex=_hex(LOGS)), _month(1))
    )
    with pytest.raises(QuantError, match="change prior"):
        rules.inspect(feed)


def test_month_chain_gap_is_not_filled():
    second = replace(_month(1), start_after="2016-10-31")
    with pytest.raises(QuantError, match="chain has a gap"):
        rules.inspect(_feed((_month(), second)))


def test_retains_input_before_invalid_endpoint():
    feed = _feed()
    s = feed.snapshots["2016-11-30"]
    invalid = replace(s.adaptive, target_hex=_hex((0.0,) * rules.A))
    feed.snapshots[s.origin] = replace(s, adaptive=invalid)
    records = []
    with pytest.raises(QuantError, match="mixture identity"):
        rules.inspect(feed, retain=records.append)
    assert records[-1]["phase"] == "before_origin" and records[-1]["origin"] == s.origin


def test_future_native_opinion_is_an_error_not_a_refresh():
    feed = _feed()
    s = feed.snapshots["2016-11-30"]
    feed.snapshots[s.origin] = replace(
        s, adaptive=replace(s.adaptive, source_known_at=("2017-01-31",) * 6)
    )
    with pytest.raises(QuantError, match="future date"):
        rules.inspect(feed)


def test_archive_and_memory_cross_same_Interface(tmp_path):
    memory = _feed((_month(), _month(1)))
    catalogue, reader = _archive(tmp_path, memory)
    assert rules.inspect(study.ArchiveAdapter(catalogue, reader=reader)) == rules.inspect(memory)


def test_archived_maturity_primitives_cannot_be_replaced(tmp_path):
    catalogue, reader = _archive(tmp_path, _feed())
    path = Path(catalogue["adaptive_loss_records"]["path"])
    rows = [json.loads(v) for v in path.read_text().splitlines()]
    rows[-2]["matured_source_interval"]["original_loss_labels_hex"] = _hex(LOGS)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    catalogue["adaptive_loss_records"]["sha256"] = file_digest(path)
    with pytest.raises(QuantError, match="primitives were changed"):
        study.ArchiveAdapter(catalogue, reader=reader)


def test_missing_archived_after_half_is_not_backfilled(tmp_path):
    catalogue, reader = _archive(tmp_path, _feed())
    path = Path(catalogue["wealth_records"]["path"])
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    catalogue["wealth_records"]["sha256"] = file_digest(path)
    with pytest.raises(QuantError, match="both halves"):
        study.ArchiveAdapter(catalogue, reader=reader)


def test_future_records_do_not_change_prefix():
    old = _feed((_month(), _month(1), _month(2)))
    new = _feed(
        (_month(), _month(1), _month(2, logs=tuple(v * 3 for v in LOGS), losses=LOGS)),
        ("2", "2", "9"),
    )
    a, b = rules.inspect(old), rules.inspect(new)
    for field in ("origin_row_hashes", "cell_row_hashes"):
        assert [r for r in a[field] if r["origin"] <= "2016-12-30"] == [
            r for r in b[field] if r["origin"] <= "2016-12-30"
        ]


def test_old_source_and_account_entrypoints_never_called(tmp_path, monkeypatch):
    from us_quant import expert_switching_loss_v1 as loss
    from us_quant import expert_switching_wealth_v1 as wealth
    from us_quant import protective_six_adaptive_hedge_v1 as original

    def forbidden(*args, **kwargs):
        raise AssertionError("A closed learner, policy, source or account was replayed.")

    monkeypatch.setattr(loss, "path_partition", forbidden)
    monkeypatch.setattr(loss, "build_targets", forbidden)
    monkeypatch.setattr(wealth, "build_targets", forbidden)
    monkeypatch.setattr(loss.AdaptiveSwitchingLoss, "advance", forbidden)
    monkeypatch.setattr(wealth.SwitchCountWealth, "advance", forbidden)
    monkeypatch.setattr(original.AdaHedge, "update", forbidden)
    monkeypatch.setattr(study.completed, "evaluate", forbidden)
    catalogue, reader = _archive(tmp_path, _feed())
    assert rules.inspect(study.ArchiveAdapter(catalogue, reader=reader))["source_origins"] == 3


def test_only_the_frozen_descriptive_policy_is_allowed():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    policy["nonaffine_or_ordinal_disagreement_is_a_failure_gate"] = True
    with pytest.raises(QuantError, match="descriptive-only"):
        rules.validate(policy)


def test_nested_rational_records_are_serializable_at_the_Interface():
    _, records = _inspect(_feed())
    serialized = json.loads(json.dumps(rules.encode(records), allow_nan=False))
    cells = _cells(serialized)
    exact = cells[0]["projection"]["parallel_energy"]
    assert set(exact) == {"numerator", "denominator"}
    assert cells[0]["projection"]["independent_operands"]["flat_status_matches"]
    assert _origins(serialized)[0]["natural_history_magnitude_shares"] is None
