from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.config import QuantError
from us_quant.selection_validation import (
    circular_positions,
    joint_maximum,
    original_bundle,
    validate_policy,
    verified_accounts,
)
from us_quant.storage import digest_json, read_json, write_json


def test_circular_blocks_share_exact_contiguous_positions_and_truncate_the_last_block():
    positions = circular_positions(55, 4, 21, np.random.default_rng(7))
    assert positions.shape == (4, 55)
    assert positions.min() >= 0 and positions.max() < 55
    for start, length in ((0, 21), (21, 21), (42, 13)):
        np.testing.assert_array_equal(
            positions[:, start : start + length],
            (positions[:, start, None] + np.arange(length)) % 55,
        )
    np.testing.assert_array_equal(
        positions, circular_positions(55, 4, 21, np.random.default_rng(7))
    )


def inputs(count=300):
    rng = np.random.default_rng(23)
    market = rng.normal(0, 0.002, count)
    return pd.DataFrame(
        {
            "a": market + rng.normal(0, 0.0003, count),
            "b": market + rng.normal(0, 0.0003, count),
            "c": -market + rng.normal(0, 0.0003, count),
        },
        index=pd.date_range("2020-01-01", periods=count),
    )


def test_deterministic_common_resampling_and_no_assumed_strategy_independence():
    frame = inputs()
    before = joint_maximum(frame, samples=512, block=21, seed=8, alpha=0.025)
    after = joint_maximum(frame, samples=512, block=21, seed=8, alpha=0.025)
    assert before == after
    assert before["comparisons"] == 3
    assert before["monte_carlo_p_value_resolution"] == 1 / 513
    assert all(
        row["simultaneous_lower_mean_daily_log_advantage"]
        <= row["observed_mean_daily_log_advantage"]
        for row in before["comparison_results"]
    )


def test_perfectly_duplicate_candidates_do_not_create_independent_bootstrap_risk():
    frame = inputs()[["a"]]
    single = joint_maximum(frame, samples=512, block=21, seed=8, alpha=0.025)
    duplicated = joint_maximum(
        frame.assign(copy=frame["a"]), samples=512, block=21, seed=8, alpha=0.025
    )
    assert duplicated["scope_limited_omnibus_p_value"] == single["scope_limited_omnibus_p_value"]
    assert duplicated["bootstrap_maximum_critical_daily_log_advantage"] == pytest.approx(
        single["bootstrap_maximum_critical_daily_log_advantage"], abs=1e-15
    )


def test_including_more_candidates_never_narrows_the_common_maximum_bound():
    frame = inputs()
    single = joint_maximum(frame[["a"]], samples=512, block=21, seed=8, alpha=0.025)
    multiple = joint_maximum(frame, samples=512, block=21, seed=8, alpha=0.025)
    assert multiple["bootstrap_maximum_critical_daily_log_advantage"] >= single[
        "bootstrap_maximum_critical_daily_log_advantage"
    ] - 1e-15


def test_joint_maximum_matches_independent_direct_index_resampling():
    frame = inputs(137)
    samples, block, seed, alpha = 256, 21, 8, 0.025
    result = joint_maximum(frame, samples=samples, block=block, seed=seed, alpha=alpha)
    means = frame.to_numpy().mean(axis=0)
    centered = frame.to_numpy() - means
    rng = np.random.default_rng(seed)
    reference = []
    for _ in range(samples):
        starts = rng.integers(0, len(frame), size=math.ceil(len(frame) / block))
        positions = np.concatenate(
            [(start + np.arange(block)) % len(frame) for start in starts]
        )[: len(frame)]
        reference.append(max(0.0, float(centered[positions].mean(axis=0).max())))
    critical = sorted(reference)[math.ceil((samples - 1) * (1 - alpha))]
    observed = max(0.0, float(means.max()))
    p_value = (1 + sum(value >= observed for value in reference)) / (samples + 1)
    assert result["scope_limited_omnibus_p_value"] == p_value
    assert result["bootstrap_maximum_critical_daily_log_advantage"] == pytest.approx(
        critical, abs=1e-15
    )
    np.testing.assert_allclose(
        [row["simultaneous_lower_mean_daily_log_advantage"] for row in result["comparison_results"]],
        means - critical,
        rtol=0,
        atol=1e-15,
    )


def test_nonpositive_observed_advantage_has_positive_part_omnibus_p_value_one():
    frame = inputs() - 0.01
    result = joint_maximum(frame, samples=512, block=21, seed=8, alpha=0.025)
    assert result["positive_part_maximum_daily_log_advantage"] == 0
    assert result["scope_limited_omnibus_p_value"] == 1
    assert not any(
        row["strictly_positive_simultaneous_lower_bound"]
        for row in result["comparison_results"]
    )


def test_large_synthetic_advantage_is_detected_but_is_not_real_research_evidence():
    frame = inputs() + 0.01
    result = joint_maximum(frame, samples=512, block=21, seed=8, alpha=0.025)
    assert result["scope_limited_omnibus_p_value"] == 1 / 513
    assert all(
        row["strictly_positive_simultaneous_lower_bound"]
        for row in result["comparison_results"]
    )


@pytest.mark.parametrize(
    "problem", ["nan", "duplicate_columns", "duplicate_dates", "unsorted_dates", "short"]
)
def test_bad_or_incomplete_inputs_are_explicitly_rejected(problem):
    frame = inputs()
    if problem == "nan":
        frame.iloc[0, 0] = np.nan
    elif problem == "duplicate_columns":
        frame.columns = ["a", "a", "c"]
    elif problem == "duplicate_dates":
        frame.index = [frame.index[0]] * len(frame)
    elif problem == "unsorted_dates":
        frame = frame.iloc[::-1]
    else:
        frame = frame.head(20)
    with pytest.raises(QuantError):
        joint_maximum(frame, samples=512, block=21, seed=8, alpha=0.025)


@pytest.mark.parametrize("setting", ["seed", "block_sessions", "included_candidate_count"])
def test_audit_scope_or_settings_cannot_be_changed_after_registration(setting):
    policy = read_json(Path(__file__).parents[1] / "config/selection-validation.json")
    policy[setting] += 1
    with pytest.raises(QuantError):
        validate_policy(policy)


def test_missing_reviewed_account_paths_cannot_be_silently_dropped():
    policy = read_json(Path(__file__).parents[1] / "config/selection-validation.json")
    with pytest.raises(QuantError, match="all four"):
        verified_accounts({"paths": []}, {"paths": []}, 10, policy)


def test_original_bundle_is_selected_by_review_digest_not_newer_output(tmp_path, monkeypatch):
    monkeypatch.setattr("us_quant.selection_validation.ROOT", tmp_path)
    original = {"as_of": "2026-10-05", "paths": []}
    altered = {"as_of": "2026-10-06", "paths": []}
    write_json(tmp_path / "reports/a/test_candidate/bundle.json", altered)
    write_json(tmp_path / "reports/b/test_candidate/bundle.json", original)
    value, _ = original_bundle(
        {"candidate_id": "test_candidate", "evidence_sha256": digest_json(original)}
    )
    assert value == original
    with pytest.raises(QuantError, match="original reviewed"):
        original_bundle({"candidate_id": "test_candidate", "evidence_sha256": "0" * 64})
