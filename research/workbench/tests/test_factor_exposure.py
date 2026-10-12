from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.factor_exposure import coverage, fit_pair, read_daily_archive, validate_policy
from us_quant.storage import read_json

MEMBER = "F-F_Research_Data_5_Factors_2x3_daily.csv"
COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]


def archive(tmp_path, rows, member=MEMBER, columns=COLUMNS):
    path = tmp_path / "factors.zip"
    with ZipFile(path, "w") as output:
        output.writestr(
            member,
            "This file was created by using the 202608 CRSP database.\n\n"
            + ",".join(["", *columns])
            + "\n"
            + "\n".join(rows)
            + "\n\nCopyright synthetic test fixture\n",
        )
    return path


def test_exact_daily_percent_units_and_negative_factor_returns_are_preserved(tmp_path):
    path = archive(
        tmp_path,
        [
            "20260828,1.25,-0.37,0.28,1.66,0.49,0.01",
            "20260831,-0.33,-0.29,-0.39,-1.09,0.03,0.01",
        ],
    )
    result = read_daily_archive(path, MEMBER, COLUMNS)
    assert list(result.columns) == COLUMNS
    assert result.index.equals(pd.to_datetime(["2026-08-28", "2026-08-31"]))
    assert result.iloc[0]["Mkt-RF"] == pytest.approx(0.0125)
    assert result.iloc[0]["SMB"] == pytest.approx(-0.0037)
    assert result.iloc[0]["RF"] == pytest.approx(0.0001)


@pytest.mark.parametrize("bad", ["-99.99", "-999", "nan", "inf", ""])
def test_missing_sentinels_or_invalid_numbers_never_become_filled_returns(tmp_path, bad):
    path = archive(tmp_path, [f"20260831,{bad},0,0,0,0,0.01"])
    with pytest.raises(QuantError):
        read_daily_archive(path, MEMBER, COLUMNS)


@pytest.mark.parametrize(
    "problem", ["member", "columns", "duplicate", "unsorted", "bad_date", "section"]
)
def test_archive_identity_chronology_and_daily_table_boundaries_are_strict(tmp_path, problem):
    rows = ["20260828,1,0,0,0,0,0.01", "20260831,1,0,0,0,0,0.01"]
    member, columns = MEMBER, COLUMNS
    if problem == "member":
        member = "unrelated.csv"
    elif problem == "columns":
        columns = COLUMNS[::-1]
    elif problem == "duplicate":
        rows[1] = rows[0]
    elif problem == "unsorted":
        rows.reverse()
    elif problem == "bad_date":
        rows[1] = "20260230,1,0,0,0,0,0.01"
    else:
        rows.insert(1, "")
    with pytest.raises(QuantError):
        read_daily_archive(archive(tmp_path, rows, member, columns), MEMBER, COLUMNS)


def test_missing_source_tail_is_explicit_and_does_not_shorten_the_formal_window():
    dates = sessions("2025-08-01", "2026-10-05")
    source = pd.DataFrame(0.001, index=dates[dates <= "2026-08-31"], columns=COLUMNS)
    report = coverage(dates, source)
    assert report["formal_sessions"] == len(dates)
    assert report["formal_end"] == "2026-10-05"
    assert report["available_overlap_end"] == "2026-08-31"
    assert len(report["unavailable_tail_sessions"]) == 24
    assert not report["covers_complete_qualification_window"]
    assert not report["formal_qualification_window_changed"]
    with pytest.raises(QuantError, match="No dropping"):
        coverage(dates, source.drop(source.index[25]))
    with pytest.raises(QuantError, match="actual factor source cut"):
        coverage(dates, source.iloc[:-1])


def example():
    rng = np.random.default_rng(13)
    dates = pd.bdate_range("2016-01-04", periods=600)
    factors = pd.DataFrame(
        rng.normal(0.0002, 0.01, (600, 4)), index=dates, columns=["m", "s", "g", "b"]
    )
    design = np.column_stack([np.ones(len(dates)), factors])
    beta = np.array([[0.0001, -0.00002], [0.6, 1.0], [0.3, 0.05], [0.2, 0.01], [0.1, -0.01]])
    excess = pd.DataFrame(
        design @ beta + rng.normal(0, 0.001, (600, 2)),
        index=dates,
        columns=["strategy", "spy"],
    )
    return factors, excess


def test_coefficients_mean_decomposition_and_pair_match_independent_normal_equations():
    factors, excess = example()
    result = fit_pair(factors, excess)
    x = np.column_stack([np.ones(len(factors)), factors])
    beta = np.linalg.solve(x.T @ x, x.T @ excess.to_numpy())
    for i, name in enumerate(excess.columns):
        actual = result[name]
        np.testing.assert_allclose(
            list(actual["coefficients"].values()), beta[:, i], atol=1e-12, rtol=0
        )
        components = beta[1:, i] * factors.mean().to_numpy()
        assert actual["mean_daily_excess_return"] == pytest.approx(
            beta[0, i] + components.sum(), abs=1e-12
        )
        assert actual["annualized_252_arithmetic_intercept"] == pytest.approx(
            beta[0, i] * 252, abs=1e-12
        )
        error = excess.iloc[:, i].to_numpy() - x @ beta[:, i]
        total = ((excess.iloc[:, i] - excess.iloc[:, i].mean()) ** 2).sum()
        assert actual["r_squared"] == pytest.approx(1 - error @ error / total, abs=1e-12)
    paired = result["paired_strategy_minus_spy"]
    np.testing.assert_allclose(
        list(paired["coefficients"].values()), beta[:, 0] - beta[:, 1], atol=1e-12
    )
    assert result["in_sample_descriptive_fit_only"]
    assert not result["independent_alpha_proven"]


@pytest.mark.parametrize(
    "problem", ["nan", "misaligned", "duplicate", "constant", "rank", "short", "shape"]
)
def test_fits_reject_bad_or_unidentified_inputs_instead_of_reporting_false_exposure(problem):
    factors, excess = example()
    if problem == "nan":
        excess.iloc[2, 0] = np.nan
    elif problem == "misaligned":
        excess = excess.iloc[::-1]
    elif problem == "duplicate":
        factors.columns = ["m", "s", "g", "m"]
    elif problem == "constant":
        excess["strategy"] = 0
    elif problem == "rank":
        factors["s"] = factors["m"]
    elif problem == "short":
        factors, excess = factors.iloc[:20], excess.iloc[:20]
    else:
        excess["extra"] = 0.001
    with pytest.raises(QuantError):
        fit_pair(factors, excess)


@pytest.mark.parametrize(
    "field,value",
    [
        ("factor_overlap_last_session", "2026-10-05"),
        ("included_candidate_count", 3),
        ("new_strategy_evaluations", 1),
        ("order_authority", True),
        ("qualification_policy_changed", True),
        ("forward_rules_changed", True),
        ("fit_is_out_of_sample_forecast", True),
        ("register_queued_factors_early", True),
    ],
)
def test_diagnostic_scope_is_not_qualification_or_an_early_factor_experiment(field, value):
    policy = read_json(Path(__file__).parents[1] / "config/factor-exposure.json")
    policy[field] = value
    with pytest.raises(QuantError):
        validate_policy(policy)
