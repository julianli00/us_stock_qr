from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from us_quant import archived_hmm_session_eligibility_v1 as review
from us_quant import hmm_session_expert_checks_v1 as checks
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.storage import read_json


@pytest.fixture
def composition():
    dates = sessions("2015-08-10", "2026-10-05")
    active = pd.DatetimeIndex(
        [d for d in dates if d >= pd.Timestamp("2016-08-31") and is_month_end(d)]
    )
    parent = pd.DataFrame(np.nan, index=dates, columns=review.UNIVERSE[:12])
    parent.loc[active] = 0.0
    parent.loc[active, list(checks.CORE)] = 0.15
    parent.loc[active, ["SOXX", "GLD", "BIL"]] = [0.2, 0.1, 0.08]
    child = parent.copy()
    child.loc[active, list(checks.CORE)] = [0.21, 0.18, 0.12, 0.09]
    audits = []
    for day in active:
        p = int(dates.get_loc(day))
        audits.append(
            {
                "decision_session": str(day.date()),
                "source_core_budget": 0.6,
                "method": "intraday",
                "first_prior_close": str(dates[p - 126].date()),
                "last_return_session": str(dates[p - 21].date()),
                "return_sessions": 105,
                "ranking": list(checks.CORE),
                "all_noncore_targets_preserved": True,
                "hmm_refitted_or_states_recomputed": False,
                "partial_session_trading_account_claimed": False,
            }
        )
    proof = {
        "monthly_decisions": 122,
        "original_calendar_opportunities": 131,
        "source_target_months": 122,
        "undefined_prehistory_months": 9,
        "additional_source_months_dropped": 0,
        "first_eligible_source_month": "2016-08-31",
        "exact_noncore_and_combined_core_budget_preserved": True,
        "hmm_refitted_or_states_recomputed": False,
        "original_source_signal_recomputed": False,
        "candidate_accounts_or_metrics_computed": False,
        "prefix_truncations": [str(active[p].date()) for p in (0, 12, 36, 60, 121)],
        "maximum_independent_target_difference": 0.0,
        "maximum_independent_unrounded_score_difference": 0.0,
        "maximum_log_split_identity_error": 0.0,
    }
    return parent, child, audits, proof


def test_full_original_opinions_warmup_and_noncore_budget_preserved(composition):
    evidence = checks.check_composition(*composition, "intraday")
    assert evidence["genuine_native_monthly_opinions"] == 122
    assert evidence["undefined_prehistory_months_retained"] == 9
    assert evidence["all_noncore_cells_exact"]


@pytest.mark.parametrize(
    "failure",
    [
        "native_date",
        "prehistory",
        "noncore",
        "budget",
        "lag",
        "method",
        "rank",
        "HMMrefit",
        "prefix",
        "proofNaN",
    ],
)
def test_no_source_clock_scope_rank_or_proof_repair(composition, failure):
    parent, child, audits, proof = deepcopy(composition)
    if failure == "native_date":
        child.loc[child.dropna().index[20]] = np.nan
    elif failure == "prehistory":
        child.iloc[20] = child.dropna().iloc[0]
    elif failure == "noncore":
        child.loc[child.dropna().index[0], "GLD"] += 0.01
    elif failure == "budget":
        child.loc[child.dropna().index[0], "MTUM"] += 0.01
    elif failure == "lag":
        audits[0]["last_return_session"] = audits[0]["decision_session"]
    elif failure == "method":
        audits[0]["method"] = "overnight"
    elif failure == "rank":
        audits[0]["ranking"] = ["MTUM"] * 4
    elif failure == "HMMrefit":
        proof["hmm_refitted_or_states_recomputed"] = True
    elif failure == "prefix":
        proof["prefix_truncations"].pop()
    else:
        proof["maximum_independent_target_difference"] = float("nan")
    with pytest.raises(QuantError):
        checks.check_composition(parent, child, audits, proof, "intraday")


@pytest.fixture
def book():
    dates = sessions("2016-10-06", "2026-10-05")
    frame = pd.DataFrame(
        {
            "equity": 10000.0,
            "return": 0.0,
            "risk_free": 0.00004,
            "cash": 200.0,
            "gross_exposure": 0.98,
            "cost": 0.0,
            "orders": 0,
        },
        index=dates,
    )
    weights = pd.DataFrame(0.0, index=dates, columns=review.UNIVERSE[:12])
    weights.loc[:, list(checks.CORE)] = 0.245
    rf = pd.Series(0.00004, index=dates, name="risk_free")
    return frame, weights, dates, rf


def test_strict_native_RF_has_no_rounding_view_or_noise_waiver(book):
    assert checks.check_book(*book)["native_RF_exact_common"]
    frame, weights, dates, rf = book
    frame.iloc[0, frame.columns.get_loc("risk_free")] = np.nextafter(0.00004, np.inf)
    evidence = checks.check_book(frame, weights, dates, rf)
    assert evidence["native_RF_exact_common"] is False
    assert evidence["native_RF_mismatched_days"] == 1
    assert evidence["RF_rounding_normalization_or_view_applied"] is False


@pytest.mark.parametrize(
    "failure", ["cash", "gross", "missing", "styles", "NaN", "borrowedweights"]
)
def test_invalid_funding_full_dates_or_style_exposure_cannot_be_admitted(book, failure):
    frame, weights, dates, rf = book
    if failure == "cash":
        frame.iloc[0, frame.columns.get_loc("cash")] = -1
    elif failure == "gross":
        frame.iloc[0, frame.columns.get_loc("gross_exposure")] = 1.01
    elif failure == "missing":
        frame = frame.iloc[1:]
    elif failure == "styles":
        weights.loc[:, list(checks.CORE)[2:]] = 0
    elif failure == "NaN":
        frame.iloc[0, frame.columns.get_loc("return")] = np.nan
    else:
        weights.iloc[0, 0] = -0.1
    with pytest.raises(QuantError):
        checks.check_book(frame, weights, dates, rf)


@pytest.mark.parametrize(
    "key,value",
    [
        ("native_monthly_opinions", 121),
        ("original_independent_BT_tolerance_USD", 0.01),
        ("RF_rounding_normalization_cropping_or_new_assessment_view", True),
        ("financial_return_performance_is_technical_admission_gate", True),
    ],
)
def test_no_new_performance_or_RF_or_tolerance_gate(key, value):
    policy = read_json(review.POLICY)
    policy[key] = value
    with pytest.raises(QuantError):
        review.validate(policy)


@pytest.fixture
def admission(book, monkeypatch):
    frame, weights, dates, rf = book
    expert = SimpleNamespace(observe=lambda day, lookback: SimpleNamespace(risk_free=rf.copy()))
    refs = [{"path": p, "sha256": "a" * 64} for p in review.OWN]
    row = {
        "status": "technical_eligibility_passed_admissible_nonselected",
        "technical_expert_admitted": True,
        "admitted_at": "2026-11-05T12:00:00Z",
        "financial_reviewed_at": "2026-10-29T08:30:00Z",
        "source_descriptor": {},
    }
    intent = {
        "created_at": "2026-11-05T11:00:00Z",
        "source_references": [],
        "complete_dependency_inputs": refs.copy(),
        "implementation_and_test_inputs": refs.copy(),
    }
    result = {
        "global_pool_changed": False,
        "new_financial_evaluations": 0,
        "order_authority": False,
        "RF_rounding_normalization_or_new_view": False,
        "eligibility_intent": {"path": "intent", "sha256": "b" * 64},
        "candidates": {review.IDS[0]: row},
    }
    monkeypatch.setattr(review.base, "checked_reference", lambda ref: Path(ref["path"]))
    monkeypatch.setattr(
        review, "read_json", lambda path: result if path == Path("admission") else intent
    )
    monkeypatch.setattr(review, "load_expert", lambda *args: (expert, {}))
    return result, intent, rf, {"path": "admission", "sha256": "c" * 64}


def test_only_passed_saved_source_consumer_uses_exact_native_RF(admission):
    result, intent, rf, ref = admission
    expert, row = review.load_admitted(ref, review.IDS[0], review.UNIVERSE, rf)
    assert row["technical_expert_admitted"] and expert.observe(None, 2512).risk_free.equals(rf)


@pytest.mark.parametrize(
    "failure", ["blocked", "RF", "view", "code", "backdate", "universe", "trading"]
)
def test_no_consumer_for_failed_repaired_stale_or_remapped_source(admission, failure):
    result, intent, rf, ref = admission
    universe = review.UNIVERSE
    if failure == "blocked":
        result["candidates"][review.IDS[0]]["technical_expert_admitted"] = False
    elif failure == "RF":
        rf = rf.copy()
        rf.iloc[0] = np.nextafter(0.00004, np.inf)
    elif failure == "view":
        result["RF_rounding_normalization_or_new_view"] = True
    elif failure == "code":
        intent["complete_dependency_inputs"].pop()
    elif failure == "backdate":
        result["candidates"][review.IDS[0]]["admitted_at"] = "2026-10-01T00:00:00Z"
    elif failure == "universe":
        universe = universe[::-1]
    else:
        result["order_authority"] = True
    with pytest.raises(QuantError):
        review.load_admitted(ref, review.IDS[0], universe, rf)
