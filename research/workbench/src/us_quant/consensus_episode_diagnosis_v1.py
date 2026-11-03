from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import ssd_research_v1 as base
from us_quant.calendar import previous_session, sessions
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.storage import file_digest, read_json, utc_now, write_json

ROOT = Path(__file__).resolve().parents[2]
FOLDER = base.REPORTS / "consensus-episode-diagnosis-20261103-v1"
INTENT = FOLDER / "diagnostic-intent-v1.json"
OUTPUT = FOLDER / "diagnostic-result-v1.json"
PUBLIC = ROOT / "evidence/consensus_episode_diagnosis_20261103_v1.json"
ROLES = {
    "new": "expert_pool_target_medoid_consensus_v1",
    "primary": "expert_pool_seven_rolling_pair_v1",
    "secondary_context": "expert_pool_seven_equal_weight_v1",
}
KEYS = ("10y-base", "10y-stress", "5y-base", "5y-stress")
SCHEME = "own_retained_peak_first_worst_trough_and_new_episode_shared_close_dates_v1"


def augmented(equity: pd.Series) -> pd.Series:
    if (
        not isinstance(equity, pd.Series)
        or equity.empty
        or not isinstance(equity.index, pd.DatetimeIndex)
        or equity.index.tz is not None
        or not equity.index.is_monotonic_increasing
        or not equity.index.is_unique
        or equity.dtype.kind not in "fi"
        or not np.isfinite(equity).all()
        or (equity <= 0).any()
        or not equity.index.equals(sessions(equity.index[0], equity.index[-1]))
    ):
        raise QuantError(
            "Use a complete, finite, positive stored model NAV calendar without filling."
        )
    return pd.concat(
        [
            pd.Series([10000.0], index=[previous_session(equity.index[0])]),
            equity.astype(float),
        ]
    )


def episode(equity: pd.Series) -> dict:
    nav = augmented(equity)
    losses = 1 - nav / nav.cummax()
    trough = losses.idxmax()
    maximum = float(losses.loc[trough])
    if maximum == 0:
        return {
            "status": "no_positive_drawdown",
            "maximum_drawdown": 0.0,
            "peak_session": None,
            "trough_session": None,
            "recovery_session": None,
        }
    peak = nav.loc[:trough].idxmax()
    recovered = nav.loc[nav.index > trough]
    recovered = recovered[recovered >= nav.loc[peak]]
    return {
        "status": "observed_loss_episode",
        "peak_session": str(peak.date()),
        "trough_session": str(trough.date()),
        "recovery_session": str(recovered.index[0].date()) if len(recovered) else None,
        "maximum_drawdown": maximum,
        "peak_is_initial_capital_anchor": bool(peak == nav.index[0]),
    }


def shared_window(equity: pd.Series, peak: str, trough: str) -> dict:
    nav = augmented(equity)
    start, end = pd.Timestamp(peak), pd.Timestamp(trough)
    if start not in nav.index or end not in nav.index or start > end:
        raise QuantError("Compare only exact stored common close dates, never reset an account.")
    return {
        "start_session": peak,
        "end_session": trough,
        "stored_close_to_close_change": float(nav.loc[end] / nav.loc[start] - 1),
        "own_retained_peak_drawdown_at_end": float(1 - nav.loc[end] / nav.loc[:end].max()),
        "new_account_or_rebased_equity_created": False,
    }


def compare(curves: dict[str, pd.Series], cached_drawdowns: dict[str, float]) -> dict:
    if set(curves) != set(ROLES) or set(cached_drawdowns) != set(ROLES):
        raise QuantError("Keep the new policy and both complete, explicitly labelled controls.")
    if any(
        isinstance(v, bool)
        or not isinstance(v, int | float)
        or not np.isfinite(v)
        or not 0 <= v < 1
        for v in cached_drawdowns.values()
    ):
        raise QuantError("Cached drawdowns must be explicit finite original numeric observations.")
    if any(not curve.index.equals(curves["new"].index) for curve in curves.values()):
        raise QuantError("Every matched model path needs the same original full return calendar.")
    episodes = {role: episode(curve) for role, curve in curves.items()}
    errors = {
        role: abs(row["maximum_drawdown"] - cached_drawdowns[role])
        for role, row in episodes.items()
    }
    if any(not np.isfinite(v) or v > 1e-12 for v in errors.values()):
        raise QuantError("The fixed diagnostic disagrees with a preserved drawdown metric; stop.")
    chosen = episodes["new"]
    intervals = {}
    if chosen["status"] == "observed_loss_episode":
        intervals = {
            role: shared_window(curve, chosen["peak_session"], chosen["trough_session"])
            for role, curve in curves.items()
        }
    primary = episodes["primary"]
    return {
        "own_episodes": episodes,
        "cached_drawdown_consistency_absolute_errors": errors,
        "new_and_primary_own_episode_clocks_identical": (
            chosen["peak_session"] == primary["peak_session"]
            and chosen["trough_session"] == primary["trough_session"]
        )
        if chosen["status"] == primary["status"] == "observed_loss_episode"
        else None,
        "stored_changes_over_new_own_episode": intervals,
        "shared_window_change_is_a_replacement_maximum_drawdown": False,
        "fee_free_or_other_counterfactual_account_computed": False,
        "causal_fee_or_unique_loss_cause_identified": False,
    }


def run() -> dict:
    lifecycle.runtime_guard()
    if OUTPUT.exists() or (FOLDER / "diagnostic-attempt-v1.json").exists():
        raise QuantError("Preserve the complete or partial first diagnostic; do not rerun it.")
    intent = read_json(INTENT)
    if (
        intent["scheme"] != SCHEME
        or intent["roles"] != ROLES
        or set(intent["cases"]) != set(KEYS)
        or intent["new_financial_trials"] != 0
    ):
        raise QuantError("Use only the one predeclared stored-evidence diagnostic.")
    for row in intent["source_references"] + intent["implementation_and_test_inputs"]:
        base.checked_reference(row)
    with lifecycle.research_job():
        history = base.journal_events(base.JOURNAL)
        before = current_counts(history)
        if before != intent["actual_baseline_counts"]:
            raise QuantError("The declared complete financial baseline changed before diagnosis.")
        write_new_receipt(
            FOLDER / "diagnostic-attempt-v1.json",
            {"started_at": utc_now(), "intent_sha256": file_digest(INTENT)},
        )
        results = {}
        try:
            with (FOLDER / "actual-diagnostic-records-v1.jsonl").open("x") as stream:
                for key in KEYS:
                    lifecycle.runtime_guard()
                    curves, cached = {}, {}
                    for role in ROLES:
                        case = intent["cases"][key][role]
                        if case["candidate_id"] != ROLES[role]:
                            raise QuantError(
                                "A cached comparison role cannot select another winner."
                            )
                        receipt = read_json(base.checked_reference(case["receipt"]))
                        if receipt["candidate_id"] != ROLES[role] or receipt["path_key"] != key:
                            raise QuantError(
                                "The cached receipt belongs to a different actual account."
                            )
                        frame = pd.read_csv(
                            base.checked_reference(case["account"]),
                            index_col=0,
                            parse_dates=True,
                            float_precision="round_trip",
                        )
                        dates = frame.index.strftime("%Y-%m-%d")
                        saved = receipt["metrics"]
                        if (dates[0], dates[-1], len(dates)) != (
                            saved["start"],
                            saved["end"],
                            saved["sessions"],
                        ):
                            raise QuantError("Do not shorten a formal stored account window.")
                        curves[role], cached[role] = frame["equity"], saved["max_drawdown"]
                    results[key] = compare(curves, cached)
                    stream.write(
                        json.dumps({"path_key": key, "result": results[key]}, allow_nan=False)
                        + "\n"
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            write_new_receipt(
                FOLDER / "failure-v1.json",
                {
                    "failed_at": utc_now(),
                    "error": str(error),
                    "completed_diagnostic_cases": list(results),
                    "partial_evidence_preserved": True,
                    "new_financial_trials": 0,
                },
            )
            raise
        if current_counts(base.journal_events(base.JOURNAL)) != before:
            raise QuantError("A read-only diagnostic cannot change financial trial counts.")
        packet = {
            "schema_version": 1,
            "completed_at": utc_now(),
            "scheme": SCHEME,
            "intent": base.reference(INTENT),
            "results": results,
            "actual_authoritative_counts": before,
            "new_financial_trials": 0,
            "new_economic_allocation_rules_selected": 0,
            "source_score_target_or_account_pipeline_replayed": False,
            "new_CAGR_or_Sharpe_or_qualification_computed": False,
            "newly_computed_diagnostic_not_recovered_original_internal_state": True,
            "episodes_are_ex_post_descriptions_not_predictive_filters": True,
            "source_failures_and_original_financial_rejection_preserved": True,
            "raw_source_prices_targets_or_daily_account_rows_exported": False,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "order_authority": False,
        }
        write_new_receipt(OUTPUT, packet)
        write_new_receipt(PUBLIC, packet)
        write_json(
            ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
            {
                "updated_at_utc": utc_now(),
                "phase": "consensus_episode_diagnosis_closed",
                "active_compute_job": None,
                "next_research_question": None,
                **before,
                "diagnostic_result": base.reference(OUTPUT),
                "next_step": "select_independent_question_without_hindsight_filter",
            },
        )
    return {"private_result": base.reference(OUTPUT), "public_projection": base.reference(PUBLIC)}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
