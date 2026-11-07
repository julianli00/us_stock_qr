from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import confidence_rated_prod_v1 as learning
from us_quant import main_capital_feedback_v1 as monthly
from us_quant.backtest import BacktestResult
from us_quant.calendar import is_month_end, next_session, previous_session, sessions
from us_quant.cash_funded_accounting_v2 import simulate
from us_quant.execution_gap_shield_v2 import comparison_report
from us_quant.strategy_experts_v1 import StrategyExpert

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/main-risk-confidence-prod-v1.json"
ID = "expert_pool_six_main_risk_confidence_prod_v1"
STUDY = "ssd_main_risk_confidence_prod_20261107_v1"
EXPERT_IDS, MECHANISMS, CLOCK = learning.EXPERT_IDS, learning.MECHANISMS, learning.CLOCK
require = learning.require


def validate(policy):
    expected = {
        "schema_version": 1,
        "study_id": STUDY,
        "candidate_id": ID,
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "actual_native_event_seq": 197702,
        "prior_journal_sequence": 1506,
        "prior_journal_head_sha256": (
            "e3979be6b5fb3ac3783e59c1232663aa6a27ea2be938941473e20b268467aadc"
        ),
        "source_inputs": {
            "path": "reports/ssd-continuous-research-v1/main-risk-confidence-prod-20261107-v1/"
            "source-inputs-v1.json",
            "sha256": "15c9da26392a2f8319bc8823ef6344c7d934aa067c38c3e34762b7ed5c8c2ac2",
        },
        "expert_ids": list(EXPERT_IDS),
        "learning_rule": "unchanged_Adapt_ML_Prod_confidence_reduction",
        "raw_masses_renormalized": False,
        "main_information_clock": "each_original_close_after_fills",
        "allocation_clock": "initial_capital_then_true_NYSE_month_ends",
        "feedback_confidence": "one_minus_original_protection_scale_of_actual_main_NAV_prefix_peak",
        "other_confidences": 1.0,
        "confidence_loss_clock": "recorded_prior_month_origin_not_current_month_end",
        "initial_capital_usd": 10000.0,
        "initial_drawdown_peak_usd": 10000.0,
        "path_local_learner_reset": True,
        "initial_partial_month_assimilated": False,
        "full_month_updates_10y": 119,
        "full_month_updates_5y": 59,
        "source_loss_labels": "original_sealed_six_source_months_unchanged",
        "participation_is_capital_cap_or_drawdown_guarantee": False,
        "confidence_is_probability_or_source_admission": False,
        "invested_target_budget": 0.98,
        "cash_reserve": 0.02,
        "actual_weight_band": 0.025,
        "primary_decimal_precision": 160,
        "independent_decimal_precision": 100,
        "numeric_target_tolerance": 1e-12,
        "canonical_account_gross": "sum_of_actual_held_weight_vector",
        "cash_comparison_operands_retained_before_assertions": True,
        "prefix_positions": [125, 503, 1006, 1509, 2012],
        "candidate_count": 1,
        "study_size_is_a_total_research_quota": False,
        "new_source_requests": 0,
        "source_policies_shadows_or_old_masters_replayed": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    require(
        policy == expected and all(type(policy[k]) is type(v) for k, v in expected.items()),
        "Freeze one monthly actual-main participation law,not a post-outcome capital knob.",
    )


class MainRiskContext:
    def __init__(self, anchor):
        require(
            isinstance(anchor, pd.Timestamp)
            and pd.notna(anchor)
            and anchor.tzinfo is None
            and len(sessions(anchor, anchor)) == 1,
            "Use a real exchange-session anchor.",
        )
        self.anchor = self.last = anchor
        self.nav = self.peak = 10000.0
        self.return_rows = 0

    def observe_close(self, day, nav):
        require(
            isinstance(day, pd.Timestamp)
            and day.tzinfo is None
            and day == next_session(self.last)
            and math.isfinite(nav)
            and nav > 0,
            "Every actual main close must be observed exactly once and in original order.",
        )
        self.last, self.nav = day, float(nav)
        self.peak = max(self.peak, self.nav)
        self.return_rows += 1

    def snapshot(self, day, nav):
        require(
            day == self.last and float(nav).hex() == self.nav.hex(),
            "Participation must read the actual then-known main NAV,not source or future wealth.",
        )
        return {
            "as_of": str(day.date()),
            "actual_main_NAV_hex": self.nav.hex(),
            "actual_main_closing_peak_hex": self.peak.hex(),
            "actual_main_return_rows": self.return_rows,
            "initial_main_capital_anchor": str(self.anchor.date()),
            "source_shadow_or_future_main_used": False,
        }


def confidence(context):
    nav = float.fromhex(context["actual_main_NAV_hex"])
    peak = float.fromhex(context["actual_main_closing_peak_hex"])
    require(
        math.isfinite(nav)
        and math.isfinite(peak)
        and 0 < nav <= peak
        and peak >= 10000
        and context["source_shadow_or_future_main_used"] is False,
        "Use actual main capital and its continuous fresh-capital peak.",
    )
    scale = float(np.clip((nav - 0.85 * peak) / (0.15 * nav), 0, 1))
    return np.array([1.0] * 5 + [1.0 - scale])


@dataclass(frozen=True)
class SourceBundle:
    experts: tuple[StrategyExpert, ...]
    periods: tuple[dict, ...]


class MainRiskProdPolicy:
    def __init__(
        self,
        bundle: SourceBundle,
        assets: tuple[str, ...],
        anchor: pd.Timestamp,
        retain_game: Callable[[dict], None] | None = None,
    ):
        require(
            len(bundle.experts) == 6
            and tuple(e.identity.identifier for e in bundle.experts) == EXPERT_IDS
            and tuple(e.identity.mechanism_group for e in bundle.experts) == MECHANISMS
            and len(assets) == len(set(assets)),
            "Keep the original six admitted complete policies and asset coordinates.",
        )
        self.bundle, self.assets = bundle, tuple(assets)
        self.context, self.state = MainRiskContext(anchor), learning.ConfidenceProd()
        self.periods = {row["end"]: row for row in bundle.periods}
        require(len(self.periods) == len(bundle.periods), "A sealed month cannot occur twice.")
        self.origin = self.prior_confidence = None
        self.last_decision = None
        self.retain = retain_game

    def decide(self, capital):
        day = capital.as_of
        require(
            self.last_decision is None or day > self.last_decision,
            "A new main program cannot replay an earlier learning decision.",
        )
        require(
            capital.initial_capital is (self.last_decision is None)
            and (capital.initial_capital or is_month_end(day))
            and tuple(capital.asset_values.index) == self.assets,
            "Initialization is not a fabricated source refresh;ordinary decisions are monthly.",
        )
        nav = float(capital.asset_values.to_numpy().sum() + capital.cash)
        main = self.context.snapshot(day, nav)
        observed = learning.observations(self.bundle.experts, day)
        require(
            len(observed) == 6
            and all(
                o.target is not None
                and tuple(o.target.index) == self.assets
                and o.as_of == day
                and o.target_known_at <= day
                for o in observed
            ),
            "No missing or future admitted opinion may become neutral capital.",
        )
        labels = saved = None
        if self.origin is not None:
            require(str(day.date()) in self.periods, "A matured sealed month is missing.")
            saved = self.periods[str(day.date())]
            require(
                saved["start_after"] == str(self.origin.date()),
                "Only a full interval after this actual path's prior month origin may train.",
            )
            labels = np.asarray(saved["input_loss_labels"], dtype=float)
        before = {
            "kind": "main_risk_learning_input",
            "decision_session": str(day.date()),
            "main_risk_context": main,
            "state_before": self.state.snapshot(),
            "prior_origin_confidence": None
            if self.prior_confidence is None
            else self.prior_confidence.tolist(),
            "matured_interval": None
            if saved is None
            else {
                "start_after": saved["start_after"],
                "end": saved["end"],
                "sealed_loss_labels_hex": [float(x).hex() for x in labels],
            },
            "source_target_known_at": [str(o.target_known_at.date()) for o in observed],
            "source_actual_introductions": [o.identity.introduced_at for o in observed],
            "source_targets_hex": [[float(x).hex() for x in o.target] for o in observed],
        }
        if self.retain is not None:
            self.retain(before)
        update = None
        if labels is not None:
            require(self.prior_confidence is not None, "A completed label needs its origin rating.")
            update = {
                "start_after": str(self.origin.date()),
                "end": str(day.date()),
                **self.state.update(labels, self.prior_confidence),
            }
        current = confidence(main)
        weights = self.state.weights(current)
        target = learning.combine(observed, weights)
        assessment = {
            "main_risk_context": main,
            "current_confidences": current.tolist(),
            "expert_weights": weights.tolist(),
            "learner_state": self.state.snapshot(),
            "complete_learning_intervals": self.state.periods,
            "period_update": update,
            "known_source_targets_hex": before["source_targets_hex"],
            "source_target_known_at": before["source_target_known_at"],
            "source_actual_introductions": before["source_actual_introductions"],
            "initial_partial_month_assimilated": False,
            "confidence_is_main_capital_cap_or_risk_certificate": False,
            "source_game_reference_is_actual_main_performance": False,
        }
        if self.retain is not None:
            self.retain(
                {
                    "kind": "main_risk_learning_resolved",
                    "decision_session": str(day.date()),
                    "assessment": assessment,
                    "unified_target_hex": [float(x).hex() for x in target],
                }
            )
        if is_month_end(day):
            self.origin = day
        self.prior_confidence, self.last_decision = current.copy(), day
        return monthly.AllocationDecision(target, assessment)


@dataclass(frozen=True)
class MainRiskRun:
    result: BacktestResult
    desired: pd.DataFrame
    issued: pd.DataFrame
    decisions: list[dict]
    pending_after_window: dict
    raw_kernel_auxiliary_gross: pd.Series


def run(data, bundle, start, end, *, cost_bps, delay, before_decision=None, retain=None):
    anchor = previous_session(pd.Timestamp(start))
    policy = MainRiskProdPolicy(bundle, tuple(data.close.columns), anchor, retain)

    def observe(event):
        if event["kind"] == "main_close":
            policy.context.observe_close(pd.Timestamp(event["session"]), event["equity"])
            event = {**event, "actual_main_prefix_peak": policy.context.peak}
        if retain is not None:
            retain(event)

    output = monthly.run(
        data,
        policy,
        start,
        end,
        cost_bps=cost_bps,
        delay=delay,
        before_decision=before_decision,
        retain=observe,
    )
    frame = output.result.frame.copy()
    raw = frame["gross_exposure"].copy()
    canonical = np.array([float(row.copy().sum()) for row in output.result.weights.to_numpy()])
    frame["gross_exposure"] = canonical
    require(
        frame.drop(columns="gross_exposure").equals(
            output.result.frame.drop(columns="gross_exposure")
        ),
        "The canonical representation Adapter cannot alter actual account primitives.",
    )
    return MainRiskRun(
        BacktestResult(frame, output.result.weights),
        output.desired,
        output.issued,
        output.decisions,
        output.pending_after_window,
        raw,
    )


def verify_learning(output):
    rows = output.decisions
    updates = [
        r["assessment"]["period_update"]
        for r in rows
        if r["assessment"]["period_update"] is not None
    ]
    oracle = learning.independent_trace(
        [(u["input_loss_labels"], u["confidence_held_from_prior_origin"]) for u in updates]
    )
    max_weight = max_target = max_state = 0.0
    for i, row in enumerate(rows):
        day = pd.Timestamp(row["signal_session"])
        audit = row["assessment"]
        if row["initial_capital"]:
            nav = peak = 10000.0
            count = 0
        else:
            known = output.result.frame.loc[:day, "equity"]
            nav, peak, count = float(known.iloc[-1]), max(10000.0, float(known.max())), len(known)
        original = audit["main_risk_context"]
        require(
            original["actual_main_NAV_hex"] == nav.hex()
            and original["actual_main_closing_peak_hex"] == peak.hex()
            and original["actual_main_return_rows"] == count,
            "Every rating must read its own actual continuous main prefix.",
        )
        rated = confidence(original)
        require(rated.tolist() == audit["current_confidences"], "Actual main rating changed.")
        periods = audit["complete_learning_intervals"]
        if periods:
            independent = oracle[periods - 1]
            masses = np.array([float(x) for x in independent["mass_after_decimal"]])
            eta = np.asarray(independent["rates_after"])
            state = audit["learner_state"]
            for name, other in (
                ("unnormalized_mass_decimal", "mass_after_decimal"),
                ("squared_confidence_excess_decimal", "squares_after_decimal"),
                ("confidence_regret_decimal", "regrets_after_decimal"),
            ):
                max_state = max(
                    max_state,
                    *[
                        abs(float(Decimal(a) - Decimal(b)))
                        for a, b in zip(state[name], independent[other], strict=True)
                    ],
                )
            max_state = max(
                max_state,
                float(
                    abs(np.array([float(x) for x in state["learning_rates_decimal"]]) - eta).max()
                ),
            )
        else:
            masses, eta = np.full(6, 1 / 6), np.full(6, 0.5)
        weighted = masses * eta * rated
        expected = weighted / math.fsum(weighted)
        actual = np.asarray(audit["expert_weights"])
        max_weight = max(max_weight, float(abs(actual - expected).max()))
        if audit["period_update"] is not None:
            update = audit["period_update"]
            require(
                i > 0
                and update["confidence_held_from_prior_origin"]
                == rows[i - 1]["assessment"]["current_confidences"],
                "End-month main risk cannot backdate the confidence on a past loss.",
            )
            max_weight = max(
                max_weight,
                float(
                    abs(
                        np.asarray(update["prequential_capital_weights_before_assimilation"])
                        - independent["weights_before"]
                    ).max()
                ),
            )
            max_state = max(
                max_state,
                abs(float(update["played_loss_decimal"]) - independent["played_loss"]),
            )
        opinions = [
            [float.fromhex(x) for x in target] for target in audit["known_source_targets_hex"]
        ]
        scalar = [
            float(
                sum(
                    Fraction.from_float(float(w)) * Fraction.from_float(target[j])
                    for w, target in zip(actual, opinions, strict=True)
                )
            )
            for j in range(len(output.desired.columns))
        ]
        max_target = max(max_target, float(abs(output.desired.loc[day].to_numpy() - scalar).max()))
    require(
        max(max_weight, max_target, max_state) <= 1e-12,
        "Independent actual-main confidence,Decimal state or rational target proof failed.",
    )
    return {
        "decisions_checked": len(rows),
        "complete_month_updates": len(updates),
        "maximum_independent_expert_weight_difference": max_weight,
        "maximum_independent_target_difference": max_target,
        "maximum_independent_decimal_state_difference": max_state,
        "all_main_closing_peak_inputs_verified": True,
        "prior_origin_not_end_month_confidence_used_for_losses": True,
        "confidence_game_regret_is_main_wealth": False,
    }


def verify_run(data, output, start, end, cost, delay, *, retain_comparison):
    require(callable(retain_comparison), "Retain comparison operands before exact assertions.")
    replay = simulate(
        data,
        output.issued,
        start,
        end,
        initial_capital=10000,
        cost_bps=cost,
        commission=1,
        delay=delay,
    )
    comparison = comparison_report(output.result, replay)
    retain_comparison(output.result, replay, comparison)
    require(
        comparison["full_frame_equals"] and comparison["full_positions_equal"],
        "Actual monthly main book must exactly match original cash-engine replay.",
    )
    dates = output.result.frame.index
    expected_dates = pd.DatetimeIndex(
        [
            previous_session(dates[0]),
            *[d for d in dates if is_month_end(d)],
        ]
    )
    require(
        [r["signal_session"] for r in output.decisions]
        == expected_dates.strftime("%Y-%m-%d").tolist(),
        "No monthly capital decision or initialization may disappear.",
    )
    expected_issued = output.issued.copy() * np.nan
    pending = None
    for i, (day, row) in enumerate(zip(expected_dates, output.decisions, strict=True)):
        if pending is not None and pending <= day:
            pending = None
        require(pending is None, "Preserve the original one-request main queue.")
        current = (
            np.zeros(len(data.close.columns))
            if i == 0
            else (output.result.weights.loc[day].to_numpy())
        )
        desired = output.desired.loc[day].to_numpy()
        gap = float(abs(current - desired).max())
        reason = (
            "initial_capital"
            if i == 0
            else ("observed_weight_gap" if gap + 1e-12 >= 0.025 else "hold")
        )
        execution = day
        for _ in range(delay):
            execution = next_session(execution)
        require(
            row["reason"] == reason
            and row["execution_session"] == str(execution.date())
            and row["initial_capital"] is (i == 0)
            and abs(row["maximum_observed_target_gap"] - gap) <= 1e-12,
            "Independent monthly band,initialization or delayed-fill proof failed.",
        )
        if reason != "hold":
            pending = execution
            expected_issued.loc[day] = desired
    require(
        expected_issued.equals(output.issued), "Issued targets differ from the main queue proof."
    )
    capital = monthly.verify_capital_inputs(output)
    learning_proof = verify_learning(output)
    return {
        **capital,
        **learning_proof,
        "all_formal_frames_positions_cash_fees_match_cash_engine": True,
        "all_column_exact_equality_acceptance_unchanged": True,
        "comparison_operands_retained_before_assertion": True,
        "maximum_pending_main_request_batches": 1,
        "broker_orders": 0,
    }
