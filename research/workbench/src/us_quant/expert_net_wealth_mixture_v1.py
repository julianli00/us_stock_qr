from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from us_quant.calendar import is_month_end, next_session, sessions
from us_quant.config import QuantError
from us_quant.strategy_expert_pool_v2 import EXPERT_IDS, MECHANISMS, combine

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/expert-net-wealth-mixture-v1.json"
PRIOR_HEAD = "a035f72f7ff0b186d9235db5ac5310c6237078725d4f37b3bd8d38520ddd7272"
MAX_PERIODS = 119
CANDIDATES = (
    {"id": "expert_pool_net_wealth_mixture_v1", "weighting": "exact_uniform_four_policy_mixture"},
)


def validate(policy):
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_expert_net_wealth_mixture_20261029_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 213,
        "prior_new_registered_candidates": 38,
        "prior_new_completed_strategy_evaluations": 35,
        "prior_new_technical_failures": 3,
        "prior_journal_sequence": 420,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "source_registration": {
            "path": "evidence/four_expert_meta_20261029_registration_v1.json",
            "sha256": "4e84ee5908ab8046f615dc0f6918aeef73aa9207c95bf58a38876c01c270fcfe",
        },
        "previous_master_results": {
            "path": "evidence/expert_master_protection_20261029_results_v1.json",
            "sha256": "a2db53a49da18e3f94918dfea9ce01c6a8194a7aaaafded3ba6b5d18aebf33c0",
        },
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "dirichlet_prior": [1, 1, 1, 1],
        "initial_decision_session": "2016-10-05",
        "shadow_reference_start": "2016-10-06",
        "observation_horizon_sessions": 2512,
        "maximum_complete_learning_intervals": MAX_PERIODS,
        "learner_mass_array_maximum_bytes": 13824000,
        "learning_clock": "completed_full_exchange_month_intervals",
        "training_memory": "expanding_all_completed_intervals",
        "validation_clock": "next_full_month_prequential_before_assimilation",
        "learning_state_resets_for_five_year_account": False,
        "temperature_or_forgetting_parameter": None,
        "switch_cost_proxy_affects_allocation": False,
        "reference_capital_usd": 10000,
        "switch_cost_bps_per_side": 5.0,
        "switch_commission_per_changed_asset_usd": 1.0,
        "changed_asset_numeric_tolerance": 1e-12,
        "invested_target_budget": 0.98,
        "cash_reserve": 0.02,
        "actual_weight_trade_band": 0.025,
        "minimum_economic_factor_families": 3,
        "new_economic_factor_definitions": 0,
        "new_shadow_account_runs": 0,
        "new_source_requests": 0,
        "candidate_count_in_this_study": 1,
        "study_size_is_a_total_research_quota": False,
        "candidates": list(CANDIDATES),
    }
    if not isinstance(policy, dict) or any(
        k not in policy or type(policy[k]) is not type(v) or policy[k] != v
        for k, v in fixed.items()
    ):
        raise QuantError("Keep the fixed four-policy net-score learner and original capital rules.")
    if not isinstance(policy.get("methodology"), dict) or any(
        policy["methodology"].get(k) is not False
        for k in (
            "score_is_actual_main_net_wealth",
            "net_wealth_identity_or_theorem_claimed",
            "algebraic_states_are_independent_economic_trials",
            "main_drawdown_guarantee",
            "independent_forward_validation",
            "investment_objective_verified",
            "order_authority",
            "automatic_live_deployment",
        )
    ):
        raise QuantError("A shadow-score integral cannot imply funded wealth or profit authority.")


class TetrahedralMoments:
    def __init__(self):
        self.degree = 0
        self.mass = np.ones((1, 1, 1))

    def counts(self):
        n = self.degree + 1
        first = np.arange(n)[:, None, None]
        second = np.arange(n)[None, :, None]
        third = np.arange(n)[None, None, :]
        return first, second, third, self.degree - first - second - third

    def weights(self):
        if (
            not 0 <= self.degree <= MAX_PERIODS
            or self.mass.shape != (self.degree + 1,) * 3
            or not np.isfinite(self.mass).all()
            or (self.mass < 0).any()
            or abs(float(self.mass.sum()) - 1) > 1e-12
        ):
            raise QuantError("Invalid bounded moment mass cannot become a neutral allocation.")
        values = np.array(
            [float(np.sum(self.mass * (count + 1))) / (self.degree + 4) for count in self.counts()]
        )
        if (
            not np.isfinite(values).all()
            or (values <= 0).any()
            or abs(float(values.sum()) - 1) > 1e-12
        ):
            raise QuantError("The four complete-expert fractions must be positive and sum to one.")
        return values

    def update(self, relative):
        relative = np.asarray(relative, dtype=float)
        if (
            relative.shape != (4,)
            or not np.isfinite(relative).all()
            or (relative <= 0).any()
            or self.degree >= MAX_PERIODS
        ):
            raise QuantError("Require four positive net relatives within the fixed interval bound.")
        self.weights()
        first, second, third, fourth = self.counts()
        weighted = self.mass / (self.degree + 4)
        new = np.zeros((self.degree + 2,) * 3)
        # Integrated monomial masses shift by the Dirichlet moment ratio (k_i+1)/(n+4).
        new[1:, :-1, :-1] += weighted * (first + 1) * relative[0]
        new[:-1, 1:, :-1] += weighted * (second + 1) * relative[1]
        new[:-1, :-1, 1:] += weighted * (third + 1) * relative[2]
        new[:-1, :-1, :-1] += weighted * np.maximum(fourth + 1, 0) * relative[3]
        total = float(new.sum())
        if not np.isfinite(new).all() or (new < 0).any() or not np.isfinite(total) or total <= 0:
            raise QuantError("A failed moment normalization cannot turn into successful weights.")
        self.mass = new / total
        self.degree += 1
        return self.weights()


def independent_weights(relatives):
    values = np.asarray(relatives, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != 4
        or len(values) > MAX_PERIODS
        or not np.isfinite(values).all()
        or (values <= 0).any()
    ):
        raise QuantError(
            "Independent integration needs complete positive four-policy observations."
        )
    if not len(values):
        return np.full(4, 0.25)
    order = (len(values) + 5) // 2
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes, weights = (nodes + 1) / 2, weights / 2
    u, v, z = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    points = np.column_stack(
        (
            u.ravel(),
            ((1 - u) * v).ravel(),
            ((1 - u) * (1 - v) * z).ravel(),
            ((1 - u) * (1 - v) * (1 - z)).ravel(),
        )
    )
    prior = (
        6
        * (1 - u) ** 2
        * (1 - v)
        * weights[:, None, None]
        * weights[None, :, None]
        * weights[None, None, :]
    ).ravel()
    score = np.zeros(len(points))
    for relative in values:
        score += np.log(points @ relative)
    mass = prior * np.exp(score - score.max())
    result = mass @ points / mass.sum()
    if not np.isfinite(result).all() or (result <= 0).any() or abs(float(result.sum()) - 1) > 1e-12:
        raise QuantError("The independent polynomial-exact integration failed.")
    return result


def observations(experts, day):
    observed = [e.observe(day, 2512) for e in experts]
    expected = (
        sessions("2016-10-06", day) if day >= pd.Timestamp("2016-10-06") else pd.DatetimeIndex([])
    )
    if (
        tuple(o.identity.identifier for o in observed) != EXPERT_IDS
        or tuple(o.identity.mechanism_group for o in observed) != MECHANISMS
        or any(
            o.as_of != day
            or o.target is None
            or o.target_known_at is None
            or o.target_known_at > day
            for o in observed
        )
        or any(
            not o.net_returns.index.equals(expected) or not o.risk_free.index.equals(expected)
            for o in observed
        )
        or any(
            not np.isfinite(o.net_returns).all()
            or (o.net_returns <= -1).any()
            or not np.isfinite(o.risk_free).all()
            for o in observed
        )
        or any(not o.risk_free.equals(observed[0].risk_free) for o in observed)
    ):
        raise QuantError("Unavailable or incomplete complete-policy observations cannot be filled.")
    return observed


def period_relative(observed, first, last):
    dates = sessions(next_session(first), last)
    if len(dates) < 2:
        raise QuantError("Only a genuinely complete month interval can train the mixture.")
    result = []
    for o in observed:
        rows = o.net_returns.loc[dates]
        if not rows.index.equals(dates) or not np.isfinite(rows).all() or (rows <= -1).any():
            raise QuantError("A missing matured net-shadow interval cannot be dropped or filled.")
        result.append(float(np.exp(np.log1p(rows).sum())))
    values = np.asarray(result)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise QuantError("Net period relatives must be finite and positive.")
    return values


def risk_cost_audit(observed, weights, previous, first_anchor, day):
    matrix = np.stack([o.target.to_numpy() for o in observed])
    change = (weights - previous) @ matrix
    changed = int((abs(change) > 1e-12).sum())
    cost = float(0.0005 * 10000 * abs(change).sum() + changed)
    result = {
        "incremental_weight_switch_proxy_at_10000_usd": cost,
        "incremental_weight_switch_changed_assets": changed,
        "proxy_is_diagnostic_not_allocation_penalty_or_total_actual_cost": True,
    }
    if first_anchor is None or day <= first_anchor:
        return {
            **result,
            "risk_status": "no_complete_learning_interval",
            "net_excess_covariance": None,
        }
    dates = sessions(next_session(first_anchor), day)
    excess = np.column_stack([o.net_returns.loc[dates] for o in observed])
    excess -= observed[0].risk_free.loc[dates].to_numpy()[:, None]
    covariance = np.cov(excess, rowvar=False, ddof=1)
    contributions = weights * (covariance @ weights)
    variance = float(contributions.sum())
    if (
        not np.isfinite(covariance).all()
        or not np.isfinite(contributions).all()
        or variance < -1e-14
    ):
        raise QuantError("Invalid known mixture risk cannot become a successful safe state.")
    return {
        **result,
        "risk_status": "observed" if variance > 0 else "degenerate_nonpositive_roundoff_variance",
        "risk_observations": len(dates),
        "net_excess_covariance": covariance.tolist(),
        "marginal_variance_contributions": contributions.tolist(),
        "mixture_variance": variance,
        "variance_contribution_fractions": (contributions / variance).tolist()
        if variance > 0
        else None,
        "risk_is_descriptive_not_an_absolute_drawdown_limit": True,
    }


def build_targets(experts, calendar, assets, policy):
    validate(policy)
    initial = pd.Timestamp(policy["initial_decision_session"])
    if (
        not isinstance(calendar, pd.DatetimeIndex)
        or calendar.empty
        or not calendar.equals(sessions(calendar[0], calendar[-1]))
        or initial not in calendar
        or len(experts) != 4
        or len(set(assets)) != len(assets)
    ):
        raise QuantError("Keep the genuine initial seed and complete admitted asset calendar.")
    decisions = [initial, *[d for d in calendar if d > initial and is_month_end(d)]]
    target = pd.DataFrame(np.nan, index=calendar, columns=assets)
    state, previous = TetrahedralMoments(), np.full(4, 0.25)
    anchor = first_anchor = None
    audits, periods = [], []
    for day in decisions:
        observed = observations(experts, day)
        validation = None
        if is_month_end(day):
            if anchor is None:
                first_anchor = day
            else:
                relative = period_relative(observed, anchor, day)
                validation = float(previous @ relative)
                periods.append(
                    {
                        "start_after": str(anchor.date()),
                        "end": str(day.date()),
                        "net_shadow_relatives": relative.tolist(),
                        "validation_weights_before_assimilation": previous.tolist(),
                        "prequential_net_shadow_score_relative": validation,
                    }
                )
                state.update(relative)
            anchor = day
        weights = state.weights()
        row = combine(observed, weights)
        if tuple(row.index) != tuple(assets):
            raise QuantError("Expert recommendations changed their admitted asset coordinates.")
        target.loc[day] = row
        audits.append(
            {
                "decision_session": str(day.date()),
                "expert_weights": weights.tolist(),
                "previous_expert_weights": previous.tolist(),
                "complete_learning_intervals": state.degree,
                "score_mature": state.degree > 0,
                "reason": "integrated_complete_net_policy_scores"
                if state.degree
                else "unscored_uniform_prior",
                "prequential_net_shadow_score_relative": validation,
                "expert_target_known_at": [str(o.target_known_at.date()) for o in observed],
                "latest_shadow_observation": [
                    str(o.net_returns.index[-1].date()) if len(o.net_returns) else None
                    for o in observed
                ],
                "adapter_requested_observation_window_filled": [o.score_mature for o in observed],
                "learner_mass_array_bytes": state.mass.nbytes,
                "algebraic_monomial_states": (state.degree + 1)
                * (state.degree + 2)
                * (state.degree + 3)
                // 6,
                "risk_and_cost": risk_cost_audit(observed, weights, previous, first_anchor, day),
                "shadow_wealth_added_to_main_capital": False,
                "score_is_actual_main_performance": False,
                "independent_forward_validation": False,
            }
        )
        previous = weights
    return target, audits, periods


def verify_targets(experts, calendar, assets, policy, target, audits, periods):
    validate(policy)
    dates = [
        pd.Timestamp(policy["initial_decision_session"]),
        *[
            d
            for d in calendar
            if d > pd.Timestamp(policy["initial_decision_session"]) and is_month_end(d)
        ],
    ]
    if (
        not target.index.equals(calendar)
        or tuple(target.columns) != tuple(assets)
        or not target.dropna(how="all").index.equals(pd.DatetimeIndex(dates))
        or (target.isna().any(axis=1) & ~target.isna().all(axis=1)).any()
        or [a["decision_session"] for a in audits] != [str(d.date()) for d in dates]
    ):
        raise QuantError(
            "Keep every actual decision, warmup return and complete target coordinate."
        )
    history, expected_periods = [], []
    previous = np.full(4, 0.25)
    anchor = first_anchor = None
    maximum_weight = maximum_target = maximum_validation = 0.0
    for day, audit in zip(dates, audits, strict=True):
        observed = observations(experts, day)
        validation = None
        if is_month_end(day):
            if anchor is None:
                first_anchor = day
            else:
                relative = period_relative(observed, anchor, day)
                validation = float(previous @ relative)
                expected_periods.append(
                    {
                        "start_after": str(anchor.date()),
                        "end": str(day.date()),
                        "net_shadow_relatives": relative.tolist(),
                        "validation_weights_before_assimilation": previous.tolist(),
                        "prequential_net_shadow_score_relative": validation,
                    }
                )
                history.append(relative)
            anchor = day
        weights = independent_weights(np.asarray(history).reshape(-1, 4))
        recorded = np.asarray(audit["expert_weights"], dtype=float)
        previous_recorded = np.asarray(audit["previous_expert_weights"], dtype=float)
        if (
            recorded.shape != (4,)
            or previous_recorded.shape != (4,)
            or not np.isfinite(recorded).all()
            or not np.isfinite(previous_recorded).all()
            or (recorded <= 0).any()
            or abs(float(recorded.sum()) - 1) > 1e-12
        ):
            raise QuantError("The recorded complete-policy fractions must remain finite and valid.")
        row = pd.Series(0.0, index=assets)
        for i, observation in enumerate(observed):
            for asset in assets:
                row[asset] += float(weights[i]) * float(observation.target[asset])
        errors = (
            float(abs(recorded - weights).max()),
            float(abs(target.loc[day] - row).max()),
        )
        if max(errors) > 1e-10:
            raise QuantError(
                "Actual allocation differs from polynomial-exact independent integration."
            )
        maximum_weight, maximum_target = (
            max(maximum_weight, errors[0]),
            max(maximum_target, errors[1]),
        )
        if validation is None:
            if audit["prequential_net_shadow_score_relative"] is not None:
                raise QuantError("An unscored prior cannot acquire a fabricated validation result.")
        else:
            recorded_score = audit["prequential_net_shadow_score_relative"]
            if not isinstance(recorded_score, int | float) or not np.isfinite(recorded_score):
                raise QuantError("A prequential score must be an actual finite observation.")
            error = abs(recorded_score - validation)
            if error > 1e-10:
                raise QuantError(
                    "Prequential validation must precede the current month's assimilation."
                )
            maximum_validation = max(maximum_validation, error)
        if (
            audit["complete_learning_intervals"] != len(history)
            or audit["score_mature"] is not bool(history)
            or audit["reason"]
            != ("integrated_complete_net_policy_scores" if history else "unscored_uniform_prior")
            or not np.allclose(previous_recorded, previous, rtol=0, atol=1e-10)
            or audit["learner_mass_array_bytes"] != (len(history) + 1) ** 3 * 8
            or audit["learner_mass_array_bytes"] > 13824000
            or audit["shadow_wealth_added_to_main_capital"] is not False
            or audit["score_is_actual_main_performance"] is not False
            or audit["independent_forward_validation"] is not False
            or audit["expert_target_known_at"] != [str(o.target_known_at.date()) for o in observed]
            or audit["latest_shadow_observation"]
            != [
                str(o.net_returns.index[-1].date()) if len(o.net_returns) else None
                for o in observed
            ]
            or audit["adapter_requested_observation_window_filled"]
            != [o.score_mature for o in observed]
            or audit["algebraic_monomial_states"]
            != (len(history) + 1) * (len(history) + 2) * (len(history) + 3) // 6
        ):
            raise QuantError(
                "The original causal state, memory bound or capital separation changed."
            )
        expected_audit = risk_cost_audit(observed, weights, previous, first_anchor, day)
        actual_audit = audit["risk_and_cost"]
        if actual_audit.keys() != expected_audit.keys():
            raise QuantError("Risk and cost diagnostics cannot be removed.")
        for key, value in expected_audit.items():
            if isinstance(value, list | float):
                if not np.allclose(actual_audit[key], value, rtol=0, atol=1e-9):
                    raise QuantError("A stored dependence or cost observation changed.")
            elif actual_audit[key] != value:
                raise QuantError("A missing risk state cannot silently become successful.")
        previous = weights
    if len(periods) != len(expected_periods):
        raise QuantError("Every complete prequential interval must remain in the audit.")
    for actual, expected in zip(periods, expected_periods, strict=True):
        for key, value in expected.items():
            if isinstance(value, list | float):
                if not np.allclose(actual[key], value, rtol=0, atol=1e-9):
                    raise QuantError("The recorded training/validation ordering changed.")
            elif actual[key] != value:
                raise QuantError("A complete learning interval cannot be redated.")
    return {
        "every_decision_independently_integrated": True,
        "decisions_checked": len(dates),
        "complete_learning_intervals": len(history),
        "maximum_weight_difference": maximum_weight,
        "maximum_target_difference": maximum_target,
        "maximum_prequential_score_difference": maximum_validation,
        "all_formal_warmup_decisions_retained": True,
        "source_expert_targets_fits_or_accounts_replayed": False,
        "candidate_accounts_or_metrics_computed": False,
        "original_source_signal_recomputed": False,
    }
