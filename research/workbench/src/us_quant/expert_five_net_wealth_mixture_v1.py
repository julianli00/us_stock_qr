from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import roots_jacobi

from us_quant import expert_net_wealth_mixture_v1 as original
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.strategy_expert_pool_v3 import EXPERT_IDS, MECHANISMS, combine

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/expert-five-net-wealth-mixture-v1.json"
PRIOR_HEAD = "b401ca3db1f96aa5729337d6527f721641d0e8c058a3cc664f571e5ccc19ace6"
MAX_PERIODS = 119
STATE_CHUNK = 32768
QUADRATURE_CHUNK = 4096
MAX_MASS_BYTES = 72629040
MAX_COEFFICIENT_BYTES = 142896160
CANDIDATES = (
    {
        "id": "expert_pool_five_net_wealth_mixture_v1",
        "weighting": "exact_uniform_five_policy_mixture",
    },
)
period_relative = original.period_relative
risk_cost_audit = original.risk_cost_audit


def validate(policy):
    fixed = {
        "schema_version": 1,
        "study_id": "ssd_expert_five_net_wealth_mixture_20261030_v1",
        "protocol_id": "ssd_dual_benchmark_net_cagr_v1",
        "prior_completed_configurations": 216,
        "prior_new_registered_candidates": 41,
        "prior_new_completed_strategy_evaluations": 38,
        "prior_new_technical_failures": 3,
        "prior_journal_sequence": 455,
        "prior_journal_head_sha256": PRIOR_HEAD,
        "expert_ids": list(EXPERT_IDS),
        "mechanism_groups": list(MECHANISMS),
        "dirichlet_prior": [1, 1, 1, 1, 1],
        "initial_decision_session": "2016-10-05",
        "shadow_reference_start": "2016-10-06",
        "observation_horizon_sessions": 2512,
        "maximum_complete_learning_intervals": MAX_PERIODS,
        "numeric_representation": "packed_valid_degree_compositions_v1",
        "maximum_algebraic_states": 9078630,
        "learner_mass_array_maximum_bytes": MAX_MASS_BYTES,
        "maximum_concurrent_coefficient_buffer_bytes": MAX_COEFFICIENT_BYTES,
        "state_chunk_size": STATE_CHUNK,
        "independent_quadrature_chunk_size": QUADRATURE_CHUNK,
        "independent_quadrature_maximum_order": 61,
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
        raise QuantError(
            "Keep the fixed five-policy score law, packed bounds and original capital."
        )
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
        raise QuantError("A score integral or numerical state cannot guarantee funded wealth.")


def _choose2(n):
    return n * (n - 1) // 2


def _choose3(n):
    return n * (n - 1) * (n - 2) // 6


def _choose4(n):
    return n * (n - 1) * (n - 2) * (n - 3) // 24


def packed_rank(degree, a, b, c, e):
    return (
        _choose4(degree + 4)
        - _choose4(degree - a + 4)
        + _choose3(degree - a + 3)
        - _choose3(degree - a - b + 3)
        + _choose2(degree - a - b + 2)
        - _choose2(degree - a - b - c + 2)
        + e
    )


class CompositionLayout:
    def __init__(self, degree):
        if type(degree) is not int or not 0 <= degree <= MAX_PERIODS:
            raise QuantError("Packed composition degree must obey the original interval bound.")
        self.degree = degree
        starts, prefixes = [], []
        for a in range(degree + 1):
            for b in range(degree - a + 1):
                c = np.arange(degree - a - b + 1, dtype=np.int64)
                starts.append(packed_rank(degree, a, b, c, 0))
                prefixes.append(
                    np.column_stack(
                        (
                            np.full(len(c), a, dtype=np.int16),
                            np.full(len(c), b, dtype=np.int16),
                            c.astype(np.int16),
                        )
                    )
                )
        self.size = math.comb(degree + 4, 4)
        self.starts = np.concatenate([*starts, np.array([self.size], dtype=np.int64)])
        self.prefixes = np.concatenate(prefixes)
        if (
            self.starts[0] != 0
            or not (np.diff(self.starts) > 0).all()
            or len(self.prefixes) != math.comb(degree + 3, 3)
        ):
            raise QuantError("The packed five-policy composition lattice is inconsistent.")

    @property
    def nbytes(self):
        return self.starts.nbytes + self.prefixes.nbytes

    def chunks(self):
        for first in range(0, self.size, STATE_CHUNK):
            last = min(self.size, first + STATE_CHUNK)
            index = np.arange(first, last, dtype=np.int64)
            segments = np.searchsorted(self.starts, index, side="right") - 1
            prefix = self.prefixes[segments].astype(np.int64)
            fourth = index - self.starts[segments]
            fifth = self.degree - prefix.sum(axis=1) - fourth
            counts = np.column_stack((prefix, fourth, fifth))
            if (counts < 0).any() or not (counts.sum(axis=1) == self.degree).all():
                raise QuantError("A packed index cannot represent an invalid monomial.")
            yield first, last, counts


class PackedFiveMoments:
    def __init__(self):
        self.degree = 0
        self.mass = np.ones(1, dtype=np.float64)
        self.layout = CompositionLayout(0)
        self.maximum_coefficient_buffer_bytes = self.mass.nbytes

    def _checked_mass(self):
        if (
            type(self.degree) is not int
            or not 0 <= self.degree <= MAX_PERIODS
            or not isinstance(self.mass, np.ndarray)
            or self.mass.dtype != np.dtype("float64")
            or self.mass.shape != (math.comb(self.degree + 4, 4),)
            or self.layout.degree != self.degree
            or not np.isfinite(self.mass).all()
            or (self.mass < 0).any()
            or abs(float(self.mass.sum()) - 1) > 1e-12
            or self.mass.nbytes > MAX_MASS_BYTES
        ):
            raise QuantError("Invalid packed mass cannot become a neutral allocation.")

    def weights(self):
        self._checked_mass()
        moments = np.zeros(5)
        for first, last, counts in self.layout.chunks():
            moments += self.mass[first:last] @ (counts + 1)
        result = moments / (self.degree + 5)
        if (
            not np.isfinite(result).all()
            or (result <= 0).any()
            or abs(float(result.sum()) - 1) > 1e-12
        ):
            raise QuantError("The five complete-policy fractions must be positive and sum to one.")
        return result

    def update(self, relative):
        relative = np.asarray(relative, dtype=float)
        if (
            relative.shape != (5,)
            or not np.isfinite(relative).all()
            or (relative <= 0).any()
            or self.degree >= MAX_PERIODS
        ):
            raise QuantError("Require five positive net relatives within the fixed interval bound.")
        self._checked_mass()
        relative = relative / relative.max()
        if (relative <= 0).any():
            raise QuantError("Positive relative ratios cannot underflow into unavailable policies.")
        degree = self.degree + 1
        new = np.zeros(math.comb(degree + 4, 4), dtype=np.float64)
        self.maximum_coefficient_buffer_bytes = max(
            self.maximum_coefficient_buffer_bytes, self.mass.nbytes + new.nbytes
        )
        if self.maximum_coefficient_buffer_bytes > MAX_COEFFICIENT_BYTES:
            raise QuantError("The packed old-plus-new coefficient bound was exceeded.")
        for first, last, counts in self.layout.chunks():
            for i in range(5):
                coordinates = [counts[:, k] + (i == k) for k in range(4)]
                destination = packed_rank(degree, *coordinates)
                # Integrated monomial masses shift by the Dirichlet ratio (k_i+1)/(n+5).
                new[destination] += (
                    self.mass[first:last] * (counts[:, i] + 1) * relative[i] / (self.degree + 5)
                )
        total = float(new.sum())
        if not math.isfinite(total) or total <= 0 or not np.isfinite(new).all() or (new < 0).any():
            raise QuantError("A failed packed normalization cannot become successful weights.")
        new /= total
        self.mass = new
        self.degree = degree
        self.layout = CompositionLayout(degree)
        return self.weights()


def independent_prefix_weights(relatives):
    values = np.asarray(relatives, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != 5
        or len(values) > MAX_PERIODS
        or not np.isfinite(values).all()
        or (values <= 0).any()
    ):
        raise QuantError(
            "Independent integration requires complete positive five-policy intervals."
        )
    if not len(values):
        return np.full((1, 5), 0.2)
    order = (len(values) + 3) // 2
    nodes, probabilities = [], []
    for remaining in (4, 3, 2, 1):
        node, probability = roots_jacobi(order, remaining - 1, 0)
        nodes.append((node + 1) / 2)
        probabilities.append(probability / probability.sum())
    denominator = np.zeros(len(values))
    numerator = np.zeros((len(values), 5))
    log_scale = np.full(len(values), -np.inf)
    for first in range(0, order**4, QUADRATURE_CHUNK):
        last = min(order**4, first + QUADRATURE_CHUNK)
        quotient = np.arange(first, last, dtype=np.int64)
        u = np.empty((len(quotient), 4))
        prior = np.ones(len(quotient))
        for axis in range(3, -1, -1):
            quotient, digit = np.divmod(quotient, order)
            u[:, axis] = nodes[axis][digit]
            prior *= probabilities[axis][digit]
        points = np.empty((len(u), 5))
        remainder = np.ones(len(u))
        for axis in range(4):
            points[:, axis] = remainder * u[:, axis]
            remainder *= 1 - u[:, axis]
        points[:, 4] = remainder
        score = points @ values.T
        if not np.isfinite(score).all() or (score <= 0).any():
            raise QuantError("Independent score integration became invalid.")
        np.log(score, out=score)
        np.cumsum(score, axis=1, out=score)
        new_scale = np.maximum(log_scale, score.max(axis=0))
        scale_old = np.exp(log_scale - new_scale)
        score -= new_scale[None, :]
        np.exp(score, out=score)
        score *= prior[:, None]
        denominator = denominator * scale_old + score.sum(axis=0)
        numerator = numerator * scale_old[:, None] + (points.T @ score).T
        log_scale = new_scale
    result = np.vstack((np.full(5, 0.2), numerator / denominator[:, None]))
    if (
        not np.isfinite(result).all()
        or (result <= 0).any()
        or not np.allclose(result.sum(axis=1), 1, rtol=0, atol=1e-12)
    ):
        raise QuantError("The independent polynomial-exact integral failed.")
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


def build_targets(experts, calendar, assets, policy, before_decision=None):
    validate(policy)
    initial = pd.Timestamp(policy["initial_decision_session"])
    if (
        not isinstance(calendar, pd.DatetimeIndex)
        or calendar.empty
        or not calendar.equals(sessions(calendar[0], calendar[-1]))
        or initial not in calendar
        or len(experts) != 5
        or len(set(assets)) != len(assets)
    ):
        raise QuantError("Keep the genuine initial seed and complete admitted asset calendar.")
    decisions = [initial, *[d for d in calendar if d > initial and is_month_end(d)]]
    target = pd.DataFrame(np.nan, index=calendar, columns=assets)
    state, previous = PackedFiveMoments(), np.full(5, 0.2)
    anchor = first_anchor = None
    audits, periods = [], []
    for day in decisions:
        if before_decision is not None:
            before_decision()
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
                "learner_maximum_coefficient_buffer_bytes": state.maximum_coefficient_buffer_bytes,
                "learner_index_layout_bytes": state.layout.nbytes,
                "algebraic_monomial_states": len(state.mass),
                "layout_is_packed_valid_degree": True,
                "array_bytes_are_total_process_RAM": False,
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
    initial = pd.Timestamp(policy["initial_decision_session"])
    dates = [initial, *[d for d in calendar if d > initial and is_month_end(d)]]
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
    history = []
    anchor = None
    for day in dates:
        observed = observations(experts, day)
        if is_month_end(day):
            if anchor is not None:
                history.append(period_relative(observed, anchor, day))
            anchor = day
    independent = independent_prefix_weights(np.asarray(history).reshape(-1, 5))
    previous = np.full(5, 0.2)
    anchor = first_anchor = None
    degree = 0
    maximum_weight = maximum_target = maximum_validation = 0.0
    expected_periods = []
    for day, audit in zip(dates, audits, strict=True):
        observed = observations(experts, day)
        validation = None
        if is_month_end(day):
            if anchor is None:
                first_anchor = day
            else:
                relative = history[degree]
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
                degree += 1
            anchor = day
        weights = independent[degree]
        recorded = np.asarray(audit["expert_weights"], dtype=float)
        if (
            recorded.shape != (5,)
            or not np.isfinite(recorded).all()
            or (recorded <= 0).any()
            or abs(float(recorded.sum()) - 1) > 1e-12
            or not np.allclose(audit["previous_expert_weights"], previous, rtol=0, atol=1e-10)
        ):
            raise QuantError("Recorded complete-policy fractions must remain finite and valid.")
        expected = np.array(
            [
                math.fsum(float(o.target[symbol]) * weights[i] for i, o in enumerate(observed))
                for symbol in assets
            ]
        )
        error = float(abs(recorded - weights).max())
        target_error = float(abs(target.loc[day].to_numpy() - expected).max())
        if max(error, target_error) > 1e-10:
            raise QuantError("Allocation differs from polynomial-exact independent integration.")
        maximum_weight, maximum_target = (
            max(maximum_weight, error),
            max(maximum_target, target_error),
        )
        if validation is None:
            if audit["prequential_net_shadow_score_relative"] is not None:
                raise QuantError("An unscored prior cannot acquire a fabricated validation result.")
        else:
            actual = audit["prequential_net_shadow_score_relative"]
            if not isinstance(actual, int | float) or not math.isfinite(actual):
                raise QuantError("A prequential score must be an actual finite observation.")
            validation_error = abs(actual - validation)
            if validation_error > 1e-10:
                raise QuantError("Prequential validation must precede assimilation.")
            maximum_validation = max(maximum_validation, validation_error)
        mass_bytes = math.comb(degree + 4, 4) * 8
        buffers = (math.comb(degree + 3, 4) * 8 + mass_bytes) if degree else 8
        if (
            audit["complete_learning_intervals"] != degree
            or audit["score_mature"] is not bool(degree)
            or audit["reason"]
            != ("integrated_complete_net_policy_scores" if degree else "unscored_uniform_prior")
            or audit["learner_mass_array_bytes"] != mass_bytes
            or mass_bytes > MAX_MASS_BYTES
            or audit["learner_maximum_coefficient_buffer_bytes"] != buffers
            or buffers > MAX_COEFFICIENT_BYTES
            or audit["learner_index_layout_bytes"] != math.comb(degree + 3, 3) * 14 + 8
            or audit["algebraic_monomial_states"] != math.comb(degree + 4, 4)
            or audit["layout_is_packed_valid_degree"] is not True
            or audit["array_bytes_are_total_process_RAM"] is not False
            or any(
                audit[k] is not False
                for k in (
                    "shadow_wealth_added_to_main_capital",
                    "score_is_actual_main_performance",
                    "independent_forward_validation",
                )
            )
            or audit["expert_target_known_at"] != [str(o.target_known_at.date()) for o in observed]
            or audit["latest_shadow_observation"]
            != [
                str(o.net_returns.index[-1].date()) if len(o.net_returns) else None
                for o in observed
            ]
            or audit["adapter_requested_observation_window_filled"]
            != [o.score_mature for o in observed]
        ):
            raise QuantError("The causal state, packed memory bound or capital separation changed.")
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
                    raise QuantError("Recorded training/validation ordering changed.")
            elif actual[key] != value:
                raise QuantError("A complete learning interval cannot be redated.")
    return {
        "every_decision_independently_integrated": True,
        "independent_integration": "streamed_polynomial_exact_four_dimensional_Gaussian_Jacobi",
        "decisions_checked": len(dates),
        "complete_learning_intervals": degree,
        "maximum_weight_difference": maximum_weight,
        "maximum_target_difference": maximum_target,
        "maximum_prequential_score_difference": maximum_validation,
        "all_formal_warmup_decisions_retained": True,
        "source_expert_targets_fits_or_accounts_replayed": False,
        "candidate_accounts_or_metrics_computed": False,
        "original_source_signal_recomputed": False,
    }
