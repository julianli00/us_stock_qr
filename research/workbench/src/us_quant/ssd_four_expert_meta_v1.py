from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import expert_initial_monthly_execution_v1 as execution
from us_quant import ssd_expert_meta_research_v1 as source
from us_quant import strategy_expert_pool_v2 as rules
from us_quant.calendar import previous_session
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import research_job, runtime_guard
from us_quant.research_weekly import verify_publication
from us_quant.ssd_research_v1 import append_event, matching_events, published_inputs, reference
from us_quant.storage import (
    digest_bytes,
    digest_json,
    file_digest,
    read_json,
    utc_now,
    write_text_atomic,
)
from us_quant.strategy_experts_v1 import ArchivedDenseExpert, ExpertIdentity

base, common, current = source.base, source.common, source.current
ROOT = base.ROOT
FOLDER = base.REPORTS / "four-expert-meta-20261029-v1"
INTENT = FOLDER / "research-intent-v1.json"
TARGETS = ROOT / "data/ssd-four-expert-meta-20261029-v1"
PREPARED = FOLDER / "prepared-target-bundle.json"
PACKET = ROOT / "evidence/four_expert_meta_20261029_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/four_expert_meta_20261029_results_v1.json"
IDS = tuple(c["id"] for c in rules.CANDIDATES)
OWN = (
    "config/strategy-expert-pool-four-v1.json",
    "src/us_quant/strategy_expert_pool_v2.py",
    "src/us_quant/ssd_four_expert_meta_v1.py",
    "tests/test_strategy_expert_pool_v2.py",
    "tests/test_ssd_four_expert_meta_v1.py",
    "src/us_quant/expert_initial_monthly_execution_v1.py",
    "tests/test_expert_initial_monthly_execution_v1.py",
)
FIRST_PUBLIC = {
    "path": "evidence/strategy_expert_pool_20261029_admission_v1.json",
    "sha256": "b6bc161acba0ac0c1c7e759c8408d7a49d5c6be969aa24d4c192e2120a462871",
}


def source_bundle():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    data, experts, parent, original, benchmarks = source.inputs()
    pool = read_json(base.checked_reference(policy["first_pool_admission"]))
    public_pool = read_json(base.checked_reference(FIRST_PUBLIC))
    admission = read_json(base.checked_reference(policy["feedback_expert_admission"]))
    record = admission["expert"]
    if (
        policy["first_pool_admission"] != parent["expert_pool"]
        or policy["source_scoring_policy"] != parent["policy"]
        or tuple(r["id"] for r in pool["experts"]) != rules.EXPERT_IDS[:3]
        or record["id"] != rules.EXPERT_IDS[-1]
        or record["mechanism_group"] != rules.MECHANISMS[-1]
        or admission["source_market_12_assets_and_RF_match_admitted_14_asset_panel"] is not True
        or admission["new_strategy_evaluations"] != 0
        or admission["new_shadow_account_runs"] != 0
        or admission["investment_objective_verified"] is not False
        or admission["expanded_pool_or_next_meta_rule_frozen"] is not False
    ):
        raise QuantError(
            "Use the unchanged first pool and the actual admitted feedback specialist."
        )
    base.checked_reference(policy["previous_three_expert_results"])
    admission_intent = read_json(base.checked_reference(admission["intent"]))
    for ref in admission_intent["inputs"]:
        base.checked_reference(ref)
    extra = ArchivedDenseExpert(
        ExpertIdentity(record["id"], record["mechanism_group"], record["introduced_at"]),
        source.execution.frame(record["targets"]),
        source.execution.frame(record["shadow"]),
        tuple(data.close.columns),
    )
    all_experts = (*experts, extra)
    if (
        tuple(e.identity.identifier for e in all_experts) != rules.EXPERT_IDS
        or tuple(e.identity.mechanism_group for e in all_experts) != rules.MECHANISMS
        or extra.observe(pd.Timestamp("2016-09-30"), 126).target is not None
    ):
        raise QuantError("Do not inflate mechanisms or backdate the specialist's availability.")
    for expert in all_experts:
        initial = expert.observe(pd.Timestamp(policy["initial_decision_session"]), 126)
        observed = expert.observe(data.close.index[-1], 2512)
        if (
            initial.target is None
            or initial.score_mature
            or len(initial.net_returns)
            or not observed.risk_free.equals(data.risk_free.loc[observed.risk_free.index])
        ):
            raise QuantError(
                "Keep real initial availability, empty initial scores and original RF."
            )
    admission_policy = read_json(base.checked_reference(record["admission_policy"]))
    legacy_packet = read_json(base.checked_reference(admission_policy["source_registration"]))
    legacy = next(c["spec"] for c in legacy_packet["candidates"] if c["spec"]["id"] == record["id"])
    legacy_code = [
        {"path": p, "sha256": digest}
        for p, digest in legacy["frozen_files"].items()
        if p.startswith("src/")
    ]
    legacy_policies = [
        {"path": p, "sha256": digest}
        for p, digest in legacy["frozen_files"].items()
        if p.startswith("config/")
    ]
    clocks = [
        {
            "id": r["id"],
            "source_policy_registered_at": r["introduced_at"],
            "actual_expert_admitted_at": public_pool["admitted_at"],
        }
        for r in pool["experts"]
    ] + [
        {
            "id": record["id"],
            "source_policy_registered_at": record["source_policy_registered_at"],
            "actual_expert_admitted_at": record["actual_expert_admitted_at"],
        }
    ]
    refs = source.unique_refs(
        [
            *admission_intent["inputs"],
            policy["first_pool_admission"],
            FIRST_PUBLIC,
            policy["feedback_expert_admission"],
            policy["source_scoring_policy"],
            policy["previous_three_expert_results"],
            admission["intent"],
            record["targets"],
            record["shadow"],
            record["admission_policy"],
        ]
    )
    return (
        data,
        all_experts,
        parent,
        original,
        benchmarks,
        legacy_code,
        legacy_policies,
        refs,
        clocks,
    )


def validation_references():
    path = FOLDER / "offline-tests-v1.xml"
    suites = ElementTree.parse(path).getroot().findall(".//testsuite")
    names = {
        case.get("name", "").split("[", 1)[0]
        for suite in suites
        for case in suite.findall("testcase")
    }
    required = {
        "test_old_three_expert_scoring_is_unchanged_on_synthetic_observations",
        "test_four_experts_produce_six_pairs_with_the_same_risk_and_score_law",
        "test_real_initial_anchor_and_prefix_causality_preserve_all_formal_days",
        "test_scalar_oracle_checks_the_actual_interface",
        "test_publication_checks_full_dependencies_before_the_explicit_packet",
        "test_original_monthly_execution_is_identical_for_valid_monthly_targets",
        "test_real_initial_target_is_not_backdated_to_a_month_end",
    }
    if (
        not required.issubset(names)
        or sum(int(s.get("tests", 0)) for s in suites) <= 0
        or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
    ):
        raise QuantError(
            "Require completed Interface, scalar-oracle and publication regression tests."
        )
    return [reference(path), reference(FOLDER / "offline-tests-v1.log")]


def initialize():
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        _, _, _, _, _, _, _, refs, clocks = source_bundle()
        events = base.journal_events(base.JOURNAL)
        counts = common.counts(events)
        if (
            len(events) != 374
            or events[-1]["sha256"] != rules.PRIOR_HEAD
            or counts
            != {
                "new_registered_candidates": 34,
                "new_completed_strategy_evaluations": 31,
                "new_reviews": 31,
                "new_technical_failures": 3,
                "new_pending_candidates": 0,
                "new_account_paths": 124,
                "new_historical_point_gate_passes": 0,
            }
        ):
            raise QuantError("Initialize only against the actual unchanged pre-study history.")
        files = [reference(ROOT / p) for p in OWN]
        tests = validation_references()
        body = {
            "created_at": utc_now(),
            "actual_continuation_event": 34875,
            "actual_delivery": "system_idle",
            "model_and_test_files": files,
            "validation_evidence": tests,
            "source_admission_references": refs,
            "source_policy_and_expert_admission_clocks": clocks,
            "question": "Fixed four-expert inclusion comparison; unchanged original ranking law.",
            "real_meta_scores_or_targets_computed": False,
            "candidate_accounts_or_metrics_computed": False,
            "new_shadow_account_runs": 0,
            "new_source_requests": 0,
            "investment_objective_verified": False,
            "order_authority": False,
        }
        common.retain_receipt(INTENT, body, ("created_at",))
        current.progress(
            "four_expert_rule_frozen_waiting_registration",
            active_compute_job=None,
            four_expert_research_intent=reference(INTENT),
            pending_steps=["register_four_expert_financial_rules_before_preparation"],
        )
        return reference(INTENT)


def inputs():
    data, experts, parent, original, benchmarks, code, policies, refs, clocks = source_bundle()
    policy, intent = read_json(rules.POLICY), read_json(INTENT)
    if (
        intent["model_and_test_files"] != [reference(ROOT / p) for p in OWN]
        or intent["validation_evidence"] != validation_references()
        or intent["source_admission_references"] != refs
        or intent["source_policy_and_expert_admission_clocks"] != clocks
        or intent["real_meta_scores_or_targets_computed"] is not False
    ):
        raise QuantError("The fixed rule, source clocks or code changed after its actual freeze.")
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": reference(rules.POLICY),
        "code": source.unique_refs(
            [
                *parent["code"],
                *code,
                *[reference(ROOT / p) for p in OWN[1:]],
                reference(ROOT / "src/us_quant/legacy_feedback_expert_v1.py"),
            ]
        ),
        "policies": source.unique_refs(
            [
                *parent["policies"],
                *policies,
                reference(rules.POLICY),
                read_json(base.checked_reference(policy["feedback_expert_admission"]))["expert"][
                    "admission_policy"
                ],
            ]
        ),
        "source_three_expert_pool": parent["expert_pool"],
        "expert_pool": reference(rules.POLICY),
        "expert_pool_definition_role": "fixed_four_members_and_source_admissions_in_policy",
        "source_policy_and_expert_admission_clocks": clocks,
        "prior_completed_configurations": 209,
        "prior_journal_sequence": 374,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "construction": "four_complete_experts_with_feedback_specialist_same_rolling_law",
        "execution_adapter": reference(
            ROOT / "src/us_quant/expert_initial_monthly_execution_v1.py"
        ),
        "execution_adapter_role": (
            "real_initial_seed_plus_unchanged_monthly_actual_holdings_feedback"
        ),
        "prior_evidence": source.unique_refs(
            [
                *parent["prior_evidence"],
                *refs,
                reference(INTENT),
            ]
        ),
        "validation_evidence": source.unique_refs(
            [
                *parent["validation_evidence"],
                *validation_references(),
            ]
        ),
        "new_shadow_account_runs": 0,
        "new_shadow_recommendation_artifacts": 0,
        "new_source_requests": 0,
    }
    check_frozen(frozen)
    return data, experts, frozen, original, benchmarks


def check_frozen(frozen):
    base.check_frozen_files(frozen)
    for ref in frozen["prior_evidence"] + frozen["validation_evidence"]:
        base.checked_reference(ref)


def verify_targets(data, experts, targets, audits):
    expected_dates = rules.decision_sessions(data.close.index, read_json(rules.POLICY))
    if (
        set(targets) != set(IDS)
        or [a["decision_session"] for a in audits] != expected_dates.strftime("%Y-%m-%d").tolist()
        or any(
            not frame.index.equals(data.close.index)
            or not frame.columns.equals(data.close.columns)
            or not frame.dropna().index.equals(expected_dates)
            or not frame.loc[~frame.index.isin(expected_dates)].isna().all().all()
            for frame in targets.values()
        )
    ):
        raise QuantError(
            "Keep the initial capital decision, every monthly target and full calendars."
        )
    previous = np.full(4, 0.25)
    maximum_target = maximum_weight = maximum_score = 0.0
    for position, (day, audit) in enumerate(zip(expected_dates, audits, strict=True)):
        observed = [e.observe(day, 126) for e in experts]
        if (
            audit["previous_rolling_expert_weights"] != previous.tolist()
            or audit["equal_expert_weights"] != np.full(4, 0.25).tolist()
            or audit["decision_role"]
            != ("initial_capital" if position == 0 else "completed_exchange_month_end")
            or tuple(o.identity.identifier for o in observed) != rules.EXPERT_IDS
            or any(
                o.target is None or o.target_known_at is None or o.target_known_at > day
                for o in observed
            )
            or audit["expert_target_known_at"] != [str(o.target_known_at.date()) for o in observed]
            or audit["preserved_archive_identity_introduced_at"]
            != [o.identity.introduced_at for o in observed]
            or audit["expert_shadow_last_observation"]
            != [
                str(o.net_returns.index[-1].date()) if len(o.net_returns) else None
                for o in observed
            ]
            or audit["shadow_wealth_added_to_main_capital"] is not False
            or audit["source_feedback_floor_transferred_to_main_account"] is not False
            or audit["counterfactual_historical_expert_availability"] is not True
            or audit["independent_forward_validation"] is not False
        ):
            raise QuantError("The expert state, source clocks or capital separation changed.")
        selection = audit["selection"]
        if not all(o.score_mature for o in observed):
            if (
                selection["scores_available"] is not False
                or selection["reason"] != "unscored_initial_equal_weights"
                or selection["pairs"] != []
                or selection["selected_pair"] is not None
                or selection["safety_or_success_claimed"] is not False
            ):
                raise QuantError("Unmatured source observations cannot receive successful scores.")
            chosen = np.full(4, 0.25)
        else:
            values = [(o.net_returns - o.risk_free).tolist() for o in observed]
            means = [math.fsum(v[:84]) / 84 for v in values]
            deviations = [
                math.sqrt(math.fsum((x - mean) ** 2 for x in v[:84]) / 83)
                for v, mean in zip(values, means, strict=True)
            ]
            pairs = {
                tuple(sorted((rules.EXPERT_IDS[i], rules.EXPERT_IDS[j])))
                for i, j in itertools.combinations(range(4), 2)
            }
            if (
                len(selection["pairs"]) != 6
                or {tuple(p["pair"]) for p in selection["pairs"]} != pairs
            ):
                raise QuantError("Retain all six scored or explicitly unscorable fixed pairs.")
            for pair in selection["pairs"]:
                i, j = [rules.EXPERT_IDS.index(name) for name in pair["pair"]]
                same = observed[i].identity.mechanism_group == observed[j].identity.mechanism_group
                reason = (
                    "same_mechanism_group"
                    if same
                    else "zero_training_variance"
                    if min(deviations[i], deviations[j]) <= 0
                    else None
                )
                weights = [0.0] * 4
                if reason is None:
                    weights[i] = deviations[j] / (deviations[i] + deviations[j])
                    weights[j] = deviations[i] / (deviations[i] + deviations[j])
                    portfolio = [
                        math.fsum(weights[k] * values[k][t] for k in range(4)) for t in range(126)
                    ]
                    if np.std(portfolio[:84], ddof=1) <= 0 or np.std(portfolio[84:], ddof=1) <= 0:
                        reason = "nonpositive_or_nonfinite_pair_risk"
                if reason is not None:
                    if pair.get("unscorable_reason") != reason:
                        raise QuantError("An unscorable pair differs from its independent reason.")
                    continue
                if "unscorable_reason" in pair:
                    raise QuantError("A valid pair was silently excluded.")
                actual_weights = np.asarray(pair["expert_weights"], dtype=float)
                contributions = np.asarray(pair["training_marginal_variance_contributions"])
                if (
                    actual_weights.shape != (4,)
                    or contributions.shape != (4,)
                    or not np.isfinite(actual_weights).all()
                    or not np.isfinite(contributions).all()
                    or (actual_weights < 0).any()
                    or abs(float(actual_weights.sum()) - 1) > 1e-12
                    or not np.isfinite(
                        [
                            pair[k]
                            for k in (
                                "train_net_risk_score",
                                "validation_net_risk_score",
                                "selection_score",
                                "selection_score_key",
                                "training_correlation",
                                "estimated_incremental_switch_cost_fraction",
                            )
                        ]
                    ).all()
                ):
                    raise QuantError("A stored pair cannot contain invalid risk weights or scores.")
                maximum_weight = max(
                    maximum_weight,
                    max(abs(a - b) for a, b in zip(weights, pair["expert_weights"], strict=True)),
                )
                covariance = (
                    math.fsum(
                        (values[i][t] - means[i]) * (values[j][t] - means[j]) for t in range(84)
                    )
                    / 83
                )
                expected_contribution = np.zeros(4)
                expected_contribution[i] = weights[i] * (
                    weights[i] * deviations[i] ** 2 + weights[j] * covariance
                )
                expected_contribution[j] = weights[j] * (
                    weights[j] * deviations[j] ** 2 + weights[i] * covariance
                )
                if (
                    abs(pair["training_correlation"] - covariance / (deviations[i] * deviations[j]))
                    > 1e-10
                    or float(abs(contributions - expected_contribution).max()) > 1e-12
                ):
                    raise QuantError(
                        "Recorded dependence or marginal risk differs from known training data."
                    )
                change = [
                    abs(
                        math.fsum(
                            (pair["expert_weights"][k] - previous[k])
                            * float(observed[k].target[name])
                            for k in range(4)
                        )
                    )
                    for name in data.close.columns
                ]
                cost = math.fsum(change) * 5 / 10000 + sum(v > 1e-12 for v in change) / 10000
                scores = [
                    source.scalar_score(portfolio[:84], cost),
                    source.scalar_score(portfolio[84:], cost),
                ]
                maximum_score = max(
                    maximum_score,
                    abs(scores[0] - pair["train_net_risk_score"]),
                    abs(scores[1] - pair["validation_net_risk_score"]),
                )
                if (
                    abs(cost - pair["estimated_incremental_switch_cost_fraction"]) > 1e-12
                    or pair["estimated_incremental_changed_assets"]
                    != sum(v > 1e-12 for v in change)
                    or pair["selection_score"]
                    != min(pair["train_net_risk_score"], pair["validation_net_risk_score"])
                    or pair["selection_score_key"] != round(pair["selection_score"], 12)
                    or pair["cost_proxy_is_actual_main_account_cost"] is not False
                ):
                    raise QuantError(
                        "The original cost proxy or deterministic selection key changed."
                    )
            scored = [p for p in selection["pairs"] if "selection_score_key" in p]
            if scored:
                best = sorted(scored, key=lambda p: (-p["selection_score_key"], p["pair"]))[0]
                dates = observed[0].net_returns.index
                if (
                    selection["selected_pair"] != best["pair"]
                    or selection["scores_available"] is not True
                    or selection["train_end"] != str(dates[83].date())
                    or selection["validation_start"] != str(dates[84].date())
                    or selection["validation_end"] != str(dates[-1].date())
                    or dates[-1] > day
                ):
                    raise QuantError(
                        "The selected pair or mature chronological score clocks changed."
                    )
                chosen = np.asarray(best["expert_weights"])
            else:
                if (
                    selection["reason"] != "unscored_degenerate_pair_equal_weights"
                    or selection["scores_available"]
                ):
                    raise QuantError("A degenerate unranked state cannot be hidden.")
                chosen = np.full(4, 0.25)
        if not np.allclose(chosen, audit["rolling_expert_weights"], rtol=0, atol=1e-12):
            raise QuantError("Issued expert weights differ from their actual selection.")
        for name, weights in ((IDS[0], np.full(4, 0.25)), (IDS[1], chosen)):
            expected = np.array(
                [
                    math.fsum(float(o.target[symbol]) * weights[i] for i, o in enumerate(observed))
                    for symbol in data.close.columns
                ]
            )
            maximum_target = max(
                maximum_target, float(np.max(abs(targets[name].loc[day] - expected)))
            )
        previous = chosen
    if maximum_target > 1e-10 or maximum_weight > 1e-10 or maximum_score > 1e-8:
        raise QuantError("Independent scalar composition, risk weights or net scores disagree.")
    return {
        "maximum_independent_target_difference": maximum_target,
        "maximum_independent_expert_weight_difference": maximum_weight,
        "maximum_independent_net_score_difference": maximum_score,
    }


def prepare(data, experts, frozen):
    if PREPARED.exists():
        saved = read_json(PREPARED)
        if saved["frozen_specification_sha256"] != digest_json(frozen):
            raise QuantError("Preserve a conflicting completed target bundle.")
        for row in saved["entries"].values():
            for ref in row.values():
                base.checked_reference(ref)
        return saved["entries"]
    attempt = FOLDER / "preparation-attempt.json"
    if attempt.exists():
        raise QuantError("Preserve the unresolved preparation instead of repeating it.")
    common.retain_receipt(
        attempt,
        {
            "started_at": utc_now(),
            "frozen_specification_sha256": digest_json(frozen),
        },
    )
    policy = read_json(rules.POLICY)
    targets, audits = rules.build_targets(
        experts,
        data.close.index,
        tuple(data.close.columns),
        policy,
        before_decision=runtime_guard,
    )
    independent = verify_targets(data, experts, targets, audits)
    dates = targets[IDS[0]].dropna().index
    if len(dates) != 121:
        raise QuantError("Preserve all121initial and monthly expert decisions.")
    prefixes = []
    for position in (0, 6, 12, 60, len(dates) - 1):
        end = dates[position]
        early, earlier = rules.build_targets(
            experts,
            data.close.index[data.close.index <= end],
            tuple(data.close.columns),
            policy,
            before_decision=runtime_guard,
        )
        if (
            any(not early[n].equals(targets[n].loc[:end]) for n in IDS)
            or earlier != audits[: position + 1]
        ):
            raise QuantError("Future inputs changed a prior expert composition.")
        prefixes.append(str(end.date()))
    common.retain_receipt(FOLDER / "information-audits.json", {"decisions": audits})
    entries = {}
    for name in IDS:
        coverage = []
        for window in read_json(base.PROTOCOL)["windows"]:
            anchor = previous_session(window["start"])
            known = targets[name].ffill().loc[anchor : window["end"]]
            actual = [s for s in source.donor.rules.CORE if (known[s] > 1e-10).any()]
            if anchor not in known.index or known.isna().any().any() or len(actual) < 3:
                raise QuantError(
                    "Every formal anchor and actual registered factor family is required."
                )
            coverage.append({"years": window["years"], "actual_factor_symbols": actual})
        directory = TARGETS / name
        path = directory / "targets.csv"
        content = targets[name].to_csv(float_format="%.17g")
        common.output_guard(path)
        if path.exists():
            if file_digest(path) != digest_bytes(content.encode()):
                raise QuantError("Preserve conflicting partial targets.")
        else:
            write_text_atomic(path, content)
        proof = {
            "checked_at": utc_now(),
            "decision_count": len(dates),
            "first_common_target_session": str(dates[0].date()),
            "initial_capital_decision_is_not_a_fabricated_month_end": True,
            "unscored_initial_equal_weight_decisions": sum(
                a["selection"]["reason"] == "unscored_initial_equal_weights" for a in audits
            ),
            "unscored_degenerate_pair_decisions": sum(
                a["selection"]["reason"] == "unscored_degenerate_pair_equal_weights" for a in audits
            ),
            "scored_decisions": sum(a["selection"]["scores_available"] for a in audits),
            "information_audits": reference(FOLDER / "information-audits.json"),
            **independent,
            "prefix_truncations": prefixes,
            "formal_factor_coverage": coverage,
            "source_policy_and_expert_admission_clocks": frozen[
                "source_policy_and_expert_admission_clocks"
            ],
            "new_shadow_account_runs": 0,
            "new_shadow_recommendation_artifacts": 0,
            "main_capital_usd_per_formal_path": 10000,
            "candidate_accounts_or_metrics_computed": False,
            "candidate_metrics_scope": (
                "formal fresh-capital main paths, not internal mature shadow scores"
            ),
            "matured_shadow_score_statistics_computed": True,
            "original_source_signal_recomputed": False,
            "independent_forward_validation": False,
        }
        common.retain_receipt(directory / "proof.json", proof)
        entries[name] = {"targets": reference(path), "proof": reference(directory / "proof.json")}
    check_frozen(frozen)
    common.retain_receipt(
        PREPARED,
        {
            "frozen_specification_sha256": digest_json(frozen),
            "entries": entries,
        },
    )
    return entries


def checked_packet(frozen):
    packet = read_json(PACKET)
    entries = [common.packet_entry(c, frozen) for c in rules.CANDIDATES]
    created = pd.Timestamp(packet["created_at"])
    if (
        packet["schema_version"] != 2
        or packet["program_id"] != "ssd_continuous_research_v1"
        or packet["study_id"] != frozen["study_id"]
        or packet["entries"] != entries
        or packet["new_registered_candidates"] != 2
        or created.tzinfo is None
        or pd.isna(created)
        or created > pd.Timestamp(utc_now())
        or any(pd.Timestamp(e["target_proof_at"]) > created for e in entries)
        or any(
            created > pd.Timestamp(e["at"]) for n in IDS for e in matching_events("path_started", n)
        )
        or any(
            packet.get(k) is not False
            for k in (
                "candidate_accounts_or_metrics_computed",
                "investment_objective_verified",
                "order_authority",
                "independent_forward_validation",
            )
        )
    ):
        raise QuantError("The actual complete pre-account registration packet changed.")
    return packet


def register():
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, experts, frozen, _, _ = inputs()
        events = base.journal_events(base.JOURNAL)
        counts = common.counts([e for e in events if e["candidate_id"] not in IDS])
        if (
            counts["new_registered_candidates"] != 34
            or counts["new_completed_strategy_evaluations"] != 31
            or counts["new_technical_failures"] != 3
            or counts["new_pending_candidates"] != 0
            or events[373]["sha256"] != rules.PRIOR_HEAD
        ):
            raise QuantError("Preserve all original completed and failed financial identities.")
        for candidate in rules.CANDIDATES:
            specification = {"candidate": candidate, "frozen": frozen}
            found = matching_events("registered", candidate["id"])
            if found and (len(found) != 1 or found[0]["body"] != specification):
                raise QuantError("A registered policy or input cannot change.")
            if not found:
                runtime_guard()
                append_event(base.JOURNAL, "registered", candidate["id"], specification)
        if any(matching_events("technical_failure", n) for n in IDS):
            raise QuantError("A terminal preparation version cannot be silently retried.")
        pending = [n for n in IDS if not matching_events("target_proof", n)]
        if pending:
            current.progress(
                "preparing_registered_four_expert_meta",
                active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            )
            try:
                prepared = prepare(data, experts, frozen)
                for name in pending:
                    runtime_guard()
                    append_event(
                        base.JOURNAL,
                        "target_proof",
                        name,
                        {
                            "registered_event_sha256": matching_events("registered", name)[0][
                                "sha256"
                            ],
                            **prepared[name],
                        },
                    )
            except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
                source.failure("four_expert_meta_preparation", pending, error)
                raise
            except KeyboardInterrupt:
                current.progress(
                    "four_expert_preparation_interrupted_preserved", active_compute_job=None
                )
                raise
        common.retain_receipt(
            PACKET,
            {
                "schema_version": 2,
                "program_id": "ssd_continuous_research_v1",
                "study_id": frozen["study_id"],
                "created_at": utc_now(),
                "entries": [common.packet_entry(c, frozen) for c in rules.CANDIDATES],
                "new_registered_candidates": 2,
                "new_economic_hypotheses": 2,
                "candidate_accounts_or_metrics_computed": False,
                "candidate_metrics_scope": "formal main paths, not internal mature shadow scores",
                "raw_inputs_and_target_matrices_remain_private": True,
                "investment_objective_verified": False,
                "order_authority": False,
                "independent_forward_validation": False,
            },
            ("created_at",),
        )
        checked_packet(frozen)
        current.progress(
            "four_expert_registered_waiting_publication",
            active_compute_job=None,
            four_expert_registration=reference(PACKET),
            pending_steps=["publish_actual_four_expert_registration", "eight_new_main_paths"],
        )
        return reference(PACKET)


def publication(frozen, commit):
    dependencies = published_inputs(frozen, commit)
    contract = read_json(base.REPO / "config/publication_allowlist.json")
    explicit = {
        r["path"]
        for r in frozen["code"] + frozen["policies"]
        if f"research/workbench/{r['path']}" in contract["files"]
    }
    own = verify_publication(ROOT, reference(PACKET), explicit, commit)
    found = [
        e for e in matching_events("publication") if e["body"].get("study_id") == frozen["study_id"]
    ]
    if found:
        if len(found) != 1 or found[0]["body"]["publication"]["published_commit"] != commit:
            raise QuantError("Resume the actual original publication, not a later patch.")
        return found[0]["body"]["publication"]
    receipt = {**own, "complete_dependency_publication": dependencies}
    append_event(
        base.JOURNAL,
        "publication",
        None,
        {
            "study_id": frozen["study_id"],
            "publication": receipt,
        },
    )
    return receipt


def states(frozen):
    result = {}
    for name in IDS:
        reviews, failed = (
            matching_events("review", name),
            matching_events("technical_failure", name),
        )
        paths = {
            e["body"]["path_key"]: base.verify_path(
                e["body"]["receipt"], frozen, name, e["body"]["path_key"]
            )
            for e in matching_events("path_completed", name)
        }
        if reviews:
            if len(reviews) != 1 or failed or reviews[0]["body"]["paths"] != paths:
                raise QuantError("The completed review disagrees with actual path receipts.")
            result[name] = reviews[0]["body"]
        else:
            result[name] = {
                "candidate_id": name,
                "status": "technical_failure" if failed else "pending",
                "paths": paths,
                "failure_event_sha256": failed[0]["sha256"] if failed else None,
                "failure_stage": failed[0]["body"]["stage"] if failed else None,
            }
    return result


def actual_factor_participation(paths, frozen):
    factor_records = frozen["eligible_actual_registered_factors"]
    symbols = source.donor.rules.CORE
    family = dict(zip(symbols, [r["family"] for r in factor_records], strict=True))
    result = {}
    for key, path in paths.items():
        refs = [r for r in path["artifacts"] if r["path"].endswith("/weights.csv")]
        if len(refs) != 1:
            raise QuantError("An actual path must retain exactly one recorded asset-weight table.")
        weights = source.execution.frame(refs[0])
        active = [s for s in symbols if (weights[s] > 1e-10).any()]
        families = sorted({family[s] for s in active})
        result[key] = {
            "actual_factor_symbols": active,
            "actual_registered_families": families,
            "minimum_three_registered_families_pass": len(families) >= 3,
        }
    return result


def evaluate(commit):
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, experts, frozen, _, benchmarks = inputs()
        checked_packet(frozen)
        if RESULTS.exists():
            saved = read_json(RESULTS)
            if saved["frozen_specification_sha256"] != digest_json(frozen) or saved[
                "candidates"
            ] != states(frozen):
                raise QuantError("Preserve completed results instead of replaying them.")
            return reference(RESULTS)
        active = IDS
        try:
            published = publication(frozen, commit)
            current.progress(
                "evaluating_four_expert_meta",
                active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            )
            targets = {
                n: source.execution.frame(matching_events("target_proof", n)[0]["body"]["targets"])
                for n in IDS
            }
            proof = read_json(
                base.checked_reference(matching_events("target_proof", IDS[0])[0]["body"]["proof"])
            )
            audits = read_json(base.checked_reference(proof["information_audits"]))["decisions"]
            verify_targets(data, experts, targets, audits)
            protocol = read_json(base.PROTOCOL)
            for name in IDS:
                active = (name,)
                if matching_events("review", name) or matching_events("technical_failure", name):
                    continue
                paths = {}
                for window in protocol["windows"]:
                    for scenario in protocol["accounts"]["scenarios"]:
                        runtime_guard()
                        check_frozen(frozen)
                        key = f"{window['years']}y-{scenario['id']}"
                        current.progress(
                            "evaluating_four_expert_meta", active_candidate=name, active_path=key
                        )
                        paths[key] = execution.evaluate_path(
                            data, targets[name], frozen, name, window, scenario, benchmarks
                        )
                participation = actual_factor_participation(paths, frozen)
                passed = all(p["all_point_gates_pass"] for p in paths.values()) and all(
                    p["minimum_three_registered_families_pass"] for p in participation.values()
                )
                append_event(
                    base.JOURNAL,
                    "review",
                    name,
                    {
                        "candidate_id": name,
                        "reviewed_at": utc_now(),
                        "status": "historical_point_gates_passed" if passed else "rejected",
                        "paths": paths,
                        "actual_factor_participation": participation,
                        "all_four_paths_same_candidate_and_frozen_parameters": True,
                        "full_search_selection_adjusted_alpha_verified": False,
                        "independent_forward_validation": False,
                        "investment_objective_verified": False,
                        "automatic_live_deployment": False,
                        "order_authority": False,
                    },
                )
            events = base.journal_events(base.JOURNAL)
            common.retain_receipt(
                RESULTS,
                {
                    "schema_version": 2,
                    "study_id": frozen["study_id"],
                    "finished_at": utc_now(),
                    "frozen_specification_sha256": digest_json(frozen),
                    "candidates": states(frozen),
                    "publication": published,
                    "legacy": frozen["legacy"],
                    "actual_journal_event_count": len(events),
                    "actual_journal_head_sha256": events[-1]["sha256"],
                    "global_counts_at_completion": common.counts(events),
                    "new_shadow_account_runs": 0,
                    "new_shadow_recommendation_artifacts": 0,
                    "investment_objective_verified": False,
                    "order_authority": False,
                },
                ("finished_at",),
            )
            current.progress(
                "four_expert_meta_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                four_expert_results=reference(RESULTS),
                pending_steps=["publish_all_four_expert_results"],
            )
            return reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            source.failure("four_expert_meta_evaluation", active, error)
            raise
        except KeyboardInterrupt:
            current.progress(
                "four_expert_meta_interrupted_preserved",
                active_compute_job=None,
                active_candidate=None,
            )
            raise


def public_result():
    base.storage_guard()
    _, _, frozen, _, benchmarks = inputs()
    checked_packet(frozen)
    saved, actual = read_json(RESULTS), states(frozen)
    if saved["candidates"] != actual or any(v["status"] == "pending" for v in actual.values()):
        raise QuantError("Only complete actual outcomes or retained failures may be published.")
    lookup = {
        (r["years"], r["scenario"], r["symbol"]): r["metrics"]["cagr"]
        for r in benchmarks["benchmarks"]
    }
    reviews = []
    for name, row in actual.items():
        paths = []
        for key, path in row["paths"].items():
            year, scenario = key.split("-")
            years = int(year[:-1])
            paths.append(
                {
                    **{
                        k: path[k]
                        for k in (
                            "path_key",
                            "metrics",
                            "execution",
                            "gates",
                            "all_point_gates_pass",
                            "excess_net_cagr_vs_spy",
                            "excess_net_cagr_vs_qqq",
                            "stress_vs_base_cost_qqq_diagnostic",
                            "independent_bt_max_equity_difference_usd",
                            "independent_monthly_feedback",
                        )
                    },
                    "years": years,
                    "scenario": scenario,
                    "benchmark_net_cagr": {s: lookup[years, scenario, s] for s in ("SPY", "QQQ")},
                    "actual_factor_participation": row.get("actual_factor_participation", {}).get(
                        key
                    ),
                }
            )
        reviews.append(
            {
                "candidate_id": name,
                "status": row["status"],
                "paths": paths,
                "reviewed_at": row.get("reviewed_at"),
                "failure_event_sha256": row.get("failure_event_sha256"),
                "failure_stage": row.get("failure_stage"),
                "all_four_paths_same_candidate_and_frozen_parameters": row.get(
                    "all_four_paths_same_candidate_and_frozen_parameters", False
                ),
                "independent_forward_validation": False,
                "order_authority": False,
            }
        )
    counts = common.counts(
        [e for e in base.journal_events(base.JOURNAL) if e["candidate_id"] in IDS]
    )
    common.retain_receipt(
        PUBLIC_RESULTS,
        {
            "schema_version": 2,
            "study_id": frozen["study_id"],
            "created_at": utc_now(),
            "actual_completed_at": saved["finished_at"],
            "actual_journal_event_count": saved["actual_journal_event_count"],
            "actual_journal_head_sha256": saved["actual_journal_head_sha256"],
            "actual_publication_before_accounts": saved["publication"],
            "registration_packet": reference(PACKET),
            "protocol": frozen["protocol"],
            "private_complete_results": reference(RESULTS),
            "reviews": reviews,
            **counts,
            "new_completed_configurations": counts["new_completed_strategy_evaluations"],
            "global_counts_at_completion": saved["global_counts_at_completion"],
            "source_experts_or_old_records_reclassified": False,
            "raw_prices_targets_or_daily_accounts_exported": False,
            "new_shadow_account_runs": 0,
            "new_shadow_recommendation_artifacts": 0,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "order_authority": False,
            "automatic_live_deployment": False,
        },
        ("created_at",),
    )
    return reference(PUBLIC_RESULTS)


def main():
    parser = argparse.ArgumentParser(description="Fixed four-expert composition; no trading.")
    parser.add_argument(
        "command", choices=("init", "register", "evaluate", "publish-result", "status")
    )
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("Four-expert research cannot request live sources.")

    sys.addaudithook(offline)
    try:
        if args.command == "init":
            result = initialize()
        elif args.command == "register":
            result = register()
        elif args.command == "evaluate":
            if args.commit is None:
                raise QuantError("Provide the actual original preregistration publication commit.")
            result = evaluate(args.commit)
        elif args.command == "publish-result":
            result = public_result()
        else:
            result = current.current_counts(base.journal_events(base.JOURNAL))
        print(json.dumps(result, indent=2, allow_nan=False))
    except QuantError as error:
        parser.exit(2, f"Four-expert research blocked: {error}\n")


if __name__ == "__main__":
    main()
