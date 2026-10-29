from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_current_progress_v1 as current
from us_quant import ssd_joint_tail_research_v1 as donor
from us_quant import ssd_research_v1 as base
from us_quant import ssd_variance_disagreement_research_v1 as execution
from us_quant import strategy_expert_meta_v1 as rules
from us_quant.calendar import is_month_end, previous_session
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import DEADLINE, research_job, runtime_guard
from us_quant.research_weekly import verify_publication
from us_quant.storage import (
    digest_bytes,
    digest_json,
    file_digest,
    read_json,
    utc_now,
    write_text_atomic,
)
from us_quant.strategy_experts_v1 import load_expert

ROOT = base.ROOT
FOLDER = base.REPORTS / "strategy-expert-meta-20261029-v1"
TRANSITION = base.REPORTS / "expert-pool-transition-20261029-v1"
TARGETS = ROOT / "data/ssd-strategy-expert-meta-20261029-v1"
PACKET = ROOT / "evidence/strategy_expert_meta_20261029_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/strategy_expert_meta_20261029_results_v1.json"
PREPARED = FOLDER / "prepared-target-bundle.json"
IDS = tuple(c["id"] for c in rules.CANDIDATES)
OWN_EXTENSIONS = (
    "config/strategy-expert-meta-v1.json",
    "src/us_quant/strategy_experts_v1.py",
    "src/us_quant/strategy_expert_meta_v1.py",
    "src/us_quant/ssd_current_progress_v1.py",
    "src/us_quant/ssd_expert_meta_research_v1.py",
    "tests/test_strategy_experts_v1.py",
    "tests/test_strategy_expert_meta_v1.py",
    "tests/test_ssd_current_progress_v1.py",
    "tests/test_ssd_expert_meta_research_v1.py",
)
EXECUTION_FILES = (
    "src/us_quant/ssd_variance_disagreement_research_v1.py",
    "src/us_quant/daily_variance_disagreement_v1.py",
    "src/us_quant/observed_weight_execution.py",
    "config/observed-weight-execution.json",
)


def unique_refs(references):
    found = {}
    for ref in references:
        if ref["path"] in found and found[ref["path"]] != ref:
            raise QuantError("Shared expert dependencies have conflicting frozen identities.")
        found[ref["path"]] = ref
    return [found[key] for key in sorted(found)]


def check_frozen(frozen):
    base.check_frozen_files(frozen)
    for ref in frozen["prior_evidence"] + frozen["validation_evidence"]:
        base.checked_reference(ref)


def inputs():
    events = base.journal_events(base.JOURNAL)
    if len(events) < 344 or events[343]["sha256"] != rules.PRIOR_HEAD:
        raise QuantError("Preserve the actual pre-meta history, including every failed candidate.")
    data, parent, original, benchmarks = donor.inputs()
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    pool = read_json(base.checked_reference(policy["pool"]))
    universe = tuple(data.close.columns)
    if (
        tuple(r["id"] for r in pool["experts"]) != rules.EXPERT_IDS
        or tuple(pool["universe"]) != universe
        or pool["market"] != parent["market"]
        or len({r["mechanism_group"] for r in pool["experts"]}) != 3
        or pool["new_strategy_trials"] != 0
    ):
        raise QuantError(
            "Use the actual fixed admitted expert pool and shared fourteen-asset market."
        )
    experts, source_refs, source_code, source_policies = [], [], [], []
    for record in pool["experts"]:
        expert, checked = load_expert(record, universe)
        admitted = next(r for r in pool["admission_checks"] if r["id"] == record["id"])
        if any(admitted[k] != value for k, value in checked.items()):
            raise QuantError("An admitted expert's evidence changed.")
        observation = expert.observe(data.close.index[-1], 2512)
        if not observation.risk_free.equals(data.risk_free.loc[observation.risk_free.index]):
            raise QuantError("Expert shadow observations must use the identical original RF.")
        experts.append(expert)
        packet = read_json(base.checked_reference(record["registration"]))
        entry = next(r for r in packet["entries"] if r["candidate_id"] == record["id"])
        frozen = entry["specification"]["frozen"]
        source_code.extend(frozen["code"])
        source_policies.extend(frozen["policies"])
        source_refs.extend(
            [
                record["registration"],
                record["results"],
                entry["target_proof"]["targets"],
                entry["target_proof"]["proof"],
                *frozen["volatility_sources"],
                *frozen["benchmark_artifacts"],
                *frozen["market"].values(),
                *frozen.get("prior_evidence", []),
                *frozen.get("validation_evidence", []),
            ]
        )
        for ref in record["completed_paths"].values():
            complete = read_json(base.checked_reference(ref))
            source_refs.extend([ref, *complete["artifacts"]])
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": base.reference(rules.POLICY),
        "code": unique_refs(
            [
                *parent["code"],
                *source_code,
                *[base.reference(ROOT / p) for p in (*OWN_EXTENSIONS[1:], *EXECUTION_FILES[:3])],
            ]
        ),
        "policies": unique_refs(
            [
                *parent["policies"],
                *source_policies,
                base.reference(rules.POLICY),
                base.reference(execution.monthly.POLICY),
            ]
        ),
        "execution_policy": base.reference(execution.monthly.POLICY),
        "expert_pool": policy["pool"],
        "prior_completed_configurations": 207,
        "prior_journal_sequence": 344,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "construction": (
            "complete_strategy_experts_with_separate_net_shadow_state_and_one_capital_account"
        ),
        "original_policy_role": "unchanged_monthly_actual_holdings_feedback_and_cost_timing",
        "eligible_actual_registered_factors": parent["used_actual_registered_factors"],
        "possible_economic_factor_families": 6,
        "minimum_economic_factor_families": 3,
        "actual_factor_participation_is_measured_in_target_proof": True,
        "prior_evidence": unique_refs(
            [
                *parent["prior_evidence"],
                *source_refs,
                policy["pool"],
                base.reference(ROOT / "STRATEGY_EXPERTS_V1.md"),
                base.reference(TRANSITION / "user-direction-native-event.json"),
                base.reference(TRANSITION / "automation-binding-confirmation.json"),
                base.reference(TRANSITION / "progress-summary-reconciliation.json"),
                base.reference(FOLDER / "research-intent.json"),
            ]
        ),
        "validation_evidence": [
            base.reference(FOLDER / name)
            for name in ("offline-tests-checked-v1.xml", "offline-tests-checked-v1.log")
        ],
        "shadow_accounts_replayed": False,
        "new_shadow_account_runs": 0,
        "new_source_requests": 0,
        "independent_forward_validation": False,
    }
    frozen.pop("used_actual_registered_factors")
    frozen.pop("used_economic_factor_families")
    check_frozen(frozen)
    execution.monthly.validate(read_json(execution.monthly.POLICY))
    return data, tuple(experts), frozen, original, benchmarks


def scalar_score(values, cost):
    mean = math.fsum(values) / len(values)
    variance = math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1)
    if variance <= 0:
        raise QuantError("A recorded scored pair cannot have zero independent variance.")
    return math.sqrt(252) * (mean - cost / 21) / math.sqrt(variance)


def verify_targets(data, experts, targets, audits):
    dates = targets[IDS[0]].dropna().index
    expected_dates = pd.DatetimeIndex(
        [day for day in data.close.index if day >= pd.Timestamp("2016-09-30") and is_month_end(day)]
    )
    if (
        dates.empty
        or str(dates[0].date()) != "2016-09-30"
        or len(dates) != 121
        or not dates.equals(expected_dates)
        or [a["decision_session"] for a in audits] != dates.strftime("%Y-%m-%d").tolist()
        or any(
            not t.index.equals(data.close.index)
            or not t.columns.equals(data.close.columns)
            or not t.dropna().index.equals(dates)
            or not t.loc[~t.index.isin(dates)].isna().all().all()
            for t in targets.values()
        )
    ):
        raise QuantError("Keep every common available month end and all original price sessions.")
    maximum_target = maximum_score = maximum_weight = 0.0
    previous = np.full(3, 1 / 3)
    for day, audit in zip(dates, audits, strict=True):
        observed = [e.observe(day, 126) for e in experts]
        if (
            audit["previous_rolling_expert_weights"] != previous.tolist()
            or audit["equal_expert_weights"] != np.full(3, 1 / 3).tolist()
            or any(o.target is None or o.target_known_at > day for o in observed)
            or audit["expert_target_known_at"] != [str(o.target_known_at.date()) for o in observed]
            or audit["expert_introduced_at"] != [o.identity.introduced_at for o in observed]
            or audit["expert_shadow_last_observation"]
            != [
                str(o.net_returns.index[-1].date()) if len(o.net_returns) else None
                for o in observed
            ]
            or audit["shadow_wealth_added_to_main_capital"] is not False
            or audit["independent_forward_validation"] is not False
        ):
            raise QuantError("Expert state or main-capital separation changed.")
        selection = audit["selection"]
        if not all(o.score_mature for o in observed):
            if (
                selection["reason"] != "unscored_initial_equal_weights"
                or selection["scores_available"]
            ):
                raise QuantError("Unmatured observations cannot receive a successful score.")
            chosen_weights = np.full(3, 1 / 3)
        else:
            dates_known = observed[0].net_returns.index
            values = [(o.net_returns - o.risk_free).tolist() for o in observed]
            expected_pairs = {
                tuple(sorted((rules.EXPERT_IDS[i], rules.EXPERT_IDS[j])))
                for i in range(3)
                for j in range(i + 1, 3)
            }
            if (
                len(selection["pairs"]) != 3
                or {tuple(p["pair"]) for p in selection["pairs"]} != expected_pairs
            ):
                raise QuantError(
                    "Every fixed pair must retain its score or explicit unscorable reason."
                )
            deviations = []
            for vector in values:
                mean = math.fsum(vector[:84]) / 84
                deviations.append(math.sqrt(math.fsum((v - mean) ** 2 for v in vector[:84]) / 83))
            for pair in selection["pairs"]:
                indices = [rules.EXPERT_IDS.index(name) for name in pair["pair"]]
                if "unscorable_reason" in pair:
                    reason = pair["unscorable_reason"]
                    if reason == "zero_training_variance":
                        if all(deviations[i] > 0 for i in indices):
                            raise QuantError("An observed nonzero risk pair was silently excluded.")
                    elif reason == "same_mechanism_group":
                        if (
                            observed[indices[0]].identity.mechanism_group
                            != observed[indices[1]].identity.mechanism_group
                        ):
                            raise QuantError(
                                "Distinct mechanisms cannot be mislabeled as duplicates."
                            )
                    elif reason == "nonpositive_or_nonfinite_pair_risk":
                        i, j = indices
                        w = [0.0, 0.0, 0.0]
                        w[i] = deviations[j] / (deviations[i] + deviations[j])
                        w[j] = 1 - w[i]
                        r = [math.fsum(w[k] * values[k][t] for k in range(3)) for t in range(126)]
                        if np.std(r[:84], ddof=1) > 0 and np.std(r[84:], ddof=1) > 0:
                            raise QuantError("A valid pair was excluded as degenerate risk.")
                    else:
                        raise QuantError(
                            "An unknown unscorable state is not an allocation fallback."
                        )
                    continue
                weights = pair["expert_weights"]
                i, j = indices
                independent_weights = [0.0, 0.0, 0.0]
                independent_weights[i] = deviations[j] / (deviations[i] + deviations[j])
                independent_weights[j] = deviations[i] / (deviations[i] + deviations[j])
                maximum_weight = max(
                    maximum_weight,
                    max(abs(a - b) for a, b in zip(weights, independent_weights, strict=True)),
                )
                change = [
                    abs(
                        math.fsum(
                            (weights[k] - previous[k]) * float(observed[k].target[name])
                            for k in range(3)
                        )
                    )
                    for name in data.close.columns
                ]
                cost = math.fsum(change) * 5 / 10000 + sum(v > 1e-12 for v in change) / 10000
                portfolio = [
                    math.fsum(weights[k] * values[k][t] for k in range(3)) for t in range(126)
                ]
                scores = [scalar_score(portfolio[:84], cost), scalar_score(portfolio[84:], cost)]
                maximum_score = max(
                    maximum_score,
                    abs(scores[0] - pair["train_net_risk_score"]),
                    abs(scores[1] - pair["validation_net_risk_score"]),
                )
                if (
                    abs(cost - pair["estimated_incremental_switch_cost_fraction"]) > 1e-12
                    or pair["selection_score"]
                    != min(pair["train_net_risk_score"], pair["validation_net_risk_score"])
                    or pair["selection_score_key"] != round(pair["selection_score"], 12)
                    or pair["cost_proxy_is_actual_main_account_cost"] is not False
                ):
                    raise QuantError("The pair's costs or deterministic score key changed.")
            scored = [p for p in selection["pairs"] if "selection_score_key" in p]
            if scored:
                best = sorted(scored, key=lambda p: (-p["selection_score_key"], p["pair"]))[0]
                if (
                    selection["selected_pair"] != best["pair"]
                    or selection["train_end"] != str(dates_known[83].date())
                    or selection["validation_start"] != str(dates_known[84].date())
                    or selection["validation_end"] != str(dates_known[-1].date())
                    or dates_known[-1] > day
                ):
                    raise QuantError("Selection or training/validation maturity changed.")
                chosen_weights = np.asarray(best["expert_weights"])
            else:
                if selection["reason"] != "unscored_degenerate_pair_equal_weights":
                    raise QuantError("A degenerate unscored state cannot be hidden.")
                chosen_weights = np.full(3, 1 / 3)
        if not np.allclose(chosen_weights, audit["rolling_expert_weights"], rtol=0, atol=1e-12):
            raise QuantError("The issued expert weights differ from their recorded selection.")
        for identifier, weights in ((IDS[0], np.full(3, 1 / 3)), (IDS[1], chosen_weights)):
            oracle = np.array(
                [
                    math.fsum(float(o.target[name]) * weights[i] for i, o in enumerate(observed))
                    for name in data.close.columns
                ]
            )
            maximum_target = max(
                maximum_target, float(np.max(abs(targets[identifier].loc[day] - oracle)))
            )
        previous = chosen_weights
    if maximum_target > 1e-10 or maximum_weight > 1e-10 or maximum_score > 1e-8:
        raise QuantError("Independent scalar composition, risk weights or net scoring disagrees.")
    return {
        "maximum_independent_target_difference": maximum_target,
        "maximum_independent_expert_weight_difference": maximum_weight,
        "maximum_independent_net_score_difference": maximum_score,
    }


def prepare(data, experts, frozen):
    if PREPARED.exists():
        bundle = read_json(PREPARED)
        if bundle["frozen_specification_sha256"] != digest_json(frozen):
            raise QuantError("Preserve a conflicting completed target bundle.")
        for entry in bundle["entries"].values():
            for ref in entry.values():
                base.checked_reference(ref)
        return bundle["entries"]
    attempt = FOLDER / "preparation-attempt.json"
    if attempt.exists():
        raise QuantError("An unresolved preparation attempt is preserved, not repeated.")
    common.retain_receipt(
        attempt, {"started_at": utc_now(), "frozen_specification_sha256": digest_json(frozen)}
    )
    targets, audits = rules.build_targets(
        experts,
        data.close.index,
        tuple(data.close.columns),
        read_json(rules.POLICY),
        before_decision=runtime_guard,
    )
    independent = verify_targets(data, experts, targets, audits)
    prefixes = []
    dates = targets[IDS[0]].dropna().index
    for position in (0, 6, 12, 60, len(dates) - 1):
        runtime_guard()
        end = dates[position]
        early, earlier_audits = rules.build_targets(
            experts,
            data.close.index[data.close.index <= end],
            tuple(data.close.columns),
            read_json(rules.POLICY),
            before_decision=runtime_guard,
        )
        if (
            any(not early[name].equals(targets[name].loc[:end]) for name in IDS)
            or earlier_audits != audits[: position + 1]
        ):
            raise QuantError("Future decisions changed an earlier expert composition.")
        prefixes.append(str(end.date()))
    common.retain_receipt(FOLDER / "information-audits.json", {"decisions": audits})
    entries = {}
    protocol = read_json(base.PROTOCOL)
    for identifier in IDS:
        target = targets[identifier]
        coverage = []
        for window in protocol["windows"]:
            anchor = previous_session(window["start"])
            known = target.ffill().loc[anchor : window["end"]]
            actual = [name for name in donor.rules.CORE if (known[name] > 1e-10).any()]
            if anchor not in known.index or known.isna().any().any() or len(actual) < 3:
                raise QuantError(
                    "Keep complete formal anchors and genuinely participating factor families."
                )
            coverage.append({"years": window["years"], "actual_factor_symbols": actual})
        directory = TARGETS / identifier
        path = directory / "targets.csv"
        content = target.to_csv(float_format="%.17g")
        common.output_guard(path)
        if path.exists():
            if file_digest(path) != digest_bytes(content.encode()):
                raise QuantError("Preserve conflicting partial expert targets.")
        else:
            write_text_atomic(path, content)
        proof = {
            "checked_at": utc_now(),
            "monthly_decisions": len(dates),
            "first_common_target_session": str(dates[0].date()),
            "unscored_initial_equal_weight_decisions": sum(
                a["selection"]["reason"] == "unscored_initial_equal_weights" for a in audits
            ),
            "unscored_degenerate_pair_decisions": sum(
                a["selection"]["reason"] == "unscored_degenerate_pair_equal_weights" for a in audits
            ),
            "scored_decisions": sum(a["selection"]["scores_available"] for a in audits),
            "information_audits": base.reference(FOLDER / "information-audits.json"),
            **independent,
            "prefix_truncations": prefixes,
            "formal_factor_coverage": coverage,
            "shadow_runs_replayed_or_newly_capitalized": 0,
            "main_capital_usd_per_formal_path": 10000,
            "candidate_accounts_or_metrics_computed": False,
            "candidate_metrics_scope": (
                "formal_fresh_capital_meta_paths_not_internal_shadow_training_scores"
            ),
            "matured_shadow_score_statistics_computed": True,
            "original_source_signal_recomputed": False,
            "independent_forward_validation": False,
        }
        common.retain_receipt(directory / "proof.json", proof)
        entries[identifier] = {
            "targets": base.reference(path),
            "proof": base.reference(directory / "proof.json"),
        }
    check_frozen(frozen)
    common.retain_receipt(
        PREPARED, {"frozen_specification_sha256": digest_json(frozen), "entries": entries}
    )
    return entries


def checked_packet(frozen):
    packet = read_json(PACKET)
    entries = [common.packet_entry(candidate, frozen) for candidate in rules.CANDIDATES]
    created = pd.Timestamp(packet["created_at"])
    if (
        packet["schema_version"] != 2
        or packet["program_id"] != "ssd_continuous_research_v1"
        or packet["study_id"] != frozen["study_id"]
        or packet["entries"] != entries
        or packet["new_registered_candidates"] != 2
        or any(
            packet.get(k) is not False
            for k in (
                "candidate_accounts_or_metrics_computed",
                "investment_objective_verified",
                "order_authority",
                "independent_forward_validation",
            )
        )
        or created.tzinfo is None
        or created > pd.Timestamp(utc_now())
        or any(pd.Timestamp(e["target_proof_at"]) > created for e in entries)
        or any(
            created > pd.Timestamp(e["at"])
            for name in IDS
            for e in base.matching_events("path_started", name)
        )
    ):
        raise QuantError("Require the actual pre-account meta-strategy packet and original clocks.")
    audit_refs = []
    for entry in entries:
        proof = read_json(base.checked_reference(entry["target_proof"]["proof"]))
        reference = proof["information_audits"]
        base.checked_reference(reference)
        audit_refs.append(reference)
    if any(ref != audit_refs[0] for ref in audit_refs):
        raise QuantError("Both meta candidates must retain the same frozen decision audit.")
    return packet


def failure(stage, identifiers, error):
    expired = pd.Timestamp(utc_now()) >= pd.Timestamp(DEADLINE)
    for identifier in identifiers:
        if (
            not expired
            and base.matching_events("registered", identifier)
            and not (
                base.matching_events("review", identifier)
                or base.matching_events("technical_failure", identifier)
            )
        ):
            base.append_event(
                base.JOURNAL,
                "technical_failure",
                identifier,
                {
                    "stage": stage,
                    "reason": str(error),
                    "partial_outputs_retained": True,
                    "candidate_accounts_started": bool(
                        base.matching_events("path_started", identifier)
                    ),
                },
            )
    current.progress(
        "expert_meta_expired_preserved" if expired else "expert_meta_blocked",
        active_compute_job=None,
        active_candidate=None,
        current_failure=str(error),
    )


def register():
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, experts, frozen, _, _ = inputs()
        events = base.journal_events(base.JOURNAL)
        prior = common.counts([e for e in events if e["candidate_id"] not in IDS])
        if (
            prior["new_registered_candidates"] != 30
            or prior["new_completed_strategy_evaluations"] != 29
            or prior["new_technical_failures"] != 1
            or prior["new_pending_candidates"] != 0
            or events[0]["body"]["legacy"] != frozen["legacy"]
        ):
            raise QuantError("Do not reset or reinterpret the preserved pre-meta trial history.")
        for candidate in rules.CANDIDATES:
            specification = {"candidate": candidate, "frozen": frozen}
            found = base.matching_events("registered", candidate["id"])
            if found and (len(found) != 1 or found[0]["body"] != specification):
                raise QuantError("A registered meta-strategy cannot be edited.")
            if not found:
                runtime_guard()
                base.append_event(base.JOURNAL, "registered", candidate["id"], specification)
        if any(base.matching_events("technical_failure", name) for name in IDS):
            raise QuantError(
                "Preserve terminal meta-preparation failures without same-version replay."
            )
        pending = [name for name in IDS if not base.matching_events("target_proof", name)]
        if pending:
            current.progress(
                "preparing_registered_expert_meta",
                active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            )
            try:
                prepared = prepare(data, experts, frozen)
                for name in pending:
                    runtime_guard()
                    base.append_event(
                        base.JOURNAL,
                        "target_proof",
                        name,
                        {
                            "registered_event_sha256": base.matching_events("registered", name)[0][
                                "sha256"
                            ],
                            **prepared[name],
                        },
                    )
            except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
                failure("expert_meta_preparation", pending, error)
                raise
            except KeyboardInterrupt:
                current.progress(
                    "expert_meta_preparation_interrupted_preserved",
                    active_compute_job=None,
                    active_candidate=None,
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
                "candidate_accounts_or_metrics_computed": False,
                "candidate_metrics_scope": (
                    "formal_fresh_capital_meta_paths_not_internal_shadow_training_scores"
                ),
                "raw_inputs_and_target_matrices_remain_private": True,
                "investment_objective_verified": False,
                "order_authority": False,
                "independent_forward_validation": False,
            },
            ("created_at",),
        )
        checked_packet(frozen)
        current.progress(
            "expert_meta_registered_waiting_publication",
            active_compute_job=None,
            active_candidate=None,
            expert_meta_registration=base.reference(PACKET),
            pending_steps=["publish_actual_expert_meta_registration", "eight_meta_funded_paths"],
        )
        return base.reference(PACKET)


def publication(frozen, original, commit):
    inherited = base.published_inputs(original, commit)
    required = {r["path"] for r in frozen["code"] + frozen["policies"]}
    own = verify_publication(ROOT, base.reference(PACKET), required, commit)
    existing = [
        e
        for e in base.matching_events("publication")
        if e["body"].get("study_id") == frozen["study_id"]
    ]
    if existing:
        if len(existing) != 1 or existing[0]["body"]["publication"]["published_commit"] != commit:
            raise QuantError("Resume with the actual original meta publication, not a later patch.")
        return existing[0]["body"]["publication"]
    receipt = {**own, "original_dependency_publication": inherited}
    base.append_event(
        base.JOURNAL, "publication", None, {"study_id": frozen["study_id"], "publication": receipt}
    )
    return receipt


def states(frozen):
    result = {}
    for name in IDS:
        reviewed = base.matching_events("review", name)
        failed = base.matching_events("technical_failure", name)
        paths = {
            e["body"]["path_key"]: base.verify_path(
                e["body"]["receipt"], frozen, name, e["body"]["path_key"]
            )
            for e in base.matching_events("path_completed", name)
        }
        if reviewed:
            if len(reviewed) != 1 or failed or reviewed[0]["body"]["paths"] != paths:
                raise QuantError("A completed meta review disagrees with its actual path records.")
            result[name] = reviewed[0]["body"]
        else:
            result[name] = {
                "candidate_id": name,
                "status": "technical_failure" if failed else "pending",
                "paths": paths,
                "failure_event_sha256": failed[0]["sha256"] if failed else None,
                "failure_stage": failed[0]["body"]["stage"] if failed else None,
            }
    return result


def evaluate(commit):
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, experts, frozen, original, benchmarks = inputs()
        checked_packet(frozen)
        if RESULTS.exists():
            saved = read_json(RESULTS)
            if saved["frozen_specification_sha256"] != digest_json(frozen) or saved[
                "candidates"
            ] != states(frozen):
                raise QuantError("Preserve completed meta results without replay.")
            return base.reference(RESULTS)
        published = publication(frozen, original, commit)
        current.progress(
            "evaluating_expert_meta",
            active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
        )
        name = None
        try:
            targets = {
                identifier: execution.frame(
                    base.matching_events("target_proof", identifier)[0]["body"]["targets"]
                )
                for identifier in IDS
            }
            proof = read_json(
                base.checked_reference(
                    base.matching_events("target_proof", IDS[0])[0]["body"]["proof"]
                )
            )
            audits = read_json(base.checked_reference(proof["information_audits"]))["decisions"]
            verify_targets(data, experts, targets, audits)
            protocol = read_json(base.PROTOCOL)
            for name in IDS:
                if base.matching_events("review", name) or base.matching_events(
                    "technical_failure", name
                ):
                    continue
                paths = {}
                for window in protocol["windows"]:
                    for scenario in protocol["accounts"]["scenarios"]:
                        runtime_guard()
                        check_frozen(frozen)
                        key = f"{window['years']}y-{scenario['id']}"
                        current.progress(
                            "evaluating_expert_meta", active_candidate=name, active_path=key
                        )
                        paths[key] = execution.evaluate_path(
                            data, targets[name], frozen, name, window, scenario, benchmarks
                        )
                check_frozen(frozen)
                base.append_event(
                    base.JOURNAL,
                    "review",
                    name,
                    {
                        "candidate_id": name,
                        "reviewed_at": utc_now(),
                        "status": "historical_point_gates_passed"
                        if all(p["all_point_gates_pass"] for p in paths.values())
                        else "rejected",
                        "paths": paths,
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
                    "investment_objective_verified": False,
                    "order_authority": False,
                },
            )
            current.progress(
                "expert_meta_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                expert_meta_results=base.reference(RESULTS),
                pending_steps=["publish_all_expert_meta_results"],
            )
            return base.reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("expert_meta_evaluation", (name,) if name else (), error)
            raise
        except KeyboardInterrupt:
            current.progress(
                "expert_meta_interrupted_preserved", active_compute_job=None, active_candidate=None
            )
            raise


def public_result():
    base.storage_guard()
    _, _, frozen, _, benchmarks = inputs()
    checked_packet(frozen)
    saved, actual = read_json(RESULTS), states(frozen)
    if saved["candidates"] != actual or any(v["status"] == "pending" for v in actual.values()):
        raise QuantError("An unfinished meta-study is not a terminal result.")
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
                    "benchmark_net_cagr": {
                        symbol: lookup[years, scenario, symbol] for symbol in ("SPY", "QQQ")
                    },
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
            "registration_packet": base.reference(PACKET),
            "protocol": frozen["protocol"],
            "private_complete_results": base.reference(RESULTS),
            "reviews": reviews,
            **counts,
            "new_completed_configurations": counts["new_completed_strategy_evaluations"],
            "global_counts_at_completion": saved["global_counts_at_completion"],
            "original_records_reclassified": False,
            "raw_prices_documents_targets_or_daily_accounts_exported": False,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "order_authority": False,
            "automatic_live_deployment": False,
        },
        ("created_at",),
    )
    return base.reference(PUBLIC_RESULTS)


def main():
    parser = argparse.ArgumentParser(
        description="Complete-strategy expert composition; no trading."
    )
    parser.add_argument("command", choices=("register", "evaluate", "publish-result", "status"))
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("Expert meta research cannot request live sources.")

    sys.addaudithook(offline)
    try:
        if args.command == "register":
            result = register()
        elif args.command == "evaluate":
            if args.commit is None:
                raise QuantError("Provide the actual pre-account publication commit.")
            result = evaluate(args.commit)
        elif args.command == "publish-result":
            result = public_result()
        else:
            result = current.current_counts(base.journal_events(base.JOURNAL))
        print(json.dumps(result, indent=2, allow_nan=False))
    except QuantError as error:
        parser.exit(2, f"Expert meta research blocked: {error}\n")


if __name__ == "__main__":
    main()
