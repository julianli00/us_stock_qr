from __future__ import annotations

import argparse
import json
import os
import sys
from importlib.metadata import version
from xml.etree import ElementTree

import pandas as pd

from us_quant import expert_initial_monthly_execution_v1 as execution
from us_quant import expert_net_wealth_mixture_v1 as rules
from us_quant import ssd_four_expert_meta_v1 as previous
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import DEADLINE, research_job, runtime_guard
from us_quant.research_program import review_market
from us_quant.research_weekly import verify_publication
from us_quant.ssd_research_v1 import append_event, matching_events, published_inputs, reference
from us_quant.storage import (
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_text_atomic,
)
from us_quant.strategy_experts_v1 import (
    ArchivedDenseExpert,
    ArchivedMonthlyExpert,
    ExpertIdentity,
)

base, common, current, ROOT = previous.base, previous.common, previous.current, previous.ROOT
FOLDER = base.REPORTS / "expert-net-mixture-20261029-v1"
INTENT = FOLDER / "research-intent-v1.json"
TARGETS = ROOT / "data/ssd-expert-net-mixture-20261029-v1"
PACKET = ROOT / "evidence/expert_net_mixture_20261029_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/expert_net_mixture_20261029_results_v1.json"
IDS = tuple(c["id"] for c in rules.CANDIDATES)
OWN = (
    "config/expert-net-wealth-mixture-v1.json",
    "src/us_quant/expert_net_wealth_mixture_v1.py",
    "src/us_quant/ssd_expert_net_mixture_v1.py",
    "tests/test_expert_net_wealth_mixture_v1.py",
    "tests/test_ssd_expert_net_mixture_v1.py",
)
PARENT_SPEC = "9543c88c522558ec470f0b7ca3154d37408c083ff475a508e648eb3f8689a62b"


def validation_references():
    path = FOLDER / "offline-study-tests-v1.xml"
    suites = ElementTree.parse(path).getroot().findall(".//testsuite")
    names = {
        case.get("name", "").split("[", 1)[0]
        for suite in suites
        for case in suite.findall("testcase")
    }
    required = {
        "test_one_period_matches_analytic_uniform_dirichlet_integral",
        "test_dense_recursion_matches_independent_polynomial_exact_integration",
        "test_identical_streams_keep_uniform_prior_at_full_bounded_history",
        "test_true_initial_anchor_and_prequential_validation_precede_assimilation",
        "test_every_allocation_matches_independent_integration_and_capital_interface",
        "test_prefix_and_future_perturbations_leave_past_targets_identical",
        "test_publication_checks_full_dependencies_before_packet_and_fence",
        "test_canonical_export_does_not_publish_daily_shadow_or_main_rows",
    }
    if (
        not required.issubset(names)
        or sum(int(s.get("tests", 0)) for s in suites) < len(required)
        or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
    ):
        raise QuantError("Require completed bounded integration, causality and publication tests.")
    return [reference(path), reference(FOLDER / "offline-study-tests-v1.log")]


def frame(ref):
    return pd.read_csv(
        base.checked_reference(ref), index_col=0, parse_dates=True, float_precision="round_trip"
    )


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    packet = read_json(base.checked_reference(policy["source_registration"]))
    parent = packet["entries"][0]["specification"]["frozen"]
    if (
        digest_json(parent) != PARENT_SPEC
        or parent["python_version"] != list(sys.version_info[:3])
        or any(version(k) != v for k, v in parent["package_versions"].items())
    ):
        raise QuantError("Use the unchanged original four-expert sources and accepted environment.")
    previous.check_frozen(parent)
    previous.checked_packet(parent)
    if read_json(previous.RESULTS)["candidates"] != previous.states(parent):
        raise QuantError("Preserve all original completed source books and reviews.")
    base.checked_reference(policy["previous_master_results"])
    source_policy = read_json(base.checked_reference(parent["policy"]))
    pool_ref, feedback_ref = (
        source_policy["first_pool_admission"],
        source_policy["feedback_expert_admission"],
    )
    pool, admission = (
        read_json(base.checked_reference(pool_ref)),
        read_json(base.checked_reference(feedback_ref)),
    )
    selected = [
        r for r in pool["admission_checks"] if r["first_pool_status"] == "admitted_selected"
    ]
    if (
        tuple(r["id"] for r in selected) != rules.EXPERT_IDS[:3]
        or any(
            r["source_technical_and_funding_checks_passed"] is not True
            or r["four_completed_paths_checked"] != 4
            for r in selected
        )
        or admission["expert"]["id"] != rules.EXPERT_IDS[-1]
        or admission["new_strategy_evaluations"] != 0
        or admission["new_shadow_account_runs"] != 0
    ):
        raise QuantError("Reuse actual technical admissions, not a financial-winner filter.")
    data = review_market({"market": parent["market"]}, ROOT)
    assets = tuple(data.close.columns)
    adapters = {"dense": ArchivedDenseExpert, "monthly": ArchivedMonthlyExpert}
    experts, refs = (
        [],
        [
            policy["source_registration"],
            policy["previous_master_results"],
            parent["policy"],
            pool_ref,
            feedback_ref,
            reference(previous.RESULTS),
        ],
    )
    for r in selected:
        identity = ExpertIdentity(r["id"], r["mechanism_group"], r["introduced_at"])
        experts.append(
            adapters[r["adapter"]](
                identity, frame(r["target_reference"]), frame(r["shadow_reference"]), assets
            )
        )
        refs.extend((r["target_reference"], r["shadow_reference"]))
    r = admission["expert"]
    experts.append(
        ArchivedDenseExpert(
            ExpertIdentity(r["id"], r["mechanism_group"], r["introduced_at"]),
            frame(r["targets"]),
            frame(r["shadow"]),
            assets,
        )
    )
    refs.extend((r["targets"], r["shadow"]))
    initial = rules.observations(experts, pd.Timestamp(policy["initial_decision_session"]))
    final = rules.observations(experts, data.close.index[-1])
    if any(len(o.net_returns) for o in initial) or any(
        len(o.net_returns) != 2512 or not o.risk_free.equals(data.risk_free.loc[o.risk_free.index])
        for o in final
    ):
        raise QuantError("Do not invent pre-shadow returns or change the original RF decoder.")
    if file_digest(ROOT / "runtime/research-program.sqlite3") != base.LEGACY_SHA256:
        raise QuantError("The original immutable ledger changed.")
    benchmarks = read_json(base.BENCHMARKS / "results.json")
    if (
        benchmarks["protocol_sha256"] != parent["protocol"]["sha256"]
        or benchmarks["benchmark_accounts"] != 8
        or benchmarks["new_strategy_evaluations"] != 0
    ):
        raise QuantError("Use only the eight unchanged funded SPY/QQQ benchmarks.")
    return data, experts, policy, parent, previous.source.unique_refs(refs), benchmarks


def initialize():
    runtime_guard()
    common.output_guard(INTENT)
    if INTENT.exists():
        raise QuantError("Read the existing freeze rather than initialize it again.")
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        _, _, policy, parent, refs, _ = source_inputs()
        events = base.journal_events(base.JOURNAL)
        counts = current.current_counts(events)
        if (
            len(events) != 420
            or events[-1]["sha256"] != rules.PRIOR_HEAD
            or counts["new_registered_candidates"] != 38
            or counts["new_completed_strategy_evaluations"] != 35
            or counts["new_technical_failures"] != 3
            or counts["new_pending_candidates"] != 0
        ):
            raise QuantError("Freeze the actual unchanged post-master-study research state.")
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "actual_continuation_event": 37925,
                "actual_delivery": "system_idle",
                "model_and_test_files": [reference(ROOT / p) for p in OWN],
                "validation_evidence": validation_references(),
                "source_references": refs,
                "source_specification_sha256": digest_json(parent),
                "question": policy["methodology"]["question"],
                "real_new_meta_scores_or_targets_computed": False,
                "candidate_accounts_or_metrics_computed": False,
                "source_expert_targets_fits_or_accounts_replayed": False,
                "new_shadow_account_runs": 0,
                "new_source_requests": 0,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        current.progress(
            "expert_net_mixture_rule_frozen",
            active_compute_job=None,
            expert_net_mixture_intent=reference(INTENT),
            pending_steps=["register_net_mixture_before_real_learning_or_targets"],
        )
        return reference(INTENT)


def inputs():
    data, experts, policy, parent, refs, benchmarks = source_inputs()
    intent = read_json(INTENT)
    if (
        intent["model_and_test_files"] != [reference(ROOT / p) for p in OWN]
        or intent["validation_evidence"] != validation_references()
        or intent["source_references"] != refs
        or intent["source_specification_sha256"] != digest_json(parent)
        or intent["real_new_meta_scores_or_targets_computed"] is not False
        or intent["candidate_accounts_or_metrics_computed"] is not False
    ):
        raise QuantError("The frozen net-mixture rule, tests or cached sources changed.")
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": reference(rules.POLICY),
        "code": previous.source.unique_refs(
            [*parent["code"], *[reference(ROOT / p) for p in OWN[1:]]]
        ),
        "policies": previous.source.unique_refs([*parent["policies"], reference(rules.POLICY)]),
        "prior_completed_configurations": 213,
        "prior_journal_sequence": 420,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "source_expert_pool_policy": parent["policy"],
        "expert_pool_definition_role": "unchanged_four_complete_policies_not_old_pair_ranking",
        "construction": "continuous_uniform_four_complete_NET_policy_score_integration",
        "construction_inputs": [reference(INTENT), *refs],
        "execution_adapter": reference(
            ROOT / "src/us_quant/expert_initial_monthly_execution_v1.py"
        ),
        "execution_adapter_role": "unchanged_initial_plus_monthly_actual_holdings_025_band",
        "prior_evidence": previous.source.unique_refs(
            [*parent["prior_evidence"], *refs, reference(INTENT)]
        ),
        "validation_evidence": previous.source.unique_refs(
            [*parent["validation_evidence"], *validation_references()]
        ),
        "actual_factor_participation_is_measured_in_target_proof": False,
        "actual_factor_participation_is_measured_in_funded_account": True,
        "shadow_net_scores_are_actual_main_wealth": False,
        "new_shadow_account_runs": 0,
        "new_source_requests": 0,
    }
    previous.check_frozen(frozen)
    return data, experts, frozen, benchmarks


def failure(stage, error):
    expired = pd.Timestamp(utc_now()) >= pd.Timestamp(DEADLINE)
    if not expired:
        for name in IDS:
            if (
                matching_events("registered", name)
                and not matching_events("review", name)
                and not matching_events("technical_failure", name)
            ):
                append_event(
                    base.JOURNAL,
                    "technical_failure",
                    name,
                    {
                        "stage": stage,
                        "reason": str(error),
                        "partial_outputs_retained": True,
                        "candidate_accounts_started": bool(matching_events("path_started", name)),
                    },
                )
    current.progress(
        "expert_net_mixture_expired" if expired else "expert_net_mixture_blocked",
        active_compute_job=None,
        active_candidate=None,
        active_path=None,
        current_failure=str(error),
    )


def prepare(data, experts, frozen):
    name = IDS[0]
    directory = TARGETS / name
    proof_path = directory / "proof.json"
    if proof_path.exists():
        proof = read_json(proof_path)
        if proof["frozen_specification_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting prior preparation, not regenerate targets.")
        for ref in (proof["targets"], proof["information_audits"]):
            base.checked_reference(ref)
        return {"targets": proof["targets"], "proof": reference(proof_path)}
    new_output_directory(directory)
    policy = read_json(base.checked_reference(frozen["policy"]))
    target, audits, periods = rules.build_targets(
        experts, data.close.index, tuple(data.close.columns), policy
    )
    verification = rules.verify_targets(
        experts, data.close.index, tuple(data.close.columns), policy, target, audits, periods
    )
    prefixes = []
    positions = sorted({0, 1, 2, len(audits) // 3, 2 * len(audits) // 3, len(audits) - 1})
    for i in positions:
        runtime_guard()
        day = pd.Timestamp(audits[i]["decision_session"])
        calendar = data.close.loc[:day].index
        earlier, aa, pp = rules.build_targets(experts, calendar, tuple(data.close.columns), policy)
        if (
            not earlier.equals(target.loc[:day])
            or aa != audits[: i + 1]
            or pp != [r for r in periods if r["end"] <= str(day.date())]
        ):
            raise QuantError("A future observation changed the actual net-mixture prefix.")
        prefixes.append(str(day.date()))
    if len(audits) != 121 or len(periods) != 119:
        raise QuantError(
            "Keep the full original decision lattice and genuine complete learning months."
        )
    write_text_atomic(directory / "targets.csv", target.to_csv(float_format="%.17g"))
    write_new_receipt(
        directory / "information-audits.json",
        {
            "expert_vector_order": list(rules.EXPERT_IDS),
            "decisions": audits,
            "complete_periods": periods,
        },
    )
    write_new_receipt(
        proof_path,
        {
            "checked_at": utc_now(),
            "frozen_specification_sha256": digest_json(frozen),
            "targets": reference(directory / "targets.csv"),
            "information_audits": reference(directory / "information-audits.json"),
            **verification,
            "prefix_truncations": prefixes,
            "candidate_accounts_or_metrics_computed": False,
            "original_source_signal_recomputed": False,
            "gross_or_shadow_score_is_actual_net_main_performance": False,
        },
    )
    return {"targets": reference(directory / "targets.csv"), "proof": reference(proof_path)}


def checked_packet(frozen):
    packet = read_json(PACKET)
    entries = [common.packet_entry(c, frozen) for c in rules.CANDIDATES]
    created = pd.Timestamp(packet["created_at"])
    if (
        packet["schema_version"] != 2
        or packet["program_id"] != "ssd_continuous_research_v1"
        or packet["study_id"] != frozen["study_id"]
        or packet["entries"] != entries
        or packet["new_registered_candidates"] != 1
        or created.tzinfo is None
        or pd.isna(created)
        or created > pd.Timestamp(utc_now())
        or any(pd.Timestamp(e["target_proof_at"]) > created for e in entries)
        or any(
            created > pd.Timestamp(e["at"])
            for name in IDS
            for e in matching_events("path_started", name)
        )
        or any(
            packet.get(k) is not False
            for k in (
                "candidate_accounts_or_metrics_computed",
                "investment_objective_verified",
                "independent_forward_validation",
                "order_authority",
            )
        )
    ):
        raise QuantError(
            "The actual preregistration identity, clock or pre-account boundary changed."
        )
    return packet


def register():
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, experts, frozen, _ = inputs()
        if any(matching_events("technical_failure", name) for name in IDS):
            raise QuantError("A failed registered version cannot be silently retried.")
        for candidate in rules.CANDIDATES:
            found = matching_events("registered", candidate["id"])
            body = {"candidate": candidate, "frozen": frozen}
            if found and (len(found) != 1 or found[0]["body"] != body):
                raise QuantError("Registered net-mixture parameters and inputs cannot change.")
            if not found:
                append_event(base.JOURNAL, "registered", candidate["id"], body)
        try:
            if not matching_events("target_proof", IDS[0]):
                current.progress(
                    "preparing_registered_expert_net_mixture",
                    active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
                )
                proof = prepare(data, experts, frozen)
                append_event(
                    base.JOURNAL,
                    "target_proof",
                    IDS[0],
                    {
                        "registered_event_sha256": matching_events("registered", IDS[0])[0][
                            "sha256"
                        ],
                        **proof,
                    },
                )
            common.retain_receipt(
                PACKET,
                {
                    "schema_version": 2,
                    "program_id": "ssd_continuous_research_v1",
                    "study_id": frozen["study_id"],
                    "created_at": utc_now(),
                    "entries": [common.packet_entry(c, frozen) for c in rules.CANDIDATES],
                    "new_registered_candidates": 1,
                    "new_economic_hypotheses": 1,
                    "candidate_accounts_or_metrics_computed": False,
                    "candidate_metrics_scope": (
                        "formal main paths; shadow-score validation is not main performance"
                    ),
                    "raw_inputs_and_target_matrices_remain_private": True,
                    "investment_objective_verified": False,
                    "independent_forward_validation": False,
                    "order_authority": False,
                },
                ("created_at",),
            )
            checked_packet(frozen)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("expert_net_mixture_preparation", error)
            raise
        current.progress(
            "expert_net_mixture_registered_waiting_publication",
            active_compute_job=None,
            expert_net_mixture_registration=reference(PACKET),
            pending_steps=["publish_exact_net_mixture_packet_before_four_accounts"],
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
            raise QuantError("Resume the original publication fence, not a later result commit.")
        return found[0]["body"]["publication"]
    receipt = {**own, "complete_dependency_publication": dependencies}
    append_event(
        base.JOURNAL, "publication", None, {"study_id": frozen["study_id"], "publication": receipt}
    )
    return receipt


def states(frozen):
    result = {}
    for name in IDS:
        reviews, failed, completed = (
            matching_events(k, name) for k in ("review", "technical_failure", "path_completed")
        )
        if (
            len({e["body"]["path_key"] for e in completed}) != len(completed)
            or len(reviews) > 1
            or len(failed) > 1
        ):
            raise QuantError("Duplicate completion or terminal events cannot select a winner.")
        paths = {
            e["body"]["path_key"]: base.verify_path(
                e["body"]["receipt"], frozen, name, e["body"]["path_key"]
            )
            for e in completed
        }
        if reviews:
            if failed or reviews[0]["body"]["paths"] != paths:
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


def evaluate(commit):
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, _, frozen, benchmarks = inputs()
        checked_packet(frozen)
        if RESULTS.exists():
            if read_json(RESULTS)["candidates"] != states(frozen):
                raise QuantError("Preserve actual completed net-mixture outcomes without replay.")
            return reference(RESULTS)
        try:
            published = publication(frozen, commit)
            current.progress(
                "evaluating_expert_net_mixture",
                active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            )
            target = frame(matching_events("target_proof", IDS[0])[0]["body"]["targets"])
            protocol = read_json(base.PROTOCOL)
            if not matching_events("review", IDS[0]) and not matching_events(
                "technical_failure", IDS[0]
            ):
                paths = {}
                for window in protocol["windows"]:
                    for scenario in protocol["accounts"]["scenarios"]:
                        runtime_guard()
                        previous.check_frozen(frozen)
                        key = f"{window['years']}y-{scenario['id']}"
                        current.progress(
                            "evaluating_expert_net_mixture",
                            active_candidate=IDS[0],
                            active_path=key,
                        )
                        paths[key] = execution.evaluate_path(
                            data, target, frozen, IDS[0], window, scenario, benchmarks
                        )
                participation = previous.actual_factor_participation(paths, frozen)
                passed = all(p["all_point_gates_pass"] for p in paths.values()) and all(
                    p["minimum_three_registered_families_pass"] for p in participation.values()
                )
                append_event(
                    base.JOURNAL,
                    "review",
                    IDS[0],
                    {
                        "candidate_id": IDS[0],
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
                    "new_source_requests": 0,
                    "investment_objective_verified": False,
                    "order_authority": False,
                },
                ("finished_at",),
            )
            current.progress(
                "expert_net_mixture_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                expert_net_mixture_results=reference(RESULTS),
                pending_steps=["publish_all_four_net_mixture_outcomes"],
            )
            return reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            failure("expert_net_mixture_evaluation", error)
            raise
        except KeyboardInterrupt:
            current.progress("expert_net_mixture_interrupted", active_compute_job=None)
            raise


def public_result():
    _, _, frozen, benchmarks = inputs()
    checked_packet(frozen)
    actual, saved = states(frozen), read_json(RESULTS)
    if saved["candidates"] != actual or any(v["status"] == "pending" for v in actual.values()):
        raise QuantError("Publish actual complete outcomes or retained technical failure only.")
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
            "original_records_reclassified": False,
            "raw_prices_documents_targets_or_daily_accounts_exported": False,
            "new_shadow_account_runs": 0,
            "new_source_requests": 0,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "order_authority": False,
            "automatic_live_deployment": False,
        },
        ("created_at",),
    )
    return reference(PUBLIC_RESULTS)


def main():
    parser = argparse.ArgumentParser(
        description="Complete net-policy mixture research; no trading."
    )
    parser.add_argument(
        "command", choices=("init", "register", "evaluate", "publish-result", "status")
    )
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("Net-mixture research cannot request live sources.")

    sys.addaudithook(offline)
    try:
        if args.command == "init":
            result = initialize()
        elif args.command == "register":
            result = register()
        elif args.command == "evaluate":
            if args.commit is None:
                raise QuantError("Provide the original actual preregistration publication commit.")
            result = evaluate(args.commit)
        elif args.command == "publish-result":
            result = public_result()
        else:
            result = current.current_counts(base.journal_events(base.JOURNAL))
        print(json.dumps(result, indent=2, allow_nan=False))
    except QuantError as error:
        parser.exit(2, f"Expert net-mixture research blocked: {error}\n")


if __name__ == "__main__":
    main()
