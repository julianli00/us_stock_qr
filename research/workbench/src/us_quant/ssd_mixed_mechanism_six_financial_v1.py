from __future__ import annotations

import argparse
import ast
import json
import os
from fractions import Fraction
from importlib.metadata import distributions
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd

from us_quant import mixed_mechanism_six_primary_v1 as primary
from us_quant import mixed_mechanism_six_sources_v1 as admitted
from us_quant import ssd_calendar_expert_inclusion_v1 as cohort
from us_quant import ssd_eight_expert_meta_v1 as template
from us_quant.calendar import is_month_end, previous_session
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.storage import (
    digest_json,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)

base, ROOT, seven, financial = template.base, template.ROOT, template.seven, template.financial
lifecycle, source = cohort.lifecycle, seven
reference, unique_refs = base.reference, template.unique_refs
FOLDER = base.REPORTS / "mixed-mechanism-six-financial-20261105-v1"
POLICY = ROOT / "config/mixed-mechanism-six-financial-v1.json"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
PREPARED = FOLDER / "prepared-target-bundle-v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC = ROOT / "evidence/mixed_mechanism_six_financial_20261105_results_v1.json"
CANDIDATES, IDS = primary.CANDIDATES, primary.IDS
PACKETS = {
    name: ROOT / "evidence" / f"mixed_mechanism_six_financial_20261105_{i}_registration_v1.json"
    for i, name in enumerate(IDS)
}
OWN = (
    "config/mixed-mechanism-six-financial-v1.json",
    "src/us_quant/ssd_mixed_mechanism_six_financial_v1.py",
    "tests/test_ssd_mixed_mechanism_six_financial_v1.py",
    "MIXED_MECHANISM_SIX_FINANCIAL_V1.md",
    *primary.OWN,
    *primary.FIXTURE_INPUTS,
    "tests/test_macro_expert_inclusion_v1.py",
)


def bound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


def validate(policy):
    prefix = "reports/ssd-continuous-research-v1/mixed-mechanism-six-primary-20261105-v1/"
    expected = {
        "schema_version": 1,
        "study_id": "ssd_mixed_mechanism_six_financial_20261105_v1",
        "primary_policy": {
            "path": "config/mixed-mechanism-six-primary-v1.json",
            "sha256": "86cd9b30af0e01e53a0fd457f7337f86ef05139bba66e6818ed3dd79ef4e8e31",
        },
        "primary_freeze": {
            "path": prefix + "primary-implementation-freeze-v1.json",
            "sha256": "f355581a7c4b562a575d1beca722de8115a8d450b98296fda14d4967f330fb5b",
        },
        "primary_completion": {
            "path": prefix + "completion-v1.json",
            "sha256": "ef012e51e5dbab0e569a71d648ec92627660853f07ffecab3057c8a07f39f73f",
        },
        "matched_original_eight_results": {
            "path": "evidence/eight_expert_inclusion_20261103_results_v1.json",
            "sha256": "d3bce69f0e921ae6265218bbde6beeed58f8435caa20eaffb11ba9ba74808e7a",
        },
        "original_financial_protocol": {
            "path": "config/dual-benchmark-v1.json",
            "sha256": "1b5b2572089ffa53746b01e72ff8303e00d6b5e5e3f815844f93834d0521e469",
        },
        "prior_journal_sequence": 1086,
        "prior_journal_head_sha256": (
            "83abc48588a2405d4389ba1d7097e805e29b04b5b63d4dd02ea0e2607a0a2f6a"
        ),
        "prior_completed_configurations": 269,
        "candidate_count": 2,
        "expert_count": 6,
        "mature_pair_count": 15,
        "static_target_packet_schema": 2,
        "both_registrations_before_either_meta_derivation": True,
        "complete_cohort_publication_before_any_funds": True,
        "new_source_requests": 0,
        "new_shadow_account_runs": 0,
        "global_pool_changed": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if policy != expected or any(type(policy[k]) is not type(v) for k, v in expected.items()):
        raise QuantError("Preserve the immutable f355 six-pool law and complete two-rule finances.")


def check_primary():
    policy = read_json(POLICY)
    validate(policy)
    for key in (
        "primary_policy",
        "primary_freeze",
        "primary_completion",
        "matched_original_eight_results",
        "original_financial_protocol",
    ):
        base.checked_reference(policy[key])
    frozen = read_json(base.checked_reference(policy["primary_freeze"]))
    for row in frozen["implementation_and_test_inputs"] + frozen["complete_dependency_inputs"]:
        base.checked_reference(row)
    if (
        frozen["actual_registered"] is not False
        or frozen["real_new_meta_scores_targets_accounts_or_financial_outcomes_computed"]
        is not False
        or frozen["financial_program_implemented"] is not False
    ):
        raise QuantError("The preparation-only primary is a historical snapshot, not funded proof.")
    primary.validate(read_json(primary.POLICY))
    return policy, frozen


def build_targets(experts, calendar, universe, policy, **kwargs):
    validate(policy)
    return primary.build_targets(experts, calendar, universe, read_json(primary.POLICY), **kwargs)


def decision_sessions(calendar, policy):
    validate(policy)
    initial = pd.Timestamp("2016-10-05")
    return pd.DatetimeIndex([initial, *[d for d in calendar if d > initial and is_month_end(d)]])


rules = SimpleNamespace(
    POLICY=POLICY,
    EXPERT_IDS=primary.EXPERT_IDS,
    CLOCK=primary.CLOCK,
    CANDIDATES=CANDIDATES,
    build_targets=build_targets,
    decision_sessions=decision_sessions,
)


def events(kind, identifier):
    return bound(template.events)(kind, identifier)


def check_frozen(frozen):
    template.check_frozen(frozen)


def dependencies():
    pending = [ROOT / path for path in OWN if path.endswith(".py")]
    result = {}
    while pending:
        path = pending.pop()
        key = path.relative_to(ROOT).as_posix()
        if key in result:
            continue
        result[key] = reference(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
                if node.module == "us_quant":
                    names.extend("us_quant." + a.name for a in node.names)
            for name in names:
                if name.startswith("us_quant"):
                    root, relative = ROOT / "src", name.replace(".", "/")
                elif name.startswith("test_") or name == "conftest":
                    root, relative = ROOT / "tests", name.replace(".", "/")
                else:
                    continue
                pending.extend(
                    p
                    for p in (root / (relative + ".py"), root / relative / "__init__.py")
                    if p.is_file()
                )
    return unique_refs([*result.values(), *[reference(ROOT / path) for path in OWN]])


def source_inputs():
    policy, primary_freeze = check_primary()
    data, experts, parent, refs, clocks, benchmarks = admitted.source_inputs(
        read_json(primary.POLICY)
    )
    refs = unique_refs(
        [
            *refs,
            *primary_freeze["source_references"],
            *[
                policy[k]
                for k in (
                    "primary_policy",
                    "primary_freeze",
                    "primary_completion",
                    "matched_original_eight_results",
                    "original_financial_protocol",
                )
            ],
        ]
    )
    for row in refs:
        base.checked_reference(row)
    return data, experts, parent, refs, clocks, benchmarks


def progress(phase, **fields):
    counts = current_counts(base.journal_events(base.JOURNAL))
    if any(k in fields and fields[k] != v for k, v in counts.items()):
        raise QuantError("Mixed-six progress cannot override authoritative journal totals.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "mixed_six_financial_" + phase,
            "study_id": read_json(POLICY)["study_id"],
            "active_compute_job": None,
            **counts,
            **fields,
        },
    )


def validation_refs():
    return bound(cohort.validation_refs)()


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the first complete or partial mixed-six financial freeze.")
    with lifecycle.research_job():
        design, policy = read_json(DESIGN), read_json(POLICY)
        if (
            design["actual_native_event_seq"] != 167767
            or design["candidates"] != list(CANDIDATES)
            or design["new_meta_targets_scores_or_accounts_computed"] is not False
        ):
            raise QuantError("Declare the same six-pool financial question before outcomes.")
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, _, parent, refs, clocks, _ = source_inputs()
        counts = current_counts(base.journal_events(base.JOURNAL))
        if (
            counts["current_journal_event_count"] != policy["prior_journal_sequence"]
            or counts["current_journal_head_sha256"] != policy["prior_journal_head_sha256"]
            or counts["total_legacy_plus_ssd_completed_configurations"] != 269
            or tuple(row["id"] for row in clocks) != primary.EXPERT_IDS
        ):
            raise QuantError("Keep the actual J1086/269 baseline and six source clocks.")
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": policy["study_id"],
            "policy": reference(primary.POLICY),
            "financial_program_policy": reference(POLICY),
            "expert_pool": reference(primary.POLICY),
            "construction": "deliberate_mixed_mechanism_six_unchanged_equal_and_point_rolling",
            "expert_pool_definition_role": "study_specific_six_original7and8_unchanged",
            "prior_journal_sequence": 1086,
            "prior_journal_head_sha256": policy["prior_journal_head_sha256"],
            "prior_completed_configurations": 269,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in [*deps, *refs] if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / "MIXED_MECHANISM_SIX_FINANCIAL_V1.md"),
                    reference(ROOT / "MIXED_MECHANISM_SIX_PRIMARY_V1.md"),
                ]
            ),
            "policies": unique_refs(
                [
                    *parent["policies"],
                    *[r for r in [*deps, *refs] if r["path"].startswith("config/")],
                ]
            ),
            "prior_evidence": unique_refs(
                [*parent["prior_evidence"], *refs, reference(DESIGN), reference(INVOCATION)]
            ),
            "validation_evidence": unique_refs(
                [*parent["validation_evidence"], *validation_refs()]
            ),
            "source_policy_and_expert_admission_clocks": clocks,
            "decision_information_clock": primary.CLOCK,
            "study_specific_expert_ids": list(primary.EXPERT_IDS),
            "study_specific_expert_count": 6,
            "global_pool_changed": False,
            "native_RF_exact_common_no_view": True,
            "new_shadow_account_runs": 0,
            "new_source_requests": 0,
        }
        check_frozen(frozen)
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "frozen": frozen,
                "source_references": refs,
                "complete_dependency_inputs": deps,
                "package_versions": sorted(
                    (d.metadata["Name"], d.version) for d in distributions()
                ),
                "real_new_scores_targets_or_accounts_computed": False,
                "both_cohort_identities_registered": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="register_both_before_actual_scores_targets")
    return reference(INTENT)


def inputs():
    return bound(template.inputs)()


def verify_targets(data, experts, targets, audits):
    maximum_rational = 0.0
    for audit in audits:
        day = pd.Timestamp(audit["decision_session"])
        observed = [e.observe(day, 126) for e in experts]
        selection = audit["selection"]
        if (
            tuple(o.identity.mechanism_group for o in observed) != primary.MECHANISMS
            or audit["preserved_archive_identity_introduced_at"]
            != [o.identity.introduced_at for o in observed]
            or audit["counterfactual_historical_expert_availability"] is not True
            or audit["independent_forward_validation"] is not False
            or selection["NET_history_mature"] != all(o.score_mature for o in observed)
            or any(
                selection[k] is not False
                for k in (
                    "source_experts_invalidated",
                    "mechanism_labels_are_independent_risks",
                    "current_opinion_exclusion_used",
                    "conditional_deletion_lower_envelope_used",
                )
            )
        ):
            raise QuantError("Keep complete source clocks, actual introductions and unaltered law.")
        for pair in selection["pairs"]:
            if "selection_score_key" not in pair:
                continue
            weights = np.asarray(pair["expert_weights"])
            marginal = np.asarray(pair["training_marginal_variance_contributions"])
            numbers = [
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
            if (
                weights.shape != (6,)
                or marginal.shape != (6,)
                or not np.isfinite(weights).all()
                or not np.isfinite(marginal).all()
                or not np.isfinite(numbers).all()
                or (weights < 0).any()
                or abs(float(weights.sum()) - 1) > 1e-12
            ):
                raise QuantError("Every scored pair needs finite six-dimensional risk evidence.")
        for identifier, key in zip(
            IDS, ("equal_expert_weights", "rolling_expert_weights"), strict=True
        ):
            expected = [
                float(
                    sum(
                        Fraction.from_float(float(w)) * Fraction.from_float(float(o.target[s]))
                        for w, o in zip(audit[key], observed, strict=True)
                    )
                )
                for s in data.close.columns
            ]
            maximum_rational = max(
                maximum_rational,
                float(np.max(np.abs(targets[identifier].loc[day].to_numpy() - expected))),
            )
    if maximum_rational > 1e-12:
        raise QuantError("Unified targets differ from the rational complete-expert allocation.")
    return {
        **bound(template.verify_targets)(data, experts, targets, audits),
        "maximum_rational_target_difference": maximum_rational,
    }


def verify_retained_records(path, experts, targets, audits):
    observed_values = 0
    with path.open() as stream:
        for position, audit in enumerate(audits):
            before, after = json.loads(stream.readline()), json.loads(stream.readline())
            day = pd.Timestamp(audit["decision_session"])
            observations = []
            for expert in experts:
                item = expert.observe(day, 126)
                observations.append(
                    {
                        "id": item.identity.identifier,
                        "mechanism_group": item.identity.mechanism_group,
                        "introduced_at": item.identity.introduced_at,
                        "target_known_at": str(item.target_known_at.date()),
                        "target_binary64_hex": [float(v).hex() for v in item.target],
                        "asset_order": list(item.target.index),
                        "return_dates": item.net_returns.index.strftime("%Y-%m-%d").tolist(),
                        "NET_returns_binary64_hex": [float(v).hex() for v in item.net_returns],
                        "RF_binary64_hex": [float(v).hex() for v in item.risk_free],
                        "score_mature": item.score_mature,
                    }
                )
                observed_values += len(item.net_returns) + len(item.risk_free)
            if before != {
                "record": "decision_input",
                "decision_index": position,
                "as_of": audit["decision_session"],
                "information_clock": primary.CLOCK,
                "previous_weights_binary64_hex": [
                    float(v).hex() for v in audit["previous_rolling_expert_weights"]
                ],
                "observations": observations,
            } or after != {
                "record": "decision_resolved",
                "decision_index": position,
                "audit": audit,
                "asset_order": list(primary.UNIVERSE),
                "complete_targets_binary64_hex": {
                    identifier: [float(v).hex() for v in targets[identifier].loc[day]]
                    for identifier in IDS
                },
            }:
                raise QuantError(
                    "Retained six-source inputs, clocks or both resolved targets differ."
                )
        if stream.readline():
            raise QuantError("Keep exactly one input and one resolved record per actual decision.")
    return {
        "actual_input_and_resolved_record_count": 2 * len(audits),
        "roundtrip_NET_RF_binary64_values_checked": observed_values,
        "actual_input_resolved_records_match_source_and_targets": True,
    }


def prepare(data, experts, frozen):
    if PREPARED.exists():
        saved = read_json(PREPARED)
        if saved["frozen_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting prior mixed-six targets.")
        for entry in saved["entries"].values():
            for row in entry.values():
                base.checked_reference(row)
        return saved["entries"]
    if (FOLDER / "preparation-attempt-v1.json").exists():
        raise QuantError("Preserve partial actual derivation instead of generating it twice.")
    if any(len(events("registered", c["id"])) != 1 for c in CANDIDATES):
        raise QuantError("Both real cohort identities must precede all new scores or targets.")
    write_new_receipt(
        FOLDER / "preparation-attempt-v1.json",
        {"started_at": utc_now(), "frozen_sha256": digest_json(frozen)},
    )
    log = FOLDER / "actual-decision-inputs-resolved-v1.jsonl"
    with log.open("x") as stream:

        def retain(row):
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        targets, audits = build_targets(
            experts,
            data.close.index,
            tuple(data.close.columns),
            read_json(POLICY),
            before_decision=lifecycle.runtime_guard,
            retain=retain,
        )
    retained = verify_retained_records(log, experts, targets, audits)
    independent = verify_targets(data, experts, targets, audits)
    if len(audits) != 121:
        raise QuantError("Preserve all original121 initial/month-end decisions.")
    prefixes = []
    for position in (0, 6, 12, 60, 120):
        day = pd.Timestamp(audits[position]["decision_session"])
        early, earlier = build_targets(
            experts,
            data.close.loc[:day].index,
            tuple(data.close.columns),
            read_json(POLICY),
            before_decision=lifecycle.runtime_guard,
        )
        if any(not early[n].equals(targets[n].loc[:day]) for n in IDS) or (
            earlier != audits[: position + 1]
        ):
            raise QuantError("Future observations changed a past six-pool decision.")
        prefixes.append(str(day.date()))
    write_new_receipt(FOLDER / "information-audits-v1.json", {"decisions": audits})
    entries = {}
    for name in IDS:
        directory = FOLDER / "targets" / name
        new_output_directory(directory)
        active = targets[name].dropna(how="all")
        coverage = []
        for window in read_json(base.PROTOCOL)["windows"]:
            anchor = previous_session(window["start"])
            known = active.loc[:anchor].tail(1)
            rows = pd.concat([known, active.loc[anchor + pd.Timedelta(days=1) : window["end"]]])
            factors = [
                a for a in ("MTUM", "VLUE", "QUAL", "USMV", "IJR", "PKW") if (rows[a] > 1e-10).any()
            ]
            if known.empty or len(factors) < 3:
                raise QuantError("Both original horizons require known anchors and style factors.")
            coverage.append({"years": window["years"], "actual_factor_symbols": factors})
        write_text_atomic(directory / "targets.csv", targets[name].to_csv(float_format="%.17g"))
        write_new_receipt(
            directory / "proof.json",
            {
                "checked_at": utc_now(),
                "frozen_specification_sha256": digest_json(frozen),
                **independent,
                **retained,
                "decision_count": len(audits),
                "all_pairs_per_mature_decision": 15,
                "study_specific_expert_count": 6,
                "prefix_truncations": prefixes,
                "formal_factor_coverage": coverage,
                "information_audits": reference(FOLDER / "information-audits-v1.json"),
                "actual_input_resolved_records": reference(log),
                "native_RF_exact_common_no_view": True,
                "standalone_source_financial_rejections_preserved": True,
                "candidate_accounts_or_metrics_computed": False,
                "original_source_signal_recomputed": False,
                "new_source_or_shadow_account_runs": 0,
                "independent_forward_validation": False,
            },
        )
        entries[name] = {
            "targets": reference(directory / "targets.csv"),
            "proof": reference(directory / "proof.json"),
        }
    write_new_receipt(PREPARED, {"frozen_sha256": digest_json(frozen), "entries": entries})
    return entries


def register():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        data, experts, frozen, _ = inputs()
        if any(events("technical_failure", name) for name in IDS):
            raise QuantError("Preserve terminal mixed-six cohort identities without retry.")
        for candidate in CANDIDATES:
            name, body = candidate["id"], {"candidate": candidate, "frozen": frozen}
            if not events("registered", name):
                base.append_event(base.JOURNAL, "registered", name, body)
            elif events("registered", name)[0]["body"] != body:
                raise QuantError("Membership or law changed after actual registration.")
        try:
            prepared = prepare(data, experts, frozen)
            for candidate in CANDIDATES:
                name = candidate["id"]
                if not events("target_proof", name):
                    base.append_event(
                        base.JOURNAL,
                        "target_proof",
                        name,
                        {
                            "registered_event_sha256": events("registered", name)[0]["sha256"],
                            **prepared[name],
                        },
                    )
                seven.common.retain_receipt(
                    PACKETS[name],
                    {
                        "schema_version": 2,
                        "program_id": "ssd_continuous_research_v1",
                        "study_id": frozen["study_id"],
                        "created_at": utc_now(),
                        "entries": [seven.common.packet_entry(candidate, frozen)],
                        "complete_fixed_cohort_ids": list(IDS),
                        "cohort_registrations": [events("registered", v)[0]["sha256"] for v in IDS],
                        "candidate_accounts_or_metrics_computed": False,
                        "order_authority": False,
                        "independent_forward_validation": False,
                        "investment_objective_verified": False,
                    },
                    ("created_at",),
                )
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            for name in IDS:
                if not events("technical_failure", name):
                    base.append_event(
                        base.JOURNAL,
                        "technical_failure",
                        name,
                        {
                            "stage": "mixed_six_cohort_preparation",
                            "reason": str(error),
                            "partial_inputs_preserved": True,
                        },
                    )
            write_new_receipt(
                FOLDER / "preparation-failure-v1.json",
                {"failed_at": utc_now(), "error": str(error), "retry_or_retuning": False},
            )
            progress("technical_failure_preserved", next_step="do_not_retry_or_retune_cohort")
            raise
        progress("registered_before_accounts", next_step="publish_both_exact_packets_before_funds")
    return {name: reference(path) for name, path in PACKETS.items()}


def scoped(function, candidate):
    return bound(template.scoped)(function, candidate)


def cohort_publication(*args):
    return bound(template.cohort_publication)(*args)


def evaluate(commit):
    return bound(template.evaluate)(commit)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(RESULTS)
        controls = read_json(
            base.checked_reference(read_json(POLICY)["matched_original_eight_results"])
        )["candidates"]
        candidates = {}
        for candidate, control in zip(
            CANDIDATES,
            ("expert_pool_eight_equal_weight_v1", "expert_pool_eight_rolling_pair_v1"),
            strict=True,
        ):
            name = candidate["id"]
            data = read_json(base.checked_reference(saved["candidate_result_refs"][name]))
            if set(data["paths"]) != base.PATH_KEYS:
                raise QuantError("Export every original funded path for both complete policies.")
            candidates[name] = {
                "status": data["review"]["status"],
                "paths": data["paths"],
                "registered_event_sha256": events("registered", name)[0]["sha256"],
                "matched_cached_original_eight_control": {
                    "candidate_id": control,
                    "paths": controls[control]["paths"],
                    "old_control_replayed": False,
                    "isolated_causal_roster_or_fee_effect": False,
                },
            }
        seven.common.retain_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "study_id": read_json(POLICY)["study_id"],
                "completed_at": saved["completed_at"],
                "candidates": candidates,
                "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
                "study_specific_pool_size": 6,
                "global_pool_changed": False,
                "mechanism_labels_are_independent_risks": False,
                "source_financial_rejections_preserved": True,
                "native_RF_exact_common_no_view": True,
                "source_models_targets_or_accounts_replayed": False,
                "new_source_requests": 0,
                "new_economic_factor_definitions": 0,
                "raw_prices_targets_or_daily_rows_exported": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
    return reference(PUBLIC)


def main():
    parser = argparse.ArgumentParser(description="Frozen mixed-mechanism six financial cohort.")
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use an actual published complete two-candidate preregistration commit.")
    actions = {"freeze": initialize, "register": register, "export": export}
    print(
        json.dumps(evaluate(args.commit) if args.action == "evaluate" else actions[args.action]())
    )


if __name__ == "__main__":
    main()
