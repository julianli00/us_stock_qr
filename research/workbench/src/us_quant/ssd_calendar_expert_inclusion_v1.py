from __future__ import annotations

import argparse
import ast
import json
import os
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import pandas as pd

from us_quant import calendar_expert_inclusion_v1 as rules
from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import ssd_eight_expert_meta_v1 as template
from us_quant.calendar import previous_session
from us_quant.calendar_program_expert_v1 import load_calendar_expert
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
source = seven
reference, unique_refs = base.reference, template.unique_refs
FOLDER = base.REPORTS / "calendar-expert-inclusion-20261105-v1"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
PREPARED = FOLDER / "prepared-target-bundle-v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC = ROOT / "evidence/calendar_expert_inclusion_20261105_results_v1.json"
CANDIDATES = rules.CANDIDATES
IDS = tuple(row["id"] for row in CANDIDATES)
PACKETS = {
    name: ROOT / "evidence" / f"calendar_expert_inclusion_20261105_{index}_registration_v1.json"
    for index, name in enumerate(IDS)
}
OWN = (
    "config/calendar-expert-inclusion-v1.json",
    "src/us_quant/calendar_expert_inclusion_v1.py",
    "src/us_quant/ssd_calendar_expert_inclusion_v1.py",
    "src/us_quant/calendar_program_expert_v1.py",
    "tests/test_calendar_program_expert_v1.py",
    "tests/test_calendar_expert_inclusion_v1.py",
    "tests/test_macro_expert_inclusion_v1.py",
    "CALENDAR_EXPERT_INCLUSION_V1.md",
    "tests/test_strategy_expert_pool_eight_v1.py",
    "tests/test_strategy_expert_pool_v5.py",
    "tests/test_strategy_expert_pool_v4.py",
    "tests/test_strategy_expert_pool_v3.py",
    "tests/conftest.py",
)


def bound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


def events(kind, identifier):
    return bound(template.events)(kind, identifier)


def check_frozen(frozen):
    template.check_frozen(frozen)


def progress(phase, **fields):
    counts = current_counts(base.journal_events(base.JOURNAL))
    if any(key in fields and fields[key] != value for key, value in counts.items()):
        raise QuantError("Calendar inclusion cannot override actual journal counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "calendar_inclusion_" + phase,
            "study_id": read_json(rules.POLICY)["study_id"],
            "active_compute_job": None,
            **counts,
            **fields,
        },
    )


def dependencies():
    pending = [ROOT / name for name in OWN if name.endswith(".py")]
    result = {}
    while pending:
        path = pending.pop()
        name = path.relative_to(ROOT).as_posix()
        if name in result:
            continue
        result[name] = reference(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
                if node.module == "us_quant":
                    names.extend("us_quant." + alias.name for alias in node.names)
            for name in names:
                if name.startswith("us_quant"):
                    path = ROOT / "src" / (name.replace(".", "/") + ".py")
                    package = ROOT / "src" / name.replace(".", "/") / "__init__.py"
                    pending.extend(
                        candidate for candidate in (path, package) if candidate.is_file()
                    )
    return unique_refs([*result.values(), *[reference(ROOT / path) for path in OWN]])


def include(experts, calendar):
    if (
        tuple(expert.identity.identifier for expert in experts) != template.rules.EXPERT_IDS
        or calendar.identity.identifier != rules.EXPERT_IDS[-1]
        or calendar.identity.mechanism_group != rules.MECHANISMS[-1]
    ):
        raise QuantError("Append only the complete admitted calendar policy to the original eight.")
    return (*experts, calendar)


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    data, experts, _, refs, clocks, benchmarks = template.source_inputs()
    calendar, technical = load_calendar_expert(
        policy["calendar_source"], tuple(data.close.columns), data.risk_free
    )
    experts = include(experts, calendar)
    for expert in experts:
        initial = expert.observe(pd.Timestamp("2016-10-05"), 126)
        final = expert.observe(data.close.index[-1], 2512)
        if (
            initial.target is None
            or initial.score_mature
            or len(initial.net_returns)
            or len(final.net_returns) != 2512
            or not final.risk_free.equals(data.risk_free.loc[final.risk_free.index])
        ):
            raise QuantError(
                "Keep complete opinions, explicit initial maturity and exact common RF."
            )
    completion = read_json(base.checked_reference(policy["calendar_source"]["completion"]))
    refs = unique_refs(
        [
            *refs,
            *completion["exact_evidence_references"],
            *[
                policy["calendar_source"][key]
                for key in ("registration", "results", "admission", "completion")
            ],
            policy["matched_original_eight_results"],
            technical["target_reference"],
            technical["shadow_reference"],
            reference(template.INTENT),
        ]
    )
    for row in refs:
        base.checked_reference(row)
    clocks = [
        *clocks,
        {
            "id": calendar.identity.identifier,
            "source_policy_registered_at": technical["introduced_at"],
            "actual_expert_admitted_at": technical["actual_expert_admitted_at"],
            "source_packet_schema": 3,
            "source_scope": "unchanged_published_calendar_policy_and_original_admitted_market",
            "native_RF_exact_common": True,
            "study_specific_inclusion_not_global_roster_promotion": True,
            "decision_information_clock": rules.CLOCK,
        },
    ]
    return data, experts, read_json(template.INTENT)["frozen"], refs, clocks, benchmarks


def validation_refs():
    proof = read_json(FOLDER / "validation-source-v1.json")
    xml, log = [base.checked_reference(row) for row in proof["test_outputs"]]
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    if (
        proof["implementation_and_test_inputs"] != [reference(ROOT / path) for path in OWN]
        or proof["exit_code"] != 0
        or xml.parent != FOLDER
        or log.parent != FOLDER
        or not suites
        or sum(int(row.get("tests", 0)) for row in suites) != proof["passed_tests"]
        or any(int(row.get(key, 0)) for row in suites for key in ("failures", "errors", "skipped"))
    ):
        raise QuantError("Bind exact source-Adapter, 36-pair and complete-cohort fence checks.")
    return [*proof["test_outputs"], reference(FOLDER / "validation-source-v1.json")]


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve complete or partial first calendar inclusion preparation.")
    with lifecycle.research_job():
        design = read_json(DESIGN)
        if (
            design["actual_native_event_seq"] != 158328
            or design["actual_new_meta_targets_or_accounts_computed"] is not False
            or design["candidates"] != list(CANDIDATES)
        ):
            raise QuantError(
                "Freeze the independently declared calendar inclusion before outcomes."
            )
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, _, parent, refs, clocks, _ = source_inputs()
        counts = current_counts(base.journal_events(base.JOURNAL))
        if (
            counts["current_journal_event_count"] != 934
            or counts["current_journal_head_sha256"] != rules.PRIOR_HEAD
        ):
            raise QuantError("Keep the actual J934/256 pre-outcome baseline.")
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": read_json(rules.POLICY)["study_id"],
            "policy": reference(rules.POLICY),
            "expert_pool": reference(rules.POLICY),
            "construction": "controlled_calendar_source_inclusion_unchanged_equal_and_rolling_laws",
            "expert_pool_definition_role": "study_specific9_original7and8_rosters_immutable",
            "prior_journal_sequence": 934,
            "prior_journal_head_sha256": rules.PRIOR_HEAD,
            "prior_completed_configurations": 256,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in [*deps, *refs] if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / "CALENDAR_EXPERT_INCLUSION_V1.md"),
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
            "calendar_source_RF_exact_common": True,
            "new_RF_or_source_scope_waiver": False,
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
                "original_active_rosters_changed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="register_both_before_actual_scores_targets")
    return reference(INTENT)


def inputs():
    return bound(template.inputs)()


def verify_targets(*args):
    return bound(template.verify_targets)(*args)


def verify_retained_records(*args):
    return bound(template.verify_retained_records)(*args)


def prepare(data, experts, frozen):
    if PREPARED.exists():
        saved = read_json(PREPARED)
        if saved["frozen_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting completed calendar inclusion targets.")
        for entry in saved["entries"].values():
            for row in entry.values():
                base.checked_reference(row)
        return saved["entries"]
    if (FOLDER / "preparation-attempt-v1.json").exists():
        raise QuantError("Preserve partial actual derivation instead of generating it twice.")
    if any(len(events("registered", candidate["id"])) != 1 for candidate in CANDIDATES):
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

        targets, audits = rules.build_targets(
            experts,
            data.close.index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            before_decision=lifecycle.runtime_guard,
            retain=retain,
        )
    retained = verify_retained_records(log, experts, targets, audits)
    independent = verify_targets(data, experts, targets, audits)
    if len(audits) != 121:
        raise QuantError("Preserve every original 121 capital/month-end decision.")
    prefixes = []
    for position in (0, 6, 12, 60, 120):
        day = pd.Timestamp(audits[position]["decision_session"])
        early, earlier = rules.build_targets(
            experts,
            data.close.loc[:day].index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            before_decision=lifecycle.runtime_guard,
        )
        if any(not early[name].equals(targets[name].loc[:day]) for name in IDS) or (
            earlier != audits[: position + 1]
        ):
            raise QuantError("Future data changed a prior nine-source inclusion decision.")
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
                asset
                for asset in ("MTUM", "VLUE", "QUAL", "USMV", "IJR", "PKW")
                if (rows[asset] > 1e-10).any()
            ]
            if known.empty or len(factors) < 3:
                raise QuantError("Both original funded horizons need known anchors and factors.")
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
                "all_pairs_per_mature_decision": 36,
                "prefix_truncations": prefixes,
                "formal_factor_coverage": coverage,
                "actual_input_resolved_records": reference(log),
                "calendar_source_financial_rejection_preserved": True,
                "calendar_native_RF_exact_common_no_view": True,
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
            raise QuantError("Preserve the terminal cohort identity without retry.")
        for candidate in CANDIDATES:
            name = candidate["id"]
            if not events("registered", name):
                base.append_event(
                    base.JOURNAL, "registered", name, {"candidate": candidate, "frozen": frozen}
                )
            elif events("registered", name)[0]["body"] != {
                "candidate": candidate,
                "frozen": frozen,
            }:
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
                        "cohort_registrations": [
                            events("registered", value)[0]["sha256"] for value in IDS
                        ],
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
                            "stage": "calendar_cohort_preparation",
                            "reason": str(error),
                            "partial_inputs_preserved": True,
                        },
                    )
            write_new_receipt(
                FOLDER / "preparation-failure-v1.json",
                {
                    "failed_at": utc_now(),
                    "error": str(error),
                    "old_source_or_failed_algorithm_rescue": False,
                },
            )
            progress("technical_failure_preserved", next_step="do_not_retry_or_retune_cohort")
            raise
        progress(
            "registered_before_accounts", next_step="publish_both_exact_packets_before_funding"
        )
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
            base.checked_reference(read_json(rules.POLICY)["matched_original_eight_results"])
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
                raise QuantError("Export every original funded path, not passing fragments.")
            candidates[name] = {
                "status": data["review"]["status"],
                "paths": data["paths"],
                "matched_cached_original_eight_control": {
                    "candidate_id": control,
                    "paths": controls[control]["paths"],
                    "old_control_replayed": False,
                },
            }
        seven.common.retain_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "study_id": read_json(rules.POLICY)["study_id"],
                "completed_at": saved["completed_at"],
                "candidates": candidates,
                "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
                "study_specific_pool_size": 9,
                "deliberate_calendar_inclusion_not_automatic_global_ninth": True,
                "original_active_rosters_changed": False,
                "calendar_source_financial_rejection_preserved": True,
                "calendar_source_RF_exact_common_no_view": True,
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
    parser = argparse.ArgumentParser(description="Controlled complete calendar expert inclusion.")
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use actual remotely published complete-cohort preregistration.")
    actions = {"freeze": initialize, "register": register, "export": export}
    print(
        json.dumps(evaluate(args.commit) if args.action == "evaluate" else actions[args.action]())
    )


if __name__ == "__main__":
    main()
