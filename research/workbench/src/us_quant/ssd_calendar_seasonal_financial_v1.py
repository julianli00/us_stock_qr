from __future__ import annotations

import argparse
import ast
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import calendar_seasonal_expert_v1 as law
from us_quant import main_capital_feedback_v1 as capital
from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import ssd_main_capital_feedback_v1 as stored_program
from us_quant import ssd_research_v1 as base
from us_quant.calendar import is_month_end, previous_session
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.research_program import review_market
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.ssd_variance_disagreement_research_v1 import verify_feedback
from us_quant.storage import (
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)
from us_quant.strategy_experts_v1 import ArchivedMonthlyExpert, ExpertIdentity

ROOT, ID = base.ROOT, law.ID
STUDY = "ssd_calendar_seasonal_financial_20261105_v1"
POLICY = ROOT / "config/calendar-seasonal-financial-v1.json"
FOLDER = base.REPORTS / "calendar-seasonal-financial-20261105-v1"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
READY = FOLDER / "program-readiness-v1.json"
PREPARED = FOLDER / "published-native-targets-v1.json"
RESULTS = FOLDER / "results.json"
PACKET = ROOT / "evidence/calendar_seasonal_financial_20261105_registration_v1.json"
PUBLIC = ROOT / "evidence/calendar_seasonal_financial_20261105_results_v1.json"
ADMISSION = FOLDER / "technical-admission-v1.json"
PUBLIC_ADMISSION = ROOT / "evidence/calendar_seasonal_financial_20261105_admission_v1.json"
KIND = "causal_main_capital_decision_program_v1"
OWN = (
    "config/calendar-seasonal-financial-v1.json",
    "src/us_quant/ssd_calendar_seasonal_financial_v1.py",
    "tests/test_ssd_calendar_seasonal_financial_v1.py",
    "CALENDAR_SEASONAL_FINANCIAL_V1.md",
    "config/calendar-seasonal-expert-v1.json",
    "src/us_quant/calendar_seasonal_expert_v1.py",
    "tests/test_calendar_seasonal_expert_v1.py",
    "CALENDAR_SEASONAL_EXPERT_V1.md",
)
reference = base.reference


def unique_refs(rows):
    result = {}
    for row in rows:
        if row["path"] in result and result[row["path"]] != row:
            raise QuantError("One dependency cannot have conflicting frozen identities.")
        result[row["path"]] = row
    return list(result.values())


def events(kind):
    return [
        row
        for row in base.journal_events(base.JOURNAL)
        if row["kind"] == kind and row["candidate_id"] == ID
    ]


def progress(phase, **fields):
    counts = current_counts(base.journal_events(base.JOURNAL))
    if any(key in fields and fields[key] != value for key, value in counts.items()):
        raise QuantError("Calendar progress cannot override actual journal counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "calendar_financial_" + phase,
            "study_id": STUDY,
            "active_compute_job": None,
            **counts,
            **fields,
        },
    )


def packages():
    return sorted((d.metadata["Name"], d.version) for d in distributions())


def dependencies():
    pending = [ROOT / path for path in OWN if path.endswith(".py")]
    pending.extend(
        ROOT / path
        for path in (
            "tests/test_main_capital_feedback_v1.py",
            "tests/test_expert_initial_monthly_execution_v1.py",
            "tests/conftest.py",
        )
    )
    result = {}
    while pending:
        path = pending.pop()
        relative = path.relative_to(ROOT).as_posix()
        if relative in result:
            continue
        result[relative] = reference(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
                if node.module == "us_quant":
                    names.extend("us_quant." + alias.name for alias in node.names)
            for name in names:
                if not name.startswith("us_quant"):
                    continue
                module = ROOT / "src" / (name.replace(".", "/") + ".py")
                package = ROOT / "src" / name.replace(".", "/") / "__init__.py"
                pending.extend(path for path in (module, package) if path.is_file())
    return unique_refs([*result.values(), *[reference(ROOT / path) for path in OWN]])


def check_frozen(frozen):
    base.check_frozen_files(frozen)
    for row in frozen["prior_evidence"] + frozen["validation_evidence"]:
        base.checked_reference(row)


def validate_policy(policy):
    fixed = {
        "schema_version": 1,
        "study_id": STUDY,
        "candidate_id": ID,
        "candidate_count": 1,
        "study_size_is_a_total_research_quota": False,
        "prior_journal_sequence": 922,
        "prior_journal_head_sha256": (
            "7913272587b33cd7573578cfed3fc34519174690caf877533e875117f36a52e8"
        ),
        "prior_completed_configurations": 255,
        "target_delivery_kind": KIND,
        "real_source_targets_before_actual_publication": False,
        "historical_target_proof_stage": "after_actual_published_program_execution",
        "initialization": (
            "fresh_capital_at_real_pre_return_anchor_using_last_known_native_month_end"
        ),
        "ordinary_decisions": "genuine_NYSE_month_end_after_existing_16_15_ET_clock",
        "capital_usd_per_formal_path": 10000.0,
        "initial_drawdown_peak_usd": 10000.0,
        "target_budget": 0.98,
        "actual_weight_band": 0.025,
        "formal_paths": ["10y-base", "10y-stress", "5y-base", "5y-stress"],
        "prefix_audit_ten_year_base_return_positions": [125, 503, 1006, 1509, 2012],
        "auxiliary_prefixes_are_independent_economic_trials": False,
        "native_source_target_prefixes": "first_seventh_thirteenth_middle_last_month_end",
        "primary_policy_or_market_or_RF_scope_changed": False,
        "new_quote_or_broker_requests": 0,
        "new_economic_factor_definitions": 0,
        "registered_expert_or_pool_admission": False,
        "standalone_financial_failure_is_technical_exclusion": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    references = {
        "primary_policy",
        "primary_preparation_freeze",
        "primary_preparation_completion",
        "market_parent_freeze",
    }
    if (
        not isinstance(policy, dict)
        or set(policy) != set(fixed) | references
        or any(
            policy.get(key) != value or type(policy.get(key)) is not type(value)
            for key, value in fixed.items()
        )
    ):
        raise QuantError("Freeze the prepared calendar law and original funded contract once.")
    if policy["primary_policy"] != reference(law.POLICY):
        raise QuantError("The primary calendar policy changed after preparation.")
    for key in references:
        base.checked_reference(policy[key])
    primary = read_json(base.checked_reference(policy["primary_preparation_freeze"]))
    if (
        primary["prior_completed_configurations"] != 255
        or primary["prior_journal_sequence"] != 922
        or primary["registered"] is not False
        or primary["real_new_policy_targets_accounts_or_metrics_computed"] is not False
    ):
        raise QuantError("Preparation is not a prior financial execution or inherited count228.")
    for row in primary["implementation_and_test_inputs"]:
        base.checked_reference(row)
    law.validate(read_json(law.POLICY))


def validation_refs():
    xml = FOLDER / "financial-tests-v3.xml"
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    proof = read_json(FOLDER / "validation-source-v3.json")
    if not suites or any(
        int(s.get(key, 0)) for s in suites for key in ("failures", "errors", "skipped")
    ):
        raise QuantError("Complete the synthetic calendar program and publication-fence tests.")
    if (
        proof["implementation_and_test_inputs"] != [reference(ROOT / path) for path in OWN]
        or proof["passed_tests"] != sum(int(s.get("tests", 0)) for s in suites)
        or proof["ruff_returncode"] != 0
        or proof["synthetic_only"] is not True
    ):
        raise QuantError("Synthetic validation must bind the exact financial program.")
    return [
        reference(FOLDER / path)
        for path in (
            "validation-source-v3.json",
            "financial-tests-v3.xml",
            "financial-tests-v3.log",
            "financial-ruff-v3.log",
        )
    ]


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve complete or partial calendar financial preparation.")
    with lifecycle.research_job():
        policy = read_json(POLICY)
        validate_policy(policy)
        counts = current_counts(base.journal_events(base.JOURNAL))
        if counts["current_journal_event_count"] != 922 or (
            counts["current_journal_head_sha256"] != policy["prior_journal_head_sha256"]
        ):
            raise QuantError("Require the actual closedJ922 baseline before new outcomes.")
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        parent = read_json(base.checked_reference(policy["market_parent_freeze"]))["frozen"]
        check_frozen(parent)
        data = review_market({"market": parent["market"]}, ROOT)
        law.checked_close(data.close)
        deps = dependencies()
        removed = {
            "expert_pool",
            "expert_pool_definition_role",
            "source_policy_and_expert_admission_clocks",
            "new_source_model_or_shadow_account_runs",
        }
        frozen = {
            **{
                key: value
                for key, value in parent.items()
                if key not in removed and not key.startswith("source_")
            },
            "study_id": STUDY,
            "policy": reference(POLICY),
            "primary_signal_policy": policy["primary_policy"],
            "execution_policy": reference(POLICY),
            "account_namespace": "calendar_seasonal_published_program_v1",
            "construction": law.CANDIDATE["construction"],
            "construction_inputs": [policy["primary_policy"]],
            "decision_program_kind": KIND,
            "expert_pool_definition_role": "one_complete_source_policy_not_pool_expansion",
            "main_capital_initialization": policy["initialization"],
            "prepublication_readiness": "synthetic_Interface_and_pinned_market_identity_only",
            "historical_target_proof_stage": policy["historical_target_proof_stage"],
            "prior_journal_sequence": 922,
            "prior_journal_head_sha256": policy["prior_journal_head_sha256"],
            "prior_completed_configurations": 255,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[row for row in deps if row["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / "CALENDAR_SEASONAL_EXPERT_V1.md"),
                    reference(ROOT / "CALENDAR_SEASONAL_FINANCIAL_V1.md"),
                ]
            ),
            "policies": unique_refs(
                [*parent["policies"], *[r for r in deps if r["path"].startswith("config/")]]
            ),
            "prior_evidence": unique_refs(
                [
                    *parent["prior_evidence"],
                    *[
                        policy[key]
                        for key in (
                            "primary_preparation_freeze",
                            "primary_preparation_completion",
                            "market_parent_freeze",
                        )
                    ],
                    reference(FOLDER / "actual-invocation-v1.json"),
                    reference(FOLDER / "financial-design-v1.json"),
                ]
            ),
            "validation_evidence": unique_refs(
                [
                    *parent["validation_evidence"],
                    *validation_refs(),
                    reference(FOLDER / "preflight-failure-v1.json"),
                    reference(FOLDER / "financial-tests-v1.xml"),
                    reference(FOLDER / "financial-tests-v1.log"),
                    reference(FOLDER / "preflight-process-v2.json"),
                    reference(FOLDER / "financial-tests-v2.xml"),
                    reference(FOLDER / "financial-tests-v2.log"),
                ]
            ),
            "decision_information_clock": law.CLOCK,
            "new_source_requests": 0,
            "new_economic_factor_definitions": 0,
            "new_shadow_account_runs": 0,
            "new_shadow_recommendation_artifacts": 0,
        }
        check_frozen(frozen)
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "frozen": frozen,
                "complete_dependency_inputs": deps,
                "package_versions": packages(),
                "new_real_source_targets_accounts_or_metrics_computed": False,
                "actual_registration_performed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="register_program_without_real_source_outputs")
    return reference(INTENT)


def inputs():
    lifecycle.runtime_guard()
    intent = read_json(INTENT)
    for row in intent["complete_dependency_inputs"]:
        base.checked_reference(row)
    if intent["package_versions"] != [list(row) for row in packages()]:
        raise QuantError("The accepted pinned calendar financial environment changed.")
    frozen = {
        **intent["frozen"],
        "prior_evidence": unique_refs([*intent["frozen"]["prior_evidence"], reference(INTENT)]),
    }
    check_frozen(frozen)
    data = review_market({"market": frozen["market"]}, ROOT)
    law.checked_close(data.close)
    references = [
        row
        for row in frozen["benchmark_artifacts"]
        if row["path"].endswith("/benchmarks-20261027-v1/results.json")
    ]
    if len(references) != 1:
        raise QuantError("Reuse the one exact complete SPY/QQQ benchmark, never replay it.")
    return data, frozen, read_json(base.checked_reference(references[0]))


def packet_body(frozen, registered, readiness):
    return {
        "schema_version": 3,
        "program_id": "ssd_continuous_research_v1",
        "study_id": STUDY,
        "target_delivery_kind": KIND,
        "complete_fixed_cohort_ids": [ID],
        "cohort_registrations": [registered["sha256"]],
        "entries": [
            {
                "candidate_id": ID,
                "specification": {
                    "financial_criteria_id": "ssd_dual_benchmark_net_cagr_v1",
                    "decision_program_kind": KIND,
                    "frozen": frozen,
                },
                "registered_event_sha256": registered["sha256"],
                "registered_at": registered["at"],
                "program_readiness": readiness,
                "historical_target_derivation_proof": None,
            }
        ],
        "historical_new_policy_targets_computed": False,
        "candidate_accounts_or_metrics_computed": False,
        "original_source_signal_recomputed": False,
        "historical_target_proof_deferred_until_published_execution": True,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }


def checked_packet(frozen):
    registered = events("registered")
    if len(registered) != 1 or registered[0]["body"] != {
        "candidate": law.CANDIDATE,
        "frozen": frozen,
    }:
        raise QuantError("Require one actual unique calendar-program registration.")
    packet = read_json(PACKET)
    if {k: v for k, v in packet.items() if k != "created_at"} != packet_body(
        frozen, registered[0], reference(READY)
    ):
        raise QuantError("The complete published program cannot become a fake target proof.")
    ready = read_json(READY)
    if (
        ready["readiness_kind"] != "synthetic_Interface_and_pinned_market_identity_only"
        or ready["historical_new_policy_targets_computed"] is not False
        or ready["historical_target_derivation_verified"] is not False
    ):
        raise QuantError("Synthetic calendar readiness is not real historical target proof.")
    return packet


def register():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        _, frozen, _ = inputs()
        if events("registered") or events("technical_failure") or PACKET.exists():
            raise QuantError("Keep an existing actual calendar registration, never duplicate it.")
        registered = base.append_event(
            base.JOURNAL, "registered", ID, {"candidate": law.CANDIDATE, "frozen": frozen}
        )
        write_new_receipt(
            READY,
            {
                "checked_at": utc_now(),
                "readiness_kind": "synthetic_Interface_and_pinned_market_identity_only",
                "implementation": reference(INTENT),
                "validation": validation_refs(),
                "historical_new_policy_targets_computed": False,
                "historical_target_derivation_verified": False,
                "future_price_or_main_account_simulation_used": False,
            },
        )
        write_new_receipt(
            PACKET, {**packet_body(frozen, registered, reference(READY)), "created_at": utc_now()}
        )
        checked_packet(frozen)
        progress("registered_before_outputs", next_step="actual_publication_before_native_targets")
    return reference(PACKET)


def publication(frozen, commit):
    checked_packet(frozen)
    verifier = FunctionType(
        base.published_inputs.__code__,
        {**base.published_inputs.__globals__, "PACKET": PACKET},
        argdefs=base.published_inputs.__defaults__,
    )
    return verifier(frozen, commit)


def publication_fence():
    rows = [
        row
        for row in base.journal_events(base.JOURNAL)
        if row["kind"] == "publication" and row["body"].get("study_id") == STUDY
    ]
    if len(rows) != 1 or len(events("registered")) != 1 or events("technical_failure"):
        raise QuantError("Actual registration and publication must precede real calendar outputs.")
    return rows[0]


def verify_targets(close, targets, audits, records):
    dates = pd.DatetimeIndex([day for day in close.index if is_month_end(day)])
    if (
        not targets.index.equals(close.index)
        or not targets.columns.equals(close.columns)
        or not targets.dropna(how="all").index.equals(dates)
        or len(audits) != len(dates)
        or len(records) != 2 * len(dates)
    ):
        raise QuantError("Preserve the whole genuine native calendar without a fake anchor.")
    for position, day in enumerate(dates):
        selected = list(law.CORE) if day.month in (10, 11, 12, 1, 2, 3) else ["BIL"]
        oracle = np.array(
            [0.98 / len(selected) if name in selected else 0.0 for name in law.UNIVERSE]
        )
        before, after = records[2 * position : 2 * position + 2]
        if (
            before
            != {
                "record": "decision_input",
                "decision_session": str(day.date()),
                "information_clock": law.CLOCK,
                "intended_calendar_month": 1 if day.month == 12 else day.month + 1,
                "future_price_or_return_observed": False,
            }
            or after["record"] != "decision_resolved"
            or after["audit"] != audits[position]
            or after["complete_target"] != targets.loc[day].tolist()
            or after["asset_order"] != list(close.columns)
            or audits[position]["selected_assets"] != selected
            or not np.array_equal(oracle, targets.loc[day].to_numpy())
        ):
            raise QuantError(
                "Independent source month, target or input-before-output proof failed."
            )
    return {"native_month_end_opinions": len(dates), "all_independent_calendar_targets_exact": True}


def prepare_source(data, frozen):
    fence = publication_fence()
    if PREPARED.exists():
        old = read_json(PREPARED)
        if old["frozen_specification_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting completed calendar source preparation.")
        for key in ("targets", "proof"):
            base.checked_reference(old[key])
        return old
    attempt = FOLDER / "source-targets-started-v1.json"
    if attempt.exists():
        raise QuantError("Preserve incomplete calendar targets instead of generating them twice.")
    write_new_receipt(
        attempt,
        {
            "started_at": utc_now(),
            "publication_event_sha256": fence["sha256"],
            "real_native_targets_previously_computed": False,
        },
    )
    trace = FOLDER / "native-inputs-resolved-v1.jsonl"
    with trace.open("x") as stream:

        def retain(row):
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        targets, audits = law.build_targets(
            data.close,
            read_json(law.POLICY),
            before_decision=lifecycle.runtime_guard,
            retain=retain,
        )
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    checked = verify_targets(data.close, targets, audits, records)
    prefixes = []
    for position in sorted({0, 6, 12, len(audits) // 2, len(audits) - 1}):
        day = pd.Timestamp(audits[position]["decision_session"])
        prefix, earlier = law.build_targets(
            data.close.loc[:day],
            read_json(law.POLICY),
            before_decision=lifecycle.runtime_guard,
        )
        if not prefix.equals(targets.loc[:day]) or earlier != audits[: position + 1]:
            raise QuantError("Future inputs changed an earlier native calendar opinion.")
        prefixes.append(str(day.date()))
    active = targets.dropna(how="all")
    for window in read_json(base.PROTOCOL)["windows"]:
        anchor = previous_session(window["start"])
        known = active.loc[:anchor].tail(1)
        covered = pd.concat([known, active.loc[window["start"] : window["end"]]])
        if known.empty or sum((covered[name] > 1e-10).any() for name in law.CORE) < 3:
            raise QuantError("Retain known original anchors and actual factor participation.")
    directory = FOLDER / "native-source"
    new_output_directory(directory)
    write_text_atomic(directory / "targets.csv", targets.to_csv(float_format="%.17g"))
    write_new_receipt(directory / "audits.json", {"decisions": audits})
    write_new_receipt(
        directory / "proof.json",
        {
            "checked_at": utc_now(),
            "publication_event_sha256": fence["sha256"],
            "source_start": reference(attempt),
            "frozen_specification_sha256": digest_json(frozen),
            **checked,
            "prefix_truncations": prefixes,
            "whole_native_prehistory_retained": True,
            "synthetic_initial_recommendation_added": False,
            "real_native_targets_first_generated_after_actual_publication": True,
            "candidate_accounts_or_metrics_computed": False,
            "original_source_signal_recomputed": False,
            "retained_input_resolved_trace": reference(trace),
        },
    )
    prepared = {
        "frozen_specification_sha256": digest_json(frozen),
        "targets": reference(directory / "targets.csv"),
        "proof": reference(directory / "proof.json"),
    }
    write_new_receipt(PREPARED, prepared)
    return prepared


def source_frame(prepared):
    return pd.read_csv(
        base.checked_reference(prepared["targets"]),
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )


def verify_opinions(output, targets):
    active = targets.dropna(how="all")
    for row in output.decisions:
        day = pd.Timestamp(row["signal_session"])
        known = active.loc[:day].tail(1)
        if (
            known.empty
            or row["assessment"]["source_opinion_known_at"] != str(known.index[0].date())
            or not np.array_equal(row["desired_target"], known.iloc[0].to_numpy())
            or known.index[0] > day
        ):
            raise QuantError("The main book used a future or different source calendar opinion.")
    return {"all_actual_source_opinions_and_known_at_dates_exact": True}


def run_calendar(data, targets, start, end, *, cost_bps, delay, retain=None):
    output = capital.run(
        data,
        capital.ArchivedOpinionPolicy(targets),
        start,
        end,
        cost_bps=cost_bps,
        delay=delay,
        before_decision=lifecycle.runtime_guard,
        retain=retain,
    )
    frame = output.result.frame.copy()
    frame["gross_exposure"] = output.result.weights.apply(lambda row: float(row.sum()), axis=1)
    return replace(output, result=capital.BacktestResult(frame, output.result.weights))


def evaluate_path(data, targets, frozen, window, scenario, benchmarks):
    key = f"{window['years']}y-{scenario['id']}"
    complete = [row for row in events("path_completed") if row["body"]["path_key"] == key]
    if complete:
        if len(complete) != 1:
            raise QuantError("A completed calendar path cannot be selected among repeats.")
        return base.verify_path(complete[0]["body"]["receipt"], frozen, ID, key)
    if any(row["body"]["path_key"] == key for row in events("path_started")):
        raise QuantError("Preserve a partial calendar account without replay.")
    publication_fence()
    lifecycle.runtime_guard()
    directory = FOLDER / "accounts" / key
    base.append_event(
        base.JOURNAL,
        "path_started",
        ID,
        {
            "path_key": key,
            "directory": directory.relative_to(ROOT).as_posix(),
            "frozen_specification_sha256": digest_json(frozen),
        },
    )
    new_output_directory(directory)
    trace = directory / "chronological-trace.jsonl"
    with trace.open("x") as stream:

        def retain(row):
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()

        output = run_calendar(
            data,
            targets,
            window["start"],
            window["end"],
            cost_bps=scenario["cost_bps_per_side"],
            delay=scenario["delay_sessions"],
            retain=retain,
        )
    feedback = verify_feedback(
        data, output.desired, output.result, output.issued, output.decisions, window, scenario
    )
    actual_state = capital.verify_capital_inputs(output)
    opinions = verify_opinions(output, targets)
    independent = base.independent_equity(
        data,
        output.issued,
        window["start"],
        window["end"],
        capital=10000,
        cost_bps=scenario["cost_bps_per_side"],
        commission=1,
        delay=scenario["delay_sessions"],
    )
    own = base.performance(output.result.frame["return"], output.result.frame["risk_free"])
    checked, difference = base.independent_metrics(
        output.result.frame, independent, data.risk_free.loc[output.result.frame.index], 10000
    )
    base.check_metrics(checked, own)
    lookup = {
        (r["years"], r["scenario"], r["symbol"]): r["metrics"] for r in benchmarks["benchmarks"]
    }
    spy, qqq = [lookup[window["years"], scenario["id"], name] for name in ("SPY", "QQQ")]
    criteria = base.gates(own, spy, qqq, lookup[window["years"], "base", "SPY"], scenario["id"])
    artifacts = stored_program.write_run(directory, output)
    write_text_atomic(directory / "independent-bt.csv", independent.to_csv(float_format="%.17g"))
    artifacts.extend([reference(trace), reference(directory / "independent-bt.csv")])
    receipt = {
        "completed_at": utc_now(),
        "candidate_id": ID,
        "path_key": key,
        "frozen_specification_sha256": digest_json(frozen),
        "metrics": own,
        "execution": base.execution_statistics(
            output.result, data, 10000, scenario["cost_bps_per_side"], 1
        ),
        "gates": criteria,
        "all_point_gates_pass": all(criteria.values()),
        "excess_net_cagr_vs_spy": own["cagr"] - spy["cagr"],
        "excess_net_cagr_vs_qqq": own["cagr"] - qqq["cagr"],
        "independent_bt_max_equity_difference_usd": difference,
        "independent_monthly_feedback": feedback,
        "actual_capital_input_verification": actual_state,
        "actual_source_opinion_verification": opinions,
        "artifacts": artifacts,
        "independent_forward_validation": False,
        "broker_orders": 0,
    }
    write_new_receipt(directory / "complete.json", receipt)
    base.append_event(
        base.JOURNAL,
        "path_completed",
        ID,
        {
            "path_key": key,
            "receipt": reference(directory / "complete.json"),
            "recovered_actual_complete_receipt": False,
        },
    )
    return receipt


def prefix_audits(data, targets):
    folder = FOLDER / "main-capital-prefix-audits-v1"
    new_output_directory(folder)
    source = FOLDER / "accounts/10y-base"
    saved = {
        name: pd.read_csv(
            source / name, index_col=0, parse_dates=True, float_precision="round_trip"
        )
        for name in ("account.csv", "weights.csv", "desired.csv", "issued.csv")
    }
    decisions = read_json(source / "decisions.json")["decisions"]
    prefixes = []
    for position in read_json(POLICY)["prefix_audit_ten_year_base_return_positions"]:
        lifecycle.runtime_guard()
        day = saved["account.csv"].index[position]
        data_prefix = MarketData(
            data.open.loc[:day],
            data.close.loc[:day],
            data.raw_close.loc[:day],
            data.volume.loc[:day],
            data.risk_free.loc[:day],
        )
        output = run_calendar(
            data_prefix,
            targets.loc[:day],
            "2016-10-06",
            str(day.date()),
            cost_bps=5,
            delay=1,
        )
        for name, frame in (
            ("account.csv", output.result.frame),
            ("weights.csv", output.result.weights),
            ("desired.csv", output.desired),
            ("issued.csv", output.issued),
        ):
            if not np.array_equal(
                frame.loc[:day].to_numpy(dtype=float),
                saved[name].loc[:day].to_numpy(dtype=float),
                equal_nan=True,
            ) or not frame.loc[:day].index.equals(saved[name].loc[:day].index):
                raise QuantError(
                    "Real calendar main-book prefix differs from its full published run."
                )
        if output.decisions != [
            row for row in decisions if pd.Timestamp(row["signal_session"]) <= day
        ]:
            raise QuantError("An earlier main decision changed in a real calendar prefix.")
        directory = folder / f"{position:04d}-{day.date()}"
        new_output_directory(directory)
        prefixes.append(
            {
                "last_session": str(day.date()),
                "formal_return_position": position,
                "capital_and_decisions_exact": True,
                "artifacts": stored_program.write_run(directory, output),
            }
        )
    path = folder / "completion.json"
    write_new_receipt(
        path,
        {
            "completed_at": utc_now(),
            "prefixes": prefixes,
            "auxiliary_main_book_audit_runs": 5,
            "extra_economic_rules_or_qualifying_windows": 0,
        },
    )
    return reference(path)


def record_failure(stage, error):
    interrupted = lifecycle.phase(datetime.now(timezone.utc)) != "research"
    if events("registered") and not events("review") and not events("technical_failure"):
        if not interrupted:
            base.append_event(
                base.JOURNAL,
                "technical_failure",
                ID,
                {"stage": stage, "reason": str(error), "partial_outputs_retained": True},
            )
    path = FOLDER / f"{stage}-failure-v1.json"
    if not path.exists():
        write_new_receipt(
            path,
            {
                "failed_at": utc_now(),
                "stage": stage,
                "reason": str(error),
                "authorization_interrupted": interrupted,
                "partial_outputs_retained": True,
            },
        )
    progress("interrupted" if interrupted else "technical_failure_preserved", failure=str(error))


def evaluate(commit):
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        data, frozen, benchmarks = inputs()
        if events("technical_failure"):
            raise QuantError("Preserve the terminal calendar financial failure.")
        if RESULTS.exists():
            saved = read_json(RESULTS)
            if saved["frozen"] != frozen or events("review")[0]["body"] != saved["review"]:
                raise QuantError("A completed calendar result cannot change on recovery.")
            for key, row in saved["review"]["paths"].items():
                base.verify_path(row, frozen, ID, key)
            return reference(RESULTS)
        published = publication(frozen, commit)
        fences = [
            row
            for row in base.journal_events(base.JOURNAL)
            if row["kind"] == "publication" and row["body"].get("study_id") == STUDY
        ]
        if not fences:
            if events("path_started") or PREPARED.exists():
                raise QuantError("No calendar historical output before its actual publication.")
            base.append_event(
                base.JOURNAL,
                "publication",
                None,
                {
                    "study_id": STUDY,
                    "verification": published,
                    "candidate_accounts_previously_started": False,
                    "real_source_targets_previously_computed": False,
                },
            )
        elif len(fences) != 1 or fences[0]["body"]["verification"]["published_commit"] != commit:
            raise QuantError("Keep the unique original calendar actual-publication fence.")
        try:
            prepared = prepare_source(data, frozen)
            targets = source_frame(prepared)
            paths = {}
            protocol = read_json(base.PROTOCOL)
            for window in protocol["windows"]:
                for scenario in protocol["accounts"]["scenarios"]:
                    lifecycle.runtime_guard()
                    key = f"{window['years']}y-{scenario['id']}"
                    paths[key] = evaluate_path(data, targets, frozen, window, scenario, benchmarks)
            audits = prefix_audits(data, targets)
            base.append_event(
                base.JOURNAL,
                "target_proof",
                ID,
                {
                    "proof_kind": "actual_published_calendar_native_and_main_book_outputs_v1",
                    "proof_stage": "after_four_actual_published_program_paths",
                    "native_source_proof": prepared,
                    "real_main_capital_prefix_audit": audits,
                    "all_original_dates_and_fixed_policy_retained": True,
                },
            )
            review = {
                "reviewed_at": utc_now(),
                "frozen_specification_sha256": digest_json(frozen),
                "status": "historical_point_gates_passed"
                if all(row["all_point_gates_pass"] for row in paths.values())
                else "rejected",
                "paths": {
                    key: reference(FOLDER / "accounts" / key / "complete.json") for key in paths
                },
                "all_four_paths_same_candidate_and_frozen_parameters": True,
                "independent_forward_validation": False,
                "order_authority": False,
            }
            base.append_event(base.JOURNAL, "review", ID, review)
            write_new_receipt(
                RESULTS,
                {
                    "completed_at": utc_now(),
                    "frozen": frozen,
                    "publication": published,
                    "review": review,
                    "paths": paths,
                    "native_source_proof": prepared,
                    "real_main_capital_prefix_audit": audits,
                },
            )
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            record_failure("funding_or_causality", error)
            raise
        progress("four_paths_complete", next_step="export_and_separate_technical_expert_admission")
    return reference(RESULTS)


def export():
    lifecycle.runtime_guard()
    data = read_json(RESULTS)
    if set(data["paths"]) != base.PATH_KEYS or len(events("review")) != 1:
        raise QuantError("Export only a complete actually reviewed calendar policy.")
    for key, ref in data["review"]["paths"].items():
        base.verify_path(ref, data["frozen"], ID, key)
    body = {
        "schema_version": 1,
        "record_type": "published_calendar_seasonal_four_path_results",
        "study_id": STUDY,
        "candidate_id": ID,
        "actual_completed_at": data["completed_at"],
        "status": data["review"]["status"],
        "registration": reference(PACKET),
        "frozen_specification_sha256": digest_json(data["frozen"]),
        "private_results_sha256": file_digest(RESULTS),
        "actual_preregistration": data["publication"],
        "target_delivery_kind": KIND,
        "paths": {
            key: {
                name: row[name]
                for name in (
                    "path_key",
                    "metrics",
                    "execution",
                    "gates",
                    "all_point_gates_pass",
                    "independent_bt_max_equity_difference_usd",
                    "independent_monthly_feedback",
                    "actual_capital_input_verification",
                    "actual_source_opinion_verification",
                )
            }
            for key, row in data["paths"].items()
        },
        "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
        "new_financial_rules_evaluated": 1,
        "new_formal_funded_paths": 4,
        "separately_versioned_auxiliary_main_book_prefix_runs": 5,
        "all_real_prefix_audits_exact": True,
        "real_native_targets_generated_after_actual_publication": True,
        "prior_policies_or_accounts_or_benchmarks_replayed": False,
        "raw_prices_targets_daily_books_private_intents_exported": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    write_new_receipt(PUBLIC, body)
    return reference(PUBLIC)


def admit():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        if ADMISSION.exists() or (FOLDER / "technical-admission-attempt-v1.json").exists():
            raise QuantError("Keep the completed or partial first calendar expert review.")
        data, frozen, _ = inputs()
        checked_packet(frozen)
        publication_fence()
        if len(events("review")) != 1 or len(events("target_proof")) != 1:
            raise QuantError(
                "Complete actual source books and post-publication proof before admission."
            )
        source = read_json(PREPARED)
        base.checked_reference(source["proof"])
        targets = source_frame(source)
        for key, ref in events("review")[0]["body"]["paths"].items():
            receipt = base.verify_path(ref, frozen, ID, key)
            if (
                receipt["execution"]["minimum_cash_usd"] < 0
                or receipt["execution"]["maximum_gross_exposure"] > 1
                or receipt["independent_bt_max_equity_difference_usd"] > 1e-6
                or receipt["broker_orders"] != 0
            ):
                raise QuantError("Invalid accounting or funding cannot become a valid expert.")
        shadow = pd.read_csv(
            FOLDER / "accounts/10y-base/account.csv",
            index_col=0,
            parse_dates=True,
            float_precision="round_trip",
        )
        before = current_counts(base.journal_events(base.JOURNAL))
        write_new_receipt(FOLDER / "technical-admission-attempt-v1.json", {"started_at": utc_now()})
        expert = ArchivedMonthlyExpert(
            ExpertIdentity(
                ID, "fixed_calendar_seasonal_equity_participation", events("registered")[0]["at"]
            ),
            targets,
            shadow,
            law.UNIVERSE,
        )
        dates = [previous_session(w["start"]) for w in read_json(base.PROTOCOL)["windows"]]
        dates.extend([pd.Timestamp("2017-04-05"), pd.Timestamp("2017-04-06"), data.close.index[-1]])
        checks = []
        for day in dates:
            observed = expert.observe(day, 126)
            if (
                observed.target is None
                or observed.target_known_at is None
                or observed.target_known_at > day
                or not observed.risk_free.equals(data.risk_free.loc[observed.risk_free.index])
            ):
                raise QuantError("Native calendar opinion or exact assessment RF is incompatible.")
            checks.append(
                {
                    "date": str(day.date()),
                    "target_known_at": str(observed.target_known_at.date()),
                    "NET_observations": len(observed.net_returns),
                    "score_mature": observed.score_mature,
                }
            )
        if (
            checks[2]["NET_observations"] != 125
            or checks[2]["score_mature"]
            or (checks[3]["NET_observations"] != 126 or not checks[3]["score_mature"])
            or current_counts(base.journal_events(base.JOURNAL)) != before
        ):
            raise QuantError("Preserve calendar availability, score maturity and trial counts.")
        packet = {
            "admitted_at": utc_now(),
            "candidate_id": ID,
            "mechanism_group": "fixed_calendar_seasonal_equity_participation",
            "adapter": "monthly",
            "introduced_at": events("registered")[0]["at"],
            "registered_event_sha256": events("registered")[0]["sha256"],
            "registration": reference(PACKET),
            "native_source_targets": source,
            "completed_paths": events("review")[0]["body"]["paths"],
            "observation_checks": checks,
            "native_RF_exact_common": True,
            "funding_and_independent_BT_checked_all_four_paths": True,
            "standalone_financial_verdict_preserved": events("review")[0]["body"]["status"],
            "technical_admission_is_final_qualification": False,
            "current_global_pool_or_roster_changed": False,
            "pool_inclusion_requires_new_separate_study": True,
            "new_strategy_evaluations": 0,
            "actual_authoritative_counts": before,
            "historical_counterfactual_availability": True,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "order_authority": False,
        }
        write_new_receipt(ADMISSION, packet)
        write_new_receipt(PUBLIC_ADMISSION, packet)
        progress("technical_admission_complete_pool_unchanged")
    return reference(ADMISSION)


def main():
    parser = argparse.ArgumentParser(description="One published calendar source policy, no orders.")
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export", "admit"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use the actual remotely published preregistration commit.")
    actions = {"freeze": initialize, "register": register, "export": export, "admit": admit}
    value = evaluate(args.commit) if args.action == "evaluate" else actions[args.action]()
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
