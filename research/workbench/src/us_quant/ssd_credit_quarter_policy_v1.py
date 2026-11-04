from __future__ import annotations

import argparse
import ast
import json
import os
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import pandas as pd

from us_quant import credit_quarter_policy_v1 as law
from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import ssd_conditional_hedge_native_v1 as native
from us_quant.calendar import previous_session
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.research_program import review_market
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.storage import (
    digest_json,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)
from us_quant.strategy_experts_v1 import load_expert

base, ROOT, source, template = native.base, native.ROOT, native.source, native.template
ID, CANDIDATE, POLICY = law.ID, law.CANDIDATE, law.POLICY
FOLDER = base.REPORTS / "credit-quarter-complete-policy-20261104-v1"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
PREPARED = FOLDER / "prepared-target-bundle-v1.json"
PACKET = ROOT / "evidence/credit_quarter_policy_20261104_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC = ROOT / "evidence/credit_quarter_policy_20261104_results_v1.json"
SOURCE_RESULTS = FOLDER / "source-result-contract-v1.json"
ADMISSION = FOLDER / "technical-admission-v1.json"
PUBLIC_ADMISSION = ROOT / "evidence/credit_quarter_policy_20261104_admission_v1.json"
PARENT = (
    base.REPORTS
    / "vxn-hmm-composition-substitution-20261104-v1/financial-implementation-freeze-v1.json"
)
OWN = (
    "config/credit-quarter-policy-v1.json",
    "src/us_quant/credit_quarter_policy_v1.py",
    "src/us_quant/ssd_credit_quarter_policy_v1.py",
    "tests/test_credit_quarter_policy_v1.py",
    "tests/test_ssd_credit_quarter_policy_v1.py",
    "config/credit-relative-input-v1.json",
    "src/us_quant/credit_relative_input_v1.py",
    "tests/test_credit_relative_input_v1.py",
    "tests/test_expert_initial_monthly_execution_v1.py",
    "tests/conftest.py",
)
METHODS = ("CREDIT_QUARTER_POLICY_V1.md", "CREDIT_RELATIVE_INPUT_V1.md")
reference, unique_refs = base.reference, source.unique_refs


def bound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


def dependencies():
    walker = source.admitted.import_template.dependencies
    return FunctionType(
        walker.__code__,
        {**globals(), "ast": ast, "prior": native},
        argdefs=walker.__defaults__,
    )()


def events(kind):
    return bound(template.events)(kind)


def progress(phase, **fields):
    counts = current_counts(base.journal_events(base.JOURNAL))
    if any(key in fields and fields[key] != value for key, value in counts.items()):
        raise QuantError("Credit progress cannot override actual journal counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "credit_policy_" + phase.removeprefix("consensus_"),
            "study_id": read_json(POLICY)["study_id"],
            "active_compute_job": None,
            **fields,
            **counts,
        },
    )


def check_frozen(frozen):
    native.check_frozen(frozen)


def credit_inputs():
    policy = read_json(POLICY)
    admission = read_json(base.checked_reference(policy["source_admission"]))
    if (
        admission["status"]
        != (
            "admitted_original_archival_lag1_quarter63_relative_input_only"
            "_NOT_expert_or_full_intraday_PIT"
        )
        or admission["observations"] != policy["source_observations"]
        or admission["new_financial_trials"] != 0
    ):
        raise QuantError("Reuse only the exact admitted archival credit input, never a new proxy.")
    frame = pd.read_csv(
        base.checked_reference(policy["source_observations"]), float_precision="round_trip"
    )
    for column in ("as_of_session", "source_start_session", "source_end_session"):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    return frame.set_index("as_of_session")


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the complete or partial first credit-policy freeze.")
    with lifecycle.research_job():
        os.umask(0o077)
        law.validate(read_json(POLICY))
        design = read_json(FOLDER / "design-decision-v1.json")
        if (
            design["actual_native_event_seq"] != 152648
            or design["actual_new_policy_targets_or_accounts_computed"] is not False
        ):
            raise QuantError("Keep the independently chosen credit policy before real targets.")
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        counts = current_counts(base.journal_events(base.JOURNAL))
        if counts["current_journal_event_count"] != 875:
            raise QuantError("Credit policy starts from the actual closed875 baseline.")
        parent = read_json(PARENT)["frozen"]
        check_frozen(parent)
        admission = read_json(
            base.checked_reference(read_json(POLICY)["source_admission_completion"])
        )
        for ref in admission["exact_evidence_references"]:
            base.checked_reference(ref)
        deps = dependencies()
        suites = (
            ElementTree.parse(FOLDER / "offline-tests-v1.xml").getroot().findall(".//testsuite")
        )
        proof = read_json(FOLDER / "validation-source-v1.json")
        if (
            not suites
            or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
            or proof["passed_tests"] != sum(int(s.get("tests", 0)) for s in suites)
            or proof["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
            or proof["outputs"]
            != [
                reference(FOLDER / "offline-tests-v1.xml"),
                reference(FOLDER / "offline-tests-v1.log"),
            ]
            or proof["exit_code"] != 0
        ):
            raise QuantError("Bind the exact successful credit law and lifecycle tests.")
        frozen = {
            **parent,
            "study_id": read_json(POLICY)["study_id"],
            "policy": reference(POLICY),
            "account_namespace": ID,
            "construction": CANDIDATE["construction"],
            "construction_inputs": [read_json(POLICY)["source_observations"]],
            "expert_pool_definition_role": "one_new_complete_source_policy_not_pool_expansion",
            "prior_journal_sequence": 875,
            "prior_journal_head_sha256": counts["current_journal_head_sha256"],
            "prior_completed_configurations": 251,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in deps if r["path"].startswith(("src/", "tests/"))],
                    *[reference(ROOT / p) for p in METHODS],
                ]
            ),
            "policies": unique_refs(
                [
                    *parent["policies"],
                    *[r for r in deps if r["path"].startswith("config/")],
                ]
            ),
            "prior_evidence": unique_refs(
                [
                    *parent["prior_evidence"],
                    *admission["exact_evidence_references"],
                    reference(PARENT),
                    reference(FOLDER / "design-decision-v1.json"),
                    read_json(POLICY)["source_admission_completion"],
                ]
            ),
            "validation_evidence": unique_refs(
                [
                    *parent["validation_evidence"],
                    reference(FOLDER / "offline-tests-v1.xml"),
                    reference(FOLDER / "offline-tests-v1.log"),
                    reference(FOLDER / "validation-source-v1.json"),
                ]
            ),
            "new_economic_factor_definitions": 0,
            "new_traded_assets": 0,
            "new_source_requests": 0,
            "new_shadow_account_runs": 0,
        }
        check_frozen(frozen)
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "frozen": frozen,
                "complete_dependency_inputs": deps,
                "package_versions": sorted(
                    (d.metadata["Name"], d.version) for d in distributions()
                ),
                "new_real_targets_or_accounts_computed": False,
                "registered": False,
                "pool_changed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="register_before_real_targets")
    return reference(INTENT)


def inputs():
    lifecycle.runtime_guard()
    intent = read_json(INTENT)
    for row in intent["complete_dependency_inputs"]:
        base.checked_reference(row)
    if intent["package_versions"] != [
        list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())
    ]:
        raise QuantError("The accepted isolated credit-policy environment changed.")
    frozen = intent["frozen"]
    check_frozen(frozen)
    frozen = {
        **frozen,
        "prior_evidence": unique_refs([*frozen["prior_evidence"], reference(INTENT)]),
    }
    data = review_market({"market": frozen["market"]}, ROOT)
    refs = [
        r
        for r in frozen["benchmark_artifacts"]
        if r["path"].endswith("/benchmarks-20261027-v1/results.json")
    ]
    if tuple(data.close.columns) != law.UNIVERSE or len(refs) != 1:
        raise QuantError("Reuse the original fourteen-asset market and completed benchmarks.")
    return data, None, frozen, read_json(base.checked_reference(refs[0]))


def verify_targets(close, credit, targets, audits, records):
    expected = pd.DatetimeIndex(
        [day for day in close.index if law.is_month_end(day) and credit.loc[day, "history_mature"]]
    )
    if (
        not targets.index.equals(close.index)
        or not targets.columns.equals(close.columns)
        or not targets.dropna(how="all").index.equals(expected)
        or len(audits) != len(expected)
        or len(records) != 2 * len(expected)
    ):
        raise QuantError("Preserve every genuine mature native month-end opinion.")
    maximum = 0.0
    for i, day in enumerate(expected):
        before, after = records[2 * i : 2 * i + 2]
        observation = credit.loc[day]
        value = float(observation["relative_total_return"])
        selected = law.CORE if value.as_integer_ratio()[0] > 0 else law.DEFENSE
        oracle = pd.Series(
            [
                0.245
                if name in selected and len(selected) == 4
                else 0.49
                if name in selected
                else 0.0
                for name in law.UNIVERSE
            ],
            index=law.UNIVERSE,
        )
        if (
            before
            != {
                "record": "decision_input",
                "decision_session": str(day.date()),
                "information_clock": law.CLOCK,
                "credit_source_start": str(observation["source_start_session"].date()),
                "credit_source_end": str(observation["source_end_session"].date()),
                "credit_relative_binary64_hex": value.hex(),
                "credit_history_mature": True,
                "source_credit_observations_recomputed": False,
            }
            or after["record"] != "decision_resolved"
            or after["audit"] != audits[i]
            or after["asset_order"] != list(close.columns)
            or after["complete_target"] != targets.loc[day].tolist()
            or audits[i]["selected_assets"] != list(selected)
        ):
            raise QuantError("The retained credit input or independent target selection changed.")
        maximum = max(maximum, float(abs(oracle - targets.loc[day]).max()))
    if maximum > 1e-12:
        raise QuantError("The independent complete credit target differs.")
    return {
        "genuine_month_end_decisions": len(expected),
        "retained_input_resolved_records": len(records),
        "maximum_independent_credit_target_difference": maximum,
    }


def prepare(data, unused, frozen):
    if PREPARED.exists():
        old = read_json(PREPARED)
        if old["frozen_specification_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting prior credit preparation.")
        for row in old["entries"][ID].values():
            base.checked_reference(row)
        return old["entries"][ID]
    attempt = FOLDER / "preparation-attempt-v1.json"
    if attempt.exists():
        raise QuantError("An incomplete credit preparation cannot be automatically repeated.")
    registered = events("registered")
    if len(registered) != 1 or registered[0]["body"] != {
        "candidate": CANDIDATE,
        "frozen": frozen,
    }:
        raise QuantError("Actual credit registration must precede all real source targets.")
    write_new_receipt(attempt, {"started_at": utc_now(), "frozen_sha256": digest_json(frozen)})
    credit = credit_inputs()
    log = FOLDER / "actual-inputs-resolved-v1.jsonl"
    with log.open("x") as stream:

        def retain(row):
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        targets, audits = law.build_targets(
            data.close,
            credit,
            read_json(POLICY),
            before_decision=lifecycle.runtime_guard,
            retain=retain,
        )
    records = [json.loads(line) for line in log.read_text().splitlines()]
    verification = verify_targets(data.close, credit, targets, audits, records)
    prefixes = []
    for index in sorted({0, 6, 12, len(audits) // 2, len(audits) - 1}):
        day = pd.Timestamp(audits[index]["decision_session"])
        early, earlier = law.build_targets(
            data.close.loc[:day],
            credit.loc[:day],
            read_json(POLICY),
            before_decision=lifecycle.runtime_guard,
        )
        if not early.equals(targets.loc[:day]) or earlier != audits[: index + 1]:
            raise QuantError("Future inputs changed a native credit recommendation.")
        prefixes.append(str(day.date()))
    active = targets.dropna(how="all")
    coverage = []
    for window in read_json(base.PROTOCOL)["windows"]:
        anchor = previous_session(window["start"])
        known = active.loc[:anchor].tail(1)
        rows = pd.concat([known, active.loc[anchor + pd.Timedelta(days=1) : window["end"]]])
        factors = [name for name in law.CORE if (rows[name] > 1e-10).any()]
        if known.empty or len(factors) < 3:
            raise QuantError(
                "Both original funded horizons need defined anchors and genuine factors."
            )
        coverage.append(
            {
                "years": window["years"],
                "actual_factor_symbols": factors,
                "known_native_recommendation_at_anchor": str(known.index[0].date()),
            }
        )
    directory = FOLDER / "native-targets"
    new_output_directory(directory)
    write_text_atomic(directory / "targets.csv", targets.to_csv(float_format="%.17g"))
    write_new_receipt(FOLDER / "information-audits-v1.json", {"decisions": audits})
    write_new_receipt(
        directory / "proof.json",
        {
            "checked_at": utc_now(),
            "frozen_specification_sha256": digest_json(frozen),
            **verification,
            "prefix_truncations": prefixes,
            "formal_factor_coverage": coverage,
            "actual_input_resolved_records": reference(log),
            "information_audits": reference(FOLDER / "information-audits-v1.json"),
            "whole_native_prehistory_retained": True,
            "synthetic_initial_recommendation_added": False,
            "candidate_accounts_or_metrics_computed": False,
            "original_source_signal_recomputed": False,
            "first_new_policy_targets_computed": True,
            "independent_forward_validation": False,
        },
    )
    entry = {
        "targets": reference(directory / "targets.csv"),
        "proof": reference(directory / "proof.json"),
    }
    write_new_receipt(
        PREPARED, {"frozen_specification_sha256": digest_json(frozen), "entries": {ID: entry}}
    )
    check_frozen(frozen)
    return entry


def record_failure(stage, error):
    return bound(template.record_failure)(stage, error)


def register():
    return bound(template.register)()


def publication(frozen, commit):
    return bound(template.publication)(frozen, commit)


def evaluate_path(data, targets, frozen, window, scenario, benchmarks):
    return bound(template.evaluate_path)(data, targets, frozen, window, scenario, benchmarks)


def evaluate(commit):
    return bound(template.evaluate)(commit)


def export():
    return bound(native.export)()


def admit():
    lifecycle.runtime_guard()
    if ADMISSION.exists() or (FOLDER / "technical-admission-attempt-v1.json").exists():
        raise QuantError("Preserve the complete or partial first credit expert admission.")
    with lifecycle.research_job():
        data, _, frozen, _ = inputs()
        before = current_counts(base.journal_events(base.JOURNAL))
        write_new_receipt(FOLDER / "technical-admission-attempt-v1.json", {"started_at": utc_now()})
        record = {
            "id": ID,
            "mechanism_group": "relative_credit_market_condition",
            "adapter": "monthly",
            "introduced_at": events("registered")[0]["at"],
            "registered_event_sha256": events("registered")[0]["sha256"],
            "registration": reference(PACKET),
            "results": reference(SOURCE_RESULTS),
            "completed_paths": events("review")[0]["body"]["paths"],
        }
        expert, technical = load_expert(record, tuple(data.close.columns))
        dates = [previous_session(w["start"]) for w in read_json(base.PROTOCOL)["windows"]]
        dates += [pd.Timestamp("2017-04-05"), pd.Timestamp("2017-04-06"), data.close.index[-1]]
        observed = []
        for day in dates:
            value = expert.observe(day, 126)
            if (
                value.target is None
                or value.target_known_at is None
                or value.target_known_at > day
                or not value.risk_free.equals(data.risk_free.loc[value.risk_free.index])
            ):
                raise QuantError(
                    "Credit native recommendations or exact common RF are incompatible."
                )
            observed.append(
                {
                    "date": str(day.date()),
                    "target_known_at": str(value.target_known_at.date()),
                    "NET_observations": len(value.net_returns),
                    "score_mature": value.score_mature,
                }
            )
        if (
            observed[2]["NET_observations"] != 125
            or observed[2]["score_mature"]
            or observed[3]["NET_observations"] != 126
            or not observed[3]["score_mature"]
            or current_counts(base.journal_events(base.JOURNAL)) != before
        ):
            raise QuantError("Preserve the unchanged actual expert availability and trial counts.")
        packet = {
            "admitted_at": utc_now(),
            "record": record,
            "technical": technical,
            "observation_checks": observed,
            "native_RF_exact_common": True,
            "funding_and_independent_BT_checked_all_four_paths": True,
            "standalone_financial_verdict_preserved": events("review")[0]["body"]["status"],
            "technical_admission_is_final_qualification": False,
            "current_global_pool_or_roster_changed": False,
            "pool_inclusion_requires_new_separate_study": True,
            "new_strategy_evaluations": 0,
            "actual_authoritative_counts": before,
            "historical_credit_input_scope_not_broadened": True,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "order_authority": False,
        }
        write_new_receipt(ADMISSION, packet)
        write_new_receipt(PUBLIC_ADMISSION, packet)
        progress("technical_admission_complete_pool_unchanged")
    return reference(ADMISSION)


def main():
    parser = argparse.ArgumentParser(
        description="One complete credit source policy, no broker orders."
    )
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export", "admit"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use an actual remotely published preregistration SHA.")
    actions = {"freeze": initialize, "register": register, "export": export, "admit": admit}
    print(
        json.dumps(evaluate(args.commit) if args.action == "evaluate" else actions[args.action]())
    )


if __name__ == "__main__":
    main()
