from __future__ import annotations

import argparse
import json
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import monthly_style_reversal_expert_v1 as law
from us_quant import ssd_calendar_seasonal_financial_v1 as shared
from us_quant import ssd_research_v1 as base
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.research_program import review_market
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.storage import digest_json, read_json, utc_now, write_json

ROOT, ID = base.ROOT, law.ID
STUDY = "ssd_monthly_style_reversal_financial_20261105_v1"
POLICY = ROOT / "config/monthly-style-reversal-financial-v1.json"
FOLDER = base.REPORTS / "monthly-style-reversal-financial-20261105-v1"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
READY = FOLDER / "program-readiness-v1.json"
PREPARED = FOLDER / "published-native-targets-v1.json"
RESULTS = FOLDER / "results.json"
PACKET = ROOT / "evidence/monthly_style_reversal_financial_20261105_registration_v1.json"
KIND = shared.KIND
lifecycle, reference, unique_refs = shared.lifecycle, base.reference, shared.unique_refs
run_calendar, verify_opinions = shared.run_calendar, shared.verify_opinions
OWN = (
    "config/monthly-style-reversal-financial-v1.json",
    "src/us_quant/ssd_monthly_style_reversal_financial_v1.py",
    "tests/test_ssd_monthly_style_reversal_financial_v1.py",
    "MONTHLY_STYLE_REVERSAL_FINANCIAL_V1.md",
    "config/monthly-style-reversal-expert-v1.json",
    "src/us_quant/monthly_style_reversal_expert_v1.py",
    "tests/test_monthly_style_reversal_expert_v1.py",
    "MONTHLY_STYLE_REVERSAL_EXPERT_V1.md",
)
REFERENCES = (
    "primary_policy",
    "primary_preparation_freeze",
    "primary_preparation_completion",
    "market_parent_freeze",
)


def _shared(name, *args, **kwargs):
    function = getattr(shared, name)
    isolated = FunctionType(
        function.__code__,
        {**function.__globals__, **globals()},
        argdefs=function.__defaults__,
    )
    return isolated(*args, **kwargs)


def events(kind):
    return _shared("events", kind)


def packages():
    return shared.packages()


def dependencies():
    return _shared("dependencies")


def check_frozen(frozen):
    return shared.check_frozen(frozen)


def inputs():
    return _shared("inputs")


def packet_body(frozen, registered, readiness):
    return _shared("packet_body", frozen, registered, readiness)


def checked_packet(frozen):
    return _shared("checked_packet", frozen)


def register():
    return _shared("register")


def publication(frozen, commit):
    return _shared("publication", frozen, commit)


def publication_fence():
    return _shared("publication_fence")


def prepare_source(data, frozen):
    return _shared("prepare_source", data, frozen)


def source_frame(prepared):
    return _shared("source_frame", prepared)


def evaluate_path(data, targets, frozen, window, scenario, benchmarks):
    return _shared("evaluate_path", data, targets, frozen, window, scenario, benchmarks)


def prefix_audits(data, targets):
    return _shared("prefix_audits", data, targets)


def record_failure(stage, error):
    return _shared("record_failure", stage, error)


def progress(phase, **fields):
    counts = current_counts(base.journal_events(base.JOURNAL))
    if any(key in fields and fields[key] != value for key, value in counts.items()):
        raise QuantError("Reversal progress cannot override actual journal counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "monthly_style_reversal_financial_" + phase,
            "study_id": STUDY,
            "active_compute_job": None,
            **counts,
            **fields,
        },
    )


def validate_policy(policy):
    original = read_json(shared.POLICY)
    references = set(REFERENCES)
    fixed = {
        **{key: value for key, value in original.items() if key not in references},
        "study_id": STUDY,
        "candidate_id": ID,
        "prior_journal_sequence": 957,
        "prior_journal_head_sha256": (
            "d66608671751e0527858252fb3b7744bfb065fa65e238b78bfcb4358f2c8371a"
        ),
        "prior_completed_configurations": 258,
    }
    if (
        not isinstance(policy, dict)
        or set(policy) != set(fixed) | references
        or any(
            policy.get(key) != value or type(policy.get(key)) is not type(value)
            for key, value in fixed.items()
        )
        or policy["primary_policy"] != reference(law.POLICY)
    ):
        raise QuantError("Preserve the prepared reversal law and original funded contract.")
    for key in references:
        base.checked_reference(policy[key])
    primary = read_json(base.checked_reference(policy["primary_preparation_freeze"]))
    if (
        primary["prior_completed_configurations"] != 258
        or primary["prior_journal_sequence"] != 957
        or primary["registered"] is not False
        or primary["real_new_policy_targets_accounts_or_metrics_computed"] is not False
    ):
        raise QuantError("Primary preparation is not a financial run or an inherited count.")
    for row in primary["implementation_and_test_inputs"]:
        base.checked_reference(row)
    law.validate(read_json(law.POLICY))


def validation_refs():
    proof = read_json(FOLDER / "validation-source-v1.json")
    suites = ElementTree.parse(FOLDER / "financial-tests-v1.xml").getroot().findall(".//testsuite")
    if (
        not suites
        or any(int(row.get(key, 0)) for row in suites for key in ("failures", "errors", "skipped"))
        or proof["implementation_and_test_inputs"] != [reference(ROOT / path) for path in OWN]
        or proof["passed_tests"] != sum(int(row.get("tests", 0)) for row in suites)
        or proof["ruff_returncode"] != 0
        or proof["synthetic_only"] is not True
    ):
        raise QuantError("Bind the exact synthetic reversal program and fence checks.")
    return [
        reference(FOLDER / name)
        for name in (
            "validation-source-v1.json",
            "financial-tests-v1.xml",
            "financial-tests-v1.log",
            "financial-ruff-v1.log",
        )
    ]


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve complete or partial reversal financial preparation.")
    with lifecycle.research_job():
        policy = read_json(POLICY)
        validate_policy(policy)
        counts = current_counts(base.journal_events(base.JOURNAL))
        if (
            counts["current_journal_event_count"] != 957
            or counts["current_journal_head_sha256"] != policy["prior_journal_head_sha256"]
            or counts["total_legacy_plus_ssd_completed_configurations"] != 258
        ):
            raise QuantError("Require the actual closedJ957 baseline before new outcomes.")
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        parent = read_json(base.checked_reference(policy["market_parent_freeze"]))["frozen"]
        check_frozen(parent)
        data = review_market({"market": parent["market"]}, ROOT)
        law.checked_close(data.close)
        deps = unique_refs(
            [*dependencies(), reference(shared.POLICY), reference(ROOT / "tests/conftest.py")]
        )
        frozen = {
            **{
                key: value
                for key, value in parent.items()
                if key not in {"expert_pool", "expert_pool_definition_role"}
                and not key.startswith("source_")
            },
            "study_id": STUDY,
            "policy": reference(POLICY),
            "primary_signal_policy": policy["primary_policy"],
            "execution_policy": reference(POLICY),
            "account_namespace": "monthly_style_reversal_published_program_v1",
            "construction": law.CANDIDATE["construction"],
            "construction_inputs": [policy["primary_policy"]],
            "decision_program_kind": KIND,
            "expert_pool_definition_role": "one_complete_source_policy_not_pool_expansion",
            "main_capital_initialization": policy["initialization"],
            "prepublication_readiness": "synthetic_Interface_and_pinned_market_identity_only",
            "historical_target_proof_stage": policy["historical_target_proof_stage"],
            "prior_journal_sequence": 957,
            "prior_journal_head_sha256": policy["prior_journal_head_sha256"],
            "prior_completed_configurations": 258,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in deps if r["path"].startswith(("src/", "tests/"))],
                    *[reference(ROOT / path) for path in OWN if path.endswith(".md")],
                ]
            ),
            "policies": unique_refs(
                [*parent["policies"], *[r for r in deps if r["path"].startswith("config/")]]
            ),
            "prior_evidence": unique_refs(
                [
                    *parent["prior_evidence"],
                    *[policy[key] for key in REFERENCES[1:]],
                    reference(FOLDER / "actual-invocation-v1.json"),
                    reference(FOLDER / "financial-design-v1.json"),
                ]
            ),
            "validation_evidence": unique_refs(
                [*parent["validation_evidence"], *validation_refs()]
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


def verify_targets(close, targets, audits, records):
    dates = pd.DatetimeIndex(
        [day for index, day in enumerate(close.index) if index >= 21 and shared.is_month_end(day)]
    )
    if (
        not targets.index.equals(close.index)
        or not targets.columns.equals(close.columns)
        or not targets.dropna(how="all").index.equals(dates)
        or len(audits) != len(dates)
        or len(records) != 2 * len(dates)
    ):
        raise QuantError("Preserve all source dates and the exact22-level reversal warmup.")
    values_checked = 0
    for index, day in enumerate(dates):
        position = int(close.index.get_loc(day))
        window = close.iloc[position - 21 : position + 1]
        ratios = {
            symbol: float(window[symbol].iloc[-1]) / float(window[symbol].iloc[0]) - 1
            for symbol in law.CORE
        }
        rounded = {symbol: round(value, 12) for symbol, value in ratios.items()}
        ranked = sorted(law.CORE, key=lambda symbol: (rounded[symbol], symbol))
        selected = ranked[:2]
        oracle = [0.49 if symbol in selected else 0.0 for symbol in law.UNIVERSE]
        expected_before = {
            "record": "decision_input",
            "decision_session": str(day.date()),
            "information_clock": law.CLOCK,
            "window_dates": window.index.strftime("%Y-%m-%d").tolist(),
            "used_source_prices_binary64_hex": {
                symbol: [float(value).hex() for value in window[symbol]] for symbol in law.CORE
            },
            "future_price_or_return_observed": False,
        }
        before, after = records[2 * index : 2 * index + 2]
        audit = audits[index]
        if (
            before != expected_before
            or after["record"] != "decision_resolved"
            or after["audit"] != audit
            or after["complete_target"] != targets.loc[day].tolist()
            or after["asset_order"] != list(close.columns)
            or audit["simple_total_returns"] != ratios
            or audit["rounded_ranking_values"] != rounded
            or audit["ranking_order"] != ranked
            or audit["selected_assets"] != selected
            or audit["source_window_start"] != str(window.index[0].date())
            or audit["source_window_end"] != str(day.date())
            or not np.array_equal(oracle, targets.loc[day].to_numpy())
        ):
            raise QuantError("Independent reversal input, scalar ratio, ranking or target failed.")
        values_checked += 22 * len(law.CORE)
    return {
        "native_month_end_opinions": len(dates),
        "all_independent_reversal_targets_exact": True,
        "source_price_binary64_inputs_checked": values_checked,
    }


def evaluate(commit):
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        data, frozen, benchmarks = inputs()
        if events("technical_failure"):
            raise QuantError("Preserve the terminal reversal financial failure.")
        if RESULTS.exists():
            saved = read_json(RESULTS)
            if saved["frozen"] != frozen or events("review")[0]["body"] != saved["review"]:
                raise QuantError("A completed reversal result cannot change on recovery.")
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
                raise QuantError("No reversal historical output before actual publication.")
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
            raise QuantError("Keep the unique actual reversal publication fence.")
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
                    "proof_kind": "actual_published_reversal_native_and_main_book_outputs_v1",
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


def main():
    parser = argparse.ArgumentParser(
        description="One published reversal source program, no orders."
    )
    parser.add_argument("action", choices=("freeze", "register", "evaluate"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use the actual remotely published preregistration commit.")
    action = initialize if args.action == "freeze" else register
    value = evaluate(args.commit) if args.action == "evaluate" else action()
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
