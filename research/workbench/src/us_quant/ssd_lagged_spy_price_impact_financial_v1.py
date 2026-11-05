from __future__ import annotations

import argparse
import json
import os
from fractions import Fraction
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import lagged_spy_price_impact_expert_v1 as law
from us_quant import ssd_calendar_seasonal_financial_v1 as shared
from us_quant import ssd_research_v1 as base
from us_quant.calendar import is_month_end, previous_session
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.research_program import review_market
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.storage import (
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)

ROOT, ID = base.ROOT, law.ID
STUDY = "ssd_lagged_spy_price_impact_financial_20261105_v1"
POLICY = ROOT / "config/lagged-spy-price-impact-financial-v1.json"
FOLDER = base.REPORTS / "lagged-spy-price-impact-financial-20261105-v1"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
READY = FOLDER / "program-readiness-v1.json"
PREPARED = FOLDER / "published-native-targets-v1.json"
RESULTS = FOLDER / "results.json"
PACKET = ROOT / "evidence/lagged_spy_price_impact_financial_20261105_registration_v1.json"
PUBLIC = ROOT / "evidence/lagged_spy_price_impact_financial_20261105_results_v1.json"
KIND = shared.KIND
lifecycle, reference, unique_refs = shared.lifecycle, base.reference, shared.unique_refs
run_calendar, verify_opinions = shared.run_calendar, shared.verify_opinions
OWN = (
    "config/lagged-spy-price-impact-financial-v1.json",
    "src/us_quant/ssd_lagged_spy_price_impact_financial_v1.py",
    "tests/test_ssd_lagged_spy_price_impact_financial_v1.py",
    "LAGGED_SPY_PRICE_IMPACT_FINANCIAL_V1.md",
    "config/lagged-spy-price-impact-expert-v1.json",
    "src/us_quant/lagged_spy_price_impact_expert_v1.py",
    "tests/test_lagged_spy_price_impact_expert_v1.py",
    "LAGGED_SPY_PRICE_IMPACT_EXPERT_V1.md",
)
REFERENCES = (
    "primary_policy",
    "primary_preparation_freeze",
    "primary_preparation_completion",
    "market_parent_freeze",
)


def _shared(name, *args, **kwargs):
    function = getattr(shared, name)
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )(*args, **kwargs)


def events(kind):
    return _shared("events", kind)


def packages():
    return shared.packages()


def dependencies():
    return _shared("dependencies")


def check_frozen(frozen):
    return shared.check_frozen(frozen)


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
        raise QuantError("Lagged-source progress cannot override actual journal counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "lagged_spy_price_impact_financial_" + phase,
            "study_id": STUDY,
            "active_compute_job": None,
            **counts,
            **fields,
        },
    )


def validate_policy(policy):
    original = read_json(shared.POLICY)
    fixed = {
        **{key: value for key, value in original.items() if key not in set(REFERENCES)},
        "study_id": STUDY,
        "candidate_id": ID,
        "prior_journal_sequence": 1027,
        "prior_journal_head_sha256": (
            "94a0ae53b6a9138e6b1776e9955491030e369a3bc1f2d840532d6d823804ede5"
        ),
        "prior_completed_configurations": 264,
        "source_input_interface": "original_MarketData_including_raw_close_and_reported_volume",
    }
    if (
        not isinstance(policy, dict)
        or set(policy) != set(fixed) | set(REFERENCES)
        or any(policy.get(k) != v or type(policy.get(k)) is not type(v) for k, v in fixed.items())
        or policy["primary_policy"] != reference(law.POLICY)
    ):
        raise QuantError("Preserve the frozen lagged OHLCV law and original funded contract.")
    for key in REFERENCES:
        base.checked_reference(policy[key])
    primary = read_json(base.checked_reference(policy["primary_preparation_freeze"]))
    if (
        primary["prior_completed_configurations"] != 264
        or primary["prior_journal_sequence"] != 1027
        or primary["registered"] is not False
        or primary["real_new_policy_indicators_targets_accounts_or_metrics_computed"] is not False
    ):
        raise QuantError(
            "Primary preparation is not a prior real indicator or financial execution."
        )
    for row in primary["implementation_and_test_inputs"]:
        base.checked_reference(row)
    law.validate(read_json(law.POLICY))


def validation_refs():
    proof = read_json(FOLDER / "validation-source-v1.json")
    suites = ElementTree.parse(FOLDER / "financial-tests-v1.xml").getroot().findall(".//testsuite")
    if (
        not suites
        or any(int(r.get(k, 0)) for r in suites for k in ("failures", "errors", "skipped"))
        or proof["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
        or proof["passed_tests"] != sum(int(r.get("tests", 0)) for r in suites)
        or proof["ruff_returncode"] != 0
        or proof["synthetic_only"] is not True
    ):
        raise QuantError("Bind the exact synthetic full-OHLCV program and publication checks.")
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
        raise QuantError("Preserve complete or partial first lagged-source financial preparation.")
    with lifecycle.research_job():
        policy = read_json(POLICY)
        validate_policy(policy)
        counts = current_counts(base.journal_events(base.JOURNAL))
        if (
            counts["current_journal_event_count"] != 1027
            or counts["current_journal_head_sha256"] != policy["prior_journal_head_sha256"]
            or counts["total_legacy_plus_ssd_completed_configurations"] != 264
        ):
            raise QuantError("Require the actual closedJ1027/264 baseline before outcomes.")
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        parent = read_json(base.checked_reference(policy["market_parent_freeze"]))["frozen"]
        check_frozen(parent)
        data = review_market({"market": parent["market"]}, ROOT)
        law.checked_market(data)
        deps = unique_refs(
            [*dependencies(), reference(shared.POLICY), reference(ROOT / "tests/conftest.py")]
        )
        frozen = {
            **{
                k: v
                for k, v in parent.items()
                if k not in {"expert_pool", "expert_pool_definition_role"}
                and not k.startswith("source_")
            },
            "study_id": STUDY,
            "policy": reference(POLICY),
            "primary_signal_policy": policy["primary_policy"],
            "execution_policy": reference(POLICY),
            "account_namespace": "lagged_spy_price_impact_published_program_v1",
            "construction": law.CANDIDATE["construction"],
            "construction_inputs": [policy["primary_policy"]],
            "decision_program_kind": KIND,
            "expert_pool_definition_role": "one_complete_source_policy_not_pool_expansion",
            "main_capital_initialization": policy["initialization"],
            "prepublication_readiness": "synthetic_Interface_and_pinned_market_identity_only",
            "historical_target_proof_stage": policy["historical_target_proof_stage"],
            "prior_journal_sequence": 1027,
            "prior_journal_head_sha256": policy["prior_journal_head_sha256"],
            "prior_completed_configurations": 264,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in deps if r["path"].startswith(("src/", "tests/"))],
                    *[reference(ROOT / p) for p in OWN if p.endswith(".md")],
                ]
            ),
            "policies": unique_refs(
                [*parent["policies"], *[r for r in deps if r["path"].startswith("config/")]]
            ),
            "prior_evidence": unique_refs(
                [
                    *parent["prior_evidence"],
                    *[policy[k] for k in REFERENCES[1:]],
                    reference(FOLDER / "actual-invocation-v1.json"),
                    reference(FOLDER / "financial-design-v1.json"),
                ]
            ),
            "validation_evidence": unique_refs(
                [*parent["validation_evidence"], *validation_refs()]
            ),
            "decision_information_clock": law.CLOCK,
            "primary_input_lag_sessions": 1,
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
                "new_real_source_indicators_targets_accounts_or_metrics_computed": False,
                "actual_registration_performed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="register_program_without_real_source_outputs")
    return reference(INTENT)


def inputs():
    lifecycle.runtime_guard()
    intent = read_json(INTENT)
    for ref in intent["complete_dependency_inputs"]:
        base.checked_reference(ref)
    if intent["package_versions"] != [list(row) for row in packages()]:
        raise QuantError("The accepted pinned lagged-source financial environment changed.")
    frozen = {
        **intent["frozen"],
        "prior_evidence": unique_refs([*intent["frozen"]["prior_evidence"], reference(INTENT)]),
    }
    check_frozen(frozen)
    data = review_market({"market": frozen["market"]}, ROOT)
    law.checked_market(data)
    found = [
        r
        for r in frozen["benchmark_artifacts"]
        if r["path"].endswith("/benchmarks-20261027-v1/results.json")
    ]
    if len(found) != 1:
        raise QuantError("Reuse one exact complete SPY/QQQ benchmark, never replay it.")
    return data, frozen, read_json(base.checked_reference(found[0]))


def independent_state(closes, reported_closes, reported_volumes):
    if (len(closes), len(reported_closes), len(reported_volumes)) != (274, 273, 273):
        raise QuantError("The independent lagged oracle needs every original window value.")
    impacts = []
    for i, (raw, volume) in enumerate(zip(reported_closes, reported_volumes, strict=True)):
        before, current, raw, volume = [float(v) for v in (closes[i], closes[i + 1], raw, volume)]
        if not all(np.isfinite(v) and v > 0 for v in (before, current, raw, volume)):
            raise QuantError("Independent source inputs cannot be missing or defaulted.")
        quotient = float(Fraction.from_float(current) / Fraction.from_float(before))
        change = float(Fraction.from_float(quotient) - 1)
        activity = float(Fraction.from_float(raw) * Fraction.from_float(volume))
        impact = float(Fraction.from_float(abs(change)) / Fraction.from_float(activity))
        impacts.append(impact)

    def rounded_binary64_mean(values):
        exact = sum((Fraction.from_float(v) for v in values), Fraction())
        return float(Fraction.from_float(float(exact)) / len(values))

    prior, current = [rounded_binary64_mean(v) for v in (impacts[:252], impacts[252:])]
    if prior <= 0:
        raise QuantError("The independent oracle cannot invent a nondegenerate reference.")
    ratio = float(Fraction.from_float(current) / Fraction.from_float(prior))
    key = round(ratio, 12)
    return {
        "reference_impact_mean": prior,
        "current_impact_mean": current,
        "relative_impact_ratio": ratio,
        "rounded_relative_ratio": key,
        "risk_off_triggered": key > 1.0,
    }


def verify_targets(data, targets, audits, records):
    dates = pd.DatetimeIndex(
        [day for i, day in enumerate(data.close.index) if i >= 274 and is_month_end(day)]
    )
    if (
        not targets.index.equals(data.close.index)
        or not targets.columns.equals(data.close.columns)
        or not targets.dropna(how="all").index.equals(dates)
        or len(audits) != len(dates)
        or len(records) != 2 * len(dates)
    ):
        raise QuantError("Preserve all source dates and the original full lagged warmup.")
    for index, day in enumerate(dates):
        i = int(data.close.index.get_loc(day))
        close = data.close["SPY"].iloc[i - 274 : i]
        raw = data.raw_close["SPY"].iloc[i - 273 : i]
        volume = data.volume["SPY"].iloc[i - 273 : i]
        state = independent_state(close.tolist(), raw.tolist(), volume.tolist())
        selected = ["BIL"] if state["risk_off_triggered"] else list(law.CORE)
        oracle = [0.98 / len(selected) if s in selected else 0.0 for s in law.UNIVERSE]
        expected_before = {
            "record": "decision_input",
            "decision_session": str(day.date()),
            "information_clock": law.CLOCK,
            "input_symbol": "SPY",
            "impact_return_session_dates": raw.index.strftime("%Y-%m-%d").tolist(),
            "adjusted_close_binary64_hex": [float(v).hex() for v in close],
            "reported_close_binary64_hex": [float(v).hex() for v in raw],
            "reported_share_volume_binary64_hex": [float(v).hex() for v in volume],
            "decision_session_own_or_future_bar_used_to_choose_weight": False,
        }
        expected_audit = {
            "decision_session": str(day.date()),
            "decision_information_clock": law.CLOCK,
            "first_prior_adjusted_close": str(close.index[0].date()),
            "reference_input_start": str(raw.index[0].date()),
            "reference_input_end": str(raw.index[251].date()),
            "current_input_start": str(raw.index[252].date()),
            "last_input_session": str(raw.index[-1].date()),
            "reference_impact_observations": 252,
            "current_impact_observations": 21,
            **state,
            "selected_assets": selected,
            "decision_session_own_market_bar_used_to_choose_weight": False,
            "proxy_is_measured_market_depth_flow_or_execution_cost": False,
            "historical_counterfactual_availability": True,
            "actual_policy_introduction_or_admission_backdated": False,
            "mature_NET_score_inferred_from_price_warmup": False,
            "independent_forward_validation": False,
        }
        before, after = records[2 * index : 2 * index + 2]
        if (
            before != expected_before
            or audits[index] != expected_audit
            or after["record"] != "decision_resolved"
            or after["audit"] != expected_audit
            or after["asset_order"] != list(data.close.columns)
            or after["complete_target"] != targets.loc[day].tolist()
            or not np.array_equal(oracle, targets.loc[day].to_numpy())
        ):
            raise QuantError("Independent lagged inputs, binary64 state or complete target failed.")
    return {
        "native_month_end_opinions": len(dates),
        "all_independent_lagged_price_impact_targets_exact": True,
        "source_price_volume_binary64_inputs_checked": 820 * len(dates),
        "decision_day_or_future_input_used": False,
        "full_MarketData_volume_Interface_preserved": True,
    }


def prepare_source(data, frozen):
    fence = publication_fence()
    if PREPARED.exists():
        old = read_json(PREPARED)
        if old["frozen_specification_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting completed lagged-source preparation.")
        for key in ("targets", "proof"):
            base.checked_reference(old[key])
        return old
    attempt = FOLDER / "source-targets-started-v1.json"
    if attempt.exists():
        raise QuantError("Preserve incomplete lagged targets instead of generating them twice.")
    write_new_receipt(
        attempt,
        {
            "started_at": utc_now(),
            "publication_event_sha256": fence["sha256"],
            "real_native_indicators_or_targets_previously_computed": False,
        },
    )
    trace = FOLDER / "native-inputs-resolved-v1.jsonl"
    with trace.open("x") as stream:

        def retain(row):
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        targets, audits = law.build_targets(
            data, read_json(law.POLICY), before_decision=lifecycle.runtime_guard, retain=retain
        )
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    checked = verify_targets(data, targets, audits, records)
    prefixes = []
    for position in sorted({0, 6, 12, len(audits) // 2, len(audits) - 1}):
        day = pd.Timestamp(audits[position]["decision_session"])
        prefix_data = MarketData(
            data.open.loc[:day],
            data.close.loc[:day],
            data.raw_close.loc[:day],
            data.volume.loc[:day],
            data.risk_free.loc[:day],
        )
        prefix, earlier = law.build_targets(
            prefix_data, read_json(law.POLICY), before_decision=lifecycle.runtime_guard
        )
        if not prefix.equals(targets.loc[:day]) or earlier != audits[: position + 1]:
            raise QuantError("Future inputs changed an earlier lagged source opinion.")
        prefixes.append(str(day.date()))
    active = targets.dropna(how="all")
    for window in read_json(base.PROTOCOL)["windows"]:
        anchor = previous_session(window["start"])
        known = active.loc[:anchor].tail(1)
        covered = pd.concat([known, active.loc[window["start"] : window["end"]]])
        if known.empty or sum((covered[name] > 1e-10).any() for name in law.CORE) < 3:
            raise QuantError("Retain known original anchors and actual style participation.")
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
            "real_native_indicators_first_generated_after_actual_publication": True,
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


def evaluate(commit):
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        data, frozen, benchmarks = inputs()
        if events("technical_failure"):
            raise QuantError("Preserve the terminal lagged-source financial failure.")
        if RESULTS.exists():
            saved = read_json(RESULTS)
            if saved["frozen"] != frozen or events("review")[0]["body"] != saved["review"]:
                raise QuantError("A completed lagged-source result cannot change on recovery.")
            for key, ref in saved["review"]["paths"].items():
                base.verify_path(ref, frozen, ID, key)
            return reference(RESULTS)
        published = publication(frozen, commit)
        fences = [
            r
            for r in base.journal_events(base.JOURNAL)
            if r["kind"] == "publication" and r["body"].get("study_id") == STUDY
        ]
        if not fences:
            if events("path_started") or PREPARED.exists():
                raise QuantError("No lagged indicator or account before actual publication.")
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
            raise QuantError("Keep the unique actual lagged-source publication fence.")
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
                    "proof_kind": (
                        "actual_published_lagged_price_impact_native_and_main_book_outputs_v1"
                    ),
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
        raise QuantError("Export only a complete actually reviewed lagged-source policy.")
    for key, ref in data["review"]["paths"].items():
        base.verify_path(ref, data["frozen"], ID, key)
    body = {
        "schema_version": 1,
        "record_type": "published_lagged_price_impact_four_path_results",
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
        "real_native_indicators_and_targets_generated_after_actual_publication": True,
        "technical_expert_admission_review_complete": False,
        "expert_pool_changed": False,
        "prior_policies_or_accounts_or_benchmarks_replayed": False,
        "raw_prices_targets_daily_books_private_intents_exported": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    write_new_receipt(PUBLIC, body)
    return reference(PUBLIC)


def main():
    parser = argparse.ArgumentParser(description="One published lagged OHLCV participation source.")
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use the actual remotely published preregistration commit.")
    actions = {"freeze": initialize, "register": register, "export": export}
    value = evaluate(args.commit) if args.action == "evaluate" else actions[args.action]()
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
