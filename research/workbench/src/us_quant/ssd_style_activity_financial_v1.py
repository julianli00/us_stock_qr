from __future__ import annotations

import argparse
import ast
import json
import math
import os
from fractions import Fraction
from types import FunctionType, SimpleNamespace
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import ssd_calendar_seasonal_financial_v1 as shared
from us_quant import ssd_lagged_spy_price_impact_financial_v1 as template
from us_quant import ssd_research_v1 as base
from us_quant import style_activity_participation_v1 as primary
from us_quant import style_activity_source_v1 as activity_source
from us_quant.calendar import previous_session
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

ROOT, ID = base.ROOT, primary.ID
STUDY = "ssd_style_activity_participation_financial_20261105_v1"
POLICY = ROOT / "config/style-activity-financial-v1.json"
FOLDER = base.REPORTS / "style-activity-participation-financial-20261105-v1"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
READY = FOLDER / "program-readiness-v1.json"
PREPARED = FOLDER / "published-native-targets-v1.json"
RESULTS = FOLDER / "results.json"
PACKET = ROOT / "evidence/style_activity_financial_20261105_registration_v1.json"
PUBLIC = ROOT / "evidence/style_activity_financial_20261105_results_v1.json"
KIND = shared.KIND
CANDIDATE = {
    "id": ID,
    "construction": "monthly_lagged_signed_activity_three_positive_styles_or_actual_BIL",
}
law = SimpleNamespace(
    ROOT=primary.ROOT,
    ID=ID,
    POLICY=primary.POLICY,
    CORE=primary.CORE,
    UNIVERSE=primary.UNIVERSE,
    CLOCK=primary.CLOCK,
    CANDIDATE=CANDIDATE,
    validate=primary.validate,
    checked_market=primary.checked_market,
    eligible_sessions=primary.eligible_sessions,
    build_targets=primary.build_targets,
)
lifecycle, reference, unique_refs = shared.lifecycle, base.reference, activity_source.unique_refs
run_calendar, verify_opinions = shared.run_calendar, shared.verify_opinions
OWN = (
    "config/style-activity-financial-v1.json",
    "src/us_quant/ssd_style_activity_financial_v1.py",
    "tests/test_ssd_style_activity_financial_v1.py",
    "STYLE_ACTIVITY_FINANCIAL_V1.md",
    *primary.OWN,
    *primary.FIXTURE_INPUTS,
    "tests/test_ssd_lagged_spy_price_impact_financial_v1.py",
    "tests/test_main_capital_feedback_v1.py",
    "tests/test_expert_initial_monthly_execution_v1.py",
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
    if any(k in fields and fields[k] != v for k, v in counts.items()):
        raise QuantError(
            "Activity financial progress cannot override authoritative journal totals."
        )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "style_activity_financial_" + phase,
            "study_id": STUDY,
            "active_compute_job": None,
            **counts,
            **fields,
        },
    )


def validate_policy(policy):
    prefix = "reports/ssd-continuous-research-v1/style-activity-participation-primary-20261105-v1/"
    refs = {
        "primary_policy": {
            "path": "config/style-activity-participation-v1.json",
            "sha256": "bd89db354cec4c98b1e0ccac4d7418b42148c740d4751564f3bca4e22c9a8835",
        },
        "primary_preparation_freeze": {
            "path": prefix + "primary-implementation-freeze-v1.json",
            "sha256": "fd291d068aeb60ec624ac142e39aa90d3e591580320fafc06fabdf0f874ab074",
        },
        "primary_preparation_completion": {
            "path": prefix + "completion-v1.json",
            "sha256": "e2dc2a8157a288be8e1b82418a93ff5aaba3352d0cc60259d80cbd36ff90386d",
        },
        "primary_source_readiness": {
            "path": prefix + "source-input-readiness-v1.json",
            "sha256": "bdc6ed7f7a3f97ff44e2d1a72e4fde9da778f7ab6c7a71224aedbf3064e1c362",
        },
        "market_parent_freeze": {
            "path": (
                "reports/ssd-continuous-research-v1/mixed-mechanism-six-financial-20261105-v1/"
                "financial-implementation-freeze-v1.json"
            ),
            "sha256": "f6d9d667554ff1205931f760c56d4365c8e81fd41c17bd485223b773cfcfb06c",
        },
        "financial_contract_template": {
            "path": "config/calendar-seasonal-financial-v1.json",
            "sha256": "354f3867368eed66d0fb8ff3c31d46ad6620db21b512d1576d5bfb1452816030",
        },
    }
    original = read_json(base.checked_reference(refs["financial_contract_template"]))
    expected = {
        **{k: v for k, v in original.items() if k not in refs},
        **refs,
        "study_id": STUDY,
        "candidate_id": ID,
        "prior_journal_sequence": 1109,
        "prior_journal_head_sha256": (
            "7c0cf8f5a2ab8bd3b2ac8bf2c1c7ae068d51f7fad17a621cf262dd1a2bddb857"
        ),
        "prior_completed_configurations": 271,
        "source_input_interface": "original_MarketData_with_admitted_four_style_activity_only",
    }
    if policy != expected or any(type(policy[k]) is not type(v) for k, v in expected.items()):
        raise QuantError("Preserve the exact fd29 activity policy and original financial contract.")
    for row in refs.values():
        base.checked_reference(row)
    frozen = read_json(base.checked_reference(policy["primary_preparation_freeze"]))
    if (
        frozen["actual_registered"] is not False
        or frozen["financial_program_implemented"] is not False
        or frozen[
            "real_new_policy_indicators_scores_opinions_targets_accounts_or_financial_outcomes_computed"
        ]
        is not False
    ):
        raise QuantError("Keep the historical preparation-only primary state, not fake real proof.")
    for row in frozen["implementation_and_test_inputs"] + frozen["complete_dependency_inputs"]:
        base.checked_reference(row)
    ready = read_json(base.checked_reference(policy["primary_source_readiness"]))
    if (
        ready["primary_freeze"] != policy["primary_preparation_freeze"]
        or ready["source_references"] != frozen["source_references"]
        or ready["matched_price_volume_values_checked"] != 44880
        or any(ready["maximum_compatibility_difference_by_field"].values())
        or ready["old_activity_source_RF_imported_or_normalized"] is not False
        or ready["actual_new_indicators_scores_opinions_targets_accounts_or_metrics_computed"]
        is not False
    ):
        raise QuantError(
            "Consume the exact completed four-field compatibility proof without replay."
        )
    primary.validate(read_json(primary.POLICY))


def dependencies():
    pending = [ROOT / p for p in OWN if p.endswith(".py")]
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
                root = ROOT / "src" if name.startswith("us_quant") else ROOT / "tests"
                if not (
                    name.startswith("us_quant") or name.startswith("test_") or name == "conftest"
                ):
                    continue
                relative = name.replace(".", "/")
                pending.extend(
                    p
                    for p in (root / (relative + ".py"), root / relative / "__init__.py")
                    if p.is_file()
                )
    return unique_refs([*result.values(), *[reference(ROOT / p) for p in OWN]])


def validation_refs():
    proof = read_json(FOLDER / "validation-source-v1.json")
    xml, _ = [base.checked_reference(r) for r in proof["test_outputs"]]
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    if (
        proof["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
        or proof["exit_code"] != 0
        or proof["synthetic_only"] is not True
        or not suites
        or sum(int(s.get("tests", 0)) for s in suites) != proof["passed_tests"]
        or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
    ):
        raise QuantError("Freeze the actual synthetic program, full OHLCV proof and money Adapter.")
    return [*proof["test_outputs"], reference(FOLDER / "validation-source-v1.json")]


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the first complete or partial activity financial freeze.")
    with lifecycle.research_job():
        policy = read_json(POLICY)
        validate_policy(policy)
        design = read_json(FOLDER / "financial-design-v1.json")
        if (
            design["actual_native_event_seq"] != 169457
            or design["candidate_id"] != ID
            or design["new_real_policy_indicators_targets_accounts_or_financial_outcomes_computed"]
            is not False
        ):
            raise QuantError("Declare the same fd29 financial question before any real output.")
        counts = current_counts(base.journal_events(base.JOURNAL))
        if (
            counts["current_journal_event_count"] != 1109
            or counts["current_journal_head_sha256"] != policy["prior_journal_head_sha256"]
            or counts["total_legacy_plus_ssd_completed_configurations"] != 271
        ):
            raise QuantError("Freeze against the actual closed J1109/271 pre-outcome baseline.")
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        parent = read_json(base.checked_reference(policy["market_parent_freeze"]))["frozen"]
        check_frozen(parent)
        data = review_market({"market": parent["market"]}, ROOT)
        law.checked_market(data)
        _, _, source_refs = activity_source.source_references()
        deps = dependencies()
        removed = {
            "expert_pool",
            "expert_pool_definition_role",
            "study_specific_expert_count",
            "study_specific_expert_ids",
            "new_source_model_or_shadow_account_runs",
        }
        frozen = {
            **{k: v for k, v in parent.items() if k not in removed and not k.startswith("source_")},
            "study_id": STUDY,
            "policy": reference(POLICY),
            "primary_signal_policy": policy["primary_policy"],
            "execution_policy": reference(POLICY),
            "account_namespace": "signed_style_activity_published_program_v1",
            "construction": CANDIDATE["construction"],
            "construction_inputs": [policy["primary_policy"]],
            "decision_program_kind": KIND,
            "expert_pool_definition_role": "one_unadmitted_complete_source_not_pool_expansion",
            "main_capital_initialization": policy["initialization"],
            "prepublication_readiness": "synthetic_Interface_and_pinned_market_identity_only",
            "historical_target_proof_stage": policy["historical_target_proof_stage"],
            "prior_journal_sequence": 1109,
            "prior_journal_head_sha256": policy["prior_journal_head_sha256"],
            "prior_completed_configurations": 271,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in [*deps, *source_refs] if r["path"].startswith(("src/", "tests/"))],
                    *[reference(ROOT / p) for p in OWN if p.endswith(".md")],
                ]
            ),
            "policies": unique_refs(
                [
                    *parent["policies"],
                    *[r for r in [*deps, *source_refs] if r["path"].startswith("config/")],
                ]
            ),
            "prior_evidence": unique_refs(
                [
                    *parent["prior_evidence"],
                    *source_refs,
                    *[
                        policy[k]
                        for k in (
                            "primary_preparation_freeze",
                            "primary_preparation_completion",
                            "primary_source_readiness",
                            "market_parent_freeze",
                            "financial_contract_template",
                        )
                    ],
                    reference(FOLDER / "actual-invocation-v1.json"),
                    reference(FOLDER / "financial-design-v1.json"),
                ]
            ),
            "validation_evidence": unique_refs(
                [*parent["validation_evidence"], *validation_refs()]
            ),
            "decision_information_clock": law.CLOCK,
            "primary_input_lag_sessions": 1,
            "admitted_activity_symbol_scope": list(law.CORE),
            "old_activity_source_RF_imported_or_normalized": False,
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
                "source_references": source_refs,
                "package_versions": packages(),
                "new_real_source_indicators_targets_accounts_or_metrics_computed": False,
                "actual_registration_performed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="actual_register_schema3_without_real_policy_outputs")
    return reference(INTENT)


def inputs():
    function = template.inputs
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )()


def independent_balance(closes, reported_closes, reported_volumes):
    if (len(closes), len(reported_closes), len(reported_volumes)) != (64, 63, 63):
        raise QuantError("The independent activity proof needs every exact lagged input.")
    prices, raw, volume = [
        [float(v) for v in values] for values in (closes, reported_closes, reported_volumes)
    ]
    if not all(math.isfinite(v) and v > 0 for row in (prices, raw, volume) for v in row) or any(
        v != math.floor(v) for v in volume
    ):
        raise QuantError("Independent source inputs cannot be missing, nonpositive or fractional.")
    activity = [
        float(Fraction.from_float(a) * Fraction.from_float(b))
        for a, b in zip(raw, volume, strict=True)
    ]
    normalizer = max(activity)
    scaled = [float(Fraction.from_float(v) / Fraction.from_float(normalizer)) for v in activity]
    if not all(math.isfinite(v) and v > 0 for v in scaled):
        raise QuantError("Independent activity arithmetic cannot invent a uniform fallback.")
    signs = [int(b > a) - int(b < a) for a, b in zip(prices[:-1], prices[1:], strict=True)]
    numerator = float(
        sum((Fraction.from_float(w) * s for w, s in zip(scaled, signs, strict=True)), Fraction())
    )
    denominator = float(sum((Fraction.from_float(w) for w in scaled), Fraction()))
    scalar = float(Fraction.from_float(numerator) / Fraction.from_float(denominator))
    vector = float(np.dot(np.asarray(scaled), np.asarray(signs)) / np.sum(scaled))
    if not math.isfinite(scalar) or abs(vector - scalar) > 1e-12:
        raise QuantError("Independent scalar and NumPy activity fractions disagree.")
    return scalar, abs(vector - scalar)


def verify_targets(data, targets, audits, records):
    dates = law.eligible_sessions(data)
    if (
        not targets.index.equals(data.close.index)
        or not targets.columns.equals(data.close.columns)
        or not targets.dropna().index.equals(dates)
        or not targets.loc[~targets.index.isin(dates)].isna().all().all()
        or len(audits) != len(dates)
    ):
        raise QuantError("Keep every genuine native opinion and the whole original data lattice.")
    rows = iter(records)
    maximum = 0.0
    for index, day in enumerate(dates):
        before, after = next(rows, None), next(rows, None)
        position = int(data.close.index.get_loc(day))
        inputs, scores = {}, {}
        for symbol in law.CORE:
            values = (
                data.close[symbol].iloc[position - 64 : position],
                data.raw_close[symbol].iloc[position - 63 : position],
                data.volume[symbol].iloc[position - 63 : position],
            )
            scores[symbol], gap = independent_balance(*values)
            maximum = max(maximum, gap)
            inputs[symbol] = {
                key: [float(v).hex() for v in values_for_key]
                for key, values_for_key in zip(
                    ("adjusted_closes", "reported_closes", "reported_share_volumes"),
                    values,
                    strict=True,
                )
            }
        positive = [s for s in law.CORE if scores[s] > 0]
        selected = sorted(positive, key=lambda s: (-round(scores[s], 12), s))[:3]
        selected = selected if len(positive) >= 3 else ["BIL"]
        oracle = pd.Series(0.0, index=law.UNIVERSE)
        oracle.loc[selected] = float(Fraction.from_float(0.98) / len(selected))
        expected_audit = {
            "decision_session": str(day.date()),
            "decision_information_clock": law.CLOCK,
            "first_prior_adjusted_close": str(data.close.index[position - 64].date()),
            "first_activity_session": str(data.close.index[position - 63].date()),
            "last_input_session": str(data.close.index[position - 1].date()),
            "return_and_activity_observations": 63,
            "activity_balances": scores,
            "ranking_keys": {s: round(scores[s], 12) for s in law.CORE},
            "strictly_positive_style_count": len(positive),
            "selected_assets": selected,
            "reason": "three_positive_activity_styles"
            if len(positive) >= 3
            else "defined_insufficient_positive_activity_participation",
            "invalid_input_defaulted_to_cash": False,
            "decision_session_own_or_future_bar_used": False,
            "reported_activity_is_actual_flow_trade_direction_or_market_breadth": False,
            "price_warmup_implies_mature_NET_score": False,
            "counterfactual_historical_availability": True,
            "actual_policy_introduction_or_admission_backdated": False,
            "independent_forward_validation": False,
        }
        expected_before = {
            "record": "decision_input",
            "decision_session": str(day.date()),
            "information_clock": law.CLOCK,
            "activity_symbols": list(law.CORE),
            "activity_session_dates": data.close.index[position - 63 : position]
            .strftime("%Y-%m-%d")
            .tolist(),
            "inputs_binary64_hex": inputs,
            "decision_session_own_or_future_bar_used": False,
        }
        expected_after = {
            "record": "decision_resolved",
            "audit": expected_audit,
            "asset_order": list(law.UNIVERSE),
            "complete_target_binary64_hex": [float(v).hex() for v in oracle],
        }
        if (
            before != expected_before
            or after != expected_after
            or audits[index] != expected_audit
            or not np.array_equal(targets.loc[day].to_numpy(), oracle.to_numpy())
        ):
            raise QuantError("Independent binary64 lagged inputs, activity state or target differ.")
    if next(rows, None) is not None:
        raise QuantError("Retain exactly one input and one resolved record per native opinion.")
    return {
        "native_month_end_opinions": len(dates),
        "all_independent_signed_activity_targets_exact": True,
        "source_price_volume_binary64_inputs_checked": 4 * (64 + 63 + 63) * len(dates),
        "independent_four_style_balance_checks": 4 * len(dates),
        "maximum_independent_scalar_numpy_balance_gap": maximum,
        "decision_day_or_future_input_used": False,
        "full_MarketData_four_fund_activity_scope_preserved": True,
        "old_activity_source_RF_imported_or_normalized": False,
    }


def prepare_source(data, frozen):
    fence = publication_fence()
    if PREPARED.exists():
        saved = read_json(PREPARED)
        if saved["frozen_specification_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting complete activity-source preparation.")
        for key in ("targets", "proof"):
            base.checked_reference(saved[key])
        return saved
    attempt = FOLDER / "source-targets-started-v1.json"
    if attempt.exists():
        raise QuantError("Preserve incomplete activity targets instead of generating them twice.")
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
    with trace.open() as stream:
        checked = verify_targets(data, targets, audits, (json.loads(row) for row in stream))
    if len(audits) != 131:
        raise QuantError("Keep every predeclared fully warmed native month end.")
    prefixes = []
    for position in (0, 6, 12, 65, 130):
        day = pd.Timestamp(audits[position]["decision_session"])
        early = MarketData(
            data.open.loc[:day],
            data.close.loc[:day],
            data.raw_close.loc[:day],
            data.volume.loc[:day],
            data.risk_free.loc[:day],
        )
        prefix, previous = law.build_targets(
            early, read_json(law.POLICY), before_decision=lifecycle.runtime_guard
        )
        if not prefix.equals(targets.loc[:day]) or previous != audits[: position + 1]:
            raise QuantError("Future source values changed a prior signed-activity opinion.")
        prefixes.append(str(day.date()))
    coverage = []
    active = targets.dropna(how="all")
    for window in read_json(base.PROTOCOL)["windows"]:
        anchor = previous_session(window["start"])
        known = active.loc[:anchor].tail(1)
        covered = pd.concat([known, active.loc[window["start"] : window["end"]]])
        styles = [s for s in law.CORE if (covered[s] > 1e-10).any()]
        if known.empty or len(styles) < 3 or anchor in active.index:
            raise QuantError("Keep real prior opinions, original seeds and actual style coverage.")
        coverage.append(
            {
                "years": window["years"],
                "capital_anchor": str(anchor.date()),
                "prior_native_opinion_session": str(known.index[0].date()),
                "actual_factor_symbols": styles,
                "capital_seed_is_new_policy_opinion": False,
            }
        )
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
            "formal_factor_coverage": coverage,
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
            raise QuantError("Preserve the terminal activity financial identity without retry.")
        if RESULTS.exists():
            saved = read_json(RESULTS)
            if saved["frozen"] != frozen or events("review")[0]["body"] != saved["review"]:
                raise QuantError("A completed activity financial result cannot change on recovery.")
            for key, ref in saved["review"]["paths"].items():
                base.verify_path(ref, frozen, ID, key)
            return reference(RESULTS)
        published = publication(frozen, commit)
        fences = [
            row
            for row in base.journal_events(base.JOURNAL)
            if row["kind"] == "publication" and row["body"].get("study_id") == STUDY
        ]
        if not fences:
            if events("path_started") or PREPARED.exists():
                raise QuantError("No real activity indicator or account may precede publication.")
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
            raise QuantError("Keep the unique actual original activity publication fence.")
        try:
            prepared = prepare_source(data, frozen)
            targets = source_frame(prepared)
            paths = {}
            for window in read_json(base.PROTOCOL)["windows"]:
                for scenario in read_json(base.PROTOCOL)["accounts"]["scenarios"]:
                    lifecycle.runtime_guard()
                    key = f"{window['years']}y-{scenario['id']}"
                    paths[key] = evaluate_path(data, targets, frozen, window, scenario, benchmarks)
            prefixes = prefix_audits(data, targets)
            base.append_event(
                base.JOURNAL,
                "target_proof",
                ID,
                {
                    "proof_kind": "actual_published_signed_style_activity_and_main_book_outputs_v1",
                    "proof_stage": "after_four_actual_published_program_paths",
                    "native_source_proof": prepared,
                    "real_main_capital_prefix_audit": prefixes,
                    "all_original_dates_and_fixed_policy_retained": True,
                },
            )
            review = {
                "reviewed_at": utc_now(),
                "frozen_specification_sha256": digest_json(frozen),
                "status": "historical_point_gates_passed"
                if all(row["all_point_gates_pass"] for row in paths.values())
                else "rejected",
                "paths": {k: reference(FOLDER / "accounts" / k / "complete.json") for k in paths},
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
                    "real_main_capital_prefix_audit": prefixes,
                },
            )
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            record_failure("funding_or_causality", error)
            raise
        progress("four_paths_complete", next_step="export_then_separate_technical_expert_review")
    return reference(RESULTS)


def export():
    lifecycle.runtime_guard()
    data = read_json(RESULTS)
    if set(data["paths"]) != base.PATH_KEYS or len(events("review")) != 1:
        raise QuantError("Export only the complete actually reviewed signed-activity policy.")
    for key, ref in data["review"]["paths"].items():
        base.verify_path(ref, data["frozen"], ID, key)
    body = {
        "schema_version": 1,
        "record_type": "published_signed_style_activity_four_path_results",
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
        "old_activity_source_RF_imported_or_normalized": False,
        "prior_policies_or_accounts_or_benchmarks_replayed": False,
        "raw_prices_targets_daily_books_private_intents_exported": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    write_new_receipt(PUBLIC, body)
    return reference(PUBLIC)


def main():
    parser = argparse.ArgumentParser(description="One published signed-style-activity policy.")
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use an actual remotely published complete schema3 program commit.")
    actions = {"freeze": initialize, "register": register, "export": export}
    print(
        json.dumps(evaluate(args.commit) if args.action == "evaluate" else actions[args.action]())
    )


if __name__ == "__main__":
    main()
