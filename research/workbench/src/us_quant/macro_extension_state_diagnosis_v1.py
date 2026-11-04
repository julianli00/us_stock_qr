from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from collections import Counter
from importlib.metadata import distributions
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import ssd_macro_expert_inclusion_v1 as saved
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.storage import digest_json, read_json, utc_now

ROOT, BASE = saved.ROOT, saved.base
FOLDER = BASE.REPORTS / "macro-extension-state-diagnosis-20261104-v1"
DESIGN = FOLDER / "design-decision-v1.json"
INTENT = FOLDER / "analysis-intent-v1.json"
RESULT = FOLDER / "diagnostic-result-v1.json"
SUMMARY = FOLDER / "summary-v1.json"
PUBLIC = ROOT / "evidence/macro_extension_state_diagnosis_20261104_v1.json"
DIAGNOSIS = "ssd_macro_extension_state_diagnosis_20261104_v1"
IDS = saved.rules.EXPERT_IDS
SCORE_TOLERANCE = 1e-8
OWN = (
    "src/us_quant/macro_extension_state_diagnosis_v1.py",
    "tests/test_macro_extension_state_diagnosis_v1.py",
    "tests/test_macro_expert_inclusion_v1.py",
    "tests/test_strategy_expert_pool_eight_v1.py",
    "tests/test_strategy_expert_pool_v5.py",
    "tests/test_strategy_expert_pool_v4.py",
    "tests/test_strategy_expert_pool_v3.py",
    "tests/conftest.py",
    "MACRO_EXTENSION_STATE_DIAGNOSIS_V1.md",
)


def exact(value):
    if isinstance(value, float):
        if not math.isfinite(value):
            raise QuantError("A nonfinite retained input cannot become a successful comparison.")
        return ("binary64", value.hex())
    if isinstance(value, list):
        return tuple(exact(v) for v in value)
    if isinstance(value, dict):
        return tuple((key, exact(value[key])) for key in sorted(value))
    return (type(value).__name__, value)


def _weights(values, size):
    if (
        not isinstance(values, list)
        or len(values) != size
        or any(type(v) is not float or not math.isfinite(v) or v < 0 for v in values)
        or abs(math.fsum(values) - 1) > 1e-12
    ):
        raise QuantError("Retain complete finite funded expert coordinates without normalization.")
    return values


def _state(before, resolved, identifiers):
    audit = resolved["audit"]
    previous = _weights(before["previous_weights"], len(identifiers))
    chosen = _weights(audit["rolling_expert_weights"], len(identifiers))
    observations = before["observations"]
    if (
        before["record"] != "decision_input"
        or resolved["record"] != "decision_resolved"
        or before["decision_index"] != resolved["decision_index"]
        or before["as_of"] != audit["decision_session"]
        or before["information_clock"] != saved.rules.CLOCK
        or audit["decision_information_clock"] != saved.rules.CLOCK
        or exact(previous) != exact(audit["previous_rolling_expert_weights"])
        or before["previous_weights_binary64_hex"] != [v.hex() for v in previous]
        or tuple(o["id"] for o in observations) != identifiers
        or before["binary64_values_truncated_to_fixed_decimals"] is not False
        or audit["shadow_wealth_added_to_main_capital"] is not False
        or audit["source_feedback_floor_transferred_to_main_account"] is not False
        or audit["counterfactual_historical_expert_availability"] is not True
        or audit["selection"]["safety_or_success_claimed"] is not False
    ):
        raise QuantError("Keep exact retained identities, original clocks and capital separation.")
    for observation in observations:
        target = observation["target"]
        if (
            observation["as_of"] != before["as_of"]
            or observation["target_known_at"] is None
            or observation["target_known_at"] > before["as_of"]
            or len(target) != len(observation["asset_order"])
            or observation["asset_order"] != resolved["asset_order"]
            or len(set(observation["asset_order"])) != len(target)
            or any(not math.isfinite(v) or v < 0 for v in target)
            or abs(math.fsum(target) - 0.98) > 1e-12
            or len(observation["NET_returns"]) != len(observation["RF"])
            or len(observation["return_dates"]) != len(observation["NET_returns"])
            or any(not math.isfinite(v) for v in observation["NET_returns"] + observation["RF"])
            or observation["score_mature"] != (len(observation["NET_returns"]) == 126)
        ):
            raise QuantError("Retain known recommendations and their exact assessment windows.")
    return audit, previous, chosen


def _pair_map(selection, identifiers):
    expected = {tuple(sorted(pair)) for pair in itertools.combinations(identifiers, 2)}
    rows = selection["pairs"]
    if (
        len(rows) != len(expected)
        or len({tuple(row["pair"]) for row in rows}) != len(rows)
        or {tuple(row["pair"]) for row in rows} != expected
    ):
        raise QuantError("Retain every shared and extension-only pair, not selected pairs alone.")
    if selection["selected_pair"] is None:
        raise QuantError("A mature scored state must retain its actual selected pair.")
    return {tuple(row["pair"]): row for row in rows}


def _components(observations, weights, cost):
    if not math.isfinite(cost) or cost < 0:
        raise QuantError("Use the recorded finite incremental switching proxy, not paid fees.")
    dates = observations[0]["return_dates"]
    if (
        len(dates) != 126
        or dates != sorted(set(dates))
        or dates[-1] != observations[0]["as_of"]
        or any(o["return_dates"] != dates or not o["score_mature"] for o in observations)
        or any(exact(o["RF"]) != exact(observations[0]["RF"]) for o in observations)
    ):
        raise QuantError("Use the same original mature126 NET/RF observations without filling.")
    portfolio = [
        math.fsum(
            weight * (observation["NET_returns"][t] - observation["RF"][t])
            for weight, observation in zip(weights, observations, strict=True)
        )
        for t in range(126)
    ]
    result = []
    for values in (portfolio[:84], portfolio[84:]):
        if all(v == values[0] for v in values):
            raise QuantError("An exactly constant pair has no defined risk score or safety claim.")
        mean = math.fsum(values) / len(values)
        variance = math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1)
        if variance <= 0 or not math.isfinite(variance):
            raise QuantError("A recorded scored pair needs positive finite risk, not a floor.")
        deviation = math.sqrt(variance)
        result.append((math.sqrt(252) * mean / deviation, math.sqrt(252) * cost / 21 / deviation))
    return result


def compare_step(control, extension):
    before8, resolved8 = control
    before9, resolved9 = extension
    audit8, previous8, chosen8 = _state(before8, resolved8, IDS[:8])
    audit9, previous9, chosen9 = _state(before9, resolved9, IDS)
    if (
        before8["decision_index"] != before9["decision_index"]
        or before8["as_of"] != before9["as_of"]
        or exact(before8["observations"]) != exact(before9["observations"][:8])
        or resolved8["asset_order"] != resolved9["asset_order"]
    ):
        raise QuantError(
            "Shared NET, RF, targets, dates and identities must be binary64-identical."
        )
    same_previous = exact(previous8 + [0.0]) == exact(previous9)
    selection8, selection9 = audit8["selection"], audit9["selection"]
    if selection8["scores_available"] is not selection9["scores_available"]:
        raise QuantError("Source extension cannot rewrite shared score maturity.")
    result = {
        "decision_index": before8["decision_index"],
        "decision_session": before8["as_of"],
        "shared_observations_binary64_identical": True,
        "shared_source_NET_RF_values_compared": sum(
            len(o["NET_returns"]) + len(o["RF"]) for o in before8["observations"]
        ),
        "control_previous_weights": previous8,
        "extension_previous_weights": previous9,
        "preceding_macro_weight": previous9[-1],
        "preceding_full_allocation_exact_equal_after_zero_extension": same_previous,
        "preceding_state": (
            "exact_equal"
            if same_previous
            else (
                "macro_mass_present"
                if previous9[-1] > 0
                else "macro_zero_but_original_source_allocation_differs"
            )
        ),
        "current_macro_weight": chosen9[-1],
        "control_selected_pair": selection8["selected_pair"],
        "extension_selected_pair": selection9["selected_pair"],
        "control_input_resolved_sha256": digest_json([before8, resolved8]),
        "extension_input_resolved_sha256": digest_json([before9, resolved9]),
        "common_pair_rows": [],
        "extension_only_pair_records": [],
    }
    if not selection8["scores_available"]:
        if any(
            selection["reason"] != "unscored_initial_equal_weights"
            or selection["pairs"] != []
            or selection["selected_pair"] is not None
            for selection in (selection8, selection9)
        ):
            raise QuantError("An unscored source state cannot become a safe successful score.")
        result["choice_class"] = "original_immature_no_score"
        return result
    pairs8, pairs9 = _pair_map(selection8, IDS[:8]), _pair_map(selection9, IDS)
    result["choice_class"] = (
        "current_macro_selected"
        if IDS[-1] in selection9["selected_pair"]
        else (
            "same_original_pair_current_macro_zero"
            if selection8["selected_pair"] == selection9["selected_pair"]
            else "different_original_pair_current_macro_zero"
        )
    )
    for selected, weights, pairs in (
        (selection8["selected_pair"], chosen8, pairs8),
        (selection9["selected_pair"], chosen9, pairs9),
    ):
        selected_row = pairs[tuple(selected)]
        if "unscorable_reason" in selected_row or exact(selected_row["expert_weights"]) != exact(
            weights
        ):
            raise QuantError(
                "Retain the actual recorded choice without running a replacement rank."
            )
    for names, old in pairs8.items():
        new = pairs9[names]
        row = {
            "pair": list(names),
            "was_control_selected": list(names) == selection8["selected_pair"],
            "was_extension_selected": list(names) == selection9["selected_pair"],
        }
        if "unscorable_reason" in old or "unscorable_reason" in new:
            if old != new:
                raise QuantError("Do not drop or relabel an unscorable common pair.")
            row.update(status="original_pair_unscorable", original_reason=old["unscorable_reason"])
            result["common_pair_rows"].append(row)
            continue
        weights8 = _weights(old["expert_weights"], 8)
        weights9 = _weights(new["expert_weights"], 9)
        if (
            weights9[-1] != 0
            or any(
                weight != 0
                for identifier, weight in zip(IDS[:8], weights8, strict=True)
                if identifier not in names
            )
            or any(
                weight != 0
                for identifier, weight in zip(IDS, weights9, strict=True)
                if identifier not in names
            )
        ):
            raise QuantError("A shared pair must not hide an extra source or macro allocation.")
        if any(pair["cost_proxy_is_actual_main_account_cost"] is not False for pair in (old, new)):
            raise QuantError("An incremental nominal penalty is not an actual paid fee.")
        components8 = _components(
            before8["observations"], weights8, old["estimated_incremental_switch_cost_fraction"]
        )
        components9 = _components(
            before9["observations"][:8],
            weights9[:8],
            new["estimated_incremental_switch_cost_fraction"],
        )
        row.update(
            status="recorded_scores_reconciled",
            pair_weights_binary64_identical=exact(weights8 + [0.0]) == exact(weights9),
            maximum_recorded_pair_weight_difference=max(
                abs(a - b) for a, b in zip(weights8, weights9[:8], strict=True)
            ),
            recorded_switch_proxy_difference=(
                new["estimated_incremental_switch_cost_fraction"]
                - old["estimated_incremental_switch_cost_fraction"]
            ),
            recorded_selection_score_key_changed=(
                old["selection_score_key"] != new["selection_score_key"]
            ),
            segments={},
        )
        for label, fields, a, b in zip(
            ("training", "validation"),
            ("train_net_risk_score", "validation_net_risk_score"),
            components8,
            components9,
            strict=True,
        ):
            if not all(math.isfinite(v) for v in (old[fields], new[fields])):
                raise QuantError("An undefined recorded score cannot be a reconstructed success.")
            error = max(abs(a[0] - a[1] - old[fields]), abs(b[0] - b[1] - new[fields]))
            if error > SCORE_TOLERANCE:
                raise QuantError(
                    "The original scalar score agreement failed; retain this identity."
                )
            observed = new[fields] - old[fields]
            source_component, proxy_component = b[0] - a[0], a[1] - b[1]
            row["segments"][label] = {
                "recorded_score_difference": observed,
                "source_NET_reference_component": source_component,
                "incremental_switch_proxy_component": proxy_component,
                "reconciliation_residual": observed - source_component - proxy_component,
                "maximum_individual_recorded_score_reconstruction_error": error,
            }
        result["common_pair_rows"].append(row)
    result["extension_only_pair_records"] = [
        row for pair, row in pairs9.items() if pair not in pairs8
    ]
    return result


def read_records(path: Path):
    with path.open() as stream:
        for index in itertools.count():
            line = stream.readline()
            if not line:
                return
            following = stream.readline()
            if not following:
                raise QuantError("A truncated retained trace cannot become a shorter success.")
            try:
                before, resolved = json.loads(line), json.loads(following)
            except json.JSONDecodeError as error:
                raise QuantError(
                    "Malformed retained JSON cannot become an empty source state."
                ) from error
            if (
                not isinstance(before, dict)
                or not isinstance(resolved, dict)
                or type(before.get("decision_index")) is not int
                or type(resolved.get("decision_index")) is not int
                or before["decision_index"] != index
                or resolved["decision_index"] != index
                or before.get("record") != "decision_input"
                or resolved.get("record") != "decision_resolved"
            ):
                raise QuantError("Keep every original input/resolved pair in its full order.")
            yield before, resolved


def _source_refs():
    design = read_json(DESIGN)
    if (
        design["diagnosis_id"] != DIAGNOSIS
        or design["actual_continuation_event"] != 149040
        or design["actual_new_comparison_or_component_statistics_computed"] is not False
        or design["new_financial_trials"] != 0
        or design["maximum_actual_stream_passes"] != 1
    ):
        raise QuantError("Keep the fixed independent question and one-pass reading convention.")
    sources, dependencies = [BASE.reference(DESIGN)], []
    for relative in design["source_folders"]:
        folder = ROOT / relative
        frozen = read_json(folder / "financial-implementation-freeze-v1.json")
        sources.extend(frozen["source_references"])
        dependencies.extend(frozen["complete_dependency_inputs"])
        sources.extend(
            BASE.reference(folder / name)
            for name in (
                "actual-decision-inputs-resolved-v1.jsonl",
                "information-audits-v1.json",
                "prepared-target-bundle-v1.json",
                "financial-implementation-freeze-v1.json",
                "completion-v1.json",
                "results.json",
            )
        )
    sources.extend(
        BASE.reference(BASE.REPORTS / folder / name)
        for folder, name in (
            ("expert-tail-cost-proxy-diagnosis-20261030-v1", "design-v1.json"),
            ("expert-tail-cost-proxy-diagnosis-20261030-v1", "summary-v1.json"),
            ("pair-risk-forecast-diagnosis-20261103-v1", "design-decision-v1.json"),
            ("pair-risk-forecast-diagnosis-20261103-v1", "summary-v1.json"),
        )
    )
    dependencies.extend(BASE.reference(ROOT / name) for name in OWN)
    return saved.unique_refs(sources), saved.unique_refs(dependencies)


def freeze():
    with lifecycle.research_job():
        if INTENT.exists():
            raise QuantError("Resume the existing diagnostic intent, never refreeze its identity.")
        os.umask(0o077)
        sources, dependencies = _source_refs()
        xml, log = FOLDER / "offline-tests-v1.xml", FOLDER / "offline-tests-v1.log"
        validation = read_json(FOLDER / "validation-source-v1.json")
        suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
        tested = {
            case.get("name", "").split("[", 1)[0]
            for suite in suites
            for case in suite.findall("testcase")
        }
        required = {
            "test_complete_shared_inputs_all_pairs_and_immature_states_preserved",
            "test_one_ulp_or_signed_zero_shared_input_change_is_not_tolerance_matched",
            "test_shared_provenance_or_clock_cannot_change",
            "test_recorded_score_gap_components_are_algebra_not_new_ranking",
            "test_corrupt_nonselected_pairs_are_not_hidden",
            "test_zero_current_macro_does_not_erase_recorded_preceding_state",
            "test_constant_scored_pair_is_undefined_not_a_variance_floor",
            "test_trace_requires_full_ordered_input_resolved_pairs",
            "test_public_summary_does_not_export_monthly_inputs_targets_or_source_weights",
            "test_second_complete_or_partial_actual_attempt_is_terminal",
        }
        outputs = [BASE.reference(xml), BASE.reference(log)]
        own = [BASE.reference(ROOT / name) for name in OWN]
        if (
            not suites
            or not required.issubset(tested)
            or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
            or validation["passed_tests"] != sum(int(s.get("tests", 0)) for s in suites)
            or validation["implementation_and_test_inputs"] != own
            or validation["outputs"] != outputs
            or validation["exit_code"] != 0
        ):
            raise QuantError(
                "Bind the exact successful synthetic implementation before arithmetic."
            )
        for ref in sources + dependencies:
            BASE.checked_reference(ref)
        history = BASE.journal_events(BASE.JOURNAL)
        if len(history) != 852 or history[-1]["sha256"] != (
            "4d9979d6234629fcb9e342c87118431c4236d316082988d4ff30967d99ab0bde"
        ):
            raise QuantError("Use the actual closed852-event financial state without new trials.")
        write_new_receipt(
            INTENT,
            {
                "diagnosis_id": DIAGNOSIS,
                "frozen_at": utc_now(),
                "source_references": sources,
                "implementation_dependencies": dependencies,
                "implementation_and_test_inputs": own,
                "validation_refs": [*outputs, BASE.reference(FOLDER / "validation-source-v1.json")],
                "package_versions": sorted(
                    (d.metadata["Name"], d.version) for d in distributions()
                ),
                "financial_counts": current_counts(history),
                "actual_new_diagnostic_values_computed": False,
                "maximum_actual_stream_passes": 1,
                "new_financial_trials": 0,
            },
        )
    return BASE.reference(INTENT)


def _distribution(values):
    return {
        "observations": len(values),
        "median_absolute": float(np.median(np.abs(values))),
        "p90_absolute": float(np.quantile(np.abs(values), 0.9)),
        "maximum_absolute": max(abs(v) for v in values),
    }


def aggregate(decisions):
    rows = [row for d in decisions for row in d["common_pair_rows"]]
    scored = [row for row in rows if row["status"] == "recorded_scores_reconciled"]
    result = {
        "decisions": len(decisions),
        "choice_classes": dict(Counter(d["choice_class"] for d in decisions)),
        "preceding_state_classes": dict(Counter(d["preceding_state"] for d in decisions)),
        "shared_source_NET_RF_values_compared": sum(
            d["shared_source_NET_RF_values_compared"] for d in decisions
        ),
        "common_pair_records": len(rows),
        "scored_common_pair_records": len(scored),
        "unscorable_common_pair_records": len(rows) - len(scored),
        "extension_only_pair_records_retained": sum(
            len(d["extension_only_pair_records"]) for d in decisions
        ),
        "common_pair_weight_nonidentity_records": sum(
            not r["pair_weights_binary64_identical"] for r in scored
        ),
        "maximum_recorded_pair_weight_difference": max(
            r["maximum_recorded_pair_weight_difference"] for r in scored
        ),
        "common_pair_selection_score_key_changed_records": sum(
            r["recorded_selection_score_key_changed"] for r in scored
        ),
        "different_original_pair_current_macro_zero_preceding_states": dict(
            Counter(
                d["preceding_state"]
                for d in decisions
                if d["choice_class"] == "different_original_pair_current_macro_zero"
            )
        ),
        "segments": {},
    }
    fields = (
        "recorded_score_difference",
        "source_NET_reference_component",
        "incremental_switch_proxy_component",
        "reconciliation_residual",
        "maximum_individual_recorded_score_reconstruction_error",
    )
    for segment in ("training", "validation"):
        result["segments"][segment] = {
            field: _distribution([row["segments"][segment][field] for row in scored])
            for field in fields
        }
    return result


def public_report(result):
    return {
        "schema_version": 1,
        "diagnosis_id": DIAGNOSIS,
        "completed_at": result["completed_at"],
        "analysis_intent_sha256": result["analysis_intent"]["sha256"],
        "aggregates": result["aggregates"],
        "original_full_shared_source_observations_binary64_identical": True,
        "source_NET_already_includes_native_shadow_costs": True,
        "incremental_switch_proxy_is_actual_paid_main_fee": False,
        "recorded_score_gap_reconciliation_is_new_chooser_or_main_PNL_attribution": False,
        "post_outcome_adaptive_pool_selection_bias_preserved": True,
        "independent_samples_or_statistical_significance_claimed": False,
        "original_rosters_admission_scope_and_financial_rejections_changed": False,
        "financial_counts_unchanged": result["financial_counts_unchanged"],
        "new_source_requests": 0,
        "new_targets_or_accounts": 0,
        "new_financial_trials": 0,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }


def run():
    lifecycle.runtime_guard()
    attempt = FOLDER / "attempt-v1.json"
    if attempt.exists() or RESULT.exists():
        raise QuantError("Preserve the first complete or partial actual diagnostic; never retry.")
    intent = read_json(INTENT)
    sources, dependencies = _source_refs()
    if (
        intent["diagnosis_id"] != DIAGNOSIS
        or intent["source_references"] != sources
        or intent["implementation_dependencies"] != dependencies
        or intent["actual_new_diagnostic_values_computed"] is not False
        or intent["maximum_actual_stream_passes"] != 1
        or intent["package_versions"]
        != [list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())]
    ):
        raise QuantError("The exact diagnostic sources, implementation or environment changed.")
    for ref in sources + dependencies + intent["validation_refs"]:
        BASE.checked_reference(ref)
    with lifecycle.research_job():
        os.umask(0o077)
        before = current_counts(BASE.journal_events(BASE.JOURNAL))
        if before != intent["financial_counts"]:
            raise QuantError("A new financial event invalidates this exact one-pass source state.")
        write_new_receipt(
            attempt,
            {
                "started_at": utc_now(),
                "pid": os.getpid(),
                "analysis_intent": BASE.reference(INTENT),
            },
        )
        completed = []
        trace = FOLDER / "decision-results-v1.jsonl"
        try:
            design = read_json(DESIGN)
            folders = [ROOT / p for p in design["source_folders"]]
            logs = [p / "actual-decision-inputs-resolved-v1.jsonl" for p in folders]
            if [BASE.reference(p)["sha256"] for p in logs] != [
                design["control_trace_sha256"],
                design["extension_trace_sha256"],
            ]:
                raise QuantError("The exact closed control/extension trace changed.")
            audits = [read_json(p / "information-audits-v1.json")["decisions"] for p in folders]
            with trace.open("x") as stream:
                for index, (control, extension) in enumerate(
                    zip(read_records(logs[0]), read_records(logs[1]), strict=True)
                ):
                    lifecycle.runtime_guard()
                    if any(
                        pair[1]["audit"] != audits[i][index]
                        or pair[1]["asset_order"] != design["common_asset_order"]
                        for i, pair in enumerate((control, extension))
                    ):
                        raise QuantError("Retain the full original audit and14-asset declaration.")
                    result = compare_step(control, extension)
                    stream.write(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    completed.append(result)
            aggregates = aggregate(completed)
            if (
                aggregates["decisions"] != 121
                or aggregates["choice_classes"].get("original_immature_no_score") != 7
                or aggregates["common_pair_records"] != 114 * 28
                or aggregates["extension_only_pair_records_retained"] != 114 * 8
                or any(len(a) != 121 for a in audits)
                or current_counts(BASE.journal_events(BASE.JOURNAL)) != before
            ):
                raise QuantError("Do not shorten source coverage or change financial trial counts.")
            for ref in sources + dependencies:
                BASE.checked_reference(ref)
            result = {
                "completed_at": utc_now(),
                "analysis_intent": BASE.reference(INTENT),
                "decisions_resolved": BASE.reference(trace),
                "aggregates": aggregates,
                "financial_counts_unchanged": before,
                "all_frozen_source_and_code_bytes_unchanged": True,
                "new_source_requests_or_financial_trials": 0,
                "new_chooser_weights_targets_or_accounts": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
            }
            write_new_receipt(RESULT, result)
            write_new_receipt(SUMMARY, public_report(result))
            write_new_receipt(PUBLIC, public_report(result))
        except (QuantError, OSError, ValueError, KeyError, TypeError, IndexError) as error:
            write_new_receipt(
                FOLDER / "failure-v1.json",
                {
                    "failed_at": utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "complete_decisions_retained": len(completed),
                    "partial_trace_preserved": True,
                    "new_financial_trials": 0,
                },
            )
            raise
    return BASE.reference(SUMMARY)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One cached source-extension state diagnosis.")
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    print(json.dumps(freeze() if args.action == "freeze" else run(), indent=2))
