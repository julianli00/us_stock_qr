from __future__ import annotations

import argparse
import json
import math
import os
from importlib.metadata import distributions
from itertools import zip_longest
from types import FunctionType
from xml.etree import ElementTree

from us_quant import hmm_session_intervention_v1 as saved
from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import ssd_research_v1 as base
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.ssd_style_activity_financial_v1 import dependencies as _dependencies
from us_quant.storage import read_json, utc_now, write_json

ROOT = base.ROOT
FOLDER = base.REPORTS / "hmm-session-intervention-diagnosis-20261105-v1"
POLICY = ROOT / "config/hmm-session-intervention-diagnosis-v1.json"
INTENT = FOLDER / "analysis-intent-v1.json"
OUTPUT = FOLDER / "diagnostic-v1.json"
PUBLIC = ROOT / "evidence/hmm_session_intervention_diagnosis_20261105_v1.json"
OWN = (
    "config/hmm-session-intervention-diagnosis-v1.json",
    "src/us_quant/hmm_session_intervention_v1.py",
    "src/us_quant/ssd_hmm_session_intervention_v1.py",
    "tests/test_hmm_session_intervention_v1.py",
    "HMM_SESSION_INTERVENTION_DIAGNOSIS_V1.md",
    "tests/conftest.py",
)
reference = base.reference


def validate(policy):
    expected = {
        "schema_version": 1,
        "study_id": "ssd_hmm_session_intervention_diagnosis_20261105_v1",
        "prior_journal_sequence": 1211,
        "prior_completed_configurations": 280,
        "input_catalogue": {
            "path": (
                "reports/ssd-continuous-research-v1/"
                "hmm-session-intervention-diagnosis-20261105-v1/source-input-catalogue-v1.json"
            ),
            "sha256": "2d1fead19061b9120e9e4dc1434f1b86551682dcce88082b9c58f87e9457ea76",
        },
        "variants": ["close", "overnight", "intraday"],
        "rules": ["equal", "rolling"],
        "all_original_decisions_per_pool": 121,
        "source_slot": 2,
        "core_assets": ["MTUM", "VLUE", "QUAL", "USMV"],
        "target_reconciliation_tolerance": saved.TOLERANCE,
        "exact_rational_binary64_identity_required": True,
        "noncore_source_opinions_exactly_unchanged": True,
        "other_seven_observations_exactly_unchanged": True,
        "channels": ["direct_opinion", "allocation_revision", "interaction"],
        "classification_uses_exact_saved_binary64_equality": True,
        "saved_target_L1_is_executed_turnover": False,
        "decomposition_is_causal_alpha_or_main_PnL": False,
        "new_policy_scores_selections_weights_or_targets_generated": False,
        "new_financial_evaluations": 0,
        "new_source_requests": 0,
        "original_sources_or_accounts_replayed": False,
        "global_pool_changed": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if policy != expected or any(type(policy[k]) is not type(v) for k, v in expected.items()):
        raise QuantError("Freeze the complete saved intervention identity, not a new policy.")


def dependencies():
    return FunctionType(
        _dependencies.__code__,
        {**_dependencies.__globals__, **globals()},
        argdefs=_dependencies.__defaults__,
    )()


def inputs():
    policy = read_json(POLICY)
    validate(policy)
    catalogue = read_json(base.checked_reference(policy["input_catalogue"]))
    refs = [policy["input_catalogue"], *catalogue["source_references"]]
    for ref in refs:
        base.checked_reference(ref)
    closure = read_json(base.checked_reference(catalogue["new_financial_completion"]))
    if (
        closure["all_six_financial_verdicts_preserved"] != "rejected"
        or closure["new_complete_formal_paths"] != 24
        or closure["actual_authoritative_counts"]["current_journal_event_count"] != 1211
        or catalogue["new_diagnostic_outcomes_computed"] is not False
        or set(catalogue["variants"]) != {"close", "intraday", "overnight"}
    ):
        raise QuantError("Consume the exact completed three-pool cohort without replay.")
    return catalogue, refs


def progress(phase, **fields):
    counts = current_counts(base.journal_events(base.JOURNAL))
    if any(k in fields and fields[k] != v for k, v in counts.items()):
        raise QuantError("A zero-trial diagnosis cannot replace actual authoritative counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "hmm_session_intervention_" + phase,
            "study_id": read_json(POLICY)["study_id"],
            "active_compute_job": None,
            **counts,
            **fields,
            "new_financial_evaluations_in_this_study": 0,
            "investment_objective_verified": False,
            "order_authority": False,
        },
    )


def freeze():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the complete or partial first intervention freeze.")
    with lifecycle.research_job():
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        design = read_json(FOLDER / "design-decision-v1.json")
        if (
            design["actual_native_event_seq"] != 173394
            or design["new_diagnostic_outcomes_computed"]
        ):
            raise QuantError("Declare the three-channel question before actual output.")
        _, refs = inputs()
        validation = read_json(FOLDER / "validation-source-v1.json")
        xml, log = [base.checked_reference(r) for r in validation["test_outputs"]]
        suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
        if (
            validation["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
            or validation["exit_code"] != 0
            or not suites
            or sum(int(s.get("tests", 0)) for s in suites) != validation["passed_tests"]
            or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
        ):
            raise QuantError("Bind the exact passing synthetic Interface before actual diagnosis.")
        counts = current_counts(base.journal_events(base.JOURNAL))
        if counts["current_journal_event_count"] != 1211 or counts["new_pending_candidates"] != 0:
            raise QuantError("Keep the actual J1211/280 pre-diagnosis baseline.")
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "criteria": read_json(POLICY),
                "implementation_and_test_inputs": [reference(ROOT / p) for p in OWN],
                "complete_dependency_inputs": dependencies(),
                "source_references": refs,
                "validation_evidence": [
                    reference(xml),
                    reference(log),
                    reference(FOLDER / "validation-source-v1.json"),
                ],
                "design": reference(FOLDER / "design-decision-v1.json"),
                "invocation": reference(FOLDER / "actual-invocation-v1.json"),
                "preservation_baseline": reference(FOLDER / "preservation-baseline-v1.json"),
                "actual_counts": counts,
                "package_versions": sorted(
                    (d.metadata["Name"], d.version) for d in distributions()
                ),
                "new_diagnostic_outcomes_computed": False,
                "new_financial_evaluations": 0,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="one_streamed_saved_intervention_identity_pass")
    return reference(INTENT)


def _pairs(path):
    with path.open() as stream:
        while line := stream.readline():
            resolved = stream.readline()
            if not resolved:
                raise QuantError("An incomplete retained pair cannot be filled or discarded.")
            yield json.loads(line), json.loads(resolved)


def _operands(pair):
    before, after = pair
    return {
        "source_opinions_binary64_hex": [
            [float(x).hex() for x in o["target"]] for o in before["observations"]
        ],
        "rolling_weights_binary64_hex": [
            float(x).hex() for x in after["audit"]["rolling_expert_weights"]
        ],
        "combined_targets_binary64_hex": {
            name: [float(x).hex() for x in values]
            for name, values in after["combined_targets"].items()
        },
    }


def analyze():
    lifecycle.runtime_guard()
    if OUTPUT.exists() or (FOLDER / "analysis-started-v1.json").exists():
        raise QuantError("Preserve first complete or partial diagnostic output; never replay it.")
    with lifecycle.research_job():
        intent = read_json(INTENT)
        catalogue, _ = inputs()
        for ref in (
            intent["complete_dependency_inputs"]
            + intent["implementation_and_test_inputs"]
            + intent["validation_evidence"]
        ):
            base.checked_reference(ref)
        if intent["package_versions"] != [
            list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())
        ]:
            raise QuantError("The frozen source environment changed.")
        write_new_receipt(
            FOLDER / "analysis-started-v1.json", {"started_at": utc_now(), "pid": os.getpid()}
        )
        policy = read_json(POLICY)
        variants = policy["variants"]
        descriptors = [catalogue["base"], *[catalogue["variants"][v] for v in variants]]
        tapes = [_pairs(base.checked_reference(row["tape"])) for row in descriptors]
        aggregate = {
            variant: {
                "decisions": 0,
                "source_core_opinion_changed": 0,
                "mapped_selected_pair_changed": 0,
                "both_scored": 0,
                "both_unscored": 0,
                "score_states_differ": 0,
                "laws": {
                    law: {
                        "saved_target_changed": 0,
                        "noncore_saved_target_changed": 0,
                        "statistical_weights_changed": 0,
                        "same_pair_weights_changed": 0,
                        "maximum_saved_reconciliation_difference": 0.0,
                        "maximum_numpy_channel_difference": 0.0,
                        "channels": {
                            name: {
                                "sum_L1": 0.0,
                                "max_L1": 0.0,
                                "nonzero_decisions": 0,
                                "sum_noncore_L1": 0.0,
                            }
                            for name in policy["channels"]
                        },
                    }
                    for law in policy["rules"]
                },
            }
            for variant in variants
        }
        count = 0
        previous_weights = [None] * 4
        try:
            with (FOLDER / "actual-inputs-resolved-v1.jsonl").open("x") as stream:
                for index, pairs in enumerate(zip_longest(*tapes)):
                    lifecycle.runtime_guard()
                    if any(pair is None for pair in pairs):
                        raise QuantError("Retain all aligned dates of all four saved pools.")
                    for j, pair in enumerate(pairs):
                        if pair[0]["decision_index"] != index or (
                            previous_weights[j] is not None
                            and pair[0]["previous_weights"] != previous_weights[j]
                        ):
                            raise QuantError("Keep each recorded chronological learning state.")
                        previous_weights[j] = pair[1]["audit"]["rolling_expert_weights"]
                    for variant, pair in zip(variants, pairs[1:], strict=True):
                        if any(row.get("source_variant") != variant for row in pair):
                            raise QuantError("A sibling tape cannot be relabeled.")
                        spec = {
                            "base": descriptors[0],
                            "current": catalogue["variants"][variant],
                            "assets": catalogue["asset_order"],
                            "slot": policy["source_slot"],
                            "core_assets": policy["core_assets"],
                        }
                        stream.write(
                            json.dumps(
                                {
                                    "record": "intervention_input",
                                    "variant": variant,
                                    "decision_index": index,
                                    "source_tapes": [
                                        descriptors[0]["tape"],
                                        spec["current"]["tape"],
                                    ],
                                    "base_operands": _operands(pairs[0]),
                                    "changed_operands": _operands(pair),
                                },
                                allow_nan=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        stream.flush()
                        os.fsync(stream.fileno())
                        row = saved.inspect_intervention(pairs[0], pair, spec)
                        stream.write(
                            json.dumps(
                                {
                                    "record": "intervention_resolved",
                                    "variant": variant,
                                    **row,
                                },
                                allow_nan=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        summary = aggregate[variant]
                        summary["decisions"] += 1
                        summary["source_core_opinion_changed"] += not row[
                            "source_core_opinion_exactly_equal"
                        ]
                        summary["mapped_selected_pair_changed"] += not row[
                            "mapped_selected_pairs_equal"
                        ]
                        summary["both_scored"] += row["old_scored"] and row["new_scored"]
                        summary["both_unscored"] += not row["old_scored"] and not row["new_scored"]
                        summary["score_states_differ"] += row["old_scored"] != row["new_scored"]
                        for law, values in row["laws"].items():
                            result = summary["laws"][law]
                            result["saved_target_changed"] += not values[
                                "saved_targets_exactly_equal"
                            ]
                            result["noncore_saved_target_changed"] += not values[
                                "noncore_saved_targets_exactly_equal"
                            ]
                            result["statistical_weights_changed"] += not values[
                                "statistical_weights_exactly_equal"
                            ]
                            result["same_pair_weights_changed"] += (
                                row["mapped_selected_pairs_equal"]
                                and not values["statistical_weights_exactly_equal"]
                            )
                            for dest, key in (
                                (
                                    "maximum_saved_reconciliation_difference",
                                    "maximum_saved_target_reconciliation_difference",
                                ),
                                (
                                    "maximum_numpy_channel_difference",
                                    "maximum_numpy_channel_difference",
                                ),
                            ):
                                result[dest] = max(result[dest], values[key])
                            for name, channel in values["channels"].items():
                                entry = result["channels"][name]
                                entry["sum_L1"] = math.fsum((entry["sum_L1"], channel["L1"]))
                                entry["max_L1"] = max(entry["max_L1"], channel["L1"])
                                entry["nonzero_decisions"] += channel["L1"] != 0
                                entry["sum_noncore_L1"] = math.fsum(
                                    (entry["sum_noncore_L1"], channel["noncore_L1"])
                                )
                    count += 1
            if count != policy["all_original_decisions_per_pool"]:
                raise QuantError("A diagnosis must retain all original121 dates per pool.")
            counts = current_counts(base.journal_events(base.JOURNAL))
            if counts != intent["actual_counts"]:
                raise QuantError("The zero-trial diagnosis changed authoritative counts.")
            record = {
                "schema_version": 1,
                "study_id": policy["study_id"],
                "completed_at": utc_now(),
                "analysis_intent": reference(INTENT),
                "aggregate": aggregate,
                "all_original_decisions_per_pool": count,
                "resolved_variant_decisions": 3 * count,
                "both_original_rules_preserved": True,
                "exact_rational_identity_and_independent_numpy_checks_passed": True,
                "private_operands_and_channels": reference(
                    FOLDER / "actual-inputs-resolved-v1.jsonl"
                ),
                "actual_authoritative_counts": counts,
                "new_financial_evaluations": 0,
                "new_source_requests": 0,
                "source_models_selectors_weights_targets_accounts_or_benchmarks_replayed": False,
                "raw_source_rows_or_dated_opinions_exported": False,
                "causal_alpha_or_main_PnL_attributed": False,
                "global_pool_changed": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            }
            write_new_receipt(OUTPUT, record)
            write_new_receipt(PUBLIC, record)
            progress("complete_pending_publication", diagnostic=reference(OUTPUT))
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            write_new_receipt(
                FOLDER / "analysis-failure-v1.json", {"failed_at": utc_now(), "error": str(error)}
            )
            progress(
                "technical_failure_preserved", next_step="do_not_retry_or_retune_this_diagnosis"
            )
            raise
    return reference(OUTPUT)


def main():
    parser = argparse.ArgumentParser(description="Saved HMM intervention accounting only.")
    parser.add_argument("action", choices=("freeze", "analyze"))
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "analyze": analyze}[args.action]()))


if __name__ == "__main__":
    main()
