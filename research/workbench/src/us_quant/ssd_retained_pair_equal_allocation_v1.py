from __future__ import annotations

import argparse
import json
import os
from importlib.metadata import distributions
from itertools import islice
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import retained_pair_equal_allocation_v1 as rules
from us_quant import ssd_eight_expert_meta_v1 as template
from us_quant.calendar import previous_session
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.ssd_style_activity_financial_v1 import dependencies as _dependencies
from us_quant.storage import (
    digest_json,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)

base, ROOT, seven, financial = template.base, template.ROOT, template.seven, template.financial
lifecycle, reference, unique_refs = template.lifecycle, template.reference, template.unique_refs
source = seven
FOLDER = base.REPORTS / "retained-pair-equal-allocation-20261105-v1"
POLICY = rules.POLICY
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
PREPARED = FOLDER / "prepared-target-bundle-v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC = ROOT / "evidence/retained_pair_equal_allocation_20261105_results_v1.json"
CANDIDATES, IDS = rules.CANDIDATES, (rules.ID,)
PACKETS = {rules.ID: ROOT / "evidence/retained_pair_equal_allocation_20261105_registration_v1.json"}
OWN = (
    "config/retained-pair-equal-allocation-v1.json",
    "src/us_quant/retained_pair_equal_allocation_v1.py",
    "src/us_quant/ssd_retained_pair_equal_allocation_v1.py",
    "tests/test_retained_pair_equal_allocation_v1.py",
    "RETAINED_PAIR_EQUAL_ALLOCATION_V1.md",
    "tests/conftest.py",
)


def bound(function, **overrides):
    return FunctionType(
        function.__code__,
        {**function.__globals__, **globals(), **overrides},
        argdefs=function.__defaults__,
    )


def events(kind, identifier):
    return bound(template.events)(kind, identifier)


def dependencies():
    return bound(_dependencies)()


def check_frozen(frozen):
    return template.check_frozen(frozen)


def progress(phase, **fields):
    counts = current_counts(base.journal_events(base.JOURNAL))
    if any(k in fields and fields[k] != v for k, v in counts.items()):
        raise QuantError("The study cannot replace actual authoritative financial counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "retained_pair_equal_" + phase,
            "study_id": read_json(POLICY)["study_id"],
            "active_compute_job": None,
            **counts,
            **fields,
            "order_authority": False,
        },
    )


def catalogue():
    policy = read_json(POLICY)
    rules.validate(policy)
    saved = read_json(base.checked_reference(policy["source_catalogue"]))
    if (
        saved["expert_ids"] != list(rules.EXPERT_IDS)
        or saved["mechanism_groups"] != list(rules.MECHANISMS)
        or saved["asset_order"] != list(rules.ASSETS)
        or saved["source_candidate_ids"] != list(template.IDS)
        or saved["advisor_candidate_id"] != policy["advisor_candidate_id"]
        or saved["new_allocation_targets_or_accounts_generated"] is not False
    ):
        raise QuantError("Keep canonical-eight membership and the original fixed advisor.")
    for ref in saved["source_references"]:
        base.checked_reference(ref)
    return saved


def source_inputs():
    saved = catalogue()
    data, experts, parent, benchmarks = template.inputs()
    if (
        tuple(data.close.columns) != rules.ASSETS
        or tuple(e.identity.identifier for e in experts) != rules.EXPERT_IDS
        or parent["source_policy_and_expert_admission_clocks"] != saved["source_clocks"]
    ):
        raise QuantError("Original source availability, universe or native RF context changed.")
    original = read_json(template.INTENT)
    refs = unique_refs(
        [
            *original["source_references"],
            *original["complete_dependency_inputs"],
            *parent["code"],
            *parent["policies"],
            *parent["prior_evidence"],
            *parent["validation_evidence"],
            *saved["source_references"],
            read_json(POLICY)["source_catalogue"],
            reference(template.INTENT),
        ]
    )
    return data, experts, parent, refs, saved["source_clocks"], benchmarks


def inputs():
    result = bound(template.inputs)()
    frozen = result[2]
    existing = events("registered", rules.ID)
    if existing and (
        len(existing) != 1 or existing[0]["body"] != {"candidate": CANDIDATES[0], "frozen": frozen}
    ):
        raise QuantError("A foreign or changed identity cannot be appended or rescued.")
    return result


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the complete or partial first allocation freeze.")
    with lifecycle.research_job():
        if any(e["candidate_id"] == rules.ID for e in base.journal_events(base.JOURNAL)):
            raise QuantError("The new allocation identity must be globally unused.")
        design = read_json(DESIGN)
        if (
            design["actual_native_event_seq"] != 174043
            or design["new_allocation_targets_or_accounts_generated"] is not False
            or design["candidate_id"] != rules.ID
        ):
            raise QuantError("Declare the symmetric allocation before any actual new target.")
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, _, parent, refs, clocks, _ = source_inputs()
        history = base.journal_events(base.JOURNAL)
        if len(history) != 1211 or history[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError("Freeze against the actual closed J1211/280 baseline.")
        validation = read_json(FOLDER / "validation-source-v1.json")
        xml, log = [base.checked_reference(ref) for ref in validation["test_outputs"]]
        suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
        if (
            validation["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
            or validation["exit_code"] != 0
            or not suites
            or sum(int(s.get("tests", 0)) for s in suites) != validation["passed_tests"]
            or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
        ):
            raise QuantError("Bind the exact causal allocation and funding-fence checks.")
        deps = dependencies()
        advice = catalogue()
        frozen = {
            **parent,
            "study_id": read_json(POLICY)["study_id"],
            "policy": reference(POLICY),
            "expert_pool": reference(template.rules.POLICY),
            "construction": "retained_original_eight_pair_advisor_symmetric_allocation",
            "declared_financial_cohort_size": 1,
            "prior_journal_sequence": 1211,
            "prior_journal_head_sha256": rules.PRIOR_HEAD,
            "prior_completed_configurations": 280,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in deps if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / "RETAINED_PAIR_EQUAL_ALLOCATION_V1.md"),
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
                    *refs,
                    reference(DESIGN),
                    reference(INVOCATION),
                ]
            ),
            "validation_evidence": unique_refs(
                [
                    *parent["validation_evidence"],
                    reference(xml),
                    reference(log),
                    reference(FOLDER / "validation-source-v1.json"),
                ]
            ),
            "source_policy_and_expert_admission_clocks": clocks,
            "retained_pair_advisor": {
                "candidate_id": advice["advisor_candidate_id"],
                "actual_registered_at": advice["advisor_actual_registered_at"],
                "source_tape": advice["source_tape"],
                "original_statistical_state_preserved": True,
                "proxy_is_new_allocation_actual_cost": False,
                "is_an_additional_strategy_expert": False,
            },
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
                "new_allocation_targets_or_accounts_generated": False,
                "global_pool_changed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="register_before_first_actual_new_allocation_target")
    return reference(INTENT)


def _pairs(path):
    with path.open() as stream:
        while line := stream.readline():
            resolved = stream.readline()
            if not resolved:
                raise QuantError("An incomplete source advisory record cannot be filled.")
            yield json.loads(line), json.loads(resolved)


def prepare(data, experts, frozen):
    if PREPARED.exists():
        saved = read_json(PREPARED)
        if saved["frozen_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting prior allocation output.")
        for entry in saved["entries"].values():
            for ref in entry.values():
                base.checked_reference(ref)
        return saved["entries"]
    if (FOLDER / "preparation-attempt-v1.json").exists():
        raise QuantError("Preserve incomplete allocation instead of automatically repeating it.")
    if len(events("registered", rules.ID)) != 1:
        raise QuantError("Actual registration must precede the first new allocation target.")
    write_new_receipt(FOLDER / "preparation-attempt-v1.json", {"started_at": utc_now()})
    saved, policy = catalogue(), read_json(POLICY)
    tape = base.checked_reference(saved["source_tape"])
    dates = rules.decision_sessions(data.close.index)
    targets = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    audits, previous_source = [], None
    count_values, maximum = 0, 0.0
    log = FOLDER / "actual-allocation-inputs-resolved-v1.jsonl"
    with log.open("x") as stream:
        for index, (before, resolved) in enumerate(_pairs(tape)):
            lifecycle.runtime_guard()
            if (
                index >= len(dates)
                or before["decision_index"] != index
                or before["as_of"] != str(dates[index].date())
                or (previous_source is not None and before["previous_weights"] != previous_source)
            ):
                raise QuantError(
                    "Retain every source date and its own chronological learning state."
                )
            for observation in before["observations"]:
                known = pd.DatetimeIndex(observation["return_dates"])
                expected = data.risk_free.loc[known].to_numpy(dtype=float)
                if not np.array_equal(np.asarray(observation["RF"], dtype=float), expected):
                    raise QuantError(
                        "Saved source RF must exactly equal the original canonical RF."
                    )
            stream.write(
                json.dumps(
                    {
                        "record": "allocation_input",
                        "source_tape": saved["source_tape"],
                        "source_before": before,
                        "source_resolved": resolved,
                    },
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
            target, audit = rules.allocate_retained_pair(before, resolved, policy)
            targets.loc[dates[index]] = target
            audits.append(audit)
            count_values += audit["NET_RF_binary64_values_validated"]
            maximum = max(maximum, audit["maximum_independent_rational_target_difference"])
            previous_source = resolved["audit"]["rolling_expert_weights"]
            stream.write(
                json.dumps(
                    {
                        "record": "allocation_resolved",
                        "audit": audit,
                        "target": target.tolist(),
                    },
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
    if len(audits) != 121 or len(dates) != 121:
        raise QuantError("Keep all original121capital/month-end decisions.")
    prefixes = []
    for position in (0, 6, 12, 60, 120):
        lifecycle.runtime_guard()
        for index, (before, resolved) in enumerate(islice(_pairs(tape), position + 1)):
            early, audit = rules.allocate_retained_pair(before, resolved, policy)
            if (
                not np.array_equal(early, targets.loc[dates[index]].to_numpy())
                or audit != audits[index]
            ):
                raise QuantError("A future source record changed an earlier new allocation.")
        prefixes.append(str(dates[position].date()))
    coverage = []
    active = targets.dropna(how="all")
    for window in read_json(base.PROTOCOL)["windows"]:
        anchor = previous_session(window["start"])
        known = active.loc[:anchor].tail(1)
        rows = pd.concat([known, active.loc[anchor + pd.Timedelta(days=1) : window["end"]]])
        factors = [
            symbol
            for symbol in ("MTUM", "VLUE", "QUAL", "USMV", "IJR", "PKW")
            if (rows[symbol] > 1e-10).any()
        ]
        if known.empty or len(factors) < 3:
            raise QuantError(
                "Each formal horizon needs its known anchor and genuine factor support."
            )
        coverage.append({"years": window["years"], "actual_factor_symbols": factors})
    directory = FOLDER / "targets" / rules.ID
    new_output_directory(directory)
    write_text_atomic(directory / "targets.csv", targets.to_csv(float_format="%.17g"))
    write_new_receipt(
        directory / "proof.json",
        {
            "checked_at": utc_now(),
            "frozen_specification_sha256": digest_json(frozen),
            "decision_count": len(audits),
            "source_scored_dates": sum(a["source_scores_available"] for a in audits),
            "source_unscored_dates": sum(not a["source_scores_available"] for a in audits),
            "actual_input_resolved_record_count": 2 * len(audits),
            "actual_input_resolved_records": reference(log),
            "NET_RF_binary64_values_validated": count_values,
            "maximum_independent_rational_target_difference": maximum,
            "prefix_truncations": prefixes,
            "formal_factor_coverage": coverage,
            "source_advisor_state_recomputed": False,
            "source_selector_replayed": False,
            "source_proxy_is_new_allocation_actual_cost": False,
            "candidate_accounts_or_metrics_computed": False,
            "original_source_signal_recomputed": False,
            "new_source_or_shadow_account_runs": 0,
            "independent_forward_validation": False,
        },
    )
    entry = {
        "targets": reference(directory / "targets.csv"),
        "proof": reference(directory / "proof.json"),
    }
    write_new_receipt(
        PREPARED, {"frozen_sha256": digest_json(frozen), "entries": {rules.ID: entry}}
    )
    return {rules.ID: entry}


def register():
    packets = bound(template.register)()
    progress("registered_before_accounts", next_step="publish_exact_single_packet_before_funding")
    return packets


def scoped(function, candidate):
    return bound(template.scoped)(function, candidate)


def cohort_publication(*args):
    return bound(template.cohort_publication)(*args)


def evaluate(commit):
    lifecycle.runtime_guard()
    if RESULTS.exists():
        for ref in read_json(RESULTS)["candidate_result_refs"].values():
            base.checked_reference(ref)
        return reference(RESULTS)
    result = scoped(financial.evaluate, CANDIDATES[0])(commit)
    write_new_receipt(
        RESULTS,
        {
            "completed_at": utc_now(),
            "candidate_result_refs": {rules.ID: result},
            "all_four_original_paths_complete": True,
            "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
            "source_advisor_state_changed": False,
            "global_pool_changed": False,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "order_authority": False,
        },
    )
    progress("four_paths_complete", next_step="publish_every_path_and_cached_controls")
    return reference(RESULTS)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(RESULTS)
        result = read_json(base.checked_reference(saved["candidate_result_refs"][rules.ID]))
        controls = read_json(base.checked_reference(catalogue()["source_public_results"]))[
            "candidates"
        ]
        if set(result["paths"]) != base.PATH_KEYS:
            raise QuantError("Publish every complete original path, not passing fragments.")
        comparisons = {
            name: {
                key: {
                    "metrics": {
                        metric: value["metrics"][metric]
                        - controls[name]["paths"][key]["metrics"][metric]
                        for metric in ("cagr", "sharpe", "max_drawdown")
                    },
                    "execution": {
                        metric: value["execution"][metric]
                        - controls[name]["paths"][key]["execution"][metric]
                        for metric in (
                            "annualized_two_sided_turnover",
                            "model_order_count",
                            "total_cost_usd",
                        )
                    },
                }
                for key, value in result["paths"].items()
            }
            for name in template.IDS
        }
        seven.common.retain_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "study_id": read_json(POLICY)["study_id"],
                "completed_at": saved["completed_at"],
                "candidates": {
                    rules.ID: {"status": result["review"]["status"], "paths": result["paths"]}
                },
                "new_minus_cached_control": comparisons,
                "old_controls_replayed": False,
                "whole_policy_difference_is_isolated_causal_fee_effect": False,
                "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
                "source_advisor_statistical_state_changed": False,
                "source_selector_or_shadow_replayed": False,
                "new_source_requests": 0,
                "raw_prices_targets_or_daily_rows_exported": False,
                "global_pool_changed": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
    return reference(PUBLIC)


def main():
    parser = argparse.ArgumentParser(
        description="Symmetric allocation from the original causal pair advisor."
    )
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use an actual remotely published complete registration packet.")
    actions = {"freeze": initialize, "register": register, "export": export}
    print(
        json.dumps(evaluate(args.commit) if args.action == "evaluate" else actions[args.action]())
    )


if __name__ == "__main__":
    main()
