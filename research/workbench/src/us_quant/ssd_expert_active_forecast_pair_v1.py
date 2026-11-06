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

from us_quant import expert_active_forecast_pair_v1 as rules
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
FOLDER = base.REPORTS / "expert-active-forecast-pair-20261106-v1"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
PREPARED = FOLDER / "prepared-target-bundle-v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC = ROOT / "evidence/expert_active_forecast_pair_20261106_results_v1.json"
CANDIDATES, IDS = rules.CANDIDATES, rules.IDS
PACKETS = {
    name: ROOT / "evidence" / f"expert_active_forecast_pair_20261106_{i}_registration_v1.json"
    for i, name in enumerate(IDS)
}
OWN = (
    "config/expert-active-forecast-pair-v1.json",
    "src/us_quant/expert_active_forecast_pair_v1.py",
    "src/us_quant/ssd_expert_active_forecast_pair_v1.py",
    "tests/test_expert_active_forecast_pair_v1.py",
    "EXPERT_ACTIVE_FORECAST_PAIR_V1.md",
    "tests/test_strategy_expert_pool_eight_v1.py",
    "tests/test_strategy_expert_pool_v5.py",
    "tests/test_strategy_expert_pool_v4.py",
    "tests/test_strategy_expert_pool_v3.py",
    "tests/conftest.py",
)


def bound(function, **overrides):
    return FunctionType(
        function.__code__,
        {**function.__globals__, **globals(), **overrides},
        argdefs=function.__defaults__,
    )


def dependencies():
    return bound(_dependencies)()


def events(kind, identifier):
    return bound(template.events)(kind, identifier)


def check_frozen(frozen):
    return template.check_frozen(frozen)


def progress(phase, **fields):
    counts = current_counts(base.journal_events(base.JOURNAL))
    if any(k in fields and fields[k] != v for k, v in counts.items()):
        raise QuantError("Derive progress from actual history, not planned account totals.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "active_forecast_pair_" + phase,
            "study_id": read_json(rules.POLICY)["study_id"],
            "active_compute_job": None,
            **counts,
            **fields,
            "order_authority": False,
        },
    )


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    data, experts, parent, benchmarks = template.inputs()
    if (
        tuple(data.close.columns) != rules.ASSETS
        or tuple(e.identity.identifier for e in experts) != rules.EXPERT_IDS
    ):
        raise QuantError("Keep the full original eight-source/fourteen-asset Interface.")
    prior = read_json(template.INTENT)
    forecast = read_json(base.checked_reference(policy["source_forecast_freeze"]))
    result = read_json(base.checked_reference(policy["source_forecast_result"]))
    closure = read_json(base.checked_reference(policy["source_forecast_completion"]))
    if (
        result["freeze"] != policy["source_forecast_freeze"]
        or result["actual_inputs_resolved"] != policy["source_forecast_tape"]
        or result["new_allocations_or_financial_trials"] != 0
        or closure["no_model_or_expert_promoted"] is not True
        or forecast["source_policy_and_expert_admission_clocks"]
        != parent["source_policy_and_expert_admission_clocks"]
    ):
        raise QuantError("Consume the exact completed causal forecasts without promoting a source.")
    refs = unique_refs(
        [
            *prior["source_references"],
            *prior["complete_dependency_inputs"],
            *parent["code"],
            *parent["policies"],
            *parent["prior_evidence"],
            *parent["validation_evidence"],
            reference(template.INTENT),
            *forecast["complete_dependency_inputs"],
            *forecast["source_references"],
            *forecast["validation_evidence"],
            *[
                policy[k]
                for k in (
                    "source_risk_tape",
                    "source_forecast_tape",
                    "source_forecast_freeze",
                    "source_forecast_result",
                    "source_forecast_completion",
                    "cached_original_eight_results",
                )
            ],
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    return (
        data,
        experts,
        parent,
        refs,
        parent["source_policy_and_expert_admission_clocks"],
        benchmarks,
    )


def inputs():
    result = bound(template.inputs)()
    frozen = result[2]
    for candidate in CANDIDATES:
        existing = events("registered", candidate["id"])
        if existing and (
            len(existing) != 1 or existing[0]["body"] != {"candidate": candidate, "frozen": frozen}
        ):
            raise QuantError("Reject either foreign cohort identity before the first append.")
    return result


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the complete or partial first forecast-allocation freeze.")
    with lifecycle.research_job():
        if any(e["candidate_id"] in IDS for e in base.journal_events(base.JOURNAL)):
            raise QuantError("Both new allocation identities must be globally unused.")
        design = read_json(DESIGN)
        if (
            design["actual_native_event_seq"] != 175938
            or design["new_allocations_or_accounts_computed"] is not False
            or design["candidate_ids"] != list(IDS)
            or design["new_forecast_fits"] != 0
        ):
            raise QuantError("Freeze both models and the new allocation before any actual outcome.")
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, _, parent, refs, clocks, _ = source_inputs()
        history = base.journal_events(base.JOURNAL)
        if len(history) != 1246 or history[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError("Freeze against actual J1246/283/0, without resetting older history.")
        validation = read_json(FOLDER / "validation-source-v1.json")
        xml, log = [base.checked_reference(ref) for ref in validation["test_outputs"]]
        suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
        if (
            validation["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
            or validation["exit_code"] != 0
            or not suites
            or validation["passed_tests"] != sum(int(s.get("tests", 0)) for s in suites)
            or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
        ):
            raise QuantError("Bind the exact join, independent score, prefix and funding checks.")
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": read_json(rules.POLICY)["study_id"],
            "policy": reference(rules.POLICY),
            "expert_pool": reference(template.rules.POLICY),
            "construction": "saved_unconditional_contextual_amount_forecast_original_pair_risk",
            "declared_financial_cohort_size": 2,
            "prior_journal_sequence": 1246,
            "prior_journal_head_sha256": rules.PRIOR_HEAD,
            "prior_completed_configurations": 283,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in [*deps, *refs] if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / "EXPERT_ACTIVE_FORECAST_PAIR_V1.md"),
                ]
            ),
            "policies": unique_refs(
                [
                    *parent["policies"],
                    *[r for r in [*deps, *refs] if r["path"].startswith("config/")],
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
            "saved_forecast_tape": read_json(rules.POLICY)["source_forecast_tape"],
            "source_risk_catalogue": read_json(rules.POLICY)["source_risk_tape"],
            "new_Ridge_fits": 0,
            "source_forecaster_is_a_strategy_expert": False,
            "new_shadow_account_runs": 0,
            "new_source_requests": 0,
            "new_source_RF_views": 0,
            "source_assessment_RF_views_reused_unchanged": True,
            "source_native_RF_identity_claimed_without_view": False,
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
                "new_allocations_or_accounts_computed": False,
                "new_forecast_fits": 0,
                "global_pool_changed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="register_both_before_either_new_allocation")
    return reference(INTENT)


def _pairs(path):
    with path.open() as stream:
        while line := stream.readline():
            resolved = stream.readline()
            if not resolved:
                raise QuantError("A partial saved input/result pair cannot be filled.")
            yield json.loads(line), json.loads(resolved)


def joined_records():
    policy = read_json(rules.POLICY)
    risk = base.checked_reference(policy["source_risk_tape"])
    forecast = base.checked_reference(policy["source_forecast_tape"])
    for (before, after), (fb, fr) in zip(_pairs(risk), _pairs(forecast), strict=True):
        yield {
            "record": "saved_forecast_allocation_input",
            "source_before": before,
            "source_resolved": after,
            "forecast_before": fb,
            "forecast_resolved": fr,
        }


def prepare(data, experts, frozen):
    if PREPARED.exists():
        saved = read_json(PREPARED)
        if saved["frozen_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting prior forecast-allocation targets.")
        for entry in saved["entries"].values():
            for ref in entry.values():
                base.checked_reference(ref)
        return saved["entries"]
    if (FOLDER / "preparation-attempt-v1.json").exists():
        raise QuantError("Preserve partial allocation rather than replaying this version.")
    if any(len(events("registered", n)) != 1 for n in IDS):
        raise QuantError("Both actual registrations must precede either actual allocation.")
    write_new_receipt(FOLDER / "preparation-attempt-v1.json", {"started_at": utc_now()})
    policy, dates = read_json(rules.POLICY), rules.decision_sessions(data.close.index)
    targets = {
        n: pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns) for n in IDS
    }
    previous = {n: np.full(8, 0.125) for n in IDS}
    audits, values = [], 0
    maxima = {}
    log = FOLDER / "actual-allocation-inputs-resolved-v1.jsonl"
    with log.open("x") as stream:

        def retain(row):
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        for i, row in enumerate(joined_records()):
            lifecycle.runtime_guard()
            before = row["source_before"]
            if (
                i >= len(dates)
                or before["decision_index"] != i
                or before["as_of"] != str(dates[i].date())
            ):
                raise QuantError("Every original date and source record must remain ordered.")
            row["previous_new_statistical_weights"] = {n: v.tolist() for n, v in previous.items()}
            retain(row)
            for o, e in zip(before["observations"], experts, strict=True):
                source_observation = e.observe(dates[i], 126)
                if (
                    source_observation.target is None
                    or o["NET_returns"] != source_observation.net_returns.tolist()
                    or o["RF"] != source_observation.risk_free.tolist()
                    or o["target"] != source_observation.target.tolist()
                    or o["RF"] != data.risk_free.loc[source_observation.risk_free.index].tolist()
                ):
                    raise QuantError(
                        "Retained forecasts and opinions need the exact current frozen source."
                    )
            current, audit = rules.allocate(row, previous, policy)
            checked = rules.verify(row, previous, current, audit, policy)
            for key, gap in checked.items():
                maxima[key] = max(maxima.get(key, 0.0), gap)
            for name in IDS:
                targets[name].loc[dates[i]] = current[name]
                previous[name] = np.asarray(audit["candidates"][name]["expert_weights"])
            audits.append(audit)
            values += audit["source_NET_RF_binary64_values_checked"]
            retain(
                {
                    "record": "saved_forecast_allocation_resolved",
                    "audit": audit,
                    "targets": {n: v.tolist() for n, v in current.items()},
                }
            )
    if len(audits) != len(dates) or len(audits) != 121:
        raise QuantError("Keep every original121capital/month-end decision.")
    prefixes = []
    for position in (0, 6, 12, 60, 120):
        lifecycle.runtime_guard()
        state = {n: np.full(8, 0.125) for n in IDS}
        for i, row in enumerate(islice(joined_records(), position + 1)):
            current, audit = rules.allocate(row, state, policy)
            if audit != audits[i] or any(
                not np.array_equal(current[n], targets[n].loc[dates[i]].to_numpy()) for n in IDS
            ):
                raise QuantError(
                    "Future records changed a prior new allocation or its learning state."
                )
            state = {n: np.asarray(audit["candidates"][n]["expert_weights"]) for n in IDS}
        prefixes.append(str(dates[position].date()))
    write_new_receipt(FOLDER / "information-audits-v1.json", {"decisions": audits})
    entries = {}
    for name in IDS:
        active = targets[name].dropna(how="all")
        coverage = []
        for window in read_json(base.PROTOCOL)["windows"]:
            anchor = previous_session(window["start"])
            known = active.loc[:anchor].tail(1)
            rows = pd.concat([known, active.loc[anchor + pd.Timedelta(days=1) : window["end"]]])
            factors = [
                s for s in ("MTUM", "VLUE", "QUAL", "USMV", "IJR", "PKW") if (rows[s] > 1e-10).any()
            ]
            if known.empty or len(factors) < 3:
                raise QuantError(
                    "Each horizon still requires a known anchor and genuine factor support."
                )
            coverage.append({"years": window["years"], "actual_factor_symbols": factors})
        directory = FOLDER / "targets" / name
        new_output_directory(directory)
        write_text_atomic(directory / "targets.csv", targets[name].to_csv(float_format="%.17g"))
        write_new_receipt(
            directory / "proof.json",
            {
                "checked_at": utc_now(),
                "frozen_specification_sha256": digest_json(frozen),
                **maxima,
                "decision_count": len(audits),
                "actual_input_resolved_record_count": 2 * len(audits),
                "source_NET_RF_binary64_values_checked": values,
                "all_pairs_per_mature_decision": 28,
                "prefix_truncations": prefixes,
                "formal_factor_coverage": coverage,
                "actual_input_resolved_records": reference(log),
                "candidate_accounts_or_metrics_computed": False,
                "source_models_selectors_forecasters_or_accounts_replayed": False,
                "new_Ridge_fits": 0,
                "new_source_requests": 0,
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
    return bound(template.register)()


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
            base.checked_reference(read_json(rules.POLICY)["cached_original_eight_results"])
        )["candidates"]
        candidates, comparisons = {}, {}
        for name in IDS:
            result = read_json(base.checked_reference(saved["candidate_result_refs"][name]))
            if set(result["paths"]) != base.PATH_KEYS:
                raise QuantError("Publish every complete funded path of both new rules.")
            candidates[name] = {"status": result["review"]["status"], "paths": result["paths"]}
            comparisons[name] = {
                control: {
                    key: {
                        "metrics": {
                            m: value["metrics"][m] - old["paths"][key]["metrics"][m]
                            for m in ("cagr", "sharpe", "max_drawdown")
                        },
                        "execution": {
                            m: value["execution"][m] - old["paths"][key]["execution"][m]
                            for m in (
                                "annualized_two_sided_turnover",
                                "model_order_count",
                                "total_cost_usd",
                            )
                        },
                    }
                    for key, value in result["paths"].items()
                }
                for control, old in controls.items()
            }
        seven.common.retain_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "study_id": read_json(rules.POLICY)["study_id"],
                "completed_at": saved["completed_at"],
                "candidates": candidates,
                "new_minus_both_cached_original_eight_controls": comparisons,
                "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
                "all_models_and_original_sources_retained": True,
                "old_controls_replayed": False,
                "source_models_selectors_forecasters_or_accounts_replayed": False,
                "new_Ridge_fits": 0,
                "new_source_requests": 0,
                "new_source_RF_views": 0,
                "raw_prices_targets_or_daily_rows_exported": False,
                "global_pool_changed": False,
                "whole_policy_difference_is_isolated_source_alpha": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
    return reference(PUBLIC)


def main():
    parser = argparse.ArgumentParser(
        description="Both saved forecasters, separate original main books."
    )
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use an actual remotely published complete two-packet cohort.")
    actions = {"freeze": initialize, "register": register, "export": export}
    print(
        json.dumps(evaluate(args.commit) if args.action == "evaluate" else actions[args.action]())
    )


if __name__ == "__main__":
    main()
