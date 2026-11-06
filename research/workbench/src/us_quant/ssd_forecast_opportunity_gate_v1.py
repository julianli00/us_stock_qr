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

from us_quant import forecast_opportunity_gate_v1 as rules
from us_quant import ssd_eight_expert_meta_v1 as template
from us_quant import ssd_expert_active_forecast_pair_v2 as advisors
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
FOLDER = base.REPORTS / "forecast-opportunity-gate-20261106-v1"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
PREPARED = FOLDER / "prepared-target-bundle-v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC = ROOT / "evidence/forecast_opportunity_gate_20261106_results_v1.json"
CANDIDATES, IDS = rules.CANDIDATES, rules.IDS
PACKETS = {
    name: ROOT / "evidence" / f"forecast_opportunity_gate_20261106_{i}_registration_v1.json"
    for i, name in enumerate(IDS)
}
OWN = (
    "config/forecast-opportunity-gate-v1.json",
    "src/us_quant/forecast_opportunity_gate_v1.py",
    "src/us_quant/ssd_forecast_opportunity_gate_v1.py",
    "tests/test_forecast_opportunity_gate_v1.py",
    "FORECAST_OPPORTUNITY_GATE_V1.md",
    "tests/test_forecast_risk_normalizer_diagnosis_v1.py",
    "tests/test_forecast_risk_normalizer_diagnosis_v2.py",
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
        raise QuantError("Participation cannot replace authoritative accounting or trial counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "forecast_opportunity_gate_" + phase,
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
    _, _, advisor_parent, _ = advisors.inputs()
    if (
        tuple(data.close.columns) != rules.ASSETS
        or tuple(e.identity.identifier for e in experts) != rules.EXPERT_IDS
    ):
        raise QuantError("Keep the full original eight experts and underlying asset coordinates.")
    for key in (
        "source_allocation_tape",
        "source_sign_certificate_tape",
        "source_sign_implementation_freeze",
        "source_sign_result",
        "source_sign_cycle",
        "source_advisor_results",
        "source_advisor_cycle",
    ):
        base.checked_reference(policy[key])
    certificate = read_json(base.checked_reference(policy["source_sign_result"]))
    sign_freeze = read_json(base.checked_reference(policy["source_sign_implementation_freeze"]))
    if (
        certificate["freeze"] != policy["source_sign_implementation_freeze"]
        or certificate["actual_input_resolved_records"] != policy["source_sign_certificate_tape"]
        or certificate["alternative_rankings_targets_or_financial_trials"] != 0
        or certificate["summary"]["source_scores_keys_or_target_records_rewritten"] is not False
    ):
        raise QuantError(
            "Consume the exact completed same-clock certificate, not a reconstructed winner."
        )
    old = read_json(template.INTENT)
    refs = unique_refs(
        [
            *old["source_references"],
            *old["complete_dependency_inputs"],
            *parent["code"],
            *parent["policies"],
            *parent["prior_evidence"],
            *parent["validation_evidence"],
            reference(template.INTENT),
            *advisor_parent["code"],
            *advisor_parent["policies"],
            *advisor_parent["prior_evidence"],
            *advisor_parent["validation_evidence"],
            reference(advisors.INTENT),
            reference(advisors.PREPARED),
            *sign_freeze["complete_dependency_inputs"],
            *sign_freeze["source_references"],
            *sign_freeze["validation_evidence"],
            *[
                policy[k]
                for k in (
                    "source_allocation_tape",
                    "source_sign_certificate_tape",
                    "source_sign_implementation_freeze",
                    "source_sign_result",
                    "source_sign_cycle",
                    "source_advisor_results",
                    "source_advisor_cycle",
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
            raise QuantError(
                "Both foreign cohort identities must be rejected before the first append."
            )
    return result


def target_proof(frozen, fields):
    if set(fields) & {
        "checked_at",
        "frozen_specification_sha256",
        "candidate_accounts_or_metrics_computed",
        "original_source_signal_recomputed",
    }:
        raise QuantError("Canonical proof fields cannot be silently overridden by a caller.")
    return {
        "checked_at": utc_now(),
        "frozen_specification_sha256": digest_json(frozen),
        **fields,
        "candidate_accounts_or_metrics_computed": False,
        "original_source_signal_recomputed": False,
        "source_ranking_forecaster_or_accounts_replayed": False,
        "new_Ridge_fits": 0,
        "new_source_requests": 0,
        "independent_forward_validation": False,
    }


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the complete or partial first opportunity-participation freeze.")
    with lifecycle.research_job():
        history = base.journal_events(base.JOURNAL)
        if len(history) != 1274 or history[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError("Freeze the new participation law against the actual J1274 baseline.")
        if any(e["candidate_id"] in IDS for e in history):
            raise QuantError("Both participation-program identities must be globally unused.")
        design = read_json(DESIGN)
        if (
            design["actual_native_event_seq"] != 177422
            or design["candidate_ids"] != list(IDS)
            or design["new_gated_targets_or_accounts_computed"] is not False
            or design["threshold_fitted_or_source_state_recomputed"] is not False
        ):
            raise QuantError("Declare the fixed zero-hurdle before any new participation outcomes.")
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, _, parent, refs, clocks, _ = source_inputs()
        validation = read_json(FOLDER / "validation-source-v1.json")
        xml, log = [base.checked_reference(r) for r in validation["test_outputs"]]
        suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
        if (
            validation["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
            or validation["exit_code"] != 0
            or not suites
            or validation["real_packet_entry_contract_exercised"] is not True
            or validation["passed_tests"] != sum(int(s.get("tests", 0)) for s in suites)
            or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
        ):
            raise QuantError(
                "Require exact gate/copy/standby/prefix and real proof-contract checks."
            )
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": read_json(rules.POLICY)["study_id"],
            "policy": reference(rules.POLICY),
            "expert_pool": reference(template.rules.POLICY),
            "construction": "feasibility_first_retained_positive_forecast_or_actual_BIL",
            "declared_financial_cohort_size": 2,
            "prior_journal_sequence": 1274,
            "prior_journal_head_sha256": rules.PRIOR_HEAD,
            "prior_completed_configurations": 285,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in [*deps, *refs] if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / "FORECAST_OPPORTUNITY_GATE_V1.md"),
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
            "retained_advisor_program_ids": list(rules.CONTROL_IDS),
            "source_advisor_state_recomputed_with_gate": False,
            "funding_standby_is_strategy_expert": False,
            "new_Ridge_fits": 0,
            "new_shadow_account_runs": 0,
            "new_source_requests": 0,
            "new_source_RF_views": 0,
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
                "new_gated_targets_or_accounts_computed": False,
                "threshold_fitted": False,
                "global_pool_changed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="register_both_before_either_new_gated_target")
    return reference(INTENT)


def _pairs(path):
    with path.open() as stream:
        while line := stream.readline():
            resolved = stream.readline()
            if not resolved:
                raise QuantError("No partial source/certificate record pair may be filled.")
            yield json.loads(line), json.loads(resolved)


def joined_records():
    policy = read_json(rules.POLICY)
    allocation = base.checked_reference(policy["source_allocation_tape"])
    certificates = base.checked_reference(policy["source_sign_certificate_tape"])
    for (before, after), (cb, cr) in zip(_pairs(allocation), _pairs(certificates), strict=True):
        yield {
            "record": "opportunity_gate_input",
            "allocation_before": before,
            "allocation_resolved": after,
            "certificate_before": cb,
            "certificate_resolved": cr,
        }


def prepare(data, experts, frozen):
    if PREPARED.exists():
        saved = read_json(PREPARED)
        if saved["frozen_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting prior gated intentions.")
        for entry in saved["entries"].values():
            for ref in entry.values():
                base.checked_reference(ref)
        return saved["entries"]
    if (FOLDER / "preparation-attempt-v1.json").exists():
        raise QuantError(
            "Preserve partial actual gate composition instead of automatically replaying."
        )
    if any(len(events("registered", n)) != 1 for n in IDS):
        raise QuantError("Both actual registrations precede either new gated target.")
    write_new_receipt(FOLDER / "preparation-attempt-v1.json", {"started_at": utc_now()})
    policy, dates = read_json(rules.POLICY), rules.decision_sessions(data.close.index)
    targets = {
        n: pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns) for n in IDS
    }
    audits = []
    log = FOLDER / "actual-gate-inputs-resolved-v1.jsonl"
    with log.open("x") as stream:

        def retain(row):
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        for i, row in enumerate(joined_records()):
            lifecycle.runtime_guard()
            source = row["allocation_resolved"]["audit"]
            if (
                i >= len(dates)
                or source["decision_index"] != i
                or source["decision_session"] != str(dates[i].date())
            ):
                raise QuantError("Keep every original date and joined input in causal order.")
            retain(row)
            current, audit = rules.gate(row, policy)
            rules.verify(row, current, audit, policy)
            for name in IDS:
                targets[name].loc[dates[i]] = current[name]
            audits.append(audit)
            retain(
                {
                    "record": "opportunity_gate_resolved",
                    "audit": audit,
                    "targets": {n: value.tolist() for n, value in current.items()},
                }
            )
    if len(audits) != len(dates) or len(audits) != 121:
        raise QuantError("Keep all121initial/month-end decisions and original formal return days.")
    prefixes = []
    for position in (0, 6, 12, 60, 120):
        lifecycle.runtime_guard()
        for i, row in enumerate(islice(joined_records(), position + 1)):
            early, audit = rules.gate(row, policy)
            if audit != audits[i] or any(
                not np.array_equal(early[n], targets[n].loc[dates[i]].to_numpy()) for n in IDS
            ):
                raise QuantError("Future records changed a prior participation intention.")
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
                    "Both horizons still require known funding anchors and real factor support."
                )
            coverage.append({"years": window["years"], "actual_factor_symbols": factors})
        directory = FOLDER / "targets" / name
        new_output_directory(directory)
        write_text_atomic(directory / "targets.csv", targets[name].to_csv(float_format="%.17g"))
        fields = {
            "decision_count": len(audits),
            "actual_input_resolved_record_count": 2 * len(audits),
            "actual_input_resolved_records": reference(log),
            "prefix_truncations": prefixes,
            "formal_factor_coverage": coverage,
            "participation_origins": sum(a["candidates"][name]["participation"] for a in audits),
            "funding_standby_origins": sum(
                not a["candidates"][name]["participation"] for a in audits
            ),
            "unscored_standby_origins": sum(
                not a["candidates"][name]["source_scores_available"] for a in audits
            ),
            "source_advisor_state_changed": False,
            "target_copy_or_BIL_identity_exact": True,
            "source_proxy_is_new_allocation_actual_cost_forecast": False,
        }
        write_new_receipt(directory / "proof.json", target_proof(frozen, fields))
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
            base.checked_reference(read_json(rules.POLICY)["source_advisor_results"])
        )["candidates"]
        candidates, comparisons = {}, {}
        for name, control in zip(IDS, rules.CONTROL_IDS, strict=True):
            result = read_json(base.checked_reference(saved["candidate_result_refs"][name]))
            if set(result["paths"]) != base.PATH_KEYS:
                raise QuantError("Publish every complete original path of both participation laws.")
            candidates[name] = {"status": result["review"]["status"], "paths": result["paths"]}
            comparisons[name] = {
                "matched_cached_advisor_id": control,
                "all_original_paths": {
                    key: {
                        "metrics": {
                            m: v["metrics"][m] - controls[control]["paths"][key]["metrics"][m]
                            for m in ("cagr", "sharpe", "max_drawdown")
                        },
                        "execution": {
                            m: v["execution"][m] - controls[control]["paths"][key]["execution"][m]
                            for m in (
                                "annualized_two_sided_turnover",
                                "model_order_count",
                                "total_cost_usd",
                            )
                        },
                    }
                    for key, v in result["paths"].items()
                },
                "old_control_replayed": False,
            }
        seven.common.retain_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "study_id": read_json(rules.POLICY)["study_id"],
                "completed_at": saved["completed_at"],
                "candidates": candidates,
                "new_minus_matched_cached_always_invested_advisor": comparisons,
                "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
                "source_advisor_state_recomputed_with_gate": False,
                "source_proxy_is_new_allocation_actual_cost_forecast": False,
                "funding_standby_is_strategy_expert": False,
                "old_controls_or_source_models_replayed": False,
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
        description="Feasibility-first retained forecast participation."
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
