from __future__ import annotations

import argparse
import json
import os
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import equal_role_projection_v1 as rules
from us_quant import ssd_protective_six_adaptive_hedge_v1 as source
from us_quant.candidate_receipts import write_new_receipt
from us_quant.ssd_forecast_opportunity_gate_v1 import target_proof
from us_quant.ssd_style_activity_financial_v1 import dependencies as _dependencies
from us_quant.storage import (
    digest_json,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)

base, ROOT, template, seven, financial = (
    source.base,
    source.ROOT,
    source.template,
    source.seven,
    source.financial,
)
lifecycle, reference, unique_refs = source.lifecycle, source.reference, source.unique_refs
current_counts = source.current_counts
FOLDER = base.REPORTS / "equal-role-projection-20261106-v1"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
PREPARED, RESULTS = FOLDER / "prepared-target-bundle-v1.json", FOLDER / "results.json"
PUBLIC = ROOT / "evidence/equal_role_projection_20261106_results_v1.json"
CANDIDATES, IDS = rules.CANDIDATES, (rules.ID,)
PACKETS = {rules.ID: ROOT / "evidence/equal_role_projection_20261106_registration_v1.json"}
OWN = (
    "config/equal-role-projection-v1.json",
    "src/us_quant/equal_role_projection_v1.py",
    "src/us_quant/ssd_equal_role_projection_v1.py",
    "tests/test_equal_role_projection_v1.py",
    "tests/conftest.py",
    "EQUAL_ROLE_PROJECTION_V1.md",
)


def bound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


def dependencies():
    return bound(_dependencies)()


def events(kind, identifier):
    return bound(template.events)(kind, identifier)


def check_frozen(frozen):
    return template.check_frozen(frozen)


def progress(phase, **fields):
    actual = current_counts(base.journal_events(base.JOURNAL))
    rules.require(
        all(key not in fields or fields[key] == value for key, value in actual.items()),
        "Keep actual financial counts instead of stale source snapshots.",
    )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "equal_role_projection_" + phase,
            "study_id": read_json(rules.POLICY)["study_id"],
            "active_compute_job": None,
            **actual,
            **fields,
            "order_authority": False,
        },
    )


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    inputs = read_json(base.checked_reference(policy["source_inputs"]))
    data, experts, _, refs, clocks, benchmarks = source.source_inputs()
    parent = source.events("registered", source.rules.ID)[0]["body"]["frozen"]
    closed = read_json(base.checked_reference(inputs["source_completed_cycle"]))
    rules.require(
        inputs["expert_ids"] == list(rules.EXPERT_IDS)
        and inputs["growth_positions"] == list(rules.GROUPS[0])
        and inputs["response_positions"] == list(rules.GROUPS[1])
        and inputs["new_projected_weights_targets_or_accounts_computed"] is False,
        "Use one controlled role allocation on the same six admitted complete sources.",
    )
    refs = unique_refs(
        [
            *refs,
            *closed["exact_evidence_references"],
            policy["source_inputs"],
            *[
                value
                for value in inputs.values()
                if isinstance(value, dict) and "path" in value and "sha256" in value
            ],
            *[
                reference(FOLDER / name)
                for name in (
                    "capture-preflight-v1.py",
                    "actual-invocation-v1.json",
                    "design-decision-v1.json",
                    "preservation-baseline-v1.json",
                )
            ],
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    return data, experts, parent, refs, clocks, benchmarks


def inputs():
    return bound(template.inputs)()


def initialize():
    lifecycle.runtime_guard()
    rules.require(
        not INTENT.exists() and not (FOLDER / "freeze-attempt-v1.json").exists(),
        "Preserve the first complete or partial role-study freeze.",
    )
    with lifecycle.research_job():
        history = base.journal_events(base.JOURNAL)
        rules.require(
            len(history) == 1464 and not any(e["candidate_id"] in IDS for e in history),
            "Freeze one unused candidate at the actual J1464 baseline.",
        )
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, _, parent, refs, clocks, _ = source_inputs()
        validation = read_json(FOLDER / "validation-source-v1.json")
        xml, log = [base.checked_reference(ref) for ref in validation["test_outputs"]]
        suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
        rules.require(
            validation["implementation_and_test_inputs"] == [reference(ROOT / p) for p in OWN]
            and validation["exit_code"] == 0
            and suites
            and validation["passed_tests"] == sum(int(s.get("tests", 0)) for s in suites)
            and not any(
                int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
            ),
            "Bind successful exact half-role,ratio,invalid-input and prefix checks.",
        )
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": read_json(rules.POLICY)["study_id"],
            "policy": reference(rules.POLICY),
            "expert_pool": reference(rules.POLICY),
            "construction": "cached_six_AdaHedge_equal_economic_role_capital_projection",
            "expert_pool_definition_role": "same_six_complete_sources_fixed_economic_roles",
            "declared_financial_cohort_size": 1,
            "prior_journal_sequence": 1464,
            "prior_journal_head_sha256": read_json(rules.POLICY)["prior_journal_head_sha256"],
            "prior_completed_configurations": 301,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in [*deps, *refs] if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / OWN[-1]),
                ]
            ),
            "policies": unique_refs(
                [
                    *parent["policies"],
                    *[r for r in [*deps, *refs] if r["path"].startswith("config/")],
                ]
            ),
            "prior_evidence": unique_refs([*parent["prior_evidence"], *refs]),
            "validation_evidence": unique_refs(
                [
                    *parent["validation_evidence"],
                    reference(xml),
                    reference(log),
                    reference(FOLDER / "validation-source-v1.json"),
                ]
            ),
            "source_policy_and_expert_admission_clocks": clocks,
            "fixed_economic_role_positions": [list(g) for g in rules.GROUPS],
            "fixed_economic_role_capital_budgets": [0.5, 0.5],
            "source_learning_state_reexecuted": False,
            "source_rate_loss_memory_or_labels_changed": False,
            "role_labels_are_independent_risks_or_protection_guarantees": False,
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
                "new_projected_weights_targets_or_accounts_computed": False,
                "source_learning_state_reexecuted": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="checkpoint_then_register_before_actual_projection")
    return reference(INTENT)


def prepare(data, experts, frozen):
    rules.require(
        not PREPARED.exists() and not (FOLDER / "preparation-attempt-v1.json").exists(),
        "Preserve the first complete or partial new projected target construction.",
    )
    rules.require(
        len(events("registered", rules.ID)) == 1,
        "Actual registration must precede all new real projected weights and targets.",
    )
    write_new_receipt(FOLDER / "preparation-attempt-v1.json", {"started_at": utc_now()})
    catalogue = read_json(base.checked_reference(read_json(rules.POLICY)["source_inputs"]))
    target = pd.DataFrame(np.nan, index=data.close.index, columns=data.close.columns)
    previous, source_previous = np.full(6, 1 / 6), None
    audits, checks = [], []
    tape = base.checked_reference(catalogue["cached_flat_six_tape"])
    trace = FOLDER / "actual-decision-inputs-resolved-v1.jsonl"
    expected = (
        source.rules.decision_sessions(data.close.index, read_json(source.rules.POLICY))
        .strftime("%Y-%m-%d")
        .tolist()
    )
    with tape.open() as stream, trace.open("x") as output:
        for index, day in enumerate(expected):
            lifecycle.runtime_guard()
            before_line, after_line = stream.readline(), stream.readline()
            rules.require(
                bool(before_line) and bool(after_line), "A missing cached pair cannot fill."
            )
            before, after = json.loads(before_line), json.loads(after_line)
            a = after["audit"]
            rules.require(
                before["decision_session"] == after["decision_session"] == day
                and a["decision_session"] == day
                and before["candidate_accounts_or_metrics_computed"] is False
                and a["score_mature"] is (index >= 2),
                "Keep every exact source origin and explicit learner maturity.",
            )
            if source_previous is not None:
                rules.require(
                    before["state"] == source_previous["adaptive_state"]
                    and before["previous_weights_hex"]
                    == [float(v).hex() for v in source_previous["expert_weights"]],
                    "Never reset,skip or rerun the cached source learning state.",
                )
            source_previous = a
            observed = source.rules.observations(experts, pd.Timestamp(day))
            rules.require(
                before["known_targets_hex"]
                == [[float(v).hex() for v in o.target] for o in observed]
                and before["known_target_dates"]
                == [str(o.target_known_at.date()) for o in observed]
                and all(o.target_known_at <= pd.Timestamp(day) for o in observed),
                "The cached source opinions and original admission clocks must remain unchanged.",
            )
            weights, details = rules.project(a["expert_weights"])
            opinions = np.stack([o.target.to_numpy() for o in observed])
            row = weights @ opinions
            checked = rules.independent(a["expert_weights"], weights, opinions, row)
            target.loc[pd.Timestamp(day)] = row
            audit = {
                "decision_session": day,
                "recorded_source_weights": a["expert_weights"],
                "projected_expert_weights": weights.tolist(),
                **details,
                "source_score_mature": a["score_mature"],
                "source_complete_learning_intervals": a["complete_learning_intervals"],
                "known_target_dates": before["known_target_dates"],
                "risk_and_cost": source.rules.risk_cost_audit(
                    observed, weights, previous, pd.Timestamp("2016-10-31"), pd.Timestamp(day)
                ),
                "source_learning_state_reexecuted": False,
                "source_wealth_guarantee_transferred_to_main": False,
            }
            for record in (
                {"phase": "before_role_projection", "source_before": before, "source_after": after},
                {
                    "phase": "after_role_projection",
                    "audit": audit,
                    "independent": checked,
                    "unified_target_hex": [float(v).hex() for v in row],
                },
            ):
                output.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
            audits.append(audit)
            checks.append(checked)
            previous = weights
        rules.require(not stream.readline(), "Extra source origins cannot be silently ignored.")
    rules.require(len(audits) == 121, "Retain the complete original decision lattice.")
    baseline = pd.read_csv(
        base.checked_reference(catalogue["cached_flat_six_targets"]),
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )
    rules.require(
        baseline.index.equals(target.index)
        and baseline.columns.equals(target.columns)
        and not baseline.equals(target),
        "A duplicate or changed-coordinate target path must not be funded.",
    )
    prefixes = []
    for position in (0, 2, 12, 60, 120):
        for i in range(position + 1):
            weights, _ = rules.project(audits[i]["recorded_source_weights"])
            rules.require(
                [float(v).hex() for v in weights]
                == [float(v).hex() for v in audits[i]["projected_expert_weights"]],
                "Future cached inputs changed earlier projected weights.",
            )
        prefixes.append(expected[position])
    coverage = []
    for window in read_json(base.PROTOCOL)["windows"]:
        rows = target.dropna(how="all").loc[: window["end"]]
        anchor = source.prior.previous_session(window["start"])
        rows = pd.concat([rows.loc[:anchor].tail(1), rows.loc[anchor + pd.Timedelta(days=1) :]])
        factors = [
            symbol
            for symbol in ("MTUM", "VLUE", "QUAL", "USMV", "IJR", "PKW")
            if (rows[symbol] > 1e-10).any()
        ]
        rules.require(len(factors) >= 3, "Preserve actual economic-factor participation.")
        coverage.append({"years": window["years"], "actual_factor_symbols": factors})
    write_new_receipt(FOLDER / "information-audits-v1.json", {"decisions": audits})
    directory = FOLDER / "targets" / rules.ID
    new_output_directory(directory)
    write_text_atomic(directory / "targets.csv", target.to_csv(float_format="%.17g"))
    write_new_receipt(
        directory / "proof.json",
        target_proof(
            data,
            target,
            CANDIDATES[0],
            frozen,
            {
                "decision_count": len(audits),
                "actual_input_and_resolved_record_count": 242,
                "initial_unscored_source_decisions": 2,
                "cached_mature_source_updates": 119,
                "prefix_truncations": prefixes,
                "formal_factor_coverage": coverage,
                "maximum_independent_weight_difference": max(
                    c["maximum_independent_weight_difference"] for c in checks
                ),
                "maximum_independent_target_difference": max(
                    c["maximum_independent_target_difference"] for c in checks
                ),
                "all_exact_half_role_masses": all(c["exact_half_role_masses"] for c in checks),
                "materially_different_cached_flat_six_origins": int(
                    (
                        (target.dropna() - baseline.loc[target.dropna().index]).abs().max(axis=1)
                        > 1e-12
                    ).sum()
                ),
                "actual_input_resolved_records": reference(trace),
                "source_learning_state_reexecuted": False,
                "source_rate_loss_memory_or_labels_changed": False,
                "source_wealth_guarantee_transferred_to_main": False,
                "new_source_or_shadow_account_runs": 0,
            },
        ),
    )
    entries = {
        rules.ID: {
            "targets": reference(directory / "targets.csv"),
            "proof": reference(directory / "proof.json"),
        }
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
    return bound(source.prior.prior.evaluate)(commit)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(RESULTS)
        result = read_json(base.checked_reference(saved["candidate_result_refs"][rules.ID]))
        catalogue = read_json(base.checked_reference(read_json(rules.POLICY)["source_inputs"]))
        control = read_json(base.checked_reference(catalogue["cached_flat_six_results"]))
        rules.require(set(result["paths"]) == base.PATH_KEYS, "Retain all four funded paths.")
        write_new_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "study_id": read_json(rules.POLICY)["study_id"],
                "candidate_id": rules.ID,
                "completed_at": saved["completed_at"],
                "status": result["review"]["status"],
                "all_four_original_paths": result["paths"],
                "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
                "cached_flat_six_control": catalogue["cached_flat_six_results"],
                "new_minus_cached_flat_six": {
                    key: {
                        metric: path["metrics"][metric]
                        - control["all_four_original_paths"][key]["metrics"][metric]
                        for metric in ("cagr", "sharpe", "max_drawdown")
                    }
                    for key, path in result["paths"].items()
                },
                "source_learning_state_reexecuted": False,
                "source_rate_loss_memory_or_labels_changed": False,
                "role_labels_are_independent_risks_or_protection_guarantees": False,
                "new_source_requests": 0,
                "new_source_RF_views": 0,
                "raw_prices_targets_or_daily_book_rows_exported": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
    return reference(PUBLIC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use an actual remotely verified singleton preregistration SHA.")
    print(
        json.dumps(
            {
                "freeze": initialize,
                "register": register,
                "evaluate": lambda: evaluate(args.commit),
                "export": export,
            }[args.action]()
        )
    )
