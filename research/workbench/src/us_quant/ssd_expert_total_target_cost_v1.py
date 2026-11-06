from __future__ import annotations

import argparse
import itertools
import json
import math
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import expert_total_target_cost_v1 as rules
from us_quant import ssd_commitment_macro_defense_hedge_v1 as prior
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_style_activity_financial_v1 import dependencies as _dependencies
from us_quant.storage import read_json, utc_now, write_json

base, ROOT, template, seven, financial = (
    prior.base,
    prior.ROOT,
    prior.template,
    prior.seven,
    prior.financial,
)
lifecycle, reference, unique_refs = prior.lifecycle, prior.reference, prior.unique_refs
current_counts = prior.current_counts
source = seven
FOLDER = base.REPORTS / "expert-total-target-cost-20261106-v1"
DESIGN, INVOCATION = FOLDER / "design-decision-v1.json", FOLDER / "actual-invocation-v1.json"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
PREPARED, RESULTS = FOLDER / "prepared-target-bundle-v1.json", FOLDER / "results.json"
PUBLIC = ROOT / "evidence/expert_total_target_cost_20261106_results_v1.json"
CANDIDATES, IDS = rules.CANDIDATES, (rules.ID,)
PACKETS = {rules.ID: ROOT / "evidence/expert_total_target_cost_20261106_registration_v1.json"}
OWN = (
    "config/expert-total-target-cost-v1.json",
    "src/us_quant/expert_total_target_cost_v1.py",
    "src/us_quant/ssd_expert_total_target_cost_v1.py",
    "tests/test_expert_total_target_cost_v1.py",
    "EXPERT_TOTAL_TARGET_COST_V1.md",
    "tests/test_commitment_macro_defense_hedge_v1.py",
    "tests/test_growth_macro_defense_hedge_v1.py",
    "tests/test_strategy_expert_pool_eight_v1.py",
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
        raise QuantError("The new cost scope cannot substitute projected financial counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "expert_total_target_cost_" + phase,
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
    inputs = read_json(base.checked_reference(policy["source_inputs"]))
    data, experts, parent, refs, clocks, benchmarks = prior.source_inputs()
    if (
        inputs["expert_ids"] != list(rules.EXPERT_IDS)
        or inputs["mechanism_groups"] != list(rules.MECHANISMS)
        or clocks != inputs["source_clocks"]
        or inputs["new_meta_scores_targets_or_accounts_computed"] is not False
        or inputs["admitted_assessment_views_reused_unchanged"] is not True
        or inputs["all_source_native_RF_claimed_equal_without_view"] is not False
    ):
        raise QuantError("Keep the already completed direct quartet and its exact source scope.")
    previous = read_json(base.checked_reference(inputs["previous_completed_cycle"]))
    refs = unique_refs(
        [
            *refs,
            *previous["exact_evidence_references"],
            *inputs["source_references"],
            policy["source_inputs"],
            *[
                inputs[k]
                for k in (
                    "previous_completed_cycle",
                    "source_quartet_policy",
                    "source_quartet_freeze",
                    "cached_direct_quartet_results",
                    "source_catalogue",
                )
            ],
            reference(FOLDER / "preservation-baseline-v1.json"),
            reference(FOLDER / "capture-preflight-v1.py"),
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    return data, experts, parent, refs, clocks, benchmarks


def inputs():
    return bound(template.inputs)()


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Keep the first complete or partial total-target-cost freeze.")
    with lifecycle.research_job():
        history = base.journal_events(base.JOURNAL)
        if len(history) != 1343 or history[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError("Freeze the single new law against actual J1343/291.")
        if any(e["candidate_id"] in IDS for e in history):
            raise QuantError("The one declared new candidate must be globally unused.")
        design = read_json(DESIGN)
        if (
            design["actual_native_event_seq"] != 182343
            or design["candidate_ids"] != list(IDS)
            or design["declared_financial_cohort_size"] != 1
            or design["new_meta_scores_targets_or_accounts_computed"] is not False
        ):
            raise QuantError("Freeze the new cost reference before its ranking or funding.")
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, _, parent, refs, clocks, _ = source_inputs()
        validation = read_json(FOLDER / "validation-source-v1.json")
        xml, log = [base.checked_reference(r) for r in validation["test_outputs"]]
        suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
        if (
            validation["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
            or validation["exit_code"] != 0
            or not suites
            or validation["passed_tests"] != sum(int(s.get("tests", 0)) for s in suites)
            or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
        ):
            raise QuantError(
                "Bind the exact cost-reference, source-state and singleton fence tests."
            )
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": read_json(rules.POLICY)["study_id"],
            "policy": reference(rules.POLICY),
            "expert_pool": reference(rules.POLICY),
            "construction": "same_four_complete_experts_cross_origin_desired_target_cost",
            "expert_pool_definition_role": "unchanged_direct_quartet_single_new_cost_law",
            "declared_financial_cohort_size": 1,
            "prior_journal_sequence": 1343,
            "prior_journal_head_sha256": rules.PRIOR_HEAD,
            "prior_completed_configurations": 291,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in [*deps, *refs] if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / "EXPERT_TOTAL_TARGET_COST_V1.md"),
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
            "cost_reference": "previous_origin_final_desired_meta_target",
            "cost_proxy_is_actual_holdings_NAV_or_main_fees": False,
            "cached_equal_and_rolling_controls_replayed": False,
            "admitted_legacy_defensive_views_reused_unchanged": True,
            "all_source_native_RF_claimed_equal_without_view": False,
            "new_source_RF_views": 0,
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
                "new_meta_scores_targets_or_accounts_computed": False,
                "global_pool_changed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="register_before_any_new_total_cost_ranking_or_targets")
    return reference(INTENT)


def verify_targets(data, experts, targets, audits):
    dates = rules.decision_sessions(data.close.index, read_json(rules.POLICY))
    if (
        set(targets) != set(IDS)
        or len(audits) != len(dates)
        or [a["decision_session"] for a in audits] != dates.strftime("%Y-%m-%d").tolist()
        or any(
            not t.index.equals(data.close.index)
            or not t.columns.equals(data.close.columns)
            or not t.dropna().index.equals(dates)
            for t in targets.values()
        )
    ):
        raise QuantError("Preserve every original source decision and formal target coordinate.")
    previous_weights, previous_target = np.full(4, 0.25), None
    target_gap = weight_gap = score_gap = cost_gap = 0.0
    for day, audit in zip(dates, audits, strict=True):
        observed = [e.observe(day, 126) for e in experts]
        if (
            tuple(o.identity.identifier for o in observed) != rules.EXPERT_IDS
            or audit["decision_information_clock"] != rules.CLOCK
            or audit["previous_rolling_expert_weights"] != previous_weights.tolist()
            or audit["previous_nominal_meta_target"] != previous_target
            or audit["previous_nominal_target_is_actual_holdings"] is not False
            or any(o.target_known_at is None or o.target_known_at > day for o in observed)
        ):
            raise QuantError("Retained nominal state, actual source clock or identity changed.")
        selection = audit["selection"]
        if not all(o.score_mature for o in observed):
            if selection["scores_available"] or selection["pairs"] or selection["selected_pair"]:
                raise QuantError("Unmatured history must remain explicitly unscored.")
            chosen = np.full(4, 0.25)
        else:
            excess = [
                [float(v) - float(rf) for v, rf in zip(o.net_returns, o.risk_free, strict=True)]
                for o in observed
            ]
            deviations = []
            for values in excess:
                mean = math.fsum(values[:84]) / 84
                deviations.append(math.sqrt(math.fsum((x - mean) ** 2 for x in values[:84]) / 83))
            alternatives = []
            pairs = selection["pairs"]
            if len(pairs) != 6:
                raise QuantError("Check every pair, not only the selected alternative.")
            for (first, second), row in zip(
                itertools.combinations(range(4), 2), pairs, strict=True
            ):
                names = sorted((rules.EXPERT_IDS[first], rules.EXPERT_IDS[second]))
                if row["pair"] != names:
                    raise QuantError("Keep exact source-order enumeration and lexical pair labels.")
                if "expert_weights" not in row:
                    continue
                w = np.zeros(4)
                total = deviations[first] + deviations[second]
                w[first], w[second] = deviations[second] / total, deviations[first] / total
                weight_gap = max(weight_gap, float(max(abs(w - row["expert_weights"]))))
                target = [
                    math.fsum(
                        float(o.target.iloc[i]) * float(x) for o, x in zip(observed, w, strict=True)
                    )
                    for i in range(len(data.close.columns))
                ]
                delta = [abs(a - b) for a, b in zip(target, previous_target, strict=True)]
                cost = math.fsum(delta) * 5 / 10000 + sum(x > 1e-12 for x in delta) / 10000
                cost_gap = max(
                    cost_gap, abs(cost - row["estimated_total_nominal_target_cost_fraction"])
                )
                scores = []
                for start, stop in ((0, 84), (84, 126)):
                    sample = [
                        math.fsum(float(w[i]) * excess[i][j] for i in range(4))
                        for j in range(start, stop)
                    ]
                    mean = math.fsum(sample) / len(sample)
                    sigma = math.sqrt(
                        math.fsum((x - mean) ** 2 for x in sample) / (len(sample) - 1)
                    )
                    scores.append(math.sqrt(252) * (mean - cost / 21) / sigma)
                score_gap = max(
                    score_gap,
                    abs(scores[0] - row["train_net_risk_score"]),
                    abs(scores[1] - row["validation_net_risk_score"]),
                )
                alternatives.append((round(min(scores), 12), names, w))
            if alternatives:
                chosen = min(alternatives, key=lambda x: (-x[0], x[1]))[2]
                wanted = min(alternatives, key=lambda x: (-x[0], x[1]))[1]
                if wanted != selection["selected_pair"]:
                    raise QuantError("Independent total-cost ranking selected another pair.")
            else:
                chosen = np.full(4, 0.25)
                if selection["scores_available"] or selection["selected_pair"] is not None:
                    raise QuantError("A degenerate menu cannot claim successful scoring.")
        expected = np.array(
            [
                math.fsum(
                    float(o.target.iloc[i]) * float(w)
                    for o, w in zip(observed, chosen, strict=True)
                )
                for i in range(len(data.close.columns))
            ]
        )
        weight_gap = max(
            weight_gap,
            float(max(abs(np.asarray(audit["rolling_expert_weights"]) - chosen))),
        )
        target_gap = max(target_gap, float(max(abs(targets[rules.ID].loc[day] - expected))))
        previous_weights = np.asarray(audit["rolling_expert_weights"])
        previous_target = targets[rules.ID].loc[day].tolist()
    if target_gap > 1e-10 or weight_gap > 1e-10 or score_gap > 1e-8 or cost_gap > 1e-10:
        raise QuantError(
            "Independent targets, ERC weights, complete scores or cost values disagree."
        )
    return {
        "maximum_independent_target_difference": target_gap,
        "maximum_independent_expert_weight_difference": weight_gap,
        "maximum_independent_net_score_difference": score_gap,
        "maximum_independent_total_nominal_cost_difference": cost_gap,
        "previous_nominal_target_is_actual_holdings": False,
        "total_nominal_cost_reference_independently_verified": True,
    }


def verify_retained_records(path, experts, targets, audits):
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != 2 * len(audits):
        raise QuantError("Keep a complete flushed input/resolution pair for each decision.")
    for i, audit in enumerate(audits):
        before, after = rows[2 * i : 2 * i + 2]
        day = pd.Timestamp(audit["decision_session"])
        observed = [e.observe(day, 126) for e in experts]
        if (
            before["record"] != "decision_input"
            or after["record"] != "decision_resolved"
            or before["decision_index"] != i
            or after["decision_index"] != i
            or before["as_of"] != audit["decision_session"]
            or before["observations"] != rules.snapshot(observed)
            or before["previous_weights"] != audit["previous_rolling_expert_weights"]
            or before["previous_weights_binary64_hex"]
            != [float(x).hex() for x in audit["previous_rolling_expert_weights"]]
            or before["previous_nominal_target"] != audit["previous_nominal_meta_target"]
            or before["previous_nominal_target_is_actual_holdings"] is not False
            or before["binary64_values_truncated_to_fixed_decimals"] is not False
            or after["audit"] != audit
            or after["asset_order"] != list(targets[rules.ID].columns)
            or after["combined_targets"] != {rules.ID: targets[rules.ID].loc[day].tolist()}
        ):
            raise QuantError(
                "Complete original source values and nominal state must be retained exactly."
            )
    return {
        "actual_input_and_resolved_record_count": len(rows),
        "exact_unrounded_source_values_and_nominal_state_retained": True,
    }


def prepare(*args):
    return bound(prior.prepare)(*args)


def register():
    return bound(template.register)()


def scoped(function, candidate):
    return bound(template.scoped)(function, candidate)


def cohort_publication(*args):
    return bound(template.cohort_publication)(*args)


def evaluate(commit):
    lifecycle.runtime_guard()
    if RESULTS.exists():
        raise QuantError("The completed single-rule financial evaluation cannot be replayed.")
    refs = {}
    for candidate in CANDIDATES:
        lifecycle.runtime_guard()
        refs[candidate["id"]] = scoped(financial.evaluate, candidate)(commit)
    write_new_receipt(
        RESULTS,
        {
            "completed_at": utc_now(),
            "candidate_result_refs": refs,
            "all_declared_candidates_and_four_original_paths_complete": True,
            "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
            "cached_equal_and_rolling_controls_replayed": False,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "order_authority": False,
        },
    )
    progress(
        "all_four_paths_complete", next_step="publish_every_path_without_source_or_control_replay"
    )
    return reference(RESULTS)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(RESULTS)
        inputs = read_json(base.checked_reference(read_json(rules.POLICY)["source_inputs"]))
        controls = read_json(base.checked_reference(inputs["cached_direct_quartet_results"]))
        result = read_json(base.checked_reference(saved["candidate_result_refs"][rules.ID]))
        if set(result["paths"]) != base.PATH_KEYS:
            raise QuantError("Publish every complete original financial path of the new law.")
        seven.common.retain_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "study_id": read_json(rules.POLICY)["study_id"],
                "completed_at": saved["completed_at"],
                "candidates": {
                    rules.ID: {"status": result["review"]["status"], "paths": result["paths"]}
                },
                "matched_cached_controls": {
                    name: {
                        "status": row["status"],
                        "paths": row["paths"],
                        "old_control_replayed": False,
                        "snapshot_is_current": False,
                    }
                    for name, row in controls["candidates"].items()
                },
                "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
                "cost_reference": "previous_origin_final_desired_meta_target",
                "cost_proxy_is_actual_holdings_NAV_or_main_fees": False,
                "source_NET_fees_already_in_history": True,
                "coefficient_window_or_risk_weighting_tuned": False,
                "old_controls_or_source_models_replayed": False,
                "global_pool_changed": False,
                "new_source_requests_or_RF_views": 0,
                "raw_prices_targets_or_daily_rows_exported": False,
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
        parser.error("Use a real remotely published complete-singleton-cohort preregistration SHA.")
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
