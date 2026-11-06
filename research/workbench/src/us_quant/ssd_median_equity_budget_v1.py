from __future__ import annotations

import argparse
import json
import math
import os
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import pandas as pd

from us_quant import median_equity_budget_v1 as rules
from us_quant import ssd_conditional_volatility_quintet_v1 as prior
from us_quant.calendar import previous_session
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
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
    prior.base,
    prior.ROOT,
    prior.template,
    prior.seven,
    prior.financial,
)
lifecycle, reference, unique_refs = prior.lifecycle, prior.reference, prior.unique_refs
current_counts, source = prior.current_counts, seven
FOLDER = base.REPORTS / "median-equity-budget-20261106-v1"
DESIGN, INVOCATION = FOLDER / "design-decision-v1.json", FOLDER / "actual-invocation-v1.json"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
PREPARED, RESULTS = FOLDER / "prepared-target-bundle-v1.json", FOLDER / "results.json"
PUBLIC = ROOT / "evidence/median_equity_budget_20261106_results_v1.json"
CANDIDATES, IDS = rules.CANDIDATES, (rules.ID,)
PACKETS = {rules.ID: ROOT / "evidence/median_equity_budget_20261106_registration_v1.json"}
OWN = (
    "config/median-equity-budget-v1.json",
    "src/us_quant/median_equity_budget_v1.py",
    "src/us_quant/ssd_median_equity_budget_v1.py",
    "tests/test_median_equity_budget_v1.py",
    "MEDIAN_EQUITY_BUDGET_V1.md",
    "tests/test_conditional_volatility_quintet_v1.py",
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
        raise QuantError("A target budget cannot substitute projected financial counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "median_equity_budget_" + phase,
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
    catalogue = read_json(base.checked_reference(policy["source_catalogue"]))
    data, experts, _, refs, clocks, benchmarks = prior.source_inputs()
    parent = prior.events("registered", prior.IDS[1])[0]["body"]["frozen"]
    prepared = read_json(base.checked_reference(catalogue["source_quintet_prepared_bundle"]))
    if (
        catalogue["expert_ids"] != list(rules.EXPERT_IDS)
        or catalogue["mechanism_groups"] != list(rules.MECHANISMS)
        or catalogue["source_clocks"] != clocks
        or prepared["frozen_sha256"] != digest_json(parent)
        or catalogue["cached_upstream_candidate_id"] != prior.IDS[1]
        or catalogue["cached_upstream_targets"] != prepared["entries"][prior.IDS[1]]["targets"]
        or catalogue["new_equity_medians_caps_targets_or_accounts_computed"] is not False
        or catalogue["upstream_ranks_weights_scores_targets_or_accounts_recomputed"] is not False
        or catalogue["admitted_legacy_defensive_views_reused_unchanged"] is not True
        or catalogue["all_source_native_RF_claimed_equal_without_view"] is not False
    ):
        raise QuantError("Keep the exact completed upstream selector and five admitted opinions.")
    previous = read_json(base.checked_reference(catalogue["previous_completed_cycle"]))
    refs = unique_refs(
        [
            *refs,
            *previous["exact_evidence_references"],
            *catalogue["source_references"],
            policy["source_catalogue"],
            *[
                catalogue[k]
                for k in (
                    "source_quintet_policy",
                    "source_quintet_freeze",
                    "source_quintet_prepared_bundle",
                    "source_quintet_actual_decision_tape",
                    "cached_upstream_targets",
                    "cached_quintet_results",
                    "previous_completed_cycle",
                )
            ],
            reference(FOLDER / "preservation-baseline-v1.json"),
            reference(FOLDER / "capture-preflight-v1.py"),
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    return data, experts, parent, refs, clocks, benchmarks


def cached_targets():
    catalogue = read_json(base.checked_reference(read_json(rules.POLICY)["source_catalogue"]))
    return pd.read_csv(
        base.checked_reference(catalogue["cached_upstream_targets"]),
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )


def inputs():
    return bound(template.inputs)()


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the first complete or partial majority-budget freeze.")
    with lifecycle.research_job():
        history = base.journal_events(base.JOURNAL)
        if len(history) != 1369 or history[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError("Freeze the single new construction against actual J1369/293.")
        if any(e["candidate_id"] in IDS for e in history):
            raise QuantError("The one declared median-budget identity must be unused.")
        design = read_json(DESIGN)
        if (
            design["actual_native_event_seq"] != 183607
            or design["candidate_ids"] != list(IDS)
            or design["declared_financial_cohort_size"] != 1
            or design["new_equity_medians_caps_targets_or_accounts_computed"] is not False
        ):
            raise QuantError("Freeze the current-opinion budget before deriving any new targets.")
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
            raise QuantError("Bind the budget, BIL transfer, causality and singleton-fence checks.")
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": read_json(rules.POLICY)["study_id"],
            "policy": reference(rules.POLICY),
            "expert_pool": reference(rules.POLICY),
            "construction": "cached_quintet_rolling_with_current_opinion_median_equity_budget",
            "expert_pool_definition_role": "unchanged_five_complete_sources_single_budget_law",
            "declared_financial_cohort_size": 1,
            "prior_journal_sequence": 1369,
            "prior_journal_head_sha256": rules.PRIOR_HEAD,
            "prior_completed_configurations": 293,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in [*deps, *refs] if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / "MEDIAN_EQUITY_BUDGET_V1.md"),
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
            "upstream_candidate_id": prior.IDS[1],
            "upstream_selection_state_recomputed_or_changed": False,
            "equity_budget_is_actual_holdings_or_drawdown_limit": False,
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
                "new_equity_medians_caps_targets_or_accounts_computed": False,
                "global_pool_changed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="register_before_any_new_median_or_capped_target")
    return reference(INTENT)


def verify_targets(data, experts, targets, audits, cached=None):
    cached = cached_targets() if cached is None else cached
    dates = rules.decision_sessions(data.close.index, read_json(rules.POLICY))
    if (
        set(targets) != set(IDS)
        or len(audits) != len(dates)
        or [a["decision_session"] for a in audits] != dates.strftime("%Y-%m-%d").tolist()
        or not targets[rules.ID].index.equals(data.close.index)
        or not targets[rules.ID].columns.equals(data.close.columns)
        or not targets[rules.ID].dropna(how="all").index.equals(dates)
    ):
        raise QuantError("Retain every original decision, formal date and asset coordinate.")
    gap = budget_gap = 0.0
    for day, audit in zip(dates, audits, strict=True):
        observed = [e.observe(day, 126) for e in experts]
        original = cached.loc[day]
        expected, budget = rules.independent_target(original, observed, day)
        target = targets[rules.ID].loc[day]
        gap = max(gap, max(abs(float(a) - float(b)) for a, b in zip(target, expected, strict=True)))
        budget_gap = max(budget_gap, abs(budget - audit["median_equity_allocation"]))
        before = math.fsum(float(original[s]) for s in rules.EQUITY)
        after = math.fsum(float(target[s]) for s in rules.EQUITY)
        if (
            audit["decision_information_clock"] != rules.CLOCK
            or audit["source_equity_allocations"]
            != [math.fsum(float(o.target[s]) for s in rules.EQUITY) for o in observed]
            or audit["cap_binding"] != (before - budget > rules.TOLERANCE)
            or audit["input_equity_allocation"] != before
            or audit["output_equity_allocation"] != after
            or audit["equity_transferred_to_BIL"] != before - after
            or audit["source_score_mature"] != [o.score_mature for o in observed]
            or audit["upstream_selection_state_recomputed_or_changed"] is not False
            or any(target[s] != original[s] for s in ("IEF", "GLD", "TLT"))
        ):
            raise QuantError("Independent budget, conserved targets or source information differs.")
    if gap > 1e-10 or budget_gap > 1e-12:
        raise QuantError("The independent rational budget construction disagrees.")
    return {
        "maximum_independent_target_difference": gap,
        "maximum_independent_median_budget_difference": budget_gap,
        "binding_budget_decisions": sum(a["cap_binding"] for a in audits),
        "new_upstream_scores_weights_covariance_or_target_runs": 0,
        "actual_holdings_or_drawdown_limit_claimed": False,
    }


def verify_retained_records(path, experts, targets, audits, cached=None):
    cached = cached_targets() if cached is None else cached
    with path.open() as stream:
        for i, audit in enumerate(audits):
            before_line, after_line = stream.readline(), stream.readline()
            if not before_line or not after_line:
                raise QuantError("A partial median-budget input stream cannot become complete.")
            before, after = json.loads(before_line), json.loads(after_line)
            day = pd.Timestamp(audit["decision_session"])
            observed = [e.observe(day, 126) for e in experts]
            if (
                before["record"] != "decision_input"
                or after["record"] != "decision_resolved"
                or before["decision_index"] != i
                or after["decision_index"] != i
                or before["as_of"] != audit["decision_session"]
                or before["observations"] != rules.snapshot(observed)
                or before["asset_order"] != list(targets[rules.ID].columns)
                or before["upstream_target_binary64_hex"]
                != [float(v).hex() for v in cached.loc[day]]
                or before["upstream_scores_weights_or_target_recomputed"] is not False
                or after["audit"] != audit
                or after["asset_order"] != before["asset_order"]
                or after["combined_targets"] != {rules.ID: targets[rules.ID].loc[day].tolist()}
            ):
                raise QuantError("Keep every exact source opinion and immutable upstream target.")
        if stream.readline():
            raise QuantError("Unexpected extra median-budget input records.")
    return {
        "actual_input_and_resolved_record_count": 2 * len(audits),
        "exact_unrounded_opinions_and_upstream_targets_retained": True,
    }


def prepare(data, experts, frozen):
    if PREPARED.exists():
        saved = read_json(PREPARED)
        if saved["frozen_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting prior capped targets.")
        for entry in saved["entries"].values():
            for ref in entry.values():
                base.checked_reference(ref)
        return saved["entries"]
    if (FOLDER / "preparation-attempt-v1.json").exists():
        raise QuantError("Keep partial capped targets instead of replaying them.")
    if len(events("registered", rules.ID)) != 1:
        raise QuantError("The actual singleton registration must precede all new budgets.")
    write_new_receipt(FOLDER / "preparation-attempt-v1.json", {"started_at": utc_now()})
    cached = cached_targets()
    log = FOLDER / "actual-decision-inputs-resolved-v1.jsonl"
    with log.open("x") as stream:

        def retain(row):
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        targets, audits = rules.build_targets(
            experts,
            data.close.index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            cached,
            before_decision=lifecycle.runtime_guard,
            retain=retain,
        )
    retained = verify_retained_records(log, experts, targets, audits, cached)
    independent = verify_targets(data, experts, targets, audits, cached)
    if len(audits) != 121:
        raise QuantError("Retain all original initial and month-end decisions.")
    prefixes = []
    for position in (0, 6, 12, 60, 120):
        lifecycle.runtime_guard()
        day = pd.Timestamp(audits[position]["decision_session"])
        early, earlier = rules.build_targets(
            experts,
            data.close.loc[:day].index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            cached.loc[:day],
            before_decision=lifecycle.runtime_guard,
        )
        if not early[rules.ID].equals(targets[rules.ID].loc[:day]) or (
            earlier != audits[: position + 1]
        ):
            raise QuantError("Future source information changed an earlier capped target.")
        prefixes.append(str(day.date()))
    write_new_receipt(FOLDER / "information-audits-v1.json", {"decisions": audits})
    directory = FOLDER / "targets" / rules.ID
    new_output_directory(directory)
    active, coverage = targets[rules.ID].dropna(how="all"), []
    for window in read_json(base.PROTOCOL)["windows"]:
        anchor = previous_session(window["start"])
        known = active.loc[:anchor].tail(1)
        rows = pd.concat([known, active.loc[anchor + pd.Timedelta(days=1) : window["end"]]])
        factors = [
            s for s in ("MTUM", "VLUE", "QUAL", "USMV", "IJR", "PKW") if (rows[s] > 1e-10).any()
        ]
        if known.empty or len(factors) < 3:
            raise QuantError("The capped policy still needs all original factor families.")
        coverage.append({"years": window["years"], "actual_factor_symbols": factors})
    write_text_atomic(directory / "targets.csv", targets[rules.ID].to_csv(float_format="%.17g"))
    write_new_receipt(
        directory / "proof.json",
        target_proof(
            frozen,
            {
                **independent,
                **retained,
                "decision_count": len(audits),
                "prefix_truncations": prefixes,
                "formal_factor_coverage": coverage,
                "actual_input_resolved_records": reference(log),
                "upstream_mature_pair_count": 10,
                "new_pair_scores_or_covariance_estimates": 0,
                "new_source_or_shadow_account_runs": 0,
                "admitted_legacy_defensive_views_reused_unchanged": True,
                "all_source_native_RF_claimed_equal_without_view": False,
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
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        data, experts, frozen, _ = inputs()
        if events("registered", rules.ID) or events("technical_failure", rules.ID):
            raise QuantError("Do not replay or rescue a registered majority-budget identity.")
        base.append_event(
            base.JOURNAL,
            "registered",
            rules.ID,
            {
                "candidate": CANDIDATES[0],
                "frozen": frozen,
            },
        )
        try:
            prepared = prepare(data, experts, frozen)
            registered = events("registered", rules.ID)[0]["sha256"]
            base.append_event(
                base.JOURNAL,
                "target_proof",
                rules.ID,
                {
                    "registered_event_sha256": registered,
                    **prepared[rules.ID],
                },
            )
            entry = seven.common.packet_entry(CANDIDATES[0], frozen)
            seven.common.retain_receipt(
                PACKETS[rules.ID],
                {
                    "schema_version": 2,
                    "program_id": "ssd_continuous_research_v1",
                    "study_id": frozen["study_id"],
                    "created_at": utc_now(),
                    "entries": [entry],
                    "complete_fixed_cohort_ids": list(IDS),
                    "cohort_registrations": [registered],
                    "candidate_accounts_or_metrics_computed": False,
                    "order_authority": False,
                    "independent_forward_validation": False,
                    "investment_objective_verified": False,
                },
                ("created_at",),
            )
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            base.append_event(
                base.JOURNAL,
                "technical_failure",
                rules.ID,
                {
                    "stage": "median_equity_budget_target_preparation",
                    "reason": str(error),
                    "partial_inputs_preserved": True,
                },
            )
            write_new_receipt(
                FOLDER / "preparation-failure-v1.json",
                {
                    "failed_at": utc_now(),
                    "error": str(error),
                    "partial_inputs_preserved": True,
                    "upstream_or_failed_algorithm_rescue": False,
                },
            )
            progress(
                "technical_failure_preserved", next_step="do_not_retry_or_retune_this_identity"
            )
            raise
        progress("registered_before_accounts", next_step="deduplicate_then_publish_actual_packet")
    return {rules.ID: reference(PACKETS[rules.ID])}


def scoped(function, candidate):
    return bound(template.scoped)(function, candidate)


def cohort_publication(*args):
    return bound(template.cohort_publication)(*args)


def evaluate(commit):
    lifecycle.runtime_guard()
    if RESULTS.exists():
        raise QuantError("Do not replay the completed majority-budget accounts.")
    ref = scoped(financial.evaluate, CANDIDATES[0])(commit)
    write_new_receipt(
        RESULTS,
        {
            "completed_at": utc_now(),
            "candidate_result_refs": {rules.ID: ref},
            "all_declared_candidates_and_four_original_paths_complete": True,
            "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
            "cached_equal_and_rolling_controls_replayed": False,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "order_authority": False,
        },
    )
    progress("all_four_paths_complete", next_step="publish_every_path_without_upstream_replay")
    return reference(RESULTS)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(RESULTS)
        catalogue = read_json(base.checked_reference(read_json(rules.POLICY)["source_catalogue"]))
        controls = read_json(base.checked_reference(catalogue["cached_quintet_results"]))
        result = read_json(base.checked_reference(saved["candidate_result_refs"][rules.ID]))
        if set(result["paths"]) != base.PATH_KEYS:
            raise QuantError("Publish every original majority-budget financial path.")
        contrasts = {
            key: {
                "metrics": {
                    m: row["metrics"][m]
                    - controls["candidates"][prior.IDS[1]]["paths"][key]["metrics"][m]
                    for m in ("cagr", "sharpe", "max_drawdown")
                },
                "execution": {
                    m: row["execution"][m]
                    - controls["candidates"][prior.IDS[1]]["paths"][key]["execution"][m]
                    for m in (
                        "annualized_two_sided_turnover",
                        "model_order_count",
                        "total_cost_usd",
                    )
                },
            }
            for key, row in result["paths"].items()
        }
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
                    n: {**v, "old_control_replayed": False, "snapshot_is_current": False}
                    for n, v in controls["candidates"].items()
                },
                "new_minus_cached_rolling_quintet": contrasts,
                "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
                "source_pool_size": 5,
                "upstream_selection_state_recomputed_or_changed": False,
                "equity_cap_is_actual_holdings_or_drawdown_bound": False,
                "new_source_requests": 0,
                "new_source_RF_views": 0,
                "old_controls_or_source_models_replayed": False,
                "raw_prices_targets_or_daily_rows_exported": False,
                "global_pool_changed": False,
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
        parser.error("Use an actual remotely published singleton preregistration SHA.")
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
