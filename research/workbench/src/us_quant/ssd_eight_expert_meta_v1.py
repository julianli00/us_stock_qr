from __future__ import annotations

import argparse
import ast
import itertools
import json
import math
import os
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import november_research_lifecycle_v1 as lifecycle
from us_quant import ssd_expert_target_consensus_v1 as financial
from us_quant import ssd_seven_expert_meta_v1 as seven
from us_quant import strategy_expert_pool_eight_v1 as rules
from us_quant.calendar import previous_session
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.ssd_current_progress_v1 import current_counts
from us_quant.ssd_expert_meta_research_v1 import scalar_score
from us_quant.storage import (
    digest_json,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)
from us_quant.strategy_experts_v1 import load_expert

base, ROOT, source = seven.base, seven.ROOT, seven
FOLDER = base.REPORTS / "eight-expert-inclusion-20261103-v1"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
PREPARED = FOLDER / "prepared-target-bundle-v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC = ROOT / "evidence/eight_expert_inclusion_20261103_results_v1.json"
CANDIDATES = rules.CANDIDATES
IDS = tuple(row["id"] for row in CANDIDATES)
PACKETS = {
    name: ROOT / "evidence" / f"eight_expert_inclusion_20261103_{i}_registration_v1.json"
    for i, name in enumerate(IDS)
}
OWN = (
    "config/strategy-expert-pool-eight-v1.json",
    "src/us_quant/strategy_expert_pool_eight_v1.py",
    "src/us_quant/ssd_eight_expert_meta_v1.py",
    "tests/test_strategy_expert_pool_eight_v1.py",
    "tests/test_ssd_eight_expert_meta_v1.py",
    "tests/test_strategy_expert_pool_v5.py",
    "tests/test_strategy_expert_pool_v4.py",
    "tests/test_strategy_expert_pool_v3.py",
    "tests/test_expert_initial_monthly_execution_v1.py",
    "tests/conftest.py",
)
reference, unique_refs = base.reference, seven.unique_refs


def events(kind, identifier):
    return [
        e
        for e in base.journal_events(base.JOURNAL)
        if e["kind"] == kind and e["candidate_id"] == identifier
    ]


def check_frozen(frozen):
    base.check_frozen_files(frozen)
    for row in frozen["prior_evidence"] + frozen["validation_evidence"]:
        base.checked_reference(row)


def dependencies():
    walker = seven.admitted.import_template.dependencies
    return FunctionType(
        walker.__code__,
        {**globals(), "ast": ast, "prior": seven},
        argdefs=walker.__defaults__,
    )()


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    data, experts, _, _, refs, clocks, admission_refs, benchmarks = seven.source_inputs()
    admission_path = base.checked_reference(policy["new_expert_admission"])
    admission = read_json(admission_path)
    expert, technical = load_expert(admission["record"], tuple(data.close.columns))
    if (
        admission["native_RF_exact_common"] is not True
        or admission["new_RF_or_recommendation_or_assessment_view"] is not False
        or technical["source_technical_and_funding_checks_passed"] is not True
        or technical["standalone_financial_verdict_preserved"] != "rejected"
        or expert.identity.identifier != rules.EXPERT_IDS[-1]
        or expert.identity.mechanism_group != rules.MECHANISMS[-1]
    ):
        raise QuantError("Append only the exact separately admitted complete hedge source.")
    experts = (*experts, expert)
    for item in experts:
        initial = item.observe(pd.Timestamp("2016-10-05"), 126)
        final = item.observe(data.close.index[-1], 2512)
        if (
            initial.target is None
            or initial.score_mature
            or len(initial.net_returns)
            or len(final.net_returns) != 2512
            or not final.risk_free.equals(data.risk_free.loc[final.risk_free.index])
        ):
            raise QuantError(
                "Preserve common exact RF, original clocks and explicit empty initial histories."
            )
    parent = read_json(base.checked_reference(reference(seven.PACKET)))["entries"][0][
        "specification"
    ]["frozen"]
    native = read_json(base.checked_reference(admission["record"]["registration"]))["entries"][0][
        "specification"
    ]["frozen"]
    extra = [
        policy["new_expert_admission"],
        admission["record"]["registration"],
        admission["record"]["results"],
        policy["source_seven_results"],
        policy["source_seven_policy"],
        reference(seven.PACKET),
        *native["code"],
        *native["policies"],
        *native["prior_evidence"],
        *native["validation_evidence"],
        technical["target_reference"],
        technical["shadow_reference"],
    ]
    primary = (
        ROOT
        / "reports/ssd-continuous-research-v1"
        / "conditional-hedge-expert-20261103-v1/primary-implementation-freeze-v1.json"
    )
    extra.extend(read_json(primary)["implementation_and_test_inputs"])
    extra.append(reference(primary))
    for row in admission["record"]["completed_paths"].values():
        extra.append(row)
        extra.extend(read_json(base.checked_reference(row))["artifacts"])
    refs = unique_refs([*refs, *admission_refs, *extra])
    for row in refs:
        base.checked_reference(row)
    clocks = [
        *clocks,
        {
            "id": expert.identity.identifier,
            "source_policy_registered_at": admission["record"]["introduced_at"],
            "actual_expert_admitted_at": admission["admitted_at"],
            "decision_information_clock": rules.CLOCK,
        },
    ]
    return data, experts, parent, refs, clocks, benchmarks


def progress(phase, **fields):
    counts = current_counts(base.journal_events(base.JOURNAL))
    if any(key in fields and fields[key] != value for key, value in counts.items()):
        raise QuantError("The study cannot override actual journal counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "eight_inclusion_" + phase,
            "study_id": read_json(rules.POLICY)["study_id"],
            "active_compute_job": None,
            **fields,
            **counts,
        },
    )


def validation_refs():
    xml = FOLDER / "offline-study-tests-v1.xml"
    suites = ElementTree.parse(xml).getroot().findall(".//testsuite")
    if not suites or any(
        int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
    ):
        raise QuantError("Finish the source, complete28-pair, causality and cohort-fence checks.")
    receipt = FOLDER / "validation-source-v1.json"
    proof = read_json(receipt)
    outputs = [reference(xml), reference(FOLDER / "offline-study-tests-v1.log")]
    if (
        proof["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
        or proof["test_outputs"] != outputs
        or proof["passed_tests"] != sum(int(s.get("tests", 0)) for s in suites)
        or proof["exit_code"] != 0
    ):
        raise QuantError("Successful canonical checks must bind these exact source/code bytes.")
    return [*outputs, reference(receipt)]


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the complete or partial first study freeze.")
    with lifecycle.research_job():
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, _, parent, refs, clocks, _ = source_inputs()
        history = base.journal_events(base.JOURNAL)
        if (
            len(history) != 699
            or history[-1]["sha256"] != read_json(rules.POLICY)["prior_journal_head_sha256"]
        ):
            raise QuantError(
                "Freeze the complete study against its real pre-outcome financial baseline."
            )
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": read_json(rules.POLICY)["study_id"],
            "policy": reference(rules.POLICY),
            "expert_pool": reference(rules.POLICY),
            "construction": "controlled_eighth_complete_expert_same_equal_and_rolling_laws",
            "expert_pool_definition_role": "study_specific_eight_only_original_seven_immutable",
            "code": unique_refs(
                [
                    *parent["code"],
                    *[
                        row
                        for row in [*deps, *refs]
                        if row["path"].startswith(("src/", "tests/")) or row["path"].endswith(".md")
                    ],
                    reference(ROOT / "EIGHT_EXPERT_INCLUSION_V1.md"),
                ]
            ),
            "policies": unique_refs(
                [
                    *parent["policies"],
                    *[row for row in [*deps, *refs] if row["path"].startswith("config/")],
                ]
            ),
            "prior_evidence": unique_refs(
                [*parent["prior_evidence"], *refs, reference(DESIGN), reference(INVOCATION)]
            ),
            "validation_evidence": unique_refs(
                [*parent["validation_evidence"], *validation_refs()]
            ),
            "source_policy_and_expert_admission_clocks": clocks,
            "decision_information_clock": rules.CLOCK,
            "new_shadow_account_runs": 0,
            "new_shadow_recommendation_artifacts": 0,
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
                "real_new_scores_targets_or_accounts_computed": False,
                "both_cohort_identities_registered": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="register_both_before_actual_scores_targets")
    return reference(INTENT)


def inputs():
    lifecycle.runtime_guard()
    intent = read_json(INTENT)
    for row in intent["source_references"] + intent["complete_dependency_inputs"]:
        base.checked_reference(row)
    if intent["package_versions"] != [
        list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())
    ]:
        raise QuantError("The frozen isolated source environment changed.")
    data, experts, _, refs, clocks, benchmarks = source_inputs()
    if (
        refs != intent["source_references"]
        or clocks != intent["frozen"]["source_policy_and_expert_admission_clocks"]
    ):
        raise QuantError("Source identities, availability or immutable inputs changed.")
    frozen = {
        **intent["frozen"],
        "prior_evidence": unique_refs([*intent["frozen"]["prior_evidence"], reference(INTENT)]),
    }
    check_frozen(frozen)
    return data, experts, frozen, benchmarks


def verify_targets(data, experts, targets, audits):
    dates = rules.decision_sessions(data.close.index, read_json(rules.POLICY))
    n = len(rules.EXPERT_IDS)
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
        raise QuantError("Keep the full original formal decision/return lattice.")
    previous, equal = np.full(n, 1 / n), np.full(n, 1 / n)
    max_target = max_weight = max_score = 0.0
    for day, audit in zip(dates, audits, strict=True):
        observed = [e.observe(day, 126) for e in experts]
        if (
            tuple(o.identity.identifier for o in observed) != rules.EXPERT_IDS
            or audit["decision_information_clock"] != rules.CLOCK
            or audit["previous_rolling_expert_weights"] != previous.tolist()
            or audit["equal_expert_weights"] != equal.tolist()
            or audit["expert_target_known_at"] != [str(o.target_known_at.date()) for o in observed]
            or any(o.target_known_at > day for o in observed)
            or audit["shadow_wealth_added_to_main_capital"] is not False
            or audit["source_feedback_floor_transferred_to_main_account"] is not False
        ):
            raise QuantError("Source clocks, full expert identity or capital separation changed.")
        selection = audit["selection"]
        if not all(o.score_mature for o in observed):
            if (
                selection["reason"] != "unscored_initial_equal_weights"
                or selection["scores_available"] is not False
                or selection["pairs"] != []
                or selection["selected_pair"] is not None
                or selection["safety_or_success_claimed"] is not False
            ):
                raise QuantError("An immature source cannot be a safe successful score.")
            chosen = equal.copy()
        else:
            values = [(o.net_returns - o.risk_free).tolist() for o in observed]
            known = observed[0].net_returns.index
            if len(known) != 126 or known[-1] > day:
                raise QuantError("Only common126 mature observations enter the fixed score.")
            means = [math.fsum(row[:84]) / 84 for row in values]
            deviations = [
                math.sqrt(math.fsum((x - m) ** 2 for x in row[:84]) / 83)
                for row, m in zip(values, means, strict=True)
            ]
            expected_pairs = {
                tuple(sorted((rules.EXPERT_IDS[i], rules.EXPERT_IDS[j])))
                for i, j in itertools.combinations(range(n), 2)
            }
            if (
                len(selection["pairs"]) != math.comb(n, 2)
                or {tuple(p["pair"]) for p in selection["pairs"]} != expected_pairs
            ):
                raise QuantError("Every one of the28pairs,including unscorable states,is required.")
            for pair in selection["pairs"]:
                i, j = [rules.EXPERT_IDS.index(name) for name in pair["pair"]]
                same = observed[i].identity.mechanism_group == observed[j].identity.mechanism_group
                reason = (
                    "same_mechanism_group"
                    if same
                    else (
                        "zero_training_variance" if min(deviations[i], deviations[j]) <= 0 else None
                    )
                )
                weights = np.zeros(n)
                if reason is None:
                    weights[i], weights[j] = (
                        deviations[j] / (deviations[i] + deviations[j]),
                        deviations[i] / (deviations[i] + deviations[j]),
                    )
                    portfolio = [
                        math.fsum(weights[k] * values[k][t] for k in range(n)) for t in range(126)
                    ]
                    if np.std(portfolio[:84], ddof=1) <= 0 or np.std(portfolio[84:], ddof=1) <= 0:
                        reason = "nonpositive_or_nonfinite_pair_risk"
                if reason is not None:
                    if pair.get("unscorable_reason") != reason:
                        raise QuantError("A pair's explicit unscorable reason disagrees.")
                    continue
                covariance = (
                    math.fsum(
                        (values[i][t] - means[i]) * (values[j][t] - means[j]) for t in range(84)
                    )
                    / 83
                )
                expected_contribution = np.zeros(n)
                expected_contribution[i] = weights[i] * (
                    weights[i] * deviations[i] ** 2 + weights[j] * covariance
                )
                expected_contribution[j] = weights[j] * (
                    weights[j] * deviations[j] ** 2 + weights[i] * covariance
                )
                actual = np.asarray(pair["expert_weights"])
                if (
                    actual.shape != (n,)
                    or not np.isfinite(actual).all()
                    or "unscorable_reason" in pair
                ):
                    raise QuantError(
                        "Every scored pair needs complete finite eight-dimensional weights."
                    )
                max_weight = max(max_weight, float(max(abs(actual - weights))))
                if (
                    abs(pair["training_correlation"] - covariance / (deviations[i] * deviations[j]))
                    > 1e-10
                    or max(
                        abs(
                            np.asarray(pair["training_marginal_variance_contributions"])
                            - expected_contribution
                        )
                    )
                    > 1e-12
                ):
                    raise QuantError(
                        "Correlation or marginal risk differs from mature source inputs."
                    )
                change = [
                    abs(
                        math.fsum(
                            (actual[k] - previous[k]) * float(observed[k].target[s])
                            for k in range(n)
                        )
                    )
                    for s in data.close.columns
                ]
                cost = math.fsum(change) * 5 / 10000 + sum(v > 1e-12 for v in change) / 10000
                score = [scalar_score(portfolio[:84], cost), scalar_score(portfolio[84:], cost)]
                max_score = max(
                    max_score,
                    abs(score[0] - pair["train_net_risk_score"]),
                    abs(score[1] - pair["validation_net_risk_score"]),
                )
                if (
                    abs(cost - pair["estimated_incremental_switch_cost_fraction"]) > 1e-12
                    or pair["estimated_incremental_changed_assets"]
                    != sum(v > 1e-12 for v in change)
                    or pair["selection_score"]
                    != min(pair["train_net_risk_score"], pair["validation_net_risk_score"])
                    or pair["selection_score_key"] != round(pair["selection_score"], 12)
                    or pair["cost_proxy_is_actual_main_account_cost"] is not False
                ):
                    raise QuantError("The unchanged incremental switch proxy or score key differs.")
            scored = [p for p in selection["pairs"] if "selection_score_key" in p]
            if scored:
                best = sorted(scored, key=lambda p: (-p["selection_score_key"], p["pair"]))[0]
                if (
                    selection["selected_pair"] != best["pair"]
                    or selection["scores_available"] is not True
                    or [
                        selection[k]
                        for k in ("train_start", "train_end", "validation_start", "validation_end")
                    ]
                    != [str(known[i].date()) for i in (0, 83, 84, 125)]
                ):
                    raise QuantError("The chronological mature sample or lexical winner changed.")
                chosen = np.asarray(best["expert_weights"])
            else:
                if (
                    selection["reason"] != "unscored_degenerate_pair_equal_weights"
                    or selection["scores_available"]
                ):
                    raise QuantError("Retain the explicit non-successful degenerate equal state.")
                chosen = equal.copy()
        if not np.allclose(chosen, audit["rolling_expert_weights"], rtol=0, atol=1e-12):
            raise QuantError("Issued source weights differ from their fixed original selection.")
        for name, weights in ((IDS[0], equal), (IDS[1], chosen)):
            expected = [
                math.fsum(float(o.target[s]) * weights[i] for i, o in enumerate(observed))
                for s in data.close.columns
            ]
            max_target = max(max_target, float(max(abs(targets[name].loc[day] - expected))))
        previous = chosen
    if max_target > 1e-10 or max_weight > 1e-10 or max_score > 1e-8:
        raise QuantError("Independent target,risk weight or net-score agreement failed.")
    return {
        "maximum_independent_target_difference": max_target,
        "maximum_independent_expert_weight_difference": max_weight,
        "maximum_independent_net_score_difference": max_score,
    }


def verify_retained_records(path, experts, targets, audits):
    return FunctionType(
        seven.verify_retained_records.__code__,
        {**seven.verify_retained_records.__globals__, **globals()},
        argdefs=seven.verify_retained_records.__defaults__,
    )(path, experts, targets, audits)


def prepare(data, experts, frozen):
    if PREPARED.exists():
        saved = read_json(PREPARED)
        if saved["frozen_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting prior eight-source targets.")
        for entry in saved["entries"].values():
            for row in entry.values():
                base.checked_reference(row)
        return saved["entries"]
    if (FOLDER / "preparation-attempt-v1.json").exists():
        raise QuantError("Preserve an unfinished actual combination instead of replaying it.")
    if any(len(events("registered", c["id"])) != 1 for c in CANDIDATES):
        raise QuantError(
            "Both real cohort identities must register before any new scores or targets."
        )
    write_new_receipt(
        FOLDER / "preparation-attempt-v1.json",
        {"started_at": utc_now(), "frozen_sha256": digest_json(frozen)},
    )
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
            before_decision=lifecycle.runtime_guard,
            retain=retain,
        )
    retained = verify_retained_records(log, experts, targets, audits)
    independent = verify_targets(data, experts, targets, audits)
    if len(audits) != 121:
        raise QuantError("Keep every original121capital/month-end decision.")
    prefixes = []
    for i in (0, 6, 12, 60, 120):
        day = pd.Timestamp(audits[i]["decision_session"])
        early, earlier = rules.build_targets(
            experts,
            data.close.loc[:day].index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            before_decision=lifecycle.runtime_guard,
        )
        if (
            any(not early[n].equals(targets[n].loc[:day]) for n in IDS)
            or earlier != audits[: i + 1]
        ):
            raise QuantError("Future data changed a prior complete-pool decision.")
        prefixes.append(str(day.date()))
    write_new_receipt(FOLDER / "information-audits-v1.json", {"decisions": audits})
    entries = {}
    for name in IDS:
        directory = FOLDER / "targets" / name
        new_output_directory(directory)
        active = targets[name].dropna(how="all")
        coverage = []
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
                raise QuantError("Both funded horizons require known anchors and genuine factors.")
            coverage.append({"years": window["years"], "actual_factor_symbols": factors})
        write_text_atomic(directory / "targets.csv", targets[name].to_csv(float_format="%.17g"))
        write_new_receipt(
            directory / "proof.json",
            {
                "checked_at": utc_now(),
                "frozen_specification_sha256": digest_json(frozen),
                **independent,
                **retained,
                "decision_count": len(audits),
                "all_pairs_per_mature_decision": 28,
                "prefix_truncations": prefixes,
                "formal_factor_coverage": coverage,
                "actual_input_resolved_records": reference(log),
                "candidate_accounts_or_metrics_computed": False,
                "original_source_signal_recomputed": False,
                "new_source_or_shadow_account_runs": 0,
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
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        data, experts, frozen, _ = inputs()
        if any(events("technical_failure", name) for name in IDS):
            raise QuantError("A terminal cohort identity cannot be rescued or repeated.")
        for candidate in CANDIDATES:
            name = candidate["id"]
            if not events("registered", name):
                base.append_event(
                    base.JOURNAL, "registered", name, {"candidate": candidate, "frozen": frozen}
                )
            elif events("registered", name)[0]["body"] != {
                "candidate": candidate,
                "frozen": frozen,
            }:
                raise QuantError("Cohort membership or law changed after actual registration.")
        try:
            prepared = prepare(data, experts, frozen)
            for candidate in CANDIDATES:
                name = candidate["id"]
                if not events("target_proof", name):
                    base.append_event(
                        base.JOURNAL,
                        "target_proof",
                        name,
                        {
                            "registered_event_sha256": events("registered", name)[0]["sha256"],
                            **prepared[name],
                        },
                    )
                entry = seven.common.packet_entry(candidate, frozen)
                seven.common.retain_receipt(
                    PACKETS[name],
                    {
                        "schema_version": 2,
                        "program_id": "ssd_continuous_research_v1",
                        "study_id": frozen["study_id"],
                        "created_at": utc_now(),
                        "entries": [entry],
                        "complete_fixed_cohort_ids": list(IDS),
                        "cohort_registrations": [events("registered", n)[0]["sha256"] for n in IDS],
                        "candidate_accounts_or_metrics_computed": False,
                        "order_authority": False,
                        "independent_forward_validation": False,
                        "investment_objective_verified": False,
                    },
                    ("created_at",),
                )
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            for name in IDS:
                if not events("technical_failure", name):
                    base.append_event(
                        base.JOURNAL,
                        "technical_failure",
                        name,
                        {
                            "stage": "eight_source_preparation",
                            "reason": str(error),
                            "partial_inputs_preserved": True,
                        },
                    )
            write_new_receipt(
                FOLDER / "preparation-failure-v1.json",
                {
                    "failed_at": utc_now(),
                    "error": str(error),
                    "original_source_or_failed_algorithm_rescue": False,
                },
            )
            progress("technical_failure_preserved", next_step="do_not_retry_or_retune_this_cohort")
            raise
        progress(
            "registered_before_accounts", next_step="publish_both_exact_packets_before_funding"
        )
    return {name: reference(path) for name, path in PACKETS.items()}


def scoped(function, candidate):
    identifier = candidate["id"]
    folder = FOLDER / "main-accounts" / identifier
    context = {
        **function.__globals__,
        **globals(),
        "ID": identifier,
        "CANDIDATE": candidate,
        "FOLDER": folder,
        "RESULTS": folder / "result.json",
        "PACKET": PACKETS[identifier],
        "events": lambda kind: events(kind, identifier),
        "publication": cohort_publication,
        "record_failure": lambda stage, error: scoped(financial.record_failure, candidate)(
            stage, error
        ),
        "evaluate_path": lambda *args: scoped(financial.evaluate_path, candidate)(*args),
    }
    return FunctionType(function.__code__, context, argdefs=function.__defaults__)


def cohort_publication(frozen, commit):
    fences = [
        row
        for row in base.journal_events(base.JOURNAL)
        if row["kind"] == "publication" and row["body"].get("study_id") == frozen["study_id"]
    ]
    if not fences and any(events("path_started", name) for name in IDS):
        raise QuantError("No cohort account may predate the one complete-cohort publication fence.")
    registrations = [events("registered", name) for name in IDS]
    if any(len(rows) != 1 for rows in registrations):
        raise QuantError("Publish every declared cohort identity's actual unique registration.")
    actual_registrations = [rows[0]["sha256"] for rows in registrations]
    checked = {}
    for candidate in CANDIDATES:
        packet = read_json(PACKETS[candidate["id"]])
        if (
            packet["complete_fixed_cohort_ids"] != list(IDS)
            or packet["cohort_registrations"] != actual_registrations
        ):
            raise QuantError("Publish every declared cohort identity before the first account.")
        checked[candidate["id"]] = scoped(financial.publication, candidate)(frozen, commit)
    return {
        "published_commit": commit,
        "all_fixed_cohort_packets_verified_before_accounts": True,
        "candidate_publication_verifications": checked,
    }


def evaluate(commit):
    lifecycle.runtime_guard()
    if RESULTS.exists():
        for path in read_json(RESULTS)["candidate_result_refs"].values():
            base.checked_reference(path)
        return reference(RESULTS)
    refs = {}
    for candidate in CANDIDATES:
        lifecycle.runtime_guard()
        ref = scoped(financial.evaluate, candidate)(commit)
        refs[candidate["id"]] = ref
    write_new_receipt(
        RESULTS,
        {
            "completed_at": utc_now(),
            "candidate_result_refs": refs,
            "all_declared_candidates_and_eight_original_paths_complete": True,
            "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
            "original_seven_policy_changed": False,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "order_authority": False,
        },
    )
    progress("all_eight_paths_complete", next_step="publish_complete_results_without_source_replay")
    return reference(RESULTS)


def export():
    results = read_json(RESULTS)
    saved = read_json(base.checked_reference(read_json(rules.POLICY)["source_seven_results"]))
    candidates = {}
    for candidate, control in zip(
        CANDIDATES,
        ("expert_pool_seven_equal_weight_v1", "expert_pool_seven_rolling_pair_v1"),
        strict=True,
    ):
        name = candidate["id"]
        data = read_json(base.checked_reference(results["candidate_result_refs"][name]))
        if set(data["paths"]) != base.PATH_KEYS:
            raise QuantError("No financial path may be omitted from the complete cohort output.")
        candidates[name] = {
            "status": data["review"]["status"],
            "paths": data["paths"],
            "registered_event_sha256": events("registered", name)[0]["sha256"],
            "matched_cached_seven_control": {
                "candidate_id": control,
                "paths": saved["candidates"][control]["paths"],
                "old_control_replayed": False,
            },
        }
    body = {
        "schema_version": 1,
        "completed_at": results["completed_at"],
        "study_id": read_json(rules.POLICY)["study_id"],
        "candidates": candidates,
        "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
        "declared_study_pool_size": 8,
        "original_seven_policy_changed": False,
        "source_expert_generators_shadows_and_accounts_replayed": False,
        "new_source_requests": 0,
        "new_economic_factor_definitions": 0,
        "raw_source_prices_targets_or_daily_book_rows_exported": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    seven.common.retain_receipt(PUBLIC, body)
    return reference(PUBLIC)


def main():
    parser = argparse.ArgumentParser(
        description="Controlled eight-source pool study; no broker orders."
    )
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use an actual remotely published complete-cohort preregistration SHA.")
    actions = {"freeze": initialize, "register": register, "export": export}
    print(
        json.dumps(
            evaluate(args.commit) if args.action == "evaluate" else actions[args.action](), indent=2
        )
    )


if __name__ == "__main__":
    main()
