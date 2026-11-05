from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from fractions import Fraction
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import expert_distinct_opinion_pair_v1 as rules
from us_quant import ssd_expert_target_consensus_v1 as financial
from us_quant import ssd_lagged_spy_price_impact_expert_inclusion_v1 as cached
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

base, ROOT, source, lifecycle = cached.base, cached.ROOT, cached.seven, cached.lifecycle
reference, unique_refs = cached.reference, cached.unique_refs
FOLDER = base.REPORTS / "distinct-opinion-pair-financial-20261105-v1"
POLICY = ROOT / "config/distinct-opinion-pair-financial-v1.json"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
PREPARED = FOLDER / "prepared-target-bundle-v1.json"
RESULTS = FOLDER / "results.json"
PACKET = ROOT / "evidence/distinct_opinion_pair_financial_20261105_registration_v1.json"
PUBLIC = ROOT / "evidence/distinct_opinion_pair_financial_20261105_results_v1.json"
ID = rules.ID
CANDIDATE = {"id": ID, "weighting": "original_rolling_pair_exact_current_opinion_distinction"}
OWN = (
    "config/distinct-opinion-pair-financial-v1.json",
    "src/us_quant/ssd_distinct_opinion_pair_financial_v1.py",
    "tests/test_ssd_distinct_opinion_pair_financial_v1.py",
    "DISTINCT_OPINION_PAIR_FINANCIAL_V1.md",
    *rules.OWN,
    "tests/conftest.py",
)


def bound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


def events(kind):
    return [
        e
        for e in base.journal_events(base.JOURNAL)
        if e["kind"] == kind and e["candidate_id"] == ID
    ]


def check_frozen(frozen):
    cached.check_frozen(frozen)


def dependencies():
    return bound(cached.previous.dependencies)()


def progress(phase, **fields):
    counts = current_counts(base.journal_events(base.JOURNAL))
    if any(key in fields and fields[key] != value for key, value in counts.items()):
        raise QuantError("Distinct-opinion financial progress cannot override actual counts.")
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "distinct_opinion_financial_" + phase.removeprefix("consensus_"),
            "study_id": read_json(POLICY)["study_id"],
            "active_compute_job": None,
            **counts,
            **fields,
        },
    )


def check_primary():
    policy = read_json(POLICY)
    expected = {
        "schema_version": 1,
        "study_id": "ssd_distinct_opinion_pair_financial_20261105_v1",
        "candidate_id": ID,
        "primary_policy": {
            "path": "config/expert-distinct-opinion-pair-v1.json",
            "sha256": "c65b8c6bbc28eb6d1e028e4877ec45d03c09bfa4e8e4d741ae8e1ac50f00db8a",
        },
        "primary_freeze": {
            "path": (
                "reports/ssd-continuous-research-v1/"
                "distinct-opinion-pair-primary-20261105-v1/primary-implementation-freeze-v1.json"
            ),
            "sha256": "cefb1739ea39637fbdc80ed7b32c75e4be922377bf974a6347f573431cf3cebe",
        },
        "primary_completion": {
            "path": (
                "reports/ssd-continuous-research-v1/"
                "distinct-opinion-pair-primary-20261105-v1/completion-v1.json"
            ),
            "sha256": "c4e7f39a5fe8af8a6aef64de5121973a446b961cf0c02aa0ffc92aa79b765e1a",
        },
        "cached_same_nine_results": {
            "path": "evidence/lagged_spy_price_impact_expert_inclusion_20261105_results_v1.json",
            "sha256": "5e0a59929a2e4166c0227478a4e1d9fcb1a41eb3a97b42ebdbc2a7c50961952b",
        },
        "original_financial_protocol": {
            "path": "config/dual-benchmark-v1.json",
            "sha256": "1b5b2572089ffa53746b01e72ff8303e00d6b5e5e3f815844f93834d0521e469",
        },
        "prior_journal_sequence": 1062,
        "prior_journal_head_sha256": (
            "ee118c53ed06e85f2df9a35116a5446e0a73c9a655742f21b587921e63dd298a"
        ),
        "prior_completed_configurations": 267,
        "candidate_count": 1,
        "static_target_packet_schema": 2,
        "new_source_requests": 0,
        "new_shadow_account_runs": 0,
        "global_pool_changed": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    if policy != expected or any(type(policy[k]) is not type(v) for k, v in expected.items()):
        raise QuantError("Keep the same frozen primary selector and original financial protocol.")
    for key in (
        "primary_policy",
        "primary_freeze",
        "primary_completion",
        "cached_same_nine_results",
        "original_financial_protocol",
    ):
        base.checked_reference(policy[key])
    primary = read_json(base.checked_reference(policy["primary_freeze"]))
    for ref in primary["implementation_and_test_inputs"] + primary["complete_dependency_inputs"]:
        base.checked_reference(ref)
    if (
        primary["actual_registered"] is not False
        or primary["real_new_policy_targets_scores_accounts_or_financial_outcomes_computed"]
        is not False
    ):
        raise QuantError("Primary preparation is not a prior real historical target run.")
    rules.validate(read_json(rules.POLICY))
    return policy


def source_inputs():
    policy = check_primary()
    data, experts, parent, refs, clocks, benchmarks = cached.source_inputs()
    closed = read_json(base.checked_reference(policy["primary_completion"]))
    refs = unique_refs(
        [
            *refs,
            *closed["exact_evidence_references"],
            *[policy[k] for k in ("primary_policy", "primary_freeze", "primary_completion")],
            policy["cached_same_nine_results"],
            reference(cached.INTENT),
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    return data, experts, parent, refs, clocks, benchmarks


def validation_refs():
    proof = read_json(FOLDER / "financial-validation-source-v1.json")
    suites = ElementTree.parse(FOLDER / "financial-tests-v1.xml").getroot().findall(".//testsuite")
    if (
        proof["implementation_and_test_inputs"] != [reference(ROOT / p) for p in OWN]
        or proof["exit_code"] != 0
        or proof["ruff_returncode"] != 0
        or not suites
        or proof["passed_tests"] != sum(int(s.get("tests", 0)) for s in suites)
        or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
    ):
        raise QuantError("Bind the exact new single-policy proof, fallback and real-packet checks.")
    return [
        reference(FOLDER / p)
        for p in (
            "financial-validation-source-v1.json",
            "financial-tests-v1.xml",
            "financial-tests-v1.log",
            "financial-ruff-v1.log",
        )
    ]


def initialize():
    lifecycle.runtime_guard()
    if INTENT.exists() or (FOLDER / "freeze-attempt-v1.json").exists():
        raise QuantError("Preserve the complete or partial first financial freeze.")
    with lifecycle.research_job():
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, _, parent, refs, clocks, _ = source_inputs()
        policy = read_json(POLICY)
        counts = current_counts(base.journal_events(base.JOURNAL))
        if (
            counts["current_journal_event_count"] != 1062
            or counts["current_journal_head_sha256"] != policy["prior_journal_head_sha256"]
            or counts["total_legacy_plus_ssd_completed_configurations"] != 267
        ):
            raise QuantError("Preserve the actual J1062/267 pre-outcome baseline.")
        deps = dependencies()
        frozen = {
            **parent,
            "study_id": policy["study_id"],
            "policy": reference(rules.POLICY),
            "financial_program_policy": reference(POLICY),
            "expert_pool": reference(rules.source.POLICY),
            "construction": "original_rolling_pair_with_exact_current_opinion_eligibility",
            "prior_journal_sequence": 1062,
            "prior_journal_head_sha256": policy["prior_journal_head_sha256"],
            "prior_completed_configurations": 267,
            "code": unique_refs(
                [
                    *parent["code"],
                    *[r for r in [*deps, *refs] if r["path"].startswith(("src/", "tests/"))],
                    reference(ROOT / "DISTINCT_OPINION_PAIR_FINANCIAL_V1.md"),
                    reference(ROOT / "EXPERT_DISTINCT_OPINION_PAIR_V1.md"),
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
                    reference(FOLDER / "baseline-v1.json"),
                    reference(FOLDER / "design-decision-v1.json"),
                    reference(FOLDER / "actual-invocation-v1.json"),
                ]
            ),
            "validation_evidence": unique_refs(
                [*parent["validation_evidence"], *validation_refs()]
            ),
            "source_policy_and_expert_admission_clocks": clocks,
            "new_shadow_account_runs": 0,
            "new_source_requests": 0,
            "new_RF_or_source_scope_waiver": False,
        }
        check_frozen(frozen)
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "frozen": frozen,
                "complete_dependency_inputs": deps,
                "source_references": refs,
                "package_versions": sorted(
                    (d.metadata["Name"], d.version) for d in distributions()
                ),
                "real_new_policy_scores_targets_or_funds_computed": False,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="actual_register_before_real_new_targets")
    return reference(INTENT)


def inputs():
    lifecycle.runtime_guard()
    intent = read_json(INTENT)
    for ref in intent["source_references"] + intent["complete_dependency_inputs"]:
        base.checked_reference(ref)
    if intent["package_versions"] != [
        list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())
    ]:
        raise QuantError("The exact accepted environment changed after freezing.")
    data, experts, _, refs, clocks, benchmarks = source_inputs()
    if (
        refs != intent["source_references"]
        or clocks != intent["frozen"]["source_policy_and_expert_admission_clocks"]
    ):
        raise QuantError("Keep the exact admitted nine sources, RF and availability clocks.")
    frozen = {
        **intent["frozen"],
        "prior_evidence": unique_refs([*intent["frozen"]["prior_evidence"], reference(INTENT)]),
    }
    check_frozen(frozen)
    return data, experts, frozen, benchmarks


def verify_retained(path, data, experts, targets, audits):
    dates = rules.source.decision_sessions(data.close.index, read_json(rules.source.POLICY))
    if (
        not targets.index.equals(data.close.index)
        or not targets.columns.equals(data.close.columns)
        or not targets.dropna(how="all").index.equals(dates)
        or [a["decision_session"] for a in audits] != dates.strftime("%Y-%m-%d").tolist()
    ):
        raise QuantError("Keep every original information date and complete target coordinate.")
    previous, equal = np.full(9, 1 / 9), np.full(9, 1 / 9)
    max_target = max_weight = max_score = 0.0
    checked_values = 0
    with path.open() as stream:
        for index, (day, audit) in enumerate(zip(dates, audits, strict=True)):
            before, after = json.loads(stream.readline()), json.loads(stream.readline())
            observed = [e.observe(day, 126) for e in experts]
            if (
                before["record"] != "decision_input"
                or before["decision_index"] != index
                or before["as_of"] != str(day.date())
                or before["information_clock"] != rules.CLOCK
                or before["previous_weights_binary64_hex"] != [float(v).hex() for v in previous]
                or len(before["observations"]) != 9
                or audit["previous_rolling_expert_weights"] != previous.tolist()
                or audit["decision_information_clock"] != rules.CLOCK
                or audit["shadow_wealth_added_to_main_capital"] is not False
                or audit["source_feedback_floor_transferred_to_main_account"] is not False
                or after["record"] != "decision_resolved"
                or after["decision_index"] != index
                or after["audit"] != audit
                or after["asset_order"] != list(targets.columns)
            ):
                raise QuantError(
                    "Retained operands, actual previous state or capital separation changed."
                )
            for i, (row, o) in enumerate(zip(before["observations"], observed, strict=True)):
                if (
                    row["id"] != rules.EXPERT_IDS[i]
                    or row["mechanism_group"] != rules.MECHANISMS[i]
                    or row["introduced_at"] != o.identity.introduced_at
                    or row["target_known_at"] != str(o.target_known_at.date())
                    or o.target_known_at > day
                    or row["asset_order"] != list(targets.columns)
                    or row["target_binary64_hex"] != [float(v).hex() for v in o.target]
                    or row["return_dates"] != o.net_returns.index.strftime("%Y-%m-%d").tolist()
                    or row["NET_returns_binary64_hex"] != [float(v).hex() for v in o.net_returns]
                    or row["RF_binary64_hex"] != [float(v).hex() for v in o.risk_free]
                    or row["score_mature"] != o.score_mature
                ):
                    raise QuantError(
                        "Retain every exact source target, mature NET/RF and known-at clock."
                    )
                checked_values += len(o.net_returns) + len(o.risk_free)
            selection = audit["selection"]
            eligibility = {
                tuple(sorted((rules.EXPERT_IDS[i], rules.EXPERT_IDS[j]))): (
                    tuple(float(v) for v in observed[i].target)
                    != tuple(float(v) for v in observed[j].target)
                )
                for i, j in itertools.combinations(range(9), 2)
            }
            expected_flags = [
                {"pair": list(pair), "exact_same_current_opinion": not allowed, "eligible": allowed}
                for pair, allowed in sorted(eligibility.items())
            ]
            mature = all(o.score_mature for o in observed)
            if (
                selection["current_opinion_pair_eligibility"] != expected_flags
                or selection["NET_history_mature"] != mature
                or selection["source_experts_invalidated"] is not False
                or selection["current_opinion_difference_is_independent_risk"] is not False
                or selection["safety_or_success_claimed"] is not False
            ):
                raise QuantError(
                    "Exact current-opinion eligibility is not a risk or source-validity gate."
                )
            if not mature:
                if (
                    selection["pairs"] != []
                    or selection["selected_pair"] is not None
                    or selection["scores_available"] is not False
                    or selection["reason"] != "unscored_initial_equal_weights"
                ):
                    raise QuantError("An immature decision cannot acquire a successful score.")
                chosen = equal.copy()
            else:
                known = observed[0].net_returns.index
                values = [(o.net_returns - o.risk_free).tolist() for o in observed]
                means = [math.fsum(v[:84]) / 84 for v in values]
                deviations = [
                    math.sqrt(math.fsum((x - m) ** 2 for x in v[:84]) / 83)
                    for v, m in zip(values, means, strict=True)
                ]
                if (
                    len(known) != 126
                    or known[-1] > day
                    or len(selection["pairs"]) != 36
                    or {tuple(row["pair"]) for row in selection["pairs"]} != set(eligibility)
                ):
                    raise QuantError(
                        "Keep all36mature pair records and chronological source inputs."
                    )
                for pair in selection["pairs"]:
                    names = tuple(pair["pair"])
                    i, j = [rules.EXPERT_IDS.index(name) for name in names]
                    if (
                        pair["eligible_by_current_opinion"] != eligibility[names]
                        or pair["exact_same_current_opinion"] == eligibility[names]
                    ):
                        raise QuantError("A pair's exact current-opinion flag changed.")
                    reason = (
                        "same_mechanism_group"
                        if observed[i].identity.mechanism_group
                        == observed[j].identity.mechanism_group
                        else "zero_training_variance"
                        if min(deviations[i], deviations[j]) <= 0
                        else None
                    )
                    weights = np.zeros(9)
                    if reason is None:
                        weights[i], weights[j] = (
                            deviations[j] / (deviations[i] + deviations[j]),
                            deviations[i] / (deviations[i] + deviations[j]),
                        )
                        portfolio = [
                            math.fsum(weights[k] * values[k][t] for k in range(9))
                            for t in range(126)
                        ]
                        if (
                            np.std(portfolio[:84], ddof=1) <= 0
                            or np.std(portfolio[84:], ddof=1) <= 0
                        ):
                            reason = "nonpositive_or_nonfinite_pair_risk"
                    if reason is not None:
                        if pair.get("unscorable_reason") != reason:
                            raise QuantError("Retain the original explicit unscorable pair reason.")
                        continue
                    actual = np.asarray(pair["expert_weights"])
                    if (
                        actual.shape != (9,)
                        or not np.isfinite(actual).all()
                        or "unscorable_reason" in pair
                    ):
                        raise QuantError(
                            "A scored pair needs complete finite original ERC weights."
                        )
                    max_weight = max(max_weight, float(max(abs(actual - weights))))
                    covariance = (
                        math.fsum(
                            (values[i][t] - means[i]) * (values[j][t] - means[j]) for t in range(84)
                        )
                        / 83
                    )
                    contribution = np.zeros(9)
                    contribution[i] = weights[i] * (
                        weights[i] * deviations[i] ** 2 + weights[j] * covariance
                    )
                    contribution[j] = weights[j] * (
                        weights[j] * deviations[j] ** 2 + weights[i] * covariance
                    )
                    if (
                        abs(
                            pair["training_correlation"]
                            - covariance / (deviations[i] * deviations[j])
                        )
                        > 1e-10
                        or max(
                            abs(
                                np.asarray(pair["training_marginal_variance_contributions"])
                                - contribution
                            )
                        )
                        > 1e-12
                    ):
                        raise QuantError("Independent source correlation or marginal risk differs.")
                    change = [
                        abs(
                            math.fsum(
                                (actual[k] - previous[k]) * float(observed[k].target[s])
                                for k in range(9)
                            )
                        )
                        for s in data.close.columns
                    ]
                    cost = math.fsum(change) * 5 / 10000 + sum(v > 1e-12 for v in change) / 10000
                    scores = [
                        scalar_score(portfolio[:84], cost),
                        scalar_score(portfolio[84:], cost),
                    ]
                    max_score = max(
                        max_score,
                        abs(scores[0] - pair["train_net_risk_score"]),
                        abs(scores[1] - pair["validation_net_risk_score"]),
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
                        raise QuantError(
                            "The unchanged original score or nominal switch proxy differs."
                        )
                scored = [p for p in selection["pairs"] if "selection_score_key" in p]
                original = (
                    min(scored, key=lambda p: (-p["selection_score_key"], p["pair"]))
                    if scored
                    else None
                )
                eligible = [p for p in scored if p["eligible_by_current_opinion"]]
                if (
                    selection["same_state_original_catalogue_selected_pair"]
                    != (original["pair"] if original else None)
                    or selection["same_state_original_catalogue_scores_available"] != bool(scored)
                    or selection["eligible_scorable_pairs"] != len(eligible)
                ):
                    raise QuantError(
                        "Saved same-state catalogue and eligible-score availability differ."
                    )
                if scored and [
                    selection[k]
                    for k in ("train_start", "train_end", "validation_start", "validation_end")
                ] != [str(known[i].date()) for i in (0, 83, 84, 125)]:
                    raise QuantError("Preserve the original chronological84/42mature sample.")
                if eligible:
                    best = min(eligible, key=lambda p: (-p["selection_score_key"], p["pair"]))
                    if (
                        selection["selected_pair"] != best["pair"]
                        or selection["scores_available"] is not True
                        or selection["reason"]
                        != "original_scored_pair_with_exact_current_opinion_distinction"
                    ):
                        raise QuantError("Eligible original-key lexical winner differs.")
                    chosen = np.asarray(best["expert_weights"])
                else:
                    reason = (
                        "no_distinct_current_opinion_pair_equal_weights"
                        if not any(eligibility.values())
                        else "no_scorable_distinct_current_opinion_pair_equal_weights"
                    )
                    if (
                        selection["selected_pair"] is not None
                        or selection["scores_available"] is not False
                        or selection["reason"] != reason
                    ):
                        raise QuantError(
                            "No-eligible-score fallback cannot be success or missing fill."
                        )
                    chosen = equal.copy()
            if not np.allclose(chosen, audit["rolling_expert_weights"], rtol=0, atol=1e-12):
                raise QuantError(
                    "Issued expert weights differ from their independently checked choice."
                )
            actual_target = targets.loc[day]
            if after["complete_target_binary64_hex"] != [float(v).hex() for v in actual_target]:
                raise QuantError("Retained resolved target differs from the actual output.")
            expected = [
                float(
                    sum(
                        Fraction.from_float(float(chosen[i]))
                        * Fraction.from_float(float(o.target[s]))
                        for i, o in enumerate(observed)
                    )
                )
                for s in data.close.columns
            ]
            max_target = max(max_target, float(max(abs(actual_target - expected))))
            previous = chosen
        if stream.readline():
            raise QuantError("Unexpected extra or partial retained decision records.")
    if max_target > 1e-10 or max_weight > 1e-10 or max_score > 1e-8:
        raise QuantError("Original independent target/ERC/score tolerances failed.")
    return {
        "verified_decisions": len(audits),
        "actual_input_resolved_records": 2 * len(audits),
        "roundtrip_NET_RF_binary64_values_checked": checked_values,
        "all36_mature_catalogues_and_exact_eligibility_verified": True,
        "maximum_independent_target_difference": max_target,
        "maximum_independent_expert_weight_difference": max_weight,
        "maximum_independent_net_score_difference": max_score,
        "original_source_signal_recomputed": False,
    }


def prepare(data, experts, frozen):
    if PREPARED.exists():
        saved = read_json(PREPARED)
        if saved["frozen_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting prior distinct-opinion targets.")
        for ref in saved["prepared"].values():
            base.checked_reference(ref)
        return saved["prepared"]
    if len(events("registered")) != 1 or (FOLDER / "preparation-attempt-v1.json").exists():
        raise QuantError("Register once before new targets and preserve partial preparations.")
    write_new_receipt(FOLDER / "preparation-attempt-v1.json", {"started_at": utc_now()})
    log = FOLDER / "actual-decision-inputs-resolved-v1.jsonl"
    with log.open("x") as stream:

        def retain(row):
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        frames, audits = rules.build_targets(
            experts,
            data.close.index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            before_decision=lifecycle.runtime_guard,
            retain=retain,
        )
    if set(frames) != {ID}:
        raise QuantError("This program has exactly one new complete policy.")
    targets = frames[ID]
    verified = verify_retained(log, data, experts, targets, audits)
    if len(audits) != 121:
        raise QuantError("Retain every original capital/month-end decision.")
    prefixes = []
    for position in (0, 6, 12, 60, 120):
        day = pd.Timestamp(audits[position]["decision_session"])
        early, earlier = rules.build_targets(
            experts,
            data.close.loc[:day].index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            before_decision=lifecycle.runtime_guard,
        )
        if not early[ID].equals(targets.loc[:day]) or earlier != audits[: position + 1]:
            raise QuantError("Future inputs changed a previous distinct-opinion choice.")
        prefixes.append(str(day.date()))
    active, coverage = targets.dropna(how="all"), []
    for window in read_json(base.PROTOCOL)["windows"]:
        anchor = previous_session(window["start"])
        known = active.loc[:anchor].tail(1)
        rows = pd.concat([known, active.loc[window["start"] : window["end"]]])
        factors = [
            s for s in ("MTUM", "VLUE", "QUAL", "USMV", "IJR", "PKW") if (rows[s] > 1e-10).any()
        ]
        if known.empty or len(factors) < 3:
            raise QuantError("Both original funded horizons need known anchors and actual factors.")
        coverage.append({"years": window["years"], "actual_factor_symbols": factors})
    directory = FOLDER / "targets"
    new_output_directory(directory)
    write_text_atomic(directory / "targets.csv", targets.to_csv(float_format="%.17g"))
    restored = pd.read_csv(
        directory / "targets.csv", index_col=0, parse_dates=True, float_precision="round_trip"
    )
    if not restored.equals(targets):
        raise QuantError("Saved targets must round-trip every binary64value exactly.")
    write_new_receipt(FOLDER / "information-audits-v1.json", {"decisions": audits})
    write_new_receipt(
        directory / "proof.json",
        {
            "checked_at": utc_now(),
            "frozen_specification_sha256": digest_json(frozen),
            **verified,
            "prefix_truncations": prefixes,
            "formal_factor_coverage": coverage,
            "actual_input_resolved_records_reference": reference(log),
            "candidate_accounts_or_metrics_computed": False,
            "original_source_signal_recomputed": False,
            "source_shadow_or_control_account_replayed": False,
            "independent_forward_validation": False,
        },
    )
    prepared = {
        "targets": reference(directory / "targets.csv"),
        "proof": reference(directory / "proof.json"),
    }
    write_new_receipt(PREPARED, {"frozen_sha256": digest_json(frozen), "prepared": prepared})
    return prepared


def register():
    return bound(financial.register)()


def publication(frozen, commit):
    return bound(financial.publication)(frozen, commit)


def record_failure(stage, error):
    return bound(financial.record_failure)(stage, error)


def evaluate_path(*args):
    return bound(financial.evaluate_path)(*args)


def evaluate(commit):
    return bound(financial.evaluate)(commit)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(RESULTS)
        if set(saved["paths"]) != base.PATH_KEYS:
            raise QuantError("Export all four original paths, not passing fragments.")
        controls = read_json(base.checked_reference(read_json(POLICY)["cached_same_nine_results"]))
        source.common.retain_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "study_id": read_json(POLICY)["study_id"],
                "candidate_id": ID,
                "completed_at": saved["completed_at"],
                "status": saved["review"]["status"],
                "paths": saved["paths"],
                "matched_cached_same_nine_controls": controls["candidates"],
                "actual_authoritative_counts": current_counts(base.journal_events(base.JOURNAL)),
                "original_source_pool_and_financial_rejections_unchanged": True,
                "old_controls_or_source_accounts_replayed": False,
                "raw_prices_targets_or_daily_rows_exported": False,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
    return reference(PUBLIC)


def main():
    parser = argparse.ArgumentParser(
        description="Frozen single distinct-opinion pair financial study."
    )
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use an actual remotely published exact preregistration SHA.")
    actions = {"freeze": initialize, "register": register, "export": export}
    print(
        json.dumps(evaluate(args.commit) if args.action == "evaluate" else actions[args.action]())
    )


if __name__ == "__main__":
    main()
