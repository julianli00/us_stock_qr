from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from importlib.metadata import version
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import expert_initial_monthly_execution_v1 as execution
from us_quant import ssd_four_expert_meta_v1 as previous
from us_quant import strategy_expert_pool_v3 as rules
from us_quant.calendar import previous_session
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.dual_benchmark_v1 import research_job, runtime_guard
from us_quant.research_program import review_market
from us_quant.research_weekly import verify_publication
from us_quant.ssd_research_v1 import append_event, matching_events, published_inputs, reference
from us_quant.storage import (
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_text_atomic,
)
from us_quant.strategy_experts_v1 import (
    ArchivedDenseExpert,
    ArchivedMonthlyExpert,
    ExpertIdentity,
)

base, common, current, ROOT = previous.base, previous.common, previous.current, previous.ROOT
FOLDER = base.REPORTS / "five-expert-meta-20261030-v1"
INTENT = FOLDER / "research-intent-v1.json"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
TARGETS = ROOT / "data/ssd-five-expert-meta-20261030-v1"
PREPARED = FOLDER / "prepared-target-bundle.json"
PACKET = ROOT / "evidence/five_expert_meta_20261030_registration_v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC_RESULTS = ROOT / "evidence/five_expert_meta_20261030_results_v1.json"
IDS = tuple(c["id"] for c in rules.CANDIDATES)
OWN = (
    "config/strategy-expert-pool-five-v1.json",
    "src/us_quant/strategy_expert_pool_v3.py",
    "src/us_quant/ssd_five_expert_meta_v1.py",
    "tests/test_strategy_expert_pool_v3.py",
    "tests/test_ssd_five_expert_meta_v1.py",
)
PARENT_SPEC = "9543c88c522558ec470f0b7ca3154d37408c083ff475a508e648eb3f8689a62b"


def frame(ref):
    return pd.read_csv(
        base.checked_reference(ref), index_col=0, parse_dates=True, float_precision="round_trip"
    )


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    packet = read_json(base.checked_reference(policy["source_registration"]))
    parent = packet["entries"][0]["specification"]["frozen"]
    if (
        digest_json(parent) != PARENT_SPEC
        or policy["source_specification_sha256"] != PARENT_SPEC
        or parent["python_version"] != list(sys.version_info[:3])
        or any(version(k) != v for k, v in parent["package_versions"].items())
        or parent["policy"] != policy["source_scoring_policy"]
    ):
        raise QuantError("Use the unchanged original four-expert sources and environment.")
    previous.check_frozen(parent)
    previous.checked_packet(parent)
    old_results = read_json(base.checked_reference(policy["previous_four_results"]))
    if (
        old_results["frozen_specification_sha256"] != PARENT_SPEC
        or set(old_results["candidates"]) != set(previous.IDS)
        or any(
            r["status"] not in {"rejected", "historical_point_gates_passed"}
            for r in old_results["candidates"].values()
        )
    ):
        raise QuantError("The previous complete controls cannot be replaced or reclassified.")
    original_policy = read_json(base.checked_reference(policy["source_scoring_policy"]))
    previous.rules.validate(original_policy)
    if any(
        policy[k] != original_policy[k]
        for k in (
            "first_pool_admission",
            "feedback_expert_admission",
            "lookback_sessions",
            "training_sessions",
            "validation_sessions",
            "selected_experts",
            "switch_cost_bps_per_side",
            "switch_commission_per_changed_asset_usd",
            "switch_cost_amortization_sessions",
            "initial_decision_session",
            "actual_weight_trade_band",
        )
    ):
        raise QuantError("Pool inclusion must not silently retune the existing scoring law.")
    pool, feedback, macro, public_macro = (
        read_json(base.checked_reference(policy[k]))
        for k in (
            "first_pool_admission",
            "feedback_expert_admission",
            "profit_expert_admission",
            "profit_public_admission",
        )
    )
    selected = [
        r for r in pool["admission_checks"] if r["first_pool_status"] == "admitted_selected"
    ]
    record = macro["expert"]
    if (
        tuple(r["id"] for r in selected) != rules.EXPERT_IDS[:3]
        or any(
            r["source_technical_and_funding_checks_passed"] is not True
            or r["four_completed_paths_checked"] != 4
            for r in selected
        )
        or feedback["expert"]["id"] != rules.EXPERT_IDS[3]
        or record["id"] != rules.EXPERT_IDS[4]
        or record["mechanism_group"] != rules.MECHANISMS[4]
        or macro["source_inputs_checked"] != 557
        or record["source_market_12_assets_and_RF_match_admitted_14"] is not True
        or record["existing_raw_snapshot_selector_and_funding_proofs_reused"] is not True
        or macro["new_strategy_evaluations"] != 0
        or macro["new_shadow_account_runs"] != 0
        or macro["expanded_pool_or_next_meta_rule_frozen"] is not False
        or public_macro["private_admission"] != policy["profit_expert_admission"]
        or public_macro["admitted_at"] != record["actual_expert_admitted_at"]
        or macro["old_ledger_sha256_unchanged"] != base.LEGACY_SHA256
    ):
        raise QuantError("Reuse actual technical admissions, not a financial-winner filter.")
    data = review_market({"market": parent["market"]}, ROOT)
    assets = tuple(data.close.columns)
    adapters = {"dense": ArchivedDenseExpert, "monthly": ArchivedMonthlyExpert}
    experts, refs = (
        [],
        [
            policy[k]
            for k in (
                "source_registration",
                "previous_four_results",
                "source_scoring_policy",
                "first_pool_admission",
                "feedback_expert_admission",
                "profit_expert_admission",
                "profit_public_admission",
            )
        ],
    )
    for r in selected:
        experts.append(
            adapters[r["adapter"]](
                ExpertIdentity(r["id"], r["mechanism_group"], r["introduced_at"]),
                frame(r["target_reference"]),
                frame(r["shadow_reference"]),
                assets,
            )
        )
        refs.extend((r["target_reference"], r["shadow_reference"]))
    r = feedback["expert"]
    experts.append(
        ArchivedDenseExpert(
            ExpertIdentity(r["id"], r["mechanism_group"], r["introduced_at"]),
            frame(r["targets"]),
            frame(r["shadow"]),
            assets,
        )
    )
    refs.extend((r["targets"], r["shadow"]))
    experts.append(
        ArchivedMonthlyExpert(
            ExpertIdentity(record["id"], record["mechanism_group"], record["introduced_at"]),
            frame(record["target_reference"]),
            frame(record["shadow_reference"]),
            assets,
        )
    )
    refs.extend((record["target_reference"], record["shadow_reference"], macro["intent"]))
    admission_intent = read_json(base.checked_reference(macro["intent"]))
    admission_refs = [
        *admission_intent["source_inputs"],
        *admission_intent["implementation_and_test_inputs"],
    ]
    for ref in admission_refs:
        base.checked_reference(ref)
    refs.extend(admission_refs)
    if tuple(e.identity.identifier for e in experts) != rules.EXPERT_IDS:
        raise QuantError("The fixed expert vector cannot change with lexical pair ordering.")
    for expert in experts:
        initial = expert.observe(pd.Timestamp(policy["initial_decision_session"]), 126)
        final = expert.observe(data.close.index[-1], 2512)
        if (
            initial.target is None
            or initial.score_mature
            or len(initial.net_returns)
            or len(final.net_returns) != 2512
            or not final.risk_free.equals(data.risk_free.loc[final.risk_free.index])
        ):
            raise QuantError("Keep genuine availability, empty initial scores and original RF.")
    clocks = [
        *parent["source_policy_and_expert_admission_clocks"],
        {
            "id": record["id"],
            "source_policy_registered_at": record["source_policy_registered_at"],
            "actual_expert_admitted_at": record["actual_expert_admitted_at"],
        },
    ]
    if file_digest(ROOT / "runtime/research-program.sqlite3") != base.LEGACY_SHA256:
        raise QuantError("The immutable original ledger changed.")
    benchmarks = read_json(base.BENCHMARKS / "results.json")
    if (
        benchmarks["protocol_sha256"] != parent["protocol"]["sha256"]
        or benchmarks["benchmark_accounts"] != 8
        or benchmarks["new_strategy_evaluations"] != 0
    ):
        raise QuantError("Reuse the original eight funded SPY/QQQ benchmarks.")
    return (
        data,
        tuple(experts),
        policy,
        parent,
        previous.source.unique_refs(refs),
        clocks,
        admission_refs,
        benchmarks,
    )


def validation_references():
    path = FOLDER / "offline-study-tests-v1.xml"
    suites = ElementTree.parse(path).getroot().findall(".//testsuite")
    names = {
        case.get("name", "").split("[", 1)[0]
        for suite in suites
        for case in suite.findall("testcase")
    }
    required = {
        "test_unchanged_pair_law_accepts_five_without_retuning",
        "test_five_experts_use_ten_pairs_and_keep_one_capital_target",
        "test_actual_interface_prefixes_keep_initial_and_maturity",
        "test_future_target_and_shadow_changes_do_not_change_past",
        "test_scalar_oracle_checks_all_ten_pairs_and_vector_order",
        "test_scalar_oracle_rejects_nonselected_pair_risk_or_cost_changes",
        "test_cached_admission_loader_never_invokes_historical_entrypoints",
        "test_publication_checks_full_dependencies_before_explicit_packet_and_fence",
        "test_failed_publication_cannot_start_any_main_account",
        "test_canonical_export_preserves_all_paths_without_daily_rows",
        "test_original_monthly_execution_is_identical_for_valid_monthly_targets",
    }
    if (
        not required.issubset(names)
        or sum(int(s.get("tests", 0)) for s in suites) < len(required)
        or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
    ):
        raise QuantError("Require completed causal, scalar, cached-Adapter and publication tests.")
    return [reference(path), reference(FOLDER / "offline-study-tests-v1.log")]


def initialize():
    runtime_guard()
    common.output_guard(INTENT)
    if INTENT.exists():
        raise QuantError("Preserve the complete or partial implementation freeze, not repeat it.")
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        _, _, policy, parent, refs, clocks, _, _ = source_inputs()
        events = base.journal_events(base.JOURNAL)
        counts = current.current_counts(events)
        if (
            len(events) != 432
            or events[-1]["sha256"] != rules.PRIOR_HEAD
            or counts["new_registered_candidates"] != 39
            or counts["new_completed_strategy_evaluations"] != 36
            or counts["new_technical_failures"] != 3
            or counts["new_pending_candidates"] != 0
        ):
            raise QuantError("Freeze against the actual unchanged post-admission history.")
        design, invocation = read_json(DESIGN), read_json(INVOCATION)
        if (
            design["actual_continuation_event"] != 39243
            or design["invocation"] != reference(INVOCATION)
            or design["expert_ids"] != list(rules.EXPERT_IDS)
            or invocation["data"]["source"] != "system"
            or invocation["data"]["delivery"] != "idle"
        ):
            raise QuantError("Preserve the actual design and scheduled invocation origin.")
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "actual_continuation_event": 39243,
                "actual_delivery": "system_idle",
                "design": reference(DESIGN),
                "actual_invocation": reference(INVOCATION),
                "model_and_test_files": [reference(ROOT / p) for p in OWN],
                "validation_evidence": validation_references(),
                "source_references": refs,
                "source_specification_sha256": digest_json(parent),
                "source_policy_and_expert_admission_clocks": clocks,
                "question": policy["methodology"]["question"],
                "real_new_meta_scores_or_targets_computed": False,
                "candidate_accounts_or_metrics_computed": False,
                "source_expert_targets_fits_or_accounts_replayed": False,
                "new_shadow_account_runs": 0,
                "new_source_requests": 0,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        current.progress(
            "five_expert_rule_frozen",
            active_compute_job=None,
            five_expert_intent=reference(INTENT),
            pending_steps=["register_five_expert_rules_before_new_scores_or_targets"],
        )
        return reference(INTENT)


def inputs():
    data, experts, policy, parent, refs, clocks, admission_refs, benchmarks = source_inputs()
    intent = read_json(INTENT)
    if (
        intent["model_and_test_files"] != [reference(ROOT / p) for p in OWN]
        or intent["validation_evidence"] != validation_references()
        or intent["source_references"] != refs
        or intent["source_specification_sha256"] != digest_json(parent)
        or intent["source_policy_and_expert_admission_clocks"] != clocks
        or intent["design"] != reference(DESIGN)
        or intent["actual_invocation"] != reference(INVOCATION)
        or intent["real_new_meta_scores_or_targets_computed"] is not False
        or intent["candidate_accounts_or_metrics_computed"] is not False
    ):
        raise QuantError("The actual pre-outcome implementation freeze changed.")
    frozen = {
        **parent,
        "study_id": policy["study_id"],
        "policy": reference(rules.POLICY),
        "code": previous.source.unique_refs(
            [
                *parent["code"],
                *[r for r in admission_refs if r["path"].startswith(("src/", "tests/"))],
                *[reference(ROOT / p) for p in OWN[1:]],
            ]
        ),
        "policies": previous.source.unique_refs(
            [
                *parent["policies"],
                *[r for r in admission_refs if r["path"].startswith("config/")],
                reference(rules.POLICY),
            ]
        ),
        "source_four_expert_pool_policy": parent["policy"],
        "expert_pool": reference(rules.POLICY),
        "expert_pool_definition_role": "fixed_five_representatives_and_cached_admissions",
        "source_policy_and_expert_admission_clocks": clocks,
        "prior_completed_configurations": 214,
        "prior_journal_sequence": 432,
        "prior_journal_head_sha256": rules.PRIOR_HEAD,
        "construction": "five_complete_experts_including_macro_same_original_pair_law",
        "execution_adapter": reference(
            ROOT / "src/us_quant/expert_initial_monthly_execution_v1.py"
        ),
        "execution_adapter_role": "unchanged_initial_plus_monthly_actual_holdings_025_band",
        "prior_evidence": previous.source.unique_refs(
            [
                *parent["prior_evidence"],
                *refs,
                reference(DESIGN),
                reference(INVOCATION),
                reference(INTENT),
            ]
        ),
        "validation_evidence": previous.source.unique_refs(
            [*parent["validation_evidence"], *validation_references()]
        ),
        "new_shadow_account_runs": 0,
        "new_shadow_recommendation_artifacts": 0,
        "new_source_requests": 0,
    }
    previous.check_frozen(frozen)
    return data, experts, frozen, benchmarks


def verify_targets(data, experts, targets, audits):
    dates = rules.decision_sessions(data.close.index, read_json(rules.POLICY))
    if (
        set(targets) != set(IDS)
        or [a["decision_session"] for a in audits] != dates.strftime("%Y-%m-%d").tolist()
        or any(
            not t.index.equals(data.close.index)
            or not t.columns.equals(data.close.columns)
            or not t.dropna().index.equals(dates)
            or not t.loc[~t.index.isin(dates)].isna().all().all()
            for t in targets.values()
        )
    ):
        raise QuantError("Keep the full initial/monthly target lattice and formal calendar.")
    previous_weights = np.full(5, 0.2)
    maximum_target = maximum_weight = maximum_score = 0.0
    for position, (day, audit) in enumerate(zip(dates, audits, strict=True)):
        observed = [e.observe(day, 126) for e in experts]
        if (
            tuple(o.identity.identifier for o in observed) != rules.EXPERT_IDS
            or audit["previous_rolling_expert_weights"] != previous_weights.tolist()
            or audit["equal_expert_weights"] != np.full(5, 0.2).tolist()
            or audit["decision_role"]
            != ("initial_capital" if position == 0 else "completed_exchange_month_end")
            or any(
                o.target is None or o.target_known_at is None or o.target_known_at > day
                for o in observed
            )
            or audit["expert_target_known_at"] != [str(o.target_known_at.date()) for o in observed]
            or audit["preserved_archive_identity_introduced_at"]
            != [o.identity.introduced_at for o in observed]
            or audit["expert_shadow_last_observation"]
            != [
                str(o.net_returns.index[-1].date()) if len(o.net_returns) else None
                for o in observed
            ]
            or audit["shadow_wealth_added_to_main_capital"] is not False
            or audit["source_feedback_floor_transferred_to_main_account"] is not False
            or audit["counterfactual_historical_expert_availability"] is not True
            or audit["independent_forward_validation"] is not False
        ):
            raise QuantError("The source clock, expert vector or capital separation changed.")
        selection = audit["selection"]
        if not all(o.score_mature for o in observed):
            if (
                selection["reason"] != "unscored_initial_equal_weights"
                or selection["scores_available"] is not False
                or selection["pairs"] != []
                or selection["selected_pair"] is not None
                or selection["safety_or_success_claimed"] is not False
            ):
                raise QuantError("Unmatured observations cannot receive successful scores.")
            chosen = np.full(5, 0.2)
        else:
            values = [(o.net_returns - o.risk_free).tolist() for o in observed]
            means = [math.fsum(v[:84]) / 84 for v in values]
            deviations = [
                math.sqrt(math.fsum((x - m) ** 2 for x in v[:84]) / 83)
                for v, m in zip(values, means, strict=True)
            ]
            pairs = {
                tuple(sorted((rules.EXPERT_IDS[i], rules.EXPERT_IDS[j])))
                for i, j in itertools.combinations(range(5), 2)
            }
            if (
                len(selection["pairs"]) != 10
                or {tuple(p["pair"]) for p in selection["pairs"]} != pairs
            ):
                raise QuantError(
                    "Retain all ten fixed pairs, including explicit unscorable states."
                )
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
                weights = [0.0] * 5
                if reason is None:
                    weights[i] = deviations[j] / (deviations[i] + deviations[j])
                    weights[j] = deviations[i] / (deviations[i] + deviations[j])
                    portfolio = [
                        math.fsum(weights[k] * values[k][t] for k in range(5)) for t in range(126)
                    ]
                    if np.std(portfolio[:84], ddof=1) <= 0 or np.std(portfolio[84:], ddof=1) <= 0:
                        reason = "nonpositive_or_nonfinite_pair_risk"
                if reason is not None:
                    if pair.get("unscorable_reason") != reason:
                        raise QuantError("An unscorable pair differs from its independent reason.")
                    continue
                actual = np.asarray(pair["expert_weights"], dtype=float)
                contributions = np.asarray(pair["training_marginal_variance_contributions"])
                if (
                    "unscorable_reason" in pair
                    or actual.shape != (5,)
                    or contributions.shape != (5,)
                    or not np.isfinite(actual).all()
                    or not np.isfinite(contributions).all()
                    or (actual < 0).any()
                    or abs(float(actual.sum()) - 1) > 1e-12
                    or not np.isfinite(
                        [
                            pair[k]
                            for k in (
                                "train_net_risk_score",
                                "validation_net_risk_score",
                                "selection_score",
                                "selection_score_key",
                                "training_correlation",
                                "estimated_incremental_switch_cost_fraction",
                            )
                        ]
                    ).all()
                ):
                    raise QuantError("Every pair needs finite, correctly dimensioned risk records.")
                maximum_weight = max(maximum_weight, max(abs(actual - weights)))
                covariance = (
                    math.fsum(
                        (values[i][t] - means[i]) * (values[j][t] - means[j]) for t in range(84)
                    )
                    / 83
                )
                expected_contribution = np.zeros(5)
                expected_contribution[i] = weights[i] * (
                    weights[i] * deviations[i] ** 2 + weights[j] * covariance
                )
                expected_contribution[j] = weights[j] * (
                    weights[j] * deviations[j] ** 2 + weights[i] * covariance
                )
                if (
                    abs(pair["training_correlation"] - covariance / (deviations[i] * deviations[j]))
                    > 1e-10
                    or max(abs(contributions - expected_contribution)) > 1e-12
                ):
                    raise QuantError("Stored correlation or marginal risk differs from known data.")
                change = [
                    abs(
                        math.fsum(
                            (pair["expert_weights"][k] - previous_weights[k])
                            * float(observed[k].target[symbol])
                            for k in range(5)
                        )
                    )
                    for symbol in data.close.columns
                ]
                cost = math.fsum(change) * 5 / 10000 + sum(v > 1e-12 for v in change) / 10000
                scores = [
                    previous.source.scalar_score(portfolio[:84], cost),
                    previous.source.scalar_score(portfolio[84:], cost),
                ]
                maximum_score = max(
                    maximum_score,
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
                    raise QuantError("The original switch proxy or score key changed.")
            scored = [p for p in selection["pairs"] if "selection_score_key" in p]
            if scored:
                best = sorted(scored, key=lambda p: (-p["selection_score_key"], p["pair"]))[0]
                known = observed[0].net_returns.index
                if (
                    selection["selected_pair"] != best["pair"]
                    or selection["scores_available"] is not True
                    or selection["train_start"] != str(known[0].date())
                    or selection["train_end"] != str(known[83].date())
                    or selection["validation_start"] != str(known[84].date())
                    or selection["validation_end"] != str(known[-1].date())
                    or known[-1] > day
                ):
                    raise QuantError("The chronological score or chosen pair changed.")
                chosen = np.asarray(best["expert_weights"])
            else:
                if (
                    selection["reason"] != "unscored_degenerate_pair_equal_weights"
                    or selection["scores_available"] is not False
                    or selection["selected_pair"] is not None
                ):
                    raise QuantError("A degenerate unranked state cannot be hidden.")
                chosen = np.full(5, 0.2)
        if not np.allclose(chosen, audit["rolling_expert_weights"], rtol=0, atol=1e-12):
            raise QuantError("Issued expert weights differ from their actual selection.")
        for name, weights in ((IDS[0], np.full(5, 0.2)), (IDS[1], chosen)):
            expected = [
                math.fsum(float(o.target[s]) * weights[i] for i, o in enumerate(observed))
                for s in data.close.columns
            ]
            maximum_target = max(maximum_target, float(max(abs(targets[name].loc[day] - expected))))
        previous_weights = chosen
    if maximum_target > 1e-10 or maximum_weight > 1e-10 or maximum_score > 1e-8:
        raise QuantError("Independent scalar composition, risk weights or net scores disagree.")
    return {
        "maximum_independent_target_difference": maximum_target,
        "maximum_independent_expert_weight_difference": maximum_weight,
        "maximum_independent_net_score_difference": maximum_score,
    }


def prepare(data, experts, frozen):
    if PREPARED.exists():
        saved = read_json(PREPARED)
        if saved["frozen_specification_sha256"] != digest_json(frozen):
            raise QuantError("Preserve conflicting prior preparation, not regenerate targets.")
        for row in saved["entries"].values():
            for ref in row.values():
                base.checked_reference(ref)
        return saved["entries"]
    attempt = FOLDER / "preparation-attempt-v1.json"
    if attempt.exists():
        raise QuantError("Preserve the unresolved preparation instead of repeating it.")
    write_new_receipt(attempt, {"started_at": utc_now(), "frozen_sha256": digest_json(frozen)})
    targets, audits = rules.build_targets(
        experts,
        data.close.index,
        tuple(data.close.columns),
        read_json(rules.POLICY),
        before_decision=runtime_guard,
    )
    independent = verify_targets(data, experts, targets, audits)
    if len(audits) != 121:
        raise QuantError("Keep all121original initial/monthly decisions.")
    prefixes = []
    for position in (0, 6, 12, 60, 120):
        runtime_guard()
        day = pd.Timestamp(audits[position]["decision_session"])
        early, earlier = rules.build_targets(
            experts,
            data.close.loc[:day].index,
            tuple(data.close.columns),
            read_json(rules.POLICY),
            before_decision=runtime_guard,
        )
        if (
            any(not early[n].equals(targets[n].loc[:day]) for n in IDS)
            or earlier != audits[: position + 1]
        ):
            raise QuantError("Future inputs changed a prior expert composition.")
        prefixes.append(str(day.date()))
    write_new_receipt(FOLDER / "information-audits-v1.json", {"decisions": audits})
    entries = {}
    for name in IDS:
        directory = TARGETS / name
        new_output_directory(directory)
        write_text_atomic(directory / "targets.csv", targets[name].to_csv(float_format="%.17g"))
        coverage = []
        for window in read_json(base.PROTOCOL)["windows"]:
            anchor = previous_session(window["start"])
            known = targets[name].ffill().loc[anchor : window["end"]]
            actual = [s for s in previous.source.donor.rules.CORE if (known[s] > 1e-10).any()]
            if anchor not in known.index or known.isna().any().any() or len(actual) < 3:
                raise QuantError("Every formal anchor and target factor coverage is required.")
            coverage.append({"years": window["years"], "actual_factor_symbols": actual})
        write_new_receipt(
            directory / "proof.json",
            {
                "checked_at": utc_now(),
                "frozen_specification_sha256": digest_json(frozen),
                "decision_count": len(audits),
                "initial_capital_decision_is_not_a_fabricated_month_end": True,
                "unscored_initial_equal_weight_decisions": sum(
                    a["selection"]["reason"] == "unscored_initial_equal_weights" for a in audits
                ),
                "scored_decisions": sum(a["selection"]["scores_available"] for a in audits),
                "information_audits": reference(FOLDER / "information-audits-v1.json"),
                **independent,
                "prefix_truncations": prefixes,
                "formal_factor_coverage": coverage,
                "source_policy_and_expert_admission_clocks": frozen[
                    "source_policy_and_expert_admission_clocks"
                ],
                "new_shadow_account_runs": 0,
                "new_shadow_recommendation_artifacts": 0,
                "candidate_accounts_or_metrics_computed": False,
                "candidate_metrics_scope": "formal main paths, not mature source-shadow scores",
                "original_source_signal_recomputed": False,
                "independent_forward_validation": False,
            },
        )
        entries[name] = {
            "targets": reference(directory / "targets.csv"),
            "proof": reference(directory / "proof.json"),
        }
    previous.check_frozen(frozen)
    write_new_receipt(
        PREPARED,
        {
            "frozen_specification_sha256": digest_json(frozen),
            "entries": entries,
        },
    )
    return entries


def checked_packet(frozen):
    packet = read_json(PACKET)
    entries = [common.packet_entry(c, frozen) for c in rules.CANDIDATES]
    created = pd.Timestamp(packet["created_at"])
    if (
        packet["schema_version"] != 2
        or packet["program_id"] != "ssd_continuous_research_v1"
        or packet["study_id"] != frozen["study_id"]
        or packet["entries"] != entries
        or packet["new_registered_candidates"] != 2
        or created.tzinfo is None
        or pd.isna(created)
        or created > pd.Timestamp(utc_now())
        or any(pd.Timestamp(e["target_proof_at"]) > created for e in entries)
        or any(
            created > pd.Timestamp(e["at"]) for n in IDS for e in matching_events("path_started", n)
        )
        or any(
            packet.get(k) is not False
            for k in (
                "candidate_accounts_or_metrics_computed",
                "investment_objective_verified",
                "independent_forward_validation",
                "order_authority",
            )
        )
    ):
        raise QuantError("The actual complete pre-account registration packet changed.")
    return packet


def register():
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, experts, frozen, _ = inputs()
        events = base.journal_events(base.JOURNAL)
        old = [e for e in events if e["candidate_id"] not in IDS]
        if len(old) != 432 or old[-1]["sha256"] != rules.PRIOR_HEAD:
            raise QuantError("Preserve all original completed and failed financial identities.")
        if any(matching_events("technical_failure", n) for n in IDS):
            raise QuantError("A terminal registered version cannot be silently retried.")
        for candidate in rules.CANDIDATES:
            found = matching_events("registered", candidate["id"])
            body = {"candidate": candidate, "frozen": frozen}
            if found and (len(found) != 1 or found[0]["body"] != body):
                raise QuantError("A registered policy or input cannot change.")
            if not found:
                runtime_guard()
                append_event(base.JOURNAL, "registered", candidate["id"], body)
        pending = [n for n in IDS if not matching_events("target_proof", n)]
        try:
            if pending:
                current.progress(
                    "preparing_registered_five_expert_meta",
                    active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
                )
                prepared = prepare(data, experts, frozen)
                for name in pending:
                    runtime_guard()
                    append_event(
                        base.JOURNAL,
                        "target_proof",
                        name,
                        {
                            "registered_event_sha256": matching_events("registered", name)[0][
                                "sha256"
                            ],
                            **prepared[name],
                        },
                    )
            common.retain_receipt(
                PACKET,
                {
                    "schema_version": 2,
                    "program_id": "ssd_continuous_research_v1",
                    "study_id": frozen["study_id"],
                    "created_at": utc_now(),
                    "entries": [common.packet_entry(c, frozen) for c in rules.CANDIDATES],
                    "new_registered_candidates": 2,
                    "new_economic_hypotheses": 2,
                    "candidate_accounts_or_metrics_computed": False,
                    "candidate_metrics_scope": "formal main paths, not source-shadow scores",
                    "raw_inputs_and_target_matrices_remain_private": True,
                    "investment_objective_verified": False,
                    "independent_forward_validation": False,
                    "order_authority": False,
                },
                ("created_at",),
            )
            checked_packet(frozen)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            previous.source.failure("five_expert_meta_preparation", pending, error)
            raise
        except KeyboardInterrupt:
            current.progress("five_expert_preparation_interrupted", active_compute_job=None)
            raise
        current.progress(
            "five_expert_registered_waiting_publication",
            active_compute_job=None,
            five_expert_registration=reference(PACKET),
            pending_steps=["publish_exact_five_expert_packet_before_eight_accounts"],
        )
        return reference(PACKET)


def publication(frozen, commit):
    dependencies = published_inputs(frozen, commit)
    contract = read_json(base.REPO / "config/publication_allowlist.json")
    explicit = {
        r["path"]
        for r in frozen["code"] + frozen["policies"]
        if f"research/workbench/{r['path']}" in contract["files"]
    }
    own = verify_publication(ROOT, reference(PACKET), explicit, commit)
    found = [
        e for e in matching_events("publication") if e["body"].get("study_id") == frozen["study_id"]
    ]
    if found:
        if len(found) != 1 or found[0]["body"]["publication"]["published_commit"] != commit:
            raise QuantError("Resume the original publication fence, not a later patch.")
        return found[0]["body"]["publication"]
    receipt = {**own, "complete_dependency_publication": dependencies}
    append_event(
        base.JOURNAL, "publication", None, {"study_id": frozen["study_id"], "publication": receipt}
    )
    return receipt


def states(frozen):
    result = {}
    for name in IDS:
        reviews, failed, completed = (
            matching_events(k, name) for k in ("review", "technical_failure", "path_completed")
        )
        if (
            len({e["body"]["path_key"] for e in completed}) != len(completed)
            or len(reviews) > 1
            or len(failed) > 1
        ):
            raise QuantError("Duplicate completion or terminal events cannot select a winner.")
        paths = {
            e["body"]["path_key"]: base.verify_path(
                e["body"]["receipt"], frozen, name, e["body"]["path_key"]
            )
            for e in completed
        }
        if reviews:
            if (
                failed
                or set(paths) != base.PATH_KEYS
                or reviews[0]["body"]["paths"] != paths
                or reviews[0]["body"]["all_four_paths_same_candidate_and_frozen_parameters"]
                is not True
            ):
                raise QuantError("The completed review disagrees with actual path receipts.")
            result[name] = reviews[0]["body"]
        else:
            result[name] = {
                "candidate_id": name,
                "status": "technical_failure" if failed else "pending",
                "paths": paths,
                "failure_event_sha256": failed[0]["sha256"] if failed else None,
                "failure_stage": failed[0]["body"]["stage"] if failed else None,
            }
    return result


def evaluate(commit):
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        data, _, frozen, benchmarks = inputs()
        checked_packet(frozen)
        if RESULTS.exists():
            saved = read_json(RESULTS)
            if saved["frozen_specification_sha256"] != digest_json(frozen) or saved[
                "candidates"
            ] != states(frozen):
                raise QuantError("Preserve actual completed outcomes without replay.")
            return reference(RESULTS)
        active = IDS
        try:
            published = publication(frozen, commit)
            current.progress(
                "evaluating_five_expert_meta",
                active_compute_job={"pid": os.getpid(), "study_id": frozen["study_id"]},
            )
            protocol = read_json(base.PROTOCOL)
            for name in IDS:
                active = (name,)
                if matching_events("review", name) or matching_events("technical_failure", name):
                    continue
                target = frame(matching_events("target_proof", name)[0]["body"]["targets"])
                paths = {}
                for window in protocol["windows"]:
                    for scenario in protocol["accounts"]["scenarios"]:
                        runtime_guard()
                        previous.check_frozen(frozen)
                        key = f"{window['years']}y-{scenario['id']}"
                        current.progress(
                            "evaluating_five_expert_meta", active_candidate=name, active_path=key
                        )
                        paths[key] = execution.evaluate_path(
                            data, target, frozen, name, window, scenario, benchmarks
                        )
                participation = previous.actual_factor_participation(paths, frozen)
                passed = all(p["all_point_gates_pass"] for p in paths.values()) and all(
                    p["minimum_three_registered_families_pass"] for p in participation.values()
                )
                append_event(
                    base.JOURNAL,
                    "review",
                    name,
                    {
                        "candidate_id": name,
                        "reviewed_at": utc_now(),
                        "status": "historical_point_gates_passed" if passed else "rejected",
                        "paths": paths,
                        "actual_factor_participation": participation,
                        "all_four_paths_same_candidate_and_frozen_parameters": True,
                        "full_search_selection_adjusted_alpha_verified": False,
                        "independent_forward_validation": False,
                        "investment_objective_verified": False,
                        "automatic_live_deployment": False,
                        "order_authority": False,
                    },
                )
            events = base.journal_events(base.JOURNAL)
            write_new_receipt(
                RESULTS,
                {
                    "schema_version": 2,
                    "study_id": frozen["study_id"],
                    "finished_at": utc_now(),
                    "frozen_specification_sha256": digest_json(frozen),
                    "candidates": states(frozen),
                    "publication": published,
                    "legacy": frozen["legacy"],
                    "actual_journal_event_count": len(events),
                    "actual_journal_head_sha256": events[-1]["sha256"],
                    "global_counts_at_completion": common.counts(events),
                    "new_shadow_account_runs": 0,
                    "new_shadow_recommendation_artifacts": 0,
                    "investment_objective_verified": False,
                    "order_authority": False,
                },
            )
            current.progress(
                "five_expert_meta_finished",
                active_compute_job=None,
                active_candidate=None,
                active_path=None,
                five_expert_results=reference(RESULTS),
                pending_steps=["publish_all_eight_five_expert_outcomes"],
            )
            return reference(RESULTS)
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            previous.source.failure("five_expert_meta_evaluation", active, error)
            raise
        except KeyboardInterrupt:
            current.progress(
                "five_expert_meta_interrupted", active_compute_job=None, active_candidate=None
            )
            raise


def public_result():
    base.storage_guard()
    _, _, frozen, benchmarks = inputs()
    checked_packet(frozen)
    actual, saved = states(frozen), read_json(RESULTS)
    if saved["candidates"] != actual or any(v["status"] == "pending" for v in actual.values()):
        raise QuantError("Publish actual complete outcomes or retained technical failures only.")
    lookup = {
        (r["years"], r["scenario"], r["symbol"]): r["metrics"]["cagr"]
        for r in benchmarks["benchmarks"]
    }
    reviews = []
    for name, row in actual.items():
        paths = []
        for key, path in row["paths"].items():
            year, scenario = key.split("-")
            years = int(year[:-1])
            paths.append(
                {
                    **{
                        k: path[k]
                        for k in (
                            "path_key",
                            "metrics",
                            "execution",
                            "gates",
                            "all_point_gates_pass",
                            "excess_net_cagr_vs_spy",
                            "excess_net_cagr_vs_qqq",
                            "stress_vs_base_cost_qqq_diagnostic",
                            "independent_bt_max_equity_difference_usd",
                            "independent_monthly_feedback",
                        )
                    },
                    "years": years,
                    "scenario": scenario,
                    "benchmark_net_cagr": {s: lookup[years, scenario, s] for s in ("SPY", "QQQ")},
                    "actual_factor_participation": row.get("actual_factor_participation", {}).get(
                        key
                    ),
                }
            )
        reviews.append(
            {
                "candidate_id": name,
                "status": row["status"],
                "paths": paths,
                "reviewed_at": row.get("reviewed_at"),
                "failure_event_sha256": row.get("failure_event_sha256"),
                "failure_stage": row.get("failure_stage"),
                "all_four_paths_same_candidate_and_frozen_parameters": row.get(
                    "all_four_paths_same_candidate_and_frozen_parameters", False
                ),
                "independent_forward_validation": False,
                "order_authority": False,
            }
        )
    counts = common.counts(
        [e for e in base.journal_events(base.JOURNAL) if e["candidate_id"] in IDS]
    )
    common.retain_receipt(
        PUBLIC_RESULTS,
        {
            "schema_version": 2,
            "study_id": frozen["study_id"],
            "created_at": utc_now(),
            "actual_completed_at": saved["finished_at"],
            "actual_journal_event_count": saved["actual_journal_event_count"],
            "actual_journal_head_sha256": saved["actual_journal_head_sha256"],
            "actual_publication_before_accounts": saved["publication"],
            "registration_packet": reference(PACKET),
            "protocol": frozen["protocol"],
            "private_complete_results": reference(RESULTS),
            "reviews": reviews,
            **counts,
            "new_completed_configurations": counts["new_completed_strategy_evaluations"],
            "global_counts_at_completion": saved["global_counts_at_completion"],
            "source_experts_or_old_records_reclassified": False,
            "raw_prices_documents_targets_or_daily_accounts_exported": False,
            "new_shadow_account_runs": 0,
            "new_shadow_recommendation_artifacts": 0,
            "independent_forward_validation": False,
            "investment_objective_verified": False,
            "order_authority": False,
            "automatic_live_deployment": False,
        },
        ("created_at",),
    )
    return reference(PUBLIC_RESULTS)


def main():
    parser = argparse.ArgumentParser(description="Controlled five-expert research; no trading.")
    parser.add_argument(
        "command", choices=("init", "register", "evaluate", "publish-result", "status")
    )
    parser.add_argument("--commit")
    args = parser.parse_args()

    def offline(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise QuantError("Five-expert research cannot request live sources.")

    sys.addaudithook(offline)
    try:
        if args.command == "init":
            result = initialize()
        elif args.command == "register":
            result = register()
        elif args.command == "evaluate":
            if args.commit is None:
                raise QuantError("Provide the original actual preregistration publication commit.")
            result = evaluate(args.commit)
        elif args.command == "publish-result":
            result = public_result()
        else:
            result = current.current_counts(base.journal_events(base.JOURNAL))
        print(json.dumps(result, indent=2, allow_nan=False))
    except QuantError as error:
        parser.exit(2, f"Five-expert research blocked: {error}\n")


if __name__ == "__main__":
    main()
