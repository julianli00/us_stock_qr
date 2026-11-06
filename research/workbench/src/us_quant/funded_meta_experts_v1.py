from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from us_quant import ssd_commitment_research_v1 as canonical
from us_quant import ssd_research_v1 as base
from us_quant.calendar import is_month_end, next_session, previous_session, sessions
from us_quant.config import QuantError
from us_quant.retained_pair_equal_allocation_v1 import ASSETS
from us_quant.storage import digest_json, read_json
from us_quant.strategy_experts_v1 import ExpertIdentity, StrategyExpert, _ArchivedExpert

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/funded-meta-expert-eligibility-v1.json"
IDS = (
    "expert_pool_eight_unconditional_forecast_pair_v2",
    "expert_pool_eight_contextual_forecast_pair_v2",
    "expert_pool_unconditional_forecast_opportunity_gate_v1",
    "expert_pool_contextual_forecast_opportunity_gate_v1",
)
GROUP = "original_eight_active_forecast_pair_program_family"
INITIAL = pd.Timestamp("2016-10-05")
CLOCK = "after_16_15_ET_index_close_before_later_open_execution"
ANCHORS = {
    key: previous_session("2016-10-06" if key.startswith("10y") else "2021-10-06")
    for key in ("10y-base", "10y-stress", "5y-base", "5y-stress")
}


def require(condition, message):
    if not condition:
        raise QuantError(message)


def validate(policy):
    expected = {
        "schema_version": 1,
        "study_id": "ssd_funded_meta_expert_eligibility_20261106_v1",
        "actual_native_event_seq": 178087,
        "prior_journal_sequence": 1297,
        "prior_journal_head_sha256": (
            "689bed6d71de7e4851b13380482bd5ae1f42c420ef74eb33876630dbe0555d73"
        ),
        "source_ids": list(IDS),
        "mechanism_group": GROUP,
        "source_family_count": 1,
        "initial_policy_opinion_session": "2016-10-05",
        "native_information_clock": CLOCK,
        "initial_opinion_requires_retained_resolved_source_record": True,
        "capital_initialization_alone_creates_policy_opinion": False,
        "source_native_RF_must_equal_canonical_default_reader": True,
        "new_RF_or_nominal_target_or_assessment_views": 0,
        "new_financial_trials": 0,
        "global_pool_changed": False,
        "pool_representative_selected": False,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
        "order_authority": False,
    }
    require(
        isinstance(policy, dict)
        and all(type(policy.get(k)) is type(v) and policy.get(k) == v for k, v in expected.items())
        and set(policy) == set(expected) | {"source_records"},
        "Freeze the four source identities, exact RF, native opinions and nonmembership scope.",
    )


def check_record(record):
    require(
        isinstance(record, dict) and record.get("id") in IDS,
        "This Adapter cannot admit terminal V1, term-event or another archived source.",
    )
    position = IDS.index(record["id"])
    advisor = position < 2
    require(
        record["mechanism_group"] == GROUP
        and record["risk_behavior"]
        == ("always_invested" if advisor else "positive_opportunity_or_BIL")
        and record["model_variant"] == ("unconditional" if position % 2 == 0 else "contextual")
        and record["initial_policy_opinion_session"] == "2016-10-05"
        and record["native_information_clock"] == CLOCK
        and record["tape_input_kind"]
        == ("saved_forecast_allocation_input" if advisor else "opportunity_gate_input")
        and record["tape_resolved_kind"]
        == ("saved_forecast_allocation_resolved" if advisor else "opportunity_gate_resolved")
        and record["tape_candidate_id"]
        == (record["id"].removesuffix("_v2") + "_v1" if advisor else record["id"])
        and record["target_proof_decision_count"] == 121
        and record["counterfactual_historical_availability"] is True
        and record["new_global_pool_membership"] is False,
        "No foreign identity, seed-derived clock, source-state rewrite or pool promotion.",
    )


class ArchivedInitialMonthlyExpert(_ArchivedExpert):
    """Observe declared source opinions, not a funded path's initialization copy."""

    def __init__(self, identity, targets, shadow, universe):
        require(
            identity.identifier in IDS
            and identity.mechanism_group == GROUP
            and tuple(universe) == tuple(ASSETS)
            and tuple(targets.columns) == tuple(ASSETS),
            "Keep this exact complete-program family and original fourteen-asset coordinates.",
        )
        super().__init__(identity, targets, shadow, tuple(universe))
        require(
            INITIAL in targets.index and self._active.index[0] == INITIAL,
            "A declared original initial opinion is required; capital cannot invent it.",
        )
        expected = pd.DatetimeIndex(
            [INITIAL, *[day for day in targets.index if day > INITIAL and is_month_end(day)]]
        )
        require(
            self._active.index.equals(expected),
            "Keep the genuine initial opinion and every later month end, with no extra events.",
        )

    def _known_target(self, as_of):
        known = self._active.loc[:as_of]
        return (None, None) if known.empty else (known.iloc[-1].copy(), known.index[-1])

    def event_context(self, as_of):
        day = pd.Timestamp(as_of)
        require(day in self._targets.index, "Use a covered actual source information session.")
        _, known = self._known_target(day)
        return {
            "as_of": str(day.date()),
            "policy_opinion_known_at": str(known.date()) if known is not None else None,
            "native_policy_opinion_refreshed_today": day in self._active.index,
            "declared_initial_policy_opinion_today": day == INITIAL,
            "ordinary_month_end_opinion_today": day in self._active.index and is_month_end(day),
            "formal_capital_anchor_paths": [
                key for key, anchor in ANCHORS.items() if anchor == day
            ],
            "capital_initialization_itself_is_policy_refresh": False,
            "observation_is_a_new_execution_request": False,
            "source_actual_holdings_copied_as_opinion": False,
            "historical_counterfactual": True,
        }

    def preservation(self):
        return {
            "native_policy_opinions": len(self._active),
            "declared_initial_policy_opinions": 1,
            "ordinary_month_end_opinions": len(self._active) - 1,
            "source_target_rows_removed_retimed_or_backfilled": False,
            "new_nominal_target_view_artifacts": 0,
            "source_family_count": 1,
        }


def frame(reference, *, round_trip=True):
    kwargs = {"float_precision": "round_trip"} if round_trip else {}
    return pd.read_csv(base.checked_reference(reference), index_col=0, parse_dates=True, **kwargs)


def verify_retained_opinions(record, targets):
    check_record(record)
    active, count = targets.dropna(how="all"), 0
    with base.checked_reference(record["resolved_opinion_tape"]).open() as stream:
        while True:
            before_line = stream.readline()
            if not before_line:
                break
            after_line = stream.readline()
            require(bool(after_line), "A partial source record cannot validate an opinion.")
            before, after = json.loads(before_line), json.loads(after_line)
            require(
                count < len(active)
                and before["record"] == record["tape_input_kind"]
                and after["record"] == record["tape_resolved_kind"],
                "Keep every exact source input/resolution pair in its declared order.",
            )
            audit = after["audit"]
            values = np.asarray(after["targets"][record["tape_candidate_id"]], dtype=float)
            require(
                audit["decision_index"] == count
                and audit["decision_session"] == str(active.index[count].date())
                and audit["decision_information_clock"] == CLOCK
                and values.shape == (len(ASSETS),)
                and np.array_equal(
                    values.view(np.uint64), active.iloc[count].to_numpy().view(np.uint64)
                ),
                "The initial and monthly opinions must equal retained source values bit-exactly.",
            )
            count += 1
    require(
        count == record["target_proof_decision_count"] == len(active),
        "No omitted opinion or incomplete initial/month-end source history.",
    )
    return {"retained_policy_opinions_bit_exact": count, "source_tape_replayed": False}


def _funding(receipt, targets, common_rf):
    key = receipt["path_key"]
    artifacts = {Path(r["path"]).name: r for r in receipt["artifacts"]}
    require(
        len(artifacts) == 5
        and set(artifacts)
        == {"account.csv", "weights.csv", "issued.csv", "independent-bt.csv", "decisions.json"},
        "Keep all five original funded-path accounting artifacts.",
    )
    book, weights, bt, issued = [
        frame(artifacts[name])
        for name in ("account.csv", "weights.csv", "independent-bt.csv", "issued.csv")
    ]
    decisions = read_json(base.checked_reference(artifacts["decisions.json"]))["decisions"]
    dates = sessions("2016-10-06" if key.startswith("10y") else "2021-10-06", "2026-10-05")
    anchor, delay = ANCHORS[key], 1 if key.endswith("base") else 2
    require(
        book.index.equals(dates)
        and weights.index.equals(dates)
        and bt.index.equals(dates)
        and len(dates) == (2512 if key.startswith("10y") else 1254)
        and tuple(weights.columns) == tuple(ASSETS)
        and issued.index.equals(targets.index)
        and issued.columns.equals(targets.columns)
        and book["risk_free"].equals(common_rf.loc[dates]),
        "Preserve every original funded date and exact native RF; no replacement RF view.",
    )
    needed = ["equity", "return", "cash", "gross_exposure", "turnover", "cost", "orders"]
    require(
        np.isfinite(book[needed].to_numpy()).all()
        and np.isfinite(weights.to_numpy()).all()
        and np.isfinite(bt["equity"]).all()
        and (book["equity"] > 0).all()
        and (book["return"] > -1).all()
        and (book[["cash", "gross_exposure", "turnover", "cost", "orders"]] >= 0).all().all()
        and (book["gross_exposure"] <= 1).all()
        and (weights >= 0).all().all()
        and np.max(abs(weights.sum(axis=1) - book["gross_exposure"])) <= 1e-12
        and np.max(abs(book["equity"] - bt["equity"])) <= 1e-6
        and receipt["independent_bt_max_equity_difference_usd"] <= 1e-6
        and receipt["broker_orders"] == 0
        and receipt["independent_monthly_feedback"]["all_actual_gaps_targets_and_delays_checked"]
        is True,
        "A financial rejection does not waive cash funding, 1x, accounting or independent BT.",
    )
    previous = book["equity"].shift(1)
    previous.iloc[0] = 10000.0
    require(
        np.allclose(book["equity"] / previous - 1, book["return"], rtol=0, atol=1e-12),
        "The original source book must retain its own fresh USD10000 and net returns.",
    )
    expected = [anchor, *[day for day in dates if is_month_end(day)]]
    require(
        len(decisions) == len(expected)
        and not (issued.isna().any(axis=1) & ~issued.isna().all(axis=1)).any(),
        "Source capital initialization and monthly holding feedback cannot lose events.",
    )
    requests = []
    active = targets.dropna(how="all")
    for i, (decision, day) in enumerate(zip(decisions, expected, strict=True)):
        execution = day
        for _ in range(delay):
            execution = next_session(execution)
        require(
            decision["signal_session"] == str(day.date())
            and decision["execution_session"] == str(execution.date())
            and decision["initial_capital"] is (i == 0)
            and (i != 0 or decision["reason"] == "initial_capital"),
            "No funding seed may move a policy opinion or an original later-open execution.",
        )
        if decision["reason"] != "hold":
            requests.append(day)
            known = active.loc[:day]
            require(
                not known.empty and issued.loc[day].equals(known.iloc[-1]),
                "A source request must use its last actually known complete opinion.",
            )
    require(
        issued.dropna(how="all").index.equals(pd.DatetimeIndex(requests)),
        "A carried opinion is not an extra source request or a fabricated capital seed.",
    )
    known_at = active.loc[:anchor].index[-1]
    return book, {
        "return_dates": len(dates),
        "native_RF_exact_canonical": True,
        "minimum_cash_usd": float(book["cash"].min()),
        "maximum_gross_exposure": float(book["gross_exposure"].max()),
        "saved_independent_BT_gap_usd": float(np.max(abs(book["equity"] - bt["equity"]))),
        "original_feedback_decisions": len(decisions),
        "actual_issued_requests": len(requests),
        "actual_capital_initialization_session": str(anchor.date()),
        "original_policy_opinion_known_at_capital_initialization": str(known_at.date()),
        "capital_initialization_creates_new_policy_opinion": False,
        "source_account_or_metrics_recomputed": False,
    }


def load_expert(record: dict, universe: tuple[str, ...]) -> tuple[StrategyExpert, dict]:
    check_record(record)
    require(
        tuple(universe) == tuple(ASSETS), "Only the original complete asset lattice is admitted."
    )
    packet = read_json(base.checked_reference(record["registration"]))
    source_result = read_json(base.checked_reference(record["results"]))
    entries = [e for e in packet["entries"] if e["candidate_id"] == record["id"]]
    reviews = base.matching_events("review", record["id"])
    require(len(entries) == len(reviews) == 1, "Use the actual unique completed source identity.")
    entry, review = entries[0], reviews[0]
    frozen = entry["specification"]["frozen"]
    expected_cohort = IDS[:2] if record["id"] in IDS[:2] else IDS[2:]
    registrations = [base.matching_events("registered", n) for n in expected_cohort]
    require(
        all(len(rows) == 1 for rows in registrations),
        "Every original cohort member needs one actual registration.",
    )
    require(
        packet["schema_version"] == 2
        and packet["program_id"] == "ssd_continuous_research_v1"
        and packet["complete_fixed_cohort_ids"] == list(expected_cohort)
        and packet["cohort_registrations"] == [rows[0]["sha256"] for rows in registrations]
        and packet["candidate_accounts_or_metrics_computed"] is False
        and packet["order_authority"] is False
        and source_result["order_authority"] is False
        and frozen["protocol"]
        == {"path": "config/dual-benchmark-v1.json", "sha256": base.PROTOCOL_SHA256}
        and entry == canonical.packet_entry(entry["specification"]["candidate"], frozen)
        and entry["registered_at"] == record["introduced_at"]
        and entry["registered_event_sha256"] == record["registered_event_sha256"]
        and review["body"]["frozen_specification_sha256"] == digest_json(frozen)
        and review["body"]["all_four_paths_same_candidate_and_frozen_parameters"] is True
        and review["body"]["status"] in {"rejected", "historical_point_gates_passed"}
        and review["body"]["paths"] == record["completed_paths"]
        and set(record["completed_paths"]) == base.PATH_KEYS
        and not base.matching_events("technical_failure", record["id"]),
        "Technical source completeness is distinct from any standalone financial verdict.",
    )
    base.check_frozen_files(frozen)
    for ref in frozen["prior_evidence"] + frozen["validation_evidence"]:
        base.checked_reference(ref)
    base.checked_reference(record["source_cycle"])
    base.checked_reference(record["source_implementation_freeze"])
    opinion_policy = read_json(base.checked_reference(record["initial_opinion_policy"]))
    require(
        record["initial_opinion_policy"] in frozen["policies"]
        and opinion_policy["initial_decision_session"] == "2016-10-05"
        and opinion_policy["decision_information_clock"] == CLOCK,
        "A capital date alone cannot establish the original initial policy announcement.",
    )
    fences = [
        e
        for e in base.journal_events(base.JOURNAL)
        if e["kind"] == "publication" and e["body"].get("study_id") == frozen["study_id"]
    ]
    starts = base.matching_events("path_started", record["id"])
    require(
        len(fences) == 1
        and len(starts) == 4
        and pd.Timestamp(packet["created_at"]) <= pd.Timestamp(fences[0]["at"])
        and all(pd.Timestamp(fences[0]["at"]) < pd.Timestamp(e["at"]) for e in starts),
        "The original complete-cohort publication must precede all four actual funds.",
    )
    common_rf = frame(frozen["market"]["risk_free"], round_trip=False).iloc[:, 0]
    targets = frame(entry["target_proof"]["targets"])
    opinions = verify_retained_opinions(record, targets)
    funding, shadow = {}, None
    for key, ref in record["completed_paths"].items():
        receipt = base.verify_path(ref, frozen, record["id"], key)
        require(
            receipt == source_result["candidates"][record["id"]]["paths"][key]
            and source_result["candidates"][record["id"]]["status"] == review["body"]["status"],
            "Preserve every public source outcome and its exact original saved receipt.",
        )
        book, funding[key] = _funding(receipt, targets, common_rf)
        if key == "10y-base":
            shadow = book
    require(shadow is not None, "A complete expert needs its own full ten-year base net book.")
    expert = ArchivedInitialMonthlyExpert(
        ExpertIdentity(record["id"], GROUP, record["introduced_at"]),
        targets,
        shadow,
        tuple(universe),
    )
    return expert, {
        "id": record["id"],
        "mechanism_group": GROUP,
        "introduced_at": record["introduced_at"],
        "source_financial_verdict_preserved": review["body"]["status"],
        "technical_source_and_funding_checks_passed": True,
        "native_RF_exact_canonical": True,
        "new_RF_or_nominal_target_or_assessment_views": 0,
        "original_full_funded_paths": funding,
        **opinions,
        **expert.preservation(),
        "source_target_reference": entry["target_proof"]["targets"],
        "source_registered_frozen_sha256": digest_json(frozen),
        "genuine_initial_policy_opinion_is_not_a_backdated_funding_seed": True,
        "global_pool_changed": False,
        "pool_representative_selected": False,
        "counterfactual_historical_availability": True,
        "independent_forward_validation": False,
        "investment_objective_verified": False,
    }
