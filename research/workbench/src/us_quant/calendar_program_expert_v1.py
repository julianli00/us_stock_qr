from __future__ import annotations

import pandas as pd

from us_quant import calendar_seasonal_expert_v1 as law
from us_quant import ssd_research_v1 as base
from us_quant.config import QuantError
from us_quant.storage import digest_json, read_json
from us_quant.strategy_experts_v1 import (
    ArchivedMonthlyExpert,
    ExpertIdentity,
    StrategyExpert,
)


def _frame(reference):
    return pd.read_csv(
        base.checked_reference(reference),
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )


def load_calendar_expert(
    record: dict, universe: tuple[str, ...], risk_free: pd.Series
) -> tuple[StrategyExpert, dict]:
    required = {
        "schema_version",
        "id",
        "mechanism_group",
        "introduced_at",
        "registration",
        "results",
        "admission",
        "completion",
    }
    if (
        not isinstance(record, dict)
        or set(record) != required
        or type(record["schema_version"]) is not int
        or record["schema_version"] != 1
        or record["id"] != law.ID
        or record["mechanism_group"] != "fixed_calendar_seasonal_equity_participation"
        or universe != law.UNIVERSE
        or not isinstance(risk_free, pd.Series)
        or not isinstance(risk_free.index, pd.DatetimeIndex)
        or not risk_free.index.is_unique
        or not risk_free.index.is_monotonic_increasing
    ):
        raise QuantError("Load only the declared complete calendar program and common RF lattice.")
    packet = read_json(base.checked_reference(record["registration"]))
    result = read_json(base.checked_reference(record["results"]))
    admission = read_json(base.checked_reference(record["admission"]))
    closure = read_json(base.checked_reference(record["completion"]))
    frozen = result["frozen"]
    entries = packet.get("entries", [])
    if (
        packet.get("schema_version") != 3
        or len(entries) != 1
        or entries[0]["candidate_id"] != law.ID
        or entries[0]["specification"]["frozen"] != frozen
        or packet["complete_fixed_cohort_ids"] != [law.ID]
        or packet["historical_new_policy_targets_computed"] is not False
        or packet["candidate_accounts_or_metrics_computed"] is not False
        or packet["order_authority"] is not False
        or packet["historical_target_proof_deferred_until_published_execution"] is not True
        or entries[0]["historical_target_derivation_proof"] is not None
        or admission["candidate_id"] != law.ID
        or admission["introduced_at"] != record["introduced_at"]
        or admission["introduced_at"] != entries[0]["registered_at"]
        or admission["registered_event_sha256"] != entries[0]["registered_event_sha256"]
        or admission["registration"] != record["registration"]
        or admission["native_RF_exact_common"] is not True
        or admission["funding_and_independent_BT_checked_all_four_paths"] is not True
        or admission["technical_admission_is_final_qualification"] is not False
        or admission["current_global_pool_or_roster_changed"] is not False
        or admission["pool_inclusion_requires_new_separate_study"] is not True
        or admission["order_authority"] is not False
        or result["review"]["order_authority"] is not False
        or frozen["order_authority"] is not False
        or closure["technical_expert_admitted"] is not True
        or closure["global_pool_changed"] is not False
        or closure["private_results"] != record["results"]
        or closure["registration"] != record["registration"]
        or closure["technical_admission"] != record["admission"]
        or result["publication"]["registration_packet"] != record["registration"]
        or result["review"]["status"] not in {"rejected", "historical_point_gates_passed"}
        or admission["standalone_financial_verdict_preserved"] != result["review"]["status"]
        or closure["financial_status"] != result["review"]["status"]
        or result["review"]["all_four_paths_same_candidate_and_frozen_parameters"] is not True
        or set(result["review"]["paths"]) != base.PATH_KEYS
        or admission["completed_paths"] != result["review"]["paths"]
    ):
        raise QuantError("Published program validity, financial rejection and pool scope differ.")
    base.check_frozen_files(frozen)
    for reference in frozen["prior_evidence"] + frozen["validation_evidence"]:
        base.checked_reference(reference)
    for reference in closure["exact_evidence_references"]:
        base.checked_reference(reference)
    history = base.journal_events(base.JOURNAL)
    registered = [
        row for row in history if row["kind"] == "registered" and row["candidate_id"] == law.ID
    ]
    reviewed = [row for row in history if row["kind"] == "review" and row["candidate_id"] == law.ID]
    published = [
        row
        for row in history
        if row["kind"] == "publication" and row["body"].get("study_id") == frozen["study_id"]
    ]
    proof_events = [
        row for row in history if row["kind"] == "target_proof" and row["candidate_id"] == law.ID
    ]
    if (
        len(registered) != 1
        or len(reviewed) != 1
        or len(published) != 1
        or len(proof_events) != 1
        or registered[0]["sha256"] != admission["registered_event_sha256"]
        or registered[0]["at"] != record["introduced_at"]
        or reviewed[0]["body"] != result["review"]
        or published[0]["body"]["verification"] != result["publication"]
        or proof_events[0]["body"]["native_source_proof"] != result["native_source_proof"]
        or not registered[0]["sequence"]
        < published[0]["sequence"]
        < proof_events[0]["sequence"]
        < reviewed[0]["sequence"]
    ):
        raise QuantError("Require actual registered, published and independently completed events.")
    source = result["native_source_proof"]
    proof = read_json(base.checked_reference(source["proof"]))
    started = read_json(base.checked_reference(proof["source_start"]))
    if (
        source != admission["native_source_targets"]
        or proof["frozen_specification_sha256"] != digest_json(frozen)
        or proof["publication_event_sha256"] != published[0]["sha256"]
        or proof["real_native_targets_first_generated_after_actual_publication"] is not True
        or proof["all_independent_calendar_targets_exact"] is not True
        or proof["whole_native_prehistory_retained"] is not True
        or proof["synthetic_initial_recommendation_added"] is not False
        or started["publication_event_sha256"] != published[0]["sha256"]
        or not pd.Timestamp(published[0]["at"])
        <= pd.Timestamp(started["started_at"])
        <= pd.Timestamp(proof["checked_at"])
        < pd.Timestamp(reviewed[0]["at"])
    ):
        raise QuantError("Native source targets cannot predate their actual publication fence.")
    saved = {}
    for key, reference in result["review"]["paths"].items():
        complete = base.verify_path(reference, frozen, law.ID, key)
        if (
            complete != result["paths"][key]
            or complete["execution"]["minimum_cash_usd"] < 0
            or not 0 <= complete["execution"]["maximum_gross_exposure"] <= 1
            or complete["independent_bt_max_equity_difference_usd"] > 1e-6
            or complete["broker_orders"] != 0
        ):
            raise QuantError("Incomplete or unfunded source books cannot be blended into a pool.")
        saved[key] = complete
    references = [
        row for row in saved["10y-base"]["artifacts"] if row["path"].endswith("/account.csv")
    ]
    if len(references) != 1:
        raise QuantError("Use the existing unique ten-year base NET source book, not new shadows.")
    shadow = _frame(references[0])
    targets = _frame(source["targets"])
    if (
        len(shadow) != 2512
        or str(shadow.index[0].date()) != "2016-10-06"
        or str(shadow.index[-1].date()) != "2026-10-05"
        or not shadow.index.isin(risk_free.index).all()
        or not shadow["risk_free"].equals(risk_free.loc[shadow.index])
        or tuple(targets.columns) != universe
    ):
        raise QuantError("The complete source cannot crop dates, change assets or substitute RF.")
    expert = ArchivedMonthlyExpert(
        ExpertIdentity(law.ID, record["mechanism_group"], record["introduced_at"]),
        targets,
        shadow,
        universe,
    )
    return expert, {
        "id": law.ID,
        "mechanism_group": record["mechanism_group"],
        "adapter": "monthly",
        "source_packet_schema": 3,
        "introduced_at": record["introduced_at"],
        "actual_expert_admitted_at": admission["admitted_at"],
        "decision_information_clock": law.CLOCK,
        "target_reference": source["targets"],
        "shadow_reference": references[0],
        "source_technical_and_funding_checks_passed": True,
        "native_RF_exact_common": True,
        "standalone_financial_verdict_preserved": result["review"]["status"],
        "original_source_targets_or_accounts_replayed": False,
        "global_pool_promotion": False,
        "historical_counterfactual_availability": True,
    }
