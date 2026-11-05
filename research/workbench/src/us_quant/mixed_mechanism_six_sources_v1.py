from __future__ import annotations

import pandas as pd

from us_quant import mixed_mechanism_six_primary_v1 as law
from us_quant import ssd_eight_expert_meta_v1 as original
from us_quant.calendar import sessions
from us_quant.calendar_program_expert_v1 import load_calendar_expert
from us_quant.config import QuantError
from us_quant.fixed_four_sleeve_admission_v1 import load_admitted_expert as load_fixed
from us_quant.monthly_style_reversal_admission_v1 import load_admitted_expert as load_reversal
from us_quant.storage import read_json

base, ROOT = original.base, law.ROOT


def source_inputs(policy):
    law.validate(policy)
    original_policy = read_json(base.checked_reference(policy["original_eight_policy"]))
    data, originals, parent, refs, clocks, benchmarks = original.source_inputs()
    if (
        tuple(original_policy["expert_ids"]) != law.original.EXPERT_IDS
        or tuple(data.close.columns) != law.UNIVERSE
    ):
        raise QuantError("The original admitted context or asset scope changed.")
    extra_configs = {
        key: read_json(base.checked_reference(policy[key]))
        for key in ("calendar_policy", "reversal_policy", "fixed_policy")
    }
    records = [
        extra_configs["calendar_policy"]["calendar_source"],
        extra_configs["reversal_policy"]["reversal_source"],
        extra_configs["fixed_policy"]["fixed_four_sleeve_source"],
    ]
    calendar, calendar_info = load_calendar_expert(records[0], law.UNIVERSE, data.risk_free)
    reversal, reversal_info = load_reversal(records[1]["admission"], law.UNIVERSE, data.risk_free)
    fixed, fixed_info = load_fixed(records[2]["admission"], law.UNIVERSE, data.risk_free)
    experts = law.assemble(originals, calendar, reversal, fixed)
    metadata = [calendar_info, reversal_info, fixed_info]
    additional = []
    original_clocks = {row["id"]: row for row in clocks}
    selected_clocks = [original_clocks[identifier] for identifier in law.EXPERT_IDS[:3]]
    for i, (record, technical) in enumerate(zip(records, metadata, strict=True)):
        identifier = technical["id"] if i == 0 else technical["candidate_id"]
        if (
            identifier != law.EXPERT_IDS[i + 3]
            or technical["mechanism_group"] != law.MECHANISMS[i + 3]
            or technical["introduced_at"] != record["introduced_at"]
            or technical["native_RF_exact_common"] is not True
            or technical["standalone_financial_verdict_preserved"] != "rejected"
        ):
            raise QuantError(
                "The admitted complete policy, native RF or actual introduction changed."
            )
        closed = read_json(base.checked_reference(record["completion"]))
        additional.extend(closed["exact_evidence_references"])
        additional.extend(record[k] for k in ("registration", "results", "admission", "completion"))
        selected_clocks.append(
            {
                "id": identifier,
                "source_policy_registered_at": record["introduced_at"],
                "actual_expert_admitted_at": technical["actual_expert_admitted_at"],
                "source_packet_schema": 3,
                "decision_information_clock": law.CLOCK,
                "native_RF_exact_common": True,
                "study_specific_representative_not_global_promotion": True,
            }
        )
    formal = sessions("2016-10-06", "2026-10-05")
    for expert in experts:
        initial = expert.observe(pd.Timestamp("2016-10-05"), 126)
        full = expert.observe(pd.Timestamp("2026-10-05"), 2512)
        if (
            initial.target is None
            or initial.score_mature
            or len(initial.net_returns)
            or tuple(initial.target.index) != law.UNIVERSE
            or not full.net_returns.index.equals(formal)
            or not full.risk_free.equals(data.risk_free.loc[formal])
        ):
            raise QuantError(
                "Keep complete opinions, empty initial NET and every original RF/date."
            )
    refs = original.unique_refs(
        [
            *refs,
            *additional,
            *[
                policy[k]
                for k in (
                    "original_eight_policy",
                    "calendar_policy",
                    "reversal_policy",
                    "fixed_policy",
                    "preceding_financial_completion",
                )
            ],
            base.reference(original.INTENT),
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    return data, experts, parent, refs, selected_clocks, benchmarks
