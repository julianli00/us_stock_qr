from __future__ import annotations

from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_research_v1 as base
from us_quant.config import QuantError
from us_quant.storage import utc_now


def current_counts(events: list[dict]) -> dict:
    if not events or events[0]["kind"] != "baseline":
        raise QuantError("Current totals need the preserved actual baseline event.")
    legacy = events[0]["body"]["legacy"]
    keys = (
        "fixed_snapshot_completed_configurations",
        "fixed_snapshot_recurring_reviews",
        "fixed_snapshot_registered_versions",
    )
    if any(type(legacy.get(key)) is not int or legacy[key] < 0 for key in keys):
        raise QuantError("Legacy counts must be actual nonnegative integer observations.")
    counts = common.counts(events)
    return {
        "original_completed_configurations": legacy[keys[0]],
        "original_reviews": legacy[keys[1]],
        "original_registered_versions": legacy[keys[2]],
        **counts,
        "total_legacy_plus_ssd_completed_configurations": (
            legacy[keys[0]] + counts["new_completed_strategy_evaluations"]
        ),
        "current_journal_event_count": len(events),
        "current_journal_head_sha256": events[-1]["sha256"],
    }


def progress(phase: str, **fields) -> None:
    counts = current_counts(base.journal_events(base.JOURNAL))
    if any(key in fields and fields[key] != value for key, value in counts.items()):
        raise QuantError("A caller cannot override current authoritative progress counts.")
    base.progress(
        phase,
        **{
            **fields,
            **counts,
            "current_summary_scope": "authoritative_journal_baseline_plus_completed_reviews",
            "current_summary_derived_at_utc": utc_now(),
        },
    )
