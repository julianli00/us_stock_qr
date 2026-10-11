from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

from us_quant.calendar import previous_session, sessions
from us_quant.config import QuantError
from us_quant.dollar_risk_guard import decision_pairs
from us_quant.financial_conditions_guard import needed_vintage_dates
from us_quant.macro_factor_tilt import align_observations
from us_quant.research_program import ResearchProgram, engine_fingerprint, safe_file
from us_quant.storage import (
    digest_json,
    file_digest,
    new_output_directory,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/real-yield-vintages.json"
SOURCE = ROOT / "data/real-yield-vintages-20261011"
FORM = {
    "url": "https://alfred.stlouisfed.org/series/downloaddata?seid=DFII10",
    "method": "POST",
    "maximum_dates_per_request": 40,
    "maximum_entered_dates_characters": 500,
}
REFERENCES = (
    "research_snapshot",
    "source_index",
    "current_source",
    "original_macro_policy",
    "original_macro_helper",
    "vintage_manifest",
)


def parse_vintage_archive(path: Path, dates: list[str]) -> pd.DataFrame:
    if (
        not dates
        or dates != sorted(set(dates))
        or len(dates) > 40
        or len(" ".join(dates)) > 500
        or any(not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day) for day in dates)
    ):
        raise QuantError(
            "Daily-vintage requests must retain the advertised bounded date-entry format."
        )
    try:
        with ZipFile(path) as archive:
            member = f"vintages_starting_{dates[0]}.csv"
            if (
                set(archive.namelist()) != {"README.txt", member}
                or len(archive.namelist()) != 2
                or any(entry.file_size > 5_000_000 for entry in archive.infolist())
            ):
                raise QuantError("DFII10 requires the exact bounded two-file daily-vintage ZIP.")
            readme = archive.read("README.txt").decode("utf-8-sig")
            if not all(
                term in readme
                for term in (
                    "Series ID: DFII10",
                    "Observations by Vintage Date, All Observations",
                    "Board of Governors of the Federal Reserve System",
                    "H.15 Selected Interest Rates",
                    "Percent",
                    "Daily",
                )
            ):
                raise QuantError(
                    "The archive does not substantiate the expected daily real-yield series."
                )
            declared = re.search(r"Vintage Dates Specified:\s*-+\s*(.*?)\s*-+\s*$", readme, re.S)
            if declared is None or declared[1].split() != dates:
                raise QuantError(
                    "The primary README must confirm every requested historical as-of date."
                )
            content = archive.read(member)
    except (BadZipFile, UnicodeError) as exc:
        raise QuantError("A DFII10 vintage must be an actual readable archive, not HTML.") from exc
    frame = pd.read_csv(io.BytesIO(content), dtype=str, na_values=["."])
    expected = ["observation_date", *(f"DFII10_{day.replace('-', '')}" for day in dates)]
    if list(frame.columns) != expected:
        raise QuantError(
            "The source columns do not identify exactly the requested DFII10 vintages."
        )
    index = pd.DatetimeIndex(pd.to_datetime(frame.pop("observation_date"), errors="raise"))
    if (
        index.empty
        or not index.is_unique
        or not index.is_monotonic_increasing
        or (index.weekday >= 5).any()
    ):
        raise QuantError("Daily real yields need distinct chronological business-day observations.")
    result = pd.DataFrame(index=index)
    for day, column in zip(dates, expected[1:], strict=True):
        values = pd.to_numeric(frame[column], errors="raise").to_numpy()
        known = np.isfinite(values)
        if not known.any() or np.isinf(values).any() or (index[known] > pd.Timestamp(day)).any():
            raise QuantError(
                "Historical vintages cannot contain future or nonfinite yield observations."
            )
        result[day] = values
    return result


def select_vintage(observed: pd.Series, query: pd.Timestamp, cutoff: str) -> tuple[float, dict]:
    if (
        pd.Timestamp(cutoff) != previous_session(query)
        or (observed.dropna().index > pd.Timestamp(cutoff)).any()
    ):
        raise QuantError(
            "Use the actual previous-NYSE-session snapshot, without future observations."
        )
    values, audit = align_observations(observed, pd.DatetimeIndex([query]))
    row = audit.loc[query]
    return float(values.loc[query]), {
        "vintage_cutoff": cutoff,
        "observation_date": str(row["observation_date"].date()),
        "assumed_observation_available_session": str(row["available_session"].date()),
        "age_calendar_days": int(row["age_calendar_days"]),
        "cutoff_strictly_before_query": pd.Timestamp(cutoff) < query,
    }


def validate_policy(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "diagnostic_id": "real_yield_vintage_validation_20261011",
        "series": "DFII10",
        "data_start": "2015-08-10",
        "as_of": "2026-10-05",
        "source_observation_start": "2014-01-01",
        "source_observation_end": "2026-10-05",
        "query_vintage_count": 225,
        "monthly_comparison_count": 131,
        "vintage_cutoff": "previous_nyse_session",
        "publication_delay_sessions": 2,
        "maximum_observation_age_days": 7,
        "change_sessions": 63,
        "stale_or_missing_required_input": "block_and_preserve_failure",
        "public_form": FORM,
        "new_strategy_evaluations": 0,
    }
    if (
        any(policy.get(key) != value for key, value in expected.items())
        or any(
            policy.get(key) is not False
            for key in (
                "order_authority",
                "qualification_policy_changed",
                "forward_rules_changed",
                "new_factor_definition_registered",
                "source_comparison_is_portfolio_performance",
            )
        )
        or any(
            not isinstance(policy.get(key), dict) or set(policy[key]) != {"path", "sha256"}
            for key in REFERENCES
        )
    ):
        raise QuantError(
            "The fixed real-yield query cohort, source lag, age limit or authority changed."
        )


def frozen_sources(policy: dict) -> tuple[dict, pd.DatetimeIndex, pd.Series, dict]:
    validate_policy(policy)
    paths = {key: safe_file(ROOT, policy[key]["path"], policy[key]["sha256"]) for key in REFERENCES}
    state, manifest = read_json(paths["research_snapshot"]), read_json(paths["vintage_manifest"])
    index = pd.read_csv(paths["source_index"], index_col=0, parse_dates=True).index
    if not index.equals(sessions(policy["data_start"], policy["as_of"])):
        raise QuantError(
            "Source comparisons must retain the exact frozen 2,805-session market index."
        )
    dates = needed_vintage_dates(index)
    if len(dates) != 225 or len(decision_pairs(index)) != 131:
        raise QuantError(
            "Do not substitute a smaller or differently dated real-yield query cohort."
        )
    current = pd.read_csv(paths["current_source"])
    if list(current.columns) != ["observation_date", "DFII10"]:
        raise QuantError("Use the original unchanged explicit DFII10 current-history source.")
    current_series = pd.Series(
        pd.to_numeric(current["DFII10"], errors="raise").to_numpy(),
        index=pd.to_datetime(current["observation_date"]),
        name="DFII10",
    )
    if (
        manifest["series"] != "DFII10"
        or manifest["vintage_dates"] != dates
        or manifest["all_requested_batches_complete"] is not True
        or manifest["source_signal_comparisons_computed"] is not False
    ):
        raise QuantError(
            "Use the sealed complete source acquisition before comparing signal states."
        )
    for name, digest in manifest["files"].items():
        safe_file(SOURCE, name, digest)
    vintages = {}
    for batch in manifest["batches"]:
        batch_dates = batch["vintage_dates"]
        request = read_json(safe_file(SOURCE, batch["request_path"], batch["request_sha256"]))
        fields = {
            "form[units]": "lin",
            "form[obs_start_date]": "2014-01-01",
            "form[obs_end_date]": "2026-10-05",
            "form[entered_vintage_dates]": " ".join(batch_dates),
            "form[file_type]": "2",
            "form[file_format]": "csv",
            "form[download_data]": "",
        }
        if (
            request["url"] != FORM["url"]
            or request["method"] != "POST"
            or request["advertised_form_fields"] != fields
            or request["http_success"] is not True
            or request["sha256"] != batch["sha256"]
            or set(vintages).intersection(batch_dates)
        ):
            raise QuantError("A dated source request changed or duplicated an existing vintage.")
        frame = parse_vintage_archive(
            safe_file(SOURCE, batch["path"], batch["sha256"]), batch_dates
        )
        vintages.update({day: frame[day].dropna() for day in batch_dates})
    if set(vintages) != set(dates):
        raise QuantError(
            "Every original required as-of snapshot is necessary; no current-history fallback."
        )
    return state, index, current_series, vintages


def fingerprint() -> str:
    folder = Path(__file__).resolve().parent
    return digest_json(
        {
            name: file_digest(folder / name)
            for name in (
                "real_yield_vintages.py",
                "macro_factor_tilt.py",
                "financial_conditions_guard.py",
                "dollar_risk_guard.py",
                "calendar.py",
                "storage.py",
            )
        }
    )


def local_path(path: Path) -> Path:
    path = path if path.is_absolute() else ROOT / path
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise QuantError("Daily-vintage evidence must remain inside the isolated workbench.")
    return path


def register(output: Path) -> dict:
    policy = read_json(POLICY)
    state, index, _, vintages = frozen_sources(policy)
    program = ResearchProgram(
        ROOT / "runtime/research-program.sqlite3",
        read_json(ROOT / "config/research-program.json"),
        root=ROOT,
    )
    try:
        if program.status() != state:
            raise QuantError("The source diagnostic must bind its exact unchanged research state.")
        output = local_path(output)
        new_output_directory(output)
        result = {
            "schema_version": 1,
            "registered_at": utc_now(),
            "policy": policy,
            "policy_sha256": file_digest(POLICY),
            "auditor_sha256": fingerprint(),
            "program_engine_sha256": engine_fingerprint(),
            "event_chain_sha256": state["event_chain_sha256"],
            "source_index_sessions": len(index),
            "actual_required_vintage_dates": len(vintages),
            "monthly_comparisons": len(decision_pairs(index)),
            "source_signal_comparisons_computed": False,
            "new_strategy_evaluations": 0,
            "order_authority": False,
        }
        write_json(output / "registration.json", result)
        return result
    finally:
        program.close()


def run(registration_path: Path, output: Path) -> dict:
    registration_path = local_path(registration_path)
    registration = read_json(safe_file(ROOT, registration_path.relative_to(ROOT).as_posix()))
    policy = read_json(POLICY)
    state, index, current, vintages = frozen_sources(policy)
    if (
        registration["policy"] != policy
        or registration["policy_sha256"] != file_digest(POLICY)
        or registration["auditor_sha256"] != fingerprint()
        or registration["program_engine_sha256"] != engine_fingerprint()
        or registration["event_chain_sha256"] != state["event_chain_sha256"]
        or registration["source_signal_comparisons_computed"] is not False
        or pd.Timestamp(registration["registered_at"]).tzinfo is None
        or pd.Timestamp(registration["registered_at"]) > pd.Timestamp(utc_now())
    ):
        raise QuantError("Compare source states only under the actual unchanged preregistration.")
    program = ResearchProgram(
        ROOT / "runtime/research-program.sqlite3",
        read_json(ROOT / "config/research-program.json"),
        root=ROOT,
    )
    try:
        if program.status() != state:
            raise QuantError("Research state changed after source-diagnostic registration.")
        original, original_audit = align_observations(current, index)
        pairs = decision_pairs(index)
        observations, values = [], {}
        for query in sorted(set(pairs) | set(pairs.values())):
            cutoff = str(previous_session(query).date())
            value, audit = select_vintage(vintages[cutoff], query, cutoff)
            values[query] = value
            selected_date = pd.Timestamp(audit["observation_date"])
            matched = current.get(selected_date, np.nan)
            observations.append(
                {
                    "query_session": str(query.date()),
                    **audit,
                    "as_published_selected_yield_percent": value,
                    "original_selected_yield_percent": float(original.loc[query]),
                    "original_observation_date": str(
                        original_audit.loc[query, "observation_date"].date()
                    ),
                    "selected_observation_date_changed": selected_date
                    != original_audit.loc[query, "observation_date"],
                    "selected_yield_difference_percentage_points": value
                    - float(original.loc[query]),
                    "current_same_observation_quote_available": bool(np.isfinite(matched)),
                    "same_observation_value_difference_percentage_points": (
                        value - float(matched) if np.isfinite(matched) else None
                    ),
                }
            )
        comparisons = []
        for query, reference in pairs.items():
            old = float(original.loc[query] - original.loc[reference])
            archived = values[query] - values[reference]
            comparisons.append(
                {
                    "query_session": str(query.date()),
                    "reference_session": str(reference.date()),
                    "original_change_percentage_points": old,
                    "as_published_change_percentage_points": archived,
                    "original_rising_real_yield": old > 0,
                    "as_published_rising_real_yield": archived > 0,
                    "rising_state_changed": (old > 0) != (archived > 0),
                }
            )
        common = [
            row["same_observation_value_difference_percentage_points"]
            for row in observations
            if row["current_same_observation_quote_available"]
        ]
        if not common:
            raise QuantError("No current/archived same-observation comparisons can be identified.")
        if program.status() != state:
            raise QuantError("A read-only source comparison cannot change prior research.")
        result = {
            "schema_version": 1,
            "created_at": utc_now(),
            "diagnostic_id": policy["diagnostic_id"],
            "registration": {
                "path": registration_path.relative_to(ROOT).as_posix(),
                "sha256": file_digest(registration_path),
            },
            "auditor_sha256": fingerprint(),
            "program_engine_sha256": engine_fingerprint(),
            "event_chain_sha256": state["event_chain_sha256"],
            "actual_archived_vintage_dates": len(vintages),
            "selected_queries": len(observations),
            "monthly_comparisons": len(comparisons),
            "changed_selected_observation_dates": sum(
                row["selected_observation_date_changed"] for row in observations
            ),
            "changed_selected_yield_values": sum(
                row["selected_yield_difference_percentage_points"] != 0 for row in observations
            ),
            "same_observation_matched_queries": len(common),
            "changed_same_observation_values": sum(value != 0 for value in common),
            "maximum_same_observation_value_difference_percentage_points": max(
                abs(value) for value in common
            ),
            "maximum_selected_yield_difference_percentage_points": max(
                abs(row["selected_yield_difference_percentage_points"]) for row in observations
            ),
            "maximum_as_published_observation_age_days": max(
                row["age_calendar_days"] for row in observations
            ),
            "changed_rising_real_yield_states": sum(
                row["rising_state_changed"] for row in comparisons
            ),
            "changed_rising_state_query_sessions": [
                row["query_session"] for row in comparisons if row["rising_state_changed"]
            ],
            "observations": observations,
            "signal_comparisons": comparisons,
            "source_signal_comparisons_computed": True,
            "exact_intraday_first_publication_verified": False,
            "original_results_or_forward_rules_rewritten": False,
            "new_strategy_evaluations": 0,
            "new_economic_factor_definitions": 0,
            "ledger_unchanged": True,
            "investment_objective_verified": False,
            "order_authority": False,
        }
        output = local_path(output)
        new_output_directory(output)
        write_json(output / "results.json", result)
        write_text_atomic(
            output / "selected-observations.csv", pd.DataFrame(observations).to_csv(index=False)
        )
        write_text_atomic(
            output / "signal-comparisons.csv", pd.DataFrame(comparisons).to_csv(index=False)
        )
        return result
    finally:
        program.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only historical real-yield provenance; never orders."
    )
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument(
        "--registration",
        type=Path,
        default=ROOT / "data/real-yield-vintage-diagnostic-20261011/registration.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / (
        "data/real-yield-vintage-diagnostic-20261011"
        if args.action == "register"
        else "reports/real-yield-vintages-20261011/primary"
    )
    try:
        result = register(output) if args.action == "register" else run(args.registration, output)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in result.items()
                    if key not in ("policy", "observations", "signal_comparisons")
                },
                indent=2,
            )
        )
    except QuantError as exc:
        parser.exit(2, f"Real-yield vintage diagnostic blocked: {exc}\n")


if __name__ == "__main__":
    main()
