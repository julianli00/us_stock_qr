from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from itertools import zip_longest
from pathlib import Path

import pandas as pd

from us_quant import ssd_commitment_research_v1 as common
from us_quant import ssd_research_v1 as base
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.data import MarketData
from us_quant.dual_benchmark_v1 import research_job, runtime_guard
from us_quant.factor_family_sources import definitions
from us_quant.research_program import (
    engine_fingerprint,
    review_market,
    validate_factor,
)
from us_quant.research_program import validate_policy as validate_original_policy
from us_quant.storage import file_digest, read_json, utc_now

ROOT = base.ROOT
POLICY = ROOT / "config/ssd-factor-catalog-v1.json"
FOLDER = base.REPORTS / "factor-catalog-20261028-v1"
DATA = ROOT / "data/ssd-factor-catalog-20261028-v1"
INTENT = FOLDER / "intent.json"
ADMISSION = FOLDER / "admission.json"
CATALOG = FOLDER / "catalog.json"
PUBLIC = ROOT / "evidence/ssd_factor_catalog_20261028_v1.json"
PREPARED = ROOT / "data/six-factor-prepared-20261011/sources/market-reference.json"
CORE = ("SPY", "IEF", "GLD", "BIL", "MTUM", "VLUE", "QUAL", "USMV")
EXISTING = (*CORE, "TLT", "QQQ", "XLK", "SOXX")
ADDED = ("IJR", "PKW")
FIELDS = ("open", "close", "raw_close", "volume")
SEMANTICS = {
    "size_exposure_ijr_v1": "ee74c4016f5a29de5fff0f4ceebc3084685b5248a180c4efd9230fc40afdfada",
    "net_buyback_exposure_pkw_v1": (
        "fea841cfac9dc1942658e0f71558de556" "478bba9830d5f0504c6dc17e17cd056"
    ),
}
AUDIT_SHA = "74e5571963b2864e874f7ebfc59c669fe6f94f7bc311f53041e69bfbf45a5002"
JOURNAL_HEAD = "d7944ac455326ecc8a6c6f6f859ce47ed05dad0899874bbf4d2f8c69620be771"


def validate_policy(policy: dict) -> None:
    expected = {
        "schema_version": 1,
        "catalog_id": "ssd_factor_catalog_20261028_v1",
        "data_scope": "factor_etf_portfolio",
        "appended_symbols": list(ADDED),
        "factor_semantic_sha256": SEMANTICS,
        "cycle_timezone": "Asia/Shanghai",
        "maximum_new_definitions_this_cycle": 2,
        "maximum_readiness_age_days": 8,
        "configuration_count_limit_before_deadline": None,
        "prior_completed_configurations": 185,
        "prior_journal_sequence": 82,
        "prior_journal_head_sha256": JOURNAL_HEAD,
        "preserve_existing_price_cell_text": True,
        "reuse_exact_original_risk_free_reference": True,
        "rewrite_legacy_ledger_or_plans": False,
        "new_market_requests": False,
        "strategy_accounts_or_metrics": False,
        "order_authority": False,
    }
    references = {
        "source_audit": (
            "reports/ssd-continuous-research-v1/prior-art-after-option-latch-v1/"
            "source-revalidation-v1.json",
            AUDIT_SHA,
        ),
        "factor_proposal": (
            "config/factor-family-expansion.json",
            "6109980a5f0d1bdb0518b1c7017eef5dde6fd7c6b93c5c4d981d57bb75a08bdf",
        ),
        "preserved_program_policy": (
            "config/research-program.json",
            "c1bb4bfc9209233f1590d3808f515250ae428fc5ad881da49d4c1e6462f29ebc",
        ),
        "preserved_benchmark_admission": (
            "reports/ssd-continuous-research-v1/benchmarks-20261027-v1/admission.json",
            "d233035837a887e98a86aea3e6f3aaa52407762ab4db81bba75086e7130c7f4f",
        ),
        "protocol": ("config/dual-benchmark-v1.json", base.PROTOCOL_SHA256),
    }
    expected.update(
        {name: {"path": path, "sha256": digest} for name, (path, digest) in references.items()}
    )
    if policy != expected or any(
        type(policy[key]) is not type(value) for key, value in expected.items()
    ):
        raise QuantError("The fixed source/catalog extension policy changed.")


def fresh_time(checked_at: str, now: str, maximum_days: int = 8) -> None:
    checked, current = pd.Timestamp(checked_at), pd.Timestamp(now)
    if (
        pd.isna(checked)
        or pd.isna(current)
        or checked.tzinfo is None
        or current.tzinfo is None
        or checked > current
        or current - checked > pd.Timedelta(days=maximum_days)
    ):
        raise QuantError("Require the actual, timezone-aware, fresh cached-source audit.")


def append_csv(original: str, additions: str) -> str:
    left = csv.reader(io.StringIO(original))
    right = csv.reader(io.StringIO(additions))
    left_header, right_header = next(left, None), next(right, None)
    if left_header != ["", *EXISTING] or right_header != ["", *CORE, *ADDED]:
        raise QuantError("Append only IJR/PKW to the unchanged twelve-asset CSV schema.")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow([*left_header, *ADDED])
    count = 0
    for a, b in zip_longest(left, right):
        if a is None or b is None or len(a) != len(left_header) or len(b) != len(right_header):
            raise QuantError("Source panels have different row counts or malformed rows.")
        if a[0] != b[0]:
            raise QuantError("Source panels do not have the same ordered exchange sessions.")
        writer.writerow([*a, *b[-2:]])
        count += 1
    if count == 0:
        raise QuantError("An empty source panel cannot be admitted.")
    return output.getvalue()


def exact_arrays(original: MarketData, additions: MarketData, extended: MarketData) -> None:
    if tuple(extended.close.columns) != (*EXISTING, *ADDED):
        raise QuantError("The new market must contain exactly the preserved and added assets.")
    for field in FIELDS:
        if not getattr(original, field).equals(getattr(extended, field).loc[:, list(EXISTING)]):
            raise QuantError(f"The original {field} values changed.")
        if (
            not getattr(additions, field)
            .loc[:, list(ADDED)]
            .equals(getattr(extended, field).loc[:, list(ADDED)])
        ):
            raise QuantError(f"The admitted IJR/PKW {field} values changed.")
    if not original.risk_free.equals(extended.risk_free):
        raise QuantError("Reuse the exact original lagged risk-free series.")


def definition_records(proposals: dict, legacy: dict, at: str) -> list[dict]:
    factors = definitions(proposals)
    clock = pd.Timestamp(at)
    if (
        pd.isna(clock)
        or clock.tzinfo is None
        or clock < pd.Timestamp(proposals["eligible_not_before"])
    ):
        raise QuantError("The new registration needs its actual eligible clock.")
    if set(factors) != set(SEMANTICS) or len(factors) > 2:
        raise QuantError("This version admits only the two queued economic definitions.")
    existing_ids = {row["id"] for row in legacy["factors"]}
    existing_semantics = {row["semantic_sha256"] for row in legacy["factors"]}
    records = []
    for identifier, factor in factors.items():
        semantic = validate_factor(factor)
        if (
            semantic != SEMANTICS[identifier]
            or identifier in existing_ids
            or semantic in existing_semantics
        ):
            raise QuantError("Do not duplicate or silently rename an existing factor definition.")
        records.append({"factor": factor, "semantic_sha256": semantic, "registered_at": at})
    return records


def source_inputs() -> tuple[dict, dict, dict, dict]:
    policy = read_json(POLICY)
    validate_policy(policy)
    for key in ("factor_proposal", "preserved_program_policy", "protocol"):
        base.checked_reference(policy[key])
    original_policy = read_json(base.checked_reference(policy["preserved_program_policy"]))
    validate_original_policy(original_policy)
    audit = read_json(base.checked_reference(policy["source_audit"]))
    fresh_time(audit["created_at"], utc_now(), policy["maximum_readiness_age_days"])
    if (
        audit["strategy_accounts_or_metrics_computed"] is not False
        or audit["new_source_admission_performed"] is not False
        or audit["new_factor_registrations"] != 0
        or audit["journal_events_unchanged"] != 82
        or audit["journal_head_sha256_unchanged"] != JOURNAL_HEAD
        or audit["legacy_ledger_sha256_unchanged"] != base.LEGACY_SHA256
    ):
        raise QuantError("Use the actual completed source-only audit, not a new success label.")
    intent = read_json(base.checked_reference(audit["intent"]))
    if len(intent["input_files"]) != audit["retained_input_files_unchanged"]:
        raise QuantError("The source audit input graph is incomplete.")
    for reference in intent["input_files"]:
        base.checked_reference(reference)
    old = read_json(base.checked_reference(policy["preserved_benchmark_admission"]))
    if old["protocol_sha256"] != base.PROTOCOL_SHA256 or old["symbols"] != list(EXISTING):
        raise QuantError("Preserve the actual original dual-benchmark admission.")
    return policy, audit, old, original_policy


def legacy_catalog(week: str) -> dict:
    ledger = ROOT / "runtime/research-program.sqlite3"
    wal = ledger.with_name(ledger.name + "-wal")
    if (
        file_digest(ledger) != base.LEGACY_SHA256
        or engine_fingerprint() != base.LEGACY_ENGINE_SHA256
        or wal.exists()
        and wal.stat().st_size != 0
    ):
        raise QuantError("The preserved original ledger, engine or WAL changed.")
    with closing(sqlite3.connect(ledger.as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        rows = list(
            db.execute("SELECT id,semantic_sha,registered_at,body FROM factors ORDER BY id")
        )
        head = db.execute("SELECT seq,sha FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        if db.execute("SELECT 1 FROM cycles WHERE week=?", (week,)).fetchone() is not None:
            raise QuantError("Do not reset or duplicate an already recorded original weekly cycle.")
    factors = []
    for row in rows:
        body = json.loads(row["body"])
        if row["semantic_sha"] != validate_factor(body) or body["id"] != row["id"]:
            raise QuantError("An original factor identity is inconsistent.")
        factors.append(
            {
                "id": row["id"],
                "family": body["family"],
                "semantic_sha256": row["semantic_sha"],
                "registered_at": row["registered_at"],
            }
        )
    if (
        len(factors) != 6
        or head["seq"] != 246
        or head["sha"] != "bb157fa3b5ebcaca94e736137211af58a198790dd92302c38da8f4e25463e65f"
    ):
        raise QuantError("The six-definition, sequence246 original catalog must remain unchanged.")
    return {
        "ledger_sha256": base.LEGACY_SHA256,
        "engine_sha256": base.LEGACY_ENGINE_SHA256,
        "event_sequence": head["seq"],
        "event_head_sha256": head["sha"],
        "factors": factors,
        "definition_count": len(factors),
        "family_count": len({row["family"] for row in factors}),
        "original_cycle_for_new_week_exists": False,
        "written_or_migrated": False,
    }


def new_text(path: Path, text: str) -> None:
    common.output_guard(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise QuantError(
                "Preserve the existing or partial admission file; do not overwrite."
            ) from exc
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        temporary.unlink(missing_ok=True)


def catalog_totals(legacy: dict, records: list[dict]) -> dict:
    return {
        "combined_definition_count": len(legacy["factors"]) + len(records),
        "combined_family_count": len(
            {row["family"] for row in legacy["factors"]}
            | {row["factor"]["family"] for row in records}
        ),
    }


def capabilities(policy: dict) -> dict:
    return {
        key: {
            "verified": True,
            "evidence_path": policy["source_audit"]["path"],
            "evidence_sha256": policy["source_audit"]["sha256"],
        }
        for key in (
            "factor_mandates",
            "actual_fund_history",
            "post_inception_and_actions",
            "unleveraged_fund_identity",
        )
    }


def public_body(catalog: dict) -> dict:
    admission = read_json(base.checked_reference(catalog["admission"]))
    return {
        "schema_version": 1,
        "catalog_id": catalog["catalog_id"],
        "registered_at": catalog["registered_at"],
        "week": catalog["week"],
        "policy": catalog["policy"],
        "private_catalog": base.reference(CATALOG),
        "legacy": catalog["legacy"],
        "new_factors": catalog["new_factors"],
        "new_definition_count": len(catalog["new_factors"]),
        "combined_definition_count": catalog["combined_definition_count"],
        "combined_family_count": catalog["combined_family_count"],
        "source_audit": catalog["source_audit"],
        "admission": {
            "receipt": catalog["admission"],
            "checked_at": admission["checked_at"],
            "recorded_at": admission["recorded_at"],
            "sessions": admission["sessions"],
            "preserved_symbols": list(EXISTING),
            "appended_symbols": list(ADDED),
            "original_price_arrays_exactly_preserved": True,
            "original_risk_free_reference_and_array_preserved": True,
            "readiness_age_limit_days": 8,
        },
        "historical_W42_plans_reclassified": False,
        "original_ledger_written_or_migrated": False,
        "new_strategy_evaluations": 0,
        "strategy_accounts_or_metrics_computed": False,
        "registered_before_new_candidate_selection": True,
        "independent_alpha_or_forward_proof": False,
        "order_authority": False,
        "limitations": admission["limitations"],
    }


def registered_catalog() -> tuple[MarketData, dict]:
    policy, audit, old, _ = source_inputs()
    public = read_json(PUBLIC)
    catalog = read_json(base.checked_reference(public["private_catalog"]))
    intent = read_json(base.checked_reference(catalog["intent"]))
    for reference in intent["implementation"]:
        base.checked_reference(reference)
    at = catalog["registered_at"]
    clock = pd.Timestamp(at)
    fresh_time(at, utc_now())
    week = clock.tz_convert(policy["cycle_timezone"]).strftime("%G-W%V")
    legacy = legacy_catalog(week)
    proposals = read_json(base.checked_reference(policy["factor_proposal"]))
    if (
        catalog["schema_version"] != 1
        or catalog["catalog_id"] != policy["catalog_id"]
        or catalog["policy"] != base.reference(POLICY)
        or catalog["week"] != week
        or catalog["legacy"] != legacy
        or catalog["new_factors"] != definition_records(proposals, legacy, at)
        or catalog["source_audit"] != policy["source_audit"]
        or catalog["new_strategy_evaluations"] != 0
        or type(catalog["new_strategy_evaluations"]) is not int
        or catalog["order_authority"] is not False
        or intent["order_authority"] is not False
        or intent["no_candidate_selected"] is not True
        or intent["journal_sequence"] != 82
        or intent["journal_head_sha256"] != JOURNAL_HEAD
        or any(
            catalog[key] != value
            for key, value in catalog_totals(legacy, catalog["new_factors"]).items()
        )
        or public != public_body(catalog)
    ):
        raise QuantError("The registered SSD factor catalog or public projection changed.")
    admission = read_json(base.checked_reference(catalog["admission"]))
    if (
        admission["schema_version"] != 1
        or admission["data_scope"] != "factor_etf_portfolio"
        or admission["checked_at"] != audit["created_at"]
        or admission["capabilities"] != capabilities(policy)
        or admission["source_audit"] != policy["source_audit"]
        or admission["preserved_benchmark_admission"] != policy["preserved_benchmark_admission"]
        or pd.Timestamp(at) < pd.Timestamp(admission["recorded_at"])
        or pd.Timestamp(admission["recorded_at"]) < pd.Timestamp(intent["created_at"])
        or admission["market"]["risk_free"] != old["market"]["risk_free"]
        or admission["protocol"] != policy["protocol"]
        or admission["strategy_accounts_or_metrics_computed"] is not False
        or admission["order_authority"] is not False
    ):
        raise QuantError("Source admission chronology, risk-free series or authority changed.")
    market = review_market(admission, ROOT)
    if admission["sessions"] != len(market.close) or len(market.close) != 2805:
        raise QuantError("Preserve all original source sessions in the new admission.")
    exact_arrays(
        review_market(old, ROOT), review_market({"market": read_json(PREPARED)}, ROOT), market
    )
    return market, catalog


def admit() -> dict:
    runtime_guard()
    with research_job(ROOT / "runtime/ssd-continuous-research-v1.lock"):
        for path in (INTENT, ADMISSION, CATALOG, PUBLIC, DATA):
            common.output_guard(path)
            if path.exists():
                raise QuantError("Inspect the existing admission; do not repeat or overwrite it.")
        policy, audit, old, original_policy = source_inputs()
        before = base.journal_events(base.JOURNAL)
        if len(before) != 82 or before[-1]["sha256"] != JOURNAL_HEAD:
            raise QuantError("Freeze another version if the disclosed prior trial history changed.")
        at = utc_now()
        week = pd.Timestamp(at).tz_convert(policy["cycle_timezone"]).strftime("%G-W%V")
        legacy = legacy_catalog(week)
        proposals = read_json(base.checked_reference(policy["factor_proposal"]))
        proposed = definition_records(proposals, legacy, at)
        if len(proposed) > original_policy["maximum_new_factors_per_cycle"]:
            raise QuantError("Keep the original two-definition cycle limit, not a strategy quota.")
        implementation = [
            base.reference(path)
            for path in (POLICY, Path(__file__), ROOT / "tests/test_ssd_factor_catalog_v1.py")
        ]
        write_new_receipt(
            INTENT,
            {
                "created_at": at,
                "implementation": implementation,
                "source_audit": policy["source_audit"],
                "legacy": legacy,
                "journal_sequence": len(before),
                "journal_head_sha256": JOURNAL_HEAD,
                "no_candidate_selected": True,
                "order_authority": False,
            },
        )
        common.progress(
            "admitting_ssd_factor_catalog",
            active_compute_job={"pid": os.getpid(), "kind": "factor_catalog_source_admission"},
        )
        try:
            additions = read_json(PREPARED)
            market = {"risk_free": old["market"]["risk_free"]}
            for field in FIELDS:
                runtime_guard()
                left = base.checked_reference(old["market"][field]).read_text()
                right = base.checked_reference(additions[field]).read_text()
                path = DATA / f"{field}.csv"
                new_text(path, append_csv(left, right))
                market[field] = base.reference(path)
            extended = review_market({"market": market}, ROOT)
            exact_arrays(
                review_market(old, ROOT),
                review_market({"market": additions}, ROOT),
                extended,
            )
            if len(extended.close) != 2805:
                raise QuantError("Retain all 2805 original source sessions.")
            limitations = [
                "A fresh cache check is not newly retrieved prices or refreshed mandates.",
                "Issuer/quote acquisition timestamps remain those recorded in October10.",
                "Current mandates do not prove historical index methodology continuity.",
                "S&P methodology403 and incomplete direct-stock PIT data remain.",
                "Size/buybacks can overlap quality/value and corporate leverage.",
                "Historical ETF proxies are not independent alpha or prospective validation.",
                "W42 plans remain unchanged and unregistered in the original ledger.",
            ]
            runtime_guard()
            write_new_receipt(
                ADMISSION,
                {
                    "schema_version": 1,
                    "recorded_at": utc_now(),
                    "checked_at": audit["created_at"],
                    "data_scope": "factor_etf_portfolio",
                    "market": market,
                    "sessions": len(extended.close),
                    "capabilities": capabilities(policy),
                    "protocol": policy["protocol"],
                    "source_audit": policy["source_audit"],
                    "preserved_benchmark_admission": policy["preserved_benchmark_admission"],
                    "strategy_accounts_or_metrics_computed": False,
                    "order_authority": False,
                    "limitations": limitations,
                },
            )
            source_inputs()
            if base.journal_events(base.JOURNAL) != before:
                raise QuantError("Source/catalog admission must not add or revise strategy trials.")
            for reference in implementation:
                base.checked_reference(reference)
            runtime_guard()
            registered_at = utc_now()
            records = definition_records(proposals, legacy, registered_at)
            registered_week = (
                pd.Timestamp(registered_at).tz_convert(policy["cycle_timezone"]).strftime("%G-W%V")
            )
            if registered_week != week:
                raise QuantError("A cycle rollover requires a fresh, explicitly dated admission.")
            catalog = {
                "schema_version": 1,
                "catalog_id": policy["catalog_id"],
                "registered_at": registered_at,
                "week": week,
                "policy": base.reference(POLICY),
                "intent": base.reference(INTENT),
                "admission": base.reference(ADMISSION),
                "source_audit": policy["source_audit"],
                "legacy": legacy,
                "new_factors": records,
                **catalog_totals(legacy, records),
                "new_strategy_evaluations": 0,
                "order_authority": False,
            }
            write_new_receipt(CATALOG, catalog)
            write_new_receipt(PUBLIC, public_body(catalog))
            registered_catalog()
            common.progress(
                "ssd_factor_catalog_admitted",
                active_compute_job=None,
                factor_catalog=base.reference(CATALOG),
                factor_catalog_public_evidence=base.reference(PUBLIC),
                factor_catalog_new_definitions=len(records),
                current_research="SSD catalog admitted; no new candidate selected",
                pending_steps=[
                    "publish_source_catalog_evidence",
                    "freeze_distinct_economic_hypothesis",
                ],
            )
            return {"catalog": base.reference(CATALOG), "public_evidence": base.reference(PUBLIC)}
        except (QuantError, OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
            common.progress(
                "ssd_factor_catalog_admission_blocked",
                active_compute_job=None,
                factor_catalog_failure=str(exc),
                partial_outputs_preserved=True,
                factor_catalog_registration_file_exists=CATALOG.exists(),
            )
            raise
        except KeyboardInterrupt:
            common.progress("ssd_factor_catalog_admission_interrupted", active_compute_job=None)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Admit the fixed SSD factor catalog; no strategy.")
    parser.add_argument("action", choices=("admit",))
    parser.parse_args()
    try:
        print(json.dumps(admit(), indent=2))
    except QuantError as exc:
        parser.exit(2, f"SSD factor catalog admission blocked: {exc}\n")


if __name__ == "__main__":
    main()
