from __future__ import annotations

import argparse
import json
import sys
from types import FunctionType
from xml.etree import ElementTree

import pandas as pd

from us_quant import rebalance_ticket_spillover_v1 as rules
from us_quant import ssd_retained_main_risk_v1 as prior
from us_quant.candidate_receipts import write_new_receipt
from us_quant.config import QuantError
from us_quant.storage import utc_now, write_json

ROOT = rules.ROOT
FOLDER = ROOT / "reports/ssd-continuous-research-v1/rebalance-ticket-spillover-20261108-v1"
CATALOGUE, INTENT = FOLDER / "source-inputs-v1.json", FOLDER / "analysis-intent-v1.json"
OUTPUT = FOLDER / "diagnostic-v1.json"
PUBLIC = ROOT / "evidence/rebalance_ticket_spillover_20261108_v1.json"
OWN = (
    "config/rebalance-ticket-spillover-v1.json",
    "src/us_quant/rebalance_ticket_spillover_v1.py",
    "src/us_quant/ssd_rebalance_ticket_spillover_v1.py",
    "tests/test_rebalance_ticket_spillover_v1.py",
    "REBALANCE_TICKET_SPILLOVER_V1.md",
    "tests/conftest.py",
)
content_path, read_json, file_digest = prior.content_path, prior.read_json, prior.file_digest
reference, checked_reference = prior.reference, prior.checked_reference


def dependencies():
    function = FunctionType(
        prior.saved.dependencies.__code__,
        {**prior.saved.dependencies.__globals__, **globals()},
        argdefs=prior.saved.dependencies.__defaults__,
    )
    function.__kwdefaults__ = prior.saved.dependencies.__kwdefaults__
    return function()


class ArchiveFeed:
    def __init__(self, path):
        self.path = content_path(path)

    def read(self, cutoff):
        with content_path(self.path).open() as stream:
            for line in stream:
                row = json.loads(line)
                observed = rules.day(
                    row["signal_session"] if row["kind"] == "decision" else row["session"]
                )
                if observed > cutoff:
                    break
                yield row


def progress(phase, **extra):
    actual = prior.saved.counts()
    rules.require(
        all(k not in extra or extra[k] == v for k, v in actual.items()),
        "A read-only diagnostic cannot alter authoritative financial counters.",
    )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "ticket_spillover_" + phase,
            "study_id": rules.STUDY,
            "active_compute_job": None,
            **actual,
            **extra,
            "new_financial_trials": 0,
            "investment_objective_verified": False,
            "order_authority": False,
        },
    )


def checked_inputs():
    policy, catalogue = read_json(rules.POLICY), read_json(CATALOGUE)
    rules.validate(policy)
    rules.require(
        reference(CATALOGUE) == policy["source_inputs"]
        and len(catalogue["books"]) == 8
        and prior.saved.counts() == catalogue["actual_authoritative_counts"]
        and catalogue["new_spillover_statistics_computed"] is False,
        "Keep the exact captured paired books and unchanged J1570 predecessor.",
    )
    refs = [
        reference(CATALOGUE),
        *catalogue["source_result_references"].values(),
        *catalogue["source_prices"].values(),
    ]
    for book in catalogue["books"].values():
        refs += [book["complete"], *book["inputs"].values()]
    for ref in refs:
        checked_reference(ref)
    return catalogue, refs


def validation():
    versions = sorted(FOLDER.glob("validation-source-v*.json"))
    rules.require(bool(versions), "Actual successful full-interface synthetic validation required.")
    proof = read_json(versions[-1])
    suites = (
        ElementTree.parse(checked_reference(proof["test_outputs"][0]))
        .getroot()
        .findall(".//testsuite")
    )
    names = {c.get("name", "").split("[", 1)[0] for s in suites for c in s.findall("testcase")}
    rules.require(
        suites
        and proof["exit_code"] == 0
        and proof["synthetic_only"] is True
        and proof["implementation_and_test_inputs"] == [reference(ROOT / name) for name in OWN]
        and not any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
        and {
            "test_archive_and_memory_same_JSONL_interface",
            "test_signal_band_spillover_partition",
            "test_commission_crossover_is_arithmetic_not_trade_cap",
            "test_future_does_not_change_prefix",
            "test_no_old_solver_or_diagnosis_called",
            "test_banned_reader_before_open",
        }.issubset(names),
        "Freeze passed arithmetic, chronology, real JSONL, old-entrypoint and reader guards.",
    )
    refs = []
    for version in versions:
        item = read_json(version)
        refs += [
            reference(version),
            *item["test_outputs"],
            item["validation_runner"],
            *item["versioned_prototypes"],
        ]
    return refs


def freeze():
    prior._guard()
    with prior.lifecycle.research_job():
        rules.require(
            not INTENT.exists() and not (FOLDER / "freeze-started-v1.json").exists(),
            "Never repeat a complete or partial first diagnostic freeze.",
        )
        catalogue, refs = checked_inputs()
        checks = validation()
        prior._permitted_refs(catalogue)
        write_new_receipt(FOLDER / "freeze-started-v1.json", {"started_at": utc_now()})
        write_new_receipt(
            INTENT,
            {
                "created_at": utc_now(),
                "criteria": read_json(rules.POLICY),
                "source_references": refs,
                "implementation_and_test_inputs": [reference(ROOT / p) for p in OWN],
                "complete_dependency_inputs": dependencies(),
                "validation_references": checks,
                "package_versions": prior._packages(),
                "python_version": sys.version,
                "actual_authoritative_counts": prior.saved.counts(),
                "new_spillover_statistics_computed": False,
                "new_financial_trials": 0,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="actual_checkpoint_then_one_readonly_audit")
        return reference(INTENT)


def checked_intent():
    intent = read_json(INTENT)
    rules.validate(intent["criteria"])
    for ref in (
        intent["source_references"]
        + intent["implementation_and_test_inputs"]
        + intent["complete_dependency_inputs"]
        + intent["validation_references"]
    ):
        checked_reference(ref)
    rules.require(
        prior._packages() == intent["package_versions"]
        and sys.version == intent["python_version"]
        and prior.saved.counts() == intent["actual_authoritative_counts"],
        "Frozen inputs, environment or financial counters changed.",
    )
    return intent


def _prices(ref):
    return pd.read_csv(
        checked_reference(ref), index_col=0, parse_dates=True, float_precision="round_trip"
    )


def _scan(directory, opening, closing, book, end):
    prior.saved.new_output_directory(directory)
    trace = directory / "input-resolution.jsonl"
    with content_path(trace).open("x") as stream:

        def retain(row):
            stream.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            )
            stream.flush()

        result = rules.scan(
            opening,
            closing,
            ArchiveFeed(checked_reference(book["inputs"]["chronological-trace.jsonl"])),
            book["saved_metrics"]["start"],
            end,
            5.0 if book["path_key"].endswith("base") else 20.0,
            1 if book["path_key"].endswith("base") else 2,
            retain=retain,
            before=prior.lifecycle.runtime_guard,
        )
    write_new_receipt(directory / "summary.json", result)
    return result, reference(trace)


def audit():
    prior._guard()
    with prior.lifecycle.research_job():
        intent = checked_intent()
        rules.require(
            not OUTPUT.exists() and not (FOLDER / "audit-started-v1.json").exists(),
            "Preserve a completed, failed or partial frozen analysis; do not rerun it.",
        )
        cp = read_json(FOLDER / "actual-postfreeze-checkpoint-v1.json")
        rules.require(
            cp["entire_state_readback_equal"] is True
            and cp["frozen_implementation"] == reference(INTENT)
            and pd.Timestamp(intent["created_at"])
            < pd.Timestamp(cp["actual_continuity_checkpoint_readback"]["saved_at"])
            <= pd.Timestamp(cp["checked_at"]),
            "An actual post-freeze checkpoint must precede diagnostic statistics.",
        )
        write_new_receipt(FOLDER / "audit-started-v1.json", {"started_at": utc_now()})
        catalogue, _ = checked_inputs()
        opening, closing = (_prices(catalogue["source_prices"][k]) for k in ("open", "close"))
        results, prefix_results = {}, []
        try:
            for key, book in catalogue["books"].items():
                summary, trace = _scan(
                    FOLDER / "books" / key.replace(":", "-"),
                    opening,
                    closing,
                    book,
                    book["saved_metrics"]["end"],
                )
                expected = book["saved_execution"]
                rules.require(
                    summary["formal_closes"] == book["saved_metrics"]["sessions"]
                    and summary["model_tickets"] == expected["model_order_count"]
                    and abs(summary["paid_cost_usd"] - expected["total_cost_usd"]) <= rules.MONEY
                    and abs(summary["actual_notional_usd"] - expected["executed_notional_usd"])
                    <= rules.MONEY,
                    "All actual tickets, costs, notional and original closes must be retained.",
                )
                results[key] = {
                    "summary": summary,
                    "retained_records": trace,
                    "saved_financial_context": book["saved_metrics"],
                }
                for cutoff in rules.CUTOFFS:
                    last = pd.Timestamp(cutoff)
                    if last < pd.Timestamp(book["saved_metrics"]["start"]):
                        prefix_results.append(
                            {
                                "book": key,
                                "cutoff": cutoff,
                                "status": "before_funding_not_applicable",
                            }
                        )
                        continue
                    directory = FOLDER / "prefixes" / (key.replace(":", "-") + "-" + cutoff)
                    prefix, actual_trace = _scan(
                        directory, opening.loc[:last], closing.loc[:last], book, cutoff
                    )
                    expected_trace = directory / "expected-input-resolution.jsonl"
                    with (
                        content_path(checked_reference(trace)).open() as source,
                        content_path(expected_trace).open("x") as target,
                    ):
                        for line in source:
                            if pd.Timestamp(json.loads(line)["session"]) > last:
                                break
                            target.write(line)
                    exact = file_digest(checked_reference(actual_trace)) == file_digest(
                        expected_trace
                    )
                    write_new_receipt(
                        directory / "comparison.json",
                        {
                            "book": key,
                            "cutoff": cutoff,
                            "all_record_prefixes_exact": exact,
                            "operands_retained_before_assertion": True,
                            "actual_trace": actual_trace,
                            "expected_trace": reference(expected_trace),
                        },
                    )
                    rules.require(exact, "Future records changed a past diagnostic prefix.")
                    prefix_results.append(
                        {
                            "book": key,
                            "cutoff": cutoff,
                            "status": "exact",
                            "comparison": reference(directory / "comparison.json"),
                        }
                    )
            rules.require(
                sum(v["summary"]["formal_closes"] for v in results.values()) == 15064
                and sum(p["status"] == "exact" for p in prefix_results) == 28
                and sum(p["status"] == "before_funding_not_applicable" for p in prefix_results)
                == 12,
                "Retain complete paired coverage and explicit inapplicable prefixes.",
            )
            write_new_receipt(
                OUTPUT,
                {
                    "completed_at": utc_now(),
                    "study_id": rules.STUDY,
                    "analysis_intent": reference(INTENT),
                    "results": results,
                    "prefixes": prefix_results,
                    "new_financial_trials": 0,
                    "actual_authoritative_counts": prior.saved.counts(),
                    "account_solver_BT_or_old_diagnosis_replayed": False,
                    "order_authority": False,
                },
            )
        except (QuantError, OSError, ValueError, KeyError, TypeError) as error:
            write_new_receipt(
                FOLDER / "audit-failure-v1.json",
                {
                    "failed_at": utc_now(),
                    "reason": str(error),
                    "partial_outputs_retained": True,
                    "new_financial_trials": 0,
                },
            )
            progress("frozen_failure_preserved", failure=str(error))
            raise
        progress("complete", next_step="sanitized_export_current_readonly_reporting")
        return reference(OUTPUT)


def export():
    prior._guard()
    with prior.lifecycle.research_job():
        result = read_json(OUTPUT)
        rules.require(
            prior.saved.counts() == result["actual_authoritative_counts"],
            "A read-only result cannot hide new financial activity.",
        )
        write_new_receipt(
            PUBLIC,
            {
                "schema_version": 1,
                "record_type": "paired_full_target_ticket_spillover_diagnosis",
                "study_id": rules.STUDY,
                "status": "complete_readonly_diagnostic",
                "actual_completed_at": result["completed_at"],
                "private_result_sha256": file_digest(OUTPUT),
                "analysis_intent_sha256": file_digest(INTENT),
                "books": {k: v["summary"] for k, v in result["results"].items()},
                "full_original_book_count": 8,
                "full_original_close_count": 15064,
                "actual_exact_diagnostic_prefixes": 28,
                "prefixes_before_funding_not_applicable_not_passes": 12,
                "raw_prices_targets_holdings_or_ticket_records_exported": False,
                "source_CAGR_Sharpe_DD_recomputed": False,
                "causal_fee_or_expert_alpha_attribution": False,
                "counterfactual_selective_rebalance_policy_evaluated": False,
                "new_financial_trials": 0,
                "actual_authoritative_counts": prior.saved.counts(),
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        return reference(PUBLIC)


def main():
    parser = argparse.ArgumentParser(description="Read-only paired ticket-spillover arithmetic.")
    parser.add_argument("action", choices=("freeze", "audit", "export"))
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](), indent=2))


if __name__ == "__main__":
    main()
