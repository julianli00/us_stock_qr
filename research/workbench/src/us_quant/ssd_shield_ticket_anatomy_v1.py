from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields
from importlib.metadata import distributions
from types import FunctionType
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from us_quant import shield_ticket_anatomy_v1 as rules
from us_quant import ssd_execution_gap_envelope_v1 as prior
from us_quant.calendar import previous_session, sessions
from us_quant.candidate_receipts import write_new_receipt
from us_quant.research_program import review_market
from us_quant.ssd_style_activity_financial_v1 import dependencies as _dependencies
from us_quant.storage import digest_json, read_json, utc_now, write_json

base, ROOT, lifecycle = prior.base, prior.ROOT, prior.lifecycle
reference, unique_refs, counts = prior.reference, prior.unique_refs, prior.counts
FOLDER = base.REPORTS / "shield-ticket-anatomy-20261107-v1"
INTENT = FOLDER / "analysis-intent-v1.json"
OUTPUT = FOLDER / "diagnostic-v1.json"
PUBLIC = ROOT / "evidence/shield_ticket_anatomy_20261107_v1.json"
OWN = (
    "config/shield-ticket-anatomy-v1.json",
    "src/us_quant/shield_ticket_anatomy_v1.py",
    "src/us_quant/ssd_shield_ticket_anatomy_v1.py",
    "tests/test_shield_ticket_anatomy_v1.py",
    "SHIELD_TICKET_ANATOMY_V1.md",
    "tests/conftest.py",
)


def dependencies():
    return FunctionType(
        _dependencies.__code__,
        {**_dependencies.__globals__, **globals()},
        argdefs=_dependencies.__defaults__,
    )()


def progress(phase, **extra):
    actual = counts()
    rules.require(
        all(k not in extra or extra[k] == v for k, v in actual.items()),
        "A descriptive ticket study cannot replace financial totals.",
    )
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "shield_ticket_anatomy_" + phase,
            "study_id": read_json(rules.POLICY)["study_id"],
            "active_compute_job": None,
            **actual,
            **extra,
            "new_financial_trials": 0,
            "investment_objective_verified": False,
            "order_authority": False,
        },
    )


def source_inputs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    catalogue = read_json(base.checked_reference(policy["source_inputs"]))
    closed = read_json(base.checked_reference(catalogue["last_completed_cycle"]))
    rules.require(
        catalogue["new_ticket_buckets_or_provenance_labels_computed"] is False
        and catalogue["strategy_account_solver_BT_or_old_diagnosis_replayed"] is False
        and counts() == closed["actual_authoritative_counts"]
        and counts()["current_journal_event_count"] == 1506,
        "Use the original completed V2 books without source or accounting replay.",
    )
    refs = unique_refs(
        [
            *closed["exact_evidence_references"],
            policy["source_inputs"],
            *[
                v
                for v in catalogue.values()
                if isinstance(v, dict) and set(v) == {"path", "sha256"}
            ],
            *catalogue["source_market"].values(),
            *[
                v
                for c in catalogue["four_saved_books"].values()
                for v in c.values()
                if isinstance(v, dict) and set(v) == {"path", "sha256"}
            ],
            *[
                reference(FOLDER / n)
                for n in (
                    "capture-preflight-v1.py",
                    "actual-invocation-v1.json",
                    "design-decision-v1.json",
                    "preservation-baseline-v1.json",
                )
            ],
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    return catalogue, refs


def freeze():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        rules.require(
            not INTENT.exists() and not (FOLDER / "freeze-attempt-v1.json").exists(),
            "Preserve the first complete or partial ticket-study freeze.",
        )
        write_new_receipt(FOLDER / "freeze-attempt-v1.json", {"started_at": utc_now()})
        _, refs = source_inputs()
        proof = read_json(FOLDER / "validation-source-v1.json")
        suites = ElementTree.parse(base.checked_reference(proof["test_outputs"][0])).getroot()
        suites = suites.findall(".//testsuite")
        rules.require(
            suites
            and proof["exit_code"] == 0
            and proof["synthetic_only"] is True
            and proof["implementation_and_test_inputs"] == [reference(ROOT / p) for p in OWN]
            and proof["passed_tests"] == sum(int(s.get("tests", 0)) for s in suites)
            and not any(
                int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
            ),
            "Bind successful exact-source synthetic ticket and provenance checks.",
        )
        write_new_receipt(
            INTENT,
            {
                "frozen_at": utc_now(),
                "criteria": read_json(rules.POLICY),
                "source_references": refs,
                "complete_dependency_inputs": dependencies(),
                "implementation_and_test_inputs": [reference(ROOT / p) for p in OWN],
                "validation_evidence": [
                    *proof["test_outputs"],
                    reference(FOLDER / "validation-source-v1.json"),
                ],
                "python_version": sys.version,
                "package_versions": sorted(
                    (d.metadata["Name"], d.version) for d in distributions()
                ),
                "actual_authoritative_counts": counts(),
                "new_ticket_buckets_or_provenance_labels_computed": False,
                "new_financial_trials": 0,
                "order_authority": False,
            },
        )
        progress("frozen", next_step="checkpoint_then_inspect_saved_execution_tickets_once")
    return reference(INTENT)


class SavedBookAdapter:
    def __init__(self, data, nominal, case, cost, delay):
        self.data, self.nominal, self.cost, self.delay = data, nominal, cost, delay
        self.frame, self.weights, self.issued = (
            pd.read_csv(
                base.checked_reference(case[name]),
                index_col=0,
                parse_dates=True,
                float_precision="round_trip",
            )
            for name in ("account.csv", "weights.csv", "issued.csv")
        )
        saved = read_json(base.checked_reference(case["decisions.json"]))
        self.decisions = saved["decisions"]
        self.closes, observed_decisions, reference_inputs = {}, [], {}
        with base.checked_reference(case["chronological-trace.jsonl"]).open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["kind"] == "main_close":
                    day = pd.Timestamp(row["session"])
                    rules.require(day not in self.closes, "No repeated retained main close.")
                    self.closes[day] = row
                elif row["kind"] == "decision":
                    observed_decisions.append({k: v for k, v in row.items() if k != "kind"})
                elif row["kind"] == "reference_input":
                    day = pd.Timestamp(row["as_of"])
                    rules.require(day not in reference_inputs, "No repeated saved reference input.")
                    reference_inputs[day] = row
                else:
                    raise ValueError("An unknown retained trace record must not be discarded.")
        expected_dates = self.frame.index
        rules.require(
            len(self.frame) == case["expected_formal_rows"]
            and expected_dates.equals(sessions(expected_dates[0], expected_dates[-1]))
            and tuple(self.frame.columns)
            == (
                "equity",
                "return",
                "cash",
                "gross_exposure",
                "turnover",
                "cost",
                "orders",
                "risk_free",
            )
            and tuple(data.close.columns)
            == tuple(nominal.columns)
            == tuple(self.weights.columns)
            == tuple(self.issued.columns)
            == rules.ASSETS
            and self.weights.index.equals(expected_dates)
            and self.issued.index.equals(data.close.index)
            and nominal.index.equals(data.close.index)
            and self.frame["risk_free"].equals(data.risk_free.loc[expected_dates])
            and pd.DatetimeIndex(self.closes).equals(expected_dates)
            and observed_decisions == self.decisions
            and [r["signal_session"] for r in self.decisions]
            == pd.DatetimeIndex([previous_session(expected_dates[0]), *expected_dates])
            .strftime("%Y-%m-%d")
            .tolist(),
            (
                "Retain every original day,primitive,issued clock and canonical RF "
                "without replacement."
            ),
        )
        active = nominal.dropna(how="all")
        self.instructions, self.unfilled = {}, []
        prior_target = prior_nominal = prior_fraction = None
        issue_dates = []
        for row in self.decisions:
            if row["execution_session"] is None:
                continue
            signal, execution = (
                pd.Timestamp(row["signal_session"]),
                pd.Timestamp(row["execution_session"]),
            )
            target = np.asarray(row["desired_target"], dtype=float)
            known = active.loc[:signal]
            rules.require(
                not known.empty and signal in reference_inputs, "No future source fallback."
            )
            nominal_row = np.asarray(reference_inputs[signal]["nominal"], dtype=float)
            rules.require(
                np.array_equal(known.iloc[-1].to_numpy(), nominal_row)
                and np.array_equal(self.issued.loc[signal].to_numpy(), target)
                and row["assessment"]["source_opinion"]["source_opinion_known_at"]
                == str(known.index[-1].date()),
                "Use the exact original saved monthly value and its actual issued instruction.",
            )
            fraction = row["assessment"]["applied_nominal_fraction"]
            intent = rules.SavedIntent(
                signal,
                execution,
                known.index[-1],
                target,
                nominal_row,
                fraction,
                prior_target,
                prior_nominal,
                prior_fraction,
            )
            rules.require(execution not in self.instructions, "Do not merge pending instructions.")
            self.instructions[execution] = intent
            if execution > expected_dates[-1]:
                self.unfilled.append(
                    {
                        "signal_session": str(signal.date()),
                        "execution_session": str(execution.date()),
                        "provenance": intent.label(delay),
                        "no_formal_fee_backfill": True,
                    }
                )
            issue_dates.append(signal)
            prior_target, prior_nominal, prior_fraction = (
                target.copy(),
                nominal_row.copy(),
                fraction,
            )
        rules.require(
            pd.DatetimeIndex(issue_dates).equals(self.issued.dropna(how="all").index),
            "Every actual issued instruction and only those instructions must remain.",
        )
        terminal = {
            str(day.date()): intent.target.tolist()
            for day, intent in self.instructions.items()
            if day > expected_dates[-1]
        }
        rules.require(
            terminal == saved["pending_after_window"], "Preserve terminal unfilled requests."
        )

    def steps(self):
        previous_units = np.zeros(rules.N)
        previous_cash = previous_nav = 10000.0
        for day in self.frame.index:
            row, trace = self.frame.loc[day], self.closes[day]
            for name in self.frame.columns:
                field = "model_tickets" if name == "orders" else name
                rules.require(
                    float(row[name]).hex() == float(trace[field]).hex(),
                    "A saved CSV primitive differs from its retained original close.",
                )
            units = rules.vector(trace["units"])
            rules.require(
                np.array_equal(self.weights.loc[day].to_numpy(), trace["held_weights"])
                and np.max(
                    abs(
                        units * self.data.close.loc[day].to_numpy()
                        - np.asarray(trace["asset_values"])
                    )
                )
                <= rules.MONEY_TOLERANCE,
                "Use actual retained units and unchanged admitted price marks.",
            )
            yield rules.SavedStep(
                day,
                self.data.open.loc[day].to_numpy(),
                self.data.close.loc[day].to_numpy(),
                previous_units.copy(),
                units.copy(),
                previous_cash,
                float(trace["cash"]),
                previous_nav,
                float(trace["equity"]),
                float(trace["return"]),
                float(trace["cost"]),
                float(trace["turnover"]),
                trace["model_tickets"],
                self.cost,
                self.delay,
                self.instructions.get(day),
            )
            previous_units, previous_cash, previous_nav = (
                units.copy(),
                trace["cash"],
                trace["equity"],
            )


def input_record(step):
    result = {}
    for field in fields(step):
        value = getattr(step, field.name)
        if isinstance(value, pd.Timestamp):
            value = str(value.date())
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, rules.SavedIntent):
            value = {
                key.name: (
                    str(getattr(value, key.name).date())
                    if isinstance(getattr(value, key.name), pd.Timestamp)
                    else getattr(value, key.name).tolist()
                    if isinstance(getattr(value, key.name), np.ndarray)
                    else getattr(value, key.name)
                )
                for key in fields(value)
            }
        result[field.name] = value
    return result


def exact_encoding(value):
    if isinstance(value, float):
        return {"binary64": value.hex()}
    if isinstance(value, list):
        return [exact_encoding(v) for v in value]
    if isinstance(value, dict):
        return {k: exact_encoding(v) for k, v in value.items()}
    return value


def analyze():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        rules.require(
            not OUTPUT.exists() and not (FOLDER / "analysis-started-v1.json").exists(),
            "Preserve completed or partial ticket anatomy rather than replaying it.",
        )
        frozen = read_json(INTENT)
        for ref in (
            frozen["source_references"]
            + frozen["complete_dependency_inputs"]
            + frozen["implementation_and_test_inputs"]
            + frozen["validation_evidence"]
        ):
            base.checked_reference(ref)
        rules.require(
            sys.version == frozen["python_version"]
            and counts() == frozen["actual_authoritative_counts"]
            and [list(v) for v in sorted((d.metadata["Name"], d.version) for d in distributions())]
            == frozen["package_versions"],
            "The original source,environment or financial journal changed.",
        )
        catalogue, refs = source_inputs()
        rules.require(refs == frozen["source_references"], "Keep every exact source reference.")
        start = read_json(FOLDER / "actual-analysis-start-v1.json")
        rules.require(
            start["actual_continuity_checkpoint_readback"]["saved_at"] < start["started_at"],
            "An actual saved checkpoint must precede new ticket labels.",
        )
        write_new_receipt(
            FOLDER / "analysis-started-v1.json",
            {
                "started_at": utc_now(),
                "pid": os.getpid(),
                "analysis_intent": reference(INTENT),
            },
        )
        data = review_market({"market": catalogue["source_market"]}, ROOT)
        nominal = pd.read_csv(
            base.checked_reference(catalogue["source_nominal_targets"]),
            index_col=0,
            parse_dates=True,
            float_precision="round_trip",
        )
        outputs, prefixes = {}, []
        trace = FOLDER / "actual-inputs-resolved-v1.jsonl"
        total_rows = 0
        with trace.open("x") as stream:

            def retain(row):
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()

            for key in rules.PATHS:
                lifecycle.runtime_guard()
                case = catalogue["four_saved_books"][key]
                cost, delay = (5.0, 1) if key.endswith("base") else (20.0, 2)
                adapter = SavedBookAdapter(data, nominal, case, cost, delay)
                steps, records = [], []
                for step in adapter.steps():
                    retain({"record": "saved_input", "path": key, "input": input_record(step)})
                    row = rules.inspect(step)
                    retain({"record": "ticket_resolved", "path": key, "resolved": row})
                    steps.append(step)
                    records.append(row)
                summary = rules.aggregate(records)
                expected = case["expected_execution_totals"]
                rules.require(
                    summary["formal_rows"] == read_json(rules.POLICY)["original_formal_rows"][key]
                    and summary["model_tickets"] == expected["model_order_count"]
                    and abs(summary["paid_fees_USD"] - expected["total_cost_usd"])
                    <= rules.MONEY_TOLERANCE
                    and abs(summary["actual_notional_USD"] - expected["executed_notional_usd"])
                    <= rules.MONEY_TOLERANCE,
                    "Every original actual ticket,dollar notional and paid fee must reconcile.",
                )
                outputs[key] = {**summary, "unfilled_terminal_requests": adapter.unfilled}
                total_rows += len(records)
                for position in read_json(rules.POLICY)["prefix_positions"]:
                    length = (
                        len(records)
                        if position == "full"
                        else len(records) // 2
                        if (position == "half")
                        else position
                    )
                    clipped = [rules.inspect(step) for step in steps[:length]]
                    rules.require(
                        exact_encoding(clipped) == exact_encoding(records[:length]),
                        "A saved-step label prefix changed a prior primitive or scalar bit.",
                    )
                    prefixes.append(
                        {
                            "path": key,
                            "formal_rows": length,
                            "last_session": records[length - 1]["session"],
                            "all_saved_ticket_scalar_and_provenance_prefix_bits_exact": True,
                            "canonical_prefix_sha256": digest_json(exact_encoding(clipped)),
                            "prefix_summary": rules.aggregate(clipped),
                        }
                    )
        rules.require(
            total_rows == 7532 and len(prefixes) == 20, "Keep all original rows/prefixes."
        )
        write_new_receipt(
            OUTPUT,
            {
                "completed_at": utc_now(),
                "study_id": read_json(rules.POLICY)["study_id"],
                "status": "complete_read_only_saved_ticket_anatomy",
                "analysis_intent": reference(INTENT),
                "source_references": refs,
                "actual_authoritative_counts": counts(),
                "four_original_books": outputs,
                "original_formal_rows": total_rows,
                "input_resolution_records": 2 * total_rows,
                "actual_prefix_checks": prefixes,
                "retained_input_resolution_trace": reference(trace),
                "source_complete_financial_result": catalogue["source_complete_financial_result"],
                "strategy_account_solver_BT_or_old_diagnosis_replayed": False,
                "source_CAGR_Sharpe_drawdown_or_qualification_recomputed": False,
                "fee_free_or_fee_removed_policy_counterfactual_claimed": False,
                "causal_fee_or_expert_alpha_attribution_claimed": False,
                "new_financial_trials": 0,
                "independent_forward_validation": False,
                "investment_objective_verified": False,
                "order_authority": False,
            },
        )
        progress("analyzed", next_step="export_sanitized_aggregate_then_close_source_only_cycle")
    return reference(OUTPUT)


def export():
    lifecycle.runtime_guard()
    with lifecycle.research_job():
        saved = read_json(OUTPUT)
        rules.require(saved["actual_authoritative_counts"] == counts(), "Keep financial totals.")
        write_new_receipt(
            PUBLIC,
            {
                key: saved[key]
                for key in (
                    "completed_at",
                    "study_id",
                    "status",
                    "actual_authoritative_counts",
                    "four_original_books",
                    "original_formal_rows",
                    "input_resolution_records",
                    "source_complete_financial_result",
                    "strategy_account_solver_BT_or_old_diagnosis_replayed",
                    "source_CAGR_Sharpe_drawdown_or_qualification_recomputed",
                    "fee_free_or_fee_removed_policy_counterfactual_claimed",
                    "causal_fee_or_expert_alpha_attribution_claimed",
                    "new_financial_trials",
                    "independent_forward_validation",
                    "investment_objective_verified",
                    "order_authority",
                )
            }
            | {
                "schema_version": 1,
                "record_type": "saved_shield_ticket_anatomy_aggregate",
                "private_diagnostic_sha256": reference(OUTPUT)["sha256"],
                "all_twenty_saved_record_prefixes_bit_exact": all(
                    p["all_saved_ticket_scalar_and_provenance_prefix_bits_exact"]
                    for p in saved["actual_prefix_checks"]
                ),
                "raw_prices_units_targets_daily_rows_or_per_ticket_rows_exported": False,
            },
        )
    return reference(PUBLIC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read-only saved shield ticket anatomy.")
    parser.add_argument("action", choices=("freeze", "analyze", "export"))
    args = parser.parse_args()
    result = {"freeze": freeze, "analyze": analyze, "export": export}[args.action]()
    print(json.dumps(result, indent=2, allow_nan=False))
