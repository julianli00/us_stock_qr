from __future__ import annotations

import argparse
import json
from types import FunctionType

from us_quant import forecast_risk_normalizer_diagnosis_v2 as rules
from us_quant import ssd_forecast_risk_normalizer_diagnosis_v1 as prior
from us_quant.config import QuantError
from us_quant.ssd_style_activity_financial_v1 import dependencies as _dependencies
from us_quant.storage import read_json, utc_now, write_json

ROOT, base, lifecycle = prior.ROOT, prior.base, prior.lifecycle
reference, unique_refs, packages = prior.reference, prior.unique_refs, prior.packages
current_counts = prior.current_counts
FOLDER = base.REPORTS / "forecast-risk-normalizer-diagnosis-20261106-v2"
INTENT = FOLDER / "diagnostic-implementation-freeze-v2.json"
RESULTS = FOLDER / "results.json"
PUBLIC = ROOT / "evidence/forecast_risk_normalizer_diagnosis_20261106_v2.json"
OWN = (
    "config/forecast-risk-normalizer-diagnosis-v2.json",
    "src/us_quant/forecast_risk_normalizer_diagnosis_v2.py",
    "src/us_quant/ssd_forecast_risk_normalizer_diagnosis_v2.py",
    "tests/test_forecast_risk_normalizer_diagnosis_v2.py",
    "FORECAST_RISK_NORMALIZER_DIAGNOSIS_V2.md",
    "tests/test_forecast_risk_normalizer_diagnosis_v1.py",
    "tests/conftest.py",
)


def bound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


def dependencies():
    return bound(_dependencies)()


def source_refs():
    policy = read_json(rules.POLICY)
    rules.validate(policy)
    keys = (
        "source_semantic_policy",
        "source_failed_diagnostic_freeze",
        "source_terminal_failure",
        "source_public_failure",
    )
    for key in keys:
        base.checked_reference(policy[key])
    failure = read_json(base.checked_reference(policy["source_terminal_failure"]))
    if failure["V1_will_not_be_patched_or_retried"] is not True:
        raise QuantError(
            "Preserve the exact terminal V1 partial diagnosis and original source keys."
        )
    refs = unique_refs([*prior.source_refs(), *[policy[k] for k in keys]])
    for ref in refs:
        base.checked_reference(ref)
    return refs


def source_audits():
    return prior.source_audits()


def progress(phase, **fields):
    write_json(
        ROOT / "reports/ssd-november-research-20261103-v1/progress.json",
        {
            "updated_at_utc": utc_now(),
            "phase": "forecast_risk_normalizer_v2_" + phase,
            "study_id": read_json(rules.POLICY)["study_id"],
            "active_compute_job": None,
            **fields,
            **current_counts(base.journal_events(base.JOURNAL)),
            "order_authority": False,
        },
    )


def initialize():
    return bound(prior.initialize)()


def check_frozen():
    return bound(prior.check_frozen)()


def run():
    return bound(prior.run)()


def main():
    parser = argparse.ArgumentParser(
        description="Exact original-source scalar-key decoder diagnosis."
    )
    parser.add_argument("action", choices=("freeze", "analyze"))
    args = parser.parse_args()
    print(json.dumps(initialize() if args.action == "freeze" else run()))


if __name__ == "__main__":
    main()
