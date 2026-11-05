from __future__ import annotations

import argparse
import json
from types import FunctionType

from us_quant import signed_style_activity_expert_inclusion_v2 as rules
from us_quant import ssd_style_activity_expert_inclusion_v1 as preserved
from us_quant.config import QuantError
from us_quant.storage import read_json

base, ROOT, template = preserved.base, preserved.ROOT, preserved.template
seven, financial, lifecycle = preserved.seven, preserved.financial, preserved.lifecycle
source = seven
reference, unique_refs = preserved.reference, preserved.unique_refs
load_admitted_expert = preserved.load_admitted_expert
FOLDER = base.REPORTS / "signed-style-activity-expert-inclusion-20261105-v2"
DESIGN = FOLDER / "design-decision-v1.json"
INVOCATION = FOLDER / "actual-invocation-v1.json"
INTENT = FOLDER / "financial-implementation-freeze-v1.json"
PREPARED = FOLDER / "prepared-target-bundle-v1.json"
RESULTS = FOLDER / "results.json"
PUBLIC = ROOT / "evidence/signed_style_activity_expert_inclusion_20261105_v2_results_v1.json"
CANDIDATES = rules.CANDIDATES
IDS = tuple(row["id"] for row in CANDIDATES)
PACKETS = {
    name: ROOT
    / "evidence"
    / f"signed_style_activity_expert_inclusion_20261105_v2_{i}_registration_v1.json"
    for i, name in enumerate(IDS)
}
OWN = (
    *preserved.OWN,
    "config/signed-style-activity-expert-inclusion-v2.json",
    "src/us_quant/signed_style_activity_expert_inclusion_v2.py",
    "src/us_quant/ssd_signed_style_activity_expert_inclusion_v2.py",
    "tests/test_signed_style_activity_expert_inclusion_v2.py",
    "SIGNED_STYLE_ACTIVITY_EXPERT_INCLUSION_V2.md",
)


def bound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


def _call(name, *args):
    return bound(getattr(preserved, name))(*args)


def events(kind, identifier):
    return _call("events", kind, identifier)


def check_frozen(frozen):
    return _call("check_frozen", frozen)


def dependencies():
    return _call("dependencies")


def validation_refs():
    return _call("validation_refs")


def validate_namespace(frozen):
    for candidate in CANDIDATES:
        rows = events("registered", candidate["id"])
        if rows and (
            len(rows) != 1 or rows[0]["body"] != {"candidate": candidate, "frozen": frozen}
        ):
            raise QuantError("Reject every foreign cohort identity before appending any new event.")


def inputs():
    result = _call("inputs")
    validate_namespace(result[2])
    return result


def source_inputs():
    data, experts, parent, refs, clocks, benchmarks = _call("source_inputs")
    collision = read_json(rules.POLICY)["preserved_unregistered_collision"]
    refs = unique_refs(
        [
            *refs,
            collision,
            reference(preserved.INTENT),
            reference(preserved.FOLDER / "namespace-collision-completion-v1.json"),
            reference(preserved.FOLDER / "identity-collision-diagnosis-v1.json"),
            reference(preserved.FOLDER / "actual-registration-execution-v1.json"),
        ]
    )
    for ref in refs:
        base.checked_reference(ref)
    return data, experts, parent, refs, clocks, benchmarks


def include(experts, extra):
    return _call("include", experts, extra)


def initialize():
    lifecycle.runtime_guard()
    if any(row["candidate_id"] in IDS for row in base.journal_events(base.JOURNAL)):
        raise QuantError(
            "Freeze only globally unused new identities; preserve every historical ID."
        )
    return _call("initialize")


def verify_targets(*args):
    return _call("verify_targets", *args)


def verify_retained_records(*args):
    return _call("verify_retained_records", *args)


def scoped(function, candidate):
    return _call("scoped", function, candidate)


def cohort_publication(*args):
    return _call("cohort_publication", *args)


def prepare(*args):
    return _call("prepare", *args)


def register():
    return _call("register")


def evaluate(commit):
    return _call("evaluate", commit)


def export():
    return _call("export")


def progress(phase, **fields):
    return bound(preserved.progress)(phase, **fields)


def main():
    parser = argparse.ArgumentParser(description="Namespace-corrected signed-activity cohort V2.")
    parser.add_argument("action", choices=("freeze", "register", "evaluate", "export"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.action == "evaluate" and args.commit is None:
        parser.error("Use actual remotely published complete-cohort preregistration.")
    actions = {"freeze": initialize, "register": register, "export": export}
    print(
        json.dumps(evaluate(args.commit) if args.action == "evaluate" else actions[args.action]())
    )


if __name__ == "__main__":
    main()
