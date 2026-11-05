from __future__ import annotations

import argparse
import json
from types import FunctionType

from us_quant import ssd_hmm_session_intervention_v1 as prior
from us_quant.config import QuantError
from us_quant.storage import read_json

base, ROOT, lifecycle, reference = prior.base, prior.ROOT, prior.lifecycle, prior.reference
FOLDER = base.REPORTS / "hmm-session-intervention-diagnosis-20261105-v2"
POLICY = ROOT / "config/hmm-session-intervention-diagnosis-v2.json"
INTENT = FOLDER / "analysis-intent-v1.json"
OUTPUT = FOLDER / "diagnostic-v1.json"
PUBLIC = ROOT / "evidence/hmm_session_intervention_diagnosis_20261105_v2.json"
OWN = (
    *prior.OWN,
    "config/hmm-session-intervention-diagnosis-v2.json",
    "src/us_quant/ssd_hmm_session_intervention_v2.py",
    "tests/test_hmm_session_intervention_v2.py",
    "HMM_SESSION_INTERVENTION_DIAGNOSIS_V2.md",
)


def bound(function):
    return FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )


def validate(policy):
    original = read_json(prior.POLICY)
    if policy["study_id"] != "ssd_hmm_session_intervention_diagnosis_20261105_v2":
        raise QuantError("The schema-only correction is a separate immutable V2 identity.")
    extra = {"schema_bridge", "terminal_v1", "original_numerical_intent"}
    if set(policy) != set(original) | extra:
        raise QuantError("Only explicit provenance bindings may extend the unchanged V1 criteria.")
    economic = {k: v for k, v in policy.items() if k not in extra}
    economic["study_id"] = original["study_id"]
    prior.validate(economic)
    if policy["original_numerical_intent"] != reference(prior.INTENT):
        raise QuantError("Keep the original pre-outcome numerical kernel and criteria.")
    if policy["terminal_v1"] != reference(prior.FOLDER / "completion-v1.json"):
        raise QuantError("Do not remove or reuse the failed V1 identity.")


def inputs():
    policy = read_json(POLICY)
    validate(policy)
    catalogue, refs = prior.inputs()
    original = read_json(base.checked_reference(policy["original_numerical_intent"]))
    for ref in original["implementation_and_test_inputs"]:
        base.checked_reference(ref)
    bridge = read_json(base.checked_reference(policy["schema_bridge"]))
    terminal = read_json(base.checked_reference(policy["terminal_v1"]))
    if (
        bridge["status"]
        != "all_split_records_exactly_parent_records_without_only_provenance_envelope_field"
        or bridge["only_missing_envelope_field"] != "source_variant"
        or bridge["exact_record_counts"] != {v: 242 for v in catalogue["variants"]}
        or bridge["split_tapes"] != {v: row["tape"] for v, row in catalogue["variants"].items()}
        or bridge["source_catalogue"] != policy["input_catalogue"]
        or bridge["values_or_opinions_or_RF_transformed"] is not False
        or terminal["numeric_intervention_outcomes_computed"] is not False
        or terminal["retry_of_this_identity_allowed"] is not False
    ):
        raise QuantError("The structural Adapter requires exact complete parent/split provenance.")
    base.checked_reference(bridge["parent_tagged_tape"])
    refs = [
        *refs,
        policy["schema_bridge"],
        policy["terminal_v1"],
        policy["original_numerical_intent"],
        bridge["parent_tagged_tape"],
        bridge["version1_failure_publication"],
    ]
    return catalogue, sorted({r["path"]: r for r in refs}.values(), key=lambda r: r["path"])


def _bindings():
    catalogue, _ = inputs()
    return base.checked_reference(catalogue["base"]["tape"]), {
        base.checked_reference(row["tape"]): variant
        for variant, row in catalogue["variants"].items()
    }


def _pairs(path):
    base_path, split_paths = _bindings()
    if path == base_path:
        yield from prior._pairs(path)
        return
    if path not in split_paths:
        raise QuantError("Unknown source paths cannot receive a provenance label.")
    variant = split_paths[path]
    for pair in prior._pairs(path):
        if any("source_variant" in row for row in pair):
            raise QuantError("A split record already carrying a label cannot be overwritten.")
        yield tuple({"source_variant": variant, **row} for row in pair)


def dependencies():
    return bound(prior.dependencies)()


def progress(*args, **kwargs):
    return bound(prior.progress)(*args, **kwargs)


def freeze():
    return bound(prior.freeze)()


def analyze():
    return bound(prior.analyze)()


def main():
    parser = argparse.ArgumentParser(
        description="Exact split-provenance Adapter, unchanged kernel."
    )
    parser.add_argument("action", choices=("freeze", "analyze"))
    args = parser.parse_args()
    print(json.dumps({"freeze": freeze, "analyze": analyze}[args.action]()))


if __name__ == "__main__":
    main()
