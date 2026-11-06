from __future__ import annotations

from us_quant import quintet_return_cost_bridge_v1 as prior
from us_quant.storage import read_json

ROOT = prior.ROOT
POLICY = ROOT / "config/quintet-return-cost-bridge-v2.json"
ROLES, PATHS, INITIAL = prior.ROLES, prior.PATHS, prior.INITIAL
TOLERANCE, MONEY_TOLERANCE, COLUMNS = prior.TOLERANCE, prior.MONEY_TOLERANCE, prior.COLUMNS
require = prior.require
decompose, compare = prior.decompose, prior.compare
paired_summary, check_prefix = prior.paired_summary, prior.check_prefix


def validate(policy):
    expected = {
        **read_json(prior.POLICY),
        "schema_version": 2,
        "study_id": "ssd_quintet_return_cost_bridge_20261106_v2",
        "same_economic_question_continuation": True,
        "original_V1_reader_contract_remains_failed": True,
        "only_reader_delivery_corrected_to_original_canonical_function": True,
        "original_V1_arithmetic_and_all_limits_reused_unchanged": True,
        "source_RF_reader": "us_quant.research_program.review_market",
        "new_RF_view_or_source_adapter": False,
        "reader_continuation_inputs": {
            "path": (
                "reports/ssd-continuous-research-v1/quintet-return-cost-bridge-20261106-v2/"
                "reader-continuation-inputs-v2.json"
            ),
            "sha256": "37382ce26c813fd4691a0c25c7b81ed8603f9cfe9b08fe074783da37b206ab06",
        },
    }
    require(
        policy == expected and all(type(policy[k]) is type(v) for k, v in expected.items()),
        "Version original reader delivery only; keep V1 failed and every exact limit unchanged.",
    )
