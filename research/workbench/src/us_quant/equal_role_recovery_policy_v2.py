from pathlib import Path

from us_quant import equal_role_projection_v1 as original
from us_quant.nested_growth_allocation_audit_v1 import require
from us_quant.storage import read_json

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "config/equal-role-projection-v2.json"
ID = "expert_pool_six_equal_role_projection_v2"
CANDIDATES = ({"id": ID, "weighting": original.CANDIDATES[0]["weighting"]},)
EXPERT_IDS, MECHANISMS, CLOCK = original.EXPERT_IDS, original.MECHANISMS, original.CLOCK


def validate(policy):
    original.validate(read_json(original.POLICY))
    expected = {
        "schema_version": 2,
        "study_id": "ssd_equal_role_projection_20261106_v2",
        "actual_native_event_seq": 192225,
        "prior_journal_sequence": 1466,
        "prior_journal_head_sha256": (
            "6d2b3713492b17d3df6c1a5bd35975581d70533d4bfef98efe77f6fddc3fda78"
        ),
        "source_inputs": {
            "path": (
                "reports/ssd-continuous-research-v1/equal-role-projection-20261106-v2/"
                "source-inputs-v2.json"
            ),
            "sha256": "bd8c3ac0a9931ff43e732cb18080c4dcc3e551e0169611d42a58b82f1af0319a",
        },
        "economic_rule_or_tolerance_changed": False,
        "failed_V1_reopened": False,
        "cached_targets_verified_not_regenerated": True,
        "proof_interface": "target_proof(frozen, fields)",
        "new_source_requests": 0,
        "new_shadow_account_runs": 0,
        "order_authority": False,
    }
    require(
        policy == expected and all(type(policy[k]) is type(v) for k, v in expected.items()),
        "Repair only the proof Interface under an explicit new identity.",
    )
