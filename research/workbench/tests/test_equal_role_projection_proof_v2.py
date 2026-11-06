import copy

import numpy as np
import pandas as pd
import pytest

from us_quant import equal_role_projection_proof_v2 as proof
from us_quant.config import QuantError
from us_quant.storage import digest_json, read_json


def fixture():
    calendar = proof.coordinates.sessions("2016-10-05", "2026-10-05")
    dates = proof.source.decision_sessions(calendar, read_json(proof.source.POLICY))
    targets = pd.DataFrame(np.nan, index=calendar, columns=proof.coordinates.ASSETS)
    records, previous = [], None
    opinions = np.zeros((6, 14))
    for i in range(6):
        opinions[i, i] = 0.98
    for index, day in enumerate(dates):
        weights = [1 / 6] * 6 if index < 2 else [0.3, 0.1, 0.1, 0.2, 0.2, 0.1]
        projected, _ = proof.allocation.project(weights)
        target = projected @ opinions
        key = str(day.date())
        state = {"complete_intervals": max(0, index - 1)}
        old = {
            "decision_session": key,
            "expert_weights": weights,
            "adaptive_state": state,
            "score_mature": index >= 2,
        }
        before = {
            "phase": "before_role_projection",
            "source_before": {
                "decision_session": key,
                "known_target_dates": [key] * 6,
                "known_targets_hex": [[float(v).hex() for v in row] for row in opinions],
                "state": {"complete_intervals": 0}
                if previous is None
                else previous["adaptive_state"].copy(),
                "previous_weights_hex": [
                    float(v).hex()
                    for v in ([1 / 6] * 6 if previous is None else previous["expert_weights"])
                ],
            },
            "source_after": {"decision_session": key, "audit": old},
        }
        after = {
            "phase": "after_role_projection",
            "audit": {"decision_session": key, "projected_expert_weights": projected.tolist()},
            "unified_target_hex": [float(v).hex() for v in target],
        }
        targets.loc[day] = target
        records.append((before, after))
        previous = old
    return targets, records


def test_actual_two_argument_proof_interface_and_canonical_guards():
    targets, records = fixture()
    frozen = {"study_id": "synthetic-interface-only"}
    result = proof.verify_cached(targets, iter(records), frozen)
    assert result["frozen_specification_sha256"] == digest_json(frozen)
    assert result["candidate_accounts_or_metrics_computed"] is False
    assert result["original_source_signal_recomputed"] is False
    assert result["decision_count"] == 121
    assert result["cached_partial_targets_verified_not_regenerated"] is True
    assert result["failed_V1_identity_reopened"] is False


@pytest.mark.parametrize("change", ["partial", "extra", "future", "state", "target", "weights"])
def test_partial_or_changed_recovery_input_remains_failed(change):
    targets, original = fixture()
    records = copy.deepcopy(original)
    if change == "partial":
        records.pop()
    elif change == "extra":
        records.append(copy.deepcopy(records[-1]))
    elif change == "future":
        records[2][0]["source_before"]["known_target_dates"][0] = "2030-01-01"
    elif change == "state":
        records[2][0]["source_before"]["state"]["complete_intervals"] = 10
    elif change == "target":
        targets.loc[
            proof.source.decision_sessions(targets.index, read_json(proof.source.POLICY))[2], "SPY"
        ] += 0.001
    else:
        records[2][1]["audit"]["projected_expert_weights"][0] += 0.01
    with pytest.raises(QuantError):
        proof.verify_cached(targets, iter(records), {"study_id": "synthetic-interface-only"})
