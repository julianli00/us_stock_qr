from copy import deepcopy
from types import FunctionType

import numpy as np
import pytest
import test_ssd_lagged_spy_price_impact_financial_v1 as existing

from us_quant import ssd_style_activity_financial_v1 as study
from us_quant.config import QuantError
from us_quant.storage import read_json


@pytest.fixture
def data(market_factory):
    return market_factory(start="2015-08-03", end="2019-12-31", symbols=study.law.UNIVERSE)


def native(data):
    records = []
    targets, audits = study.law.build_targets(
        data, read_json(study.law.POLICY), retain=records.append
    )
    return targets, audits, records


def packet(monkeypatch):
    fn = FunctionType(
        existing.packet.__code__,
        {**existing.packet.__globals__, **globals()},
        argdefs=existing.packet.__defaults__,
    )
    return fn(monkeypatch)


def bind_test(function):
    rebound = FunctionType(
        function.__code__, {**function.__globals__, **globals()}, argdefs=function.__defaults__
    )
    rebound.__dict__.update(function.__dict__)
    return rebound


for name in (
    "test_actual_financial_policy_preserves_full_OHLCV_law_without_hidden_knobs",
    "test_scope_and_original_financial_contract_cannot_drift",
    "test_schema3_defers_indicator_proof_and_preserves_old_namespace",
    "test_readiness_cannot_be_relabelled_actual_indicator_proof",
    "test_actual_publication_required_before_any_indicator_generation",
    "test_partial_source_is_never_silently_regenerated",
    "test_actual_source_Seam_passes_MarketData_not_close_frame",
    "test_failed_remote_fence_never_starts_source_or_capital",
    "test_source_adapter_preserves_original_money_and_scalar_BT",
    "test_real_money_prefix_exact_and_no_old_namespace_mutation",
):
    globals()[name] = bind_test(getattr(existing, name))


@pytest.mark.parametrize(
    "tamper", ["reported_price", "own_session", "state", "target", "missing_record", "extra_record"]
)
def test_independent_four_style_fraction_oracle_binds_all_native_state_and_targets(data, tamper):
    targets, audits, records = native(data)
    proof = study.verify_targets(data, targets, audits, records)
    assert proof["all_independent_signed_activity_targets_exact"] is True
    assert np.isfinite(proof["maximum_independent_scalar_numpy_balance_gap"])
    assert proof["source_price_volume_binary64_inputs_checked"] == 760 * len(audits)
    assert proof["independent_four_style_balance_checks"] == 4 * len(audits)
    if tamper == "reported_price":
        records[0]["inputs_binary64_hex"]["MTUM"]["reported_closes"][0] = (999.0).hex()
    elif tamper == "own_session":
        records[0]["activity_session_dates"][-1] = records[0]["decision_session"]
    elif tamper == "state":
        audits[0]["activity_balances"]["MTUM"] += 0.01
    elif tamper == "target":
        targets.loc[targets.dropna().index[0], "SPY"] = 0.1
    elif tamper == "missing_record":
        records.pop()
    else:
        records.append({})
    with pytest.raises(QuantError):
        study.verify_targets(data, targets, audits, records)


@pytest.mark.parametrize(
    "key,value",
    [
        ("primary_policy_or_market_or_RF_scope_changed", True),
        ("standalone_financial_failure_is_technical_exclusion", True),
        ("source_input_interface", "close_only"),
        ("prior_completed_configurations", 269),
    ],
)
def test_no_source_native_RF_family_or_prior_rescue(key, value):
    policy = deepcopy(read_json(study.POLICY))
    policy[key] = value
    with pytest.raises(QuantError):
        study.validate_policy(policy)


def test_new_adapter_does_not_mutate_frozen_primary_or_old_financial_namespace():
    assert not hasattr(study.primary, "CANDIDATE")
    assert study.law.CANDIDATE == study.CANDIDATE
    assert study.law.build_targets is study.primary.build_targets
    old = study.shared.ID, study.shared.FOLDER, study.shared.PACKET, study.shared.law
    paths = {r["path"] for r in study.dependencies()}
    assert set(study.OWN).issubset(paths)
    assert "tests/test_ssd_lagged_spy_price_impact_financial_v1.py" in paths
    assert "tests/conftest.py" in paths
    assert old == (study.shared.ID, study.shared.FOLDER, study.shared.PACKET, study.shared.law)
