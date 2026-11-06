from types import SimpleNamespace

import pandas as pd
import pytest

from us_quant import quintet_return_cost_bridge_v1 as old
from us_quant import quintet_return_cost_bridge_v2 as rules
from us_quant import ssd_quintet_return_cost_bridge_v2 as study
from us_quant.config import QuantError
from us_quant.storage import read_json


def test_v1_arithmetic_and_every_limit_are_the_same_function_objects():
    assert rules.decompose is old.decompose
    assert rules.compare is old.compare
    assert rules.paired_summary is old.paired_summary
    assert rules.check_prefix is old.check_prefix
    assert rules.TOLERANCE == old.TOLERANCE == 1e-12
    assert rules.MONEY_TOLERANCE == old.MONEY_TOLERANCE == 1e-6


def test_v2_only_calls_original_pinned_market_reader(monkeypatch):
    inputs = read_json(study.base.checked_reference(read_json(rules.POLICY)["source_inputs"]))
    continuation = read_json(
        study.base.checked_reference(read_json(rules.POLICY)["reader_continuation_inputs"])
    )
    rf = pd.Series([0.01, 0.02], index=pd.date_range("2016-10-06", periods=2))
    calls = []

    def original_reader(definition, root):
        calls.append((definition, root))
        return SimpleNamespace(risk_free=rf)

    monkeypatch.setattr(study.base, "review_market", original_reader)
    assert study.canonical_rf(inputs, continuation) is rf
    assert calls == [({"market": continuation["original_frozen_market"]}, study.ROOT)]
    different = {**inputs, "risk_free": {**inputs["risk_free"], "sha256": "0" * 64}}
    with pytest.raises(QuantError):
        study.canonical_rf(different, continuation)


@pytest.mark.parametrize(
    "key,value",
    [
        ("new_RF_view_or_source_adapter", True),
        ("source_RF_reader", "raw_pandas_round_trip"),
        ("return_and_log_identity_tolerance", 1e-8),
        ("money_identity_tolerance_USD", 0.01),
        ("original_V1_reader_contract_remains_failed", False),
        ("original_V1_arithmetic_and_all_limits_reused_unchanged", False),
        ("source_CAGR_Sharpe_or_qualification_recomputed", True),
        ("new_strategy_targets_accounts_BT_or_financial_trials", 1),
    ],
)
def test_no_RF_view_tolerance_or_failed_V1_rescue(key, value):
    with pytest.raises(QuantError):
        rules.validate({**read_json(rules.POLICY), key: value})
