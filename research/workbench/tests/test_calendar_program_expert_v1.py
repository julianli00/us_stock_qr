from copy import deepcopy

import pandas as pd
import pytest

from us_quant import calendar_expert_inclusion_v1 as rules
from us_quant import calendar_program_expert_v1 as adapter
from us_quant.config import QuantError
from us_quant.research_program import review_market
from us_quant.storage import read_json


@pytest.fixture(scope="module")
def inputs():
    record = read_json(rules.POLICY)["calendar_source"]
    result = read_json(adapter.base.checked_reference(record["results"]))
    data = review_market({"market": result["frozen"]["market"]}, adapter.base.ROOT)
    return record, data.risk_free


def test_published_source_observes_existing_book_opinions_and_maturity_without_replay(inputs):
    record, risk = inputs
    expert, evidence = adapter.load_calendar_expert(record, adapter.law.UNIVERSE, risk)
    assert expert.identity.identifier == record["id"]
    assert expert.identity.introduced_at == record["introduced_at"]
    assert evidence["source_packet_schema"] == 3
    assert evidence["standalone_financial_verdict_preserved"] == "rejected"
    assert evidence["source_technical_and_funding_checks_passed"]
    assert evidence["original_source_targets_or_accounts_replayed"] is False
    first = expert.observe(pd.Timestamp("2016-10-05"), 126)
    assert first.target is not None and first.target_known_at == pd.Timestamp("2016-09-30")
    assert len(first.net_returns) == 0 and not first.score_mature
    cold = expert.observe(pd.Timestamp("2017-04-05"), 126)
    mature = expert.observe(pd.Timestamp("2017-04-06"), 126)
    assert len(cold.net_returns) == 125 and not cold.score_mature
    assert len(mature.net_returns) == 126 and mature.score_mature
    assert mature.risk_free.equals(risk.loc[mature.risk_free.index])
    summer = expert.observe(pd.Timestamp("2017-05-15"), 126)
    assert summer.target_known_at == pd.Timestamp("2017-04-28")
    assert summer.target["BIL"] == 0.98
    assert summer.target_known_at < summer.as_of


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", "calendar_variant"),
        ("mechanism_group", "independent_risk"),
        ("introduced_at", "2015-08-31T00:00:00Z"),
        ("schema_version", True),
    ],
)
def test_source_identity_or_historical_introduction_is_not_relabelled(inputs, field, value):
    record, risk = inputs
    bad = deepcopy(record)
    bad[field] = value
    with pytest.raises(QuantError):
        adapter.load_calendar_expert(bad, adapter.law.UNIVERSE, risk)


@pytest.mark.parametrize(
    "field",
    [
        "native_RF_exact_common",
        "funding_and_independent_BT_checked_all_four_paths",
        "technical_admission_is_final_qualification",
        "current_global_pool_or_roster_changed",
        "order_authority",
    ],
)
def test_false_source_checks_and_qualification_or_trading_laundering_stop(
    inputs, field, monkeypatch
):
    record, risk = inputs
    path = adapter.base.checked_reference(record["admission"])
    original = adapter.read_json
    bad = deepcopy(original(path))
    bad[field] = not bad[field]
    monkeypatch.setattr(adapter, "read_json", lambda p: bad if p == path else original(p))
    with pytest.raises(QuantError):
        adapter.load_calendar_expert(record, adapter.law.UNIVERSE, risk)


def test_no_new_rf_view_or_asset_crop_is_accepted(inputs):
    record, risk = inputs
    changed = risk.copy()
    changed.loc["2017-04-06"] += 1e-8
    with pytest.raises(QuantError, match="substitute RF"):
        adapter.load_calendar_expert(record, adapter.law.UNIVERSE, changed)
    with pytest.raises(QuantError):
        adapter.load_calendar_expert(record, tuple(reversed(adapter.law.UNIVERSE)), risk)


def test_actual_publication_and_source_start_chronology_cannot_be_backdated(inputs, monkeypatch):
    record, risk = inputs
    result = read_json(adapter.base.checked_reference(record["results"]))
    proof = read_json(adapter.base.checked_reference(result["native_source_proof"]["proof"]))
    path = adapter.base.checked_reference(proof["source_start"])
    original = adapter.read_json
    bad = deepcopy(original(path))
    bad["started_at"] = "2026-11-05T03:33:31.698701Z"
    monkeypatch.setattr(adapter, "read_json", lambda p: bad if p == path else original(p))
    with pytest.raises(QuantError, match="publication fence"):
        adapter.load_calendar_expert(record, adapter.law.UNIVERSE, risk)
