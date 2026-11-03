import copy

import pandas as pd
import pytest

from us_quant import expert_native_arrival_equal_v1 as rules
from us_quant.calendar import is_month_end, sessions
from us_quant.config import QuantError
from us_quant.storage import read_json
from us_quant.strategy_experts_v1 import ExpertIdentity, ExpertObservation

ASSETS = ("SPY", "BIL", "MTUM", "VLUE", "QUAL", "USMV", "IJR", "PKW")


class OpinionFixture:
    def __init__(self, number, calendar):
        self.number = number
        self.calendar = calendar
        self.identity = ExpertIdentity(
            rules.EXPERT_IDS[number], rules.MECHANISMS[number], "2026-11-03T00:00:00Z"
        )
        self.calls = []
        self.missing = set()
        self.bad_known_at = False
        self.return_level = 0.0001
        self.future_start = None

    def observe(self, as_of, lookback):
        assert lookback == 2
        self.calls.append(as_of)
        if self.number == 3:
            refresh = as_of
        else:
            refresh = max(day for day in self.calendar if day <= as_of and is_month_end(day))
        target = pd.Series(0.0, index=ASSETS)
        target.iloc[self.number] = 0.98
        if refresh.month == 11 or self.future_start is not None and refresh >= self.future_start:
            target[:] = 0
            target["BIL"] = 0.98
        if self.number == 3:
            target[:] = 0
            share = 0.35 + refresh.day / 100
            target["SPY"], target["BIL"] = share, 0.98 - share
        if as_of in self.missing:
            target = None
        known = as_of if self.number in (0, 1, 3) else refresh
        if self.bad_known_at:
            known += pd.Timedelta(days=1)
        history = self.calendar[self.calendar <= as_of][-2:]
        return ExpertObservation(
            self.identity,
            as_of,
            target,
            known,
            pd.Series(self.return_level, index=history),
            pd.Series(0.00001, index=history),
            len(history) == lookback,
        )


def examples(end="2017-01-05"):
    calendar = sessions("2016-09-01", end)
    return calendar, tuple(OpinionFixture(i, calendar) for i in range(8))


def test_dense_monthly_availability_does_not_create_new_policy_refresh():
    calendar, experts = examples()
    view = rules.NativeOpinionAdapter(experts[0], calendar).observe(pd.Timestamp("2016-10-12"))
    assert view.meta_as_of == pd.Timestamp("2016-10-12")
    assert view.policy_refresh_at == pd.Timestamp("2016-09-30")
    assert view.observation.as_of == view.observation.target_known_at == view.policy_refresh_at
    assert view.arrived_today is False
    assert experts[0].calls == [pd.Timestamp("2016-09-30")]


def test_daily_feedback_refresh_is_not_a_source_execution_claim():
    calendar, experts = examples()
    view = rules.NativeOpinionAdapter(experts[3], calendar).observe(pd.Timestamp("2016-10-12"))
    assert view.arrived_today is True
    assert view.native_clock == "daily_feedback"
    assert view.policy_refresh_at == view.observation.as_of == pd.Timestamp("2016-10-12")


def test_first_master_capital_and_monthly_source_refresh_have_different_roles():
    calendar, experts = examples()
    target, audits = rules.build_targets(experts, calendar, ASSETS, read_json(rules.POLICY))
    assert audits[0]["decision_session"] == "2016-10-05"
    assert audits[0]["decision_role"] == "initial_capital"
    assert audits[0]["fresh_source_opinion_ids"] == [rules.EXPERT_IDS[3]]
    assert [row["policy_refreshed_at"] for row in audits[0]["source_opinions"]] == [
        "2016-10-05" if i == 3 else "2016-09-30" for i in range(8)
    ]
    assert target.loc[:"2016-10-04"].isna().all().all()
    assert target.dropna().index.equals(calendar[calendar >= pd.Timestamp("2016-10-05")])
    assert target.dropna().sum(axis=1).sub(0.98).abs().max() < 1e-12


def test_monthly_sources_are_queried_only_on_native_month_ends():
    calendar, experts = examples()
    _, audits = rules.build_targets(experts, calendar, ASSETS, read_json(rules.POLICY))
    for i, expert in enumerate(experts):
        if i != 3:
            assert all(is_month_end(day) for day in expert.calls)
    month = next(row for row in audits if row["decision_session"] == "2016-10-31")
    assert month["fresh_source_opinion_ids"] == list(rules.EXPERT_IDS)
    assert all(
        row["source_native_order_or_execution_claimed"] is False for row in month["source_opinions"]
    )


def test_return_scores_do_not_affect_equal_opinion_allocation():
    calendar, experts = examples()
    before, audits = rules.build_targets(experts, calendar, ASSETS, read_json(rules.POLICY))
    for expert in experts:
        expert.return_level = 3.0
    after, later = rules.build_targets(experts, calendar, ASSETS, read_json(rules.POLICY))
    assert before.equals(after)
    assert audits == later
    assert all(row["source_scores_used"] is False for row in later)


def test_prefix_truncation_and_future_opinions_do_not_change_prior_targets():
    calendar, experts = examples()
    full, audits = rules.build_targets(experts, calendar, ASSETS, read_json(rules.POLICY))
    for stamp in ("2016-10-05", "2016-10-06", "2016-10-31", "2016-11-30", "2016-12-30"):
        day = pd.Timestamp(stamp)
        early, earlier = rules.build_targets(
            experts, calendar[calendar <= day], ASSETS, read_json(rules.POLICY)
        )
        assert early.equals(full.loc[:day])
        assert earlier == [row for row in audits if row["decision_session"] <= stamp]
    experts[0].future_start = pd.Timestamp("2016-12-30")
    changed, later = rules.build_targets(experts, calendar, ASSETS, read_json(rules.POLICY))
    assert changed.loc[:"2016-12-29"].equals(full.loc[:"2016-12-29"])
    assert later[:10] == audits[:10]


def test_actual_query_and_operands_are_retained_before_aggregate():
    calendar, experts = examples("2016-10-07")
    rows = []
    target, audits = rules.build_targets(
        experts, calendar, ASSETS, read_json(rules.POLICY), retain=rows.append
    )
    assert [row["record"] for row in rows] == [
        "opinion_query",
        "opinion_input",
        "opinion_resolved",
    ] * 3
    assert rows[0]["native_query_sessions"] == [
        "2016-10-05" if i == 3 else "2016-09-30" for i in range(8)
    ]
    assert rows[1]["source_opinions"][0]["target_binary64_hex"]
    assert rows[2]["audit"] == audits[0]
    assert rows[2]["target_binary64_hex"] == [float(x).hex() for x in target.loc["2016-10-05"]]


def test_missing_expected_native_update_preserves_query_and_never_carries_safe():
    calendar, experts = examples("2016-10-31")
    experts[0].missing.add(pd.Timestamp("2016-10-31"))
    rows = []
    with pytest.raises(QuantError, match="no carried safe fallback"):
        rules.build_targets(experts, calendar, ASSETS, read_json(rules.POLICY), retain=rows.append)
    assert rows[-1]["record"] == "opinion_query"
    assert rows[-1]["meta_as_of"] == "2016-10-31"


def test_future_source_known_at_and_preformal_scope_are_explicit_errors():
    calendar, experts = examples()
    experts[0].bad_known_at = True
    with pytest.raises(QuantError, match="no carried safe fallback"):
        rules.NativeOpinionAdapter(experts[0], calendar).observe(pd.Timestamp("2016-10-12"))
    with pytest.raises(QuantError, match="raw source prehistory is unchanged"):
        rules.NativeOpinionAdapter(experts[1], calendar).observe(pd.Timestamp("2016-09-07"))


def test_native_clock_parameter_or_wealth_guarantee_changes_are_rejected():
    policy = read_json(rules.POLICY)
    for key, value in (
        ("source_opinion_clocks", ["daily_feedback"] * 8),
        ("source_feedback_floor_or_shadow_wealth_guarantee_transferred", True),
    ):
        changed = copy.deepcopy(policy)
        changed[key] = value
        with pytest.raises(QuantError):
            rules.validate(changed)
