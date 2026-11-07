import builtins
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant import retained_main_risk_v1 as model
from us_quant import ssd_expert_adanormalhedge_v1 as old
from us_quant import ssd_retained_main_risk_v1 as driver
from us_quant.calendar import is_month_end, previous_session, sessions
from us_quant.config import QuantError


def points(calendar):
    origins = [model.ANCHOR, *[d for d in calendar[1:] if is_month_end(d)]]
    result = []
    for index, origin in enumerate(origins):
        p = np.roll(np.arange(1, 7, dtype=float) / 21, index)
        targets = np.array([np.roll(np.arange(1, 15), 2 * i) / 105 * 0.98 for i in range(6)])
        result.append(
            model.SourcePoint(
                origin,
                tuple(p),
                tuple(tuple(row) for row in targets),
                tuple(p @ targets),
                (origin,) * 6,
                max(0, index - 1),
            )
        )
    return tuple(result)


@pytest.fixture
def example():
    calendar = sessions("2016-10-05", "2016-11-30")
    price_returns = np.sin(np.arange(len(calendar))[:, None]) * np.arange(1, 15) * 0.0003
    close = pd.DataFrame(
        100 * np.cumprod(1 + price_returns, axis=0), index=calendar, columns=model.ASSETS
    )
    dates = calendar[1:]
    nav = np.linspace(10000, 7900, len(dates))
    actual = np.full((len(dates), len(model.ASSETS)), 0.49 / len(model.ASSETS))
    account = pd.DataFrame(
        {
            "equity": nav,
            "return": nav / np.r_[10000, nav[:-1]] - 1,
            "cash": nav * 0.51,
            "gross_exposure": 0.49,
            "turnover": 0.1,
            "cost": 1.5,
            "orders": 2,
            "risk_free": 0.0001,
        },
        index=dates,
    )
    source = model.MemoryFeed(points(calendar))
    desired = pd.DataFrame(np.nan, index=calendar, columns=model.ASSETS)
    for origin in source.origins:
        desired.loc[origin] = source.latest(origin).mixed_target
    return model.PathEvidence(
        "10y-base", account, pd.DataFrame(actual, index=dates, columns=model.ASSETS), desired, close
    ), source


def tape(path, feed):
    with path.open("x") as stream:
        for point in feed.points:
            day = str(point.origin.date())
            known = [str(d.date()) for d in point.known_at]
            stream.write(
                json.dumps(
                    {
                        "phase": "before_decision",
                        "decision_session": day,
                        "source_native_known_at": known,
                        "source_targets_hex": [
                            [float(v).hex() for v in row] for row in point.targets
                        ],
                    }
                )
                + "\n"
            )
            stream.write(
                json.dumps(
                    {
                        "phase": "after_decision",
                        "decision_session": day,
                        "audit": {
                            "decision_session": day,
                            "source_native_known_at": known,
                            "expert_weights": point.probabilities,
                            "complete_source_intervals": point.complete_intervals,
                        },
                        "unified_target_hex": [float(v).hex() for v in point.mixed_target],
                    }
                )
                + "\n"
            )


def test_policy_is_fixed_descriptive_not_a_new_cap():
    policy = driver.read_json(model.POLICY)
    model.validate(policy)
    with pytest.raises(QuantError):
        model.validate({**policy, "signed_hedging_contributions_clipped": True})


def test_archive_and_memory_cross_same_interface(example, tmp_path):
    evidence, feed = example
    path = tmp_path / "source.jsonl"
    tape(path, feed)
    assert model.audit_path(evidence, feed) == model.audit_path(evidence, driver.ArchiveFeed(path))


def test_future_does_not_change_then_known_prefix(example):
    evidence, feed = example
    cutoff = pd.Timestamp("2016-10-27")
    records = model.audit_path(evidence, feed)["daily_digests"]
    altered = evidence.close.copy()
    altered.loc[altered.index > cutoff] *= 1.2
    future = []
    for point in feed.points:
        if point.origin > cutoff:
            p = tuple(np.roll(point.probabilities, 2))
            point = replace(
                point, probabilities=p, mixed_target=tuple(np.array(p) @ np.asarray(point.targets))
            )
        future.append(point)
    short = replace(
        evidence,
        account=evidence.account.loc[:cutoff],
        held_weights=evidence.held_weights.loc[:cutoff],
        close=altered,
    )
    assert model.audit_path(short, model.MemoryFeed(tuple(future)))["daily_digests"] == {
        d: h for d, h in records.items() if d <= str(cutoff.date())
    }


def test_old_pipelines_never_called(example, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No old learner, account or verifier may execute.")

    for name in ("build_targets", "run", "verify_run"):
        monkeypatch.setattr(old.rules, name, forbidden)
    assert model.audit_path(*example)["summary"]["rows"] == len(example[0].account)


def test_zero_variance_is_undefined(example):
    evidence, feed = example
    rows = []
    result = model.audit_path(
        replace(evidence, close=evidence.close * 0 + 100), feed, retain=rows.append
    )
    assert result["summary"]["defined_source_variance_share_days"] == 0
    later = [r["result"] for r in rows if r["phase"] == "after_risk_validation"][-1]
    assert later["risk"]["source_contribution_shares"] is None
    assert later["risk"]["share_status"] == "undefined_zero_variance"


def test_signed_hedging_share_not_clipped(example):
    evidence, feed = example
    covariance = np.zeros((14, 14))
    covariance[:2, :2] = [[0.01, -0.01], [-0.01, 0.01]]
    targets = np.zeros((6, 14))
    targets[:, 0] = 0.98
    targets[2] = 0
    targets[2, 1] = 0.98
    p = np.array([0.18, 0.18, 0.1, 0.18, 0.18, 0.18])
    mixture = p @ targets
    result = model._risks(p, targets, mixture, mixture, covariance, covariance.copy())
    assert result["source_contribution_shares"][2] < 0
    assert sum(result["source_contribution_shares"]) == pytest.approx(1, abs=1e-12)


def test_initial_peak_and_original_costs_are_preserved(example):
    result = model.audit_path(*example)
    assert result["summary"]["maximum_drawdown_from_saved_NAV"] == pytest.approx(0.21)
    assert result["contexts"]["worst_drawdown"]["peak"]["initial_capital_peak"]
    assert result["summary"]["actual_model_cost_sum"] == 1.5 * len(example[0].account)
    assert result["contexts"]["first_strict_15pct_breach"]["drawdown"] > 0.15
    assert result["summary"]["undefined_covariance_days"] == 1


@pytest.mark.parametrize("field", ["cash", "gross_exposure", "orders", "return"])
def test_invalid_main_primitives_fail(example, field):
    evidence, feed = example
    bad = evidence.account.copy()
    bad.iloc[0, bad.columns.get_loc(field)] = -1
    with pytest.raises(QuantError):
        model.audit_path(replace(evidence, account=bad), feed)


def test_retains_reference_before_failure(example):
    evidence, feed = example
    bad = evidence.account.copy()
    bad.iloc[0, 0] = np.nan
    rows = []
    with pytest.raises(QuantError):
        model.audit_path(replace(evidence, account=bad), feed, retain=rows.append)
    assert rows[0]["phase"] == "before_input_validation"


def test_no_missing_price_or_unmatured_source_becomes_neutral(example):
    evidence, feed = example
    with pytest.raises(QuantError, match="price history"):
        model.audit_path(replace(evidence, close=evidence.close.iloc[1:]), feed)
    bad = replace(feed.points[0], known_at=(pd.Timestamp("2016-12-01"),) * 6)
    with pytest.raises(QuantError, match="then-known"):
        model.audit_path(evidence, model.MemoryFeed((bad, *feed.points[1:])))


def test_future_adapter_cannot_launder_equal_target(example):
    evidence, feed = example
    future = replace(
        feed.points[0],
        origin=pd.Timestamp("2016-11-30"),
        known_at=(pd.Timestamp("2016-11-30"),) * 6,
    )

    class FutureFeed:
        origins = feed.origins

        def latest(self, day):
            return future

    with pytest.raises(QuantError, match="then-known"):
        model.audit_path(evidence, FutureFeed())


def test_boolean_probability_and_account_columns_fail(example):
    evidence, feed = example
    bad = replace(feed.points[0], probabilities=(True, 0, 0, 0, 0, 0))
    with pytest.raises(QuantError, match="boolean"):
        model.audit_path(evidence, model.MemoryFeed((bad, *feed.points[1:])))
    account = evidence.account.copy()
    account["orders"] = True
    with pytest.raises(QuantError, match="Boolean"):
        model.audit_path(replace(evidence, account=account), feed)


def test_fresh_main_anchor_is_not_prior_market_wealth(example):
    evidence, feed = example
    dates = evidence.account.index[evidence.account.index >= pd.Timestamp("2016-11-01")]
    account = evidence.account.loc[dates].copy()
    account["equity"] = 10000
    account["cash"] = 10000
    account["gross_exposure"] = 0
    account["return"] = 0
    account["cost"] = 0
    account["turnover"] = 0
    account["orders"] = 0
    held = evidence.held_weights.loc[dates] * 0
    desired = evidence.desired.copy()
    anchor = previous_session(dates[0])
    desired.loc[anchor] = feed.latest(anchor).mixed_target
    result = model.audit_path(
        replace(evidence, account=account, held_weights=held, desired=desired), feed
    )
    assert result["summary"]["maximum_drawdown_from_saved_NAV"] == 0
    assert result["summary"]["undefined_covariance_days"] == 0
    assert result["summary"]["mean_actual_invested_asset_hhi"] is None


def test_archive_guard_rejects_before_open(tmp_path, monkeypatch):
    fake = tmp_path / "synthetic-prohibited.json"
    monkeypatch.setattr(old, "PROTECTED", fake)
    opened = []

    def forbidden(*args, **kwargs):
        opened.append(True)
        raise AssertionError("Reject the synthetic forbidden path before open.")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(builtins, "open", forbidden)
    with pytest.raises(PermissionError, match="before open"):
        driver.ArchiveFeed(fake)
    assert not opened
