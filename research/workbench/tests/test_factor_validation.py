from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_quant.calendar import sessions
from us_quant.config import QuantError
from us_quant.factor_validation import check_metrics, independent_metrics
from us_quant.metrics import performance


@pytest.fixture
def ledgers():
    index = sessions("2020-01-02", "2020-02-28")
    returns = pd.Series(np.sin(np.arange(len(index))) * 0.005 + 0.001, index=index)
    risk_free = pd.Series(0.0001, index=index)
    frame = pd.DataFrame(
        {
            "equity": 10000 * (1 + returns).cumprod(),
            "return": returns,
            "risk_free": risk_free,
        }
    )
    return frame, frame[["equity", "return"]].copy(), risk_free


def test_independent_stats_agree_without_using_primary_metrics(ledgers):
    frame, independent, risk_free = ledgers
    actual, difference = independent_metrics(frame, independent, risk_free, 10000)
    expected = performance(frame["return"], risk_free)
    check_metrics(actual, expected)
    assert difference == 0


@pytest.mark.parametrize("field", ["cagr", "sharpe", "max_drawdown"])
def test_incorrect_reported_metric_cannot_pass_audit(ledgers, field):
    frame, independent, risk_free = ledgers
    actual, _ = independent_metrics(frame, independent, risk_free, 10000)
    wrong = dict(actual)
    wrong[field] += 0.05
    with pytest.raises(QuantError, match="disagrees"):
        check_metrics(actual, wrong)


def test_independent_equity_mismatch_is_not_ignored(ledgers):
    frame, independent, risk_free = ledgers
    independent.iloc[-1, independent.columns.get_loc("equity")] += 10
    with pytest.raises(QuantError, match="equity"):
        independent_metrics(frame, independent, risk_free, 10000)


def test_different_risk_free_or_missing_session_is_rejected(ledgers):
    frame, independent, risk_free = ledgers
    with pytest.raises(QuantError, match="inconsistent"):
        independent_metrics(frame, independent, risk_free * 2, 10000)
    with pytest.raises(QuantError, match="inconsistent"):
        independent_metrics(frame.drop(frame.index[5]), independent, risk_free, 10000)
