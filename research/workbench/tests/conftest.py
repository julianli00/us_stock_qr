from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from us_quant.calendar import sessions
from us_quant.config import load_config
from us_quant.data import MarketData
from us_quant.paper import PaperConfig
from us_quant.storage import read_json


@pytest.fixture
def config():
    return load_config(Path(__file__).parents[1] / "config/research.json")


@pytest.fixture
def small_config(config):
    return replace(config, momentum_lookbacks=(3, 5, 10), trend_lookback=6, volatility_lookback=5)


@pytest.fixture
def market_factory():
    def create(start="2019-12-02", end="2022-01-31", symbols=("SPY", "QQQ")):
        index = sessions(start, end)
        rng = np.random.default_rng(12345)
        returns = rng.normal(0.0004, 0.008, (len(index), len(symbols)))
        close = pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=index, columns=symbols)
        opening = close.shift(1).fillna(100) * (1 + returns * 0.2)
        return MarketData(
            open=opening,
            close=close,
            raw_close=close.copy(),
            volume=pd.DataFrame(1000000.0, index=index, columns=symbols),
            risk_free=pd.Series(0.00004, index=index, name="risk_free"),
        )

    return create


@pytest.fixture
def paper_config(tmp_path):
    raw = read_json(Path(__file__).parents[1] / "config/paper.example.json")
    raw.update(
        {
            "account": "DU1234567",
            "paper_acknowledged": True,
            "allowed_symbols": tuple(raw["allowed_symbols"]),
            "state_file": str(tmp_path / "paper.sqlite3"),
            "kill_switch_file": str(tmp_path / "STOP"),
        }
    )
    return PaperConfig(**raw)
