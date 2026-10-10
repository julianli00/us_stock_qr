from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import us_quant.research as research
from us_quant.cli import main
from us_quant.config import QuantError
from us_quant.research import run_development, run_holdout, verify_freeze, verify_qualification
from us_quant.storage import (
    digest_json,
    file_digest,
    read_json,
    utc_now,
    write_json,
    write_text_atomic,
)


def write_fixture_dataset(config, data, output: Path, phase: str):
    start = config.data_start if phase == "development" else config.development_end
    end = config.development_end if phase == "development" else config.as_of
    files = {}
    for symbol in config.symbols:
        frame = pd.DataFrame(
            {
                "open": data.open[symbol],
                "close": data.raw_close[symbol],
                "adj_open": data.open[symbol],
                "adj_close": data.close[symbol],
                "volume": data.volume[symbol],
            }
        ).loc[start:end]
        frame.index.name = "date"
        target = output / f"{symbol}.csv"
        write_text_atomic(target, frame.to_csv(float_format="%.12g"))
        files[target.name] = file_digest(target)
    rate_dates = pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=14), end, freq="B")
    rates = pd.DataFrame({"close": 1.3}, index=rate_dates)
    rates.index.name = "date"
    target = output / "IRX.csv"
    write_text_atomic(target, rates.to_csv())
    files[target.name] = file_digest(target)
    write_json(
        output / "manifest.json",
        {
            "phase": phase,
            "protocol_sha256": digest_json(config.to_dict()),
            "files": files,
            "retrieved_at": utc_now(),
            "start": start,
            "end": end,
            "provider": "synthetic test fixture",
        },
    )


@pytest.fixture
def protocol(small_config):
    return replace(
        small_config,
        data_start="2019-12-02",
        simulation_start="2020-01-02",
        development_end="2021-12-31",
        holdout_start="2022-01-03",
        as_of="2022-01-31",
        selection=replace(
            small_config.selection,
            training_years=1,
            test_years=1,
            first_test_year=2021,
            final_training_start="2021-01-01",
        ),
        targets=replace(small_config.targets, minimum_holdout_sessions=2),
        stress=replace(small_config.stress, bootstrap_samples=100, bootstrap_block_sessions=5),
    )


def test_complete_frozen_workflow_is_reproducible_and_honest(tmp_path, protocol, market_factory):
    market = market_factory(symbols=protocol.symbols)
    development = tmp_path / "development"
    holdout = tmp_path / "holdout"
    development_output = tmp_path / "dev_report"
    write_fixture_dataset(protocol, market, development, "development")
    dev = run_development(protocol, development, development_output)
    assert dev["walk_forward"]["start"] == "2021-01-04"
    assert len(dev["folds"]) == 1
    freeze = development_output / "frozen.json"
    original_hash = file_digest(freeze)
    write_fixture_dataset(protocol, market, holdout, "holdout")
    first = run_holdout(protocol, development, holdout, freeze, tmp_path / "evaluation")
    second = run_holdout(protocol, development, holdout, freeze, tmp_path / "reproduction")
    assert first["holdout"] == second["holdout"]
    assert first["bootstrap"] == second["bootstrap"]
    assert first["selected_candidate_id"] == dev["final_selection"]["selected_candidate_id"]
    assert first["holdout"]["start"] == "2022-01-03"
    assert first["holdout"]["end"] == protocol.as_of
    assert first["objective_verified"] is False and first["forward_paper_sessions"] == 0
    assert file_digest(freeze) == original_hash
    verify_qualification(protocol, freeze, tmp_path / "evaluation/results.json")
    with pytest.raises(QuantError, match="overwrite"):
        run_holdout(protocol, development, holdout, freeze, tmp_path / "evaluation")
    signal = read_json(tmp_path / "evaluation/latest_signal.json")
    assert signal["executable"] is False
    assert "stale_completed_session" in signal["block_reasons"]


def test_code_or_protocol_changes_invalidate_freeze(
    tmp_path, protocol, market_factory, monkeypatch
):
    market = market_factory(symbols=protocol.symbols)
    development = tmp_path / "development"
    output = tmp_path / "results"
    write_fixture_dataset(protocol, market, development, "development")
    run_development(protocol, development, output)
    freeze = output / "frozen.json"
    verify_freeze(protocol, development, freeze)
    monkeypatch.setattr(research, "implementation_fingerprint", lambda: "changed-code")
    with pytest.raises(QuantError, match="implementation changed"):
        verify_freeze(protocol, development, freeze)


def test_peeked_holdout_is_not_accepted_as_a_fresh_holdback(tmp_path, protocol, market_factory):
    market = market_factory(symbols=protocol.symbols)
    development, holdout, output = (
        tmp_path / name for name in ("development", "holdout", "output")
    )
    write_fixture_dataset(protocol, market, development, "development")
    write_fixture_dataset(protocol, market, holdout, "holdout")
    run_development(protocol, development, output)
    with pytest.raises(QuantError, match="predates the freeze"):
        run_holdout(protocol, development, holdout, output / "frozen.json", tmp_path / "evaluation")


def test_cli_probe_reports_blocker_without_claiming_an_account(monkeypatch, capsys):
    import us_quant.cli as cli

    monkeypatch.setattr(
        cli,
        "probe_paper_ports",
        lambda: {
            "paper_ports_listening": {"7497": False, "4002": False},
            "account_verified": False,
            "orders_sent": 0,
        },
    )
    assert main(["doctor"]) == 0
    assert "blocked_no_local_paper_service" in capsys.readouterr().out


def test_cli_discovery_is_metadata_only(monkeypatch, capsys):
    import us_quant.cli as cli

    monkeypatch.setattr(
        cli,
        "probe_paper_ports",
        lambda: {
            "paper_ports_listening": {"7497": False, "4002": True},
            "account_verified": False,
            "orders_sent": 0,
        },
    )
    monkeypatch.setattr(
        cli,
        "discover_paper_identity",
        lambda port: {
            "port": port,
            "account": "DU***4567",
            "identity_verified": True,
            "account_data_requested": False,
            "market_data_requested": False,
            "orders_sent": 0,
        },
    )
    assert main(["doctor", "--discover-paper"]) == 0
    output = capsys.readouterr().out
    assert "paper_identity_verified_account_not_synced" in output
    assert '"account_verified": false' in output


def test_cli_binding_requires_an_unambiguous_paper_endpoint(monkeypatch, capsys, tmp_path):
    import us_quant.cli as cli

    monkeypatch.setattr(
        cli,
        "probe_paper_ports",
        lambda: {
            "paper_ports_listening": {"7497": True, "4002": True},
            "account_verified": False,
            "orders_sent": 0,
        },
    )
    target = tmp_path / "paper.json"
    assert main(["doctor", "--save-paper-config", str(target)]) == 2
    assert "unambiguous" in capsys.readouterr().err
    assert not target.exists()


def test_cli_handles_asyncio_timeouts_explicitly(monkeypatch, capsys):
    import asyncio

    import us_quant.cli as cli

    def timed_out(args):
        raise asyncio.TimeoutError()

    monkeypatch.setattr(cli, "dispatch", timed_out)
    assert main(["doctor"]) == 2
    assert "TimeoutError" in capsys.readouterr().err


def test_cli_redacts_broker_account_ids_from_warnings_and_errors(monkeypatch, capsys):
    import logging

    import us_quant.cli as cli

    message = "Trade(account='DUQ1234567') compared with U7654321"
    masked = cli.redact_account_ids(message)
    assert "DUQ1234567" not in masked and "U7654321" not in masked
    assert "DUQ***4567" in masked and "U***4321" in masked
    record = logging.LogRecord("broker", logging.WARNING, "", 0, "%s", (message,), None)
    assert cli.AccountLogRedactor().filter(record)
    assert record.getMessage() == masked

    def rejected(args):
        raise QuantError(message)

    monkeypatch.setattr(cli, "dispatch", rejected)
    assert main(["doctor"]) == 2
    assert "DUQ1234567" not in capsys.readouterr().err
    monkeypatch.setattr(cli, "dispatch", lambda args: {"warning": message})
    assert main(["doctor"]) == 0
    assert "DUQ1234567" not in capsys.readouterr().out


def test_example_paper_config_never_sends_an_order(capsys, monkeypatch):
    import us_quant.cli as cli

    monkeypatch.setattr(
        cli,
        "probe_paper_ports",
        lambda: {
            "paper_ports_listening": {"7497": False, "4002": False},
            "account_verified": False,
            "orders_sent": 0,
        },
    )
    assert main(["doctor", "--paper-config", "config/paper.example.json"]) == 2
    assert "DU paper account" in capsys.readouterr().err


def test_changed_signal_weights_are_recomputed_before_connection(
    tmp_path, config, monkeypatch, capsys
):
    import us_quant.cli as cli

    freeze_path = tmp_path / "frozen.json"
    data_path = tmp_path / "data"
    signal_path = tmp_path / "signal.json"
    write_json(freeze_path, {"test_fixture": True})
    write_json(data_path / "manifest.json", {"test_fixture": True})
    qualification = {"research_gates_passed": True}
    frozen = {"selected_candidate_id": "cross_asset_top3_12"}
    signal = {
        "freeze_sha256": file_digest(freeze_path),
        "qualification_sha256": digest_json(qualification),
        "protocol_sha256": digest_json(config.to_dict()),
        "market_data_manifest_sha256": file_digest(data_path / "manifest.json"),
        "strategy_id": frozen["selected_candidate_id"],
        "research_qualified": True,
        "executable": True,
        "mode": "paper_only",
        "block_reasons": [],
        "signal_date": "2026-09-30",
        "execution_session": "2026-10-01",
        "weights": {"SPY": 0.3},
        "cash_weight": 0.7,
    }
    write_json(signal_path, signal)
    monkeypatch.setattr(cli, "verify_freeze", lambda *args: frozen)
    monkeypatch.setattr(cli, "verify_qualification", lambda *args: qualification)
    monkeypatch.setattr(cli, "load_market", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        cli, "make_signal", lambda *args: {**signal, "weights": {"SPY": 0.2}, "cash_weight": 0.8}
    )

    def forbid_broker(*args, **kwargs):
        pytest.fail("A mutated signal must never initiate a broker connection.")

    monkeypatch.setattr(cli, "IBPaperBroker", forbid_broker)
    result = main(
        [
            "paper",
            "--freeze",
            str(freeze_path),
            "--signal",
            str(signal_path),
            "--market-data",
            str(data_path),
        ]
    )
    assert result == 2
    assert "weights differs from recomputed" in capsys.readouterr().err


def test_observation_remains_read_only(tmp_path, monkeypatch, paper_config, capsys):
    from dataclasses import asdict

    import us_quant.cli as cli
    from us_quant.paper import Snapshot

    path = tmp_path / "paper.json"
    write_json(path, asdict(paper_config))

    class ReadOnlyBroker:
        def __init__(self, config, *, readonly):
            assert readonly is True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def snapshot(self):
            return Snapshot(paper_config.account, "USD", 10000.0, 10000.0, {})

    monkeypatch.setattr(cli, "IBPaperBroker", ReadOnlyBroker)
    assert main(["observe", "--paper-config", str(path)]) == 0
    output = capsys.readouterr().out
    assert '"orders_sent": 0' in output
    assert paper_config.account not in output


def test_readonly_doctor_surfaces_unknown_cash_as_an_execution_blocker(
    tmp_path, monkeypatch, paper_config, capsys
):
    from dataclasses import asdict

    import us_quant.cli as cli
    from us_quant.paper import Snapshot

    path = tmp_path / "paper.json"
    write_json(path, asdict(paper_config))

    class ReadOnlyBroker:
        def __init__(self, config, *, readonly):
            assert readonly

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def snapshot(self):
            return Snapshot(paper_config.account, "USD", 10000.0, None, {})

    monkeypatch.setattr(cli, "IBPaperBroker", ReadOnlyBroker)
    monkeypatch.setattr(
        cli,
        "probe_paper_ports",
        lambda: {
            "paper_ports_listening": {"7497": True, "4002": False},
            "account_verified": False,
            "orders_sent": 0,
        },
    )
    assert main(["doctor", "--paper-config", str(path)]) == 0
    output = capsys.readouterr().out
    assert '"account_verified": true' in output
    assert '"settled_cash": null' in output
    assert "settled_usd_cash_not_reported" in output
    assert '"orders_sent": 0' in output
