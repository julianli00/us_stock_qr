from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError, URLError


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.us_quant.research_delivery import (  # noqa: E402
    Receipt, configured_webhook, deliver_report, outbox_path, post_to_slack, read_outbox,
)
from src.us_quant.research_reporting import (  # noqa: E402
    CALENDAR_HASH, WATCH_ID, ExchangeCalendar, ReportError, ReportInputs, build_report,
    canonical_hash, load_inputs, read_json, report_from_root,
)


NOW = datetime(2026, 10, 6, 22, 15, tzinfo=timezone.utc)
PUBLIC_INPUTS = (
    "config/nyse_calendar.json",
    "config/research_strategy_registry.json",
    "docs/research_progress.json",
    "research/workbench/evidence/research_status.json",
)


def json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def concurrent_delivery(root: str, queue: object) -> None:
    def transport(text: str) -> Receipt:
        with (Path(root) / "transport_calls.txt").open("a") as stream:
            stream.write("one-post\n")
        time.sleep(0.1)
        return Receipt("SENT", "synthetic_receipt", 200)

    result = deliver_report(Path(root), now=NOW, dry_run=False, transport=transport)
    queue.put(result["reason"])


class ResearchReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="research_report_guard_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in PUBLIC_INPUTS:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        self.calendar = ExchangeCalendar(read_json(self.root / PUBLIC_INPUTS[0]))

    def snapshot(self, count: int = 1) -> dict:
        session = self.calendar.latest_completed(NOW)
        names = [f"AA{chr(ord('A') + index)}" for index in range(count)]
        decision = session.closed + timedelta(minutes=45)
        inputs = {
            "prices": {"rows": [
                {
                    "ticker": ticker, "session": session.day, "close": 100.0,
                    "known_at": (session.closed + timedelta(minutes=1)).isoformat(),
                    "basis": "adjusted_close",
                }
                for ticker in [*names, "SPY", "QQQ", "^VIX"]
            ]},
            "events": {"rows": [
                {
                    "event_id": f"event-{ticker}", "ticker": ticker,
                    "published_at": (session.closed - timedelta(hours=2)).isoformat(),
                    "known_at": (session.closed - timedelta(hours=1)).isoformat(),
                    "url": f"https://example.com/research/{ticker}",
                }
                for ticker in names
            ]},
            "universe": {
                "session": session.day, "point_in_time": True, "members": names,
                "known_at": (session.closed - timedelta(hours=3)).isoformat(),
            },
        }
        return {
            "schema_version": 1, "origin": "us_stock_qr", "strategy_id": WATCH_ID,
            "order_authority": False, "session": session.day, "decision_at": decision.isoformat(),
            "input_sha256": {name: canonical_hash(value) for name, value in inputs.items()},
            "inputs": inputs,
            "ideas": [
                {
                    "signal_id": f"{ticker}-{session.day}", "ticker": ticker,
                    "signal_session": session.day, "first_observed_session": session.day,
                    "event_ids": [f"event-{ticker}"], "rationale": "Synthetic timestamped research observation.",
                    "levels": {"entry_low": 99.0, "entry_high": 101.0, "stop": 95.0, "take_profit_1": 105.0, "take_profit_2": 110.0},
                }
                for ticker in names
            ],
        }

    def write_snapshot(self, snapshot: dict) -> None:
        directory = self.root / "artifacts/private/research_reporting/watch"
        directory.mkdir(parents=True, exist_ok=True)
        envelope = copy.deepcopy(snapshot)
        inputs = envelope.pop("inputs")
        envelope["input_sha256"] = {name: canonical_hash(value) for name, value in inputs.items()}
        for name, value in inputs.items():
            (directory / f"{name}.json").write_bytes(json_bytes(value))
        (directory / "snapshot.json").write_bytes(json_bytes(envelope))

    def assert_blocked(self, snapshot: dict, code: str) -> None:
        self.write_snapshot(snapshot)
        report = report_from_root(self.root, NOW)
        self.assertEqual(report.idea_keys, ())
        self.assertIn(code, report.blockers)
        self.assertNotIn("Research levels:", report.text)

    def test_stale_baseline_is_useful_status_only(self) -> None:
        report = report_from_root(self.root, NOW)
        self.assertEqual(report.idea_keys, ())
        self.assertIn("29.54%", report.text)
        self.assertIn("1.1867", report.text)
        self.assertIn("28.31%", report.text)
        self.assertIn("52", report.text)
        self.assertIn("2026-05-08", report.text)
        self.assertIn("early_watch_future_news_vs_signal", report.blockers)
        self.assertNotIn("Research levels:", report.text)
        english = report_from_root(self.root, NOW, language="en")
        self.assertIn("*Independent early watch*", english.text)
        self.assertIn("*Independent imported research*", english.text)

    def test_preview_has_no_state_or_input_mutation(self) -> None:
        before = {name: (self.root / name).read_bytes() for name in PUBLIC_INPUTS}
        with patch("src.us_quant.research_delivery.urlopen", side_effect=AssertionError("network")):
            first = deliver_report(self.root, now=NOW, dry_run=True)
            second = deliver_report(self.root, now=NOW + timedelta(seconds=2), dry_run=True)
        self.assertEqual(first["digest"], second["digest"])
        self.assertFalse(outbox_path(self.root).parent.exists())
        self.assertEqual(before, {name: (self.root / name).read_bytes() for name in PUBLIC_INPUTS})

    def test_calendar_before_close_after_close_and_six_am(self) -> None:
        before = self.calendar.latest_completed(datetime(2026, 10, 6, 19, 59, tzinfo=timezone.utc))
        after = self.calendar.latest_completed(datetime(2026, 10, 6, 20, 1, tzinfo=timezone.utc))
        self.assertEqual(before.day, "2026-10-05")
        self.assertEqual(after.day, "2026-10-06")
        self.assertEqual(after.report_after.astimezone(timezone.utc).hour, 22)
        result = deliver_report(
            self.root, now=datetime(2026, 10, 6, 21, 59, tzinfo=timezone.utc),
            dry_run=False, transport=lambda _: self.fail("before six"),
        )
        self.assertEqual(result["reason"], "before_0600_shanghai")
        self.assertFalse(outbox_path(self.root).parent.exists())

    def test_holiday_early_close_dst_and_ad_hoc_closure(self) -> None:
        self.assertNotIn("2026-11-26", self.calendar.sessions)
        self.assertNotIn("2025-01-09", self.calendar.sessions)
        self.assertEqual(self.calendar.sessions["2026-11-27"].closed.hour, 18)
        self.assertEqual(self.calendar.sessions["2026-10-30"].closed.hour, 20)
        self.assertEqual(self.calendar.sessions["2026-11-02"].closed.hour, 21)
        self.assertEqual(
            self.calendar.latest_completed(datetime(2025, 1, 9, 23, tzinfo=timezone.utc)).day,
            "2025-01-08",
        )

    def test_calendar_missing_corrupt_out_of_range_and_naive_clock(self) -> None:
        with self.assertRaisesRegex(ReportError, "out_of_range"):
            self.calendar.latest_completed(datetime(2029, 1, 2, tzinfo=timezone.utc))
        with self.assertRaisesRegex(ReportError, "timezone_missing"):
            self.calendar.latest_completed(datetime(2026, 10, 6))
        payload = read_json(self.root / PUBLIC_INPUTS[0])
        payload["sessions"][0]["close_utc"] = "2024-01-02T22:00:00+00:00"
        with self.assertRaisesRegex(ReportError, "hash_mismatch"):
            ExchangeCalendar(payload)
        (self.root / PUBLIC_INPUTS[0]).unlink()
        with self.assertRaisesRegex(ReportError, "input_missing"):
            deliver_report(self.root, now=NOW)

    def test_new_ideas_zero_one_and_six_without_quota_filling(self) -> None:
        for count, expected in ((0, 0), (1, 1), (6, 5)):
            with self.subTest(count=count):
                self.write_snapshot(self.snapshot(count))
                report = report_from_root(self.root, NOW)
                self.assertEqual(len(report.idea_keys), expected)
                self.assertEqual(report.text.count("Research levels:"), expected)

    def test_postclose_1730_decision_is_valid(self) -> None:
        snapshot = self.snapshot()
        snapshot["decision_at"] = "2026-10-06T17:30:00-04:00"
        self.write_snapshot(snapshot)
        self.assertEqual(len(report_from_root(self.root, NOW).idea_keys), 1)

    def test_decision_before_close_buffer_is_incomplete(self) -> None:
        snapshot = self.snapshot()
        snapshot["decision_at"] = "2026-10-06T16:29:59-04:00"
        self.assert_blocked(snapshot, "watch_decision_outside_cutoff")

    def test_decision_at_next_open_is_too_late(self) -> None:
        snapshot = self.snapshot()
        snapshot["decision_at"] = "2026-10-07T09:30:00-04:00"
        self.write_snapshot(snapshot)
        report = report_from_root(self.root, datetime(2026, 10, 7, 14, tzinfo=timezone.utc))
        self.assertEqual(report.idea_keys, ())
        self.assertIn("watch_decision_outside_cutoff", report.blockers)

    def test_decision_cannot_follow_report_generation(self) -> None:
        snapshot = self.snapshot()
        snapshot["decision_at"] = (NOW + timedelta(minutes=1)).isoformat()
        self.assert_blocked(snapshot, "watch_decision_outside_cutoff")

    def test_zh_blockers_and_progress_are_readable_with_real_canonical_link(self) -> None:
        report = report_from_root(self.root, NOW)
        self.assertNotIn("top30_price_session_mismatch", report.text)
        self.assertIn("Top30行情仍停在历史截止日", report.text)
        self.assertIn("同一截止日", report.text)
        self.assertIn("https://github.com/julianli00/us_stock_qr/pull/1", report.text)
        self.assertIsNone(report.metadata()["signal_date"])
        self.assertEqual(report.metadata()["session"], "2026-10-06")

    def test_future_or_old_price_any_required_symbol_blocks_lane(self) -> None:
        for symbol, bad_day in (("AAA", "2026-10-05"), ("SPY", "2026-10-07")):
            with self.subTest(symbol=symbol):
                snapshot = self.snapshot()
                for row in snapshot["inputs"]["prices"]["rows"]:
                    if row["ticker"] == symbol:
                        row["session"] = bad_day
                self.assert_blocked(snapshot, "watch_price_session_mismatch")

    def test_missing_benchmark_price_is_not_global_max_coverage(self) -> None:
        snapshot = self.snapshot()
        snapshot["inputs"]["prices"]["rows"].pop()
        self.assert_blocked(snapshot, "watch_required_price_missing")

    def test_future_event_and_missing_first_known_at_are_rejected(self) -> None:
        snapshot = self.snapshot()
        event = snapshot["inputs"]["events"]["rows"][0]
        event["published_at"] = "2026-10-07T12:00:00+00:00"
        event["known_at"] = "2026-10-07T12:01:00+00:00"
        self.assert_blocked(snapshot, "watch_event_known_at_invalid")
        snapshot = self.snapshot()
        del snapshot["inputs"]["events"]["rows"][0]["known_at"]
        self.assert_blocked(snapshot, "timestamp_missing")

    def test_event_availability_not_published_date_alone(self) -> None:
        snapshot = self.snapshot()
        snapshot["inputs"]["events"]["rows"][0]["known_at"] = "2026-10-07T00:00:00+00:00"
        self.assert_blocked(snapshot, "watch_event_known_at_invalid")

    def test_universe_and_price_known_at_must_precede_decision(self) -> None:
        snapshot = self.snapshot()
        snapshot["inputs"]["universe"]["point_in_time"] = False
        self.assert_blocked(snapshot, "watch_universe_not_point_in_time")
        snapshot = self.snapshot()
        snapshot["inputs"]["prices"]["rows"][0]["known_at"] = "2026-10-07T00:00:00+00:00"
        self.assert_blocked(snapshot, "watch_price_known_at_invalid")

    def test_hash_mismatch_and_missing_input_block_without_raw_errors(self) -> None:
        self.write_snapshot(self.snapshot())
        path = self.root / "artifacts/private/research_reporting/watch/prices.json"
        path.write_text('{"changed": true}')
        self.assertIn("watch_input_hash_mismatch", report_from_root(self.root, NOW).blockers)
        path.unlink()
        self.assertIn("watch_input_missing", report_from_root(self.root, NOW).blockers)

    def test_malformed_rows_are_explicitly_blocked(self) -> None:
        snapshot = self.snapshot()
        snapshot["inputs"]["prices"]["rows"][0] = 42
        self.assert_blocked(snapshot, "watch_price_row_invalid")
        snapshot = self.snapshot()
        snapshot["ideas"][0]["event_ids"] = [{}]
        self.assert_blocked(snapshot, "watch_evidence_missing")

    def test_nonfinite_provenance_is_not_accepted(self) -> None:
        with self.assertRaisesRegex(ReportError, "provenance_serialization_invalid"):
            canonical_hash({"value": float("nan")})
        with self.assertRaisesRegex(ReportError, "provenance_serialization_invalid"):
            canonical_hash({"value": float("inf")})

    def test_old_idea_cannot_be_relabelled_new(self) -> None:
        snapshot = self.snapshot()
        snapshot["ideas"][0]["first_observed_session"] = "2026-10-05"
        self.assert_blocked(snapshot, "watch_not_new_session")

    def test_sent_ideas_are_not_new_on_preview(self) -> None:
        self.write_snapshot(self.snapshot())
        sent = deliver_report(self.root, now=NOW, dry_run=False, transport=lambda _: Receipt("SENT", "ok", 200))
        preview = deliver_report(self.root, now=NOW, dry_run=True)
        self.assertTrue(sent["sent"])
        self.assertEqual(preview["new_watch_ideas"], 0)

    def test_invalid_price_levels_and_nonfinite_numbers_block(self) -> None:
        snapshot = self.snapshot()
        snapshot["ideas"][0]["levels"]["stop"] = 200
        self.assert_blocked(snapshot, "watch_levels_invalid")
        snapshot = self.snapshot()
        snapshot["inputs"]["prices"]["rows"][0]["close"] = 0
        self.assert_blocked(snapshot, "number_invalid")

    def test_no_private_fields_or_credentials_are_serialized(self) -> None:
        snapshot = self.snapshot()
        snapshot["ideas"][0]["positions"] = [{"ticker": "PRIVATE_CANARY"}]
        self.write_snapshot(snapshot)
        report = report_from_root(self.root, NOW)
        self.assertIn("private_fields_rejected", report.blockers)
        self.assertNotIn("PRIVATE_CANARY", report.text)
        snapshot = self.snapshot()
        snapshot["ideas"][0]["rationale"] = "authorization=PRIVATE_CANARY"
        self.write_snapshot(snapshot)
        report = report_from_root(self.root, NOW)
        self.assertIn("credential_text_rejected", report.blockers)
        self.assertNotIn("PRIVATE_CANARY", report.text)

    def test_holdings_and_legacy_delivery_ledgers_are_never_opened(self) -> None:
        sentinels = [
            "artifacts/current/slack_signal_state.json",
            "artifacts/current/personal_signal_manual_position_ledger_v1.csv",
            "artifacts/current/ibkr_paper_positions.csv",
        ]
        for name in sentinels:
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("PRIVATE_SENTINEL_DO_NOT_READ")
        original = Path.open

        def guarded(path: Path, *args: object, **kwargs: object):
            if str(path.relative_to(self.root)) in sentinels:
                raise AssertionError("Private ledger read attempted")
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", guarded):
            report = deliver_report(self.root, now=NOW, dry_run=True)
        self.assertNotIn("PRIVATE_SENTINEL", report["text"])
        self.assertTrue(all((self.root / name).read_text() == "PRIVATE_SENTINEL_DO_NOT_READ" for name in sentinels))

    def test_lane_identity_and_authority_cannot_be_overwritten(self) -> None:
        original = load_inputs(self.root)
        source = copy.deepcopy(original.source)
        source["order_authority"] = True
        with self.assertRaisesRegex(ReportError, "identity_or_authority"):
            build_report(ReportInputs(original.registry, source, original.progress), self.calendar, NOW)
        source = copy.deepcopy(original.source)
        source["strategy_id"] = original.registry["incumbent"]["strategy_id"]
        with self.assertRaisesRegex(ReportError, "identity_or_authority"):
            build_report(ReportInputs(original.registry, source, original.progress), self.calendar, NOW)

    def test_repeat_restart_generation_and_content_changes_send_once(self) -> None:
        calls = []
        transport = lambda text: calls.append(text) or Receipt("SENT", "ok", 200)
        first = deliver_report(self.root, now=NOW, dry_run=False, transport=transport)
        progress = read_json(self.root / "docs/research_progress.json")
        progress["completed"].append("A new code checkpoint, not a new market session.")
        (self.root / "docs/research_progress.json").write_bytes(json_bytes(progress))
        second = deliver_report(self.root, now=NOW + timedelta(minutes=30), dry_run=False, transport=transport)
        self.assertTrue(first["sent"])
        self.assertEqual(second["reason"], "session_already_reported")
        self.assertEqual(len(calls), 1)
        self.assertEqual(read_outbox(outbox_path(self.root))["deliveries"].popitem()[1]["state"], "SENT")

    def test_failed_delivery_never_marks_sent_and_can_retry(self) -> None:
        failed = deliver_report(self.root, now=NOW, dry_run=False, transport=lambda _: Receipt("FAILED", "http_400", 400))
        self.assertFalse(failed["sent"])
        self.assertFalse(failed["receipt_verified"])
        retried = deliver_report(self.root, now=NOW, dry_run=False, transport=lambda _: Receipt("SENT", "ok", 200))
        self.assertTrue(retried["sent"])
        state = read_outbox(outbox_path(self.root))
        self.assertEqual(next(iter(state["deliveries"].values()))["attempt_count"], 2)

    def test_unknown_delivery_blocks_automatic_resend(self) -> None:
        first = deliver_report(self.root, now=NOW, dry_run=False, transport=lambda _: Receipt("UNKNOWN", "timeout"))
        second = deliver_report(self.root, now=NOW, dry_run=False, transport=lambda _: self.fail("ambiguous resend"))
        self.assertEqual(first["reason"], "delivery_outcome_unknown")
        self.assertTrue(second["requires_review"])
        self.assertFalse(second["sent"])

    def test_crash_pending_is_unknown_not_success(self) -> None:
        def crash(_: str) -> Receipt:
            raise InterruptedError("synthetic interrupted process")

        with self.assertRaises(InterruptedError):
            deliver_report(self.root, now=NOW, dry_run=False, transport=crash)
        retry = deliver_report(self.root, now=NOW, dry_run=False, transport=lambda _: self.fail("resend after crash"))
        self.assertEqual(retry["reason"], "delivery_outcome_unknown")
        self.assertEqual(next(iter(read_outbox(outbox_path(self.root))["deliveries"].values()))["state"], "UNKNOWN")

    def test_concurrent_processes_send_exactly_once(self) -> None:
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        processes = [ctx.Process(target=concurrent_delivery, args=(str(self.root), queue)) for _ in range(3)]
        for process in processes:
            process.start()
        for process in processes:
            process.join(30)
            self.assertEqual(process.exitcode, 0)
        reasons = [queue.get(timeout=2) for _ in processes]
        self.assertEqual(reasons.count("posted"), 1)
        self.assertEqual((self.root / "transport_calls.txt").read_text(), "one-post\n")
        self.assertEqual(os.stat(outbox_path(self.root)).st_mode & 0o777, 0o600)

    def test_channel_and_stream_keys_are_explicit(self) -> None:
        for alias in ("canonical-slack", "approved-test-channel"):
            result = deliver_report(self.root, now=NOW, channel_alias=alias, dry_run=False, transport=lambda _: Receipt("SENT", "ok", 200))
            self.assertTrue(result["sent"])
        self.assertEqual(len(read_outbox(outbox_path(self.root))["deliveries"]), 2)

    def test_corrupt_outbox_fails_closed(self) -> None:
        path = outbox_path(self.root)
        path.parent.mkdir(parents=True)
        path.write_text("not-json")
        with self.assertRaisesRegex(ReportError, "input_unreadable"):
            deliver_report(self.root, now=NOW, dry_run=False, transport=lambda _: self.fail("corrupt state"))

    def test_outbox_symlink_is_rejected_before_reading(self) -> None:
        path = outbox_path(self.root)
        path.parent.mkdir(parents=True)
        path.symlink_to(self.root / "private-do-not-read.json")
        with self.assertRaisesRegex(ReportError, "symlink_rejected"):
            deliver_report(self.root, now=NOW, dry_run=True)

    def test_legacy_force_and_preview_never_enter_legacy_builder(self) -> None:
        from src.us_quant.slack_signal_notifier import SlackSignalConfig, build_signal_message, send_latest_signal

        config = SlackSignalConfig(
            webhook_url="", report_root=self.root,
            signal_path=self.root / "PRIVATE_DO_NOT_READ",
            state_path=self.root / "PRIVATE_DO_NOT_WRITE",
            single_stock=True, top_n=99,
        )
        with patch("src.us_quant.slack_signal_notifier._signal_hash", side_effect=AssertionError("legacy hash")), \
             patch("src.us_quant.slack_signal_notifier._legacy_build_signal_message", side_effect=AssertionError("legacy payload")), \
             patch("src.us_quant.slack_signal_notifier.post_research_report", side_effect=AssertionError("HTTP")):
            text, _, meta = build_signal_message(config)
            result = send_latest_signal(config, force=True, dry_run=True)
        self.assertEqual(meta["signal_rows"], 0)
        self.assertIn("Top30", text)
        self.assertIn("text", result)
        self.assertFalse((self.root / "PRIVATE_DO_NOT_WRITE").exists())

    def test_legacy_monitor_flags_do_not_refresh_broker_or_force_send(self) -> None:
        import scripts.slack_signal_monitor as monitor

        args = ["monitor", "--max-loops", "1", "--refresh-ibkr-snapshot", "--force-first", "--heartbeat-minutes", "0"]
        with patch.object(sys, "argv", args), \
             patch.object(monitor, "deliver_report", return_value={"sent": False, "reason": "dry_run"}) as sender, \
             patch.object(monitor, "configured_webhook", side_effect=AssertionError("credentials")), \
             patch.object(monitor, "post_to_slack", side_effect=AssertionError("HTTP")), \
             patch("builtins.print"):
            self.assertEqual(monitor.main(), 0)
        self.assertTrue(sender.call_args.kwargs["dry_run"])
        self.assertIsNone(sender.call_args.kwargs["transport"])

    def test_independent_discord_artifact_preserves_selected_metadata(self) -> None:
        import scripts.generate_personal_single_stock_signal_v1 as generator
        import src.us_quant.slack_signal_notifier as notifier

        config = notifier.SlackSignalConfig(webhook_url="", single_stock=True)
        original = ("synthetic artifact", "digest", {"selected_ticker": "SYNTHETIC", "selected_action": "watch"})
        with patch.object(notifier, "_legacy_build_signal_message", return_value=original):
            self.assertEqual(generator.build_legacy_single_stock_artifact(config), original)
        with self.assertRaisesRegex(ValueError, "requires_single_stock"):
            notifier.build_legacy_single_stock_artifact(notifier.SlackSignalConfig(webhook_url=""))

    def test_transport_errors_never_expose_secret_urls(self) -> None:
        from src.us_quant.research_delivery import validate_webhook

        webhook = "https://" + "hooks.slack.com/services/TEST/TEST/PRIVATE_CANARY"
        self.assertEqual(validate_webhook(webhook), webhook)
        for error, expected in (
            (URLError("PRIVATE_CANARY"), "UNKNOWN"),
            (HTTPError(webhook, 400, "PRIVATE_CANARY", {}, None), "FAILED"),
            (HTTPError(webhook, 500, "PRIVATE_CANARY", {}, None), "UNKNOWN"),
        ):
            with self.subTest(error=type(error).__name__), patch("src.us_quant.research_delivery.urlopen", side_effect=error):
                receipt = post_to_slack(webhook, "synthetic report")
                self.assertEqual(receipt.state, expected)
                self.assertNotIn("PRIVATE_CANARY", repr(receipt))

    def test_only_real_ok_response_is_a_successful_receipt(self) -> None:
        class Response:
            status = 200

            def __init__(self, body: bytes) -> None:
                self.body = body

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, limit: int) -> bytes:
                return self.body[:limit]

        webhook = "https://" + "hooks.slack.com/services/TEST/TEST/SYNTHETIC"
        for body, state in ((b"ok", "SENT"), (b"unexpected response", "UNKNOWN")):
            with patch("src.us_quant.research_delivery.urlopen", return_value=Response(body)):
                self.assertEqual(post_to_slack(webhook, "synthetic").state, state)

    def test_scheduler_uses_exact_bounded_reporter_and_no_broker_flags(self) -> None:
        import scripts.install_research_report_launchd as installer

        config = installer.configuration(self.root)
        self.assertEqual(config["Label"], "com.usstockqr.daily-research-report")
        self.assertEqual(config["StartInterval"], 900)
        self.assertTrue(config["RunAtLoad"])
        self.assertNotIn("KeepAlive", config)
        self.assertEqual(config["EnvironmentVariables"], {"TZ": "Asia/Shanghai"})
        self.assertEqual(config["ProgramArguments"], [
            str(self.root / ".venv/bin/python"), "-B", str(self.root / "scripts/send_slack_signal_once.py"),
            "--send", "--env-file", ".env.local", "--channel-alias", "canonical-slack", "--language", "zh",
        ])
        self.assertFalse((self.root / "artifacts/private").exists())

    def test_scheduler_requires_reviewed_receipt_before_bootstrap(self) -> None:
        import scripts.install_research_report_launchd as installer

        with self.assertRaisesRegex(ReportError, "receipt_required"):
            installer.reviewed_session_receipt(self.root, NOW)
        deliver_report(self.root, now=NOW, dry_run=False, transport=lambda _: Receipt("SENT", "ok", 200))
        self.assertEqual(installer.reviewed_session_receipt(self.root, NOW), "2026-10-06")

    def test_scheduler_accepts_both_launchctl_disabled_formats(self) -> None:
        import scripts.install_research_report_launchd as installer

        for value in ("true", "disabled"):
            responses = [
                SimpleNamespace(returncode=113, stdout="", stderr="Could not find service"),
                SimpleNamespace(returncode=0, stdout=f'"{installer.OLD_LABEL}" => {value}', stderr=""),
            ]
            with patch.object(installer.subprocess, "run", side_effect=responses):
                self.assertTrue(installer.old_monitor_contained("gui/501"))
        with patch.object(installer.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")):
            self.assertFalse(installer.old_monitor_contained("gui/501"))

    def test_env_configuration_is_explicit_and_not_serialized(self) -> None:
        env = self.root / ".env.local"
        env.write_text("SLACK_WEBHOOK_URL=not-a-valid-url\n")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ReportError, "configuration_invalid"):
                configured_webhook(env)
            preview = deliver_report(self.root, now=NOW, dry_run=True)
        self.assertNotIn("not-a-valid-url", preview["text"])

    def test_import_manifest_and_package_boundary(self) -> None:
        bundle = ROOT / "research/workbench"
        manifest = read_json(bundle / "IMPORT_MANIFEST.json")
        self.assertEqual(hashlib.sha256((bundle / "IMPORT_MANIFEST.json").read_bytes()).hexdigest(),
                         "3b2ef6789348ac88b8fdf82805415f518da85bb72c1d652735e7aeaaa63666b6")
        for name, digest in manifest["file_sha256"].items():
            self.assertEqual(hashlib.sha256((bundle / name).read_bytes()).hexdigest(), digest, name)
        self.assertFalse(any(name == "us_quant" or name.startswith("us_quant.") for name in sys.modules))
        self.assertEqual(read_json(ROOT / "config/nyse_calendar.json")["sessions_sha256"], CALENDAR_HASH)


if __name__ == "__main__":
    unittest.main()
