from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.us_quant.research_delivery import configured_webhook, deliver_report, post_to_slack  # noqa: E402
from src.us_quant.research_reporting import ReportError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Compatibility monitor for guarded research reports; never refreshes brokers.")
    parser.add_argument("--send", action="store_true", help="Explicit opt-in; legacy installed arguments only preview.")
    parser.add_argument("--env-file", default=".env.local")
    parser.add_argument("--webhook-env-var", default="SLACK_WEBHOOK_URL")
    parser.add_argument("--channel-alias", default="canonical-slack")
    parser.add_argument("--language", choices=["zh", "en"], default="zh")
    parser.add_argument("--interval-seconds", type=int, default=900)
    parser.add_argument("--max-loops", type=int, default=0)
    for flag in ("--market-hours-only", "--force-first", "--refresh-ibkr-snapshot"):
        parser.add_argument(flag, action="store_true", help="Deprecated and ignored.")
    for flag in ("--signal-file", "--status-file", "--heartbeat-minutes", "--top-n", "--account-equity",
                 "--ibkr-host", "--ibkr-port", "--ibkr-client-id", "--ibkr-account", "--market-data-type"):
        parser.add_argument(flag, help="Deprecated and ignored.")
    args = parser.parse_args()
    loops = 0
    while True:
        loops += 1
        try:
            transport = None
            if args.send:
                webhook = configured_webhook(ROOT / args.env_file, args.webhook_env_var)
                transport = lambda text: post_to_slack(webhook, text)
            result = deliver_report(
                ROOT, channel_alias=args.channel_alias, language=args.language,
                dry_run=not args.send, transport=transport,
            )
            print(json.dumps({key: value for key, value in result.items() if key != "text"}, ensure_ascii=False), flush=True)
        except ReportError as exc:
            print(json.dumps({"sent": False, "reason": str(exc)}), file=sys.stderr, flush=True)
            return 1
        if args.max_loops and loops >= args.max_loops:
            return 1 if result["reason"] in {"delivery_failed", "delivery_outcome_unknown"} else 0
        time.sleep(max(30, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
