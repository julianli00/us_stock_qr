from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.us_quant.research_delivery import configured_webhook, deliver_report, post_to_slack  # noqa: E402
from src.us_quant.research_reporting import ReportError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview or deliver the canonical daily research status.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--send", action="store_true", help="Explicitly enable a real Slack POST.")
    mode.add_argument("--dry-run", action="store_true", help="Side-effect-free preview (the default).")
    parser.add_argument("--env-file", default=".env.local")
    parser.add_argument("--webhook-env-var", default="SLACK_WEBHOOK_URL")
    parser.add_argument("--channel-alias", default="canonical-slack")
    parser.add_argument("--language", choices=["zh", "en"], default="zh")
    parser.add_argument("--json", action="store_true", help="Print the safe report envelope.")
    parser.add_argument("--force", action="store_true", help="Deprecated; never bypasses session or freshness gates.")
    parser.add_argument("--signal-file", help="Deprecated; legacy trading payloads are not read.")
    parser.add_argument("--status-file", help="Deprecated; canonical lane identities are preserved.")
    parser.add_argument("--top-n", type=int, help="Deprecated; new watch ideas are always capped at five.")
    parser.add_argument("--account-equity", type=float, help="Deprecated; holdings are outside this reporter.")
    args = parser.parse_args()
    try:
        transport = None
        if args.send:
            webhook = configured_webhook(ROOT / args.env_file, args.webhook_env_var)
            transport = lambda text: post_to_slack(webhook, text)
        result = deliver_report(
            ROOT,
            channel_alias=args.channel_alias,
            language=args.language,
            dry_run=not args.send,
            transport=transport,
        )
    except ReportError as exc:
        print(json.dumps({"sent": False, "reason": str(exc)}), file=sys.stderr)
        return 1
    if not args.send and not args.json:
        print(result["text"])
    else:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 1 if result["reason"] in {"delivery_failed", "delivery_outcome_unknown"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
