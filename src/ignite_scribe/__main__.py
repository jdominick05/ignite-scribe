# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""python -m ignite_scribe snapshot [--json] [--policy FILE] [--no-text]
python -m ignite_scribe watch [--seconds N] [--policy FILE] [--raw]
python -m ignite_scribe mcp"""
import argparse
import json
import sys
import time


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ignite-scribe", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot", help="print the redacted screen state once")
    s.add_argument("--json", action="store_true")
    s.add_argument("--policy", help="policy file (JSON), merged with the defaults")
    s.add_argument("--no-text", action="store_true", help="windows only, no UI Automation content")
    s.add_argument("--max-windows", type=int, default=6, help="windows to read, focused first")
    w = sub.add_parser("watch", help="print change events as JSON lines until Ctrl+C or --seconds")
    w.add_argument("--seconds", type=float, default=0, help="stop after this long (0: until Ctrl+C)")
    w.add_argument("--policy", help="policy file (JSON), merged with the defaults")
    w.add_argument("--raw", action="store_true", help="every event, not merged per window per second")
    sub.add_parser("mcp", help="serve the MCP tools on stdio")
    a = ap.parse_args(argv)
    if a.cmd == "mcp":
        from .mcp_server import main as serve
        serve()
        return 0
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from .redact import Policy
    policy = Policy.load(a.policy)
    if a.cmd == "watch":
        from .watch import Watcher
        watcher = Watcher(policy).start()
        end = time.monotonic() + a.seconds if a.seconds else None
        since = watcher.changes(10 ** 12)["seq"]
        try:
            while end is None or time.monotonic() < end:
                time.sleep(1.0)
                out = watcher.changes(since, 10 ** 6, merge=not a.raw)
                for e in out["events"]:
                    print(json.dumps(e, ensure_ascii=False), flush=True)
                since = out["next_since"]
        except KeyboardInterrupt:
            pass
        finally:
            st = watcher.status()
            watcher.stop()
            print(json.dumps({k: st[k] for k in ("dirty_source", "refreshes", "walks", "refresh_ms_mean",
                                                 "thread_cpu_percent")}), file=sys.stderr)
        return 0
    from .collect import collect
    snap, counts = policy.apply(collect(policy, text=not a.no_text, max_windows=a.max_windows))
    print(snap.to_json() if a.json else snap.to_markdown())
    print(f"redaction: {counts}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
