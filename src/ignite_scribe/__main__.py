# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""python -m ignite_scribe snapshot [--json] [--policy FILE] [--no-text]  |  python -m ignite_scribe mcp"""
import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ignite-scribe", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot", help="print the redacted screen state once")
    s.add_argument("--json", action="store_true")
    s.add_argument("--policy", help="policy file (JSON), merged with the defaults")
    s.add_argument("--no-text", action="store_true", help="windows only, no UI Automation content")
    s.add_argument("--max-windows", type=int, default=6, help="windows to read, focused first")
    sub.add_parser("mcp", help="serve the MCP tools on stdio")
    a = ap.parse_args(argv)
    if a.cmd == "mcp":
        from .mcp_server import main as serve
        serve()
        return 0
    from .collect import collect
    from .redact import Policy
    policy = Policy.load(a.policy)
    snap, counts = policy.apply(collect(policy, text=not a.no_text, max_windows=a.max_windows))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(snap.to_json() if a.json else snap.to_markdown())
    print(f"redaction: {counts}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
