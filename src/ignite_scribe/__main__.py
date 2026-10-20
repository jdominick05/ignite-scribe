# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ignite-scribe setup [--history|--no-history] [--always-on|--no-always-on] [--service|--no-service]
              [--mcp-config FILE] [--yes]
ignite-scribe service {install,uninstall,start,stop,restart,status}
ignite-scribe status
python -m ignite_scribe snapshot [--json] [--policy FILE] [--no-text]
python -m ignite_scribe watch [--seconds N] [--policy FILE] [--raw]
python -m ignite_scribe history {status,search,purge} [QUERY] [--since 2h]
python -m ignite_scribe pause | resume       (always-on OCR)
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
    h = sub.add_parser("history", help="the opt-in screen history: status, search, or purge (delete it all)")
    h.add_argument("action", choices=["status", "search", "purge"])
    h.add_argument("query", nargs="?", default="")
    h.add_argument("--since", help="ISO 8601 or relative (90m, 2h, 1d)")
    h.add_argument("--match", help="only windows whose title or process contains this")
    h.add_argument("--yes", action="store_true", help="purge without asking")
    sub.add_parser("pause", help="pause always-on OCR (it holds no NPU context while paused)")
    sub.add_parser("resume", help="resume always-on OCR")
    sub.add_parser("mcp", help="serve the MCP tools on stdio")
    st = sub.add_parser("setup", help="choose history, always-on OCR and the background service; connect an agent")
    for flag, name in (("history", "history"), ("always-on", "always_on"), ("service", "service")):
        g = st.add_mutually_exclusive_group()
        g.add_argument(f"--{flag}", dest=name, action="store_true", default=None)
        g.add_argument(f"--no-{flag}", dest=name, action="store_false")
    st.add_argument("--mcp-config", help="also add the server to this JSON config file (an mcpServers object)")
    st.add_argument("--yes", action="store_true", help="no questions: keep current answers where not given")
    sv = sub.add_parser("service", help="the background process started at logon")
    sv.add_argument("action", choices=["install", "uninstall", "start", "stop", "restart", "status"])
    sub.add_parser("daemon", help="run the background process in the foreground (what the logon entry runs)")
    sub.add_parser("status", help="what ignite-scribe is doing now, across sessions")
    a = ap.parse_args(argv)
    if a.cmd == "setup":
        from .configure import run
        return run(a.history, a.always_on, a.service, a.mcp_config, a.yes)
    if a.cmd == "daemon":
        from .service import daemon
        return daemon()
    if a.cmd == "service":
        from . import service as svc
        if a.action == "install":
            print(f"installed: {svc.install()}", file=sys.stderr)
        elif a.action == "uninstall":
            svc.uninstall()
            print("uninstalled", file=sys.stderr)
        elif a.action == "start":
            print(f"running, pid {svc.start()}", file=sys.stderr)
        elif a.action == "stop":
            print("stopped" if svc.stop() else "did not stop in time", file=sys.stderr)
        elif a.action == "restart":
            svc.stop()
            print(f"running, pid {svc.start()}", file=sys.stderr)
        print(json.dumps(svc.status(), indent=1))
        return 0
    if a.cmd == "status":
        from . import service as svc
        from .always_on import PAUSE_FILE
        from .tray import read_beacons, summarize
        beacons = read_beacons()
        state, tip = summarize(beacons, PAUSE_FILE.exists())
        print(json.dumps(dict(state=state, summary=tip, service=svc.status(), sessions=beacons), indent=1))
        return 0
    if a.cmd in ("pause", "resume"):
        from .always_on import PAUSE_FILE
        if a.cmd == "pause":
            PAUSE_FILE.parent.mkdir(parents=True, exist_ok=True)
            PAUSE_FILE.write_bytes(b"")
        else:
            PAUSE_FILE.unlink(missing_ok=True)
        print(f"always-on OCR {'paused' if a.cmd == 'pause' else 'resumed'}", file=sys.stderr)
        return 0
    if a.cmd == "mcp":
        from .mcp_server import main as serve
        serve()
        return 0
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if a.cmd == "history":
        from .history import History, default_path, parse_time
        if not default_path().exists():
            print(f"no history at {default_path()} (it is opt-in: IGNITE_SCRIBE_HISTORY=1)", file=sys.stderr)
            return 1
        hist = History(record=False)
        try:
            if a.action == "status":
                print(json.dumps(hist.status(), indent=1))
            elif a.action == "search":
                print(json.dumps(hist.search(a.query, parse_time(a.since), None, a.match), indent=1,
                                 ensure_ascii=False))
            else:
                if not a.yes and input(f"delete all screen history in {hist.path}? [y/N] ").lower() != "y":
                    return 1
                print(f"deleted {hist.purge()} chunks", file=sys.stderr)
        finally:
            hist.close()
        return 0
    from .redact import Policy, policy_path
    policy = Policy.load(a.policy or policy_path())
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
