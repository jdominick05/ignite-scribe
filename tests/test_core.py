# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline checks: redaction, rendering, and the MCP protocol over a fake screen. The live Win32 source has one
smoke test, skipped off Windows."""
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ignite_scribe.mcp_server import Server, main
from ignite_scribe.redact import WITHHELD, Policy, luhn, scrub
from ignite_scribe.snapshot import UNTRUSTED, Rect, Snapshot, TextBlock, Window


def fake():
    ws = [Window(1, "README.md - Visual Studio Code", "Code.exe", 10, Rect(0, 0, 1920, 1080), 0, focused=True,
                 text=[TextBlock("token ghp_" + "a" * 36 + " end"), TextBlock("card 4111 1111 1111 1111")]),
          Window(2, "Vault", "KeePassXC.exe", 11, Rect(100, 100, 900, 700), 1),
          Window(3, "Bank of Example - Sign in", "msedge.exe", 12, Rect(0, 0, 800, 600), 2, minimized=True)]
    return Snapshot.now(ws, [Rect(0, 0, 1920, 1080)])


def test_scrub_secrets_and_cards():
    t, n = scrub("a sk-" + "x" * 20 + " b AKIA" + "A" * 16 + " c 4111-1111-1111-1111 d 1234 5678 9012 3456")
    assert "[api key withheld]" in t and "[aws key withheld]" in t and "[card number withheld]" in t
    assert "1234 5678 9012 3456" in t   # fails Luhn: an ordinary number, kept
    assert n == 3
    assert luhn("4111111111111111") and not luhn("1234567890123456")


def test_policy_withholds_excluded_windows_and_keeps_position():
    snap, counts = Policy().apply(fake())
    by = {w.hwnd: w for w in snap.windows}
    assert by[2].title == WITHHELD and by[2].redacted and by[2].rect.right == 900   # password manager
    assert by[3].redacted                                                          # title pattern: bank / sign in
    assert "[github token withheld]" in by[1].text[0].text and "withheld" in by[1].text[1].text
    assert counts == dict(windows_withheld=2, secrets_withheld=2)
    assert fake().windows[1].title == "Vault"   # the source snapshot is not modified


def test_policy_file_extends_defaults(tmp_path):
    f = tmp_path / "policy.json"
    f.write_text(json.dumps({"exclude_processes": ["Code.exe"]}), encoding="utf-8")
    snap, counts = Policy.load(f).apply(fake())
    assert counts["windows_withheld"] == 3


def test_renderings_carry_the_untrusted_note():
    snap, _ = Policy().apply(fake())
    md, js = snap.to_markdown(), json.loads(snap.to_json())
    assert UNTRUSTED in md and js["note"] == UNTRUSTED
    assert "Focused: **README.md - Visual Studio Code**" in md and "| 1 | " + WITHHELD in md


def test_mcp_protocol_over_stdio_framing():
    msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "get_screen_state",
                                                                            "arguments": {"format": "json"}}},
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "list_windows",
                                                                            "arguments": {"match": "code"}}},
            {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "nope"}},
            {"jsonrpc": "2.0", "id": 6, "method": "bogus"}]
    out = io.StringIO()
    main(Server(fake, Policy()), io.StringIO("\n".join(json.dumps(m) for m in msgs) + "\n"), out)
    replies = [json.loads(ln) for ln in out.getvalue().splitlines()]
    assert [r["id"] for r in replies] == [1, 2, 3, 4, 5, 6]   # no reply to the notification
    assert replies[0]["result"]["serverInfo"]["name"] == "ignite-scribe"
    assert {t["name"] for t in replies[1]["result"]["tools"]} == {"scribe_status", "get_screen_state", "list_windows"}
    state = json.loads(replies[2]["result"]["content"][0]["text"])
    assert state["note"] == UNTRUSTED and state["redaction"]["windows_withheld"] == 2
    listed = json.loads(replies[3]["result"]["content"][0]["text"])["windows"]
    assert [w["process"] for w in listed] == ["Code.exe"]
    assert replies[4]["result"]["isError"] and replies[5]["error"]["code"] == -32601


@pytest.mark.skipif(sys.platform != "win32", reason="Win32 source")
def test_live_windows_snapshot_smoke():
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ, PYTHONPATH=src + os.pathsep + os.environ.get("PYTHONPATH", ""))
    out = subprocess.run([sys.executable, "-m", "ignite_scribe", "snapshot", "--json"], capture_output=True,
                         text=True, encoding="utf-8", timeout=60, env=env)
    assert out.returncode == 0, out.stderr
    d = json.loads(out.stdout)
    assert d["note"] == UNTRUSTED and d["monitors"] and isinstance(d["windows"], list)
