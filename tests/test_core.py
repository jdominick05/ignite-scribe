# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline checks: redaction, rendering, the MCP protocol over a fake screen, and that excluded windows are never
walked. Live checks (Windows only): the Win32 snapshot, and UI Automation on a native window holding a password
field whose value must appear nowhere."""
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ignite_scribe import collect as collect_mod
from ignite_scribe.mcp_server import Server, main
from ignite_scribe.redact import WITHHELD, Policy, luhn, scrub
from ignite_scribe.snapshot import UNTRUSTED, Rect, Snapshot, TextBlock, Window


def fake(text=True, max_windows=6):
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
    assert {t["name"] for t in replies[1]["result"]["tools"]} == {"scribe_status", "get_screen_state",
                                                                 "screen_changes", "find_ui_elements", "ocr_windows",
                                                                 "search_screen_history", "screen_at", "list_windows",
                                                                 "wait_for"}
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


def test_excluded_windows_are_never_walked(monkeypatch):
    walked = []
    monkeypatch.setattr(collect_mod.windows, "snapshot", lambda: fake())
    monkeypatch.setattr(collect_mod.uia, "available", lambda: True)
    monkeypatch.setattr(collect_mod.uia, "walk",
                        lambda hwnd, cap, sec: (walked.append(hwnd), ([TextBlock(f"in {hwnd}")], []))[1])
    snap = collect_mod.collect(Policy())
    assert walked == [1]           # KeePassXC (process) and the bank sign-in (title, and minimized) are skipped
    assert "uia" in snap.sources and snap.windows[0].text[0].text == "in 1"


def test_missing_uia_degrades_to_windows_only(monkeypatch):
    monkeypatch.setattr(collect_mod.windows, "snapshot", lambda: fake())
    monkeypatch.setattr(collect_mod.uia, "available", lambda: False)
    snap = collect_mod.collect(Policy())
    assert snap.sources[-1].startswith("uia unavailable")


@pytest.fixture
def native_window():
    texts = dict(title="scribe uia probe", label="Probe label 7Q", plain="plain value 3K", secret="hunter2-Zx9-secret",
                 lines="first line 4R|second line 8W||fourth line 2P")
    proc = subprocess.Popen([sys.executable, str(Path(__file__).with_name("password_window.py")), *texts.values()],
                            stdout=subprocess.PIPE, text=True)
    try:
        hwnd = int(proc.stdout.readline())
        yield hwnd, texts
    finally:
        proc.kill()
        proc.wait(timeout=10)


@pytest.mark.skipif(sys.platform != "win32", reason="UI Automation")
def test_uia_reads_text_but_never_a_password(native_window):
    pytest.importorskip("comtypes")
    from ignite_scribe import uia
    hwnd, t = native_window
    blocks, notes = uia.walk(hwnd, node_cap=200, seconds=5.0)
    dump = json.dumps([b.__dict__ for b in blocks], default=str)
    assert t["label"] in dump and t["plain"] in dump, (dump, notes)
    assert "[password field]" in dump
    assert t["secret"] not in dump and "hunter2" not in dump
    assert any(b.role == "Edit" for b in blocks)
    multi = [b for b in blocks if "first line 4R" in b.text]
    assert len(multi) == 1 and multi[0].text.endswith("first line 4R\nsecond line 8W\n\nfourth line 2P"), multi


@pytest.mark.skipif(sys.platform != "win32", reason="UI Automation")
def test_uia_skips_the_window_when_policy_excludes_its_title(native_window, monkeypatch):
    pytest.importorskip("comtypes")
    from ignite_scribe import uia
    hwnd, t = native_window
    calls = []
    real = uia.walk
    monkeypatch.setattr(uia, "walk", lambda h, *a: (calls.append(h), real(h, *a))[1])
    policy = Policy(exclude_title_patterns=[r"uia probe"])
    snap, counts = policy.apply(collect_mod.collect(policy, max_windows=50))
    assert hwnd not in calls
    w = next(w for w in snap.windows if w.hwnd == hwnd)
    assert w.redacted and not w.text and t["plain"] not in snap.to_json()


class FakeElement:
    """Stands in for a cached UIA element whose Value and Text pattern would leak if they were ever requested."""
    def __init__(self, password, name="Field", value="hunter2-Zx9-secret", text=None):
        self.CachedIsPassword, self.CachedName, self.CachedControlType = password, name, 50004
        self.value, self.value_requests = value, 0
        self.text, self.pattern_requests = text, 0

    @property
    def CachedBoundingRectangle(self):
        from types import SimpleNamespace
        return SimpleNamespace(left=0, top=0, right=10, bottom=10)

    def GetCachedPropertyValue(self, pid):
        from ignite_scribe import uia
        return {uia.IS_VALUE_AVAILABLE: self.value is not None, uia.IS_TEXT_AVAILABLE: self.text is not None}[pid]

    def GetCurrentPattern(self, pid):
        from types import SimpleNamespace
        self.pattern_requests += 1
        rng = SimpleNamespace(GetText=lambda n: self.text)
        ranges = SimpleNamespace(Length=1, GetElement=lambda i: rng)
        tp = SimpleNamespace(GetVisibleRanges=lambda: ranges)
        return SimpleNamespace(QueryInterface=lambda iface: tp)

    def GetCurrentPropertyValue(self, pid):
        self.value_requests += 1
        return self.value


def test_password_value_is_never_requested():
    """The live test cannot show this: Windows itself returns '' for a password Edit's Value. Here the provider
    would hand the secret over, so only ignite-scribe's own guard keeps it out."""
    from ignite_scribe import uia
    uia._UIA = uia._UIA or type("UIA", (), {"IUIAutomationTextPattern": None})
    pw = FakeElement(True, text="hunter2-Zx9-secret")
    plain = FakeElement(False, value="plain value")
    b, role, covered = uia._block(pw)
    assert b.text == "[password field]" and role == "Edit" and not covered
    assert pw.value_requests == 0 and pw.pattern_requests == 0
    b, _, _ = uia._block(plain)
    assert b.text == "Field: plain value" and plain.value_requests == 1


def test_text_pattern_multiline_covers_subtree_and_drops_repeats():
    from ignite_scribe import uia
    uia._UIA = uia._UIA or type("UIA", (), {"IUIAutomationTextPattern": None})
    term = FakeElement(False, name="Shell", value=None, text="PS> dir   \r\n\r\n\r\n\r\nfile.txt    \r\n      ")
    b, _, covered = uia._block(term)
    assert b.text == "Shell\nPS> dir\n\nfile.txt" and covered and term.pattern_requests == 1
    same = FakeElement(False, name="Address", value="C:\\data", text="C:\\data")
    b, _, covered = uia._block(same)
    assert b.text == "Address: C:\\data" and not covered       # the text only repeated the value


def test_clean_caps_long_text():
    from ignite_scribe import uia
    out = uia.clean("x" * (uia.MAX_TEXT + 50))
    assert len(out) == uia.MAX_TEXT + len("\n[... truncated]") and out.endswith("[... truncated]")


def test_markdown_fences_multiline_blocks():
    ws = [Window(1, "Terminal", "WindowsTerminal.exe", 1, Rect(0, 0, 10, 10), 0,
                 text=[TextBlock("PS> dir\n| not a table |\n~~~~", None, "uia", "Text")])]
    md = Snapshot.now(ws).to_markdown()
    assert "~~~~text\nPS> dir\n| not a table |\n~ ~ ~ ~\n~~~~" in md
