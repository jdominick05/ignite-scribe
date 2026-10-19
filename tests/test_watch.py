# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Change-driven capture. Offline: the diff, the merging, and the MCP tools over a fake watcher. Live (Windows):
the watcher sees a native window open, its text change, and close, and never reports its password."""
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ignite_scribe.changes import compact, diff
from ignite_scribe.mcp_server import Server
from ignite_scribe.redact import Policy
from ignite_scribe.snapshot import UNTRUSTED, Rect, Snapshot, TextBlock, Window


def win(hwnd, title="A", z=0, text=(), **kw):
    return Window(hwnd, title, "app.exe", 100 + hwnd, kw.pop("rect", Rect(0, 0, 100, 100)), z,
                  text=[TextBlock(t) for t in text], **kw)


def test_diff_window_lifecycle_and_focus():
    old = Snapshot.now([win(1, focused=True), win(2, "B", 1)])
    new = Snapshot.now([win(1, "A2", 1, rect=Rect(5, 5, 100, 100)), win(3, "C", 0, focused=True)])
    kinds = {(e["kind"], e["hwnd"]) for e in diff(old, new)}
    assert kinds == {("opened", 3), ("closed", 2), ("focus", 3), ("title", 1), ("moved", 1)}
    assert diff(None, new) == []


def test_diff_content_lists_new_lines_only_when_both_sides_were_read():
    old = Snapshot.now([win(1, text=["PS> dir\nfile.txt"]), win(2, text=["kept"])])
    new = Snapshot.now([win(1, text=["PS> dir\nfile.txt\nPS> echo hi\nhi"]), win(2)])
    (e,) = diff(old, new)
    assert e["kind"] == "content" and e["hwnd"] == 1 and e["added"] == ["PS> echo hi", "hi"] and e["removed"] == 0
    # window 2 was not read this time: no "content removed" noise


def test_diff_minimize_is_not_also_a_move():
    old = Snapshot.now([win(1)])
    new = Snapshot.now([win(1, minimized=True, rect=Rect(9, 9, 99, 99))])
    assert [e["kind"] for e in diff(old, new)] == ["minimized"]


def test_compact_merges_repeats_per_window_and_keeps_order():
    evs = [dict(seq=1, t=1, kind="title", hwnd=1, title="◐ x", old="◓ x", process="t"),
           dict(seq=2, t=2, kind="focus", hwnd=2, title="B", process="b"),
           dict(seq=3, t=3, kind="title", hwnd=1, title="◑ x", old="◐ x", process="t"),
           dict(seq=4, t=4, kind="content", hwnd=1, title="◑ x", process="t", added=["a"], added_total=1, removed=0),
           dict(seq=5, t=5, kind="content", hwnd=1, title="◑ x", process="t", added=["a", "b"], added_total=2,
                removed=1)]
    out = compact(evs)
    assert [(e["kind"], e["seq"]) for e in out] == [("title", 3), ("focus", 2), ("content", 5)]
    assert out[0]["old"] == "◓ x" and out[0]["title"] == "◑ x" and out[0]["merged"] == 2
    assert out[2]["added"] == ["a", "b"] and out[2]["added_total"] == 3 and out[2]["removed"] == 1
    assert evs[0]["title"] == "◐ x"        # inputs untouched


class FakeWatcher:
    max_windows = 6

    def __init__(self):
        self.snap = Snapshot.now([win(1, "From watcher", focused=True, text=["watched text"])])

    def snapshot(self):
        return self.snap, dict(windows_withheld=0, secrets_withheld=0), self.snap.taken_at

    def changes(self, since=0, limit=200, merge=True):
        evs = [dict(seq=7, t=1.0, kind="opened", hwnd=1, title="From watcher", process="app.exe")]
        evs = [e for e in evs if e["seq"] > since]
        return dict(seq=7, since=since, next_since=7, truncated=False, more=False, events=evs)

    def status(self):
        return dict(running=True, dirty_source="fake")


def test_mcp_tools_use_the_watcher_state():
    fresh = Snapshot.now([win(9, "Read now", focused=True)])
    s = Server(lambda text=True, max_windows=6: fresh, Policy(), watcher=FakeWatcher())
    assert "From watcher" in s.call("get_screen_state", {})
    assert "watched text" in s.call("get_screen_state", {})
    assert "Read now" in s.call("get_screen_state", {"refresh": True})
    assert "Read now" in s.call("get_screen_state", {"max_windows": 50})      # more than the watcher reads
    ch = json.loads(s.call("screen_changes", {"since": 0}))
    assert ch["note"] == UNTRUSTED and ch["events"][0]["kind"] == "opened" and ch["next_since"] == 7
    assert json.loads(s.call("screen_changes", {"since": 7}))["events"] == []
    assert s.call("scribe_status", {})["watch"]["dirty_source"] == "fake"


def test_screen_changes_reports_when_watching_is_off():
    s = Server(lambda text=True, max_windows=6: Snapshot.now([]), Policy())     # injected source: no watcher
    with pytest.raises(RuntimeError, match="watcher is off"):
        s.call("screen_changes", {})
    assert s.call("scribe_status", {})["watch"]["running"] is False


@pytest.mark.skipif(sys.platform != "win32", reason="WinEvents and Desktop Duplication")
def test_live_watcher_sees_open_edit_close_and_no_password():
    pytest.importorskip("comtypes")
    from ignite_scribe.watch import Watcher
    w = Watcher(Policy(), interval=0.1).start()
    try:
        seq0 = w.changes(10 ** 12)["seq"]
        p = subprocess.Popen([sys.executable, str(Path(__file__).with_name("password_window.py")), "scribe watch probe",
                              "label W1", "plain W2", "secret-W3-hunter2", "line W4|line W5"],
                             stdout=subprocess.PIPE, text=True)
        try:
            hwnd = int(p.stdout.readline())

            def events():
                return [e for e in w.changes(seq0, 10 ** 6, merge=False)["events"] if e.get("hwnd") == hwnd]

            def wait_for(pred, timeout=10.0):
                end = time.monotonic() + timeout
                while time.monotonic() < end:
                    if pred(events()):
                        return True
                    time.sleep(0.1)
                return False

            assert wait_for(lambda es: any(e["kind"] == "opened" for e in es)), events()
            snap, _, _ = w.snapshot()
            assert any(x.hwnd == hwnd for x in snap.windows)
            user32 = ctypes.WinDLL("user32")
            user32.FindWindowExW.restype = ctypes.c_void_p
            user32.FindWindowExW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_wchar_p]
            edit = user32.FindWindowExW(hwnd, None, "EDIT", None)             # the first Edit: the plain one
            user32.SendMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_wchar_p]
            user32.SendMessageW(edit, 0x000C, None, "changed W6")              # WM_SETTEXT
            assert wait_for(lambda es: any(e["kind"] == "content" and any("changed W6" in a for a in e["added"])
                                           for e in es)), events()
        finally:
            p.kill()
            p.wait(timeout=10)
        assert wait_for(lambda es: any(e["kind"] == "closed" for e in es)), events()
        everything = json.dumps(w.changes(0, 10 ** 6, merge=False), ensure_ascii=False)
        state = w.snapshot()[0].to_json()
        assert "hunter2" not in everything and "hunter2" not in state
    finally:
        w.stop()
    assert not w.running
