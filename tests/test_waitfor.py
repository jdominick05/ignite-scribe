# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""wait_for against the real Watcher's event ring and condition variable, driven from a thread (no desktop)."""
import collections
import json
import threading
import time

from ignite_scribe.changes import diff
from ignite_scribe.mcp_server import Server
from ignite_scribe.snapshot import Rect, Snapshot, TextBlock, Window
from ignite_scribe.waitfor import wait_for
from ignite_scribe.watch import Watcher


def snap(lines, title="build - Terminal", extra=()):
    ws = [Window(1, title, "WindowsTerminal.exe", 5, Rect(0, 0, 800, 600), 0, focused=True,
                 text=[TextBlock("\n".join(lines))])]
    ws += [Window(2 + i, t, "app.exe", 6, Rect(0, 0, 100, 100), 1 + i) for i, t in enumerate(extra)]
    return Snapshot(time.time(), ws)


def watcher(first):
    w = Watcher.__new__(Watcher)
    w.events, w.seq, w._lock = collections.deque(maxlen=100), 0, threading.Lock()
    w._changed = threading.Condition(w._lock)
    w.state, w.counts, w.as_of = first, {}, first.taken_at
    return w


def publish(w, new_snap):
    with w._lock:
        w._append(diff(w.state, new_snap), new_snap.taken_at)
        w.state = new_snap


def later(delay, fn, *a):
    t = threading.Timer(delay, fn, a)
    t.start()
    return t


def test_text_that_appears_later_wakes_the_waiter_promptly():
    w = watcher(snap(["compiling"]))
    later(0.3, publish, w, snap(["compiling", "Build succeeded in 4.2 s"]))
    t = time.monotonic()
    r = wait_for(w, text="build SUCCEEDED", timeout_s=5)
    assert r["result"] == "matched" and r["hit"]["line"] == "Build succeeded in 4.2 s" and r["hit"]["via"] == "content"
    assert 0.25 < time.monotonic() - t < 1.0                  # woken by the event, not by the timeout


def test_text_already_on_screen_counts_at_once():
    w = watcher(snap(["error: linker failed"]))
    r = wait_for(w, text="linker failed", timeout_s=5)
    assert r["result"] == "matched" and r["hit"]["via"] == "on screen" and r["waited_s"] < 0.2


def test_text_that_came_and_went_since_the_baseline_is_not_missed():
    w = watcher(snap(["a"]))
    base = w.seq
    publish(w, snap(["a", "Saved."]))
    publish(w, snap(["a"]))                                   # gone again before the agent asked
    r = wait_for(w, text="saved", since=base, timeout_s=1)
    assert r["result"] == "matched" and r["hit"]["via"] == "content"


def test_gone_and_timeout():
    w = watcher(snap(["x"], extra=["Installing..."]))
    later(0.2, publish, w, snap(["x"]))
    assert wait_for(w, gone="installing", timeout_s=5)["result"] == "matched"
    r = wait_for(w, text="never shown", timeout_s=0.3)
    assert r["result"] == "timeout" and 0.25 < r["waited_s"] < 1.0


def test_action_check_reports_what_changed_after_it_settles():
    w = watcher(snap(["form"]))
    seq = w.seq
    later(0.1, publish, w, snap(["form"], extra=["Confirm delete?"]))
    later(0.3, publish, w, snap(["form", "1 item selected"], extra=["Confirm delete?"]))
    t = time.monotonic()
    r = wait_for(w, since=seq, quiet_s=0.4, timeout_s=5)
    kinds = {e["kind"] for e in r["events"]}
    assert r["result"] == "matched" and {"opened", "content"} <= kinds
    assert time.monotonic() - t >= 0.65                       # waited out the quiet period after the last change
    assert r["next_since"] == w.seq
    nothing = wait_for(w, since=w.seq, quiet_s=0.2, timeout_s=0.4)
    assert nothing["result"] == "timeout" and nothing["events"] == []   # the action changed nothing


def test_match_and_kinds_narrow_what_counts():
    w = watcher(snap(["a"]))
    seq = w.seq
    publish(w, snap(["a"], extra=["Other window"]))           # an 'opened' in another window
    r = wait_for(w, since=seq, match="terminal", timeout_s=0.3)
    assert r["result"] == "timeout"
    r = wait_for(w, since=seq, kinds=["opened"], timeout_s=0.3)
    assert r["result"] == "matched" and r["events"][0]["title"] == "Other window"


def test_redacted_windows_never_match():
    w = watcher(Snapshot(time.time(), [Window(9, "Vault", "KeePassXC.exe", 1, Rect(0, 0, 9, 9), 0, redacted=True,
                                              text=[TextBlock("hunter2")])]))
    assert wait_for(w, text="hunter2", timeout_s=0.2)["result"] == "timeout"


def test_mcp_wait_for_tool():
    w = watcher(snap(["done: 3 tests passed"]))
    s = Server(source=lambda **k: w.state)
    s._watcher = lambda: w
    out = json.loads(s.call("wait_for", {"text": "tests passed", "timeout_s": 2}))
    assert out["result"] == "matched" and "untrusted" in out["note"].lower()
    try:
        s.call("wait_for", {"text": "a", "gone": "b"})
        assert False, "text and gone together must be refused"
    except ValueError:
        pass


def test_a_text_wait_lists_no_events_unless_asked():
    w = watcher(snap(["a"]))
    seq = w.seq
    publish(w, snap(["a", "b", "done"]))
    assert "events" not in wait_for(w, text="done", since=seq, timeout_s=1)
    assert wait_for(w, text="done", since=seq, timeout_s=1, events=True)["events"]
