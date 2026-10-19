# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Screen history, on a temp file: opt-in, encrypted at rest, retention, search, the past snapshot, the MCP tools."""
import json
import time

import pytest

from ignite_scribe import history as H
from ignite_scribe.changes import diff
from ignite_scribe.mcp_server import Server
from ignite_scribe.redact import Policy
from ignite_scribe.snapshot import Rect, Snapshot, TextBlock, Window, from_dict


def win(hwnd, title, process, lines=(), z=0, focused=False):
    return Window(hwnd, title, process, 100 + hwnd, Rect(0, 0, 800, 600), z, focused,
                  text=[TextBlock("\n".join(lines))] if lines else [])


def snap(t, *ws):
    return Snapshot(t, list(ws), sources=["windows", "uia"])


def wait_flushed(h, n):
    end = time.time() + 5
    while time.time() < end:
        h.flush()
        if h.status()["chunks"] >= n:
            return
        time.sleep(0.05)
    raise AssertionError("history did not flush")


def record_pair(h, policy, s0, s1):
    """What the watcher does: redact, diff the redacted states, hand both to the recorder."""
    r0, _ = policy.apply(s0)
    r1, _ = policy.apply(s1)
    evs = diff(r0, r1)
    for i, e in enumerate(evs, 1):
        e["seq"], e["t"] = i, r1.taken_at
    h.record([], r0)
    h.record(evs, r1)


@pytest.fixture
def hist(tmp_path):
    h = H.History(tmp_path / "h.db", snapshot_s=0, flush_s=0.05)
    yield h
    h.close()


def test_history_is_opt_in(monkeypatch):
    monkeypatch.delenv("IGNITE_SCRIBE_HISTORY", raising=False)
    assert not H.enabled(Policy())
    assert H.enabled(Policy(history=True))
    monkeypatch.setenv("IGNITE_SCRIBE_HISTORY", "0")
    assert not H.enabled(Policy(history=True))
    monkeypatch.setenv("IGNITE_SCRIBE_HISTORY", "1")
    assert H.enabled(Policy())
    monkeypatch.delenv("IGNITE_SCRIBE_HISTORY")
    s = Server(source=lambda **k: snap(time.time()), policy=Policy())
    with pytest.raises(RuntimeError, match="opt-in"):
        s.call("search_screen_history", {"query": "x"})
    assert s.history is None


def test_nothing_on_disk_is_readable_and_excluded_content_never_recorded(hist, tmp_path):
    now = time.time()
    p = Policy()
    s0 = snap(now - 10, win(1, "Notes", "notepad.exe", ["shopping list"], focused=True),
              win(2, "Vault", "keepass.exe", ["hunter2 the master"], z=1))
    s1 = snap(now, win(1, "Notes", "notepad.exe", ["shopping list", "zanzibar meeting at noon"], focused=True),
              win(2, "Vault", "keepass.exe", ["hunter2 the master", "bank pin 7731"], z=1))
    assert p.excluded(s0.windows[1])
    record_pair(hist, p, s0, s1)
    wait_flushed(hist, 3)
    hist.close()
    raw = (tmp_path / "h.db").read_bytes() + b"".join(f.read_bytes() for f in tmp_path.glob("h.db-*"))
    for secret in (b"zanzibar", b"shopping", b"Notes", b"notepad", b"hunter2", b"7731"):
        assert secret not in raw
    h2 = H.History(tmp_path / "h.db")
    try:
        hits = h2.search("zanzibar", since=now - 60)["hits"]
        assert [x["line"] for x in hits] == ["zanzibar meeting at noon"]
        everything = json.dumps(h2.search("", since=now - 60, limit=1000))
        assert "hunter2" not in everything and "7731" not in everything
    finally:
        h2.close()


def test_secrets_are_scrubbed_before_recording(hist):
    now = time.time()
    s0 = snap(now - 5, win(1, "Terminal", "wt.exe", ["$ ls"], focused=True))
    s1 = snap(now, win(1, "Terminal", "wt.exe", ["$ ls", "export KEY=sk-abcdefghijklmnopqrstuvwxyz0123"], focused=True))
    record_pair(hist, Policy(), s0, s1)
    wait_flushed(hist, 3)
    out = json.dumps(hist.search("", since=now - 60, limit=1000))
    assert "sk-abcdefghijklmnopqrstuvwxyz0123" not in out and "export KEY" in out


def test_search_terms_window_filter_and_time_range(hist):
    now = time.time()
    s0 = snap(now - 3600, win(1, "Mail", "outlook.exe", ["quarterly report draft"]),
              win(2, "Chat", "teams.exe", ["the quarterly numbers"], z=1))
    s1 = snap(now - 3500, win(1, "Mail", "outlook.exe", ["quarterly report draft", "report sent to finance"]),
              win(2, "Chat", "teams.exe", ["the quarterly numbers"], z=1))
    record_pair(hist, Policy(), s0, s1)
    wait_flushed(hist, 3)
    assert {x["line"] for x in hist.search("quarterly", since=now - 7200)["hits"]} == {
        "quarterly report draft", "the quarterly numbers"}
    assert [x["process"] for x in hist.search("quarterly", since=now - 7200, match="teams")["hits"]] == ["teams.exe"]
    assert [x["line"] for x in hist.search("report finance", since=now - 7200)["hits"]] == ["report sent to finance"]
    assert hist.search("quarterly", since=now - 60)["hits"] == []
    seen = hist.search("quarterly report draft", since=now - 7200)["hits"][0]
    assert seen["seen_count"] == 2                 # both snapshots held it: one hit with first and last seen


def test_retention_deletes_old_chunks(hist):
    now = time.time()
    old = snap(now - 8 * 86400, win(1, "Old", "a.exe", ["ancient line"]))
    hist.record([], old)
    hist.record([dict(kind="opened", hwnd=1, title="New", process="a.exe", seq=1, t=now)],
                snap(now, win(1, "New", "a.exe", ["fresh line"])))   # a change, so a new snapshot is due
    wait_flushed(hist, 2)                           # each flush prunes: the 8-day-old snapshot is already gone
    assert hist.stats["deleted"] >= 1 and hist.prune() == 0
    assert hist.search("ancient", since=0)["hits"] == []
    assert hist.search("fresh", since=0)["hits"]


def test_screen_at_returns_the_past_snapshot_and_following_events(hist):
    now = time.time()
    s0 = snap(now - 100, win(1, "Editor", "code.exe", ["def main():"], focused=True))
    s1 = snap(now - 50, win(1, "Editor", "code.exe", ["def main():", "    return 42"], focused=True),
              win(2, "Calc", "calc.exe", z=1))
    hist.snapshot_s = 10 ** 9                       # only the first snapshot is stored
    record_pair(hist, Policy(), s0, s1)
    wait_flushed(hist, 2)
    out = hist.at(now - 10)
    assert from_dict(out["snapshot"]).windows[0].title == "Editor"
    kinds = {e["kind"] for e in out["events"]}
    assert {"content", "opened"} <= kinds
    assert hist.at(now - 1000)["snapshot"] is None


def test_mcp_history_tools_and_status(tmp_path):
    now = time.time()
    h = H.History(tmp_path / "m.db", snapshot_s=0, flush_s=0.05)

    class FakeWatcher:
        max_windows = 6

        def start(self):
            return self

        def stop(self):
            pass

        def status(self):
            return dict(running=True)

    s = Server(source=lambda **k: snap(now), policy=Policy(), watcher=FakeWatcher(), history=h)
    try:
        record_pair(h, Policy(), snap(now - 20, win(1, "Docs", "word.exe", ["alpha"])),
                    snap(now - 10, win(1, "Docs", "word.exe", ["alpha", "beta gamma"])))
        wait_flushed(h, 3)
        out = json.loads(s.call("search_screen_history", {"query": "gamma", "since": "1h"}))
        assert out["note"] and [x["line"] for x in out["hits"]] == ["beta gamma"]
        md = s.call("screen_at", {"time": "5s"})
        assert "Docs" in md and "beta gamma" in md
        st = s.call("scribe_status", {})["history"]
        assert "DPAPI" in st["encrypted"] and st["retention_days"] == 7
    finally:
        s.close()


def test_parse_time_forms():
    assert H.parse_time("2h", now=10_000.0) == 10_000.0 - 7200
    assert H.parse_time("90m", now=10_000.0) == 10_000.0 - 5400
    assert H.parse_time(None) is None
    assert abs(H.parse_time("2026-10-19T12:00:00+00:00") - 1_792_411_200) < 1


def test_tampered_metadata_fails_authentication(hist):
    now = time.time()
    hist.record([dict(kind="opened", hwnd=1, title="A", process="a.exe", seq=1, t=now)], snap(now, win(1, "A", "a.exe")))
    wait_flushed(hist, 2)
    hist.db.execute("UPDATE chunks SET t0 = t0 - 1 WHERE kind = 'events'")   # move a chunk in time
    hist._cache.clear()
    with pytest.raises(ValueError, match="authentication"):
        hist.search("", since=now - 60)


def test_search_budget_reports_how_far_it_got(hist):
    now = time.time()
    for i in range(30):
        t = now - 3000 + i * 100
        hist._write("events", t, t, 1, [dict(kind="content", hwnd=1, title="T", process="t.exe",
                                            added=[f"line {i}"], seq=i, t=t)])
    full = hist.search("line", since=now - 4000, limit=100)
    assert full["complete"] and full["total"] == 30
    part = hist.search("line", since=now - 4000, limit=100, budget_s=0)
    assert not part["complete"] and "searched_back_to" in part


def test_purge_empties_and_replaces_the_key(hist):
    now = time.time()
    hist.record([dict(kind="opened", hwnd=1, title="A", process="a.exe", seq=1, t=now)], snap(now, win(1, "A", "a.exe")))
    wait_flushed(hist, 2)
    old_key = hist.db.execute("SELECT v FROM meta WHERE k = 'key'").fetchone()[0]
    assert hist.purge() == 2 and hist.status()["chunks"] == 0
    assert hist.db.execute("SELECT v FROM meta WHERE k = 'key'").fetchone()[0] != old_key
