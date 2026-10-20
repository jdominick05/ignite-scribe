# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The tray indicator: beacons carry no screen content, stale ones are dropped, the summary is right, one icon
per machine, and the real Win32 icon comes up and goes away."""
import json
import os
import time
import types

from ignite_scribe import tray as T


def fake_server(watching=True, vision=False, always=None, recording=False):
    a = None if always is None else types.SimpleNamespace(status=lambda: always)
    return types.SimpleNamespace(watcher=types.SimpleNamespace(running=watching),
                                 vision=types.SimpleNamespace(running=vision), always=a,
                                 history=types.SimpleNamespace(recorder=recording) if recording else None)


def test_beacon_holds_flags_only_and_is_removed_on_stop(tmp_path):
    s = fake_server(always=dict(running=True, paused=None, npu_held=True, reads=7), recording=True)
    b = T.Beacon(s, every_s=0.1, directory=tmp_path).start()
    time.sleep(0.3)
    d = json.loads(b.path.read_bytes())
    assert set(d) == {"pid", "t", "watching", "vision_running", "always_on", "recording"}
    assert d["always_on"] == dict(running=True, paused=None, reading=True, reads=7) and d["recording"] is True
    b.stop()
    assert not b.path.exists()


def test_stale_and_dead_beacons_are_dropped(tmp_path):
    now = time.time()
    (tmp_path / "1.json").write_text(json.dumps(dict(pid=1, t=now, watching=True)))
    (tmp_path / "2.json").write_text(json.dumps(dict(pid=2, t=now - 60, watching=True)))
    (tmp_path / "3.json").write_text(json.dumps(dict(pid=3, t=now, watching=True)))
    live = T.read_beacons(tmp_path, alive=lambda pid: pid != 3)
    assert [b["pid"] for b in live] == [1] and not (tmp_path / "3.json").exists()


def test_summary_states():
    w = dict(watching=True)
    r = dict(watching=True, always_on=dict(running=True, reading=True))
    assert T.summarize([], False)[0] == "idle"
    assert T.summarize([w], False) == ("watching", "ignite-scribe: 1 session, watching windows (1)")
    st, tip = T.summarize([w, r, dict(recording=True)], False)
    assert st == "reading" and "reading pixels" in tip and "recording history" in tip and "3 sessions" in tip
    st, tip = T.summarize([r], True)
    assert st == "paused" and "always-on OCR paused" in tip and "reading pixels" not in tip
    assert T.summarize([dict(vision_running=True)], False)[0] == "reading"
    assert len(T.summarize([w] * 50 + [r, dict(recording=True)], False)[1]) <= 127


def test_one_icon_per_machine_and_the_icon_comes_and_goes(tmp_path):
    lock, status = tmp_path / "tray.lock", tmp_path / "status"
    status.mkdir()
    (status / f"{os.getpid()}.json").write_text(json.dumps(dict(pid=os.getpid(), t=time.time() + 30,
                                                                 watching=True)))
    a = T.Tray(takeover_s=0.2, refresh_ms=200, status_dir=status, lock=lock, pause_file=tmp_path / "p").start()
    b = T.Tray(takeover_s=0.2, refresh_ms=200, status_dir=status, lock=lock, pause_file=tmp_path / "p").start()
    t0 = time.time()
    while not (a.shown or b.shown) and time.time() - t0 < 5:
        time.sleep(0.05)
    try:
        assert (a.shown, b.shown).count(True) == 1 and (a.owner, b.owner).count(True) == 1
        first, second = (a, b) if a.owner else (b, a)
        assert first.state == "watching" and "watching windows" in first.tip
        first.stop()
        t0 = time.time()
        while not second.shown and time.time() - t0 < 5:        # the other one takes over
            time.sleep(0.05)
        assert second.shown and second.owner
    finally:
        a.stop()
        b.stop()
    assert not a.shown and not b.shown
