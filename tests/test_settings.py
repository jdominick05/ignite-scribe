# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Settings from the tray: the policy file keeps its other keys, and a running server applies a change at once,
above all turning history, always-on OCR and secure mode off, so no process goes on recording after "off"."""
import json
import time
import types

from ignite_scribe import settings as S
from ignite_scribe import tray as T
from ignite_scribe.redact import DEFAULT_EXCLUDE_PROCESSES, Policy


def test_changes_keep_the_other_keys(tmp_path):
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(dict(history=True, exclude_title_patterns=["secret"], other=1)))
    S.set_flag("always_on_ocr", True, p)
    S.add_exclusion("Signal.exe", p)
    S.add_exclusion("signal.exe", p)                                   # once, lower case
    S.add_exclusion(DEFAULT_EXCLUDE_PROCESSES[0], p)                   # built in: not written
    d = json.loads(p.read_text())
    assert d == dict(history=True, exclude_title_patterns=["secret"], other=1, always_on_ocr=True,
                     exclude_processes=["signal.exe"])
    S.remove_exclusion("SIGNAL.EXE", p)
    assert S.user_exclusions(p) == [] and json.loads(p.read_text())["other"] == 1
    try:
        S.set_flag("anything", True, p)
        raise AssertionError("unknown keys are refused")
    except ValueError:
        pass


class Closable:
    def __init__(self, **kw):
        self.closed = False
        self.__dict__.update(kw)

    def close(self):
        self.closed = True

    stop = close


def fake_server(policy, history=None, always=None, scanner=None):
    w = types.SimpleNamespace(recorder=history, reapplied=0)
    w.reapply = lambda: setattr(w, "reapplied", w.reapplied + 1)
    policy.scanner = scanner
    s = types.SimpleNamespace(policy=policy, history=history, history_on=history is not None, watcher=w,
                              indexer=Closable(), always=always, _always_lock=Closable() if always else None,
                              always_on_wanted=always is not None, secure_wanted=scanner is not None,
                              _secure_worker=Closable() if scanner else None, embedder=None, secure_error=None)
    s.start_always_on = lambda: setattr(s, "started_always", True)
    return s


def test_off_is_applied_at_once(monkeypatch):
    for v in ("IGNITE_SCRIBE_HISTORY", "IGNITE_SCRIBE_ALWAYS_ON", "IGNITE_SCRIBE_SECURE"):
        monkeypatch.delenv(v, raising=False)
    pol = Policy(history=True, always_on_ocr=True, secure_mode=True)
    h, a, sc = Closable(recorder=True), Closable(), Closable()
    s = fake_server(pol, history=h, always=a, scanner=sc)
    lock, worker, procs = s._always_lock, s._secure_worker, pol.exclude_processes
    new = Policy()
    new.exclude_processes.append("signal.exe")
    S.apply_live(s, new)
    assert h.closed and s.history is None and not s.history_on and s.watcher.recorder is None   # no recording
    assert a.closed and s.always is None and lock.closed and s._always_lock is None
    assert sc.closed and pol.scanner is None and worker.closed
    assert pol.exclude_processes is procs and "signal.exe" in procs        # in place: the watcher sees it
    assert not pol.history and not pol.always_on_ocr and not pol.secure_mode
    assert s.watcher.reapplied == 1


def test_always_on_is_not_started_in_the_service(monkeypatch):
    monkeypatch.delenv("IGNITE_SCRIBE_ALWAYS_ON", raising=False)
    s = fake_server(Policy())
    S.apply_live(s, Policy(always_on_ocr=True), is_service=True)
    assert not hasattr(s, "started_always")
    S.apply_live(s, Policy(always_on_ocr=True))
    assert s.started_always


def test_the_beacon_applies_a_changed_file(tmp_path, monkeypatch):
    monkeypatch.delenv("IGNITE_SCRIBE_HISTORY", raising=False)
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(dict(history=True)))
    h = Closable(recorder=True)
    s = fake_server(Policy(history=True), history=h)
    s.vision, s.tray = None, None
    b = T.Beacon(s, directory=tmp_path / "status", policy_file=p)
    b.check_policy()
    assert not h.closed                                                # unchanged: nothing to do
    p.write_text(json.dumps(dict(history=False)))
    t = time.time() + 5
    import os
    os.utime(p, (t, t))
    b.check_policy()
    assert h.closed and s.history is None and b.applied == p.stat().st_mtime
    b.write()
    assert json.loads(b.path.read_bytes())["policy"] == p.stat().st_mtime


def test_processes_that_cannot_reload_are_named():
    m = time.time() - 60
    d = T.describe([dict(pid=1, policy=m), dict(pid=2)], False, m)     # pid 2: older code, no "policy"
    assert "pid 2" in d["stale"] and "pid 1" not in d["stale"]
    assert T.describe([dict(pid=1, policy=m)], False, m)["stale"] == ""
