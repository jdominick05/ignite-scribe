# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The configurator and the background service, on temporary paths. The real registry and policy are untouched."""
import json
import threading
import time

import pytest

from ignite_scribe import configure as C
from ignite_scribe import redact
from ignite_scribe import service as S


def test_policy_path_prefers_the_environment_then_the_user_file(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delenv("IGNITE_SCRIBE_POLICY", raising=False)
    assert redact.policy_path() is None                       # no user file yet
    f = redact.default_policy_path()
    f.parent.mkdir(parents=True)
    f.write_text('{"history": true}')
    assert redact.policy_path() == str(f) and redact.Policy.load(redact.policy_path()).history
    monkeypatch.setenv("IGNITE_SCRIBE_POLICY", "")
    assert redact.policy_path() is None                       # empty: no policy file at all


def test_merge_mcp_config_keeps_other_servers_and_backs_up(tmp_path):
    cfg = tmp_path / "agent.json"
    cfg.write_text(json.dumps({"theme": "dark", "mcpServers": {"other": {"command": "x"}}}))
    backup = C.merge_mcp_config(cfg, {"command": "scribe.exe", "args": ["mcp"]})
    d = json.loads(cfg.read_text())
    assert d["theme"] == "dark" and d["mcpServers"]["other"] == {"command": "x"}
    assert d["mcpServers"]["ignite-scribe"] == {"command": "scribe.exe", "args": ["mcp"]}
    assert json.loads(backup.read_text())["mcpServers"] == {"other": {"command": "x"}}
    new = tmp_path / "sub" / "new.json"
    assert C.merge_mcp_config(new, {"command": "y"}) is None and json.loads(new.read_text())["mcpServers"]


@pytest.fixture
def no_registry(monkeypatch):
    calls = []
    monkeypatch.setattr(S, "installed", lambda: None)
    monkeypatch.setattr(S, "install", lambda start_now=True: calls.append("install"))
    monkeypatch.setattr(S, "uninstall", lambda: calls.append("uninstall"))
    monkeypatch.setattr(S, "running_pid", lambda: None)
    monkeypatch.setattr(S, "status", lambda: dict(running=False, pid=None))
    return calls


def test_setup_writes_the_policy_keeping_other_settings(tmp_path, no_registry, capsys):
    pol = tmp_path / "policy.json"
    pol.write_text(json.dumps({"exclude_processes": ["secret.exe"], "history": False}))
    assert C.run(history=True, always_on=False, service=False, yes=True, policy_file=pol) == 0
    d = json.loads(pol.read_text())
    assert d == {"exclude_processes": ["secret.exe"], "history": True, "always_on_ocr": False, "secure_mode": False}
    out = capsys.readouterr().out
    assert '"mcpServers"' in out and '"ignite-scribe"' in out and no_registry == []
    C.run(yes=True, policy_file=pol)                    # --yes keeps the answers; history on implies the service
    assert json.loads(pol.read_text())["history"] is True
    C.run(history=False, always_on=False, service=True, yes=True, policy_file=pol)
    assert no_registry == ["install", "install"]


def test_daemon_runs_until_asked_to_stop(tmp_path, monkeypatch):
    for name, value in (("LOCK", "d.lock"), ("PID_FILE", "d.pid"), ("STOP_FILE", "d.stop"), ("LOG", "d.log")):
        monkeypatch.setattr(S, name, tmp_path / value)
    monkeypatch.setattr(S, "SCRIBE_DIR", tmp_path)
    for k, v in (("IGNITE_SCRIBE_TRAY", "0"), ("IGNITE_SCRIBE_HISTORY", "0"), ("IGNITE_SCRIBE_ALWAYS_ON", "0"),
                 ("IGNITE_SCRIBE_WATCH", "0")):
        monkeypatch.setenv(k, v)
    import sys
    err = sys.stderr
    t = threading.Thread(target=S.daemon, daemon=True)
    t.start()
    try:
        t0 = time.time()
        while S.running_pid() is None and time.time() - t0 < 10:
            time.sleep(0.05)
        assert S.running_pid() is not None
        assert S.daemon() == 0                                    # a second one exits at once
        assert S.stop(wait_s=10) is True
        t.join(5)
        assert not t.is_alive() and not (tmp_path / "d.pid").exists()
        log = (tmp_path / "d.log").read_text()
        assert '"event": "started"' in log and "stopped" in log
    finally:
        sys.stderr = err
