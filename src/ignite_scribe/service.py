# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The background service: one ignite-scribe process per user, started at logon, independent of agent sessions.

It holds what should not depend on an agent being open: the history recorder (when history is on) and the tray
icon. MCP servers started by agents then only read history: its lock is already taken. Always-on OCR is not run
here: its text joins the live state of the process that runs it, so it stays with the first agent's server, where
the tools read it. It runs in the user's own session, because screen
access needs the interactive desktop, so it is not a Windows service but a per-user logon entry
(``HKCU\\...\\Run``, no administrator rights), started with ``pythonw`` (no console window).

``ignite-scribe service install | uninstall | start | stop | restart | status``; ``ignite-scribe daemon`` is what
the logon entry runs. Stopping writes a stop request, so the process flushes its history before it exits. Its
log (``daemon.log``, errors and lifecycle only, never screen content) is cut back to its last 256 KB at each start.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .always_on import SCRIBE_DIR, pid_alive, take_lock

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN_NAME = "ignite-scribe"
LOCK = SCRIBE_DIR / "daemon.lock"
PID_FILE = SCRIBE_DIR / "daemon.pid"
STOP_FILE = SCRIBE_DIR / "daemon.stop"
LOG = SCRIBE_DIR / "daemon.log"


def pythonw() -> Path:
    """The windowless interpreter of this installation (``pythonw.exe`` beside ``python.exe``)."""
    exe = Path(sys.executable)
    w = exe.with_name("pythonw.exe")
    return w if w.is_file() else exe


def command() -> str:
    return f'"{pythonw()}" -m ignite_scribe daemon'


def installed() -> str | None:
    """The logon entry's command, or None."""
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
            return winreg.QueryValueEx(k, RUN_NAME)[0]
    except OSError:
        return None


def running_pid() -> int | None:
    try:
        pid = int(PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None
    if not pid_alive(pid):
        return None
    probe = take_lock(LOCK)                       # the daemon holds this lock while it runs
    if probe is not None:
        probe.close()
        return None
    return pid


def install(start_now: bool = True) -> str:
    import winreg
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
        winreg.SetValueEx(k, RUN_NAME, 0, winreg.REG_SZ, command())
    if start_now:
        start()
    return command()


def uninstall():
    import winreg
    stop()
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
            winreg.DeleteValue(k, RUN_NAME)
    except OSError:
        pass


def start(wait_s: float = 10.0) -> int | None:
    pid = running_pid()
    if pid:
        return pid
    STOP_FILE.unlink(missing_ok=True)
    flags = 0x00000008 | 0x00000200 | 0x08000000    # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
    subprocess.Popen([str(pythonw()), "-m", "ignite_scribe", "daemon"], creationflags=flags, close_fds=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait_s:
        pid = running_pid()
        if pid:
            return pid
        time.sleep(0.2)
    return None


def stop(wait_s: float = 15.0) -> bool:
    """Ask the daemon to exit (it flushes history first). -> True when none runs afterwards."""
    if running_pid() is None:
        return True
    SCRIBE_DIR.mkdir(parents=True, exist_ok=True)
    STOP_FILE.write_bytes(b"")
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait_s:
        if running_pid() is None:
            STOP_FILE.unlink(missing_ok=True)
            return True
        time.sleep(0.2)
    return False


def status() -> dict:
    from .tray import read_beacons
    pid = running_pid()
    mine = [b for b in read_beacons() if pid and b.get("pid") == pid]
    return dict(installed=installed() is not None, command=installed(), running=pid is not None, pid=pid,
                state=mine[0] if mine else None, log=str(LOG))


def _log(msg: str):
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    except OSError:
        pass


def daemon() -> int:
    """Run until a stop request: the history recorder and the tray, as the policy says."""
    SCRIBE_DIR.mkdir(parents=True, exist_ok=True)
    lock = take_lock(LOCK)
    if lock is None:
        return 0                                  # already running
    try:
        if LOG.is_file() and LOG.stat().st_size > 256 * 1024:
            LOG.write_bytes(LOG.read_bytes()[-256 * 1024:])
        log = open(LOG, "a", encoding="utf-8", buffering=1)
        sys.stderr = log                              # pythonw has no stderr; errors go to the log
        if sys.stdout is None:
            sys.stdout = log
        PID_FILE.write_text(str(os.getpid()))
        STOP_FILE.unlink(missing_ok=True)
        from . import tray
        from .mcp_server import Server
        server = Server()
        try:
            h = server._history() if server.history_on else None
            if h is not None and h.recorder:
                server._watcher()
            if tray.enabled():
                server.beacon = tray.Beacon(server, service=True).start()   # agents' servers yield the tray to it
                server.tray = tray.Tray(takeover_s=1.0).start()
            _log(json.dumps(dict(event="started", pid=os.getpid(), history=bool(h and h.recorder),
                                 tray=tray.enabled())))
            while not STOP_FILE.exists():
                time.sleep(1.0)
            _log("stop requested")
        finally:
            server.close()
            PID_FILE.unlink(missing_ok=True)
            _log("stopped")
        return 0
    finally:
        lock.close()
