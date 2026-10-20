# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Settings changed while ignite-scribe runs: the tray menu writes them, and every running process applies them.

The policy file is the one place settings live (``policy.json``, as ``ignite-scribe setup`` writes it). The tray
changes it with ``set_flag`` / ``add_exclusion`` / ``remove_exclusion``, keeping every other key. Each process
checks the file's modification time every two seconds (from its beacon thread) and, when it changed, calls
``apply_live``, which brings the running parts in line at once:
- exclusions and ``capture_elevated`` are updated in place on the shared ``Policy`` (the watcher, always-on OCR and
  the vision tools read it on every pass), and the watcher re-applies it to what it shows;
- history off closes the store, which flushes it and releases the recorder lock, so no process records or takes
  recording over; on opens it (the process that gets the lock records);
- always-on OCR off stops it and releases its lock; secure mode off stops the scanner.
Turning a part on again is otherwise lazy, as at start. ``IGNITE_SCRIBE_*`` variables still override the file.
A process too old to reload says so in its beacon (no ``policy`` key), and the tray lists it.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .configure import read_policy, write_policy
from .redact import DEFAULT_EXCLUDE_PROCESSES, Policy, default_policy_path

FLAGS = ("history", "always_on_ocr", "secure_mode", "capture_elevated")


def path() -> Path | None:
    """The policy file to edit: ``IGNITE_SCRIBE_POLICY`` when set (empty: none, so nothing is editable), else the
    user's ``policy.json`` (created on the first change)."""
    v = os.environ.get("IGNITE_SCRIBE_POLICY")
    if v is not None:
        return Path(v) if v else None
    return default_policy_path()


def mtime(p: Path | None) -> float | None:
    try:
        return p.stat().st_mtime if p is not None else None
    except OSError:
        return None


def current(p: Path | None = None) -> dict:
    p = p or path()
    return read_policy(p) if p is not None else {}


def set_flag(key: str, on: bool, p: Path | None = None):
    if key not in FLAGS:
        raise ValueError(key)
    p = p or path()
    d = read_policy(p)
    d[key] = bool(on)
    write_policy(p, d)


def user_exclusions(p: Path | None = None) -> list[str]:
    """The processes the user excluded (the defaults are always excluded and are not in the file)."""
    return [s.lower() for s in current(p).get("exclude_processes", [])]


def add_exclusion(process: str, p: Path | None = None):
    p = p or path()
    d = read_policy(p)
    have = [s.lower() for s in d.get("exclude_processes", [])]
    if process.lower() not in have and process.lower() not in DEFAULT_EXCLUDE_PROCESSES:
        d["exclude_processes"] = have + [process.lower()]
        write_policy(p, d)


def remove_exclusion(process: str, p: Path | None = None):
    p = p or path()
    d = read_policy(p)
    d["exclude_processes"] = [s for s in d.get("exclude_processes", []) if s.lower() != process.lower()]
    write_policy(p, d)


def running_apps() -> list[str]:
    """Process names of the open windows the taskbar would show (Win32 only: no window content is read)."""
    from .windows import windows
    return sorted({w.process.lower() for w in windows() if w.process})


def open_file(p: Path | None = None):
    """Open the policy file in the user's editor, creating it when missing (for title patterns and the rest)."""
    p = p or path()
    if not p.is_file():
        write_policy(p, {})
    os.startfile(str(p))  # noqa: S606 - the user's own settings file, opened at their request


def service(action: str):
    """``install`` / ``uninstall`` / ``restart`` the background service from a separate process. Never from the
    tray's own thread: the tray may run inside the service, whose stop waits for this process to exit."""
    if action not in ("install", "uninstall", "restart"):
        raise ValueError(action)
    exe = Path(sys.executable)
    w = exe.with_name("pythonw.exe")
    flags = 0x00000008 | 0x00000200 | 0x08000000    # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
    subprocess.Popen([str(w if w.is_file() else exe), "-m", "ignite_scribe", "service", action], creationflags=flags,
                     close_fds=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def apply_live(server, new: Policy, is_service: bool = False):
    """Bring a running server in line with ``new`` (see the module docstring). Turning things off is immediate."""
    from . import always_on, history, secure
    p = server.policy
    p.exclude_processes[:] = new.exclude_processes
    p.exclude_title_patterns[:] = new.exclude_title_patterns
    p.capture_elevated = new.capture_elevated
    p.history, p.always_on_ocr, p.secure_mode = new.history, new.always_on_ocr, new.secure_mode

    want = history.enabled(p)
    if not want and server.history_on:
        server.history_on = False
        if server.watcher is not None:
            server.watcher.recorder = None
        if server.indexer is not None:
            server.indexer.stop()
            server.indexer = None
        h, server.history = server.history, None
        if h is not None:
            h.close()                                      # flushes, then releases the recorder lock
    elif want and not server.history_on:
        server.history_on = True
        h = server._history()
        if h is not None and h.recorder:
            if server.watcher is not None:
                server.watcher.recorder = h
            server._on_recorder()                          # the recorder watches from now on, and indexes

    server.always_on_wanted = always_on.enabled(p)
    if not server.always_on_wanted:
        if server.always is not None:
            server.always.stop()
            server.always = None
        if server._always_lock is not None:
            server._always_lock.close()
            server._always_lock = None
    elif not is_service:                                   # always-on stays with an agent's server (service.py)
        server.start_always_on()

    server.secure_wanted = secure.enabled(p)
    if not server.secure_wanted and p.scanner is not None:
        p.scanner.stop()
        p.scanner = None
        if server._secure_worker is not None and server._secure_worker is not server.embedder:
            server._secure_worker.close()
        server._secure_worker = None
        server.secure_error = None
    elif server.secure_wanted and p.scanner is None:
        server.secure_error = None                         # a cached "not installed" must not block a new try
        if server.watcher is not None:
            server._start_secure()                         # else it starts with the watcher, as at start

    if server.watcher is not None and hasattr(server.watcher, "reapply"):
        server.watcher.reapply()                           # exclusions and flags apply to what it shows now
