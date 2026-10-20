# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``ignite-scribe setup``: choose what ignite-scribe may do, start the background service, connect an agent.

Asks (or takes flags for) three things, each off unless chosen:
- **history**: record the redacted screen state, encrypted, kept 7 days, searchable by agents;
- **always-on OCR**: read the text of windows UI Automation cannot describe, continuously, on the NPU only;
- **the background service**: a per-user process started at logon that records history and shows the tray
  icon without an agent open (always-on OCR runs in the first agent's server, whose tools read its text).
It writes them to the user's policy file (``%LOCALAPPDATA%\\ignite-scribe\\policy.json``), keeping any other
settings in it, restarts the service when it runs, and prints the MCP server entry for an agent's config.
``--mcp-config FILE`` merges that entry into a JSON config file with an ``mcpServers`` object, after a backup.
Run it again at any time to change the answers.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from .redact import default_policy_path


def server_entry() -> dict:
    """The MCP server entry for this installation: the console script when present, else the interpreter."""
    exe = Path(sys.executable)
    script = exe.with_name("ignite-scribe.exe")
    if script.is_file():
        return {"command": str(script), "args": ["mcp"]}
    return {"command": str(exe), "args": ["-m", "ignite_scribe", "mcp"]}


def merge_mcp_config(path: Path, entry: dict, name: str = "ignite-scribe") -> Path | None:
    """Add or replace ``mcpServers[name]`` in a JSON config file. -> the backup's path (None for a new file)."""
    backup = None
    data = {}
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        data = json.loads(text) if text.strip() else {}
        backup = path.with_name(path.name + ".bak")
        shutil.copy2(path, backup)
    data.setdefault("mcpServers", {})[name] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return backup


def read_policy(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except ValueError:
        return {}


def write_policy(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _ask(question: str, default: bool) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        a = input(f"{question} [{hint}] ").strip().lower()
        if not a:
            return default
        if a in ("y", "yes", "n", "no"):
            return a.startswith("y")


def capabilities() -> dict:
    """What this machine can do, from files and imports only (nothing is started)."""
    from . import uia, vision_client
    from .always_on import npu_recognizer_ready
    ok_vision, why_vision = vision_client.available()
    ok_ocr, why_ocr = vision_client.ocr_available()
    ready, why_ready = npu_recognizer_ready()
    try:
        from . import semantic
        ok_sem, why_sem = semantic.available()
    except Exception as e:  # noqa: BLE001
        ok_sem, why_sem = False, str(e)
    return dict(ui_automation=uia.available(), vision=ok_vision, vision_why=why_vision or None,
                ocr=ok_ocr, ocr_why=why_ocr or None, always_on_ready=ready, always_on_why=why_ready or None,
                semantic_history=ok_sem, semantic_why=why_sem or None)


def run(history: bool | None = None, always_on: bool | None = None, service: bool | None = None,
        mcp_config: str | None = None, yes: bool = False, policy_file: Path | None = None) -> int:
    from . import service as svc
    path = policy_file or default_policy_path()
    pol = read_policy(path)
    interactive = not yes and sys.stdin is not None and sys.stdin.isatty()
    cap = capabilities()
    print("ignite-scribe setup\n")
    print(f"  UI Automation text   {'yes' if cap['ui_automation'] else 'no (pip install ignite-scribe[uia])'}")
    print(f"  NPU vision and OCR   {'yes' if cap['vision'] else 'not set up (optional, see the README)'}")
    print(f"  always-on OCR ready  {'yes' if cap['always_on_ready'] else 'no: ' + (cap['always_on_why'] or '')}")
    print(f"  semantic history     {'yes' if cap['semantic_history'] else 'keyword search only'}\n")
    if history is None:
        history = (_ask("Record an encrypted, redacted screen history (7 days) that agents can search?",
                        bool(pol.get("history", False))) if interactive else bool(pol.get("history", False)))
    if always_on is None:
        cur = bool(pol.get("always_on_ocr", False))
        always_on = (_ask("Read text in windows UI Automation cannot describe, continuously, on the NPU?", cur)
                     if interactive and cap["always_on_ready"] else cur)
    if service is None:
        want = history or svc.installed() is not None
        service = _ask("Run ignite-scribe in the background at logon (tray icon; records history without an agent)?",
                       want) if interactive else want
    pol["history"], pol["always_on_ocr"] = bool(history), bool(always_on)
    write_policy(path, pol)
    print(f"\n  policy      {path}")
    print(f"  history     {'on' if history else 'off'}")
    print(f"  always-on   {'on' if always_on else 'off'}" + ("" if cap["always_on_ready"] or not always_on
                                                             else " (waits until the NPU recognizer is installed)"))
    if service:
        if svc.running_pid():
            svc.stop()
        svc.install(start_now=True)
        st = svc.status()
        print(f"  service     {'running, pid ' + str(st['pid']) if st['running'] else 'installed (starts at logon)'}")
    else:
        if svc.installed() is not None:
            svc.uninstall()
        print("  service     off")
    entry = server_entry()
    if mcp_config:
        backup = merge_mcp_config(Path(mcp_config).expanduser(), entry)
        print(f"  agent       added to {mcp_config}" + (f" (backup: {backup})" if backup else ""))
    print("\nAdd this to your agent's MCP configuration (most clients use an \"mcpServers\" object):\n")
    print(json.dumps({"mcpServers": {"ignite-scribe": entry}}, indent=2))
    print("\nThen restart the agent and ask it what is on your screen.")
    return 0
