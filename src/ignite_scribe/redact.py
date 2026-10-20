# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What never leaves the engine: excluded apps, and secret-shaped text.

Redaction runs on every snapshot before any consumer sees it (MCP, hook, history). Two layers:
- **Excluded processes / titles.** A matching window keeps its position (so an agent knows something is there) but
  its title and content are replaced. Defaults cover common password managers; extend them in a policy file.
- **Secret-shaped text.** API keys and tokens, private-key blocks, JWTs, and card numbers that pass the Luhn check
  are replaced in any text that survives the first layer.
UI Automation's ``IsPassword`` flag is honoured when that source exists (password fields are never read).

The policy file is JSON: ``{"exclude_processes": [...], "exclude_title_patterns": [...], "capture_elevated":
false, "history": false}``, merged with the defaults. Patterns are case-insensitive regular expressions.
``capture_elevated`` lets the vision stage copy the pixels of elevated windows, which UI Automation cannot read
(off by default). ``history`` records the redacted screen to disk, encrypted, for 7 days (off by default;
``IGNITE_SCRIBE_HISTORY`` overrides it either way).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import injection
from .snapshot import Snapshot, TextBlock, Window

DEFAULT_EXCLUDE_PROCESSES = ["keepass.exe", "keepassxc.exe", "1password.exe", "bitwarden.exe", "dashlane.exe",
                             "enpass.exe", "lastpass.exe", "credentialuibroker.exe", "consent.exe"]
DEFAULT_EXCLUDE_TITLES = [r"\bpassword\b", r"\bsign[ -]?in\b", r"\bbank(ing)?\b", r"\bprivate browsing\b",
                          r"\binprivate\b", r"\bincognito\b"]
WITHHELD = "[withheld by ignite-scribe policy]"

SECRET_PATTERNS = [
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("api key", re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}\b")),
    ("github token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}\b")),
    ("aws key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("hf token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
]
CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def scrub(text: str) -> tuple[str, int]:
    """-> (text with secret-shaped spans replaced, number replaced)."""
    n = 0
    for name, rx in SECRET_PATTERNS:
        text, k = rx.subn(f"[{name} withheld]", text)
        n += k

    def card(m):
        nonlocal n
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and luhn(digits):
            n += 1
            return "[card number withheld]"
        return m.group(0)
    return CARD.sub(card, text), n


def default_policy_path() -> Path:
    """Where ``ignite-scribe setup`` writes the user's policy."""
    return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "ignite-scribe" / "policy.json"


def policy_path() -> str | None:
    """The policy file in force: ``IGNITE_SCRIBE_POLICY`` when set (empty: none), else the user's
    ``policy.json`` when it exists."""
    v = os.environ.get("IGNITE_SCRIBE_POLICY")
    if v is not None:
        return v or None
    p = default_policy_path()
    return str(p) if p.is_file() else None


@dataclass
class Policy:
    exclude_processes: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE_PROCESSES))
    exclude_title_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE_TITLES))
    capture_elevated: bool = False   # may vision copy the pixels of elevated windows (UIA cannot read them)?
    history: bool = False            # record the redacted events and snapshots to disk (history.py)?
    always_on_ocr: bool = False      # read changed pixels of windows UIA cannot describe, on the NPU (always_on.py)?
    secure_mode: bool = False        # also score window text with the injection classifier, on the iGPU (secure.py)?
    scanner: object = field(default=None, repr=False, compare=False)   # secure.Scanner, set by the server

    @staticmethod
    def load(path: str | Path | None) -> Policy:
        p = Policy()
        if path and Path(path).is_file():
            d = json.loads(Path(path).read_text(encoding="utf-8"))
            p.exclude_processes += [s.lower() for s in d.get("exclude_processes", [])]
            p.exclude_title_patterns += d.get("exclude_title_patterns", [])
            p.capture_elevated = bool(d.get("capture_elevated", False))
            p.history = bool(d.get("history", False))
            p.always_on_ocr = bool(d.get("always_on_ocr", False))
            p.secure_mode = bool(d.get("secure_mode", False))
        return p

    def excluded(self, w: Window) -> bool:
        if w.process.lower() in {s.lower() for s in self.exclude_processes}:
            return True
        return any(re.search(rx, w.title, re.I) for rx in self.exclude_title_patterns)

    def apply(self, snap: Snapshot) -> tuple[Snapshot, dict]:
        """-> (redacted copy, counts). The input is not modified."""
        windows, withheld, secrets = [], 0, 0
        for w in snap.windows:
            if self.excluded(w):
                windows.append(replace(w, title=WITHHELD, text=[], elements=[], flags=[], redacted=True))
                withheld += 1
                continue
            title, k = scrub(w.title)
            secrets += k
            blocks = []
            for b in w.text:
                t, k = scrub(b.text)
                secrets += k
                d = b.detail
                if d is not None:                      # every string in it came from the screen too
                    d = dict(d)
                    for key, v in d.items():
                        if isinstance(v, str):
                            d[key], k = scrub(v)
                            secrets += k
                blocks.append(replace(b, text=t, detail=d))
            clean = replace(w, title=title, text=blocks)
            clean.flags = injection.flag_window(clean)     # on scrubbed text only: an excerpt holds no secret
            if self.scanner is not None:                   # secure mode: a score arrives on a later refresh
                f = self.scanner.flag(clean, injection.category(clean.process))
                if f is not None:
                    clean.flags = clean.flags + [f]
            windows.append(clean)
        return replace(snap, windows=windows), dict(windows_withheld=withheld, secrets_withheld=secrets,
                                                    windows_flagged=sum(1 for w in windows if w.flags))
