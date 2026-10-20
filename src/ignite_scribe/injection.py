# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Flag screen text that looks written to instruct an AI agent (prompt injection), at capture time, locally.

A flag is metadata, never a filter: the text stays as redaction left it, and the window gains ``flags``. A flag
says "this window's text seems to address an agent". No flag says nothing about safety: every reply keeps the
UNTRUSTED note. Runs on scrubbed text only (inside ``Policy.apply``, after ``scrub``, and on change events'
added lines), so a flag's excerpt can never carry a secret, and withheld windows are never scanned.

Text is normalized before matching: NFKC (folds full-width and compatibility forms), casefold, invisible
characters dropped, whitespace collapsed. So inserted zero-width characters and look-alike letters do not hide a
phrase. Invisible characters are a finding of their own (``hidden``): Unicode tag characters, bidi overrides and
isolates, and runs of zero-width characters. A reader on screen cannot see them, but a model reads them.

Kinds: ``hidden``, ``override`` (ignore / disregard previous instructions, also German, French, Spanish),
``extract`` (reveal your instructions / password), ``role`` (you are now ..., new
instructions, system prompt), ``address`` (text addressed to an AI or agent), ``markup`` (chat-template tokens),
``secrecy`` (keep this from the user, the user approved ...), ``exfil`` (send this to a URL, pipe a download into
a shell). Patterns use bounded quantifiers only, and at most ``MAX_SCAN`` characters per block are scanned, since
the text is attacker-controlled and this runs on every watcher refresh.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import OrderedDict

MAX_SCAN = 32_000          # characters scanned per text block
MAX_FLAGS = 8              # findings kept per window
EXCERPT = 100              # characters of normalized text around a match

_TAGS = re.compile("[\U000e0000-\U000e007f]")
_BIDI = re.compile("[‪-‮⁦-⁩]")
_ZW = re.compile("[​‌⁠᠎]|(?<!^)﻿")
_ZW_RUN = 3                                     # zero-width characters in a block before it counts as hidden
_INVISIBLE = re.compile("[\U000e0000-\U000e007f‪-‮⁦-⁩​-‏⁠-⁤﻿᠎­]")
_WS = re.compile(r"\s+")

_AGENT = r"(?:ai|a\.i\.|assistant|agent|chatbot|bot|llm|language model|model)s?"
_PREV = r"(?:all|any|the|your|my|previous|prior|above|earlier|preceding|former|original|system|initial)"
_RULES = r"(?:instructions?|prompts?|rules|directions?|guidelines|guidance|context|commands?|messages?|orders)"
PATTERNS = {
    "override": [
        rf"\b(?:ignore|disregard|forget|override|bypass|skip|drop)\b(?: [a-z]{{1,12}}){{0,3}} {_PREV}\b(?: [a-z]{{1,12}}){{0,2}} {_RULES}\b",
        rf"\b(?:do not|don't|never) (?:follow|obey) (?:the |your |any )?{_PREV}? ?{_RULES}\b",
        r"\binstead,? (?:you (?:must|should|will)|do the following)\b",
        r"\b(?:ignore|disregard|forget)\b(?: \S{1,15}){0,4}? (?:instruct\w*|istruct\w*|instruc\w*|prompts?|rules)\b",
        r"\bforget (?:about )?(?:everything|all of (?:this|that|it)|all that)\b",
        r"\b(?:leave|put) (?:all |any )?(?:previous|prior|earlier) (?:information|instructions?|context) behind\b",
        # German, French, Spanish
        r"\b(?:vergiss|vergessen sie) (?:alles|alle|die|deine|ihre)\b",
        r"\bignorier(?:e|en sie)? (?:\S{1,15} ){0,3}(?:anweisungen|instruktionen|befehle|vorgaben|regeln)\b",
        r"\boublie[zr]? (?:tout|toutes? les|les instructions)\b",
        r"\bignore[zr]? (?:\S{1,15} ){0,3}(?:instructions|consignes|règles)\b",
        r"\b(?:olvida|olvide|ignora|ignore) (?:todo|todas? las|las instrucciones|\S{1,15} instrucciones)\b",
    ],
    "extract": [
        r"\b(?:reveal|share|print|list|show|repeat|output|display|leak|dump|spell out|tell me)\b(?: \S{1,15}){0,3}? "
        r"(?:your|the|previous|above|secret|system|initial|hidden|original)(?: \S{1,15}){0,2}? "
        r"(?:instruct\w*|prompts?|rules|password|passphrase|secret)\b",
        r"\bwhat (?:are|were) (?:the |your )?(?:\S{1,15} ){0,2}(?:instructions|rules) (?:you|that)\b",
    ],
    "role": [
        rf"\byou are (?:now|no longer)\b.{{0,40}}?\b(?:{_AGENT}|mode|unrestricted|unfiltered|jailbroken|free (?:of|from))\b",
        rf"\bfrom now on,? you\b",
        r"\b(?:new|updated|real|actual|true) (?:system )?instructions?\s?:",
        r"\bsystem prompt\b",
        r"\b(?:developer|god|jailbreak|dan) mode\b",
        rf"\bact as (?:an? |the )?(?:unrestricted|unfiltered|different) {_AGENT}",
        r"\b(?:a |an )?(?:new|another|additional|further|different) (?:instruction|task|order|command)s? for you\b",
        r"\b(?:now|next) (?:there (?:are|is)|follows?|comes?)(?: \S{1,12}){0,3}? (?:new|more|further|additional|other) (?:tasks|instructions|commands)\b",
        r"\b(?:jetzt|nun) (?:bist du|sind sie|folgen (?:neue|weitere))\b",
        r"\bdu bist (?:jetzt|nun|ab sofort)\b",
        r"\b(?:neue|weitere) (?:anweisung|aufgabe)n?(?: für dich)?\s?:",
    ],
    "address": [
        rf"\b(?:attention|note|message|instructions?|important|warning|notice) (?:to|for) (?:all |any )?(?:the )?{_AGENT}\b",
        rf"\bif you are an? {_AGENT}\b",
        rf"\b(?:dear|hey|hi|hello) {_AGENT}\b",
        rf"\b{_AGENT} (?:reading|browsing|processing|summari[sz]ing|viewing) this\b",
        rf"\b(?:to|for) (?:any|the) {_AGENT} (?:that|who) (?:reads?|sees?|processe?s?)\b",
        r"\b(?:add|include|insert|append|incorporate|embed|blend|integrate|render|mention|put|weave)\b.{0,80}?"
        r"\b(?:in|into|to|within|as (?:a )?part of) your (?:response|answer|output|summary|algorithm|code|codebase|"
        r"program|logic|solution|script|completion)s?\b",
    ],
    "markup": [
        r"<\|(?:im_start|im_end|system|user|assistant|endoftext|start_header_id|eot_id)\|>",
        r"\[/?inst\]", r"<</?sys>>", r"</?system>", r"^\s*#{2,4} ?(?:system|instruction|instructions)\s?:?\s*$",
        r"\bbegin (?:system|hidden|secret) (?:prompt|instructions?|message)\b",
    ],
    "secrecy": [
        r"\b(?:do not|don't|never) (?:tell|inform|mention|reveal|show|alert|notify)(?: this| it)?(?: to)? (?:the |your )?user\b",
        r"\bwithout (?:telling|informing|asking|alerting|notifying) (?:the |your )?user\b",
        r"\bthe user (?:has |had )?(?:already )?(?:authori[sz]ed|approved|consented|wants you|asked you|instructed you)\b",
        r"\b(?:keep|hide) (?:this|it) (?:secret |hidden )?from (?:the |your )?user\b",
    ],
    "exfil": [
        r"\b(?:send|post|upload|forward|email|e-mail|submit|transmit|leak)\b (?:this|it|them|the|all|every|your|any|my)\b.{0,60}?\b(?:to|at|into)\b.{0,10}?(?:https?://|\b[\w.-]{1,40}@[\w-]{1,40}\.)",
        r"\b(?:curl|wget|iwr|invoke-webrequest|irm|invoke-restmethod)\b[^|\n]{0,200}\|\s?(?:sh|bash|zsh|iex|invoke-expression|python3?|powershell)\b",
        r"\bpowershell(?:\.exe)?\b[^\n]{0,40}\s-e(?:nc|ncodedcommand)?\s+[a-z0-9+/=]{20,}",
        r"\b(?:api keys?|passwords?|credentials?|tokens?|secrets?|cookies?|ssh keys?)\b.{0,40}?\b(?:to|at)\b.{0,10}?https?://",
    ],
}
_COMPILED = [(kind, re.compile(p, re.IGNORECASE | re.MULTILINE)) for kind, ps in PATTERNS.items() for p in ps]

BROWSERS = {"msedge.exe", "chrome.exe", "firefox.exe", "brave.exe", "opera.exe", "vivaldi.exe", "arc.exe",
            "iexplore.exe", "librewolf.exe", "waterfox.exe"}
MAIL = {"outlook.exe", "olk.exe", "thunderbird.exe", "hxoutlook.exe", "mailbird.exe"}
CHAT = {"teams.exe", "ms-teams.exe", "slack.exe", "discord.exe", "telegram.exe", "whatsapp.exe", "signal.exe",
        "zoom.exe", "element.exe"}
TERMINALS = {"windowsterminal.exe", "cmd.exe", "powershell.exe", "pwsh.exe", "conhost.exe", "openconsole.exe",
             "wezterm-gui.exe", "alacritty.exe", "mintty.exe"}
EDITORS = {"code.exe", "notepad.exe", "notepad++.exe", "pycharm64.exe", "idea64.exe", "devenv.exe", "sublime_text.exe",
           "winword.exe", "obsidian.exe", "cursor.exe", "zed.exe"}


def category(process: str) -> str:
    """Where the text came from, roughly: browser, mail, chat, terminal, editor or other."""
    p = (process or "").lower()
    for name, s in (("browser", BROWSERS), ("mail", MAIL), ("chat", CHAT), ("terminal", TERMINALS),
                    ("editor", EDITORS)):
        if p in s:
            return name
    return "other"


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text[:MAX_SCAN])
    t = _INVISIBLE.sub("", t).casefold()
    return "\n".join(_WS.sub(" ", ln).strip() for ln in t.split("\n"))


def hidden(text: str) -> str | None:
    """-> a description of the invisible characters in ``text``, or None."""
    t = text[:MAX_SCAN]
    found = []
    if n := len(_TAGS.findall(t)):
        found.append(f"{n} tag characters")
    if n := len(_BIDI.findall(t)):
        found.append(f"{n} bidi controls")
    if (n := len(_ZW.findall(t))) >= _ZW_RUN:
        found.append(f"{n} zero-width characters")
    return ", ".join(found) or None


_CACHE: OrderedDict = OrderedDict()   # digest of a block -> its findings; the watcher re-redacts unchanged text
_CACHE_SIZE = 1024


def scan(text: str) -> list[dict]:
    """-> findings in one block of (already scrubbed) text: [{kind, excerpt}], at most one per pattern. Results
    are cached by a digest of the text (the text itself is not kept)."""
    key = hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=16).digest()
    got = _CACHE.get(key)
    if got is not None:
        _CACHE.move_to_end(key)
        return [dict(f) for f in got]
    out = _scan(text)
    _CACHE[key] = out
    if len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return [dict(f) for f in out]


def _scan(text: str) -> list[dict]:
    out = []
    h = hidden(text)
    if h:
        out.append(dict(kind="hidden", excerpt=h))
    t = normalize(text)
    for kind, rx in _COMPILED:
        m = rx.search(t)
        if m:
            a = max(0, m.start() - (EXCERPT - (m.end() - m.start())) // 2)
            out.append(dict(kind=kind, excerpt=t[a:a + EXCERPT].replace("\n", " ")))
    return out


def flag_window(w) -> list[dict]:
    """Findings for one (redacted) window: its title and every text block, with provenance."""
    if getattr(w, "redacted", False):
        return []
    cat = category(w.process)
    flags, kinds_seen = [], set()
    sources = [("title", None, w.title or "")] + [(b.source, b.role, b.text) for b in w.text]
    for src, role, text in sources:
        for f in scan(text):
            key = (f["kind"], src)
            if key in kinds_seen:
                continue
            kinds_seen.add(key)
            flags.append(dict(f, source=src, role=role, app=cat))
            if len(flags) >= MAX_FLAGS:
                return flags
    return flags


def flag_lines(lines: list[str], process: str) -> list[dict]:
    """Findings in lines that appeared (a change event), with the window's app category."""
    cat = category(process)
    seen, out = set(), []
    for f in scan("\n".join(lines)):
        if f["kind"] not in seen:
            seen.add(f["kind"])
            out.append(dict(f, app=cat))
    return out[:MAX_FLAGS]
