# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Screen history on disk: the watcher's redacted events and periodic redacted snapshots, encrypted, kept 7 days.

Opt-in: nothing is recorded unless ``IGNITE_SCRIBE_HISTORY=1``, or the policy sets ``"history": true``.
- **What is stored:** only what the watcher already redacted. That is its change events (``changes.diff`` of
  redacted snapshots) and, every ``snapshot_s`` while the screen changes, the redacted snapshot. Excluded windows
  appear as the placeholder title, never their content. No pixels and no OCR text (OCR runs per call).
- **How:** batched chunks in SQLite (``%LOCALAPPDATA%\\ignite-scribe\\history.db``; ``IGNITE_SCRIBE_HISTORY_DB``
  overrides). Each chunk is JSON, compressed, then sealed with AES-256-GCM (``aesgcm``, Windows CNG) under one
  random data key; that key is stored only wrapped with DPAPI for the current Windows user (``dpapi``), so another
  account, or the file copied elsewhere, cannot read it. Only a chunk's kind, time range and count are in the
  clear, and they are the cipher's associated data, so they cannot be altered unnoticed. ``purge`` also replaces
  the key.
- **Retention:** chunks older than ``days`` (7) are deleted, and the oldest go first while the file exceeds
  ``max_mb`` (200). It is checked at every flush.
- **Search:** ``search`` decrypts the chunks in a time range and matches words (case-insensitive, all must
  appear) in window titles and text lines, optionally within windows whose title or process contains ``match``.
  ``at`` returns the snapshot in force at a time, plus the events after it.
Writes happen on this module's own thread, so a slow disk never stalls the watcher.
"""
from __future__ import annotations

import json
import os
import queue
import sqlite3
import threading
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

from . import aesgcm, dpapi

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, kind TEXT NOT NULL, t0 REAL NOT NULL, t1 REAL NOT NULL,
                                   n INTEGER NOT NULL, blob BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS chunks_t ON chunks (kind, t1);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v BLOB NOT NULL);
"""
FORMAT = b"\x01"                                # chunk format: zlib(JSON) sealed with AES-256-GCM


def default_path() -> Path:
    env = os.environ.get("IGNITE_SCRIBE_HISTORY_DB")
    if env:
        return Path(env)
    return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "ignite-scribe" / "history.db"


def enabled(policy=None) -> bool:
    v = os.environ.get("IGNITE_SCRIBE_HISTORY")
    if v is not None:
        return v == "1"
    return bool(getattr(policy, "history", False))


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).astimezone().isoformat(timespec="seconds")


UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_time(v, now: float | None = None) -> float | None:
    """Epoch seconds from a number, a relative ``"90m"`` / ``"2h"`` / ``"1d"`` (ago), or ISO 8601 (local time if
    no offset is given). None stays None."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    now = time.time() if now is None else now
    if s[-1:].lower() in UNITS and s[:-1].replace(".", "", 1).isdigit():
        return now - float(s[:-1]) * UNITS[s[-1].lower()]
    d = datetime.fromisoformat(s)
    return (d if d.tzinfo else d.astimezone()).timestamp()


def aad(kind, t0, t1, n) -> bytes:
    return f"{kind}|{float(t0)!r}|{float(t1)!r}|{int(n)}".encode()


class History:
    def __init__(self, path: str | Path | None = None, days: float = 7.0, max_mb: float = 200.0,
                 flush_s: float = 60.0, flush_events: int = 500, snapshot_s: float = 300.0):
        self.path = Path(path) if path else default_path()
        self.days, self.max_mb, self.flush_s, self.flush_events = days, max_mb, flush_s, flush_events
        self.snapshot_s = snapshot_s
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db_lock = threading.Lock()
        self.db = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA auto_vacuum = INCREMENTAL")
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.executescript(SCHEMA)
        self._aead = aesgcm.AesGcm(self._data_key())
        self._q: queue.Queue = queue.Queue()
        self._pending: list[dict] = []           # events not yet flushed (searchable already)
        self._pending_lock = threading.Lock()
        self._last_snap_t = 0.0
        self._snap_dirty = True
        self._cache: dict[int, object] = {}
        self.stats = dict(chunks_written=0, events_written=0, snapshots_written=0, deleted=0, errors=0)
        self.last_error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ignite-scribe-history", daemon=True)
        self._thread.start()

    def _data_key(self, new: bool = False) -> bytes:
        row = None if new else self.db.execute("SELECT v FROM meta WHERE k = 'key'").fetchone()
        if row is not None:
            return dpapi.unprotect(row[0])
        key = os.urandom(32)
        self.db.execute("INSERT OR REPLACE INTO meta (k, v) VALUES ('key', ?)", (dpapi.protect(key),))
        return key

    def seal(self, kind, t0, t1, n, obj) -> bytes:
        raw = zlib.compress(json.dumps(obj, ensure_ascii=False).encode("utf-8"), 6)
        return FORMAT + self._aead.seal(raw, aad(kind, t0, t1, n))

    def unseal(self, kind, t0, t1, n, blob: bytes):
        if blob[:1] != FORMAT:
            raise ValueError("unknown history chunk format")
        return json.loads(zlib.decompress(self._aead.open(blob[1:], aad(kind, t0, t1, n))).decode("utf-8"))

    # ---- writing (the watcher calls ``record``; everything else happens on this module's thread) -----------
    def record(self, events: list[dict], snapshot=None):
        """Queue redacted events (and the redacted snapshot they came from). Never blocks on the disk."""
        self._q.put((list(events), snapshot))

    def _run(self):
        last_flush = time.time()
        while not self._stop.is_set() or not self._q.empty():
            try:
                events, snap = self._q.get(timeout=0.5)
                with self._pending_lock:
                    self._pending += events
                if events:
                    self._snap_dirty = True
                now = time.time()
                if snap is not None and self._snap_dirty and now - self._last_snap_t >= self.snapshot_s:
                    self._write("snapshot", snap.taken_at, snap.taken_at, 1, snap.to_dict())
                    self._last_snap_t, self._snap_dirty = now, False
                    self.stats["snapshots_written"] += 1
            except queue.Empty:
                pass
            except Exception as e:  # noqa: BLE001 - history must never take the watcher down
                self._error(e)
            if self._pending and (time.time() - last_flush >= self.flush_s or len(self._pending) >= self.flush_events
                                  or self._stop.is_set()):
                self.flush()
                last_flush = time.time()

    def flush(self):
        with self._pending_lock:
            evs, self._pending = self._pending, []
        if not evs:
            return
        try:
            self._write("events", evs[0]["t"], evs[-1]["t"], len(evs), evs)
            self.stats["events_written"] += len(evs)
            self.prune()
        except Exception as e:  # noqa: BLE001
            self._error(e)

    def _write(self, kind, t0, t1, n, obj):
        blob = self.seal(kind, t0, t1, n, obj)
        with self._db_lock:
            self.db.execute("INSERT INTO chunks (kind, t0, t1, n, blob) VALUES (?, ?, ?, ?, ?)", (kind, t0, t1, n, blob))
        self.stats["chunks_written"] += 1

    def _error(self, e):
        self.stats["errors"] += 1
        self.last_error = f"{type(e).__name__}: {e}"

    def prune(self, now: float | None = None) -> int:
        now = time.time() if now is None else now
        with self._db_lock:
            n = self.db.execute("DELETE FROM chunks WHERE t1 < ?", (now - self.days * 86400,)).rowcount
            while self.size_mb() > self.max_mb:
                ids = [r[0] for r in self.db.execute("SELECT id FROM chunks ORDER BY t1 LIMIT 50")]
                if not ids:
                    break
                n += self.db.execute(f"DELETE FROM chunks WHERE id IN ({','.join('?' * len(ids))})", ids).rowcount
                self.db.execute("PRAGMA incremental_vacuum")
                self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            if n:
                self.db.execute("PRAGMA incremental_vacuum")
        self.stats["deleted"] += n
        if n:
            self._cache.clear()
        return n

    def size_mb(self) -> float:
        page = self.db.execute("PRAGMA page_size").fetchone()[0]
        count = self.db.execute("PRAGMA page_count").fetchone()[0]
        free = self.db.execute("PRAGMA freelist_count").fetchone()[0]
        return (count - free) * page / 1e6

    def close(self):
        self._stop.set()
        self._thread.join(10)
        self.flush()
        with self._db_lock:
            self.db.close()
        self._aead.close()

    def purge(self) -> int:
        """Delete everything recorded (the file stays, empty)."""
        with self._pending_lock:
            self._pending = []
        with self._db_lock:
            n = self.db.execute("DELETE FROM chunks").rowcount
            self._aead.close()
            self._aead = aesgcm.AesGcm(self._data_key(new=True))
            self.db.execute("VACUUM")
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self._cache.clear()
        return n

    # ---- reading ---------------------------------------------------------------------------------------------
    def _chunks(self, kind, since, until):
        with self._db_lock:
            rows = self.db.execute("SELECT id, t0, t1, n, blob FROM chunks WHERE kind = ? AND t1 >= ? AND t0 <= ? "
                                   "ORDER BY t0 DESC", (kind, since, until)).fetchall()
        for cid, t0, t1, n, blob in rows:
            if cid not in self._cache:
                if len(self._cache) > 256:
                    self._cache.clear()
                self._cache[cid] = self.unseal(kind, t0, t1, n, blob)
            yield t0, t1, self._cache[cid]

    def search(self, query: str = "", since: float | None = None, until: float | None = None,
               match: str | None = None, limit: int = 50, budget_s: float = 8.0) -> dict:
        """Hits newest first. Stops after ``budget_s`` (``complete`` False, ``searched_back_to`` says how far)."""
        now = time.time()
        since = now - self.days * 86400 if since is None else since
        until = now if until is None else until
        terms = [t.lower() for t in query.split()]
        m = (match or "").lower()
        hits, seen = [], {}

        def consider(t, kind, title, process, line):
            if not since <= t <= until:
                return
            if m and m not in (title or "").lower() and m not in (process or "").lower():
                return
            hay = f"{title}\n{line}".lower() if line else (title or "").lower()
            if terms and not all(x in hay for x in terms):
                return
            key = (process, title, line)
            if key in seen:
                h = seen[key]
                h["first_seen"], h["seen_count"] = min(h["first_seen"], t), h["seen_count"] + 1
                h["last_seen"] = max(h["last_seen"], t)
                return
            h = dict(first_seen=t, last_seen=t, seen_count=1, kind=kind, title=title, process=process, line=line)
            seen[key] = h
            hits.append(h)

        def from_events(evs):
            for e in reversed(evs):
                if e["kind"] == "content":
                    for ln in e.get("added", []):
                        consider(e["t"], "content", e.get("title"), e.get("process"), ln)
                elif e["kind"] in ("opened", "focus", "title"):
                    consider(e["t"], e["kind"], e.get("title"), e.get("process"), None)

        end = time.monotonic() + budget_s
        reached = {"snapshot": until, "events": until}
        with self._pending_lock:
            pending = list(self._pending)
        from_events(pending)
        for kind in ("snapshot", "events"):             # newest first; snapshots are few and hold whole windows
            for t0, _, chunk in self._chunks(kind, since, until):
                if time.monotonic() > end:
                    break
                if kind == "events":
                    from_events(chunk)
                else:
                    for w in chunk.get("windows", []):
                        consider(t0, "snapshot", w.get("title"), w.get("process"), None)
                        for b in w.get("text", []):
                            for ln in b.get("text", "").split("\n"):
                                if ln.strip():
                                    consider(t0, "snapshot", w.get("title"), w.get("process"), ln)
                reached[kind] = max(since, t0)
            else:
                reached[kind] = since
        hits.sort(key=lambda h: h["last_seen"], reverse=True)
        out = hits[:limit]
        for h in out:
            h["first_seen"], h["last_seen"] = iso(h["first_seen"]), iso(h["last_seen"])
        back = max(reached.values())
        res = dict(query=query, since=iso(since), until=iso(until), total=len(hits), hits=out,
                   complete=back <= since)
        if back > since:
            res["searched_back_to"] = iso(back)
            res["note_partial"] = (f"stopped after the {budget_s:g} s budget; narrow `since` or ask again with "
                                   f"`until` = searched_back_to for older history")
        return res

    def at(self, t: float, events_limit: int = 200) -> dict:
        """The redacted snapshot in force at ``t`` (the last one recorded before it) and the events after it."""
        with self._db_lock:
            row = self.db.execute("SELECT id, t0, t1, n, blob FROM chunks WHERE kind = 'snapshot' AND t0 <= ? "
                                  "ORDER BY t0 DESC LIMIT 1", (t,)).fetchone()
        if row is None:
            return dict(at=iso(t), snapshot=None, events=[], note="no snapshot recorded before this time")
        snap = self.unseal("snapshot", *row[1:])
        evs = []
        for _, _, chunk in self._chunks("events", row[1], t):
            evs += [e for e in chunk if row[1] < e["t"] <= t]
        with self._pending_lock:
            evs += [e for e in self._pending if row[1] < e["t"] <= t]
        evs.sort(key=lambda e: e["seq"] if "seq" in e else e["t"])
        return dict(at=iso(t), snapshot_taken=iso(row[1]), snapshot=snap, events=evs[-events_limit:],
                    events_total=len(evs))

    def status(self) -> dict:
        with self._db_lock:
            r = self.db.execute("SELECT COUNT(*), MIN(t0), MAX(t1), SUM(n) FROM chunks").fetchone()
            mb = self.size_mb()
        return dict(recording=self._thread.is_alive(), path=str(self.path), chunks=r[0], oldest=iso(r[1]) if r[1] else None,
                    newest=iso(r[2]) if r[2] else None, size_mb=round(mb, 2), retention_days=self.days,
                    max_mb=self.max_mb, pending_events=len(self._pending), encrypted="AES-256-GCM, key wrapped with DPAPI (current user)",
                    **self.stats, last_error=self.last_error)
