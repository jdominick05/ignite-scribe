# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Screen history on disk: the watcher's redacted events and periodic redacted snapshots, encrypted, kept 7 days.

Opt-in: nothing is recorded unless ``IGNITE_SCRIBE_HISTORY=1``, or the policy sets ``"history": true``.
- **What is stored:** only what the watcher already redacted. That is its change events (``changes.diff`` of
  redacted snapshots) and, every ``snapshot_s`` while the screen changes, the redacted snapshot. Excluded windows
  appear as the placeholder title, never their content. No pixels and no OCR text (OCR runs per call). Derived
  from those: window passages (``passages.py``) and their embedding vectors (``semantic.py``), sealed the same way.
- **How:** batched chunks in SQLite (``%LOCALAPPDATA%\\ignite-scribe\\history.db``; ``IGNITE_SCRIBE_HISTORY_DB``
  overrides). Each chunk is JSON, compressed, then sealed with AES-256-GCM (``aesgcm``, Windows CNG) under one
  random data key; that key is stored only wrapped with DPAPI for the current Windows user (``dpapi``), so another
  account, or the file copied elsewhere, cannot read it. Only a chunk's kind, time range and count are in the
  clear, and they are the cipher's associated data, so they cannot be altered unnoticed. ``purge`` also replaces
  the key.
- **Retention:** chunks older than ``days`` (7) are deleted, and the oldest go first while the file exceeds
  ``max_mb`` (200). It runs on open, at every flush and every ``prune_s`` (also while the screen is quiet), and
  reads never reach past it. ``secure_delete`` overwrites deleted chunks instead of leaving them in free pages.
- **One recorder:** see ``History``. A chunk that fails authentication is skipped and counted, never trusted.
- **Search:** ``search`` decrypts the chunks in a time range and matches words (case-insensitive, all must
  appear) in window titles and text lines, optionally within windows whose title or process contains ``match``.
  ``at`` returns the snapshot in force at a time, plus the events after it.
Writes happen on this module's own thread, so a slow disk never stalls the watcher.
"""
from __future__ import annotations

import json
import msvcrt
import os
import queue
import sqlite3
import threading
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

from . import aesgcm, dpapi
from .passages import Passages

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
    """One file, many readers, one recorder.

    The MCP server runs in every client session, so several processes may open the same file. Only the one that
    holds the recorder lock (an exclusive byte-range lock on ``<file>.lock``, released by Windows if the process
    dies) records; the others only search, and every ``takeover_s`` they try the lock, so when the recording
    session ends another one carries on (``on_recorder`` is then called, on this module's thread, to start its
    watcher). ``record=False`` never records (the CLI)."""

    def __init__(self, path: str | Path | None = None, days: float = 7.0, max_mb: float = 200.0,
                 flush_s: float = 60.0, flush_events: int = 500, snapshot_s: float = 300.0, record: bool = True,
                 prune_s: float = 600.0, takeover_s: float = 30.0):
        self.path = Path(path) if path else default_path()
        self.days, self.max_mb, self.flush_s, self.flush_events = days, max_mb, flush_s, flush_events
        self.snapshot_s, self.prune_s, self.takeover_s, self.want_record = snapshot_s, prune_s, takeover_s, record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db_lock = threading.RLock()                  # purge holds it while it replaces the key
        self.db = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None, timeout=15)
        self.db.execute("PRAGMA auto_vacuum = INCREMENTAL")
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.execute("PRAGMA secure_delete = ON")      # deleted chunks are overwritten, not left in free pages
        self.db.executescript(SCHEMA)
        self._lockfile = None
        self.recorder = record and self._take_lock()
        self.on_recorder = None
        self._key_blob = None
        self._aead = aesgcm.AesGcm(self._data_key())
        self.stats = dict(chunks_written=0, events_written=0, snapshots_written=0, deleted=0, errors=0,
                          dropped_not_recorder=0)
        self._cache: dict[int, object] = {}
        self.prune()                                     # retention holds from open, not only after new events
        self._q: queue.Queue = queue.Queue()
        self._pending: list[dict] = []           # events not yet flushed (searchable already)
        self._pending_lock = threading.Lock()
        self._last_snap_t = 0.0
        self._snap_dirty = True
        self.passages = Passages()                       # window passages for semantic search (semantic.py)
        self.last_error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ignite-scribe-history", daemon=True)
        self._thread.start()

    # ---- the recorder lock and the key ------------------------------------------------------------------------
    def _take_lock(self) -> bool:
        if self._lockfile is not None:
            return True
        f = open(self.path.with_name(self.path.name + ".lock"), "a+b")
        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            f.close()
            return False
        self._lockfile = f
        return True

    def _release_lock(self):
        if self._lockfile is not None:
            try:
                self._lockfile.seek(0)
                msvcrt.locking(self._lockfile.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            self._lockfile.close()
            self._lockfile = None

    def _data_key(self, new: bool = False) -> bytes:
        """The data key, created once per file inside one write transaction, so processes opening a fresh file
        at the same time all end up with the key that is stored."""
        with self._db_lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                row = self.db.execute("SELECT v FROM meta WHERE k = 'key'").fetchone()
                if row is None or new:
                    blob = dpapi.protect(os.urandom(32))
                    self.db.execute("INSERT OR REPLACE INTO meta (k, v) VALUES ('key', ?)", (blob,))
                else:
                    blob = row[0]
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
        self._key_blob = blob
        return dpapi.unprotect(blob)

    def _reload_key(self) -> bool:
        """Pick up a key replaced by another process's purge. True if it changed."""
        with self._db_lock:
            row = self.db.execute("SELECT v FROM meta WHERE k = 'key'").fetchone()
        if row is None or row[0] == self._key_blob:
            return False
        old, self._aead = self._aead, aesgcm.AesGcm(self._data_key())
        old.close()
        self._cache.clear()
        return True

    def seal(self, kind, t0, t1, n, obj) -> bytes:
        raw = zlib.compress(json.dumps(obj, ensure_ascii=False).encode("utf-8"), 6)
        return FORMAT + self._aead.seal(raw, aad(kind, t0, t1, n))

    def unseal(self, kind, t0, t1, n, blob: bytes):
        if blob[:1] != FORMAT:
            raise ValueError("unknown history chunk format")
        return json.loads(zlib.decompress(self._aead.open(blob[1:], aad(kind, t0, t1, n))).decode("utf-8"))

    # ---- writing (the watcher calls ``record``; everything else happens on this module's thread) -----------
    def record(self, events: list[dict], snapshot=None):
        """Queue redacted events (and the redacted snapshot they came from). Never blocks on the disk. Dropped
        unless this instance is the recorder."""
        if not self.recorder:
            self.stats["dropped_not_recorder"] += len(events)
            return
        self._q.put((list(events), snapshot))

    def _maintain(self, now, last):
        """Retention on a timer (also while nothing changes), and recorder takeover."""
        if now - last["prune"] >= self.prune_s:
            last["prune"] = now
            self.prune()
        if self.want_record and not self.recorder and now - last["takeover"] >= self.takeover_s:
            last["takeover"] = now
            if self._take_lock():
                self._reload_key()
                self.recorder = True
                if self.on_recorder is not None:
                    self.on_recorder()

    def _run(self):
        last_flush = time.time()
        last = dict(prune=time.time(), takeover=time.time())
        while not self._stop.is_set() or not self._q.empty():
            try:
                self._maintain(time.time(), last)
            except Exception as e:  # noqa: BLE001
                self._error(e)
            try:
                events, snap = self._q.get(timeout=0.5)
                with self._pending_lock:
                    self._pending += events
                if events:
                    self._snap_dirty = True
                now = time.time()
                sd = None
                if snap is not None and self._snap_dirty and now - self._last_snap_t >= self.snapshot_s:
                    sd = snap.to_dict()
                    self._write("snapshot", snap.taken_at, snap.taken_at, 1, sd)
                    self._last_snap_t, self._snap_dirty = now, False
                    self.stats["snapshots_written"] += 1
                self.passages.feed(events, sd)
            except queue.Empty:
                pass
            except Exception as e:  # noqa: BLE001 - history must never take the watcher down
                self._error(e)
            if self._pending and (time.time() - last_flush >= self.flush_s or len(self._pending) >= self.flush_events
                                  or self._stop.is_set()):
                self.flush()
                last_flush = time.time()
            self._write_passages(None if self._stop.is_set() else time.time())

    def _write_passages(self, now):
        """Store the passages of the buckets that have closed (all of them at close)."""
        try:
            ps = self.passages.close(now)
            if ps:
                self._write("passages", min(x["t0"] for x in ps), max(x["t1"] for x in ps), len(ps), ps)
                self.stats["passages_written"] = self.stats.get("passages_written", 0) + len(ps)
        except Exception as e:  # noqa: BLE001
            self._error(e)

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

    def _write(self, kind, t0, t1, n, obj) -> int:
        self._reload_key()                               # never seal with a key a purge has replaced
        blob = self.seal(kind, t0, t1, n, obj)
        with self._db_lock:
            cid = self.db.execute("INSERT INTO chunks (kind, t0, t1, n, blob) VALUES (?, ?, ?, ?, ?)",
                                  (kind, t0, t1, n, blob)).lastrowid
        self.stats["chunks_written"] += 1
        return cid

    # ---- semantic index (semantic.py drives it; only the recorder writes) -------------------------------------
    def _embedded_upto(self, model: str) -> int:
        with self._db_lock:
            row = self.db.execute("SELECT v FROM meta WHERE k = ?", (f"embedded:{model}",)).fetchone()
        return int(row[0]) if row else 0

    def unembedded(self, model: str, limit: int = 8) -> list[tuple[int, float, float, list]]:
        """Passage chunks not yet embedded with ``model``, oldest first: (id, t0, t1, passages)."""
        upto = self._embedded_upto(model)
        with self._db_lock:
            rows = self.db.execute("SELECT id, t0, t1, n, blob FROM chunks WHERE kind = 'passages' AND id > ? "
                                   "AND t1 >= ? ORDER BY id LIMIT ?", (upto, self.horizon(), limit)).fetchall()
        out, bad = [], []
        for cid, t0, t1, n, blob in rows:
            ps = self._open("passages", t0, t1, n, blob, bad)
            out.append((cid, t0, t1, ps or []))
        return out

    def add_vectors(self, model: str, src: int, t0: float, t1: float, meta: list[dict], vecs: bytes):
        """Store the vectors of passage chunk ``src`` (float16, row-major) and mark it embedded."""
        if not self.recorder:
            return
        if meta:
            self._write("vectors", t0, t1, len(meta), dict(model=model, src=src, meta=meta, vecs=vecs.hex()))
        with self._db_lock:
            self.db.execute("INSERT OR REPLACE INTO meta (k, v) VALUES (?, ?)", (f"embedded:{model}", src))

    def vector_chunks(self, model: str, since: float, until: float, bad: list):
        """(t0, t1, chunk) for the vectors chunks of ``model`` in range, newest first."""
        for t0, t1, c in self._chunks("vectors", max(since, self.horizon()), until, bad):
            if c.get("model") == model:
                yield t0, t1, c

    def passages_of(self, src: int) -> list[dict]:
        with self._db_lock:
            row = self.db.execute("SELECT t0, t1, n, blob FROM chunks WHERE id = ? AND kind = 'passages'",
                                  (src,)).fetchone()
        return (self._open("passages", *row, []) or []) if row else []

    def index_status(self, model: str) -> dict:
        upto = self._embedded_upto(model)
        with self._db_lock:
            r = self.db.execute("SELECT COUNT(*), COALESCE(SUM(n), 0) FROM chunks WHERE kind = 'passages' AND id > ?",
                                (upto,)).fetchone()
            v = self.db.execute("SELECT COALESCE(SUM(n), 0) FROM chunks WHERE kind = 'vectors'").fetchone()
        return dict(model=model, vectors=v[0], backlog_chunks=r[0], backlog_passages=r[1])

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
            if n:
                self.db.execute("PRAGMA incremental_vacuum")
                self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
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
        if self.recorder:
            self._write_passages(None)
        with self._db_lock:
            self.db.close()
        self._aead.close()
        self._release_lock()

    def purge(self) -> int:
        """Delete everything recorded (the file stays, empty) and replace the key. Refused while another process
        records, since it would go on sealing with the old key."""
        if not self.recorder and not self._take_lock():
            raise RuntimeError("an ignite-scribe server is recording this history now; close its client sessions "
                               "(or set IGNITE_SCRIBE_HISTORY=0 for them) and purge again")
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
    def _open(self, kind, t0, t1, n, blob, bad: list):
        """A chunk's contents, or None (counted in ``bad``) when it fails authentication even with the stored key."""
        for attempt in (0, 1):
            try:
                return self.unseal(kind, t0, t1, n, blob)
            except ValueError:
                if attempt or not self._reload_key():
                    bad.append(1)
                    return None

    def _chunks(self, kind, since, until, bad: list):
        with self._db_lock:
            rows = self.db.execute("SELECT id, t0, t1, n, blob FROM chunks WHERE kind = ? AND t1 >= ? AND t0 <= ? "
                                   "ORDER BY t0 DESC", (kind, since, until)).fetchall()
        for cid, t0, t1, n, blob in rows:
            got = self._cache.get(cid)
            if got is None:
                got = self._open(kind, t0, t1, n, blob, bad)
                if got is None:
                    continue
                if len(self._cache) > 256:
                    self._cache.clear()
                self._cache[cid] = got
            yield t0, t1, got

    def horizon(self, now: float | None = None) -> float:
        return (time.time() if now is None else now) - self.days * 86400

    def search(self, query: str = "", since: float | None = None, until: float | None = None,
               match: str | None = None, limit: int = 50, budget_s: float = 8.0) -> dict:
        """Hits newest first. Stops after ``budget_s`` (``complete`` False, ``searched_back_to`` says how far)."""
        now = time.time()
        since = self.horizon(now) if since is None else max(since, self.horizon(now))   # never past retention
        until = now if until is None else until
        bad: list = []
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
            for t0, _, chunk in self._chunks(kind, since, until, bad):
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
        if bad:
            res["unreadable_chunks"] = len(bad)        # failed authentication: skipped, never trusted
        if back > since:
            res["searched_back_to"] = iso(back)
            res["note_partial"] = (f"stopped after the {budget_s:g} s budget; narrow `since` or ask again with "
                                   f"`until` = searched_back_to for older history")
        return res

    def at(self, t: float, events_limit: int = 200) -> dict:
        """The redacted snapshot in force at ``t`` (the last readable one recorded before it, within retention)
        and the events after it."""
        h = self.horizon()
        if t < h:
            return dict(at=iso(t), snapshot=None, events=[], note=f"older than the {self.days:g}-day retention")
        bad: list = []
        with self._db_lock:
            rows = self.db.execute("SELECT id, t0, t1, n, blob FROM chunks WHERE kind = 'snapshot' AND t0 <= ? "
                                   "AND t0 >= ? ORDER BY t0 DESC", (t, h)).fetchall()
        snap = row = None
        for row in rows:
            snap = self._open("snapshot", *row[1:], bad)
            if snap is not None:
                break
        if snap is None:
            return dict(at=iso(t), snapshot=None, events=[], note="no snapshot recorded before this time",
                        **({"unreadable_chunks": len(bad)} if bad else {}))
        evs = []
        for _, _, chunk in self._chunks("events", row[1], t, bad):
            evs += [e for e in chunk if row[1] < e["t"] <= t]
        with self._pending_lock:
            evs += [e for e in self._pending if row[1] < e["t"] <= t]
        evs.sort(key=lambda e: e["seq"] if "seq" in e else e["t"])
        return dict(at=iso(t), snapshot_taken=iso(row[1]), snapshot=snap, events=evs[-events_limit:],
                    events_total=len(evs), **({"unreadable_chunks": len(bad)} if bad else {}))

    def status(self) -> dict:
        with self._db_lock:
            r = self.db.execute("SELECT COUNT(*), MIN(t0), MAX(t1), SUM(n) FROM chunks").fetchone()
            mb = self.size_mb()
        role = ("recorder" if self.recorder else
                "reader: another ignite-scribe process records; this one takes over when it exits" if self.want_record
                else "reader")
        return dict(recording=self.recorder and self._thread.is_alive(), role=role, path=str(self.path),
                    chunks=r[0], oldest=iso(r[1]) if r[1] else None, newest=iso(r[2]) if r[2] else None,
                    size_mb=round(mb, 2), retention_days=self.days, max_mb=self.max_mb,
                    pending_events=len(self._pending),
                    encrypted="AES-256-GCM, key wrapped with DPAPI (current user)", **self.stats,
                    last_error=self.last_error)
