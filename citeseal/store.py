"""Append-only store of cited records.

Two properties matter, and both are about trust rather than performance:

**Append-only.** A changed record does not overwrite its predecessor. The old
row is archived and the new one takes version+1, so "what did this say last
month, and to what source?" is answerable. A store that silently overwrites
cannot be audited, and an unauditable store is just a cache.

**Change events.** A new or changed record emits an event. This is what makes a
"what changed?" tool possible, and it is the difference between a search index
and a monitored corpus.

Re-ingesting an unchanged record is a no-op, not an event. Without that, every
refresh would look like a change and the feed would be noise.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .record import Record, utcnow

__all__ = ["Store", "UpsertOutcome"]

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- Current state of every record.
CREATE TABLE IF NOT EXISTS records (
    record_id     TEXT    PRIMARY KEY,
    key           TEXT    NOT NULL,
    kind          TEXT    NOT NULL,
    title         TEXT    NOT NULL,
    body          TEXT    NOT NULL,
    source_url    TEXT    NOT NULL,          -- NOT NULL: the hard rule, at the schema level
    source_name   TEXT    NOT NULL,
    citation_text TEXT,
    attributes    TEXT    NOT NULL DEFAULT '{}',
    observed_at   TEXT    NOT NULL,
    version       INTEGER NOT NULL DEFAULT 1,
    content_hash  TEXT    NOT NULL,
    retired       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_records_kind ON records(kind);
CREATE INDEX IF NOT EXISTS idx_records_key  ON records(key);

-- Every superseded version, retained.
CREATE TABLE IF NOT EXISTS record_versions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id  TEXT    NOT NULL,
    version    INTEGER NOT NULL,
    snapshot   TEXT    NOT NULL,
    recorded_at TEXT   NOT NULL,
    UNIQUE(record_id, version)
);

-- The change feed.
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type  TEXT    NOT NULL,            -- 'record_new' | 'record_changed'
    record_id   TEXT    NOT NULL,
    kind        TEXT    NOT NULL,
    title       TEXT    NOT NULL,
    detail      TEXT,
    source_url  TEXT    NOT NULL,
    happened_at TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_at ON events(happened_at DESC);

-- Upstreams we have ingested from, so provenance is checkable.
CREATE TABLE IF NOT EXISTS sources (
    key         TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    url         TEXT,
    last_run_at TEXT,
    record_count INTEGER NOT NULL DEFAULT 0
);

-- Monotonic counter bumped on every write, used to invalidate the search index
-- without comparing whole tables.
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT OR IGNORE INTO meta(key, value) VALUES ('data_version', '0');
"""


@dataclass(frozen=True, slots=True)
class UpsertOutcome:
    record_id: str
    status: str          # 'new' | 'changed' | 'unchanged'
    version: int

    @property
    def wrote(self) -> bool:
        return self.status in ("new", "changed")


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # NOTE: `with sqlite3.connect(...)` commits on exit but does NOT close
        # the connection -- it is a transaction context manager, not a resource
        # one. Relying on it here leaked a handle per Store, which on Windows
        # kept the database file locked and broke temp-directory cleanup.
        conn = self._connect()
        try:
            conn.executescript(SCHEMA)
            conn.commit()
        finally:
            conn.close()

    # -- connection --------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Connection]:
        """One connection per thread, committed on clean exit.

        The COMMIT IS LOAD-BEARING. An earlier version rolled back on exception
        but never committed on success, so every write was discarded when the
        process exited. In-process tests could not see it: they read back through
        the same open connection, where uncommitted rows are visible. It only
        appeared when a second process opened the database and found nothing.
        ``test_writes_survive_reopening_the_store`` is the guard.
        """
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def close(self) -> None:
        """Release this thread's connection.

        Not optional on Windows: an open SQLite handle keeps the file locked, so
        a store whose directory is later removed raises WinError 32. Tests and
        short-lived scratch stores must close before cleaning up.
        """
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            finally:
                self._local.conn = None

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writes ------------------------------------------------------------

    def upsert(self, record: Record) -> UpsertOutcome:
        """Insert or supersede a record. Never overwrites in place."""
        record_id = record.record_id
        with self._cursor() as conn:
            existing = conn.execute(
                "SELECT version, content_hash FROM records WHERE record_id = ?",
                (record_id,),
            ).fetchone()

            if existing is None:
                conn.execute(
                    """INSERT INTO records
                       (record_id, key, kind, title, body, source_url, source_name,
                        citation_text, attributes, observed_at, version, content_hash)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        record_id, record.key, record.kind, record.title, record.body,
                        record.citation.url, record.citation.source_name,
                        record.citation.text,
                        json.dumps(record.attributes, ensure_ascii=False, default=str),
                        record.observed_at, 1, record.content_hash,
                    ),
                )
                self._emit(conn, "record_new", record, None)
                self._bump(conn)
                return UpsertOutcome(record_id, "new", 1)

            if existing["content_hash"] == record.content_hash:
                # Re-ingest of identical content. Record that we looked, but do
                # NOT emit an event: a refresh is not a change.
                conn.execute(
                    "UPDATE records SET observed_at = ? WHERE record_id = ?",
                    (record.observed_at, record_id),
                )
                return UpsertOutcome(record_id, "unchanged", existing["version"])

            previous = conn.execute(
                "SELECT * FROM records WHERE record_id = ?", (record_id,)
            ).fetchone()
            new_version = int(existing["version"]) + 1
            conn.execute(
                """INSERT OR REPLACE INTO record_versions
                   (record_id, version, snapshot, recorded_at) VALUES (?,?,?,?)""",
                (
                    record_id, previous["version"],
                    json.dumps(dict(previous), ensure_ascii=False, default=str),
                    utcnow(),
                ),
            )
            updated = record.with_version(new_version)
            conn.execute(
                """UPDATE records SET title=?, body=?, source_url=?, source_name=?,
                   citation_text=?, attributes=?, observed_at=?, version=?,
                   content_hash=? WHERE record_id=?""",
                (
                    updated.title, updated.body, updated.citation.url,
                    updated.citation.source_name, updated.citation.text,
                    json.dumps(updated.attributes, ensure_ascii=False, default=str),
                    updated.observed_at, new_version, updated.content_hash, record_id,
                ),
            )
            self._emit(conn, "record_changed", updated, dict(previous))
            self._bump(conn)
            return UpsertOutcome(record_id, "changed", new_version)

    def _emit(self, conn: sqlite3.Connection, event_type: str,
              record: Record, previous: dict[str, Any] | None) -> None:
        detail = None
        if previous is not None and previous.get("title") != record.title:
            detail = f"title: {previous.get('title')!r} -> {record.title!r}"
        conn.execute(
            """INSERT INTO events
               (event_type, record_id, kind, title, detail, source_url, happened_at)
               VALUES (?,?,?,?,?,?,?)""",
            (
                event_type, record.record_id, record.kind, record.title,
                detail, record.citation.url, record.observed_at,
            ),
        )

    def _bump(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "UPDATE meta SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT) "
            "WHERE key = 'data_version'"
        )

    def register_source(self, key: str, name: str, url: str | None,
                        record_count: int) -> None:
        with self._cursor() as conn:
            conn.execute(
                """INSERT INTO sources(key, name, url, last_run_at, record_count)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(key) DO UPDATE SET
                     name=excluded.name, url=excluded.url,
                     last_run_at=excluded.last_run_at,
                     record_count=excluded.record_count""",
                (key, name, url, utcnow(), record_count),
            )

    def retire(self, record_id: str) -> bool:
        """Withdraw a record from the corpus without deleting it.

        A record can be withdrawn upstream (a recall gets rescinded) and the
        corpus has to reflect that. Retiring keeps the row and its citation for
        audit while removing it from queries.

        This must go through the store rather than a direct UPDATE: it bumps the
        data version and emits an event, which is what invalidates the search
        index and tells subscribers something changed. Retiring with raw SQL
        leaves a stale index serving the withdrawn record.

        Returns True if the record moved to retired, False if it was already
        retired or does not exist.
        """
        with self._cursor() as conn:
            row = conn.execute(
                "SELECT * FROM records WHERE record_id = ?", (record_id,)
            ).fetchone()
            if row is None or row["retired"]:
                return False
            conn.execute(
                "UPDATE records SET retired = 1, observed_at = ? WHERE record_id = ?",
                (utcnow(), record_id),
            )
            conn.execute(
                """INSERT INTO events
                   (event_type, record_id, kind, title, detail, source_url, happened_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    "record_retired", record_id, row["kind"], row["title"],
                    "withdrawn from the corpus; retained for audit",
                    row["source_url"], utcnow(),
                ),
            )
            self._bump(conn)
            return True

    def unretire(self, record_id: str) -> bool:
        """Return a retired record to the corpus."""
        with self._cursor() as conn:
            row = conn.execute(
                "SELECT retired FROM records WHERE record_id = ?", (record_id,)
            ).fetchone()
            if row is None or not row["retired"]:
                return False
            conn.execute(
                "UPDATE records SET retired = 0, observed_at = ? WHERE record_id = ?",
                (utcnow(), record_id),
            )
            self._bump(conn)
            return True

    # -- reads -------------------------------------------------------------

    def _row_to_record(self, row: sqlite3.Row) -> Record:
        from .record import Citation

        return Record(
            key=row["key"],
            kind=row["kind"],
            title=row["title"],
            body=row["body"],
            citation=Citation(
                url=row["source_url"],
                source_name=row["source_name"],
                text=row["citation_text"],
            ),
            attributes=json.loads(row["attributes"] or "{}"),
            observed_at=row["observed_at"],
            version=int(row["version"]),
        )

    def all_records(self, kind: str | None = None) -> list[Record]:
        query = "SELECT * FROM records WHERE retired = 0"
        params: tuple[Any, ...] = ()
        if kind:
            query += " AND kind = ?"
            params = (kind,)
        with self._cursor() as conn:
            # Materialise immediately: a live cursor would hold the file open.
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get(self, record_id: str) -> Record | None:
        with self._cursor() as conn:
            row = conn.execute(
                "SELECT * FROM records WHERE record_id = ?", (record_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def history(self, record_id: str) -> list[dict[str, Any]]:
        with self._cursor() as conn:
            rows = conn.execute(
                "SELECT version, snapshot, recorded_at FROM record_versions "
                "WHERE record_id = ? ORDER BY version DESC",
                (record_id,),
            ).fetchall()
        return [
            {
                "version": r["version"],
                "recorded_at": r["recorded_at"],
                "snapshot": json.loads(r["snapshot"]),
            }
            for r in rows
        ]

    def events(self, since: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        query = "SELECT * FROM events"
        params: list[Any] = []
        if since:
            query += " WHERE happened_at >= ?"
            params.append(since)
        query += " ORDER BY happened_at DESC, id DESC LIMIT ?"
        params.append(limit)
        with self._cursor() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [
            {
                "event_type": r["event_type"],
                "record_id": r["record_id"],
                "kind": r["kind"],
                "title": r["title"],
                "detail": r["detail"],
                "source_url": r["source_url"],
                "happened_at": r["happened_at"],
            }
            for r in rows
        ]

    def data_version(self) -> str:
        with self._cursor() as conn:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'data_version'"
            ).fetchone()
        return row["value"] if row else "0"

    def counts(self) -> dict[str, int]:
        with self._cursor() as conn:
            total = conn.execute(
                "SELECT COUNT(*) c FROM records WHERE retired = 0"
            ).fetchone()["c"]
            by_kind = conn.execute(
                "SELECT kind, COUNT(*) c FROM records WHERE retired = 0 GROUP BY kind"
            ).fetchall()
            events = conn.execute("SELECT COUNT(*) c FROM events").fetchone()["c"]
            versions = conn.execute(
                "SELECT COUNT(*) c FROM record_versions"
            ).fetchone()["c"]
        return {
            "records": total,
            "events": events,
            "superseded_versions": versions,
            "by_kind": {r["kind"]: r["c"] for r in by_kind},
        }

    def last_ingested_at(self) -> str | None:
        with self._cursor() as conn:
            row = conn.execute(
                "SELECT MAX(observed_at) m FROM records"
            ).fetchone()
        return row["m"] if row else None
