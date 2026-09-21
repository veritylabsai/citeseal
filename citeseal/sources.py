"""Corpus adapters: how an upstream becomes a set of cited records.

A ``Source`` is the only thing a new corpus has to implement. It fetches from
somewhere and yields ``Record`` objects; the framework handles storage, indexing,
matching, the change feed and serving.

The adapter owns normalisation, because normalisation is inherently
source-specific -- deciding which upstream field is the "body" is a judgement
about that source, not something a generic layer can make.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Protocol, runtime_checkable

from .record import Record, utcnow
from .store import Store

__all__ = ["Source", "IngestReport", "ingest", "SourceRegistry", "build_records"]


def build_records(
    rows: Iterable[Any],
    factory: Any,
    *,
    on_error: Any = None,
) -> Iterator[Record]:
    """Turn raw upstream rows into Records, skipping rows that cannot be built.

    THIS EXISTS BECAUSE OF A REAL FAILURE. ``Source.fetch`` is a generator, so an
    exception raised while building row 1 of 10,000 does not skip row 1 -- it
    terminates the generator, and rows 2..10,000 are silently lost. A single
    record with, say, a missing source URL would empty the corpus.

    Adapters should therefore build through this helper rather than raising:

        def fetch(self):
            rows = fetch_upstream()
            return build_records(rows, self._to_record, on_error=self.errors.append)

    ``on_error`` receives the underlying exception, so an adapter can surface
    per-row problems in its own report instead of hiding them.
    """
    for row in rows:
        try:
            yield factory(row)
        except Exception as exc:  # noqa: BLE001 - deliberate: one bad row must not
            # cost the other 9,999. The exception is reported, never swallowed.
            if on_error is not None:
                on_error(exc)
            continue


@runtime_checkable
class Source(Protocol):
    """Anything that can produce cited records."""

    key: str
    name: str

    def fetch(self) -> Iterable[Record]:
        """Yield records. Must not yield anything without a citation -- and
        cannot, since ``Record`` refuses to exist without one."""
        ...


@dataclass(slots=True)
class IngestReport:
    source_key: str
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    rejected: int = 0
    errors: list[str] = field(default_factory=list)
    started_at: str = field(default_factory=utcnow)
    finished_at: str | None = None

    @property
    def total(self) -> int:
        return self.new + self.changed + self.unchanged

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_key": self.source_key,
            "new": self.new,
            "changed": self.changed,
            "unchanged": self.unchanged,
            "rejected": self.rejected,
            "total": self.total,
            "errors": self.errors[:20],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def ingest(store: Store, source: Source, *, url: str | None = None) -> IngestReport:
    """Fetch a source and merge it into the store.

    Idempotent and incremental, so the correct response to a partial failure is
    simply to run it again: records already written stay written, and
    re-ingesting unchanged data is a no-op.
    """
    report = IngestReport(source_key=source.key)
    iterator = iter(source.fetch())
    while True:
        try:
            record = next(iterator)
        except StopIteration:
            break
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            # A generator cannot be resumed after it raises, so anything not yet
            # yielded is lost. Adapters that care use build_records() to skip bad
            # rows instead; this is the framework refusing to hide the problem.
            report.errors.append(
                f"source aborted mid-stream ({type(exc).__name__}: {exc}); "
                "remaining records were not ingested"
            )
            break
        try:
            outcome = store.upsert(record)
        except Exception as exc:  # noqa: BLE001
            report.rejected += 1
            report.errors.append(f"rejected {getattr(record, 'key', '?')!r}: {exc}")
            continue
        if outcome.status == "new":
            report.new += 1
        elif outcome.status == "changed":
            report.changed += 1
        else:
            report.unchanged += 1

    report.finished_at = utcnow()
    store.register_source(source.key, source.name, url, report.total)
    return report


class SourceRegistry:
    """A named set of sources, so a service can be declared rather than coded."""

    def __init__(self) -> None:
        self._sources: dict[str, Source] = {}

    def register(self, source: Source) -> Source:
        if not source.key:
            raise ValueError("source.key must be non-empty")
        self._sources[source.key] = source
        return source

    def get(self, key: str) -> Source | None:
        return self._sources.get(key)

    def keys(self) -> list[str]:
        return sorted(self._sources)

    def all(self) -> list[Source]:
        return [self._sources[k] for k in self.keys()]

    def ingest_all(self, store: Store) -> list[IngestReport]:
        return [ingest(store, source) for source in self.all()]
