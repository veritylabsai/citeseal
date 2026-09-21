"""Query layer: the thing an agent actually calls.

THE GUARANTEE, STRUCTURALLY
    ``Answer`` validates itself. ``found=True`` with no citation, or with a
    result missing ``source_url``, raises. ``found=False`` carrying results also
    raises. So the contract is not a convention callers must respect -- an
    uncited answer cannot be constructed, and therefore cannot be returned.

    Everything here is a lookup. There is no model, no summarisation, no
    paraphrase. A result is the stored record or it does not exist.

WHY THE INDEX IS CACHED
    Tokenising a corpus per request is the difference between ~0.4ms and ~200ms
    on a realistic corpus. On a public endpoint that gap is a remote CPU-burn, so
    the prepared index is cached per store and invalidated by the store's
    monotonic data version.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Iterable

from .matching import Index, token_set
from .record import BODY_MAX, Record, UncitedRecordError
from .store import Store

__all__ = ["Answer", "Engine", "MAX_QUERY_CHARS"]

# Bound query input: it goes straight into tokenisation, so an unbounded string
# is a cheap way to burn CPU.
MAX_QUERY_CHARS = 512

_INDEX_CACHE: dict[str, tuple[str, Index]] = {}
_INDEX_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class Answer:
    """The only shape the engine returns.

    Invariants are enforced here so that no caller, and no future refactor of a
    query path, can produce an uncited answer.
    """

    found: bool
    query: str
    results: tuple[dict[str, Any], ...] = ()
    reason: str | None = None
    note: str = ""
    searched_kinds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.found:
            if not self.results:
                raise ValueError("Answer(found=True) must carry at least one result")
            for result in self.results:
                if not result.get("source_url"):
                    raise UncitedRecordError(
                        "Answer(found=True) carried a result with no source_url; "
                        "this is the one outcome the framework forbids"
                    )
        else:
            if self.results:
                raise ValueError(
                    "Answer(found=False) must not carry results; an explicit "
                    "negative is not a partial match"
                )
            if not self.reason:
                raise ValueError("Answer(found=False) must state a reason")

    def as_dict(self) -> dict[str, Any]:
        return {
            "found": self.found,
            "query": self.query,
            "reason": self.reason,
            "count": len(self.results),
            "results": list(self.results),
            "searched_kinds": list(self.searched_kinds),
            "note": self.note,
        }


NO_RECORD_NOTE = (
    "Absence here means 'not in this store', not a negative claim about the "
    "world. Confirm against the cited source before acting."
)


class Engine:
    def __init__(self, store: Store):
        self.store = store
        self._cache_key = str(store.path)

    # -- index -------------------------------------------------------------

    def _kinds(self) -> tuple[str, ...]:
        """Distinct record kinds, cached by data version.

        This used to call ``store.counts()``, which runs four aggregate queries
        including full-table COUNTs over records, events and versions. Doing that
        on every search cost ~24ms per query at 50k records for an answer that
        almost never changes -- measured, not guessed. The kinds are a property of
        the data version, so they are cached with it.
        """
        version = self.store.data_version()
        cache_key = f"{self._cache_key}|__kinds__"
        with _INDEX_LOCK:
            cached = _INDEX_CACHE.get(cache_key)
            if cached is not None and cached[0] == version:
                return cached[1]
        kinds = tuple(sorted(self.store.counts()["by_kind"]))
        with _INDEX_LOCK:
            _INDEX_CACHE[cache_key] = (version, kinds)
        return kinds

    def index(self, kind: str | None = None) -> Index:
        """Prepared index for one kind (or all kinds), cached by data version."""
        version = self.store.data_version()
        cache_key = f"{self._cache_key}|{kind or '*'}"

        with _INDEX_LOCK:
            cached = _INDEX_CACHE.get(cache_key)
            if cached is not None and cached[0] == version:
                return cached[1]

        records = self.store.all_records(kind=kind)
        prepared = [
            (
                # Only what scoring and deterministic ordering need. The record
                # body is deliberately NOT cached: returned results are read back
                # from the store, so they are verbatim by construction.
                #
                # An earlier version cached a 600-character prefix of the body and
                # returned that prefix as the result. Every result longer than 600
                # characters was therefore silently truncated -- a direct
                # violation of the one guarantee this framework exists to provide,
                # and invisible because every test corpus had short bodies. Live
                # data exposed it on the first run.
                {"record_id": r.record_id, "kind": r.kind, "title": r.title},
                token_set(r.index_text),
            )
            for r in records
        ]
        index = Index(prepared, version=version)

        with _INDEX_LOCK:
            _INDEX_CACHE[cache_key] = (version, index)
        return index

    # -- queries -----------------------------------------------------------

    def search(self, query: str, kind: str | None = None, limit: int = 10) -> Answer:
        cleaned = (query or "").strip()[:MAX_QUERY_CHARS]
        kinds = (kind,) if kind else self._kinds()

        if not cleaned:
            return Answer(
                found=False,
                query="",
                reason="empty query",
                note="Supply a search string.",
                searched_kinds=kinds,
            )

        scored = self.index(kind).search(cleaned, limit=max(1, min(limit, 100)))

        # Hydrate every hit from the store rather than trusting the cached copy.
        # This is what makes "results are stored records, verbatim" true by
        # construction rather than by keeping two representations in step.
        results = []
        for hit in scored:
            stored = self.store.get(hit.payload["record_id"])
            if stored is None:
                # Withdrawn between the index being built and this query. It is
                # no longer in the corpus, so it is not a result.
                continue
            payload = stored.as_dict()
            payload["score"] = hit.score
            payload["matched_terms"] = list(hit.matched)
            results.append(payload)

        if not results:
            return Answer(
                found=False,
                query=cleaned,
                reason="no record in the store matches this query",
                note=NO_RECORD_NOTE,
                searched_kinds=kinds,
            )

        return Answer(
            found=True,
            query=cleaned,
            results=tuple(results),
            searched_kinds=kinds,
            note="Every result is a stored record with a resolvable source_url.",
        )

    def verify(self, query: str, limit: int = 10) -> Answer:
        """Check a claim against the whole store.

        Deliberately just ``search`` across all kinds: a separate verification
        path that could disagree with search would be one more place for the two
        to drift, and the value here is that there is exactly one lookup.
        """
        return self.search(query, kind=None, limit=limit)

    def get(self, record_id: str) -> Answer:
        record = self.store.get(record_id)
        if record is None:
            return Answer(
                found=False,
                query=record_id,
                reason="no record with that id",
                note=NO_RECORD_NOTE,
            )
        return Answer(
            found=True,
            query=record_id,
            results=(record.as_dict(),),
            note="Stored record with its citation.",
        )

    def changes(self, since: str | None = None, limit: int = 100) -> dict[str, Any]:
        """The change feed. Not an Answer: it is a list, not a claim."""
        events = self.store.events(since=since, limit=max(1, min(limit, 500)))
        return {
            "since": since,
            "count": len(events),
            "changes": events,
            "note": "New and superseded records, each with its source URL.",
        }
