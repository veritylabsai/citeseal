"""The cited record: the only kind of thing this framework will ever return.

THE HARD RULE
    A record without a usable citation cannot exist. It is enforced in
    ``Citation.__post_init__`` and ``Record.__post_init__``, so it holds for
    every construction path -- there is no builder, loader or adapter that can
    bypass it. Callers do not have to remember to check; the type refuses.

That is the whole point of the framework. Everything else here is plumbing to
make a corpus queryable; this file is the guarantee.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

__all__ = [
    "Citation",
    "Record",
    "UncitedRecordError",
    "utcnow",
    "SHORT_MAX",
    "BODY_MAX",
]

# Bounds. An ingest adapter that pulls a 40MB HTML blob into `body` would make
# every query slow for everyone, so the store refuses it at the door rather than
# discovering the problem under load.
SHORT_MAX = 300
BODY_MAX = 20_000

# Keys come from upstream data, so this is deliberately permissive: it only
# rules out control characters, leading/trailing space and absurd length. An
# earlier, stricter allow-list (letters, digits, dot, dash, colon) rejected
# legitimate values like "children's-product-certificate", and because a source's
# fetch() is a generator, one rejected key aborted the rest of the corpus.
_KEY_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,200}$")


class UncitedRecordError(ValueError):
    """A record or citation was constructed without a usable source.

    Deliberately loud: this is the one invariant the framework exists to
    protect, and a silent fallback here would defeat the entire product.
    """


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_url(url: Any) -> str:
    if not isinstance(url, str) or not url.strip():
        raise UncitedRecordError("citation url must be a non-empty string")
    cleaned = url.strip()
    try:
        parsed = urlparse(cleaned)
    except ValueError as exc:
        # urlparse raises a bare ValueError for some malformed inputs, notably
        # anything with an unbalanced '[' ("Invalid IPv6 URL"). Callers are told
        # to catch UncitedRecordError for a bad citation, so letting a different
        # type escape here would break that contract -- and it did, until fuzzing
        # found it.
        raise UncitedRecordError(f"citation url could not be parsed: {cleaned!r}") from exc
    if parsed.scheme not in ("http", "https"):
        raise UncitedRecordError(
            f"citation url must be http(s), got {cleaned!r}"
        )
    if not parsed.netloc:
        raise UncitedRecordError(f"citation url has no host: {cleaned!r}")
    return cleaned


@dataclass(frozen=True, slots=True)
class Citation:
    """Where a record came from. The url is non-optional by construction."""

    url: str
    source_name: str
    text: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "url", _require_url(self.url))
        if not isinstance(self.source_name, str) or not self.source_name.strip():
            raise UncitedRecordError("citation source_name must be a non-empty string")
        object.__setattr__(self, "source_name", self.source_name.strip())
        if self.text is not None:
            object.__setattr__(self, "text", str(self.text)[:BODY_MAX])

    def as_dict(self) -> dict[str, Any]:
        return {"url": self.url, "source_name": self.source_name, "text": self.text}


@dataclass(frozen=True, slots=True)
class Record:
    """One unit of ground truth.

    ``kind`` is source-defined ('recall', 'requirement', 'policy', ...) so the
    framework does not have to know any domain. ``body`` is the text that gets
    indexed and matched; ``attributes`` carries whatever structured fields the
    source has, and is returned verbatim rather than interpreted.
    """

    key: str
    kind: str
    title: str
    body: str
    citation: Citation
    attributes: dict[str, Any] = field(default_factory=dict)
    observed_at: str = field(default_factory=utcnow)
    version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not _KEY_RE.match(self.key):
            raise ValueError(
                f"record key must be 1-200 printable characters with no leading "
                f"or trailing space, got {self.key!r}"
            )
        if self.key != self.key.strip():
            raise ValueError(f"record key must not have surrounding whitespace: {self.key!r}")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("record kind must be a non-empty string")
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("record title must be a non-empty string")
        if not isinstance(self.body, str):
            raise ValueError("record body must be a string (may be empty)")
        if len(self.body) > BODY_MAX:
            raise ValueError(f"record body exceeds {BODY_MAX} characters")
        if not isinstance(self.citation, Citation):
            # A dict here is the most likely mistake, and accepting it would
            # quietly move citation validation to some later, less safe place.
            raise UncitedRecordError(
                "record requires a Citation instance, not "
                f"{type(self.citation).__name__}"
            )
        if not isinstance(self.attributes, dict):
            raise ValueError("record attributes must be a dict")

    # -- identity ----------------------------------------------------------

    @property
    def record_id(self) -> str:
        """Globally unique within a store. Kind is part of identity so two
        sources can legitimately use the same key."""
        return f"{self.kind}:{self.key}"

    @property
    def content_hash(self) -> str:
        """Changes when the record's substance changes.

        Excludes ``observed_at``: re-fetching an unchanged record from upstream
        must NOT look like a change, or every refresh would emit events and the
        change feed would be useless.
        """
        payload = json.dumps(
            {
                "kind": self.kind,
                "title": self.title,
                "body": self.body,
                "citation": self.citation.as_dict(),
                "attributes": self.attributes,
            },
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def index_text(self) -> str:
        """Text handed to the matcher.

        The key is included: for a corpus where keys are product ids, model
        numbers or statute references, a user searching for exactly that
        identifier is the most obvious query there is, and leaving the key
        unindexed made it unanswerable. Titles are repeated because a title match
        is a stronger signal than one buried in the body.
        """
        return f"{self.key} {self.title} {self.title} {self.body}".strip()

    def as_dict(self) -> dict[str, Any]:
        """The only shape this framework shows to a caller."""
        return {
            "record_id": self.record_id,
            "key": self.key,
            "kind": self.kind,
            "title": self.title,
            "body": self.body,
            "source_url": self.citation.url,
            "source_name": self.citation.source_name,
            "citation_text": self.citation.text,
            "attributes": self.attributes,
            "observed_at": self.observed_at,
            "version": self.version,
        }

    def with_version(self, version: int, observed_at: str | None = None) -> "Record":
        return Record(
            key=self.key,
            kind=self.kind,
            title=self.title,
            body=self.body,
            citation=self.citation,
            attributes=self.attributes,
            observed_at=observed_at or utcnow(),
            version=version,
        )
