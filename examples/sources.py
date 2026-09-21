"""Two unrelated corpora, one framework.

This is the whole argument for Citeseal being a framework rather than a library
extracted from one product. The two sources below share no fields, no upstream
shape and no domain:

  * ``RecallFeed``   -- a stream of structured event records (product recalls).
  * ``GlossaryCorpus`` -- a small curated set of definitions, closer to
    documentation than to a feed.

Both become the same thing: cited records that a query layer can return, and a
change feed that reports what is new. Neither source implements storage,
indexing, matching, negative reporting or citation enforcement, because none of
those are source-specific.

Run ``python examples/demo.py`` to build both and check them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from citeseal import Citation, Record
from citeseal.sources import build_records

DATA = Path(__file__).parent / "data"


class RecallFeed:
    """A structured feed. One upstream record becomes one cited Record."""

    key = "demo-recalls"
    name = "Demo Recall Feed"

    def __init__(self, path: Path | None = None):
        self.path = path or (DATA / "recalls.json")
        self.errors: list[str] = []

    def _to_record(self, raw: dict) -> Record:
        # Normalisation is the adapter's job: deciding that hazard+remedy is
        # the searchable body is a judgement about this source, not something
        # a generic layer could make.
        body = " ".join(
            part for part in (raw.get("hazard"), raw.get("remedy")) if part
        )
        return Record(
            key=raw["id"],
            kind="recall",
            title=raw["title"],
            body=body,
            citation=Citation(
                url=raw["url"],
                source_name=raw["agency"],
                text=raw.get("hazard"),
            ),
            attributes={
                "recall_date": raw.get("date"),
                "units": raw.get("units"),
            },
        )

    def fetch(self) -> Iterable[Record]:
        rows = json.loads(self.path.read_text(encoding="utf-8"))
        # build_records so one malformed row cannot cost us the whole feed.
        return build_records(rows, self._to_record, on_error=self.errors.append)


class GlossaryCorpus:
    """A curated glossary. Different shape entirely -- no dates, no agency feed,
    and the whole value is in the definition text."""

    key = "demo-glossary"
    name = "Demo Compliance Glossary"

    def __init__(self, path: Path | None = None):
        self.path = path or (DATA / "glossary.json")
        self.errors: list[str] = []

    def _to_record(self, raw: dict) -> Record:
        # Keys must be stable and URL-safe; upstream terms are free text, so the
        # adapter slugifies rather than trusting the term to be well behaved.
        slug = "".join(
            ch if ch.isalnum() else "-" for ch in raw["term"].lower()
        ).strip("-")
        while "--" in slug:
            slug = slug.replace("--", "-")
        return Record(
            key=slug,
            kind="definition",
            title=raw["term"],
            body=raw["definition"],
            citation=Citation(
                url=raw["source"],
                source_name=raw["authority"],
                text=raw["definition"],
            ),
            attributes={"market": raw.get("market")},
        )

    def fetch(self) -> Iterable[Record]:
        rows = json.loads(self.path.read_text(encoding="utf-8"))
        return build_records(rows, self._to_record, on_error=self.errors.append)
