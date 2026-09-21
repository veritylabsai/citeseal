# Citeseal

**Cited-or-nothing retrieval.** Turn any corpus into a ground-truth MCP server
that cannot invent an answer.

```python
from citeseal import Citation, Engine, Record, Store

store = Store("corpus.sqlite3")
store.upsert(Record(
    key="northwind-aurelia-kettle",
    kind="recall",
    title="Northwind Recalls Aurelia Kettle Due to Burn Hazard",
    body="The kettle's lid can detach during pouring, releasing scalding water.",
    citation=Citation(
        url="https://example.gov/recalls/northwind-aurelia-kettle",
        source_name="Consumer Product Safety Commission",
    ),
))

engine = Engine(store)
engine.verify("kettle burn hazard").found        # True  -> cited records
engine.verify("something not in the corpus")     # False -> explicit negative
```

## The one idea

Most retrieval systems answer with *something*. Citeseal answers with **a stored
record or nothing at all**. There is no model in the query path, no
summarisation, no paraphrase.

That rule is not a convention you have to remember. It is enforced in the types:

```python
Citation(url="", source_name="X")          # raises UncitedRecordError
Record(..., citation={"url": "..."})       # raises UncitedRecordError
Answer(found=True, results=({},))          # raises UncitedRecordError
```

No construction path produces an uncited record, so no query can return one.
`Answer` also refuses to be `found=False` *and* carry results — an explicit
negative is not a partial match.

## Conformance, not vibes

A corpus either honours the guarantees or it does not, and that is a decidable
question. So it is a command:

```console
$ python -m citeseal.conformance --db corpus.sqlite3

Citeseal conformance report
  corpus: 6 records  kinds: {'definition': 3, 'recall': 3}

  [PASS] G1  every record carries a resolvable citation
  [PASS] G2  no match returns an explicit negative
  [PASS] G3  results are stored records, not generated text
  [PASS] G4  a positive answer cites every result
  [PASS] G5  re-ingesting unchanged data emits no events
  [PASS] G6  a changed record is superseded, not overwritten

RESULT: PASS
```

**G3 is the interesting one.** It re-queries the corpus and compares each result
byte-for-byte against the store. If any result text differs, something
synthesised it and the corpus fails. That is the anti-hallucination property,
mechanically checked rather than promised.

Exit code is 0 on pass. Wire it into CI and the guarantee cannot silently rot.

## Why a framework rather than a library

The engine is small — a few hundred lines. The value is not in the code; it is in
the properties the code makes unavoidable. A second corpus has to implement
**one thing**:

```python
class MyCorpus:
    key = "my-corpus"
    name = "My Upstream"

    def fetch(self) -> Iterable[Record]:
        rows = fetch_my_upstream()
        return build_records(rows, self._to_record, on_error=self.errors.append)

    def _to_record(self, raw) -> Record:
        return Record(
            key=raw["id"],
            kind="notice",
            title=raw["title"],
            body=raw["text"],
            citation=Citation(url=raw["url"], source_name=raw["authority"]),
        )
```

Storage, indexing, matching, the change feed, negative reporting and citation
enforcement are the framework's problem, not the adapter's. `examples/demo.py`
builds two corpora with **no fields and no domain in common** — a structured
recall feed and a curated glossary — and both pass all six guarantees.

## Install

```console
pip install citeseal              # core: no dependencies at all
pip install "citeseal[serve]"     # optional: FastAPI + MCP serving
```

The core deliberately has zero dependencies. Storage is SQLite from the standard
library. A framework whose selling point is auditability should be installable
and readable without a dependency tree.

## CLI

```console
citeseal ingest  --db corpus.sqlite3 --source mycorpus.sources:MyCorpus
citeseal query   --db corpus.sqlite3 "kettle burn hazard"
citeseal changes --db corpus.sqlite3 --since 2026-01-01T00:00:00Z
citeseal check   --db corpus.sqlite3
citeseal stats   --db corpus.sqlite3
```

`query` exits non-zero when nothing matches, so it composes in a shell.

## What this is not

Worth being blunt, because the category invites overclaiming:

- **Not a model, and not RAG.** There is no generation step to ground. If you
  want an LLM to write prose over a corpus, this is not that, and it will not
  pretend to be.
- **Not a guarantee about the world.** `found: false` means *not in this store*,
  never *not true*. Every negative response says so in the payload. A well-built
  corpus can still be wrong, stale, or incomplete — Citeseal makes it
  **auditable**, not infallible.
- **Not a cure for a bad corpus.** It enforces that every record has a citation.
  It cannot tell you whether the citation supports the claim. That judgement
  stays with whoever curates the corpus.
- **Not an embedder.** Matching is IDF-weighted token overlap: fast, inspectable,
  dependency-free, and weaker than a vector index on paraphrase. It was chosen so
  that a result can always be explained in terms of terms that matched. A vector
  backend is a reasonable future addition, not a missing feature.

## Design notes

Five bugs found while extracting this from a production service, each of which
would have been much harder to find later, are documented in
[`docs/design.md`](docs/design.md):

- per-request corpus rebuilds turning a sub-millisecond query into ~200ms
- an absolute distinctiveness threshold that made small corpora unmatchable
- a `with sqlite3.connect(...)` that leaked a file handle, because it commits but
  does not close
- **writes that were never committed** — invisible to in-process tests, which
  read back through the same open connection; only a second process revealed it
- a generator that silently dropped a whole corpus when one row was malformed

The missing-commit bug is why the CLI is exercised as a separate surface. A test
suite that only ever touches one long-lived process is not testing persistence.

## Tests

No test framework required:

```console
python tests/test_guarantees.py   # 19 checks
python examples/demo.py           # two corpora, built and checked
```

## Licence

MIT. The durable asset is a curated, refreshed, cited corpus — not this code.
Keeping the engine permissively licensed is deliberate.
