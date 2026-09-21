# Design notes

Citeseal was not designed on a whiteboard. It was extracted from a working
service (a product-recall lookup) after that service had been in production long
enough to fail in interesting ways. The interesting failures are recorded here,
because each one is a trap that any corpus built on this framework would
otherwise have to rediscover.

## 1. Rebuilding the corpus per request

**Symptom.** A single-record lookup took ~215ms against a 14,000-record corpus.
On a public, unauthenticated endpoint that is an availability problem and a
trivial remote CPU-burn: a few concurrent clients could saturate the instance.

**Cause.** The naive query path read every record and re-tokenised the entire
corpus on every call. Tokenisation dominated; the actual scoring was noise.

**Fix.** Prepare the index once per process and cache it, keyed by the store path
and invalidated by a monotonic `data_version` counter that every write
increments. Measured result: **~215ms → ~0.4ms**, roughly 500×.

**Why a counter rather than a hash of the data.** Hashing the corpus to decide
whether to rebuild costs a full read — precisely the work being avoided. A
counter bumped inside the same transaction as the write is O(1) and cannot drift.

**Consequence for the cache.** The cache must hold plain data, never live
database objects. An early version cached `sqlite3.Row` instances; each Row holds
a reference to its cursor, which keeps the database file open, which on Windows
prevents the file from being deleted or moved. The cache now materialises dicts.

## 2. A scale-dependent notion of "distinctive"

**Symptom.** A corpus containing exactly one record could not match its own
content. Queries for the only record's own words returned nothing.

**Cause.** Distinctiveness was an *absolute* threshold on inverse document
frequency: a term counted as distinctive only if `idf > 0.7`. In a one-document
corpus, every term appears in the only document, so `idf` for all of them is
near zero, and nothing ever qualified. The bug was invisible on a large corpus
and total on a small one — the worst combination, because tests used a large
corpus and users start small.

**Fix.** Judge distinctiveness by *document frequency relative to corpus size*:
a term is distinctive if `df <= max(1, n * 0.5)`. This behaves identically at 10
records and at 10 million. `test_small_corpus_can_still_match` is the regression
guard.

**The general lesson.** Any threshold expressed in absolute terms encodes an
assumption about corpus size. Those assumptions are usually unstated and often
wrong at the boundaries.

## 3. A connection leak hiding behind a context manager

**Symptom.** Temp-directory cleanup failed with
`PermissionError: [WinError 32] ... being used by another process` on the SQLite
file. Confusing, because the store had been explicitly closed.

**Cause.** This:

```python
with sqlite3.connect(path) as conn:   # does NOT close the connection
    conn.executescript(SCHEMA)
```

A `sqlite3.Connection` used as a context manager is a **transaction** context
manager: it commits or rolls back on exit. It does not close the connection. So
every `Store()` construction leaked a file handle. `Store.close()` then closed
only the *cached per-thread* connection, not the leaked one.

**Fix.** Open explicitly, use `try/finally`, close explicitly. `Store` gained
`close()` and `__enter__`/`__exit__`, and conformance's scratch store closes
before its temporary directory is removed.

**Why this is in the notes.** It is a one-line mistake that produces a resource
leak invisible on Linux (where unlinking an open file is permitted) and fatal on
Windows. Anything that creates files and is tested on both platforms should close
them explicitly rather than relying on a context manager's apparent meaning.

## 4. Writes that were never committed

The most instructive bug of the four, and the one that best justifies the CLI
existing at all.

**Symptom.** `citeseal ingest` reported `3 new, 0 changed` for each of two
sources. `citeseal stats` then reported **0 records**. The stores were empty.

**Cause.** The connection context manager rolled back on exception but never
committed on success:

```python
@contextmanager
def _cursor(self):
    conn = self._connection()
    try:
        yield conn            # writes happen here...
    except Exception:
        conn.rollback()
        raise                 # ...but nothing ever commits on the happy path
```

Every write was discarded when the process exited.

**Why the tests did not catch it.** All 19 tests ran in one process, and they
read back through the *same open connection*. SQLite shows a connection its own
uncommitted rows, so `store.counts()` returned the right answer from data that
had never reached disk. The bug is invisible to any test that does not cross a
process boundary — and it only surfaced because the CLI ingests in one process
and reports in another.

**Fix.** Commit on clean exit; keep the rollback on failure. Plus a regression
test that intentionally crosses the boundary: write, `close()`, reopen, read.

**The general lesson.** A test suite that only ever exercises one long-lived
process is not testing persistence — it is testing an in-memory cache that
happens to be backed by a file. Any durability claim needs a test that closes
and reopens, and ideally one that runs in a separate process.

## 5. A generator that loses the corpus when one row is bad

Found while writing the example adapters, which is exactly what examples are for.

**Symptom.** A three-record glossary ingested zero records.

**Cause.** One term was `Children's Product Certificate`. The adapter built the
key by lowercasing and replacing spaces, producing
`children's-product-certificate`. The key validator used a conservative
allow-list that excluded apostrophes, so `Record(...)` raised.

That alone would have skipped one record. But `Source.fetch` is a **generator**,
and an exception inside a generator terminates it. The raise on row 1 meant rows
2 and 3 were never yielded. One malformed upstream row silently emptied the
corpus.

**Two fixes**, because there were two faults:

1. The key validator was wrong. Keys come from upstream data; the rule now only
   excludes control characters, surrounding whitespace and absurd length. It
   accepts apostrophes, slashes and non-ASCII, which real data contains.
2. The framework was missing an affordance. `build_records()` wraps row-to-record
   construction so a bad row is skipped and reported rather than fatal, and
   `ingest()` now reports a mid-stream abort explicitly instead of letting it
   look like an empty source.

**The general lesson.** A generator is a poor boundary for untrusted input,
because its failure mode is silent truncation rather than a single bad item.
Anywhere a partial failure should be survivable, iterate defensively.

## 6. Coverage that counted missing terms as matches

Found by the adversarial suite, and the most consequential ranking bug of the set.

**Symptom.** Searching `"toy choke hazard"` against a corpus whose only record
was a kettle recall returned that record with a score of **1.083** — higher than
a genuine three-term match. Also, `"lithium battery fire"` returned *something*.

**Cause.** `idf()` returned `0.0` for any term the corpus had never seen:

```python
def idf(self, token):
    return self._idf.get(token, 0.0)      # unseen term -> weight 0
```

Coverage is `matched_weight / total_query_weight`, and both sides are summed over
query terms. For `"toy choke hazard"` against a one-record corpus, `toy` and
`choke` were absent and therefore contributed **zero to the denominator**, while
`hazard` contributed its full weight to the numerator. Coverage came out at
exactly **1.0**.

In other words: the more of your query the corpus had never heard of, the better
your match looked. Since a real user's query almost always contains words absent
from the corpus, this was not an edge case — it was the normal path.

**Fix.** An unseen term is treated as maximally rare (`df = 0`), giving it the
highest possible weight. The same query now scores **0.177** and is rejected.
Coverage became meaningful: it measures what fraction of the *query* was found,
not what fraction of the *found terms* were in the query.

**How it was caught.** Not by a ranking test. Ranking tests use queries built
from corpus vocabulary, where every term is present and the bug is invisible. It
surfaced from a documentation test asserting a negative: *an unrelated query
returns nothing*. Writing down what the system claims and executing that claim is
what exposed it.

## 7. Three smaller gaps found by adversarial testing

**A `key` that could not be searched.** The index was built from title and body
only, so a user searching for a product id, model number or statute reference —
the most obvious query for a keyed corpus — could never match. The key is now
part of the indexed text.

**An ASCII-only tokeniser.** The word pattern was `[a-z0-9]+`, so CJK, Cyrillic
and Arabic text tokenised to *nothing*, and any corpus in those scripts was
permanently unsearchable. Now `\w+` with Unicode semantics. A Japanese key that
previously could not be retrieved at all now round-trips, including across a
reopen.

**No way to retire a record.** A recall can be rescinded upstream, and the corpus
has to reflect that. There was no API for it, so the only option was a raw
`UPDATE` — which does not bump the data version, so the search index kept serving
the withdrawn record. `Store.retire()` and `unretire()` now exist, bump the
version, and emit an event. Reaching into the database bypassing the store is
exactly the kind of shortcut a framework should make unnecessary.

**The general lesson.** Two of these three (unsearchable keys, unsearchable
scripts) were complete functional failures for whole classes of corpus that the
existing tests could never have noticed, because every test used Latin text and
body-only queries. Adversarial tests are worth writing precisely because they
attack the assumptions the happy-path tests were built on.

## Open questions

- **Matching quality.** Token overlap is inspectable and dependency-free but
  weaker than embeddings on paraphrase. The honest framing in the README is that
  a vector backend is a future addition, not a missing feature. Any such backend
  must preserve the property that a result can be explained.
- **Citation sufficiency.** The framework guarantees a citation *exists*. It
  cannot check that the citation *supports* the claim. That remains a curation
  responsibility, and no amount of type enforcement changes it.
- **Staleness.** `observed_at` records when a record was last confirmed, and the
  change feed reports what moved, but nothing yet alerts on a source that has
  gone quiet. For a monitored corpus that is the obvious next feature.
