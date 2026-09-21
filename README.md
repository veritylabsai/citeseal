# Citeseal

**Retrieval that returns a source, or nothing at all.**

Ask a normal search system a question it has no good answer to and it still
returns its closest matches — and an LLM on top will write a confident paragraph
about them. That is how an agent ends up telling someone a product isn't
recalled when it simply has no record either way.

Citeseal is the opposite by construction. You point it at a corpus. It stores
cited records and, at query time, returns **those records or an explicit
"no verified record"**. There is no model in the query path, so it cannot
paraphrase, summarise, or invent — it has no mechanism for any of it.

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

engine.verify("kettle burn hazard").results[0]["source_url"]
# 'https://example.gov/recalls/northwind-aurelia-kettle'

engine.verify("lithium battery fire")
# found=False  reason='no record in the store matches this query'
```

That second result is the product. A weaker system would have returned the
kettle anyway, because it is the only thing in the corpus.

## Is this for you?

**Use it if** an agent under your control answers questions where being wrong is
expensive — compliance, safety, policy, eligibility, anything with a regulator
or a lawyer behind it — and you can point it at an authoritative source.

**Don't use it if** you want the model to write prose over documents. This is not
RAG and does not pretend to be. It never generates text. See
[What this is not](#what-this-is-not).

## How it works

```
  upstream API ──▶  Source  ──▶  Store  ──▶  Index  ──▶  Engine  ──▶  Answer
   (any shape)     you write    SQLite      cached      lookup     found, or
                                  │                              not found
                                  └──▶  change feed (what moved, and when)
```

You write one class — a `Source` that fetches from your upstream and yields
`Record`s. Everything else is the framework's job.

Three objects matter:

| | |
|---|---|
| **`Record`** | One unit of ground truth: a key, a kind, a title, searchable text, and a **citation**. |
| **`Source`** | How your upstream becomes `Record`s. The only thing a new corpus implements. |
| **`Answer`** | What a query returns: cited results, or an explicit negative with a reason. |

## The rule is enforced, not promised

Most "no hallucination" claims are a system prompt and a hope. Here it is a type
invariant, checked when objects are constructed:

```python
Citation(url="", source_name="X")        # UncitedRecordError
Citation(url="ftp://x/y", source_name="X")  # UncitedRecordError
Record(..., citation={"url": "..."})     # UncitedRecordError (needs a Citation)
Answer(found=True, results=({},))        # UncitedRecordError (result has no source)
Answer(found=False, results=(...))       # ValueError (a negative carries no results)
```

There is no constructor that skips these, so no query path can return an uncited
answer. You cannot forget to check, because you cannot build the object.

## Prove it on your corpus, in one command

A corpus either honours the guarantees or it does not, and that is decidable — so
it is a command rather than a paragraph:

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

**G3 is the one that matters.** It re-queries your corpus and compares every
result byte-for-byte against the store. If any result text differs, something
synthesised it and the corpus fails. That is the anti-hallucination property,
mechanically checked.

Exit code is 0 on pass, so it drops straight into CI and cannot silently rot.

## Building a corpus

Implement one class:

```python
from citeseal import Citation, Record
from citeseal.sources import build_records

class MyCorpus:
    key = "my-corpus"
    name = "My Upstream"

    def fetch(self):
        rows = fetch_my_upstream()            # whatever your API returns
        return build_records(rows, self._to_record, on_error=self.errors.append)

    def _to_record(self, raw) -> Record:
        return Record(
            key=raw["id"],
            kind="notice",
            title=raw["title"],
            body=raw["text"],                 # the text that gets searched
            citation=Citation(url=raw["url"], source_name=raw["authority"]),
        )
```

Then:

```console
citeseal ingest  --db corpus.sqlite3 --source mycorpus.sources:MyCorpus
citeseal query   --db corpus.sqlite3 "kettle burn hazard"
citeseal check   --db corpus.sqlite3
```

`build_records` is not decoration. A `fetch()` generator that raises on one bad
row **terminates**, silently dropping every row after it — my first version lost a
whole corpus to one apostrophe in a key. `build_records` skips and reports the bad
row instead.

### It has to work for more than one corpus

A framework that only ever served one corpus is a library with delusions. So the
repository proves it: `examples/demo.py` builds two corpora with **no fields and
no domain in common** — a structured recall feed and a curated glossary — plus a
combined registry, and all three pass every guarantee.

```console
$ python examples/demo.py
...
ALL CORPORA PASSED
```

## Install

```console
pip install citeseal              # core: zero dependencies
pip install "citeseal[serve]"     # optional: FastAPI + MCP serving
```

The core has no runtime dependencies. Storage is SQLite from the standard
library; the only network call is `urllib`. A framework whose selling point is
auditability should be installable and readable without a dependency tree.

## What this is not

Blunt, because the category invites overclaiming:

- **Not a model, and not RAG.** There is no generation step to ground. If you
  want an LLM writing prose over documents, this is not that.
- **Not a claim about the world.** `found: false` means *not in this store*,
  never *not true*. Every negative says so in the payload, and so should your
  agent.
- **Not a cure for a bad corpus.** It enforces that a citation *exists*. It
  cannot check that the citation *supports* the claim. That judgement stays with
  whoever curates the corpus — and it is the harder half.
- **Not a vector search.** Matching is IDF-weighted token overlap: fast,
  inspectable, dependency-free, and explainable in terms of the terms that
  matched. It is weaker than embeddings on paraphrase. A vector backend is a
  reasonable future addition, not a missing feature.
- **Not English-only.** The tokeniser is Unicode-aware, but the stemmer is
  English-oriented. Other languages index and match correctly, just without
  stemming.

## Status

`0.1.0`. Storage, matching, guarantees and the conformance suite are built and
tested against a real upstream (OSV, ~400 live vulnerabilities) as well as
synthetic corpora. **MCP and HTTP serving is not written yet** — the `[serve]`
extra is a placeholder, and until it lands Citeseal is a library plus a CLI, not
a network service.

### Known limit: query time scales with corpus size

Scoring is O(documents matching the query), and a common term matches a large
fraction of a large corpus. Measured on this machine, warm cache:

| records | index build | mean query | worst |
|---|---|---|---|
| 5,000 | 0.1s | 5 ms | 8 ms |
| 14,000 | 0.3s | 16 ms | 37 ms |
| 20,000 | 0.4s | 25 ms | 49 ms |
| 50,000 | 1.2s | 65 ms | 144 ms |
| 100,000 | 2.8s | 163 ms | 255 ms |

Comfortable to roughly 50k records. Beyond that, query latency grows linearly and
would need candidate pruning or a vector backend. It is stated here rather than
discovered by you. Note the index is built once per process and reused — the
numbers above do not include a rebuild, and a rebuild is what the original
production bug was.

## Tests

No test framework required. 73 checks across five suites:

```console
python tests/test_guarantees.py    # 20 checks: the promises
python tests/test_adversarial.py   # 26 checks: trying to break it
python tests/test_stress.py        # 10 checks: fuzzing and scale
python tests/test_real_corpus.py   #  9 checks: a real upstream, replayed
python tests/test_readme_claims.py #  8 checks: this README is not lying
python examples/demo.py            # two corpora, built and checked
```

Plus a live test that needs the network and hits the real OSV API:

```console
python tests/live_smoke.py --no-save
```

What the non-obvious suites actually do:

- **Adversarial** — SQL injection through keys and queries, Unicode keys and
  text, eight concurrent writers with no lost writes, readers staying consistent
  under concurrent mutation, a corrupt database failing loudly, simulated
  crashes, boundary sizes, determinism.
- **Stress** — 4,000 fuzz cases against citations, records, queries and the
  `Answer` invariant with a fixed seed so failures reproduce; plus a scale run at
  20k (and 100k with `--big`).
- **Real corpus** — replays payloads recorded from live OSV, so the adapter is
  tested against the upstream's actual shape rather than one its author invented.
  Includes the hard rule: an OSV entry with no reference must be refused, not
  admitted without a citation.
- **README claims** — executes the examples on this page against the real
  library, asserts every row of the type-guarantee table raises, and fails if a
  stated test count drifts. It caught two false statements in my own draft.

CI runs everything on Linux across Python 3.10–3.13 plus one Windows leg, checks
the core has no third-party imports, and runs the live smoke test nightly.

## Design notes

Bugs found while building this, with the reasoning, are in
[`docs/design.md`](docs/design.md). Four worth knowing before you build on it:

- A missing `commit()` discarded **every write** at process exit. In-process
  tests could not detect it, because they read back through the same open
  connection where uncommitted rows are visible.
- The index cached a 600-character prefix of each record body and returned it, so
  **every result longer than 600 characters was silently truncated** — a direct
  violation of the guarantee this framework exists to provide. Every test corpus
  had short bodies; live data caught it on the first run.
- Unseen query terms were weighted zero, so **coverage counted missing terms as
  matches** and an unrelated query scored higher than a real one.
- Every search ran full-table `COUNT`s to discover the record kinds, costing
  ~24 ms per query at 50k records for an answer that changes only on write.

## Licence

MIT. The durable asset in a system like this is a curated, current, cited corpus
— not the engine. Keeping the engine permissive is deliberate.
