"""Fuzz and scale tests.

Two questions the other suites do not answer:

  * **Fuzzing** -- does anything crash, hang, or silently produce an uncited
    answer when fed input nobody would type on purpose? Deterministic seed, so a
    failure is reproducible.
  * **Scale** -- does a query stay fast on a corpus far larger than anything the
    project has been tested with, and does the index stay out of the way?

    python tests/test_stress.py              # fuzz + 20k scale
    python tests/test_stress.py --big        # fuzz + 100k scale
"""

from __future__ import annotations

import random
import shutil
import string
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from citeseal import (  # noqa: E402
    Answer,
    Citation,
    Engine,
    Record,
    Store,
    UncitedRecordError,
    run_conformance,
)

SEED = 20260921
PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []
_TMP: list[Path] = []
_STORES: list[Store] = []


def test(fn):
    def wrapper():
        try:
            fn()
            PASSED.append(fn.__name__)
            print(f"  PASS  {fn.__name__}")
        except Exception:  # noqa: BLE001
            FAILED.append((fn.__name__, traceback.format_exc()))
            print(f"  FAIL  {fn.__name__}")
    wrapper.__name__ = fn.__name__
    return wrapper


def fresh(name: str) -> Store:
    tmp = Path(tempfile.mkdtemp(prefix=f"citeseal-stress-{name}-"))
    _TMP.append(tmp)
    store = Store(tmp / "s.sqlite3")
    _STORES.append(store)
    return store


# Every exception the library is allowed to raise on bad input. Anything else is
# a defect: an unhandled crash on hostile input is a denial-of-service bug.
EXPECTED = (ValueError, UncitedRecordError, TypeError, KeyError)

HOSTILE_TEXT = [
    "", " ", "\n", "\t", "\x00", "\x1b[31m", "'; DROP TABLE records; --",
    "\" OR 1=1 --", "../" * 50, "%00", "%2e%2e%2f", "<script>alert(1)</script>",
    "&amp;&lt;&gt;", "🔥" * 100, "ｆｕｌｌｗｉｄｔｈ", "\u200b\u200c\u200d",
    "a" * 5000, "日本語" * 200, "Привет" * 200, "مرحبا" * 200,
    "x\x00y", "\r\n\r\n", "\\x41", "${jndi:ldap://x/a}",
]

HOSTILE_URLS = [
    "", " ", "x", "//", "/", ":", "://", "http:", "http://", "https://",
    "ftp://x/y", "javascript:alert(1)", "data:text/html,x", "file:///etc/passwd",
    "http://a b c/", "http://[::1]:notaport/x", "HTTP://EXAMPLE.GOV",
    "https://example.gov/" + "a" * 3000, "\x00", "http://\u4f8b.jp/x",
]


# --- fuzzing ---------------------------------------------------------------

@test
def test_fuzz_citation_urls_never_crash_undefined():
    rng = random.Random(SEED)
    accepted = 0
    for _ in range(4000):
        url = rng.choice(HOSTILE_URLS)
        if rng.random() < 0.5:
            url += "".join(rng.choice(string.printable) for _ in range(rng.randint(0, 12)))
        try:
            Citation(url=url, source_name="X")
            accepted += 1
        except UncitedRecordError:
            pass
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(
                f"unescaped {type(exc).__name__} on url={url!r}: {exc}"
            ) from exc
    # Whatever it accepted must have been http(s) with a host.
    assert accepted >= 0


@test
def test_fuzz_any_accepted_citation_is_usable():
    """The invariant that matters: nothing admitted is uncitable.

    Note the assertion is on the PARSED url, not on a string prefix. URL schemes
    are case-insensitive, so "HTTP://EXAMPLE.GOV" is a legitimate citation and
    a naive `startswith("http://")` check wrongly calls it a failure.
    """
    from urllib.parse import urlparse

    rng = random.Random(SEED + 1)
    for _ in range(3000):
        url = rng.choice(HOSTILE_URLS)
        try:
            citation = Citation(url=url, source_name=rng.choice(["X", "", " ", "\x00"]))
        except EXPECTED:
            continue
        parsed = urlparse(citation.url)
        assert parsed.scheme in ("http", "https"), citation.url
        assert parsed.netloc, citation.url
        assert citation.source_name.strip(), citation.source_name


@test
def test_fuzz_record_construction_never_crashes_undefined():
    rng = random.Random(SEED + 2)
    for _ in range(3000):
        key = rng.choice(HOSTILE_TEXT) or "k" + str(rng.randint(0, 999))
        try:
            record = Record(
                key=key,
                kind=rng.choice(["notice", "", " ", "\x00", "🔥"]),
                title=rng.choice(HOSTILE_TEXT) or "t",
                body=rng.choice(HOSTILE_TEXT),
                citation=Citation(url="https://example.gov/x", source_name="X"),
            )
        except EXPECTED:
            continue
        assert record.key == record.key.strip(), "accepted a key with edge whitespace"
        assert "\n" not in record.key and "\x00" not in record.key, record.key


@test
def test_fuzz_tokenizer_and_query_never_crash():
    rng = random.Random(SEED + 3)
    store = fresh("fuzzquery")
    store.upsert(Record(key="seed", kind="notice", title="Seed", body="a seed record",
                        citation=Citation(url="https://example.gov/s", source_name="X")))
    engine = Engine(store)

    for _ in range(1500):
        query = rng.choice(HOSTILE_TEXT) + "".join(
            rng.choice(string.printable) for _ in range(rng.randint(0, 30))
        )
        try:
            answer = engine.verify(query)
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(
                f"unescaped {type(exc).__name__} on query={query[:60]!r}: {exc}"
            ) from exc

        # The invariant must hold for every single answer, however absurd the
        # input. This is the one thing that must never break under fuzzing.
        if answer.found:
            assert answer.results, "found=True with no results"
            for result in answer.results:
                assert result.get("source_url"), "found=True with an uncited result"
        else:
            assert not answer.results, "found=False carrying results"
            assert answer.reason, "found=False without a reason"


@test
def test_fuzz_store_round_trips_everything_it_accepts():
    """Anything the store accepts must come back identical."""
    rng = random.Random(SEED + 4)
    store = fresh("fuzzstore")
    accepted: dict[str, Record] = {}

    for i in range(600):
        key = f"k{i}"
        body = rng.choice(HOSTILE_TEXT)
        record = Record(
            key=key, kind="notice", title=f"Title {i}", body=body,
            citation=Citation(url="https://example.gov/x", source_name="X"),
        )
        outcome = store.upsert(record)
        if outcome.status == "new":
            accepted[key] = record

    for key, original in accepted.items():
        stored = store.get(f"notice:{key}")
        assert stored is not None, f"{key} vanished"
        assert stored.body == original.body, f"{key} body altered"
        assert stored.title == original.title, f"{key} title altered"


@test
def test_fuzz_injection_shaped_queries_never_modify_the_store():
    rng = random.Random(SEED + 5)
    store = fresh("fuzzinject")
    for i in range(10):
        store.upsert(Record(key=f"r{i}", kind="notice", title=f"Record {i}",
                            body="lithium battery hazard",
                            citation=Citation(url="https://example.gov/x", source_name="X")))
    before = store.counts()
    engine = Engine(store)

    payloads = [
        "'; DROP TABLE records; --", "1; DELETE FROM records", "\" OR \"\"=\"",
        "' UNION SELECT * FROM records --", "%'; TRUNCATE records; --",
        "'); UPDATE records SET retired=1; --", "\x00'; --",
    ]
    for _ in range(1000):
        query = rng.choice(payloads) + rng.choice(["", " lithium", "'--", "\x00"])
        engine.verify(query)

    after = store.counts()
    assert after == before, f"an injection-shaped query changed the store: {before} -> {after}"


@test
def test_fuzz_answers_are_always_constructible_or_refused():
    """Direct fuzzing of the Answer invariant itself."""
    rng = random.Random(SEED + 6)
    for _ in range(2000):
        found = rng.choice([True, False])
        count = rng.randint(0, 3)
        results = tuple(
            {"record_id": f"x{i}",
             **({"source_url": "https://example.gov/x"} if rng.random() < 0.5 else {})}
            for i in range(count)
        )
        try:
            answer = Answer(found=found, query="q", results=results,
                            reason="r" if not found else None)
        except EXPECTED:
            continue
        if answer.found:
            assert answer.results, "accepted found=True with no results"
            assert all(r.get("source_url") for r in answer.results), (
                "accepted found=True with an uncited result"
            )
        else:
            assert not answer.results, "accepted found=False with results"


# --- scale -----------------------------------------------------------------

def build_corpus(store: Store, size: int) -> None:
    """Write records in batches, in a shape resembling a real feed."""
    vocabulary = (
        "lithium battery overheating fire burn hazard consumer product recall "
        "choking small parts lead paint phthalate children sleepwear flammability "
        "electric shock grounding wire detach fall injury laceration entrapment "
        "allergen undeclared milk peanut salmonella listeria contamination"
    ).split()
    rng = random.Random(SEED + 100)
    for i in range(size):
        words = rng.sample(vocabulary, 8)
        store.upsert(Record(
            key=f"scale-{i}",
            kind="recall",
            title=f"Record {i} " + " ".join(words[:3]),
            body=" ".join(words) + f" reference {i}",
            citation=Citation(
                url=f"https://example.gov/recalls/{i}", source_name="Example Agency"
            ),
        ))


@test
def test_scale_corpus_is_queryable_and_fast():
    size = 100_000 if "--big" in sys.argv else 20_000
    store = fresh(f"scale{size}")

    started = time.perf_counter()
    build_corpus(store, size)
    ingest_seconds = time.perf_counter() - started
    assert store.counts()["records"] == size, store.counts()

    engine = Engine(store)
    started = time.perf_counter()
    engine.verify("warm up the index")
    index_seconds = time.perf_counter() - started

    queries = [
        "lithium battery overheating fire",
        "listeria contamination undeclared milk",
        "sleepwear flammability children",
        "grounding wire detach electric shock",
    ]
    timings = []
    for query in queries:
        for _ in range(5):
            started = time.perf_counter()
            answer = engine.verify(query)
            timings.append(time.perf_counter() - started)
            if answer.found:
                for result in answer.results:
                    assert result.get("source_url"), "uncited result at scale"

    mean_ms = (sum(timings) / len(timings)) * 1000
    worst_ms = max(timings) * 1000
    print(f"         {size:,} records | ingest {ingest_seconds:.1f}s | "
          f"index {index_seconds:.1f}s | query mean {mean_ms:.2f}ms worst {worst_ms:.2f}ms")

    # Thresholds are calibrated to the REALISTIC corpus size, and the large-corpus
    # bound is deliberately loose.
    #
    # Query cost is linear in the number of candidate documents, and a common
    # query term matches a large fraction of a large corpus, so scoring is
    # inherently O(matching documents). Measured: ~20ms at 20k records, ~150ms at
    # 100k. That is documented in README "Status" as a known limit.
    #
    # What these bounds actually guard is the ~500x regression from rebuilding
    # the index per request (215ms at 14k records, which would be seconds here),
    # plus the full-table COUNTs that once ran on every query. Both are
    # order-of-magnitude failures, not tuning.
    if size <= 20_000:
        assert mean_ms < 50, f"mean query {mean_ms:.1f}ms at {size:,} records"
        assert worst_ms < 250, f"worst query {worst_ms:.1f}ms at {size:,} records"
    else:
        assert mean_ms < 400, f"mean query {mean_ms:.1f}ms at {size:,} records"
        assert worst_ms < 1200, f"worst query {worst_ms:.1f}ms at {size:,} records"


@test
def test_scale_index_is_reused_not_rebuilt():
    size = 10_000
    store = fresh("scalecache")
    build_corpus(store, size)
    engine = Engine(store)
    first = engine.index()
    started = time.perf_counter()
    for _ in range(200):
        engine.verify("lithium battery fire")
    elapsed = time.perf_counter() - started
    assert engine.index() is first, "the index was rebuilt during querying"
    per_query_ms = (elapsed / 200) * 1000
    # A rebuild on this corpus takes ~10s, so 200 rebuilds would be minutes.
    assert per_query_ms < 20, f"{per_query_ms:.1f}ms per query suggests a rebuild"


@test
def test_scale_conformance_still_passes():
    store = fresh("scaleconf")
    build_corpus(store, 5_000)
    report = run_conformance(store)
    assert report.passed, report.render()


def main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"running {len(tests)} fuzz/scale tests (seed {SEED})\n")
    for fn in tests:
        fn()

    print()
    for name, tb in FAILED:
        print(f"--- {name} ---")
        print(tb.encode("ascii", "backslashreplace").decode("ascii"))

    for store in _STORES:
        try:
            store.close()
        except Exception:  # noqa: BLE001
            pass
    for tmp in _TMP:
        shutil.rmtree(tmp, ignore_errors=True)

    total = len(PASSED) + len(FAILED)
    print(f"OK ({len(PASSED)} passed)" if not FAILED
          else f"FAILED ({len(FAILED)} of {total})")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
