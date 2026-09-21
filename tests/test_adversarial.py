"""Adversarial tests: an honest attempt to break Citeseal.

The guarantee tests check that the framework does what it claims. These check
what happens when it is misused, stressed, or fed hostile input. Anything that
fails here is a real defect, not a nitpick.

    python tests/test_adversarial.py
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from citeseal import (  # noqa: E402
    Answer,
    Citation,
    Engine,
    Record,
    Store,
    UncitedRecordError,
    ingest,
)

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
    tmp = Path(tempfile.mkdtemp(prefix=f"citeseal-adv-{name}-"))
    _TMP.append(tmp)
    store = Store(tmp / "store.sqlite3")
    _STORES.append(store)
    return store


def rec(key: str, body: str = "widget hazard", kind: str = "notice",
        title: str = "Acme Widget", url: str = "https://example.gov/a") -> Record:
    return Record(key=key, kind=kind, title=title, body=body,
                  citation=Citation(url=url, source_name="Example"))


# --- hostile input ---------------------------------------------------------

@test
def test_sql_injection_in_key_and_query_is_inert():
    store = fresh("sqli")
    hostile = "'; DROP TABLE records; --"
    store.upsert(rec(hostile))
    store.upsert(rec("normal-1", body="a perfectly ordinary hazard"))

    answer = Engine(store).verify(hostile)
    assert answer.found, "a hostile but valid key should still be findable verbatim"

    # The table must still exist and hold both rows.
    assert store.counts()["records"] == 2, "SQL in a key altered the corpus"
    answer = Engine(store).verify("ordinary hazard")
    assert answer.found, "the table did not survive a hostile key"


@test
def test_sql_injection_in_query_is_inert():
    store = fresh("sqli2")
    ingest_ok(store)
    for hostile in ("'; DELETE FROM records; --", "%' OR 1=1 --", "\" OR \"\"=\""):
        Engine(store).verify(hostile)
    assert store.counts()["records"] == 2, "an injection-shaped query modified the store"


def ingest_ok(store: Store) -> None:
    store.upsert(rec("a-1", body="lithium battery fire risk"))
    store.upsert(rec("a-2", body="choking hazard small parts"))


@test
def test_unicode_keys_titles_and_bodies():
    store = fresh("unicode")
    cases = [
        ("キー-1", "リチウム電池 発火", "株式会社テスト"),
        ("ключ-2", "литиевая батарея пожар", "ООО Тест"),
        ("مفتاح-3", "بطارية الليثيوم حريق", "شركة اختبار"),
        ("emoji-🔥-4", "battery fire 🔥 hazard", "Acme 🏭"),
        ("ümlaut-5", "Lithiumbatterie Brandgefahr", "Müller GmbH"),
    ]
    for key, body, title in cases:
        store.upsert(rec(key, body=body, title=title))

    counts = store.counts()["records"]
    assert counts == len(cases), f"expected {len(cases)} records, stored {counts}"

    for key, body, _ in cases:
        answer = Engine(store).verify(key)
        assert answer.found, f"could not retrieve a unicode key: {key!r}"
        stored = store.get(f"notice:{key}")
        assert stored is not None and stored.body == body


@test
def test_unicode_survives_a_reopen():
    tmp = Path(tempfile.mkdtemp(prefix="citeseal-adv-unicode-"))
    _TMP.append(tmp)
    path = tmp / "store.sqlite3"
    w = Store(path)
    w.upsert(rec("キー-1", body="リチウム電池 発火", title="株式会社テスト"))
    w.close()

    r = Store(path)
    _STORES.append(r)
    try:
        got = r.get("notice:キー-1")
        assert got is not None, "unicode key vanished across a reopen"
        assert got.body == "リチウム電池 発火", got.body
    finally:
        r.close()


@test
def test_citation_url_is_not_silently_repaired():
    """A bad URL must raise, not be coerced into something plausible."""
    bad = [
        "javascript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "file:///etc/passwd",
        "http://",              # scheme but no host
        "https:///path",        # no host
        "//example.gov/x",      # scheme-relative
        "example.gov/x",        # no scheme
        "   ",                  # whitespace only
        None,
        123,
    ]
    for value in bad:
        try:
            Citation(url=value, source_name="X")
        except UncitedRecordError:
            continue
        raise AssertionError(f"Citation accepted an unusable url: {value!r}")


@test
def test_control_characters_rejected_in_key():
    for key in ("bad\nkey", "bad\tkey", "bad\x00key", "x" * 201, " leading", "trailing "):
        try:
            rec(key)
        except ValueError:
            continue
        raise AssertionError(f"Record accepted a bad key: {key!r}")


# --- boundary conditions ---------------------------------------------------

@test
def test_empty_store_answers_everything_gracefully():
    store = fresh("empty")
    engine = Engine(store)
    for query in ("", "   ", "anything at all"):
        answer = engine.verify(query)
        assert not answer.found and answer.results == ()
        assert answer.reason
    changes = engine.changes()
    assert changes["count"] == 0
    assert store.counts()["records"] == 0


@test
def test_oversized_query_does_not_explode():
    store = fresh("bigquery")
    ingest_ok(store)
    answer = Engine(store).verify("x" * 100_000)
    assert not answer.found or answer.results
    answer = Engine(store).verify("lithium " * 20_000)
    assert not answer.found or answer.results


@test
def test_body_at_the_size_limit_is_accepted_and_rejected_past_it():
    store = fresh("bodysize")
    store.upsert(rec("at-limit", body="x" * 20_000))
    assert store.counts()["records"] == 1
    try:
        store.upsert(rec("past-limit", body="x" * 20_001))
    except ValueError:
        return
    raise AssertionError("store accepted a body over the limit")


@test
def test_extremely_long_title_is_stored():
    store = fresh("longtitle")
    store.upsert(rec("long-title", title="T" * 5_000))
    assert store.get("notice:long-title") is not None


@test
def test_duplicate_upsert_is_idempotent_many_times():
    store = fresh("idem")
    record = rec("same")
    for _ in range(50):
        store.upsert(record)
    assert store.counts()["records"] == 1
    assert store.counts()["events"] == 1, "re-ingest emitted extra events"
    assert store.counts()["superseded_versions"] == 0


@test
def test_many_sequential_changes_build_correct_history():
    store = fresh("history")
    for i in range(25):
        store.upsert(rec("evolving", body=f"revision number {i}"))
    assert store.get("notice:evolving").version == 25
    history = store.history("notice:evolving")
    assert len(history) == 24, f"expected 24 archived versions, got {len(history)}"
    for entry in history:
        assert entry["snapshot"]["body"].startswith("revision number")
    versions = sorted(e["version"] for e in history)
    assert versions == list(range(1, 25))


@test
def test_retired_records_leave_the_corpus():
    store = fresh("retired")
    store.upsert(rec("keep-me", body="visible shelf life notice"))
    store.upsert(rec("retire-me", body="quarantine withdrawal notice"))
    assert store.counts()["records"] == 2

    assert store.retire("notice:retire-me") is True
    assert store.counts()["records"] == 1

    # The precise property: the retired record is gone from query results. (The
    # corpus is not expected to return nothing for the query -- the surviving
    # record may still share a term, and in a one-record corpus every term it
    # contains is distinctive.)
    ids = [r["record_id"] for r in Engine(store).verify("quarantine withdrawal notice").results]
    assert "notice:retire-me" not in ids, (
        f"a retired record was still being served: {ids} -- index not invalidated"
    )

    # The citation is retained for audit even though it is out of the corpus.
    assert store.get("notice:retire-me") is not None
    assert any(e["event_type"] == "record_retired" for e in store.events())

    assert store.retire("notice:retire-me") is False, "double-retire should be a no-op"
    assert store.unretire("notice:retire-me") is True
    ids = [r["record_id"] for r in Engine(store).verify("quarantine withdrawal notice").results]
    assert "notice:retire-me" in ids, "unretire did not restore the record to the corpus"


# --- determinism and ranking ----------------------------------------------

@test
def test_repeated_queries_return_identical_ordering():
    store = fresh("determinism")
    for i in range(30):
        store.upsert(rec(f"r-{i}", body=f"lithium battery hazard variant {i}",
                         title=f"Device {i}"))
    engine = Engine(store)
    first = [r["record_id"] for r in engine.verify("lithium battery hazard").results]
    for _ in range(5):
        again = [r["record_id"] for r in engine.verify("lithium battery hazard").results]
        assert again == first, "result ordering is not deterministic"


@test
def test_a_more_relevant_record_outranks_a_weaker_one():
    store = fresh("ranking")
    store.upsert(rec("strong", title="Northwind Kettle Burn Hazard",
                     body="the kettle lid detaches and scalds the user"))
    store.upsert(rec("weak", title="Unrelated Notice",
                     body="a kettle was mentioned once in passing alongside many other items"))
    results = Engine(store).verify("kettle burn hazard scalds").results
    assert results, "expected at least one hit"
    assert results[0]["record_id"] == "notice:strong", [
        (r["record_id"], r["score"]) for r in results
    ]


@test
def test_rare_term_beats_common_term():
    store = fresh("idf")
    for i in range(20):
        store.upsert(rec(f"common-{i}", body="general safety notice about products"))
    store.upsert(rec("rare", body="general safety notice about a graphene dehumidifier"))
    results = Engine(store).verify("graphene").results
    assert results, "rare-term query returned nothing"
    assert results[0]["record_id"] == "notice:rare"


# --- cache correctness -----------------------------------------------------

@test
def test_index_reflects_a_write_from_a_second_store_handle():
    """Two Store objects on one file must not serve staleness to each other.
    The index cache is keyed by path, so this is a real risk."""
    tmp = Path(tempfile.mkdtemp(prefix="citeseal-adv-twohandle-"))
    _TMP.append(tmp)
    path = tmp / "store.sqlite3"

    a = Store(path)
    b = Store(path)
    _STORES.extend([a, b])
    a.upsert(rec("first", body="alpha shelf life notice"))

    # "second" exists only in b's write; if a keeps serving a cached index, this
    # never becomes visible.
    b.upsert(rec("second", body="beta shelf life notice"))
    answer = Engine(a).verify("beta shelf life notice")
    assert answer.found, "a write through one handle was invisible to the other"
    assert any(r["key"] == "second" for r in answer.results), answer.results


@test
def test_generic_word_overlap_alone_is_not_a_match():
    """A query must not match a record on a word common to the whole corpus.

    Regression guard for a real defect: searching "kettle burn hazard" over a
    three-record recall corpus returned an unrelated child carrier, because both
    bodies contain "hazard". The result was cited, so no guarantee broke -- but
    an agent asking "is this recalled?" would have been handed it as evidence.
    """
    store = fresh("generic")
    store.upsert(rec("kettle", title="Northwind Kettle Burn Hazard",
                     body="the kettle lid detaches and scalds the user"))
    store.upsert(rec("carrier", title="Halcyon Child Carrier Fall Hazard",
                     body="the carrier hip belt buckle can release unexpectedly"))
    store.upsert(rec("salmon", title="Vantage Smoked Salmon Listeria Risk",
                     body="the product may be contaminated with listeria"))

    results = Engine(store).verify("kettle burn hazard").results
    ids = [r["record_id"] for r in results]
    assert "notice:kettle" in ids, ids
    assert "notice:salmon" not in ids, f"matched on generic overlap alone: {ids}"


@test
def test_index_is_not_rebuilt_when_nothing_changed():
    store = fresh("nocache")
    ingest_ok(store)
    engine = Engine(store)
    first = engine.index()
    for _ in range(10):
        assert engine.index() is first, "index rebuilt without any write"


@test
def test_index_rebuilds_after_every_write():
    store = fresh("rebuild")
    engine = Engine(store)
    store.upsert(rec("one", body="alpha"))
    i1 = engine.index()
    store.upsert(rec("two", body="beta"))
    i2 = engine.index()
    assert i1 is not i2, "index served stale data after a write"
    assert Engine(store).verify("beta").found


# --- concurrency -----------------------------------------------------------

@test
def test_concurrent_writers_do_not_lose_records():
    store = fresh("concurrent")
    errors: list[str] = []
    barrier = threading.Barrier(8)

    def writer(worker: int) -> None:
        try:
            barrier.wait(timeout=10)
            for i in range(15):
                store.upsert(rec(f"w{worker}-{i}", body=f"hazard {worker} {i}"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"writer {worker}: {type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=writer, args=(w,)) for w in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors, errors
    expected = 8 * 15
    got = store.counts()["records"]
    assert got == expected, f"lost writes: expected {expected}, stored {got}"


@test
def test_concurrent_readers_and_writers_stay_consistent():
    store = fresh("rw")
    ingest_ok(store)
    engine = Engine(store)
    errors: list[str] = []
    stop = threading.Event()

    def reader() -> None:
        try:
            while not stop.is_set():
                answer = engine.verify("lithium battery")
                # The invariant must hold under concurrent mutation.
                if answer.found:
                    for r in answer.results:
                        if not r.get("source_url"):
                            errors.append("uncited result returned during concurrency")
                elif answer.results:
                    errors.append("negative answer carried results")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"reader: {type(exc).__name__}: {exc}")

    def writer() -> None:
        try:
            for i in range(40):
                store.upsert(rec(f"live-{i}", body=f"lithium battery item {i}"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"writer: {type(exc).__name__}: {exc}")

    readers = [threading.Thread(target=reader) for _ in range(4)]
    w = threading.Thread(target=writer)
    for t in readers:
        t.start()
    w.start()
    w.join(timeout=60)
    stop.set()
    for t in readers:
        t.join(timeout=30)

    assert not errors, errors
    assert store.counts()["records"] == 2 + 40


# --- durability / corruption ----------------------------------------------

@test
def test_store_reopens_after_simulated_crash():
    """WAL should recover; nothing committed may be lost."""
    tmp = Path(tempfile.mkdtemp(prefix="citeseal-adv-crash-"))
    _TMP.append(tmp)
    path = tmp / "store.sqlite3"

    store = Store(path)
    for i in range(25):
        store.upsert(rec(f"c-{i}", body=f"hazard {i}"))
    # Deliberately do NOT close: drop the handle to simulate a crash.
    del store

    recovered = Store(path)
    _STORES.append(recovered)
    assert recovered.counts()["records"] == 25, recovered.counts()


@test
def test_corrupt_database_fails_loudly_not_silently():
    tmp = Path(tempfile.mkdtemp(prefix="citeseal-adv-corrupt-"))
    _TMP.append(tmp)
    path = tmp / "store.sqlite3"
    path.write_bytes(b"this is not a sqlite database, it is a text file" * 40)

    try:
        store = Store(path)
        _STORES.append(store)
        store.counts()
    except sqlite3.DatabaseError:
        return  # correct: refuse to pretend
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"unexpected error type on corrupt db: {type(exc).__name__}") from exc
    raise AssertionError(
        "a corrupt database was opened and queried as though it were valid"
    )


@test
def test_events_are_ordered_newest_first_and_bounded():
    store = fresh("events")
    for i in range(30):
        store.upsert(rec(f"e-{i}", body=f"hazard {i}"))
    events = Engine(store).changes(limit=10)
    assert events["count"] == 10, events["count"]
    stamps = [e["happened_at"] for e in events["changes"]]
    assert stamps == sorted(stamps, reverse=True), "events are not newest-first"


# --- performance sanity ----------------------------------------------------

@test
def test_query_is_fast_on_a_realistic_corpus():
    """A public endpoint must not let one request burn CPU.

    Generous bound: the point is to catch a return of per-request rebuilds, which
    is a ~500x regression, not to benchmark.
    """
    store = fresh("perf")
    body = ("the product may overheat and pose a burn hazard to consumers "
            "during normal use in household environments")
    for i in range(4_000):
        store.upsert(rec(f"p-{i}", body=f"{body} reference {i}", title=f"Device {i}"))

    engine = Engine(store)
    engine.verify("warm the cache")  # build the index once

    start = time.perf_counter()
    for _ in range(20):
        engine.verify("overheat burn hazard consumers")
    elapsed = (time.perf_counter() - start) / 20

    assert elapsed < 0.05, f"mean query took {elapsed * 1000:.1f}ms on 4,000 records"


def main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"running {len(tests)} adversarial tests\n")
    for fn in tests:
        fn()

    print()
    for name, tb in FAILED:
        print(f"--- {name} ---")
        # The console on Windows is cp1252, so a traceback containing a Unicode
        # key raises while being printed and hides the real failure.
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
