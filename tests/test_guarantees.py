"""Citeseal's own guarantees, tested.

Deliberately dependency-free and self-running, so `python tests/test_guarantees.py`
works on a bare interpreter with nothing installed:

    python tests/test_guarantees.py

Exits non-zero on failure. Every test here is a promise the framework makes to
whoever builds a corpus on it, so a failure is a broken contract, not a bug in
the caller's code.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
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
    run_conformance,
)
from examples.sources import GlossaryCorpus, RecallFeed  # noqa: E402

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


def fresh_store(name: str) -> Store:
    tmp = Path(tempfile.mkdtemp(prefix=f"citeseal-{name}-"))
    _TMP.append(tmp)
    store = Store(tmp / "store.sqlite3")
    _STORES.append(store)
    return store


def sample(key: str = "a-1", kind: str = "notice", body: str = "widget fire risk") -> Record:
    return Record(
        key=key,
        kind=kind,
        title="Acme recalls Widget",
        body=body,
        citation=Citation(url="https://example.gov/a", source_name="Example Agency"),
    )


# --- the hard rule ---------------------------------------------------------

@test
def test_record_without_citation_is_impossible():
    """The core promise: you cannot build an uncited record at all."""
    for bad in ("", "   ", "not-a-url", "ftp://example.gov/x", "/relative/path"):
        try:
            Citation(url=bad, source_name="X")
        except UncitedRecordError:
            continue
        raise AssertionError(f"Citation accepted a bad url: {bad!r}")


@test
def test_citation_requires_a_source_name():
    try:
        Citation(url="https://example.gov/a", source_name="  ")
    except UncitedRecordError:
        return
    raise AssertionError("Citation accepted a blank source_name")


@test
def test_record_rejects_a_dict_in_place_of_a_citation():
    """A dict is the likely mistake, and accepting it would move citation
    validation somewhere less safe."""
    try:
        Record(key="k", kind="notice", title="t", body="b",
               citation={"url": "https://example.gov/a", "source_name": "X"})
    except UncitedRecordError:
        return
    raise AssertionError("Record accepted a dict where a Citation was required")


@test
def test_answer_cannot_claim_an_uncited_result():
    """Guard the answer type itself, not just the call sites."""
    try:
        Answer(found=True, query="q", results=({"record_id": "x"},))
    except UncitedRecordError:
        pass
    else:
        raise AssertionError("Answer allowed found=True with an uncited result")

    try:
        Answer(found=True, query="q", results=())
    except ValueError:
        pass
    else:
        raise AssertionError("Answer allowed found=True with no results")

    try:
        Answer(found=False, query="q", results=({"source_url": "https://x.gov"},))
    except ValueError:
        pass
    else:
        raise AssertionError("Answer allowed found=False carrying results")


# --- storage ---------------------------------------------------------------

@test
def test_unchanged_reingest_emits_no_event():
    store = fresh_store("reingest")
    record = sample()
    assert store.upsert(record).status == "new"
    before = len(store.events(limit=1000))
    for _ in range(3):
        assert store.upsert(record).status == "unchanged"
    assert len(store.events(limit=1000)) == before, "a refresh was reported as a change"


@test
def test_changed_record_is_superseded_not_overwritten():
    store = fresh_store("supersede")
    store.upsert(sample(body="original hazard text"))
    revised = Record(
        key="a-1", kind="notice", title="Acme recalls Widget",
        body="revised hazard text",
        citation=Citation(url="https://example.gov/a", source_name="Example Agency"),
    )
    outcome = store.upsert(revised)
    assert outcome.status == "changed" and outcome.version == 2, outcome

    history = store.history("notice:a-1")
    assert history, "no archived version was kept"
    assert history[0]["snapshot"]["body"] == "original hazard text", history[0]
    assert store.get("notice:a-1").body == "revised hazard text"


@test
def test_record_id_includes_kind():
    """Two sources may legitimately use the same key."""
    store = fresh_store("kinds")
    store.upsert(sample(key="same", kind="recall"))
    store.upsert(sample(key="same", kind="definition"))
    assert store.counts()["records"] == 2


@test
def test_body_size_is_bounded():
    store = fresh_store("bounds")
    try:
        store.upsert(sample(body="x" * 30_000))
    except ValueError:
        return
    raise AssertionError("store accepted an oversized body")


# --- query behaviour -------------------------------------------------------

@test
def test_no_match_is_an_explicit_negative():
    store = fresh_store("negative")
    ingest(store, RecallFeed())
    answer = Engine(store).verify("zzqx flurble wumpus 9999")
    assert not answer.found, "a nonsense query claimed a match"
    assert answer.results == ()
    assert answer.reason, "a negative must state a reason"


@test
def test_results_are_verbatim_stored_records():
    store = fresh_store("verbatim")
    ingest(store, RecallFeed())
    engine = Engine(store)
    answer = engine.verify("kettle burn hazard")
    assert answer.found, "expected a match for a term present in the corpus"
    for result in answer.results:
        stored = store.get(result["record_id"])
        assert stored is not None
        assert result["body"] == stored.body
        assert result["title"] == stored.title
        assert result["source_url"] == stored.citation.url


@test
def test_every_result_is_cited():
    store = fresh_store("cited")
    ingest(store, GlossaryCorpus())
    answer = Engine(store).verify("childrens product certificate")
    assert answer.found
    for result in answer.results:
        assert result["source_url"].startswith("http")


@test
def test_empty_query_is_rejected_not_answered():
    store = fresh_store("empty")
    ingest(store, RecallFeed())
    answer = Engine(store).verify("   ")
    assert not answer.found and answer.reason == "empty query"


@test
def test_small_corpus_can_still_match():
    """Regression guard: an absolute IDF cutoff once made a one-record corpus
    unmatchable, because even its rarest term failed the threshold."""
    store = fresh_store("tiny")
    store.upsert(Record(
        key="only-1", kind="notice", title="Only Record Here",
        body="a singular hazard about a specific gizmo",
        citation=Citation(url="https://example.gov/only", source_name="Example"),
    ))
    answer = Engine(store).verify("singular gizmo")
    assert answer.found, "a one-record corpus could not match its own content"


@test
def test_query_length_is_bounded():
    store = fresh_store("longquery")
    ingest(store, RecallFeed())
    answer = Engine(store).verify("kettle " * 5000)
    assert not answer.found or answer.results


@test
def test_index_is_reused_across_queries():
    store = fresh_store("cache")
    ingest(store, RecallFeed())
    engine = Engine(store)
    first = engine.index()
    second = engine.index()
    assert first is second, "the index was rebuilt for a second query on unchanged data"

    store.upsert(sample(key="new-1"))
    assert engine.index() is not first, "the index was not invalidated after a write"


# --- sources ---------------------------------------------------------------

@test
def test_partial_source_failure_keeps_what_it_wrote():
    class Exploding:
        key = "exploding"
        name = "Exploding Source"

        def fetch(self):
            yield sample(key="ok-1")
            raise RuntimeError("upstream went away mid-stream")

    store = fresh_store("partial")
    report = ingest(store, Exploding())
    assert report.new == 1, report
    assert report.errors and "upstream went away" in report.errors[0]
    assert store.get("notice:ok-1") is not None


@test
def test_writes_survive_reopening_the_store():
    """Durability across a process boundary.

    In-process tests read back through the same open connection, where
    uncommitted rows are perfectly visible -- so they cannot detect a missing
    commit. This closes the store and opens it again, which is the only way to
    observe that the data actually reached disk.
    """
    tmp = Path(tempfile.mkdtemp(prefix="citeseal-durability-"))
    _TMP.append(tmp)
    path = tmp / "store.sqlite3"

    writer = Store(path)
    ingest(writer, RecallFeed())
    expected = writer.counts()["records"]
    writer.close()
    assert expected == 3, expected

    reader = Store(path)
    _STORES.append(reader)
    try:
        assert reader.counts()["records"] == expected, (
            "records did not survive reopening the store -- writes are not being committed"
        )
        assert Engine(reader).verify("kettle burn hazard").found
    finally:
        reader.close()


@test
def test_registry_runs_every_source():
    from citeseal import SourceRegistry

    registry = SourceRegistry()
    registry.register(RecallFeed())
    registry.register(GlossaryCorpus())
    store = fresh_store("registry")
    reports = registry.ingest_all(store)
    assert len(reports) == 2
    kinds = store.counts()["by_kind"]
    assert kinds.get("recall") == 3 and kinds.get("definition") == 3, kinds


# --- conformance -----------------------------------------------------------

@test
def test_conformance_passes_on_both_corpora():
    for source in (RecallFeed(), GlossaryCorpus()):
        store = fresh_store(f"conf-{source.key}")
        ingest(store, source)
        report = run_conformance(store)
        assert report.passed, f"{source.key} failed conformance:\n{report.render()}"
        assert len(report.checks) == 6


@test
def test_conformance_leaves_no_probe_in_the_corpus():
    store = fresh_store("probe-cleanup")
    ingest(store, RecallFeed())
    before = store.counts()["records"]
    run_conformance(store)
    assert store.counts()["records"] == before, "the conformance probe was left behind"
    assert not Engine(store).verify("citeseal conformance probe").found


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"running {len(tests)} guarantee tests\n")
    for fn in tests:
        fn()

    print()
    for name, tb in FAILED:
        print(f"--- {name} ---")
        print(tb)

    # Close every handle before removing directories: on Windows an open SQLite
    # connection keeps the file locked and rmtree raises WinError 32.
    for store in _STORES:
        store.close()
    for tmp in _TMP:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"OK ({len(PASSED)} passed)" if not FAILED else f"FAILED ({len(FAILED)} of {len(PASSED) + len(FAILED)})")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
