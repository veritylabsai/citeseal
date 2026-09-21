"""Hermetic test over a real upstream: the OSV adapter, replayed from a fixture.

tests/live_smoke.py proves the adapter works against the live API. This proves
the same code path keeps working in CI, by replaying the exact payloads that
were recorded from it (tests/fixtures/osv_sample.json), with no network.

The distinction matters. A fixture the adapter's author invented tests the
adapter against its own assumptions. A fixture recorded from the live service
tests it against the real thing.

If the fixture is missing (a fresh clone with the file removed), this suite
reports SKIP rather than failing, and tells you how to regenerate it.

    python tests/test_real_corpus.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from citeseal import Engine, Store, ingest, run_conformance  # noqa: E402
from examples.osv_source import OSVCorpus  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "osv_sample.json"

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []
SKIPPED: list[str] = []
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


def load_fixture() -> list[dict[str, Any]] | None:
    if not FIXTURE.exists():
        return None
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def replayer(entries: list[dict[str, Any]]):
    """Serve recorded responses; refuse anything not recorded."""
    def fetch(url: str, payload: dict[str, Any] | None = None) -> Any:
        for entry in entries:
            if entry["url"] == url and entry.get("payload") == payload:
                if "error" in entry:
                    raise OSError(entry["error"])
                return entry["response"]
        raise AssertionError(
            f"the fixture has no recorded response for {url} {payload!r}. "
            "Re-run `python tests/live_smoke.py` to refresh it."
        )
    return fetch


def corpus() -> Store:
    entries = load_fixture()
    if entries is None:
        raise unittest_skip("fixture missing")
    tmp = Path(tempfile.mkdtemp(prefix="citeseal-real-"))
    _TMP.append(tmp)
    store = Store(tmp / "osv.sqlite3")
    _STORES.append(store)
    ingest(store, OSVCorpus(fetch_json=replayer(entries)))
    return store


class unittest_skip(Exception):
    pass


def guarded(fn):
    """Turn a missing fixture into a SKIP rather than a failure."""
    def wrapper():
        try:
            fn()
        except unittest_skip:
            SKIPPED.append(fn.__name__)
            print(f"  SKIP  {fn.__name__}  (fixture missing)")
            return
        except Exception:  # noqa: BLE001
            FAILED.append((fn.__name__, traceback.format_exc()))
            print(f"  FAIL  {fn.__name__}")
            return
        PASSED.append(fn.__name__)
        print(f"  PASS  {fn.__name__}")
    wrapper.__name__ = fn.__name__
    return wrapper


@guarded
def test_real_upstream_yields_records():
    store = corpus()
    counts = store.counts()
    assert counts["records"] > 10, counts
    assert counts["by_kind"] == {"vulnerability": counts["records"]}, counts["by_kind"]


@guarded
def test_every_real_record_is_cited():
    store = corpus()
    for record in store.all_records():
        assert record.citation.url.startswith(("http://", "https://")), record.record_id
        assert record.citation.source_name, record.record_id


@guarded
def test_citations_come_from_varied_real_hosts():
    store = corpus()
    hosts = {r.citation.url.split("/")[2] for r in store.all_records()}
    assert len(hosts) > 3, f"expected varied citation hosts, got {hosts}"
    assert all("." in h for h in hosts), hosts


@guarded
def test_a_real_record_is_retrievable_by_its_own_id():
    store = corpus()
    engine = Engine(store)
    for record in store.all_records()[:10]:
        answer = engine.verify(record.key)
        assert answer.found, f"{record.key} not retrievable by its own id"
        assert any(r["record_id"] == record.record_id for r in answer.results)


@guarded
def test_long_real_bodies_survive_intact():
    """The regression guard for the truncation bug.

    The index once cached a 600-character prefix of each body and returned it,
    so every result longer than that was silently altered. Only a body longer
    than the old cap can detect a return of it, and real OSV details comfortably
    exceed it.
    """
    store = corpus()
    engine = Engine(store)
    longest = max(store.all_records(), key=lambda r: len(r.body))
    assert len(longest.body) > 600, (
        f"fixture has no body over 600 chars (longest {len(longest.body)}), so it "
        "cannot detect truncation. Re-run tests/live_smoke.py to refresh it."
    )
    answer = engine.verify(longest.key)
    assert answer.found
    result = next(r for r in answer.results if r["record_id"] == longest.record_id)
    assert result["body"] == longest.body, (
        f"body was altered: stored {len(longest.body)} chars, "
        f"returned {len(result['body'])}"
    )
    assert result["title"] == longest.title
    assert result["source_url"] == longest.citation.url


@guarded
def test_nested_osv_fields_are_searchable():
    """OSV expresses impact as package names nested several levels deep. The
    adapter flattens them; if it stopped, these queries would return nothing."""
    store = corpus()
    engine = Engine(store)
    for query in ("django", "lodash", "axios"):
        answer = engine.verify(query)
        assert answer.found, f"{query!r} did not match anything in a real OSV corpus"


@guarded
def test_unrelated_domain_returns_a_negative():
    store = corpus()
    answer = Engine(store).verify("childrens pajamas flammability sleepwear")
    assert not answer.found, "a consumer-products query matched a security corpus"


@guarded
def test_conformance_passes_on_real_data():
    store = corpus()
    report = run_conformance(store)
    assert report.passed, report.render()


@guarded
def test_the_adapter_rejects_an_uncitable_vulnerability():
    """THE HARD RULE, against the real adapter.

    An OSV entry with no resolvable reference has nothing to cite. It must be
    refused and reported, not admitted without a source.
    """
    uncitable = {
        "id": "OSV-TEST-0001",
        "summary": "A vulnerability with no references at all",
        "details": "No advisory link exists.",
        "references": [],
        "affected": [],
    }
    tmp = Path(tempfile.mkdtemp(prefix="citeseal-real-uncitable-"))
    _TMP.append(tmp)
    store = Store(tmp / "s.sqlite3")
    _STORES.append(store)

    source = OSVCorpus(packages=[("npm", "x")], fetch_json=lambda u, p=None: {"vulns": [uncitable]})
    report = ingest(store, source)

    assert report.new == 0, "an uncitable vulnerability was admitted to the corpus"
    assert report.rejected == 1, report.as_dict()
    assert report.errors, "the rejection was not reported"
    assert store.counts()["records"] == 0


def main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"running {len(tests)} real-corpus tests"
          + ("" if FIXTURE.exists() else "  (FIXTURE MISSING -- will skip)\n"))
    print()
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
    if FAILED:
        print(f"FAILED ({len(FAILED)} of {total})")
        return 1
    print(f"OK ({len(PASSED)} passed"
          + (f", {len(SKIPPED)} skipped" if SKIPPED else "") + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
