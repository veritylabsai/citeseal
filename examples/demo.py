"""Build two unrelated corpora on Citeseal and check both.

    python examples/demo.py

Demonstrates that the framework carries the guarantees for a structured feed and
for a curated glossary without either adapter implementing them.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from citeseal import Engine, SourceRegistry, Store, ingest, run_conformance  # noqa: E402
from examples.sources import GlossaryCorpus, RecallFeed  # noqa: E402


def build(store: Store, sources: list) -> None:
    for source in sources:
        report = ingest(store, source)
        print(
            f"  ingested {source.key}: {report.new} new, {report.changed} changed, "
            f"{report.unchanged} unchanged"
            + (f", errors={report.errors}" if report.errors else "")
        )


def show(engine: Engine, query: str) -> None:
    answer = engine.verify(query)
    if answer.found:
        print(f"  {query!r} -> {len(answer.results)} cited result(s)")
        for result in answer.results[:2]:
            print(f"      [{result['kind']}] {result['title'][:70]}")
            print(f"          {result['source_url']}")
    else:
        print(f"  {query!r} -> NO VERIFIED RECORD ({answer.reason})")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stores: list[Store] = []

        def new_store(name: str) -> Store:
            store = Store(root / name)
            stores.append(store)
            return store

        try:
            print("=" * 72)
            print("Corpus A: a structured recall feed")
            print("=" * 72)
            store_a = new_store("recalls.sqlite3")
            build(store_a, [RecallFeed()])
            engine_a = Engine(store_a)
            show(engine_a, "kettle burn hazard")
            show(engine_a, "listeria smoked salmon")
            show(engine_a, "a query that matches nothing at all")
            report_a = run_conformance(store_a)
            print()
            print(report_a.render())

            print()
            print("=" * 72)
            print("Corpus B: a curated glossary, wholly different shape")
            print("=" * 72)
            store_b = new_store("glossary.sqlite3")
            build(store_b, [GlossaryCorpus()])
            engine_b = Engine(store_b)
            show(engine_b, "childrens product certificate")
            show(engine_b, "substance of very high concern")
            show(engine_b, "a query that matches nothing at all")
            report_b = run_conformance(store_b)
            print()
            print(report_b.render())

            print()
            print("=" * 72)
            print("One registry, both corpora")
            print("=" * 72)
            registry = SourceRegistry()
            registry.register(RecallFeed())
            registry.register(GlossaryCorpus())
            store_c = new_store("combined.sqlite3")
            registry.ingest_all(store_c)
            print(f"  sources: {registry.keys()}")
            print(f"  records: {store_c.counts()}")
            engine_c = Engine(store_c)
            show(engine_c, "listeria")
            report_c = run_conformance(store_c)
            print()
            print(report_c.render())

            ok = report_a.passed and report_b.passed and report_c.passed
            print()
            print("ALL CORPORA PASSED" if ok else "SOME CORPORA FAILED")
            return 0 if ok else 1
        finally:
            # Close before the temporary directory is torn down; an open SQLite
            # handle keeps the file locked on Windows.
            for store in stores:
                store.close()


if __name__ == "__main__":
    sys.exit(main())
