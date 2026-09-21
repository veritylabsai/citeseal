"""Citeseal -- turn a corpus into a ground-truth MCP server that cannot invent.

The framework exists to make one property structural rather than aspirational:
**no answer without a citation.**

    from citeseal import Citation, Record, Store, Engine

    store = Store("corpus.sqlite3")
    store.upsert(Record(
        key="acme-2026-01",
        kind="notice",
        title="Acme recalls Model X",
        body="Acme recalled Model X for a battery fire risk.",
        citation=Citation(
            url="https://example.gov/recalls/acme-model-x",
            source_name="Example Agency",
        ),
    ))

    answer = Engine(store).verify("acme model x")
    assert answer.found and answer.results[0]["source_url"]

    answer = Engine(store).verify("something not in the corpus")
    assert not answer.found and answer.results == ()

What this is not: a model, a RAG pipeline, or a summariser. It never writes
prose. It stores what an authority published, and returns it or refuses.
"""

from .conformance import ConformanceReport, run_conformance
from .engine import Answer, Engine
from .matching import Index, token_set, tokenize
from .record import BODY_MAX, SHORT_MAX, Citation, Record, UncitedRecordError
from .sources import IngestReport, Source, SourceRegistry, ingest
from .store import Store

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "BODY_MAX",
    "Citation",
    "ConformanceReport",
    "Engine",
    "Index",
    "IngestReport",
    "Record",
    "SHORT_MAX",
    "Source",
    "SourceRegistry",
    "Store",
    "UncitedRecordError",
    "ingest",
    "run_conformance",
    "token_set",
    "tokenize",
    "__version__",
]
