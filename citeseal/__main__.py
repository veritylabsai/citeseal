"""Command line entry point.

    python -m citeseal ingest   --db corpus.sqlite3 --source module:attr
    python -m citeseal query    --db corpus.sqlite3 "some query"
    python -m citeseal changes  --db corpus.sqlite3 --since 2026-01-01T00:00:00Z
    python -m citeseal check    --db corpus.sqlite3
    python -m citeseal stats    --db corpus.sqlite3
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any

from . import __version__
from .conformance import run_conformance
from .engine import Engine
from .sources import Source, ingest
from .store import Store


def _load_source(spec: str) -> Source:
    """Load a Source from ``module.path:attribute``.

    Imported lazily so that using the CLI for query-only work never imports the
    corpus's HTTP client stack.
    """
    if ":" not in spec:
        raise SystemExit(
            f"--source must look like 'mycorpus.sources:cpsc', got {spec!r}"
        )
    module_name, attr = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(f"could not import {module_name!r}: {exc}") from exc
    try:
        target = getattr(module, attr)
    except AttributeError as exc:
        raise SystemExit(f"{module_name!r} has no attribute {attr!r}") from exc

    # A class must be instantiated. Checking `hasattr(target, "fetch")` is not
    # enough: on a class that finds the *unbound* function, so the class passes
    # the check and then fails at call time with a missing `self`.
    if isinstance(target, type):
        source = target()
    elif not hasattr(target, "fetch") and callable(target):
        source = target()  # a factory, so a source can take configuration
    else:
        source = target

    for required in ("key", "name", "fetch"):
        if not hasattr(source, required):
            raise SystemExit(
                f"{spec!r} is not a Citeseal Source: it has no {required!r}"
            )
    return source


def _emit(payload: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return
    if isinstance(payload, dict) and "results" in payload:
        print(f"found={payload['found']}  query={payload['query']!r}  count={payload['count']}")
        for result in payload["results"]:
            print(f"  [{result['kind']}] {result['title']}")
            print(f"      score={result['score']}  source={result['source_url']}")
        if not payload["found"]:
            print(f"  reason: {payload['reason']}")
        return
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="citeseal", description=__doc__)
    parser.add_argument("--version", action="version", version=f"citeseal {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="fetch sources into the store")
    p_ingest.add_argument("--db", required=True)
    p_ingest.add_argument("--source", action="append", required=True,
                          help="module.path:attribute (repeatable)")
    p_ingest.add_argument("--json", action="store_true")

    p_query = sub.add_parser("query", help="search the store")
    p_query.add_argument("--db", required=True)
    p_query.add_argument("query")
    p_query.add_argument("--kind")
    p_query.add_argument("--limit", type=int, default=10)
    p_query.add_argument("--json", action="store_true")

    p_changes = sub.add_parser("changes", help="the change feed")
    p_changes.add_argument("--db", required=True)
    p_changes.add_argument("--since")
    p_changes.add_argument("--limit", type=int, default=50)
    p_changes.add_argument("--json", action="store_true")

    p_check = sub.add_parser("check", help="run the conformance guarantees")
    p_check.add_argument("--db", required=True)
    p_check.add_argument("--json", action="store_true")

    p_stats = sub.add_parser("stats", help="corpus counts")
    p_stats.add_argument("--db", required=True)
    p_stats.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    store = Store(args.db)

    if args.command == "ingest":
        reports = [ingest(store, _load_source(spec)).as_dict() for spec in args.source]
        _emit(reports if len(reports) > 1 else reports[0], args.json)
        return 0

    if args.command == "query":
        answer = Engine(store).search(args.query, kind=args.kind, limit=args.limit)
        _emit(answer.as_dict(), args.json)
        return 0 if answer.found else 1

    if args.command == "changes":
        _emit(Engine(store).changes(since=args.since, limit=args.limit), args.json)
        return 0

    if args.command == "check":
        report = run_conformance(store)
        if args.json:
            _emit(report.as_dict(), True)
        else:
            print(report.render())
        return 0 if report.passed else 1

    if args.command == "stats":
        counts = store.counts()
        counts["last_ingested_at"] = store.last_ingested_at()
        counts["data_version"] = store.data_version()
        _emit(counts, args.json)
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
