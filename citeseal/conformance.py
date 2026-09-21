"""Conformance: the guarantees, as something you can run against a corpus.

This is the part that makes the framework more than a library. A corpus built on
Citeseal either honours the guarantees or it does not, and that is a decidable
question -- so it should be a command, not a promise in a README.

    python -m citeseal.conformance --db path/to/store.sqlite3

Exit code 0 means the corpus passed. Anything else means it did not, and the
report says which guarantee failed and on what evidence.

Six guarantees are checked:

  G1  Every stored record carries a resolvable http(s) citation.
  G2  A query with no match returns an explicit negative, never a partial guess.
  G3  Every result is a stored record reproduced verbatim, not generated text.
  G4  An answer claiming results always cites every one of them.
  G5  Re-ingesting unchanged data emits no events (a refresh is not a change).
  G6  A changed record is superseded, not overwritten, and the old version
      remains retrievable from history.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from .engine import Answer, Engine
from .record import Citation, Record, UncitedRecordError, utcnow
from .store import Store

__all__ = ["Check", "ConformanceReport", "run_conformance"]

# Queries that should never match anything in a real corpus. Kept nonsensical on
# purpose: a plausible-looking query that happens to miss would test the wrong
# thing, and would start failing the day the corpus grew.
NONSENSE_QUERIES = (
    "zzqx nonexistent transmogrifier gibberish",
    "flurble wumpus 9999 quux",
)


@dataclass(slots=True)
class Check:
    code: str
    name: str
    passed: bool
    detail: str

    def line(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"  [{mark}] {self.code}  {self.name}\n         {self.detail}"


@dataclass(slots=True)
class ConformanceReport:
    checks: list[Check] = field(default_factory=list)
    corpus_size: int = 0
    kinds: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks) and bool(self.checks)

    def render(self) -> str:
        lines = [
            "Citeseal conformance report",
            f"  corpus: {self.corpus_size} records  kinds: {self.kinds or '{}'}",
            "",
        ]
        lines.extend(c.line() for c in self.checks)
        lines.append("")
        lines.append("RESULT: PASS" if self.passed else "RESULT: FAIL")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "corpus_size": self.corpus_size,
            "kinds": self.kinds,
            "checks": [
                {"code": c.code, "name": c.name, "passed": c.passed, "detail": c.detail}
                for c in self.checks
            ],
        }


def _g1_citations(store: Store, records: list[Record]) -> Check:
    bad = [r.record_id for r in records if not r.citation.url.startswith(("http://", "https://"))]
    return Check(
        "G1", "every record carries a resolvable citation",
        not bad,
        "all records cited" if not bad else f"uncited records: {bad[:5]}",
    )


def _g2_explicit_negative(engine: Engine) -> Check:
    for query in NONSENSE_QUERIES:
        answer = engine.verify(query)
        if answer.found:
            return Check(
                "G2", "no match returns an explicit negative", False,
                f"nonsense query {query!r} claimed a match",
            )
        if not answer.reason or answer.results:
            return Check(
                "G2", "no match returns an explicit negative", False,
                f"negative answer for {query!r} lacked a reason or carried results",
            )
    return Check(
        "G2", "no match returns an explicit negative", True,
        "nonsense queries returned a stated reason and no results",
    )


def _g3_verbatim_results(store: Store, engine: Engine) -> Check:
    """Results must be stored records, byte for byte.

    This is the anti-generation check: if any result text differs from the store,
    something synthesised it.
    """
    sample = store.all_records()[:25]
    checked = 0
    for record in sample:
        query = record.title.split("(")[0][:120]
        if not query.strip():
            continue
        answer = engine.verify(query)
        if not answer.found:
            continue
        stored = store.get(record.record_id)
        for result in answer.results:
            if result["record_id"] != record.record_id:
                continue
            checked += 1
            if stored is None or result["body"] != stored.body or result["title"] != stored.title:
                return Check(
                    "G3", "results are stored records, not generated text", False,
                    f"result {result['record_id']} differs from the stored record",
                )
    return Check(
        "G3", "results are stored records, not generated text", True,
        f"{checked} matching result(s) compared byte-for-byte against the store",
    )


def _g4_answer_cites_everything(engine: Engine, records: list[Record]) -> Check:
    """Belt and braces: Answer already enforces this, so this checks the runtime
    agrees with the type rather than trusting the type alone."""
    probes = [r.title[:80] for r in records[:10] if r.title.strip()]
    if not probes:
        return Check("G4", "a positive answer cites every result", True, "no records to probe")
    for query in probes:
        answer = engine.verify(query)
        if answer.found:
            missing = [r.get("record_id") for r in answer.results if not r.get("source_url")]
            if missing:
                return Check(
                    "G4", "a positive answer cites every result", False,
                    f"results without a citation: {missing}",
                )
    try:
        Answer(found=True, query="x", results=({"record_id": "y"},))
    except UncitedRecordError:
        pass
    else:
        return Check(
            "G4", "a positive answer cites every result", False,
            "Answer accepted an uncited result; the guard is not active",
        )
    return Check(
        "G4", "a positive answer cites every result", True,
        f"{len(probes)} positive answer(s) fully cited; uncited construction refused",
    )


def _g5_refresh_is_not_a_change(store: Store, records: list[Record]) -> Check:
    if not records:
        return Check("G5", "re-ingesting unchanged data emits no events", True, "empty corpus")
    before = len(store.events(limit=10_000))
    for record in records[:20]:
        store.upsert(record)
    after = len(store.events(limit=10_000))
    if after != before:
        return Check(
            "G5", "re-ingesting unchanged data emits no events", False,
            f"{after - before} spurious event(s) from re-ingesting identical records",
        )
    return Check(
        "G5", "re-ingesting unchanged data emits no events", True,
        f"{min(len(records), 20)} record(s) re-ingested; event count unchanged at {before}",
    )


def _g6_append_only() -> Check:
    """Prove supersession on a scratch store, so the real corpus is untouched.

    An earlier version wrote its probe into the corpus under test and then tried
    to tidy up. That was wrong twice over: it appended to data it was only
    supposed to inspect, and the cleanup was a special case that could fail. A
    throwaway store proves exactly the same property with no side effects.
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(prefix="citeseal-conformance-") as tmp:
        scratch = Store(Path(tmp) / "probe.sqlite3")
        try:
            probe = Record(
                key="append-only-probe",
                kind="probe",
                title="Citeseal conformance probe",
                body="temporary record used to verify append-only behaviour",
                citation=Citation(
                    url="https://example.invalid/conformance", source_name="citeseal"
                ),
                observed_at=utcnow(),
            )
            first = scratch.upsert(probe)
            revised = Record(
                key=probe.key,
                kind=probe.kind,
                title="Citeseal conformance probe (revised)",
                body=probe.body,
                citation=probe.citation,
                observed_at=utcnow(),
            )
            second = scratch.upsert(revised)
            history = scratch.history(probe.record_id)
        finally:
            # Must close before the temp directory is removed: an open handle
            # keeps the file locked on Windows.
            scratch.close()

    if first.status != "new":
        return Check(
            "G6", "a changed record is superseded, not overwritten", False,
            f"first insert reported {first.status!r}, expected 'new'",
        )
    if second.version != 2 or not history:
        return Check(
            "G6", "a changed record is superseded, not overwritten", False,
            f"expected version 2 with history; got version {second.version} "
            f"and {len(history)} archived version(s)",
        )
    if history[0]["snapshot"].get("title") != probe.title:
        return Check(
            "G6", "a changed record is superseded, not overwritten", False,
            "the archived version does not hold the previous content",
        )
    return Check(
        "G6", "a changed record is superseded, not overwritten", True,
        "version 2 written on a scratch store; version 1 remained retrievable",
    )


def run_conformance(store: Store, on_check: Callable[[Check], None] | None = None) -> ConformanceReport:
    engine = Engine(store)
    records = store.all_records()
    counts = store.counts()

    report = ConformanceReport(corpus_size=len(records), kinds=counts["by_kind"])

    checks: list[Check] = [
        _g6_append_only(),
        _g1_citations(store, records),
        _g2_explicit_negative(engine),
        _g3_verbatim_results(store, engine),
        _g4_answer_cites_everything(engine, records),
        _g5_refresh_is_not_a_change(store, records),
    ]

    order = {"G1": 1, "G2": 2, "G3": 3, "G4": 4, "G5": 5, "G6": 6}
    checks.sort(key=lambda c: order.get(c.code, 99))
    for check in checks:
        report.checks.append(check)
        if on_check:
            on_check(check)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="citeseal-conformance",
        description="Check that a Citeseal corpus honours the framework guarantees.",
    )
    parser.add_argument("--db", required=True, help="path to the Citeseal SQLite store")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args(argv)

    store = Store(args.db)
    report = run_conformance(store)

    if args.json:
        import json
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(report.render())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
