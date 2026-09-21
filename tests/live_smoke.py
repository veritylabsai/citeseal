"""Live smoke test: build a real corpus from the real OSV API.

Network required, so this is NOT part of the default suite. It proves two things
the hermetic tests cannot:

  1. the adapter works against the live upstream, not a fixture shaped by the
     adapter's own expectations
  2. the framework's guarantees hold on data nobody involved here designed

It also records the upstream payloads so the hermetic suite can exercise the same
adapter without the network.

    python tests/live_smoke.py            # run and refresh the fixture
    python tests/live_smoke.py --no-save  # run without touching the fixture
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from citeseal import Engine, Store, ingest, run_conformance  # noqa: E402
from examples.osv_source import _http_json  # noqa: E402
from examples.osv_source import OSVCorpus  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "osv_sample.json"

# Keep the recorded fixture small: OSV `details` can run to several kilobytes
# each, and the fixture needs to exercise shape and the long-body path, not
# volume. Twenty-five vulns per package is plenty for both.
DETAIL_CAP = 900
VULNS_PER_RESPONSE = 25


def _trim(response: Any) -> Any:
    """Shrink an OSV response for storage, without changing its structure."""
    if not isinstance(response, dict):
        return response
    trimmed = dict(response)
    vulns = []
    for vuln in (response.get("vulns") or [])[:VULNS_PER_RESPONSE]:
        copy = dict(vuln)
        if isinstance(copy.get("details"), str):
            copy["details"] = copy["details"][:DETAIL_CAP]
        # Keep only the first few affected entries and trim long version lists.
        affected = []
        for entry in (copy.get("affected") or [])[:3]:
            entry = dict(entry)
            if entry.get("versions"):
                entry["versions"] = entry["versions"][:8]
            affected.append(entry)
        copy["affected"] = affected
        copy["references"] = (copy.get("references") or [])[:6]
        vulns.append(copy)
    trimmed["vulns"] = vulns
    return trimmed


def main() -> int:
    save = "--no-save" not in sys.argv
    failures: list[str] = []
    captured: list[dict[str, Any]] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}"
              + (f"\n         {detail}" if detail else ""))
        if not ok:
            failures.append(label)

    def recording_fetch(url: str, payload: dict[str, Any] | None = None) -> Any:
        """Real fetch, with the request/response pair kept for the fixture."""
        try:
            body = _http_json(url, payload)
        except Exception as exc:  # noqa: BLE001 - record the failure too
            captured.append({"url": url, "payload": payload,
                             "error": f"{type(exc).__name__}: {exc}"})
            raise
        captured.append({"url": url, "payload": payload, "response": _trim(body)})
        return body

    tmp = Path(tempfile.mkdtemp(prefix="citeseal-live-"))
    store = Store(tmp / "osv.sqlite3")
    try:
        print("Citeseal live smoke test -- Open Source Vulnerabilities (OSV)")
        print()

        source = OSVCorpus(fetch_json=recording_fetch)
        print(f"  querying {len(source.packages)} packages on api.osv.dev ...")
        started = time.perf_counter()
        report = ingest(store, source)
        elapsed = time.perf_counter() - started

        print(f"  ingested {report.new} new, {report.changed} changed, "
              f"{report.unchanged} unchanged, {report.rejected} rejected "
              f"in {elapsed:.1f}s")
        for error in source.errors[:5]:
            print(f"      adapter note: {error[:130]}")

        check("the live upstream produced records", report.total > 0,
              f"total={report.total}")
        check("the live upstream had a clean fetch", not report.errors,
              "; ".join(report.errors[:2]) or "no errors")

        counts = store.counts()
        print(f"  corpus: {counts['records']} records, kinds={counts['by_kind']}")

        records = store.all_records()
        uncited = [r.record_id for r in records
                   if not r.citation.url.startswith("http")]
        check("every live record carries an http(s) citation", not uncited,
              f"uncited: {uncited[:3]}")

        hosts = {r.citation.url.split("/")[2] for r in records}
        check("citations point at real, varied hosts", len(hosts) > 1,
              f"{len(hosts)} hosts, e.g. {sorted(hosts)[:3]}")

        engine = Engine(store)

        # Retrieval by the record's own identifier, over a sample. Includes at
        # least one long body, because a truncated body is exactly the failure
        # live data exposed the first time it was run.
        longest = max(records, key=lambda r: len(r.body))
        sample = records[:4] + [longest]
        for record in sample:
            answer = engine.verify(record.key)
            ok = answer.found and any(
                r["record_id"] == record.record_id for r in answer.results
            )
            check(f"retrievable by its own id ({record.key})", ok)

        # THE REGRESSION GUARD. The index once cached a 600-character prefix of
        # the body and returned it, so results were silently truncated. Only a
        # body longer than that can catch it.
        answer = engine.verify(longest.key)
        if answer.found:
            result = next(
                (r for r in answer.results if r["record_id"] == longest.record_id),
                None,
            )
            check(
                "a long body comes back in full, not truncated",
                result is not None and result["body"] == longest.body,
                f"stored {len(longest.body)} chars, returned "
                f"{len(result['body']) if result else 0}",
            )
            check(
                "the returned title is verbatim",
                result is not None and result["title"] == longest.title,
            )

        negative = engine.verify("childrens pajamas flammability sleepwear")
        check("an unrelated domain returns an explicit negative", not negative.found,
              negative.reason or "")

        conformance = run_conformance(store)
        check("all six guarantees hold on live data", conformance.passed)
        print()
        print(conformance.render())

        if save and captured:
            FIXTURE.parent.mkdir(parents=True, exist_ok=True)
            FIXTURE.write_text(
                json.dumps(captured, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            size = FIXTURE.stat().st_size
            print(f"\n  recorded {len(captured)} upstream response(s) to "
                  f"{FIXTURE.relative_to(ROOT)} ({size / 1024:.0f} KB)")

        print()
        print("LIVE SMOKE: PASS" if not failures
              else f"LIVE SMOKE: FAIL ({len(failures)}): {failures}")
        return 1 if failures else 0
    finally:
        store.close()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
