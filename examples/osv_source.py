"""A real, live corpus: the Open Source Vulnerabilities database.

This exists to test the framework's central claim. A framework that has only
ever served the corpus it was extracted from is a library with delusions, and
the honest test of an abstraction is whether a second, unrelated corpus is cheap
to stand up.

OSV is a good adversary for that claim because it shares nothing with the recall
corpus this was extracted from:

  * different domain          -- software supply-chain security, not consumer products
  * different upstream shape  -- nested `affected[].ranges[].events[]`, not flat records
  * different identifier      -- GHSA/CVE ids, not recall numbers
  * different citation source -- NVD advisories and package pages, not a regulator

Everything below is adapter code. It contains no storage, no indexing, no
matching, no citation enforcement and no negative reporting, because none of
those are OSV-specific.

Usage:

    from examples.osv_source import OSVCorpus
    ingest(store, OSVCorpus(packages=[("npm", "lodash")]))

Network access is required. Pass ``fetch_json=`` to inject a stub for hermetic
tests -- see tests/test_real_corpus.py.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable, Iterable

from citeseal import Citation, Record
from citeseal.sources import build_records

OSV_QUERY = "https://api.osv.dev/v1/query"
USER_AGENT = "citeseal-example/0.1 (+https://github.com/veritylabsai/citeseal)"

# (ecosystem, package). Deliberately popular packages: they have long, messy
# advisory histories, which is what we want to test against.
DEFAULT_PACKAGES: tuple[tuple[str, str], ...] = (
    ("npm", "lodash"),
    ("npm", "axios"),
    ("PyPI", "requests"),
    ("PyPI", "django"),
    ("Go", "github.com/gin-gonic/gin"),
)

# Preferences for which reference becomes THE citation. An advisory page is a
# better citation than a bare package link, because it is the document that
# actually states the vulnerability.
REFERENCE_PRIORITY = ("ADVISORY", "ARTICLE", "REPORT", "FIX", "WEB", "PACKAGE")


def _http_json(url: str, payload: dict[str, Any] | None = None) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _pick_reference(references: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose the most authoritative reference, or None if there are none."""
    if not references:
        return None
    ranked = sorted(
        references,
        key=lambda ref: REFERENCE_PRIORITY.index(ref["type"])
        if ref.get("type") in REFERENCE_PRIORITY
        else len(REFERENCE_PRIORITY),
    )
    for reference in ranked:
        if str(reference.get("url", "")).startswith(("http://", "https://")):
            return reference
    return None


def _summarise_affected(raw: dict[str, Any]) -> str:
    """Flatten OSV's nested affected[] into something searchable.

    This is the adapter's central judgement call. OSV expresses impact as
    package names, version ranges and commit events several levels deep; a
    searcher wants to match on "log4j", "django", "before 2.0.0". Flattening that
    is a decision about OSV, which is why it lives here and not in the framework.
    """
    parts: list[str] = []
    for affected in raw.get("affected") or []:
        package = affected.get("package") or {}
        name = package.get("name")
        if name:
            parts.append(str(name))
        ecosystem = package.get("ecosystem")
        if ecosystem:
            parts.append(str(ecosystem))
        for version in (affected.get("versions") or [])[:25]:
            parts.append(str(version))
        for range_ in affected.get("ranges") or []:
            for event in range_.get("events") or []:
                for key, value in event.items():
                    parts.append(f"{key} {value}")
    parts.extend(raw.get("aliases") or [])
    return " ".join(parts)


class OSVCorpus:
    """Source over the OSV API. One vulnerability becomes one cited Record."""

    key = "osv"
    name = "Open Source Vulnerabilities (OSV)"

    def __init__(
        self,
        packages: Iterable[tuple[str, str]] | None = None,
        fetch_json: Callable[..., Any] | None = None,
    ):
        self.packages = tuple(packages or DEFAULT_PACKAGES)
        self._fetch_json = fetch_json or _http_json
        self.errors: list[str] = []

    # -- fetching ----------------------------------------------------------

    def _query(self, ecosystem: str, package: str) -> list[dict[str, Any]]:
        payload = {"package": {"ecosystem": ecosystem, "name": package}}
        try:
            body = self._fetch_json(OSV_QUERY, payload)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            # A failing package must not cost the whole corpus.
            self.errors.append(f"{ecosystem}/{package}: {type(exc).__name__}: {exc}")
            return []
        return list((body or {}).get("vulns") or [])

    def fetch(self) -> Iterable[Record]:
        rows: list[dict[str, Any]] = []
        for ecosystem, package in self.packages:
            rows.extend(self._query(ecosystem, package))
        return build_records(rows, self._to_record, on_error=self.errors.append)

    # -- normalisation -----------------------------------------------------

    def _to_record(self, raw: dict[str, Any]) -> Record:
        reference = _pick_reference(raw.get("references") or [])
        if reference is None:
            # THE HARD RULE IN PRACTICE. An OSV entry with no usable reference
            # cannot become a Record -- there is nothing to cite. It is rejected
            # and reported rather than admitted without a source.
            raise ValueError(
                f"{raw.get('id')!r} has no resolvable reference URL, so it cannot "
                "be cited and is not admitted to the corpus"
            )

        vuln_id = str(raw["id"])
        summary = (raw.get("summary") or "").strip()
        details = (raw.get("details") or "").strip()
        aliases = [str(a) for a in (raw.get("aliases") or [])]

        title = summary or f"Vulnerability {vuln_id}"
        if aliases:
            title = f"{title} ({', '.join(aliases[:3])})"

        body = " ".join(
            part for part in (summary, details, _summarise_affected(raw)) if part
        )

        severities = []
        for entry in raw.get("severity") or []:
            score = entry.get("score")
            if score:
                severities.append(str(score))

        return Record(
            key=vuln_id,
            kind="vulnerability",
            title=title[:300],
            body=body[:20_000],
            citation=Citation(
                url=str(reference["url"]),
                source_name=reference.get("type", "OSV reference"),
                text=summary or None,
            ),
            attributes={
                "aliases": aliases,
                "published": raw.get("published"),
                "modified": raw.get("modified"),
                "severity": severities,
                "ecosystems": sorted({
                    str((a.get("package") or {}).get("ecosystem"))
                    for a in (raw.get("affected") or [])
                    if (a.get("package") or {}).get("ecosystem")
                }),
                "reference_count": len(raw.get("references") or []),
            },
        )
