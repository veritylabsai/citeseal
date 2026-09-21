"""Tokenisation and IDF-weighted matching.

Extracted from a production service, including two bugs that were found the hard
way and are worth not re-introducing:

1. **The corpus must not be rebuilt per request.** Tokenising every record on
   every call turned a sub-millisecond query into a ~200ms one, which on a
   public endpoint is a remote CPU-burn. The index is built once per process and
   invalidated by a store version counter.

2. **Distinctiveness must be scale-invariant.** An absolute IDF cutoff meant a
   small corpus could never match anything, because even the rarest term failed
   the threshold. Ranking by document frequency relative to corpus size fixes it.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

__all__ = ["tokenize", "token_set", "stem", "Index", "Scored"]

_WORD_RE = re.compile(r"\w+", re.UNICODE)

STOPWORDS = frozenset(
    """a an and are as at be been by for from has have if in into is it its of on or
    that the their there these this to was were will with within without""".split()
)

# A candidate counts as a match only if it rests on something meaningful. Two
# conditions, either of which is sufficient:
#
#   * at least one matched term is DISTINCTIVE (rare in this corpus), or
#   * the query is almost entirely covered (>= COVERAGE_ESCAPE).
#
# WHY THIS EXISTS. Without it, a query matched a record on a single generic word
# and returned found=true. Searching "kettle burn hazard" over a three-record
# recall corpus returned both the kettle recall AND an unrelated child carrier,
# because both bodies contain the word "hazard". Every result was properly
# cited, so no guarantee was broken -- but an agent asking "is this recalled?"
# would have been handed an irrelevant record as evidence.
#
# The coverage escape matters too: require distinctiveness alone and an
# uninformative query ("product safety" against a corpus where every record
# mentions both) would return nothing, which is worse than returning the corpus.
COVERAGE_ESCAPE = 0.8


# The score below which a candidate is not a match at all.
#
# WHY. Distinctiveness alone is not enough in a tiny corpus: with one record,
# every term has df=1 and so counts as distinctive, and a query sharing a single
# word with the only record matched it. Searching "toy choke hazard" returned a
# kettle recall because both contained "hazard". Cited, but worthless as
# evidence, and actively dangerous for an agent answering "is this recalled?".
#
# The floor is expressed on the composite score, so ~0.5 means "either roughly
# half the query's information matched, or a distinctive term matched with
# substantial coverage". Every real match in the test suite scores well above it
# (typical: 1.0-1.25) and every generic-overlap false positive scores below
# (typical: 0.03-0.42).
MIN_SCORE = 0.5


def stem(word: str) -> str:
    """Very small suffix-stripper.

    Not a linguist's stemmer and not trying to be. It exists so that "recalls"
    matches "recall" and "certified" matches "certification"-ish, which is what
    actually matters for product-name and hazard-text search. Over-stemming is
    worse than under-stemming here, so the rules are deliberately conservative.

    English-oriented by design. Other languages simply skip stemming rather than
    being mangled by it.
    """
    if len(word) <= 4:
        return word
    for suffix in ("ations", "ation", "ings", "ing", "ies", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            base = word[: -len(suffix)]
            if suffix == "ies":
                return base + "y"
            return base
    return word


def tokenize(text: str) -> list[str]:
    return [
        stem(tok)
        for tok in _WORD_RE.findall((text or "").lower())
        if tok not in STOPWORDS and len(tok) > 1
    ]


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


@dataclass(frozen=True, slots=True)
class Scored:
    payload: Any
    score: float
    matched: tuple[str, ...]


class Index:
    """An inverted index over prepared documents.

    Build once, query many times. ``documents`` is a sequence of
    ``(payload, token_set)`` pairs; the tokenising happens before construction so
    that a cache can hold the prepared form.
    """

    def __init__(self, documents: Iterable[tuple[Any, set[str]]], version: str = ""):
        docs = list(documents)
        self.version = version
        self.size = len(docs)
        self._payloads: list[Any] = []
        self._tokens: list[set[str]] = []
        self._postings: dict[str, list[int]] = {}

        for position, (payload, tokens) in enumerate(docs):
            self._payloads.append(payload)
            self._tokens.append(tokens)
            for token in tokens:
                self._postings.setdefault(token, []).append(position)

        n = max(self.size, 1)
        self._idf: dict[str, float] = {
            token: math.log((n + 1) / (len(positions) + 0.5))
            for token, positions in self._postings.items()
        }
        # The weight of a query term the corpus has never seen. It must NOT be
        # zero: coverage divides by the total weight of the query, so scoring
        # absent terms at zero made them vanish from the denominator and
        # inflated coverage to 1.0 whenever a query's other terms were missing.
        # Concretely, "toy choke hazard" against a single kettle record scored
        # 1.08 -- a perfect match -- on the strength of one shared common word.
        # Treating an unseen term as maximally rare (df=0) is both correct and
        # the only choice that keeps coverage honest.
        self._unseen_idf = math.log((n + 1) / 0.5)
        # A term is "distinctive" if it appears in at most half the corpus. This
        # is a ratio, not an absolute count, so it behaves the same whether the
        # corpus has 10 records or 10 million.
        self.df_threshold = max(1, int(n * 0.5))

    def idf(self, token: str) -> float:
        if token in self._idf:
            return self._idf[token]
        return self._unseen_idf

    def document_frequency(self, token: str) -> int:
        return len(self._postings.get(token, ()))

    def is_distinctive(self, token: str) -> bool:
        df = self.document_frequency(token)
        return 0 < df <= self.df_threshold

    def search(self, query: str, limit: int = 10,
               min_score: float = MIN_SCORE) -> list[Scored]:
        """Rank documents against a query.

        Scoring is IDF-weighted overlap with a coverage term, so a record
        matching many rare query terms beats one matching a single common term.
        Candidates resting on nothing but common terms are discarded -- see
        COVERAGE_ESCAPE for why that matters.
        """
        if self.size == 0:
            return []
        query_tokens = list(token_set(query))
        if not query_tokens:
            return []

        # Everything below depends only on the QUERY plus which tokens a document
        # contains, never on the document's other content. Hoisting it out of the
        # candidate loop matters: in a large corpus a common query term matches
        # tens of thousands of documents, so per-candidate dictionary lookups
        # dominated the query time (measured at ~80ms over 50k records).
        token_idf = {t: self.idf(t) for t in query_tokens}
        distinctive_tokens = {t for t in query_tokens if self.is_distinctive(t)}
        total_idf = sum(token_idf.values()) or 1.0
        inv_total = 1.0 / total_idf
        inv_query_len = 1.0 / len(query_tokens)

        # Counter.update over an iterable runs in C (_count_elements), whereas a
        # Python-level `for position in postings: overlap[position] += 1` loop
        # costs roughly a microsecond per posting. At 50k records a common query
        # term reaches tens of thousands of documents, so this loop was the
        # single largest remaining cost in a query.
        overlap: Counter[int] = Counter()
        for token in query_tokens:
            postings = self._postings.get(token)
            if postings:
                overlap.update(postings)

        if not overlap:
            return []

        results: list[Scored] = []
        for position, hits in overlap.items():
            tokens = self._tokens[position]
            matched = [t for t in query_tokens if t in tokens]
            matched_idf = sum(token_idf[t] for t in matched)
            # Coverage of the query, so a long document cannot win on volume.
            coverage = matched_idf * inv_total
            # Reward distinctive terms over generic ones.
            distinctive = sum(1 for t in matched if t in distinctive_tokens)

            # Reject matches that rest entirely on terms common to this corpus.
            if distinctive == 0 and coverage < COVERAGE_ESCAPE:
                continue

            score = coverage + (0.25 * distinctive * inv_query_len)
            if score > min_score:
                results.append(
                    Scored(
                        payload=self._payloads[position],
                        score=round(score, 4),
                        matched=tuple(sorted(matched)),
                    )
                )

        results.sort(key=lambda s: (-s.score, str(s.payload)))
        return results[:limit]
