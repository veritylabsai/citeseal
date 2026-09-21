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

_WORD_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """a an and are as at be been by for from has have if in into is it its of on or
    that the their there these this to was were will with within without""".split()
)


def stem(word: str) -> str:
    """Very small suffix-stripper.

    Not a linguist's stemmer and not trying to be. It exists so that "recalls"
    matches "recall" and "certified" matches "certification"-ish, which is what
    actually matters for product-name and hazard-text search. Over-stemming is
    worse than under-stemming here, so the rules are deliberately conservative.
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
        # A term is "distinctive" if it appears in at most half the corpus. This
        # is a ratio, not an absolute count, so it behaves the same whether the
        # corpus has 10 records or 10 million.
        self.df_threshold = max(1, int(n * 0.5))

    def idf(self, token: str) -> float:
        return self._idf.get(token, 0.0)

    def document_frequency(self, token: str) -> int:
        return len(self._postings.get(token, ()))

    def is_distinctive(self, token: str) -> bool:
        df = self.document_frequency(token)
        return 0 < df <= self.df_threshold

    def search(self, query: str, limit: int = 10, min_score: float = 0.0) -> list[Scored]:
        """Rank documents against a query.

        Scoring is IDF-weighted overlap with a coverage term, so a record
        matching many rare query terms beats one matching a single common term.
        """
        if self.size == 0:
            return []
        query_tokens = token_set(query)
        if not query_tokens:
            return []

        overlap: Counter[int] = Counter()
        for token in query_tokens:
            for position in self._postings.get(token, ()):
                overlap[position] += 1

        if not overlap:
            return []

        total_idf = sum(self.idf(t) for t in query_tokens) or 1.0
        results: list[Scored] = []
        for position, hits in overlap.items():
            tokens = self._tokens[position]
            matched_idf = sum(self.idf(t) for t in query_tokens if t in tokens)
            # Coverage of the query, so a long document cannot win on volume.
            coverage = matched_idf / total_idf
            # Reward distinctive terms over generic ones.
            distinctive = sum(
                1 for t in query_tokens if t in tokens and self.is_distinctive(t)
            )
            score = coverage + (0.25 * distinctive / len(query_tokens))
            if score > min_score:
                results.append(
                    Scored(
                        payload=self._payloads[position],
                        score=round(score, 4),
                        matched=tuple(sorted(t for t in query_tokens if t in tokens)),
                    )
                )

        results.sort(key=lambda s: (-s.score, str(s.payload)))
        return results[:limit]
