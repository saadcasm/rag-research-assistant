"""Small, explicit BM25 implementation for lexical chunk retrieval."""

from collections import Counter
from math import log
import re
from typing import List, Sequence

from .models import Chunk, SearchResult


_WORD = re.compile(r"[^\W_]+", flags=re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Extract case-insensitive Unicode word tokens without stemming."""

    return _WORD.findall(text.casefold())


class BM25Index:
    """Precompute corpus statistics and rank chunks with Okapi BM25."""

    def __init__(
        self, chunks: Sequence[Chunk], *, k1: float = 1.5, b: float = 0.75
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1")
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self.document_lengths = [sum(counts.values()) for counts in self.term_frequencies]
        self.average_document_length = (
            sum(self.document_lengths) / len(self.document_lengths)
            if self.document_lengths
            else 0.0
        )
        self.document_frequencies: Counter[str] = Counter()
        for counts in self.term_frequencies:
            self.document_frequencies.update(counts.keys())

    def inverse_document_frequency(self, term: str) -> float:
        """Return Robertson/Sparck Jones IDF with a positive BM25 variant."""

        document_count = len(self.chunks)
        tokens = tokenize(term)
        if document_count == 0:
            return 0.0
        frequency = self.document_frequencies[tokens[0]] if tokens else 0
        return log(1.0 + (document_count - frequency + 0.5) / (frequency + 0.5))

    def score(self, query: str) -> List[float]:
        """Calculate one BM25 relevance score per corpus chunk."""

        query_terms = tokenize(query)
        if not query_terms:
            raise ValueError("query must contain at least one word token")
        if not self.chunks:
            return []

        scores: List[float] = []
        for counts, document_length in zip(
            self.term_frequencies, self.document_lengths
        ):
            score = 0.0
            length_ratio = (
                document_length / self.average_document_length
                if self.average_document_length
                else 0.0
            )
            for term in dict.fromkeys(query_terms):
                term_frequency = counts[term]
                if term_frequency == 0:
                    continue
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * length_ratio
                )
                score += self.inverse_document_frequency(term) * (
                    term_frequency * (self.k1 + 1.0) / denominator
                )
            scores.append(score)
        return scores

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Rank chunks by BM25 score; corpus order breaks exact ties."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scores = self.score(query)
        ranked_rows = sorted(range(len(scores)), key=lambda row: (-scores[row], row))
        return [
            SearchResult(scores[row], self.chunks[row])
            for row in ranked_rows[: min(top_k, len(ranked_rows))]
        ]
