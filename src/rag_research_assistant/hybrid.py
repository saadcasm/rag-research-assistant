"""Rank-based fusion for combining heterogeneous retrievers."""

from typing import List, Sequence

from .models import SearchResult


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchResult]], *, top_k: int, rrf_k: int = 60
) -> List[SearchResult]:
    """Fuse rankings without assuming their raw score scales are comparable."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if rrf_k < 0:
        raise ValueError("rrf_k cannot be negative")

    scores: dict[str, float] = {}
    chunks = {}
    first_seen: dict[str, int] = {}
    seen_counter = 0
    for ranking in rankings:
        seen_in_ranking = set()
        for rank, result in enumerate(ranking, start=1):
            chunk_id = result.chunk.chunk_id
            if chunk_id in seen_in_ranking:
                continue
            seen_in_ranking.add(chunk_id)
            if chunk_id not in first_seen:
                first_seen[chunk_id] = seen_counter
                seen_counter += 1
                chunks[chunk_id] = result.chunk
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    ranked_ids = sorted(scores, key=lambda item: (-scores[item], first_seen[item]))
    return [
        SearchResult(scores[chunk_id], chunks[chunk_id])
        for chunk_id in ranked_ids[: min(top_k, len(ranked_ids))]
    ]
