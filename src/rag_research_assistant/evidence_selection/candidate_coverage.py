"""Gold-labelled oracle evaluation over frozen, independently retrieved pools."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence, Tuple

from ..models import Chunk, SearchResult
from ..multihop import MultiHopExample, hop_matches_result


@dataclass(frozen=True)
class CandidatePool:
    """One frozen ranking reconstructed without access to evaluation labels."""

    question_id: str
    question: str
    candidates: Tuple[SearchResult, ...]
    duplicate_chunk_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class HopAvailability:
    hop_index: int
    description: str
    evidence_available: bool
    first_candidate_rank: int | None


@dataclass(frozen=True)
class DepthAvailability:
    depth: int
    depth_available: bool
    recovered_hop_indexes: Tuple[int, ...]
    recovered_available_hop_indexes: Tuple[int, ...]
    required_hop_count: int
    available_hop_count: int
    recovered_hops: int
    recovered_available_hops: int
    full: bool | None
    partial: float | None
    available_full: bool | None
    available_partial: float | None


@dataclass(frozen=True)
class CandidatePoolEvaluation:
    question_id: str
    question: str
    answerability: str
    category: str
    required_hop_count: int
    available_hop_count: int
    candidate_count: int
    candidate_chunk_ids: Tuple[str, ...]
    candidate_ranks: Tuple[int, ...]
    hop_availability: Tuple[HopAvailability, ...]
    depths: Mapping[int, DepthAvailability]
    duplicate_chunk_ids: Tuple[str, ...]
    warnings: Tuple[str, ...]
    evaluation_seconds: float

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["depths"] = {str(key): asdict(item) for key, item in self.depths.items()}
        return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_phase10g_candidate_pools(
    artifact: Mapping[str, Any], chunks: Sequence[Chunk]
) -> list[CandidatePool]:
    """Reconstruct baseline hybrid pools without reading any gold hop fields.

    This function is the label-free side of the boundary: it accepts only the
    persisted retrieval artifact and chunk corpus. Gold labels enter later in
    ``evaluate_candidate_pool`` and can therefore score, but never alter, pools.
    """

    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    pools: list[CandidatePool] = []
    for record in artifact.get("questions", []):
        identifier = record.get("question_id")
        question = record.get("question")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("Phase 10G artifact contains an invalid question_id")
        if not isinstance(question, str) or not question:
            raise ValueError(f"{identifier}: artifact contains an invalid question")
        try:
            rows = record["conditions"]["baseline"]["branch_rankings"][0]["results"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"{identifier}: baseline candidate ranking is missing") from exc
        if not isinstance(rows, list):
            raise ValueError(f"{identifier}: baseline candidate ranking must be a list")
        candidates: list[SearchResult] = []
        duplicates: list[str] = []
        seen: set[str] = set()
        for expected_rank, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                raise ValueError(f"{identifier}: candidate row must be an object")
            chunk_id = row.get("chunk_id")
            if chunk_id not in chunk_by_id:
                raise ValueError(f"{identifier}: unknown candidate chunk {chunk_id!r}")
            if row.get("rank") != expected_rank:
                raise ValueError(f"{identifier}: non-contiguous candidate ranks")
            if chunk_id in seen:
                duplicates.append(chunk_id)
                continue
            seen.add(chunk_id)
            score = row.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(f"{identifier}: candidate score must be numeric")
            candidates.append(SearchResult(float(score), chunk_by_id[chunk_id]))
        pools.append(CandidatePool(identifier, question, tuple(candidates), tuple(duplicates)))
    if not pools:
        raise ValueError("Phase 10G artifact contains no candidate pools")
    return pools


def evaluate_candidate_pool(
    example: MultiHopExample,
    pool: CandidatePool,
    *,
    requested_depths: Sequence[int] = (5, 10, 20, 50),
) -> CandidatePoolEvaluation:
    """Apply gold hops after retrieval solely to measure oracle availability."""

    if example.id != pool.question_id or example.question != pool.question:
        raise ValueError("candidate pool does not correspond to the evaluation example")
    if not requested_depths or any(
        isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0
        for depth in requested_depths
    ):
        raise ValueError("requested depths must be positive integers")
    started = perf_counter()
    first_ranks: list[int | None] = []
    for hop in example.hops:
        first_ranks.append(next(
            (
                rank
                for rank, candidate in enumerate(pool.candidates, 1)
                if hop_matches_result(hop, candidate)
            ),
            None,
        ))
    available_indexes = tuple(
        index for index, hop in enumerate(example.hops) if hop.evidence_available
    )
    depth_results: dict[int, DepthAvailability] = {}
    for depth in requested_depths:
        if depth > len(pool.candidates):
            depth_results[depth] = DepthAvailability(
                depth, False, (), (), len(example.hops), len(available_indexes),
                0, 0, None, None, None, None,
            )
            continue
        recovered = tuple(
            index for index, rank in enumerate(first_ranks)
            if rank is not None and rank <= depth
        )
        recovered_available = tuple(
            index for index in recovered if index in available_indexes
        )
        required_count = len(example.hops)
        available_count = len(available_indexes)
        depth_results[depth] = DepthAvailability(
            depth=depth,
            depth_available=True,
            recovered_hop_indexes=recovered,
            recovered_available_hop_indexes=recovered_available,
            required_hop_count=required_count,
            available_hop_count=available_count,
            recovered_hops=len(recovered),
            recovered_available_hops=len(recovered_available),
            full=len(recovered) == required_count,
            partial=len(recovered) / required_count,
            available_full=len(recovered_available) == available_count,
            available_partial=(
                len(recovered_available) / available_count if available_count else 1.0
            ),
        )
    warnings: list[str] = []
    unavailable_depths = [depth for depth, row in depth_results.items() if not row.depth_available]
    if unavailable_depths:
        warnings.append("unavailable_candidate_depths:" + ",".join(map(str, unavailable_depths)))
    if pool.duplicate_chunk_ids:
        warnings.append("duplicate_chunk_ids:" + ",".join(pool.duplicate_chunk_ids))
    return CandidatePoolEvaluation(
        question_id=example.id,
        question=example.question,
        answerability=example.answerability,
        category=example.category,
        required_hop_count=len(example.hops),
        available_hop_count=len(available_indexes),
        candidate_count=len(pool.candidates),
        candidate_chunk_ids=tuple(item.chunk.chunk_id for item in pool.candidates),
        candidate_ranks=tuple(range(1, len(pool.candidates) + 1)),
        hop_availability=tuple(
            HopAvailability(index, hop.description, hop.evidence_available, first_ranks[index])
            for index, hop in enumerate(example.hops)
        ),
        depths=depth_results,
        duplicate_chunk_ids=pool.duplicate_chunk_ids,
        warnings=tuple(warnings),
        evaluation_seconds=perf_counter() - started,
    )


def aggregate_candidate_coverage(
    evaluations: Sequence[CandidatePoolEvaluation], depth: int
) -> dict[str, Any] | None:
    if not evaluations:
        return None
    rows = [item.depths[depth] for item in evaluations]
    if not all(row.depth_available for row in rows):
        return None
    count = len(rows)
    return {
        "questions": count,
        "oracle_full_hop_availability": sum(bool(row.full) for row in rows) / count,
        "partial_hop_availability": sum(float(row.partial) for row in rows) / count,
        "oracle_available_hop_full_availability": sum(bool(row.available_full) for row in rows) / count,
        "available_hop_partial_availability": sum(float(row.available_partial) for row in rows) / count,
        "average_recovered_hops": sum(row.recovered_hops for row in rows) / count,
        "average_recovered_available_hops": sum(row.recovered_available_hops for row in rows) / count,
        "zero_hops": sum(row.recovered_hops == 0 for row in rows),
        "one_hop": sum(row.recovered_hops == 1 for row in rows),
        "all_hops": sum(bool(row.full) for row in rows),
        "all_available_hops": sum(bool(row.available_full) for row in rows),
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value
