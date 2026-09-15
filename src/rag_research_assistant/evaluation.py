"""Transparent retrieval and generation evaluation for the local RAG pipeline."""

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .context import build_context
from .embeddings import Embedder
from .generation import DEFAULT_TEMPERATURE, GenerationError, TextGenerator
from .index import EmbeddingIndex
from .models import CorpusIndexMetadata, SearchResult
from .prompting import build_grounded_prompt
from .retrievers import DenseRetriever, Retriever


ANSWERABILITY_CATEGORIES = {
    "answerable",
    "partially_answerable",
    "unanswerable",
}
DEFAULT_HIT_KS = (1, 3, 5)
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_REFUSAL_PHRASES = (
    "insufficient evidence",
    "supplied evidence is insufficient",
    "provided evidence is insufficient",
    "not enough information",
    "cannot answer from the supplied context",
    "can't answer from the supplied context",
    "cannot answer from the provided context",
    "can't answer from the provided context",
    "supplied sources do not contain enough information",
    "provided sources do not contain enough information",
    "supplied evidence does not contain enough information",
    "provided evidence does not contain enough information",
)


@dataclass(frozen=True)
class ExpectedSource:
    """A manually labelled source that should support an evaluation question."""

    document: str
    page_number: Optional[int] = None
    chunk_id: Optional[str] = None


@dataclass(frozen=True)
class EvaluationExample:
    """One human-curated question and its expected evidence."""

    id: str
    question: str
    answerability: str
    expected_sources: Tuple[ExpectedSource, ...]
    notes: str = ""


@dataclass(frozen=True)
class RetrievedSource:
    """One ranked retrieval result saved for later diagnosis."""

    rank: int
    similarity_score: float
    document: str
    page_number: int
    chunk_id: str
    text: str


@dataclass(frozen=True)
class MappedCitation:
    """A valid answer citation resolved to the supplied context item."""

    citation_number: int
    document: str
    page_number: int
    chunk_id: str


@dataclass(frozen=True)
class GenerationEvaluation:
    """Deterministic checks around one generated answer."""

    answer: Optional[str]
    detected_citations: List[int]
    invalid_citations: List[int]
    mapped_citations: List[MappedCitation]
    refusal_detected: Optional[bool]
    error: Optional[str] = None


@dataclass(frozen=True)
class QuestionEvaluation:
    """Retrieval and optional generation diagnostics for one question."""

    id: str
    question: str
    answerability: str
    expected_sources: Tuple[ExpectedSource, ...]
    retrieved_sources: List[RetrievedSource]
    top_similarity_score: Optional[float]
    first_correct_rank: Optional[int]
    first_correct_similarity_score: Optional[float]
    hit_at_1: Optional[bool]
    hit_at_3: Optional[bool]
    hit_at_5: Optional[bool]
    expected_source_recall_at_1: Optional[float]
    expected_source_recall_at_3: Optional[float]
    expected_source_recall_at_5: Optional[float]
    generation: Optional[GenerationEvaluation]
    notes: str


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate metrics, keeping retrieval and generation separate."""

    questions_evaluated: int
    retrieval_questions: int
    answerable_questions: int
    partially_answerable_questions: int
    unanswerable_questions: int
    hit_at_1: Optional[float]
    hit_at_3: Optional[float]
    hit_at_5: Optional[float]
    mean_first_correct_rank: Optional[float]
    expected_source_recall_at_1: Optional[float]
    expected_source_recall_at_3: Optional[float]
    expected_source_recall_at_5: Optional[float]
    generation_questions: int
    unanswerable_refusals: int
    unanswerable_refusal_accuracy: Optional[float]
    invalid_citation_references: int
    invalid_citation_question_ids: List[str]
    retrieval_failure_ids: List[str]
    unanswerable_non_refusal_ids: List[str]
    generation_error_ids: List[str]


@dataclass(frozen=True)
class EvaluationReport:
    """Serializable Phase 4 report."""

    schema_version: int
    created_at: str
    embedding_model: str
    embedding_dimension: int
    indexed_chunks: int
    retrieval_strategy: str
    dense_backend: Optional[str]
    retrieval_score_type: str
    reranker_model: Optional[str]
    generation_model: Optional[str]
    generation_temperature: Optional[float]
    retrieval_depth: int
    dataset_path: str
    summary: EvaluationSummary
    questions: List[QuestionEvaluation]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _require_nonempty_string(value: Any, field: str, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line_number}: {field} must be a non-empty string")
    return value.strip()


def _parse_expected_source(value: Any, line_number: int) -> ExpectedSource:
    if not isinstance(value, dict):
        raise ValueError(f"line {line_number}: each expected source must be an object")
    document = _require_nonempty_string(value.get("document"), "document", line_number)
    page_number = value.get("page_number")
    if page_number is not None and (
        isinstance(page_number, bool)
        or not isinstance(page_number, int)
        or page_number <= 0
    ):
        raise ValueError(
            f"line {line_number}: page_number must be a positive integer or null"
        )
    chunk_id = value.get("chunk_id")
    if chunk_id is not None:
        chunk_id = _require_nonempty_string(chunk_id, "chunk_id", line_number)
    return ExpectedSource(document, page_number, chunk_id)


def load_evaluation_dataset(path: Path) -> List[EvaluationExample]:
    """Load and validate a human-editable JSONL evaluation dataset."""

    examples: List[EvaluationExample] = []
    seen_ids = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"line {line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"line {line_number}: evaluation example must be an object"
                )

            example_id = _require_nonempty_string(value.get("id"), "id", line_number)
            if example_id in seen_ids:
                raise ValueError(f"line {line_number}: duplicate id {example_id!r}")
            seen_ids.add(example_id)
            question = _require_nonempty_string(
                value.get("question"), "question", line_number
            )
            answerability = _require_nonempty_string(
                value.get("answerability"), "answerability", line_number
            )
            if answerability not in ANSWERABILITY_CATEGORIES:
                allowed = ", ".join(sorted(ANSWERABILITY_CATEGORIES))
                raise ValueError(
                    f"line {line_number}: answerability must be one of {allowed}"
                )
            raw_sources = value.get("expected_sources")
            if not isinstance(raw_sources, list):
                raise ValueError(f"line {line_number}: expected_sources must be a list")
            expected_sources = tuple(
                _parse_expected_source(item, line_number) for item in raw_sources
            )
            if len(set(expected_sources)) != len(expected_sources):
                raise ValueError(
                    f"line {line_number}: expected_sources contains duplicates"
                )
            if answerability == "unanswerable" and expected_sources:
                raise ValueError(
                    f"line {line_number}: unanswerable questions cannot have expected_sources"
                )
            if answerability != "unanswerable" and not expected_sources:
                raise ValueError(
                    f"line {line_number}: {answerability} questions need expected_sources"
                )
            notes = value.get("notes", "")
            if not isinstance(notes, str):
                raise ValueError(f"line {line_number}: notes must be a string")
            examples.append(
                EvaluationExample(
                    id=example_id,
                    question=question,
                    answerability=answerability,
                    expected_sources=expected_sources,
                    notes=notes.strip(),
                )
            )
    if not examples:
        raise ValueError(f"evaluation dataset is empty: {path}")
    return examples


def expected_source_matches(expected: ExpectedSource, result: SearchResult) -> bool:
    """Match a labelled page when it falls anywhere in a chunk's page span."""

    chunk = result.chunk
    if expected.document != chunk.document:
        return False
    if expected.page_number is not None and not (
        chunk.start_page <= expected.page_number <= chunk.end_page
    ):
        return False
    if expected.chunk_id is not None and expected.chunk_id != chunk.chunk_id:
        return False
    return True


def _matching_expected_indexes(
    expected_sources: Sequence[ExpectedSource], results: Sequence[SearchResult], k: int
) -> set[int]:
    return {
        expected_index
        for expected_index, expected in enumerate(expected_sources)
        if any(expected_source_matches(expected, result) for result in results[:k])
    }


def extract_citations(answer: str) -> List[int]:
    """Extract bracketed numeric citations in first-seen order."""

    return list(
        dict.fromkeys(int(match) for match in _CITATION_PATTERN.findall(answer))
    )


def invalid_citations(citations: Sequence[int], context_count: int) -> List[int]:
    """Return citation numbers that do not identify a supplied context item."""

    return [
        citation for citation in citations if citation < 1 or citation > context_count
    ]


def detects_refusal(answer: str) -> bool:
    """Detect common refusal phrases; this is intentionally only a heuristic."""

    normalized = " ".join(answer.lower().split())
    return any(phrase in normalized for phrase in _REFUSAL_PHRASES)


def _average(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _evaluate_question(
    example: EvaluationExample,
    results: List[SearchResult],
    generator: Optional[TextGenerator],
    temperature: float,
) -> QuestionEvaluation:
    retrieved = [
        RetrievedSource(
            rank=rank,
            similarity_score=result.score,
            document=result.chunk.document,
            page_number=result.chunk.page_number,
            chunk_id=result.chunk.chunk_id,
            text=result.chunk.text,
        )
        for rank, result in enumerate(results, start=1)
    ]
    top_score = results[0].score if results else None

    first_correct_rank: Optional[int] = None
    first_correct_score: Optional[float] = None
    hits: Dict[int, Optional[bool]] = {k: None for k in DEFAULT_HIT_KS}
    recalls: Dict[int, Optional[float]] = {k: None for k in DEFAULT_HIT_KS}
    if example.answerability != "unanswerable":
        for rank, result in enumerate(results, start=1):
            if any(
                expected_source_matches(expected, result)
                for expected in example.expected_sources
            ):
                first_correct_rank = rank
                first_correct_score = result.score
                break
        for k in DEFAULT_HIT_KS:
            matched = _matching_expected_indexes(example.expected_sources, results, k)
            hits[k] = bool(matched)
            recalls[k] = len(matched) / len(example.expected_sources)

    generation: Optional[GenerationEvaluation] = None
    if generator is not None:
        if not results:
            generation = GenerationEvaluation(
                answer=None,
                detected_citations=[],
                invalid_citations=[],
                mapped_citations=[],
                refusal_detected=None,
                error="no context was retrieved",
            )
        else:
            context, sources = build_context(results)
            prompt = build_grounded_prompt(example.question, context)
            try:
                answer = generator.generate(prompt, temperature=temperature)
            except (GenerationError, ValueError) as exc:
                generation = GenerationEvaluation(
                    answer=None,
                    detected_citations=[],
                    invalid_citations=[],
                    mapped_citations=[],
                    refusal_detected=None,
                    error=str(exc),
                )
            else:
                citations = extract_citations(answer)
                invalid = invalid_citations(citations, len(sources))
                mapped = [
                    MappedCitation(
                        citation_number=number,
                        document=sources[number - 1].document,
                        page_number=sources[number - 1].page_number,
                        chunk_id=sources[number - 1].chunk_id,
                    )
                    for number in citations
                    if number not in invalid
                ]
                generation = GenerationEvaluation(
                    answer=answer,
                    detected_citations=citations,
                    invalid_citations=invalid,
                    mapped_citations=mapped,
                    refusal_detected=detects_refusal(answer),
                )

    return QuestionEvaluation(
        id=example.id,
        question=example.question,
        answerability=example.answerability,
        expected_sources=example.expected_sources,
        retrieved_sources=retrieved,
        top_similarity_score=top_score,
        first_correct_rank=first_correct_rank,
        first_correct_similarity_score=first_correct_score,
        hit_at_1=hits[1],
        hit_at_3=hits[3],
        hit_at_5=hits[5],
        expected_source_recall_at_1=recalls[1],
        expected_source_recall_at_3=recalls[3],
        expected_source_recall_at_5=recalls[5],
        generation=generation,
        notes=example.notes,
    )


def evaluate_retrieval_results(
    example: EvaluationExample, results: List[SearchResult]
) -> QuestionEvaluation:
    """Apply the established relevance labels to externally retrieved results."""

    return _evaluate_question(example, results, generator=None, temperature=0.0)


def _summarize(results: Sequence[QuestionEvaluation]) -> EvaluationSummary:
    retrieval_results = [r for r in results if r.answerability != "unanswerable"]
    generated = [r for r in results if r.generation is not None]
    generated_unanswerable = [r for r in generated if r.answerability == "unanswerable"]
    refusal_successes = sum(
        r.generation is not None and r.generation.refusal_detected is True
        for r in generated_unanswerable
    )

    def rates(field: str) -> Optional[float]:
        return _average([float(getattr(result, field)) for result in retrieval_results])

    found_ranks = [
        float(result.first_correct_rank)
        for result in retrieval_results
        if result.first_correct_rank is not None
    ]
    invalid_count = sum(
        len(result.generation.invalid_citations)
        for result in generated
        if result.generation is not None
    )
    return EvaluationSummary(
        questions_evaluated=len(results),
        retrieval_questions=len(retrieval_results),
        answerable_questions=sum(r.answerability == "answerable" for r in results),
        partially_answerable_questions=sum(
            r.answerability == "partially_answerable" for r in results
        ),
        unanswerable_questions=sum(r.answerability == "unanswerable" for r in results),
        hit_at_1=rates("hit_at_1"),
        hit_at_3=rates("hit_at_3"),
        hit_at_5=rates("hit_at_5"),
        mean_first_correct_rank=_average(found_ranks),
        expected_source_recall_at_1=rates("expected_source_recall_at_1"),
        expected_source_recall_at_3=rates("expected_source_recall_at_3"),
        expected_source_recall_at_5=rates("expected_source_recall_at_5"),
        generation_questions=len(generated),
        unanswerable_refusals=refusal_successes,
        unanswerable_refusal_accuracy=(
            refusal_successes / len(generated_unanswerable)
            if generated_unanswerable
            else None
        ),
        invalid_citation_references=invalid_count,
        invalid_citation_question_ids=[
            r.id
            for r in generated
            if r.generation is not None and r.generation.invalid_citations
        ],
        retrieval_failure_ids=[r.id for r in retrieval_results if r.hit_at_5 is False],
        unanswerable_non_refusal_ids=[
            r.id
            for r in generated_unanswerable
            if r.generation is not None and r.generation.refusal_detected is False
        ],
        generation_error_ids=[
            r.id
            for r in generated
            if r.generation is not None and r.generation.error is not None
        ],
    )


def evaluate(
    examples: Sequence[EvaluationExample],
    index: Optional[EmbeddingIndex],
    embedder: Optional[Embedder],
    *,
    retrieval_depth: int = 5,
    generator: Optional[TextGenerator] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    dataset_path: str = "",
    retriever: Optional[Retriever] = None,
    corpus_metadata: Optional[CorpusIndexMetadata] = None,
) -> EvaluationReport:
    """Run deterministic retrieval evaluation and optional local generation."""

    if retrieval_depth < max(DEFAULT_HIT_KS):
        raise ValueError("retrieval_depth must be at least 5 to calculate Hit@5")
    if not examples:
        raise ValueError("at least one evaluation example is required")
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature must be between 0 and 2")

    if retriever is not None:
        selected_retriever = retriever
    else:
        if embedder is None or index is None:
            raise ValueError(
                "index and embedder are required when no retriever is supplied"
            )
        selected_retriever = DenseRetriever(index, embedder)
    if corpus_metadata is None:
        if index is None:
            raise ValueError("corpus_metadata is required without a NumPy index")
        corpus_metadata = CorpusIndexMetadata(
            index.model_name, index.dimension, len(index.chunks)
        )
    question_results = [
        _evaluate_question(
            example,
            selected_retriever.search(example.question, top_k=retrieval_depth),
            generator,
            temperature,
        )
        for example in examples
    ]
    return EvaluationReport(
        schema_version=3,
        created_at=datetime.now(timezone.utc).isoformat(),
        embedding_model=corpus_metadata.embedding_model,
        embedding_dimension=corpus_metadata.embedding_dimension,
        indexed_chunks=corpus_metadata.chunk_count,
        retrieval_strategy=selected_retriever.name,
        dense_backend=getattr(selected_retriever, "backend_name", None),
        retrieval_score_type=selected_retriever.score_name,
        reranker_model=getattr(selected_retriever, "reranker_model", None),
        generation_model=generator.model_name if generator is not None else None,
        generation_temperature=temperature if generator is not None else None,
        retrieval_depth=retrieval_depth,
        dataset_path=dataset_path,
        summary=_summarize(question_results),
        questions=question_results,
    )


def write_evaluation_report(report: EvaluationReport, path: Path) -> None:
    """Write a complete machine-readable report as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(report.to_dict(), output, indent=2, ensure_ascii=False)
        output.write("\n")
