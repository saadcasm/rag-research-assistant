"""Command-line interface for ingestion, retrieval, generation, and evaluation."""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .bm25 import BM25Index
from .comparison import compare_reports, write_comparison_report
from .embeddings import DEFAULT_MODEL, Embedder, SentenceTransformerEmbedder
from .evaluation import (
    EvaluationReport,
    evaluate as run_evaluation,
    load_evaluation_dataset,
    write_evaluation_report,
)
from .generation import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_TEMPERATURE,
    GenerationError,
    OllamaGenerator,
)
from .index import EmbeddingIndex, InvalidIndexError, build_index, load_index
from .models import SearchResult
from .pdf import discover_pdfs
from .pipeline import build_chunks, read_jsonl, write_jsonl
from .rag import answer_question
from .reranking import DEFAULT_RERANKER_MODEL, CrossEncoderReranker
from .retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankingRetriever,
    Retriever,
)


DEFAULT_INPUT = Path("data/papers")
DEFAULT_OUTPUT = Path("data/processed/chunks.jsonl")
DEFAULT_INDEX = Path("data/processed/embedding_index")
DEFAULT_EVALUATION_DATASET = Path("data/evaluation/questions.jsonl")
DEFAULT_EVALUATION_OUTPUT = Path("data/evaluation/results/latest.json")
DEFAULT_COMPARISON_OUTPUT = Path("data/evaluation/results/comparison.json")


def _add_strategy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--retriever", choices=("dense", "bm25", "hybrid"), default="dense"
    )
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--candidate-depth", type=int, default=20)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local RAG ingestion, retrieval, and grounded generation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Extract and chunk local PDFs")
    ingest.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ingest.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ingest.add_argument("--chunk-size", type=int, default=1_200)
    ingest.add_argument("--overlap", type=int, default=200)

    inspect = subparsers.add_parser("inspect", help="Print sample chunks")
    inspect.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    inspect.add_argument("--limit", type=int, default=3)

    embed = subparsers.add_parser(
        "embed", help="Build a persistent local embedding index"
    )
    embed.add_argument("--chunks", type=Path, default=DEFAULT_OUTPUT)
    embed.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    embed.add_argument("--model", default=DEFAULT_MODEL)
    embed.add_argument("--batch-size", type=int, default=32)
    embed.add_argument(
        "--device", help="Optional Sentence Transformers device, e.g. cpu or mps"
    )

    reranker_download = subparsers.add_parser(
        "reranker-download", help="Download and cache the optional local reranker"
    )
    reranker_download.add_argument("--model", default=DEFAULT_RERANKER_MODEL)
    reranker_download.add_argument(
        "--device", help="Optional model device, e.g. cpu or mps"
    )

    search_parser = subparsers.add_parser(
        "search", help="Search indexed chunks semantically"
    )
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--chunks", type=Path, default=DEFAULT_OUTPUT)
    search_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    search_parser.add_argument(
        "--device", help="Optional Sentence Transformers device, e.g. cpu or mps"
    )
    search_parser.add_argument("--preview-chars", type=int, default=400)
    _add_strategy_arguments(search_parser)

    ask = subparsers.add_parser(
        "ask", help="Answer from retrieved evidence with local Ollama"
    )
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--chunks", type=Path, default=DEFAULT_OUTPUT)
    ask.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ask.add_argument("--device", help="Optional embedding device, e.g. cpu or mps")
    ask.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    ask.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ask.add_argument("--timeout", type=float, default=180.0)
    ask.add_argument("--show-context", action="store_true")
    ask.add_argument("--preview-chars", type=int, default=400)
    _add_strategy_arguments(ask)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate retrieval and optional local generation"
    )
    evaluate_parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_EVALUATION_DATASET
    )
    evaluate_parser.add_argument("--top-k", type=int, default=5)
    evaluate_parser.add_argument("--chunks", type=Path, default=DEFAULT_OUTPUT)
    evaluate_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    evaluate_parser.add_argument(
        "--device", help="Optional embedding device, e.g. cpu or mps"
    )
    evaluate_parser.add_argument("--with-generation", action="store_true")
    evaluate_parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    evaluate_parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE
    )
    evaluate_parser.add_argument("--timeout", type=float, default=180.0)
    evaluate_parser.add_argument(
        "--output", type=Path, default=DEFAULT_EVALUATION_OUTPUT
    )
    evaluate_parser.add_argument("--verbose", action="store_true")
    _add_strategy_arguments(evaluate_parser)

    compare = subparsers.add_parser(
        "compare", help="Benchmark dense, BM25, hybrid, and reranked retrieval"
    )
    compare.add_argument("--dataset", type=Path, default=DEFAULT_EVALUATION_DATASET)
    compare.add_argument("--top-k", type=int, default=5)
    compare.add_argument("--chunks", type=Path, default=DEFAULT_OUTPUT)
    compare.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    compare.add_argument("--device", help="Optional model device, e.g. cpu or mps")
    compare.add_argument("--candidate-depth", type=int, default=20)
    compare.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    compare.add_argument("--output", type=Path, default=DEFAULT_COMPARISON_OUTPUT)
    return parser


def _print_retrieval(
    results: List[SearchResult], preview_chars: int, score_name: str = "cosine"
) -> None:
    print("\nRetrieved context")
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        preview_text = " ".join(chunk.text.split())
        preview = preview_text[:preview_chars]
        if len(preview) < len(preview_text):
            preview += "..."
        print(f"\n{rank}. {score_name}_score={result.score:.4f}")
        print(
            f"   source={chunk.document} page={chunk.page_number} chunk={chunk.chunk_id}"
        )
        print(f"   {preview}")


def _percentage(value: Optional[float]) -> str:
    return "not run" if value is None else f"{value:.1%}"


def _print_evaluation(report: EvaluationReport, verbose: bool) -> None:
    summary = report.summary
    print("\nRAG Evaluation")
    print("==============")
    print(f"Questions evaluated: {summary.questions_evaluated}")
    print(f"Retrieval questions: {summary.retrieval_questions}")
    print(f"Unanswerable questions: {summary.unanswerable_questions}")
    print(f"Strategy: {report.retrieval_strategy} ({report.retrieval_score_type} score)")
    print("\nRetrieval")
    print(f"Hit@1: {_percentage(summary.hit_at_1)}")
    print(f"Hit@3: {_percentage(summary.hit_at_3)}")
    print(f"Hit@5: {_percentage(summary.hit_at_5)}")
    if summary.mean_first_correct_rank is not None:
        print(
            f"Mean first-correct rank (found questions): {summary.mean_first_correct_rank:.2f}"
        )

    print("\nGeneration")
    if report.generation_model is None:
        print("Not run (use --with-generation).")
    else:
        print(f"Model: {report.generation_model}")
        print(
            "Unanswerable refusal accuracy: "
            f"{_percentage(summary.unanswerable_refusal_accuracy)} "
            f"({summary.unanswerable_refusals}/{summary.unanswerable_questions})"
        )
        print(f"Invalid citation references: {summary.invalid_citation_references}")

    print("\nFailures")
    failure_lines = []
    for question_id in summary.retrieval_failure_ids:
        failure_lines.append(f"- {question_id}: expected source not found in top 5")
    for question_id in summary.unanswerable_non_refusal_ids:
        failure_lines.append(f"- {question_id}: no refusal phrase detected")
    for question_id in summary.invalid_citation_question_ids:
        failure_lines.append(f"- {question_id}: answer contains an invalid citation")
    for question_id in summary.generation_error_ids:
        failure_lines.append(f"- {question_id}: generation returned an error")
    print("\n".join(failure_lines) if failure_lines else "None at the measured checks.")

    if verbose:
        print("\nPer-question diagnostics")
        for result in report.questions:
            first_rank = result.first_correct_rank or "not found"
            print(f"\n- {result.id} ({result.answerability})")
            if result.answerability != "unanswerable":
                print(f"  first correct rank: {first_rank}")
                print(
                    f"  Hit@1/3/5: {result.hit_at_1}/{result.hit_at_3}/{result.hit_at_5}"
                )
            for source in result.retrieved_sources:
                print(
                    f"  {source.rank}. {source.document} p.{source.page_number} "
                    f"{report.retrieval_score_type}_score={source.similarity_score:.4f}"
                )
            if result.generation is not None:
                print(f"  refusal detected: {result.generation.refusal_detected}")
                print(f"  citations: {result.generation.detected_citations}")
                print(f"  invalid citations: {result.generation.invalid_citations}")


def _load_retriever(
    args,
) -> Tuple[EmbeddingIndex, Optional[Embedder], Retriever]:
    """Load shared corpus state and construct the requested ranking strategy."""

    if args.rerank and args.retriever != "hybrid":
        raise ValueError("--rerank requires --retriever hybrid")
    if args.candidate_depth <= 0:
        raise ValueError("candidate_depth must be positive")
    index = load_index(args.chunks, args.index)
    bm25 = BM25Retriever(BM25Index(index.chunks))
    embedder = None
    dense = None
    if args.retriever != "bm25":
        embedder = SentenceTransformerEmbedder(
            model_name=index.model_name,
            device=args.device,
            local_files_only=True,
        )
        dense = DenseRetriever(index, embedder)
    selected: Retriever
    if args.retriever == "dense":
        assert dense is not None
        selected = dense
    elif args.retriever == "bm25":
        selected = bm25
    else:
        assert dense is not None
        selected = HybridRetriever(
            dense, bm25, candidate_depth=args.candidate_depth, rrf_k=60
        )
    if args.rerank:
        reranker = CrossEncoderReranker(
            model_name=args.reranker_model,
            device=args.device,
            local_files_only=True,
        )
        selected = RerankingRetriever(
            selected, reranker, candidate_depth=args.candidate_depth
        )
    return index, embedder, selected


def _print_comparison(report) -> None:
    print("\nRetrieval strategy comparison")
    print("Strategy          Hit@1   Hit@3   Hit@5   Mean first rank")
    print("----------------  ------  ------  ------  ---------------")
    for metrics in report.strategies:
        mean_rank = (
            "n/a"
            if metrics.mean_first_correct_rank is None
            else f"{metrics.mean_first_correct_rank:.2f}"
        )
        print(
            f"{metrics.strategy:<16}"
            f"{_percentage(metrics.hit_at_1):>8}"
            f"{_percentage(metrics.hit_at_3):>8}"
            f"{_percentage(metrics.hit_at_5):>8}"
            f"{mean_rank:>18}"
        )
    for comparison in report.comparisons:
        print(f"\n{comparison.baseline} -> {comparison.contender}")
        improved = ", ".join(
            f"{item.question_id} ({item.from_rank}->{item.to_rank})"
            for item in comparison.improved
        )
        degraded = ", ".join(
            f"{item.question_id} ({item.from_rank}->{item.to_rank})"
            for item in comparison.degraded
        )
        print(f"Improved: {improved or 'none'}")
        print(f"Degraded: {degraded or 'none'}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "ingest":
        pdfs = discover_pdfs(args.input)
        if not pdfs:
            print(f"No PDF files found in {args.input}")
            return 0
        chunks = build_chunks(
            args.input, chunk_size=args.chunk_size, overlap=args.overlap
        )
        count = write_jsonl(chunks, args.output)
        print(f"Processed {len(pdfs)} PDF(s) into {count} chunk(s): {args.output}")
        return 0

    if args.command == "inspect":
        chunks = read_jsonl(args.input)
        for chunk in chunks[: max(0, args.limit)]:
            print(f"\n[{chunk.chunk_id}] {chunk.document}, page {chunk.page_number}")
            print(f"characters {chunk.char_start}:{chunk.char_end}")
            print(chunk.text)
        print(
            f"\nShowing {min(max(0, args.limit), len(chunks))} of {len(chunks)} chunk(s)."
        )
        return 0

    if args.command == "embed":
        embedder = SentenceTransformerEmbedder(
            model_name=args.model, batch_size=args.batch_size, device=args.device
        )
        index = build_index(args.chunks, args.index, embedder)
        print(
            f"Embedded {len(index.chunks)} chunk(s) as a "
            f"{index.embeddings.shape} matrix with {index.model_name}: {args.index}"
        )
        return 0

    if args.command == "reranker-download":
        try:
            CrossEncoderReranker(model_name=args.model, device=args.device)
        except (OSError, ValueError) as exc:
            print(f"Error: reranker download failed: {exc}", file=sys.stderr)
            return 2
        print(f"Cached local reranker: {args.model}")
        return 0

    if args.command == "evaluate":
        if args.top_k < 5:
            print("Error: top_k must be at least 5 to calculate Hit@5", file=sys.stderr)
            return 2
        if not 0.0 <= args.temperature <= 2.0:
            print("Error: temperature must be between 0 and 2", file=sys.stderr)
            return 2

        generator = None
        if args.with_generation:
            try:
                generator = OllamaGenerator(model_name=args.model, timeout=args.timeout)
                generator.ensure_model_available()
            except (GenerationError, ValueError) as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 2
        try:
            examples = load_evaluation_dataset(args.dataset)
            index, embedder, retriever = _load_retriever(args)
            report = run_evaluation(
                examples,
                index,
                embedder,
                retrieval_depth=args.top_k,
                generator=generator,
                temperature=args.temperature,
                dataset_path=str(args.dataset),
                retriever=retriever,
            )
            write_evaluation_report(report, args.output)
        except (FileNotFoundError, InvalidIndexError, OSError, ValueError) as exc:
            print(f"Error: evaluation failed: {exc}", file=sys.stderr)
            return 2

        _print_evaluation(report, args.verbose)
        print(f"\nMachine-readable report: {args.output}")
        return 0

    if args.command == "compare":
        if args.top_k < 5:
            print("Error: top_k must be at least 5 to calculate Hit@5", file=sys.stderr)
            return 2
        if args.candidate_depth <= 0:
            print("Error: candidate_depth must be positive", file=sys.stderr)
            return 2
        try:
            examples = load_evaluation_dataset(args.dataset)
            index = load_index(args.chunks, args.index)
            embedder = SentenceTransformerEmbedder(
                model_name=index.model_name,
                device=args.device,
                local_files_only=True,
            )
            dense = DenseRetriever(index, embedder)
            bm25 = BM25Retriever(BM25Index(index.chunks))
            hybrid = HybridRetriever(
                dense, bm25, candidate_depth=args.candidate_depth, rrf_k=60
            )
            reranked = RerankingRetriever(
                hybrid,
                CrossEncoderReranker(
                    model_name=args.reranker_model,
                    device=args.device,
                    local_files_only=True,
                ),
                candidate_depth=args.candidate_depth,
            )
            reports = [
                run_evaluation(
                    examples,
                    index,
                    embedder,
                    retrieval_depth=args.top_k,
                    dataset_path=str(args.dataset),
                    retriever=retriever,
                )
                for retriever in (dense, bm25, hybrid, reranked)
            ]
            comparison = compare_reports(reports)
            write_comparison_report(comparison, args.output)
        except (FileNotFoundError, InvalidIndexError, OSError, ValueError) as exc:
            print(f"Error: comparison failed: {exc}", file=sys.stderr)
            return 2
        _print_comparison(comparison)
        print(f"\nMachine-readable comparison: {args.output}")
        return 0

    if args.command == "ask":
        if args.preview_chars <= 0:
            print("Error: preview_chars must be positive", file=sys.stderr)
            return 2
        if not 0.0 <= args.temperature <= 2.0:
            print("Error: temperature must be between 0 and 2", file=sys.stderr)
            return 2
        try:
            generator = OllamaGenerator(
                model_name=args.model,
                timeout=args.timeout,
            )
            generator.ensure_model_available()
        except (GenerationError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

        try:
            index, embedder, retriever = _load_retriever(args)
        except (FileNotFoundError, InvalidIndexError, OSError, ValueError) as exc:
            print(f"Error: could not load the embedding index: {exc}", file=sys.stderr)
            return 2
        debug_callback = None
        if args.show_context:

            def debug_callback(results: List[SearchResult]) -> None:
                _print_retrieval(results, args.preview_chars, retriever.score_name)

        try:
            grounded_answer = answer_question(
                args.question,
                index,
                embedder,
                generator,
                top_k=args.top_k,
                temperature=args.temperature,
                on_retrieved=debug_callback,
                retriever=retriever,
            )
        except (GenerationError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

        print("\nAnswer\n")
        print(grounded_answer.text)
        print("\nSources")
        if not grounded_answer.sources:
            print("No sources retrieved.")
        for source in grounded_answer.sources:
            print(
                f"[{source.citation_number}] {source.document}, page {source.page_number}, "
                f"chunk {source.chunk_id} "
                f"({retriever.score_name}_score={source.score:.4f})"
            )
        return 0

    if args.preview_chars <= 0:
        print("Error: preview_chars must be positive", file=sys.stderr)
        return 2
    try:
        _, _, retriever = _load_retriever(args)
        results = retriever.search(args.query, top_k=args.top_k)
    except (FileNotFoundError, InvalidIndexError, OSError, ValueError) as exc:
        print(f"Error: search failed: {exc}", file=sys.stderr)
        return 2
    _print_retrieval(results, args.preview_chars, retriever.score_name)
    print(f"\nReturned {len(results)} result(s) for: {args.query}")
    return 0
