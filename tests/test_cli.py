import numpy as np
from types import SimpleNamespace

from rag_research_assistant import cli
from rag_research_assistant.generation import OllamaUnavailableError
from rag_research_assistant.index import EmbeddingIndex
from rag_research_assistant.evaluation import EvaluationExample, ExpectedSource
from rag_research_assistant.models import Chunk
from rag_research_assistant.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankingRetriever,
)


class FakeQueryEmbedder:
    model_name = "test/model"

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


def test_search_command_displays_score_and_chunk_metadata(monkeypatch, capsys) -> None:
    chunk = Chunk("paper.pdf:p4:c2", "paper.pdf", 4, 2, 10, 24, "Relevant passage")
    index = EmbeddingIndex(
        embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        chunks=[chunk],
        model_name="test/model",
    )
    monkeypatch.setattr(cli, "load_index", lambda chunks_path, index_dir: index)
    monkeypatch.setattr(
        cli,
        "SentenceTransformerEmbedder",
        lambda **kwargs: FakeQueryEmbedder(),
    )

    assert cli.main(["search", "natural language question", "--top-k", "1"]) == 0

    output = capsys.readouterr().out
    assert "score=1.0000" in output
    assert "source=paper.pdf page=4 chunk=paper.pdf:p4:c2" in output
    assert "Relevant passage" in output


def test_ask_command_reports_unavailable_ollama_without_traceback(
    monkeypatch, capsys
) -> None:
    class UnavailableGenerator:
        def __init__(self, **kwargs) -> None:
            pass

        def ensure_model_available(self) -> None:
            raise OllamaUnavailableError("Install Ollama with the documented command")

    monkeypatch.setattr(cli, "OllamaGenerator", UnavailableGenerator)

    assert cli.main(["ask", "What is RAG?"]) == 2

    captured = capsys.readouterr()
    assert "Error: Install Ollama" in captured.err
    assert "Traceback" not in captured.err


def test_evaluate_command_requires_depth_needed_for_hit_at_five(capsys) -> None:
    assert cli.main(["evaluate", "--top-k", "3"]) == 2

    captured = capsys.readouterr()
    assert "top_k must be at least 5" in captured.err


def test_retrieval_only_evaluate_command_does_not_construct_ollama(
    monkeypatch, tmp_path, capsys
) -> None:
    chunk = Chunk("paper:p1:c1", "paper.pdf", 1, 1, 0, 8, "Evidence")
    index = EmbeddingIndex(
        embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        chunks=[chunk],
        model_name="test/model",
    )
    example = EvaluationExample(
        "one",
        "Question",
        "answerable",
        (ExpectedSource("paper.pdf", 1),),
    )
    monkeypatch.setattr(cli, "load_evaluation_dataset", lambda path: [example])
    monkeypatch.setattr(cli, "load_index", lambda chunks_path, index_dir: index)
    monkeypatch.setattr(
        cli,
        "SentenceTransformerEmbedder",
        lambda **kwargs: FakeQueryEmbedder(),
    )

    def unexpected_ollama(**kwargs):
        raise AssertionError("retrieval-only evaluation must not construct Ollama")

    monkeypatch.setattr(cli, "OllamaGenerator", unexpected_ollama)
    output = tmp_path / "report.json"

    assert cli.main(["evaluate", "--output", str(output)]) == 0

    captured = capsys.readouterr()
    assert "Hit@1: 100.0%" in captured.out
    assert "Not run (use --with-generation)" in captured.out
    assert output.exists()


def test_retriever_modes_construct_expected_strategy(monkeypatch) -> None:
    chunk = Chunk("one", "paper.pdf", 1, 1, 0, 8, "Evidence")
    index = EmbeddingIndex(
        embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        chunks=[chunk],
        model_name="test/model",
    )
    monkeypatch.setattr(cli, "load_index", lambda chunks, path: index)
    monkeypatch.setattr(cli, "SentenceTransformerEmbedder", lambda **kwargs: FakeQueryEmbedder())

    def args(strategy, rerank=False):
        return SimpleNamespace(
            retriever=strategy,
            rerank=rerank,
            candidate_depth=20,
            reranker_model="fake/reranker",
            chunks="chunks.jsonl",
            index="index",
            device=None,
        )

    assert isinstance(cli._load_retriever(args("dense"))[2], DenseRetriever)
    assert isinstance(cli._load_retriever(args("bm25"))[2], BM25Retriever)
    assert isinstance(cli._load_retriever(args("hybrid"))[2], HybridRetriever)

    fake_reranker = SimpleNamespace(model_name="fake/reranker", rerank=lambda *a: [])
    monkeypatch.setattr(cli, "CrossEncoderReranker", lambda **kwargs: fake_reranker)
    assert isinstance(
        cli._load_retriever(args("hybrid", rerank=True))[2], RerankingRetriever
    )


def test_cli_rejects_reranking_without_hybrid(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "load_index",
        lambda *args: (_ for _ in ()).throw(AssertionError("must validate first")),
    )

    assert cli.main(["search", "question", "--retriever", "dense", "--rerank"]) == 2
    assert "--rerank requires --retriever hybrid" in capsys.readouterr().err
