import numpy as np

from rag_research_assistant import cli
from rag_research_assistant.index import EmbeddingIndex
from rag_research_assistant.models import Chunk


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
