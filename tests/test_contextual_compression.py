import numpy as np
import pytest

from rag_research_assistant.compression import ContextualCompressor, Segmenter
from rag_research_assistant.compression.evaluation import evaluate_retention
from rag_research_assistant.models import Chunk, SearchResult


class FakeScorer:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def score_texts(self, query, texts):
        self.calls.append((query, list(texts)))
        if callable(self.scores):
            return np.asarray([self.scores(text) for text in texts], dtype=np.float32)
        return np.asarray(self.scores[: len(texts)], dtype=np.float32)


def result(
    text: str,
    chunk_id: str = "c1",
    document: str = "paper.pdf",
    page: int = 2,
    score: float = 0.8,
) -> SearchResult:
    return SearchResult(
        score,
        Chunk(
            chunk_id=chunk_id,
            document=document,
            page_number=page,
            chunk_index=0,
            char_start=0,
            char_end=len(text),
            text=text,
            start_page=page,
            end_page=page,
            section_title="Methods",
        ),
    )


def split_segmenter() -> Segmenter:
    return Segmenter(min_characters=1, target_characters=1)


def test_sentence_splitting_protects_abbreviations_decimals_and_numbers() -> None:
    text = "Fig. 2 reports 97.5% accuracy. Smith et al. confirm it. Final result is 42."

    segments = split_segmenter().segment(result(text), 1)

    assert [item.text for item in segments] == [
        "Fig. 2 reports 97.5% accuracy.",
        "Smith et al. confirm it.",
        "Final result is 42.",
    ]
    assert "97.5%" in segments[0].text
    assert "42" in segments[2].text


def test_segment_offsets_are_stable_and_reconstruct_exact_source_text() -> None:
    source = result("  First sentence.\n\nSecond sentence!  ")

    segments = split_segmenter().segment(source, 3)

    assert [source.chunk.text[item.start_char : item.end_char] for item in segments] == [
        item.text for item in segments
    ]
    assert [(item.source_rank, item.segment_index) for item in segments] == [(3, 0), (3, 1)]


def test_prepare_preserves_provenance_and_batches_all_segment_scores() -> None:
    scorer = FakeScorer([0.1, 0.2, 0.3])
    compressor = ContextualCompressor(scorer, segmenter=split_segmenter())
    sources = [result("One. Two.", "a", "a.pdf", 4), result("Three.", "b", "b.pdf", 8)]

    prepared = compressor.prepare("question", sources)

    assert len(scorer.calls) == 1
    assert scorer.calls[0][1] == ["One.", "Two.", "Three."]
    last = prepared.segments_by_source[1][0]
    assert (last.source_chunk_id, last.document, last.start_page, last.section_title) == (
        "b", "b.pdf", 8, "Methods"
    )


def test_per_chunk_uses_score_for_selection_but_restores_source_order() -> None:
    source = result("Alpha evidence. Distractor. Gamma evidence.")
    compressor = ContextualCompressor(
        FakeScorer([0.8, 0.1, 0.9]), segmenter=split_segmenter()
    )
    prepared = compressor.prepare("question", [source])

    compressed = compressor.compress_per_chunk(prepared, 0.75)

    assert [item.segment_index for item in compressed.contexts[0].selected_segments] == [0, 2]
    assert compressed.contexts[0].compressed_text == "Alpha evidence.\n\nGamma evidence."


def test_fixed_budget_and_compression_ratio_are_character_based() -> None:
    source = result("Relevant long sentence. Noise long sentence.")
    compressor = ContextualCompressor(
        FakeScorer([1.0, 0.0]), segmenter=split_segmenter()
    )

    compressed = compressor.compress_per_chunk(compressor.prepare("q", [source]), 0.50)

    assert len(compressed.contexts[0].selected_segments) == 1
    assert compressed.compression_ratio == pytest.approx(
        len("Relevant long sentence.") / len(source.chunk.text)
    )
    assert compressed.characters_removed == len(source.chunk.text) - len("Relevant long sentence.")


def test_per_chunk_minimum_safeguard_keeps_one_oversized_segment_per_source() -> None:
    compressor = ContextualCompressor(FakeScorer([0.1, 0.9]), segmenter=split_segmenter())
    sources = [result("Only evidence.", "a"), result("Other evidence.", "b")]

    compressed = compressor.compress_per_chunk(compressor.prepare("q", sources), 0.01)

    assert len(compressed.contexts) == 2
    assert all(len(context.selected_segments) == 1 for context in compressed.contexts)
    assert compressed.empty_output_prevention_count == 2


def test_global_minimum_safeguard_retains_one_segment_overall() -> None:
    compressor = ContextualCompressor(FakeScorer([0.1, 0.9]), segmenter=split_segmenter())
    sources = [result("Only evidence.", "a"), result("Best evidence.", "b")]

    compressed = compressor.compress_global(compressor.prepare("q", sources), 0.01)

    assert compressed.segments_retained == 1
    assert compressed.contexts[0].source_rank == 2
    assert compressed.omitted_source_ranks == (1,)
    assert compressed.empty_output_prevention_count == 1


def test_global_compression_keeps_retrieval_order_for_multiple_sources() -> None:
    scorer = FakeScorer(lambda text: {"A low.": 0.7, "A best.": 1.0, "B good.": 0.9}[text])
    compressor = ContextualCompressor(scorer, segmenter=split_segmenter())
    sources = [result("A low. A best.", "a"), result("B good.", "b")]

    compressed = compressor.compress_global(compressor.prepare("q", sources), 0.80)

    assert [context.source_rank for context in compressed.contexts] == sorted(
        context.source_rank for context in compressed.contexts
    )
    if len(compressed.contexts[0].selected_segments) == 2:
        assert [item.segment_index for item in compressed.contexts[0].selected_segments] == [0, 1]


def test_no_compression_preserves_exact_text_and_retrieval_order() -> None:
    compressor = ContextualCompressor(FakeScorer([0.2, 0.1]), segmenter=split_segmenter())
    sources = [result("First.", "a"), result("Second.", "b")]

    baseline = compressor.no_compression(compressor.prepare("q", sources))

    assert [context.source_result.chunk.chunk_id for context in baseline.contexts] == ["a", "b"]
    assert [context.compressed_text for context in baseline.contexts] == ["First.", "Second."]
    assert baseline.compression_ratio == 1.0
    assert baseline.compression_seconds >= 0.0


def test_empty_and_very_short_chunks_are_handled_without_fabricated_text() -> None:
    compressor = ContextualCompressor(FakeScorer([1.0]), segmenter=split_segmenter())
    prepared = compressor.prepare("q", [result(""), result("X", "short")])

    compressed = compressor.compress_per_chunk(prepared, 0.30)

    assert compressed.contexts[0].compressed_text == ""
    assert compressed.contexts[0].selected_segments == ()
    assert compressed.contexts[1].compressed_text == "X"


def test_retention_uses_matching_document_page_and_labels_proxy() -> None:
    scorer = FakeScorer([0.9, 0.1])
    compressor = ContextualCompressor(scorer, segmenter=split_segmenter())
    prepared = compressor.prepare(
        "What accuracy was reported?",
        [result("The model reports 97.5% accuracy. Unrelated text.")],
    )
    baseline = compressor.no_compression(prepared)
    compressed = compressor.compress_per_chunk(prepared, 0.75)
    evidence = [{"document": "paper.pdf", "pages": [2], "passage": "model reports 97.5% accuracy"}]

    metrics = evaluate_retention(
        "What accuracy was reported?", evidence, baseline, compressed
    )

    assert metrics["gold_evidence_term_coverage_compressed"] == 1.0
    assert metrics["gold_evidence_term_retention_conditional"] == 1.0
    assert "proxy" in metrics["metric_warning"].lower()


def test_invalid_segment_scores_are_rejected() -> None:
    compressor = ContextualCompressor(FakeScorer([float("nan")]), segmenter=split_segmenter())

    with pytest.raises(ValueError, match="invalid scores"):
        compressor.prepare("q", [result("Evidence.")])
