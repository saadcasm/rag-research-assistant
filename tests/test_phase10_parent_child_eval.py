from rag_research_assistant.evaluation import EvaluationExample, ExpectedSource
from rag_research_assistant.experiments.phase10_parent_child_eval import (
    evaluate_experiment,
)
from rag_research_assistant.models import Chunk, PageText, SearchResult
from rag_research_assistant.parent_child.index import build_parent_index
from rag_research_assistant.parent_child.retrieval import ParentChildRetriever


def chunk(chunk_id: str, document: str, text: str) -> Chunk:
    return Chunk(
        chunk_id, document, 1, 1, 0, len(text), text,
        1, 1, "Methods", "semantic",
    )


class QueryRetriever:
    name = "hybrid+rerank"
    score_name = "cross-encoder"

    def __init__(self, result):
        self.result = result

    def search(self, query, top_k=5):
        return [self.result]


def test_question_report_classifies_child_improvement_and_parent_ranks() -> None:
    right = chunk("right", "right.pdf", "relevant evidence")
    wrong = chunk("wrong", "wrong.pdf", "wrong evidence")
    pages = [
        PageText("right.pdf", 1, "Methods\nrelevant evidence and context"),
    ]
    structural = [
        Chunk("s1", "right.pdf", 1, 1, 0, 29, "relevant evidence and context", 1, 1, "Methods", "structural")
    ]
    index = build_parent_index([right], pages, structural)
    parent_child = ParentChildRetriever(QueryRetriever(SearchResult(2.0, right)), index)
    example = EvaluationExample(
        "q1", "question", "answerable", (ExpectedSource("right.pdf", 1),)
    )

    report = evaluate_experiment(
        [example],
        QueryRetriever(SearchResult(1.0, wrong)),
        parent_child,
        dataset_path="frozen.jsonl",
    )

    item = report["questions"][0]
    assert item["baseline_first_relevant_rank"] is None
    assert item["child_first_relevant_rank"] == 1
    assert item["page_parent_first_relevant_rank"] == 1
    assert item["structural_parent_first_relevant_rank"] == 1
    assert item["child_vs_baseline_outcome"] == "improved"
    assert item["page_parent_vs_baseline_outcome"] == "improved"
    assert item["structural_parent_vs_child_outcome"] == "unchanged"
    assert report["summary"]["child_improved"] == 1
    assert report["summary"]["page_parent"]["hit_at_1"] == 1.0
    assert report["summary"]["structural_parent"]["relevant_child_containment_rate"] == 1.0
