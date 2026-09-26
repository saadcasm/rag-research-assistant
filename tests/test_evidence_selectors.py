import json
import numpy as np
from rag_research_assistant.evidence_selection.selectors import (
    cosine_similarity, extract_query_facets, rank_normalized_relevance,
    select_evidence, selection_diagnostics,
)
from rag_research_assistant.models import Chunk, SearchResult


def result(i,text,doc="a.pdf"):
    return SearchResult(float(10-i),Chunk(f"c{i}",doc,1,i,0,len(text),text))


def test_rank_normalization_and_facets_are_deterministic_and_deduplicated():
    assert rank_normalized_relevance(3).tolist()==[1.0,0.5,0.0]
    assert extract_query_facets("How do DPR and ColBERT compare with DPR?")==("DPR","ColBERT")


def test_cosine_and_embedding_reuse_drive_diversity_without_document_quota():
    candidates=[result(0,"DPR alpha"),result(1,"DPR duplicate"),result(2,"ColBERT beta")]
    vectors={"c0":np.array([1.,0.]),"c1":np.array([1.,0.]),"c2":np.array([0.,1.])}
    assert cosine_similarity(vectors["c0"],vectors["c1"])==1.0
    selected=select_evidence("DPR ColBERT",candidates,vectors,top_k=3,strategy="diversity",diversity_weight=0.6)
    assert selected.selected[0].chunk.chunk_id=="c0"
    assert len(selected.selected)==3
    assert [x.chunk.document for x in selected.selected].count("a.pdf")==3


def test_coverage_and_combined_selectors_track_auditable_decisions():
    candidates=[result(0,"DPR details"),result(1,"more DPR"),result(2,"ColBERT details")]
    vectors={f"c{i}":np.eye(3)[i] for i in range(3)}
    coverage=select_evidence("How do DPR and ColBERT differ?",candidates,vectors,top_k=2,strategy="coverage",coverage_weight=1.2)
    combined=select_evidence("How do DPR and ColBERT differ?",candidates,vectors,top_k=2,strategy="coverage_diversity",coverage_weight=1.2,diversity_weight=0.2)
    assert [x.chunk.chunk_id for x in coverage.selected]==["c0","c2"]
    assert coverage.decisions[1].new_facets_added==("ColBERT",)
    assert len(combined.selected)==2
    assert json.loads(json.dumps(combined.to_dict()))["decisions"][0]["original_rank"]==1


def test_missing_candidate_vector_is_rejected_and_diagnostics_are_serializable():
    candidates=[result(0,"DPR"),result(1,"ColBERT","b.pdf")]
    vectors={"c0":np.array([1.,0.]),"c1":np.array([0.,1.])}
    selected=select_evidence("DPR ColBERT",candidates,vectors,top_k=2,strategy="baseline")
    metrics=selection_diagnostics(selected.selected,vectors)
    assert metrics["unique_documents"]==2
    assert metrics["average_pairwise_similarity"]==0.0
    try:
        select_evidence("DPR",candidates,{"c0":vectors["c0"]},top_k=2,strategy="baseline")
        assert False
    except ValueError as exc:
        assert "embedding" in str(exc)


def test_selector_api_has_no_gold_label_input():
    import inspect
    assert "example" not in inspect.signature(select_evidence).parameters
    assert "hop" not in inspect.signature(select_evidence).parameters
