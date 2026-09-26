# Phase 10J: Coverage-Aware Evidence-Set Selection

> Experimental only; production is unchanged.

## Multi-hop coverage

| Depth | Strategy | Oracle | Full@3 | Full@5 | Full@10 | Partial@10 | Selector ms | Total s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 20 | baseline | 0.3077 | 0.0769 | 0.1154 | 0.2308 | 0.5192 | 1.674 | 0.170 |
| 20 | diversity | 0.3077 | 0.0385 | 0.0769 | 0.2308 | 0.5192 | 1.591 | 0.170 |
| 20 | coverage | 0.3077 | 0.0769 | 0.1154 | 0.2308 | 0.5192 | 1.584 | 0.170 |
| 20 | coverage_diversity | 0.3077 | 0.0385 | 0.1154 | 0.2308 | 0.5192 | 1.590 | 0.170 |
| 50 | baseline | 0.5385 | 0.0385 | 0.1154 | 0.2692 | 0.4423 | 4.959 | 0.372 |
| 50 | diversity | 0.5385 | 0.0385 | 0.0385 | 0.1538 | 0.3846 | 4.866 | 0.371 |
| 50 | coverage | 0.5385 | 0.0385 | 0.1154 | 0.2692 | 0.4423 | 4.849 | 0.371 |
| 50 | coverage_diversity | 0.5385 | 0.0385 | 0.0385 | 0.2308 | 0.4231 | 4.840 | 0.371 |

## Frozen regression

```json
{
  "20": {
    "baseline": {
      "hit_at_1": 0.45555555555555555,
      "hit_at_3": 0.7222222222222222,
      "hit_at_5": 0.8444444444444444,
      "recall_at_1": 0.45,
      "recall_at_3": 0.7166666666666667,
      "recall_at_5": 0.8222222222222222,
      "mean_first_correct_rank": 2.4166666666666665
    },
    "diversity": {
      "hit_at_1": 0.45555555555555555,
      "hit_at_3": 0.7,
      "hit_at_5": 0.8111111111111111,
      "recall_at_1": 0.45,
      "recall_at_3": 0.6944444444444444,
      "recall_at_5": 0.7888888888888889,
      "mean_first_correct_rank": 2.5476190476190474
    },
    "coverage": {
      "hit_at_1": 0.45555555555555555,
      "hit_at_3": 0.7333333333333333,
      "hit_at_5": 0.8444444444444444,
      "recall_at_1": 0.45,
      "recall_at_3": 0.7277777777777777,
      "recall_at_5": 0.8277777777777777,
      "mean_first_correct_rank": 2.380952380952381
    },
    "coverage_diversity": {
      "hit_at_1": 0.45555555555555555,
      "hit_at_3": 0.7333333333333333,
      "hit_at_5": 0.8444444444444444,
      "recall_at_1": 0.45,
      "recall_at_3": 0.7277777777777777,
      "recall_at_5": 0.8277777777777777,
      "mean_first_correct_rank": 2.392857142857143
    }
  },
  "50": {
    "baseline": {
      "hit_at_1": 0.45555555555555555,
      "hit_at_3": 0.7222222222222222,
      "hit_at_5": 0.8555555555555555,
      "recall_at_1": 0.45,
      "recall_at_3": 0.7166666666666667,
      "recall_at_5": 0.8388888888888889,
      "mean_first_correct_rank": 2.3452380952380953
    },
    "diversity": {
      "hit_at_1": 0.45555555555555555,
      "hit_at_3": 0.6222222222222222,
      "hit_at_5": 0.8222222222222222,
      "recall_at_1": 0.45,
      "recall_at_3": 0.6166666666666667,
      "recall_at_5": 0.8,
      "mean_first_correct_rank": 2.7023809523809526
    },
    "coverage": {
      "hit_at_1": 0.45555555555555555,
      "hit_at_3": 0.7333333333333333,
      "hit_at_5": 0.8555555555555555,
      "recall_at_1": 0.45,
      "recall_at_3": 0.7277777777777777,
      "recall_at_5": 0.8388888888888889,
      "mean_first_correct_rank": 2.3333333333333335
    },
    "coverage_diversity": {
      "hit_at_1": 0.45555555555555555,
      "hit_at_3": 0.7111111111111111,
      "hit_at_5": 0.8222222222222222,
      "recall_at_1": 0.45,
      "recall_at_3": 0.7055555555555556,
      "recall_at_5": 0.8111111111111111,
      "mean_first_correct_rank": 2.4642857142857144
    }
  }
}
```

## Movements

```json
{
  "20": {
    "diversity": {
      "improved": [],
      "unchanged": [
        "p10g_dpr_ance_negatives",
        "p10g_coil_colbert_interactions",
        "p10g_hyde_query2doc",
        "p10g_rag_replug_integration",
        "p10g_beir_bright_focus",
        "p10g_dpr_colbert_architecture",
        "p10g_splade_deepimpact_sparse",
        "p10g_bert_monot5_reranking",
        "p10g_sbert_simcse_representation",
        "p10g_lostmiddle_ruler_longcontext",
        "p10g_selfrag_crag_control",
        "p10g_raptor_graphrag_structure",
        "p10g_hnsw_faiss_ann",
        "p10g_realm_rag_memory",
        "p10g_latechunk_texttiling",
        "p10g_beir_mteb_scope",
        "p10g_gtr_sgpt_scaling",
        "p10g_colbert_versions",
        "p10g_rewrite_query2doc",
        "p10g_fid_replug_evidence_use",
        "p10g_dpr_two_stage_fact",
        "p10g_raptor_build_use",
        "p10g_partial_colbert_plaid",
        "p10g_partial_splade_unicoil",
        "p10g_partial_hnsw_diskann",
        "p10g_partial_beir_miracl"
      ],
      "degraded": []
    },
    "coverage": {
      "improved": [],
      "unchanged": [
        "p10g_dpr_ance_negatives",
        "p10g_coil_colbert_interactions",
        "p10g_hyde_query2doc",
        "p10g_rag_replug_integration",
        "p10g_beir_bright_focus",
        "p10g_dpr_colbert_architecture",
        "p10g_splade_deepimpact_sparse",
        "p10g_bert_monot5_reranking",
        "p10g_sbert_simcse_representation",
        "p10g_lostmiddle_ruler_longcontext",
        "p10g_selfrag_crag_control",
        "p10g_raptor_graphrag_structure",
        "p10g_hnsw_faiss_ann",
        "p10g_realm_rag_memory",
        "p10g_latechunk_texttiling",
        "p10g_beir_mteb_scope",
        "p10g_gtr_sgpt_scaling",
        "p10g_colbert_versions",
        "p10g_rewrite_query2doc",
        "p10g_fid_replug_evidence_use",
        "p10g_dpr_two_stage_fact",
        "p10g_raptor_build_use",
        "p10g_partial_colbert_plaid",
        "p10g_partial_splade_unicoil",
        "p10g_partial_hnsw_diskann",
        "p10g_partial_beir_miracl"
      ],
      "degraded": []
    },
    "coverage_diversity": {
      "improved": [],
      "unchanged": [
        "p10g_dpr_ance_negatives",
        "p10g_coil_colbert_interactions",
        "p10g_hyde_query2doc",
        "p10g_rag_replug_integration",
        "p10g_beir_bright_focus",
        "p10g_dpr_colbert_architecture",
        "p10g_splade_deepimpact_sparse",
        "p10g_bert_monot5_reranking",
        "p10g_sbert_simcse_representation",
        "p10g_lostmiddle_ruler_longcontext",
        "p10g_selfrag_crag_control",
        "p10g_raptor_graphrag_structure",
        "p10g_hnsw_faiss_ann",
        "p10g_realm_rag_memory",
        "p10g_latechunk_texttiling",
        "p10g_beir_mteb_scope",
        "p10g_gtr_sgpt_scaling",
        "p10g_colbert_versions",
        "p10g_rewrite_query2doc",
        "p10g_fid_replug_evidence_use",
        "p10g_dpr_two_stage_fact",
        "p10g_raptor_build_use",
        "p10g_partial_colbert_plaid",
        "p10g_partial_splade_unicoil",
        "p10g_partial_hnsw_diskann",
        "p10g_partial_beir_miracl"
      ],
      "degraded": []
    }
  },
  "50": {
    "diversity": {
      "improved": [],
      "unchanged": [
        "p10g_dpr_ance_negatives",
        "p10g_coil_colbert_interactions",
        "p10g_hyde_query2doc",
        "p10g_rag_replug_integration",
        "p10g_beir_bright_focus",
        "p10g_dpr_colbert_architecture",
        "p10g_bert_monot5_reranking",
        "p10g_sbert_simcse_representation",
        "p10g_selfrag_crag_control",
        "p10g_raptor_graphrag_structure",
        "p10g_realm_rag_memory",
        "p10g_latechunk_texttiling",
        "p10g_beir_mteb_scope",
        "p10g_gtr_sgpt_scaling",
        "p10g_colbert_versions",
        "p10g_rewrite_query2doc",
        "p10g_fid_replug_evidence_use",
        "p10g_dpr_two_stage_fact",
        "p10g_raptor_build_use",
        "p10g_partial_colbert_plaid",
        "p10g_partial_splade_unicoil",
        "p10g_partial_hnsw_diskann",
        "p10g_partial_beir_miracl"
      ],
      "degraded": [
        "p10g_splade_deepimpact_sparse",
        "p10g_lostmiddle_ruler_longcontext",
        "p10g_hnsw_faiss_ann"
      ]
    },
    "coverage": {
      "improved": [],
      "unchanged": [
        "p10g_dpr_ance_negatives",
        "p10g_coil_colbert_interactions",
        "p10g_hyde_query2doc",
        "p10g_rag_replug_integration",
        "p10g_beir_bright_focus",
        "p10g_dpr_colbert_architecture",
        "p10g_splade_deepimpact_sparse",
        "p10g_bert_monot5_reranking",
        "p10g_sbert_simcse_representation",
        "p10g_lostmiddle_ruler_longcontext",
        "p10g_selfrag_crag_control",
        "p10g_raptor_graphrag_structure",
        "p10g_hnsw_faiss_ann",
        "p10g_realm_rag_memory",
        "p10g_latechunk_texttiling",
        "p10g_beir_mteb_scope",
        "p10g_gtr_sgpt_scaling",
        "p10g_colbert_versions",
        "p10g_rewrite_query2doc",
        "p10g_fid_replug_evidence_use",
        "p10g_dpr_two_stage_fact",
        "p10g_raptor_build_use",
        "p10g_partial_colbert_plaid",
        "p10g_partial_splade_unicoil",
        "p10g_partial_hnsw_diskann",
        "p10g_partial_beir_miracl"
      ],
      "degraded": []
    },
    "coverage_diversity": {
      "improved": [],
      "unchanged": [
        "p10g_dpr_ance_negatives",
        "p10g_coil_colbert_interactions",
        "p10g_hyde_query2doc",
        "p10g_rag_replug_integration",
        "p10g_beir_bright_focus",
        "p10g_dpr_colbert_architecture",
        "p10g_splade_deepimpact_sparse",
        "p10g_bert_monot5_reranking",
        "p10g_sbert_simcse_representation",
        "p10g_lostmiddle_ruler_longcontext",
        "p10g_selfrag_crag_control",
        "p10g_raptor_graphrag_structure",
        "p10g_realm_rag_memory",
        "p10g_latechunk_texttiling",
        "p10g_beir_mteb_scope",
        "p10g_gtr_sgpt_scaling",
        "p10g_colbert_versions",
        "p10g_rewrite_query2doc",
        "p10g_fid_replug_evidence_use",
        "p10g_dpr_two_stage_fact",
        "p10g_raptor_build_use",
        "p10g_partial_colbert_plaid",
        "p10g_partial_splade_unicoil",
        "p10g_partial_hnsw_diskann",
        "p10g_partial_beir_miracl"
      ],
      "degraded": [
        "p10g_hnsw_faiss_ann"
      ]
    }
  }
}
```

## Interpretation status

The generated metrics are reviewed in `docs/phase-10j-coverage-reranking.md`.
No selector is promoted; production remains unchanged.
