# Phase 10I-2: Controlled Candidate-Depth Comparison

> Experimental only. Production candidate depth remains unchanged.

## Multi-hop final reranked coverage

| Depth | Candidates reranked | Full@3 | Full@5 | Full@10 | Partial@3 | Partial@5 | Partial@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 20.0 | 0.0769 | 0.1154 | 0.2308 | 0.2115 | 0.3077 | 0.5192 |
| 50 | 50.0 | 0.0385 | 0.1154 | 0.2692 | 0.1538 | 0.2885 | 0.4423 |
| 100 | 100.0 | 0.0385 | 0.0769 | 0.1923 | 0.1538 | 0.2500 | 0.4038 |

## Candidate-pool oracle

| Depth | Full | Partial |
|---:|---:|---:|
| 20 | 0.3077 | 0.5962 |
| 50 | 0.5385 | 0.7500 |
| 100 | 0.7308 | 0.8718 |

## Frozen 100-question regression

| Depth | Hit@1 | Hit@3 | Hit@5 | Recall@1 | Recall@3 | Recall@5 | MFR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.4556 | 0.7222 | 0.8444 | 0.4500 | 0.7167 | 0.8222 | 2.4167 |
| 50 | 0.4556 | 0.7222 | 0.8556 | 0.4500 | 0.7167 | 0.8389 | 2.3452 |
| 100 | 0.4444 | 0.7111 | 0.8444 | 0.4389 | 0.7056 | 0.8278 | 2.3614 |

## Movements and latency

```json
{
  "movements": {
    "50": {
      "improved": [
        "p10g_splade_deepimpact_sparse"
      ],
      "unchanged": [
        "p10g_dpr_ance_negatives",
        "p10g_coil_colbert_interactions",
        "p10g_hyde_query2doc",
        "p10g_rag_replug_integration",
        "p10g_beir_bright_focus",
        "p10g_dpr_colbert_architecture",
        "p10g_sbert_simcse_representation",
        "p10g_lostmiddle_ruler_longcontext",
        "p10g_raptor_graphrag_structure",
        "p10g_hnsw_faiss_ann",
        "p10g_realm_rag_memory",
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
        "p10g_bert_monot5_reranking",
        "p10g_selfrag_crag_control",
        "p10g_latechunk_texttiling",
        "p10g_beir_mteb_scope",
        "p10g_gtr_sgpt_scaling"
      ],
      "transitions": {
        "0-hop_to_0-hop": 5,
        "1-hop_to_1-hop": 13,
        "1-hop_to_full-hop": 1,
        "2-hop_to_2-hop": 2,
        "partial-hop_to_0-hop": 5
      }
    },
    "100": {
      "improved": [],
      "unchanged": [
        "p10g_dpr_ance_negatives",
        "p10g_coil_colbert_interactions",
        "p10g_hyde_query2doc",
        "p10g_rag_replug_integration",
        "p10g_beir_bright_focus",
        "p10g_dpr_colbert_architecture",
        "p10g_splade_deepimpact_sparse",
        "p10g_sbert_simcse_representation",
        "p10g_raptor_graphrag_structure",
        "p10g_hnsw_faiss_ann",
        "p10g_realm_rag_memory",
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
        "p10g_bert_monot5_reranking",
        "p10g_lostmiddle_ruler_longcontext",
        "p10g_selfrag_crag_control",
        "p10g_latechunk_texttiling",
        "p10g_beir_mteb_scope",
        "p10g_gtr_sgpt_scaling"
      ],
      "transitions": {
        "0-hop_to_0-hop": 5,
        "1-hop_to_1-hop": 14,
        "2-hop_to_2-hop": 1,
        "full-hop_to_partial-hop": 1,
        "partial-hop_to_0-hop": 5
      }
    }
  },
  "multihop_latency": {
    "20": {
      "dense": {
        "mean": 0.012969458460597357,
        "p50": 0.008133166993502527,
        "p95": 0.008947541995439678
      },
      "bm25": {
        "mean": 0.013905416577472351,
        "p50": 0.013586832996224985,
        "p95": 0.01617291700677015
      },
      "rrf": {
        "mean": 3.278530829657729e-05,
        "p50": 3.1167000997811556e-05,
        "p95": 3.749999450519681e-05
      },
      "reranker": {
        "mean": 0.14296738303808246,
        "p50": 0.13959704199805856,
        "p95": 0.16383104099077173
      },
      "total": {
        "mean": 0.16987504338444873,
        "p50": 0.16216683300444856,
        "p95": 0.186887581992778
      }
    },
    "50": {
      "dense": {
        "mean": 0.00853750315526178,
        "p50": 0.00844333300483413,
        "p95": 0.00885291698796209
      },
      "bm25": {
        "mean": 0.013649615230231294,
        "p50": 0.013347874992177822,
        "p95": 0.01568224999937229
      },
      "rrf": {
        "mean": 6.351765366092038e-05,
        "p50": 6.0165999457240105e-05,
        "p95": 7.329099753405899e-05
      },
      "reranker": {
        "mean": 0.34299452411505627,
        "p50": 0.340538958000252,
        "p95": 0.37675279199902434
      },
      "total": {
        "mean": 0.36524516015421027,
        "p50": 0.36249537400726695,
        "p95": 0.3984822499915026
      }
    },
    "100": {
      "dense": {
        "mean": 0.009227528882687554,
        "p50": 0.009153249993687496,
        "p95": 0.00959737499943003
      },
      "bm25": {
        "mean": 0.018537681077409964,
        "p50": 0.013622625003335997,
        "p95": 0.015701333002652973
      },
      "rrf": {
        "mean": 0.00011337188446374897,
        "p50": 0.00010683400614652783,
        "p95": 0.00013087499246466905
      },
      "reranker": {
        "mean": 0.6703824135015916,
        "p50": 0.6544847500044852,
        "p95": 0.7292457919975277
      },
      "total": {
        "mean": 0.6982609953461528,
        "p50": 0.6797762089845492,
        "p95": 0.7523847089905757
      }
    }
  }
}
```

## Interpretation status

No production depth decision is encoded. Review oracle, reranked coverage, regression, question movement, and latency together.
