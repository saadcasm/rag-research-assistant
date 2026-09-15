# Phase 10B: multi-query retrieval results

The frozen baseline is compared with original+2 and original+3 generated alternatives. Each query receives dense+BM25 hybrid retrieval; those rankings are fused before one unchanged cross-encoder rerank against the original question.

Dataset: `data/evaluation/questions-phase-7.5.jsonl`
Generation model: `qwen3.5:4b`

## Aggregate retrieval metrics

| Metric | Baseline | Multi-query-2 | Multi-query-3 |
|---|---:|---:|---:|
| Hit@1 | 0.4556 | 0.4556 | 0.4556 |
| Hit@3 | 0.7222 | 0.7333 | 0.7333 |
| Hit@5 | 0.8444 | 0.8667 | 0.8667 |
| Recall@1 | 0.4500 | 0.4500 | 0.4500 |
| Recall@3 | 0.7167 | 0.7278 | 0.7278 |
| Recall@5 | 0.8222 | 0.8444 | 0.8444 |
| Mean first relevant rank | 1.8947 | 1.9359 | 1.9487 |

## Behavior and cost

- Questions: 100 (90 scored; 10 unanswerable)
- MQ2 improved/degraded/unchanged: 5/0/85
- MQ3 improved/degraded/unchanged: 4/0/86
- Generation failures: 9; MQ2/MQ3 shortfall questions: 9/25
- Average valid variants: 2.57; rejected variants: 16
- Drift-warning questions: 45
- Average generation: 6.336s (p50 6.462s; p95 7.635s)
- Average baseline/MQ2/MQ3 retrieval: 0.171s / 0.304s / 0.222s
- Average MQ2/MQ3 total: 6.640s / 6.558s

## Validation

The result is internally consistent with the frozen experiment: the dataset SHA-256 is `36e9707071d722e327ff1987118caff1fc71d646dbeef77d139b32aa1445fcbe`, baseline metrics exactly reproduce Phase 10A, and the index/model/RRF/candidate settings are unchanged.

## Informative rank changes

| ID | Baseline | MQ2 | MQ3 | Original question |
|---|---:|---:|---:|---|
| `p75_beir_composition` | not found | 5 | 5 | How broad is BEIR in terms of datasets and retrieval architectures evaluated? |
| `p75_monot5_reformulation` | 3 | 2 | 3 | How does monoT5 cast document ranking as a sequence-to-sequence task? |
| `p75_rankgpt_noveleval` | not found | 5 | 5 | Why did RankGPT introduce the NovelEval test set? |
| `p75_realm_signal` | 4 | 3 | 3 | What learning signal allows REALM to pretrain its retriever without supervised retrieval labels? |
| `p75_hyde_false_details` | 5 | 4 | 4 | How does HyDE limit the effect of false details in its generated hypothetical document? |

MQ2 improved five scored questions and MQ3 improved four. The third variant moved `p75_monot5_reformulation` back from rank 2 to its baseline rank 3, although it did not create a baseline-relative degradation. The gains covered one exact-number, three section-detail, and one difficult-distractor question for MQ2.

## Generation reliability

Nine questions fell back to original-only retrieval because Qwen returned invalid JSON:

- `p75_doct5query_direction`
- `p75_lost_middle_position`
- `p75_query2doc_bm25_gain`
- `p75_beir_tradeoff`
- `p75_deepimpact_token_value`
- `p75_e5_evaluation`
- `p75_fid_aggregation`
- `p75_mteb_no_winner`
- `p75_unanswerable_cost`

The fallback behaved correctly: all nine produced the same final chunk ordering as baseline. Another 16 responses repeated the original query as one generated variant; those duplicates were rejected safely, leaving two valid alternatives. Consequently MQ2 had 9 shortfalls and MQ3 had 25.

The 45 drift-warning questions are not 45 confirmed semantic errors. The intentionally lightweight Phase 10A detector treats many hyphenated phrases as model-like tokens, so warnings such as a dropped `long-document` phrase are high-recall audit signals. None of the five MQ2 improvements carried a warning, and no warned query produced a measured baseline-relative degradation in this run.

## Latency interpretation

Baseline retrieval averaged 0.171 seconds. Adding generation raised average end-to-end latency to 6.640 seconds for MQ2 and 6.558 seconds for MQ3—about 39 times baseline—with generation alone averaging 6.336 seconds.

MQ2's measured retrieval time was unexpectedly above MQ3's despite executing fewer queries. The conditions ran in a fixed MQ2-then-MQ3 order, so model/index warm-up and caching can bias this micro-timing. It is not evidence that three alternatives are cheaper. Generation dominates either condition, making the broad cost conclusion reliable even though the MQ2-versus-MQ3 retrieval delta is not.

The slightly worse mean first-relevant rank does not contradict the absence of degraded questions. The mean includes only questions where evidence was found: recovering previously missed evidence at ranks 4–5 adds those values to the denominator, increasing the mean while Hit@5 improves.

## Recommendation

The original-query safeguard solved the destructive-replacement failure mode observed in Phase 10A for this run: no scored question degraded relative to baseline. MQ2 increased Hit@3 from 0.7222 to 0.7333 and Hit@5 from 0.8444 to 0.8667, with matching recall gains.

Keep multi-query-2 as an optional experimental recall mode, not the production default. Its gain is real but modest, generation dominates latency, and 9% of model responses violated the JSON contract. Multi-query-3 is not justified by this benchmark: it produced no aggregate improvement over MQ2, had more shortfalls, and lost one of MQ2's rank improvements.

The companion JSON retains complete per-query hybrid rankings, fused candidates, generated/rejected variants, final results, configuration, failures, and latency records. Phase 10B does not alter the production retrieval path.
