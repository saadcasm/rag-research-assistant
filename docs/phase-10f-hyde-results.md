# Phase 10F: HyDE results

HyDE generates one answer-like passage as a semantic probe. The hypothetical text is never evidence, and every final cross-encoder call uses the original question against real legacy chunks.

Questions: 100 (90 scored; 10 unanswerable)

| Metric | Baseline | HyDE-only | Baseline + HyDE |
|---|---:|---:|---:|
| Hit@1 | 0.4556 | 0.4556 | 0.4556 |
| Hit@3 | 0.7222 | 0.7333 | 0.7222 |
| Hit@5 | 0.8444 | 0.8333 | 0.8444 |
| Recall@1 | 0.4500 | 0.4500 | 0.4500 |
| Recall@3 | 0.7167 | 0.7278 | 0.7167 |
| Recall@5 | 0.8222 | 0.8167 | 0.8222 |
| Mean first relevant rank | 1.8947 | 1.8267 | 1.8947 |

## Category Hit@5

| Category | Questions | Baseline | HyDE-only | Fused |
|---|---:|---:|---:|---:|
| comparison | 3 | 0.6667 | 0.6667 | 0.6667 |
| cross_page_context | 3 | 0.6667 | 0.6667 | 0.6667 |
| difficult_distractor | 19 | 0.8421 | 0.8421 | 0.8421 |
| exact_number | 23 | 0.8261 | 0.7391 | 0.8261 |
| exact_terminology | 13 | 1.0000 | 0.9231 | 1.0000 |
| multi_source | 5 | 0.6000 | 0.6000 | 0.6000 |
| section_detail | 13 | 0.8462 | 1.0000 | 0.8462 |
| semantic_paraphrase | 11 | 0.9091 | 0.9091 | 0.9091 |

## Rank movement

- HyDE-only improved/unchanged/degraded: 17/66/7
- Fused improved/unchanged/degraded: 0/90/0
- HyDE-only rescued/lost top-five evidence: 4/5
- Fused rescued/lost top-five evidence: 0/0

## Probe and latency

- Average hypothetical length: 92.0 words / 709.5 characters
- Fallbacks/errors/truncations: 0/0/0
- Missing-protected-term / numeric-detail / citation-like output questions: 9/33/0
- Introduced-entity / any-warning questions: 100/39
- Mean baseline retrieval: 0.198s
- Mean generation / embedding / HyDE dense retrieval: 4.077s / 0.009s / 0.003s
- Mean HyDE-only / fused total: 4.228s / 4.422s
- Unanswerable top-result changes, HyDE-only/fused: 8/0
- Mean unanswerable fused overlap with baseline top five: 4.80/5

## Interpretation boundary

Do not infer a recommendation from this generated table alone. Inspect rescues, regressions, numeric/entity drift, unanswerable probes, and fusion safeguards before deciding whether HyDE merits retention as an optional experiment.

## Analysis

HyDE-only does not beat the frozen baseline. Hit@3 increases from 0.7222 to 0.7333, but Hit@5 falls from 0.8444 to 0.8333 and Recall@5 falls from 0.8222 to 0.8167. It improves 17 questions, degrades 7, and leaves 66 unchanged. More importantly, it rescues four top-five misses but turns five baseline top-five hits into misses. Its lower mean first-relevant rank (1.8267 versus 1.8947) is misleading in isolation because the set of found questions changed and became smaller.

The category results match the technique's expected strengths and risks. Difficult-distractor Hit@1 rises from 0.3684 to 0.4737; section-detail Hit@5 rises from 0.8462 to 1.0000; and semantic-paraphrase Hit@3 rises from 0.8182 to 0.9091. Conversely, exact-number Hit@5 falls from 0.8261 to 0.7391, exact-terminology Hit@5 falls from 1.0000 to 0.9231, and comparison Hit@3 falls from 0.6667 to 0.3333. These category samples—especially comparison and cross-page—are small.

The four rescues were `p75_rankgpt_noveleval`, `p75_replug_integration`, `p75_doct5query_latency`, and `p75_gtr_data_efficiency`. The latter two probes invented numeric details, while the RankGPT and GTR passages invented benchmark or architecture details. Their retrieval success does not validate those statements; false details happened to move the embedding toward useful real chunks.

The five found-to-not-found losses were `p75_doct5query_direction`, `p75_dpr_bm25_margin`, `p75_instructor_tasks`, `p75_rocketqav2_distillation`, and `p75_beir_tradeoff`. The probes respectively described the wrong representation flow, invented a 4.2-point DPR margin, invented 100 INSTRUCTOR tasks, replaced RocketQAv2's training mechanism with generic joint optimization, and named the wrong BEIR model families and metric. This is retrieval drift caused by answer-shaped hallucination.

Fusion performs its safety role perfectly but provides no benefit: improved/unchanged/degraded is 0/90/0, with zero rescues and zero losses. The fused top-five list is byte-for-byte identical in chunk order to baseline for 88 of 100 questions; the remaining changes do not alter any scored question's first-relevant rank. On unanswerables, fused top results never change and average top-five overlap is 4.8/5.

Probe generation followed the requested size: 92 words and 710 characters on average, with no truncations, fallbacks, generation errors, or citation-like output. Nine questions lost at least one protected term, 33 introduced numerical details, and 39 triggered a drift warning. The high-recall title-case entity heuristic flags every question, so its 100/100 count is descriptive rather than a useful discriminator.

All ten unanswerable probes confidently invent plausible facts, including HNSW parameters, 2026 model results, Mac energy use, German retention periods, OCR thresholds, Qdrant latency, chunking winners, cloud cost, Arabic accuracy, and user-study preferences. HyDE-only changes the top result for eight of ten. This does not directly create unsupported final evidence because only real chunks survive, but it shows why the hypothetical passage must remain isolated and why HyDE-only is unsafe.

Mean baseline retrieval is 0.198 seconds. HyDE-only costs 4.228 seconds and fused HyDE costs 4.422 seconds; generation alone costs 4.077 seconds, roughly 92% of fused latency. Phase 10B Multi-query-2 averaged 6.640 seconds, so HyDE is about 33% faster, but Multi-query-2 improved top-three/top-five metrics without measured baseline-relative degradation. HyDE fusion adds substantial latency and produces no gain.

## Recommendation

Keep the implementation as a reproducible educational experiment only. Do not integrate HyDE into FastAPI, the CLI, `RAGApplication`, LangGraph, or default retrieval. HyDE-only is too volatile, while equal fusion is safe but economically unjustified. A smaller generator would reduce cost but would not solve the zero-benefit fused result, so it is not the next experiment.
