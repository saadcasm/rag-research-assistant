# Phase 10D: parent-child retrieval results

Semantic child chunks are retrieved through the frozen dense+BM25+RRF+cross-encoder stack, then expanded to physical-page or conservatively matched structural parents. Parent relevance is provenance-based: the parent document/page span must cover an expected source.

## Child retrieval

| Metric | Legacy baseline | Semantic child |
|---|---:|---:|
| Hit@1 | 0.4556 | 0.4000 |
| Hit@3 | 0.7222 | 0.7556 |
| Hit@5 | 0.8444 | 0.8444 |
| Recall@1 | 0.4500 | 0.3833 |
| Recall@3 | 0.7167 | 0.7333 |
| Recall@5 | 0.8222 | 0.8222 |
| Mean first relevant rank | 1.8947 | 1.8158 |

## Parent retrieval

| Metric | Page parent | Structural parent |
|---|---:|---:|
| Hit@1 | 0.2667 | 0.3778 |
| Hit@3 | 0.6778 | 0.7222 |
| Hit@5 | 0.8333 | 0.8444 |
| Recall@5 | 0.8056 | 0.8222 |
| Mean first relevant rank | 2.3600 | 2.0921 |
| Average characters | 4373.5700 | 2395.5020 |
| Median characters | 4224.0000 | 1179.0000 |
| p95 characters | 6126.0000 | 4964.2000 |
| Average expansion ratio | 7.9755 | 4.0513 |
| Deduplication rate | 0.2615 | 0.0665 |

## Coverage and latency

- Structural exact mappings: 3310/5124 (64.6%); page fallbacks: 1814
- Average baseline/child retrieval: 0.158s / 0.144s
- Average parent mapping: 0.000018s
- Average page/structural aggregation: 0.000170s / 0.000124s

## Result validation

The report contains all 100 frozen questions (90 scored and 10 unanswerable), 849 extracted pages from 52 PDFs, 5,124 semantic children, and the expected models and candidate depth. The dataset SHA-256 is `36e9707071d722e327ff1987118caff1fc71d646dbeef77d139b32aa1445fcbe`. Legacy and semantic-child metrics exactly reproduce their Phase 7.5 results, confirming that the retrieval implementations and labels did not change.

## Child localization

Semantic children do not beat legacy uniformly. They trade lower early precision for stronger rank-three retrieval:

- Hit@1 falls from 45.6% to 40.0%.
- Hit@3 rises from 72.2% to 75.6%.
- Hit@5 and Recall@5 remain tied at 84.4% and 82.2%.
- Twenty questions improve, 26 degrade, and 44 remain unchanged.

This reproduces Phase 7.5 rather than creating a new chunking conclusion. Small semantic units help some exact localization cases—including `p75_cocondenser_loss`, `p75_realm_signal`, and `p75_hyde_false_details`—but lose several legacy hits such as `p75_colbert_late_interaction`, `p75_monot5_reformulation`, and `p75_rocketqav2_distillation`.

## What parent aggregation changes

Page aggregation recovers six semantic-child misses but loses seven child hits. Relevant evidence is usually ranked later: page-parent Hit@1 is 26.7%, Hit@3 is 67.8%, and Hit@5 is 83.3%. It improves 26 question ranks relative to children, degrades 38, and leaves 26 unchanged.

Structural aggregation recovers two child misses and loses two hits. It preserves semantic-child Hit@5 and Recall@5 exactly, while Hit@1 declines only from 40.0% to 37.8%. Relative to children it improves 16, degrades 34, and leaves 40 unchanged.

The two structural losses are `p75_graphrag_index` and `p75_late_chunking_order`. It recovers `p75_rocketqav2_distillation` at rank 1 and `p75_texttiling_validation` at rank 3. These changes occur because parent RRF can promote evidence from the 20-child candidate pool, but parent deduplication and consensus can also demote a strong individual child.

## Context size and fallback effects

Page parents average 4,374 characters (approximately 677 whitespace tokens), with a median of 4,224 and p95 of 6,126. They expand the best contributing child by 7.98× on average. That is broad context, but also a substantial noise risk.

The structural condition is bimodal:

- 299/500 returned contexts are true structural parents, averaging about 1,120 characters and 2.13× expansion.
- 201/500 are page fallbacks, averaging about 4,293 characters and 6.90× expansion.

Across the complete child corpus, 3,310/5,124 children (64.6%) receive an exact structural mapping and 1,814 (35.4%) fall back. Although every semantic child carries a detected section title, title presence is not treated as proof of reliability; unique same-section text containment is required.

Page aggregation deduplicates 26.2% of the 20 child candidates, compared with 6.7% for the structural condition. Pages naturally collect more children, which explains both their stronger consensus effects and their greater loss of child-level ranking precision.

## Evidence containment

Of 209 relevant child occurrences in the top-20 rankings:

- page parents retain 189 in returned top-five parents (90.4%), but only 31 satisfy strict full-child contiguous-text containment (14.8%);
- structural parents retain 141 (67.5%), with 105 satisfying strict containment (50.2%).

The stored `relevant_child_mapping_rate` means “the mapped parent survived into the returned top five,” not merely that a mapping exists. Every child has a mapping.

Containment is deliberately strict. Semantic chunk construction may join text around removed heading lines or span page boundaries, so a page can contain the underlying evidence without reproducing the entire normalized child as one contiguous substring. The low page figure therefore exposes a representation mismatch; it is not equivalent to saying 85% of evidence disappeared. Structural exact mappings score better because containment is part of their acceptance rule.

## Latency

Average legacy and semantic-child retrieval were 0.158s and 0.144s. Mapping plus both aggregation passes add about 0.00031s per query—negligible compared with model retrieval. The apparent semantic speed advantage is small run-to-run variation, not evidence that the strategy is intrinsically faster.

Building page text and all mappings at startup took 21.6 seconds. A future integrated implementation should persist the parent map rather than rebuild it for every process, but persistence is intentionally outside this experiment.

## Recommendation

Keep legacy as the production retrieval default. Parent-child retrieval does not improve child localization overall, and page parents introduce too much context while weakening early ranking.

Retain the conservative structural-parent design as the promising experimental branch. It preserves top-five retrieval, provides approximately 2× context when a true structural match exists, and costs almost nothing per query after construction. It is not ready for production because structural coverage is only 64.6%, fallbacks are much larger, strict containment is incomplete, and provenance Hit does not measure whether generation actually benefits from the added context.

If Phase 10D continues later, the next controlled step should improve or persist structural mappings and run a very small qualitative generation/context-usefulness study. Do not adopt full-page expansion as the primary parent strategy from these results.
