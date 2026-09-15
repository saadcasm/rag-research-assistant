# Phase 10G: Frozen Benchmark Regression

The frozen Phase 7.5 dataset remains unchanged: 100 questions, including 90
retrieval-scored questions and 10 unanswerable controls.

## Metrics

| Condition | Hit@1 | Hit@3 | Hit@5 | Recall@1 | Recall@3 | Recall@5 | Mean first rank* |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.4556 | 0.7222 | **0.8444** | 0.4500 | **0.7167** | **0.8222** | 1.8947 |
| Always decompose | 0.4556 | 0.7222 | 0.8222 | 0.4500 | 0.7111 | 0.8000 | 1.8378 |
| Original + decomposed | 0.4556 | 0.7222 | **0.8444** | 0.4500 | **0.7167** | **0.8222** | 1.8947 |

`*` The apparently lower mean for always-decompose is not an overall win: mean
first rank excludes questions for which no relevant result was found.

Always-decompose loses top-five retrieval for `p75_beir_tradeoff` and
`p75_raptor_levels`, with no top-five rescue. Guarded fusion exactly reproduces the
baseline aggregate metrics and preserves those two questions through its original
branch.

Qwen requested decomposition for 41 of 100 ordinary benchmark questions: 37 of 90
answerable and 4 of 10 unanswerable. That rate is too high to serve as a selective
router without independent validation. One RocketQAv2 response safely fell back
after the validator rejected a second sub-question as a superficial paraphrase.

Decomposition generation averaged 1.637 seconds and consumed about 163.7 seconds
across this regression run. The unchanged guarded metrics show that the original
branch is an effective safety net, not that decomposition adds value.

## Conclusion

The regression experiment reinforces the multi-hop result: decomposed-only
retrieval can lose relevant evidence, while guarded fusion prevents measured
regression but supplies no aggregate gain. Production remains unchanged.
