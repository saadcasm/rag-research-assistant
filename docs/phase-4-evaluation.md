# Phase 4: systematic local RAG evaluation

## Why evaluation is necessary

"The answer looked correct" is not repeatable evidence. It does not identify
whether the right passage was retrieved, whether the model used that passage,
whether a citation exists, or whether a different configuration is better.
Phase 4 freezes a small set of human-labelled questions and records each stage so
future changes can be compared against the same task.

This phase evaluates two separate systems:

1. **Retrieval:** did the ranked chunks include the page that a human identified
   as relevant?
2. **Generation:** given the actual retrieved context, what did the model answer,
   were its citation numbers resolvable, and did it refuse unavailable answers?

Retrieval evaluation is deterministic, fast, and independent of Ollama.
Generation evaluation is optional because it is slower, can vary, and has
different failure modes.

## Dataset and source matching

`data/evaluation/questions.jsonl` contains 20 records: 14 answerable, 2 partially
answerable, and 4 unanswerable. The expected filename/page labels were checked
against the locally extracted text from:

- `rag-lewis-2020.pdf`
- `dense-passage-retrieval.pdf`
- `sgpt-semantic-search.pdf`

Each line is independently editable JSON. IDs must be unique. Answerable and
partially answerable records require at least one expected source. Unanswerable
records require none. Matching always requires the filename; an included page
number and chunk ID add exact constraints. Exact generated answer text is never
required.

`partially_answerable` means the indexed material supports a useful but
incomplete response. It is still included in retrieval metrics because relevant
evidence is labelled. The category lets manual generation review distinguish it
from a fully supported question.

## Retrieval metrics

For one question:

```text
Hit@k = 1 if any expected source occurs in ranks 1..k, otherwise 0
```

The aggregate is the mean across answerable and partially answerable questions.
Hit@1 asks whether the first chunk is on an expected page. Hit@3 and Hit@5 allow
more opportunities, so they can only stay equal or increase for a fixed ranking.

First-correct rank is the rank of the earliest matching source. "Not found"
means no expected source appeared within the configured retrieval depth. The
reported mean uses questions where a match was found; failures remain separately
listed so the mean cannot hide them.

When a question has several expected sources, recall@k is:

```text
distinct expected sources represented in top-k / total expected sources
```

Hit@k remains the primary metric: it needs any labelled evidence. Recall adds a
small diagnostic for questions that benefit from several pages.

Unanswerable questions have null Hit@k values. Cosine similarity calculates a
score for every indexed chunk and sorting always returns a top result, even when
all scores describe irrelevant material. Returning top-k is not an answerability
decision.

## Generation checks

With `--with-generation`, the evaluator uses the same numbered context and
grounded prompt as `ask`. It records the answer, retrieval results, numeric
citations, resolved citation sources, invalid citation numbers, refusal result,
and any generation error.

Refusal detection lowercases and normalizes the answer, then searches for a
short list of phrases such as "insufficient evidence" and "not enough
information." This is easy to inspect but imperfect: a legitimate refusal can
use unexpected wording, and cautious text can contain a phrase without actually
refusing.

A citation is reference-valid when `[n]` identifies one of the `n` context items
that the model received. `[7]` is invalid if only five chunks were supplied.
Validity does not establish correctness. A model can cite an existing source
that does not support its claim, so the saved answer and full contexts still
require human review.

We intentionally do not calculate an automatic groundedness percentage. No
RAGAS, DeepEval, TruLens, or LLM-as-a-judge is used yet. Those tools can later
add claim decomposition, semantic entailment, answer relevance, reference-answer
comparison, and judge-model scoring, but first we need an explicit baseline and
an understanding of what each score assumes.

## Commands and reports

Fast retrieval-only run:

```bash
rag-research-assistant evaluate
rag-research-assistant evaluate --top-k 5 --verbose
```

Optional local generation run:

```bash
rag-research-assistant evaluate \
  --with-generation \
  --model qwen3.5:4b \
  --temperature 0.1 \
  --output data/evaluation/results/latest.json
```

The JSON report contains summary metrics plus per-question expected sources,
ranked source text and scores, first-correct rank, Hit@k, recall@k, generation
details, and notes. `data/evaluation/results/` is ignored because reports depend
on local generated artifacts and models.

The existing stale-index checks run before evaluation. If `chunks.jsonl` differs
from the indexed hash, or the embedding model/dimensions are incompatible, the
command fails rather than producing misleading scores.

## Measured local baseline

The initial retrieval-only run used 209 chunks from all three papers and the
persisted `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` index:

| Metric | Result |
| --- | ---: |
| Questions in dataset | 20 |
| Retrieval-scored questions | 16 |
| Hit@1 | 50.0% |
| Hit@3 | 62.5% |
| Hit@5 | 93.8% |
| Mean first-correct rank, when found | 2.20 |

`dpr_runtime` was the only expected page not found in the top five. Its top
results were still from the DPR paper, especially pages 1 and 2, while the label
points to the runtime encoder description on page 3. That is a useful retrieval
ranking failure; the dataset was not altered to improve the score.

A real local `qwen3.5:4b` pass was completed for the four unanswerable questions
with temperature `0.1`. All 4 contained a recognized refusal phrase (100%), and
there were 0 out-of-range citation references. Two answers cited supplied
passages while explaining that they discussed different retrieval hardware or
methods. Those references were numerically valid, but a human must still decide
whether they were useful and correctly attached.

The unrestricted full 20-question generation run was stopped after an
impractically long local runtime and produced no partial report. This does not
affect the retrieval baseline. The full command remains supported; generation
latency and variability are reasons it is opt-in and excluded from CI.

These results do not mean the RAG system is "93.8% accurate." They show only that
at least one manually labelled filename/page appeared in the top five for 15 of
16 retrieval-scored questions in this small dataset.

## Learning summary

1. Retrieval evaluation asks whether known supporting evidence appears early in
   the ranking, not whether the final prose sounds good.
2. Hit@1 means the first returned chunk matches at least one expected source.
3. Hit@3 means at least one expected source appears among the first three.
4. Hit@5 can be higher because the retriever gets two additional chances to
   surface a labelled source.
5. Retrieval success supplies evidence; generation success interprets and writes
   from it. They are different operations.
6. A model can still misread, omit, combine, or invent claims after correct
   retrieval.
7. Generation can sound correct after retrieval failure because the model may
   rely on memorized parametric knowledge. That answer is not grounded in the
   retrieved evidence.
8. Unanswerable questions test whether the system admits missing evidence instead
   of rewarding confident invention.
9. Cosine similarity always assigns comparable numeric scores and therefore an
   ordering; it has no inherent relevance threshold or "none" result.
10. Citation validity checks that `[n]` exists. Citation correctness asks whether
    source `[n]` actually supports the attached claim.
11. Human-curated questions encode the evidence and edge cases we care about in
    a form that is inspectable and reusable.
12. Keeping the set fixed makes before/after comparisons meaningful; changing
    questions to fit a new method moves the target.
13. Current limits include a small corpus and dataset, page-level labels,
    incomplete expected-source annotations, heuristic refusals, no semantic
    citation checking, no reference-answer scoring, and variable generation.
14. An advanced framework could add standardized datasets, richer IR metrics,
    claim-level faithfulness, answer relevance, automated judges, experiment
    tracking, and confidence intervals. Those additions still require careful
    human labels and metric interpretation.
