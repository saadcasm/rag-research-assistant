# Phase 7.5: expanded evaluation dataset

`data/evaluation/questions-phase-7.5.jsonl` is a new, reviewable benchmark for
the verified 52-PDF corpus. It is separate from the original 20-question
`data/evaluation/questions.jsonl`; no existing labels were replaced.

## Selection and verification

The interrupted drafting pass produced 104 candidates from the current
`pypdf` page extraction. Four weaker or duplicative candidates were removed:

- `p75_atlas_index_update`: generic and redundant with the retained Atlas
  few-shot question.
- `p75_layoutlm_pretraining`: asked about multiple objectives while the located
  evidence directly supported only one.
- `p75_rankt5_zero_shot`: used indirect evidence and was weaker than the retained
  RankT5 loss question.
- `p75_sbert_computations`: duplicated the retained SBERT speed result.

The final 100 records contain 90 answerable and 10 unanswerable questions. For
each answerable record, a separate verification pass freshly extracted the
declared PDF pages, normalized whitespace, line-break hyphenation, and common
ligatures, and re-located the stored passage. All 99 evidence passages were
found on their declared pages. Source/page labels and evidence locations also
matched exactly. A few passages were then narrowed to remove title-block text
or malformed adjacent font tokens; the ColBERT fine-grained token was
human-normalized and explicitly noted in those records.

The evidence-locator pass corrected anchors rather than changing claims. It
handled extraction-specific hyphenation or ligatures for COIL, FiD, BEIR,
ColBERT, CRAG, DPR, GTR, M3, RULER, SBERT, and query2doc; selected clearer text
for GRF, LongBench, monoT5, MS MARCO, RankT5, Lost-in-the-Middle, RAPTOR, and
SimCSE; and corrected the supporting pages/passages for the COIL/ColBERT and
RAG/RePlug comparisons. Final review also expanded HyDE's too-short evidence,
removed title-block spillover from HYRR and query2doc, narrowed DPR and RETRO
snippets, and restored the beginning of BEIR's comparison passage. None of
these corrections changed the expected paper identity or the 90/10 split.

Unanswerable questions are domain-plausible but ask for facts not established
by the paper corpus, such as this project's future Qdrant latency, the winning
chunk strategy on this new benchmark, energy use, legal retention policy, or a
user study. They deliberately contain retrieval vocabulary so related passages
may still rank highly.

## Composition

| Category | Questions |
|---|---:|
| comparison | 3 |
| cross-page context | 3 |
| difficult distractor | 19 |
| exact number | 23 |
| exact terminology | 13 |
| multi-source | 5 |
| section detail | 13 |
| semantic paraphrase | 11 |
| unanswerable | 10 |
| **Total** | **100** |

Difficulty labels are 17 easy, 35 medium, and 48 hard. Every approved PDF has
at least one grounded question, so the benchmark represents 52 distinct papers
with no zero-question paper. Some multi-source questions add more than one
topic-cluster mention; their labelled-source coverage is:

| Primary cluster | Question-source mentions | Distinct papers |
|---|---:|---:|
| RAG | 16 | 9 |
| Dense retrieval | 16 | 8 |
| Retrieval evaluation | 11 | 5 |
| Embeddings | 10 | 6 |
| Sparse / lexical | 9 | 4 |
| Query transformation / multi-query | 8 | 4 |
| Long-context vs retrieval | 6 | 3 |
| Advanced retrieval | 5 | 3 |
| Document chunking / parsing | 5 | 3 |
| Reranking | 4 | 4 |
| Vector similarity search | 4 | 2 |
| Hybrid retrieval | 1 | 1 |

The single hybrid-retrieval paper still has direct coverage; its lower count
reflects the validated corpus composition, not an omitted label.

## What the benchmark stresses

Easy questions often test a distinctive reported number, such as RETRO's
database size. Hard questions distinguish nearby concepts—for example, HyDE's
hypothetical-document embedding from hybrid rank fusion, or COIL's exact-token
overlap from ColBERT's broader late interaction. Cross-page questions require
evidence spanning two labelled pages, and five multi-source questions require
comparing claims from two papers.

The questions are intentionally not uniform trivia. Exact terms test lexical
retrieval, paraphrases test semantic retrieval, overlapping method descriptions
create plausible distractors, and unanswerable items test whether later answer
generation can avoid treating a top-ranked related passage as proof.

## Limitations and review points

- Evidence is page-level ground truth, not a gold chunk ID. Chunk strategies
  can split the same page differently, which is part of the later experiment.
- Question authorship is synthetic and was verified against the PDFs, but has
  not been independently reviewed by a second human annotator.
- The corpus is English-heavy and research-paper-heavy; it does not represent
  conversational, multilingual, enterprise, or scanned-document retrieval.
- Abstract and first-page evidence is common because it often states methods
  and headline results most clearly. Section-detail and cross-page items reduce,
  but do not eliminate, that bias.
- Ten unanswerable labels are corpus-level judgments. They should be reviewed
  again if the corpus changes.

This task creates labels only. It does not rebuild Qdrant, regenerate embeddings,
or run the four-strategy benchmark.
