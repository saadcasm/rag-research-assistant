# Phase 2A: embeddings and semantic retrieval

## Model choice

The initial candidate, `all-MiniLM-L6-v2`, is still a reasonable compact general
sentence embedding model. Phase 2A uses
[`multi-qa-MiniLM-L6-cos-v1`](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1)
instead because this project performs asymmetric search: a short question must
retrieve a longer passage. The selected MiniLM model was trained specifically on
question-answer pairs for semantic search. It remains small enough for local
laptop inference and produces 384-dimensional vectors.

This is a starting baseline, not a claim that it is the universally best model.
Model selection should eventually be based on retrieval evaluation using the
project's own papers and questions.

## What an embedding is

An embedding model transforms variable-length text into a fixed-length array of
numbers. For this model, every query and chunk becomes 384 floating-point values.
The training process gives those coordinates useful geometry: semantically
related texts tend to be closer or point in similar directions.

The individual dimensions are learned features, not human-assigned labels such
as “biology” or “methodology.” Dimensionality simply means how many coordinates
each vector contains. More dimensions can represent more distinctions, but they
also consume more storage and computation and do not automatically imply a
better model.

## Why one model must encode both sides

Each embedding model learns its own coordinate system. Dimension 17 from one
model does not mean the same thing as dimension 17 from another—even when both
models happen to output 384 numbers. Documents and queries must therefore use
the exact same model (and compatible query/document encoding behavior) before
their vectors can be compared meaningfully.

The persisted manifest records the document model name. Search loads that name
and refuses a mismatched query embedder.

## Cosine similarity and top-k

Cosine similarity is the dot product divided by both vector lengths:

```text
cosine(a, b) = (a · b) / (||a|| × ||b||)
```

It measures the angle between two non-zero vectors. A value near `1` means they
point in a similar direction, `0` means roughly orthogonal, and `-1` means
opposite directions. In an embedding model, higher generally means the model
considers the texts more semantically related. The value is not a probability.

For one query, Phase 2A calculates a score against every chunk, sorts all rows by
descending score, and returns top-k: the `k` highest-ranked chunks. If the corpus
has fewer than `k` chunks, all chunks are returned.

## Why this can beat literal keyword matching

Keyword search depends on overlapping surface forms. Semantic retrieval can
place paraphrases, synonyms, abbreviations, and related concepts near each other
even when they share few literal words. A question about “reducing hallucination
with external evidence” may therefore retrieve a passage about “grounding model
responses in retrieved documents.”

This advantage is not universal. Exact names, codes, equations, citations, and
rare terminology are often better served by lexical search. A later hybrid
retriever could combine both signals after each is understood separately.

## Important limitations

- The model is English-focused and may perform poorly on other languages.
- It is a general question-passage model, not a specialist in every scientific
  field, mathematical notation, tables, or source code.
- Its model card notes a 512-word-piece limit and training on inputs up to about
  250 word pieces. Longer chunks can be truncated and silently lose content.
- Dense retrieval may favor broadly related passages over exact factual matches.
- Similarity scores are not calibrated confidence values and are comparable
  primarily within the same query and model setup.
- The model cannot compensate for broken PDF reading order, OCR gaps, headers,
  or poor chunk boundaries.
- Brute-force NumPy search scales linearly with the number of chunks. That is a
  teaching choice, not a production-scale indexing strategy.

These limitations are why the next engineering step should be retrieval
evaluation—not immediate LLM generation.
