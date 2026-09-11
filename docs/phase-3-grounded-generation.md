# Phase 3: local grounded answer generation

## What the local LLM receives

The generation model receives one text prompt with four parts:

1. rules requiring evidence-only answers and numbered citations;
2. the retrieved chunks, each labelled with `[n]`, filename, page, chunk ID, and
   full text;
3. the user's question;
4. an `ANSWER` marker indicating where generation should begin.

It does not receive the PDFs, vector matrix, cosine scores, or unretrieved chunks.
Retrieval is completed in application code before Ollama is called.

A context entry looks like this:

```text
SOURCE [1]
filename: paper.pdf
page: 4
chunk_id: paper.pdf:p4:c2
text:
The complete text of the retrieved chunk appears here.
```

The final source list maps `[1]` back to the same metadata and also shows its
retrieval score.

## Context windows

A model's context window is the maximum token budget available for its input and
generated output during one request. The prompt instructions, question, every
retrieved chunk, metadata, and answer all consume that budget.

Retrieved evidence must fit inside the effective Ollama context window. Content
beyond the limit may be rejected or truncated, and evidence that is truncated is
effectively unavailable to the model. A model may advertise a very large maximum
while the locally configured runtime uses a smaller context to control memory.
Top-k is therefore also a context-budget decision: more chunks provide more
coverage but use more memory and can dilute the useful evidence.

## Grounding and hallucination

Prompt grounding reduces hallucination risk by limiting the requested task,
putting evidence directly in the prompt, requiring citations, and defining how to
respond when evidence is insufficient. It does not mathematically constrain the
model's output. An LLM still predicts tokens probabilistically and can ignore
instructions, misread evidence, combine claims incorrectly, or attach the wrong
citation. Grounded answers still need evaluation and source checking.

Retrieved documents are also untrusted data. The prompt explicitly tells the
model to treat context as quoted evidence rather than instructions, reducing the
risk that instruction-like text inside a paper controls generation.

## Retrieval failure versus generation failure

- **Retrieval failure:** the relevant chunk is absent from top-k, the wrong
  passages rank highly, extraction lost the needed text, or chunking separated
  essential context. Inspect `--show-context`, then improve ingestion, chunking,
  embeddings, top-k, or later retrieval evaluation.
- **Generation failure:** the correct evidence is present, but Ollama fails,
  times out, returns no response, or the model creates an inaccurate or poorly
  cited answer. Investigate the runtime, model, prompt, context size, and
  generation settings.

This distinction prevents a common mistake: changing the generation prompt when
the answer was never present in the retrieved evidence.

## Temperature

Temperature controls how sharply the model samples among possible next tokens.
Lower values make output more repeatable and favor high-probability wording;
higher values increase variation and creativity. Temperature does not measure
truthfulness and setting it to zero does not guarantee a correct answer.

Phase 3 defaults to `0.1`. Grounded research QA usually benefits from low
variation because the goal is faithful synthesis, not creativity. The value is
configurable with `--temperature` for experiments.

## Ollama, Qwen, and the embedding model

- **Ollama** is the local runtime. It downloads and stores compatible model
  weights, exposes the localhost HTTP API, loads a model into memory, and runs
  inference on the Mac.
- **Qwen 3.5 4B** is the default generation model. It reads the grounded prompt
  and predicts the cited natural-language answer. Ollama's `qwen3.5:4b` build is
  a roughly 3.4 GB Q4 quantization, a practical baseline for a 16 GB M1 Pro.
- **multi-qa-MiniLM-L6-cos-v1** is the embedding model. It never writes the
  answer. It creates vectors used to rank chunks by semantic similarity.

The two models are independent and serve different purposes. Changing the Qwen
generation model does not require rebuilding embeddings. Changing the embedding
model does require rebuilding the index because it changes the vector space.

## Setup and commands

Install and start Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

In another terminal, pull one model and ask a question:

```bash
ollama pull qwen3.5:4b
rag-research-assistant ask "How does RAG work?" --top-k 5 --show-context
```

Ollama model files live in Ollama's machine-level storage, not this repository.
They must never be committed.
