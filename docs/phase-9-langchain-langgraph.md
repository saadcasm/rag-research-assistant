# Phase 9: LangChain and LangGraph experiments

Phase 9 adds two isolated framework exercises. The manual CLI, FastAPI service,
retrieval stack, prompt, models, and `rag.answer_question` remain the default.
This is a comparison, not a migration.

```text
                         production/default
CLI / FastAPI -> RAGApplication -> answer_question
                         |
                         +-- existing Retriever, context, prompt, OllamaGenerator

                         educational/parallel
existing Retriever -> LangChain Document + LCEL -> ChatOllama
                  \-> LangGraph bounded retrieve/rewrite workflow
```

Only three direct framework packages are declared:

- `langchain-core`: `Document`, prompt templates, Runnables, LCEL, parsers;
- `langchain-ollama`: the provider-specific `ChatOllama` model wrapper;
- `langgraph`: state graphs and conditional control flow.

There is no `langchain` umbrella package, community integration bundle,
LangChain Qdrant wrapper, or LangChain embedding wrapper. The experiment reuses
our established hybrid + reranking `Retriever`, so it does not rebuild the
corpus or hide BM25, RRF, Qdrant, and cross-encoder behavior behind new layers.

## LangChain path

The path in `frameworks/langchain_pipeline.py` is:

```text
question: str
  -> Runnable-wrapped existing Retriever.search
  -> list[SearchResult]
  -> list[LangChain Document]
  -> context formatting
  -> PromptTemplate
  -> ChatOllama
  -> StrOutputParser
  -> existing GroundedAnswer
```

### Document

A LangChain `Document` combines `page_content` with an open-ended metadata
dictionary. Our adapter records the complete `Chunk` metadata plus retrieval
score. A reverse adapter validates required provenance and reconstructs the
domain `SearchResult`. This is more code than simply dropping metadata into a
dictionary, but it makes citation loss detectable.

The framework does not know that `document`, `start_page`, `end_page`, and
`chunk_id` are essential. That remains our application contract. Flexible
metadata is convenient at integration boundaries but is less type-safe than a
dedicated `Chunk` dataclass.

### Retriever abstraction and Runnable

Our manual `Retriever` protocol exposes `search(query, top_k)`. A
`RunnableLambda` adapts it to LangChain's standard `invoke` interface and emits
Documents. Nothing about the ranking implementation changes.

A Runnable is a component with a common invocation/composition interface.
Runnables can be invoked individually, batched, streamed, or joined into larger
pipelines. The framework provides that uniformity; it does not guarantee that
the component's hidden behavior or metadata is correct.

### LCEL and composition

LCEL means LangChain Expression Language. In Python, the `|` operator composes
Runnables, while `RunnableParallel` and `.assign()` build dictionaries and add
derived values. Phase 9 uses LCEL to combine retrieval, context, prompt, model,
and parsing without a custom orchestration loop.

The composed chain is approximately:

```python
prepared = RunnableParallel(
    question=identity,
    documents=retriever_runnable,
) | add_context

answer_chain = prompt | model | StrOutputParser()
chain = prepared.assign(answer=generate_or_return_insufficient)
```

The empty-retrieval guard preserves the manual pipeline's behavior and avoids
calling Ollama with no evidence.

### Prompt template and model wrapper

`GROUNDED_PROMPT_TEMPLATE` is now a named constant used by both the existing
manual formatter and LangChain `PromptTemplate`. This is a behavior-preserving
refactor: there are not two prompts to drift apart.

`build_ollama_chat_model()` configures `ChatOllama` with the established
`qwen3.5:4b`, temperature `0.1`, local URL, and reasoning disabled. Its
`validate_model_on_init` check fails early if Ollama or the model is unavailable.
Unlike our small `OllamaGenerator`, the wrapper brings standard message,
streaming, async, callback, and Runnable integration behavior.

### Minimal use

```python
from rag_research_assistant.application import (
    ApplicationSettings,
    default_retrieval_config,
    load_retriever,
)
from rag_research_assistant.frameworks.langchain_pipeline import (
    LangChainRAGPipeline,
    build_ollama_chat_model,
)

settings = ApplicationSettings.from_environment()
loaded = load_retriever(default_retrieval_config(settings))
try:
    model = build_ollama_chat_model(
        settings.ollama_model,
        settings.ollama_url,
    )
    pipeline = LangChainRAGPipeline(loaded.retriever, model, top_k=5)
    answer = pipeline.invoke("How does DPR represent passages?")
finally:
    loaded.close()
```

The caller owns the existing retriever/Qdrant lifecycle just as it does in the
manual application layer.

## Manual versus LangChain

| Dimension | Manual path | LangChain experiment |
|---|---|---|
| Orchestration | ordinary typed Python calls | standard Runnable/LCEL composition |
| Retrieval | explicit `Retriever.search` | same call behind `RunnableLambda` |
| Data | typed dataclasses | flexible `Document.metadata` at the boundary |
| Prompt | explicit string builder | same template through `PromptTemplate` |
| Model | 120-line local HTTP wrapper | provider wrapper with many capabilities |
| Debugging | direct call stack and values | must inspect Runnable inputs/outputs |
| Dependency coupling | project code + standard library | LangChain protocols and provider packages |
| Integration convenience | adapters written per integration | shared invoke/batch/stream conventions |
| Hidden behavior | low | more validation, conversion, callback, and message behavior |

The manual orchestration, context, and prompt modules are 136 lines combined.
The LangChain module is 178 lines because robust provenance adapters and domain
conversion dominate it; the central LCEL expression itself is short. Line count
therefore does not prove simplicity. LangChain becomes more attractive when
many replaceable models, retrievers, callbacks, or streaming modes must share a
uniform interface. For one stable local pipeline, our manual path remains
clearer and less coupled.

## LangGraph path

LangChain composes a mostly linear data path. LangGraph adds explicit control
flow when the next step depends on state.

```text
START
  -> retrieve
  -> evaluate
       | sufficient OR retry limit reached
       |        -> answer -> END
       |
       + insufficient and retries remain
                -> rewrite -> retrieve
```

### State schema

`RetrievalState` is a `TypedDict` with:

| Field | Purpose |
|---|---|
| `question` | immutable user intent used for the final answer |
| `query` | current retrieval query; may be rewritten |
| `documents` | evidence from the most recent retrieval |
| `evidence_sufficient` | latest deterministic evaluation decision |
| `rewrite_count` | completed rewrites |
| `max_rewrites` | explicit bound, zero through three |
| `query_history` | observable state-transition trail |
| `answer` | final domain `GroundedAnswer`, initially `None` |

Each node returns only a state update. LangGraph merges the update into state
before following the next edge.

### Nodes and edges

- `retrieve`: invokes the LangChain pipeline's Document retriever with `query`.
- `evaluate`: runs an injected `EvidenceEvaluator`. The default is a cheap,
  deterministic term-coverage heuristic; authorship questions additionally
  require title-page evidence.
- conditional edge after `evaluate`: routes to `answer` when sufficient or
  exhausted, otherwise to `rewrite`.
- `rewrite`: invokes an injected `QueryRewriter`, increments the counter, and
  appends to history. Recognized corpus method names can be expanded to official
  titles for authorship queries. Other intents receive visible generic hints.
- normal edge from `rewrite` to `retrieve`: creates the bounded loop.
- `answer`: generates from the latest documents while asking the original
  question, then follows a normal edge to `END`.

`max_rewrites=1` is the default. The constructor rejects negative values and
values above three. Even if evidence is always insufficient, the graph answers
after `max_rewrites + 1` retrieval attempts. Node exceptions propagate to the
caller instead of being converted into invented answers.

### Why this is not an agent

The graph has fixed nodes, fixed capabilities, deterministic routing, and a
strict retry limit. It cannot choose arbitrary tools, invent new goals, or run
indefinitely. A graph describes control flow; it becomes agentic only when
model-driven decisions and open-ended action selection are deliberately added.

### Checkpoints

A checkpointer stores graph state between steps so a workflow can resume after
a failure, wait for human input, or retain thread history. This experiment is
short, local, and synchronous, so it compiles without a checkpointer. Adding a
database merely to demonstrate persistence would obscure the state/edge lesson.

## Controlled experiment

The experiment used the actual 3,793-chunk legacy corpus and the unchanged
Qdrant dense + BM25 + RRF + cross-encoder retriever. The manual call and the
LangChain Document adapter returned identical chunk IDs for each original
query, which is expected because they share the same retriever.

| Question | Original retrieval | Graph decision | Rewritten retrieval |
|---|---|---|---|
| Who wrote the DPR paper? | DPR pages 5 and 2 were present, but no title page | insufficient; rewrite once | official-title query returned DPR page 1 at rank 1 |
| How does DPR represent passages? | DPR page 2 at rank 1 | sufficient; direct answer | not rewritten |
| What supervision source trains E5 embeddings? | E5 page 1 at rank 1 | sufficient; direct answer | not rewritten |
| How do late-interaction models differ from single-vector retrievers? | ColBERTv2 page 1 at rank 1 and COIL page 2 at rank 2 | sufficient; direct answer | not rewritten |

This is a control-flow demonstration, not a statistically meaningful quality
claim. It shows one known failure mode improved without forcing every question
through rewriting. It also exposed a useful negative result: a generic query
like “paper title page author names” displaced DPR with unrelated papers. The
explicit official-title expansion succeeded, illustrating why query rewriting
must preserve identity rather than merely append keywords.

An end-to-end Qwen generation run was attempted after the real retrieval check.
The installed Ollama 0.34.0 server could list `qwen3.5:4b`, but its llama-server
failed to allocate a Metal command queue/context, even with the Python models
released and a reduced CPU/context request. The run therefore stopped with a
real resource error rather than substituting fake output. Model composition is
covered deterministically in tests, and Phase 8 previously proved this same
model/prompt/corpus combination end to end. The Phase 9 result claims only the
real retrieval and graph transitions observed here.

## Tests

Tests inject a fake Retriever, Runnable model, evaluator, and rewriter. They
require no network, model downloads, corpus, Qdrant, or Ollama and cover:

- LCEL retrieval -> prompt -> model composition;
- bidirectional Document/domain adaptation;
- page-span, score, filename, and chunk-ID preservation;
- no-evidence short circuit;
- sufficient-evidence branch;
- rewrite branch and state history;
- maximum retry enforcement;
- node failure propagation;
- official-title expansion for a known method.

## Adoption decision and limits

No framework object enters the default CLI, FastAPI, `RAGApplication`, or manual
RAG modules. The most promising future adoption candidates are Runnable-based
streaming/provider interchange and LangGraph for workflows that genuinely need
human approval, persistence, or branching. The typed `Chunk`, retrieval math,
provenance rules, evaluation, and simple linear orchestration should remain
manual until a concrete integration need outweighs the dependency and debugging
cost.

Phase 9 deliberately adds no agent, tool calling, memory, checkpoint database,
LangSmith tracing, framework vector store, framework embeddings, API endpoint,
or default-path switch.
