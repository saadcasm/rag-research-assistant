"""Orchestrate retrieval and generation without merging their responsibilities."""

from typing import Callable, List, Optional

from .context import build_context
from .embeddings import Embedder
from .generation import DEFAULT_TEMPERATURE, TextGenerator
from .index import EmbeddingIndex
from .models import GroundedAnswer, SearchResult
from .prompting import build_grounded_prompt
from .retrieval import search


INSUFFICIENT_CONTEXT_ANSWER = (
    "I cannot answer from the supplied evidence because no relevant context was retrieved."
)


def answer_question(
    question: str,
    index: EmbeddingIndex,
    embedder: Embedder,
    generator: TextGenerator,
    top_k: int = 5,
    temperature: float = DEFAULT_TEMPERATURE,
    on_retrieved: Optional[Callable[[List[SearchResult]], None]] = None,
) -> GroundedAnswer:
    """Retrieve evidence, build a grounded prompt, and invoke generation."""

    if not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature must be between 0 and 2")
    results = search(question, index, embedder, top_k=top_k)
    if on_retrieved is not None:
        on_retrieved(results)
    if not results:
        return GroundedAnswer(
            text=INSUFFICIENT_CONTEXT_ANSWER,
            sources=[],
            retrieved=[],
        )

    context, sources = build_context(results)
    prompt = build_grounded_prompt(question, context)
    answer = generator.generate(prompt, temperature=temperature)
    return GroundedAnswer(text=answer, sources=sources, retrieved=results)
