"""Corpus-grounded, single-query rewriting through the existing generator."""

import re
from dataclasses import dataclass
from time import perf_counter
from typing import List, Protocol, Sequence, Tuple

from ..generation import TextGenerator
from ..models import SearchResult


REWRITE_PROMPT_TEMPLATE = """You improve a search query for a research-paper retrieval system.

Your only task is to output ONE rewritten retrieval query. Do not answer the question.

Rules:
1. Preserve the user's original intent.
2. Preserve important named entities, paper names, method names, model names, dataset names, and quoted terms.
3. Preserve acronyms unless the evidence clearly resolves them.
4. When the evidence clearly resolves an acronym or term, you may include BOTH the original term and its expansion.
5. Use only facts visible in the preliminary evidence. Do not invent expansions, names, titles, or facts.
6. Never replace a specific entity with a generic concept.
7. Avoid semantic drift and unnecessary extra terms.
8. If rewriting is unlikely to improve retrieval, return the original query unchanged.
9. Output only the query: no label, quotation marks, explanation, or answer.

ORIGINAL QUERY
{original_query}

PRELIMINARY EVIDENCE
{evidence}

REWRITTEN RETRIEVAL QUERY
"""


@dataclass(frozen=True)
class IdentityDiagnostics:
    """Lightweight reporting of protected tokens lost during rewriting."""

    protected_terms: Tuple[str, ...]
    preserved_terms: Tuple[str, ...]
    missing_terms: Tuple[str, ...]

    @property
    def identity_preserved(self) -> bool:
        return not self.missing_terms


@dataclass(frozen=True)
class RewriteResult:
    """Observable outcome of one rewrite attempt, including safe fallback state."""

    original_query: str
    rewritten_query: str
    changed: bool
    latency_seconds: float
    model_name: str
    fallback: bool
    error: str | None
    identity: IdentityDiagnostics


class QueryRewriter(Protocol):
    """Behavior required by the Phase 10A experiment runner."""

    model_name: str

    def rewrite(
        self, original_query: str, preliminary_results: Sequence[SearchResult]
    ) -> RewriteResult: ...


_QUOTED_PHRASE = re.compile(r'(["“”])([^"“”]+)["“”]')
_ACRONYM = re.compile(r"(?<!\w)(?:[A-Z]{2,}[A-Z0-9-]*|[A-Z][0-9][A-Z0-9-]*)(?!\w)")
_MODEL_LIKE = re.compile(
    r"(?<!\w)[A-Za-z][A-Za-z0-9]*(?:[-./:][A-Za-z0-9]+)+(?!\w)"
)


def extract_protected_terms(query: str) -> Tuple[str, ...]:
    """Find quoted, acronym, and model-like identity signals without NER."""

    quoted_matches = list(_QUOTED_PHRASE.finditer(query))
    model_matches = list(_MODEL_LIKE.finditer(query))
    protected_spans = [match.span() for match in [*quoted_matches, *model_matches]]
    terms: List[str] = [match.group(2).strip() for match in quoted_matches]
    terms.extend(
        match.group(0)
        for match in _ACRONYM.finditer(query)
        if not any(
            outer_start <= match.start() and match.end() <= outer_end
            for outer_start, outer_end in protected_spans
        )
    )
    terms.extend(match.group(0) for match in model_matches)
    return tuple(dict.fromkeys(term for term in terms if term))


def diagnose_identity_preservation(
    original_query: str, rewritten_query: str
) -> IdentityDiagnostics:
    """Report protected terms absent from the rewrite; do not block the rewrite."""

    protected = extract_protected_terms(original_query)
    lowered = rewritten_query.casefold()
    preserved = tuple(term for term in protected if term.casefold() in lowered)
    missing = tuple(term for term in protected if term.casefold() not in lowered)
    return IdentityDiagnostics(protected, preserved, missing)


def format_preliminary_evidence(
    results: Sequence[SearchResult], *, max_chars_per_chunk: int
) -> str:
    """Format bounded evidence with enough provenance to audit a rewrite."""

    if max_chars_per_chunk <= 0:
        raise ValueError("max_chars_per_chunk must be positive")
    if not results:
        return "No preliminary evidence was retrieved."
    entries = []
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        text = " ".join(chunk.text.split())
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk].rstrip() + "..."
        page = (
            str(chunk.start_page)
            if chunk.start_page == chunk.end_page
            else f"{chunk.start_page}-{chunk.end_page}"
        )
        entries.append(
            "\n".join(
                [
                    f"EVIDENCE [{rank}]",
                    f"document: {chunk.document}",
                    f"page: {page}",
                    f"chunk_id: {chunk.chunk_id}",
                    f"text: {text}",
                ]
            )
        )
    return "\n\n".join(entries)


def build_rewrite_prompt(
    original_query: str,
    preliminary_results: Sequence[SearchResult],
    *,
    max_chars_per_chunk: int,
) -> str:
    if not original_query.strip():
        raise ValueError("original_query cannot be empty")
    evidence = format_preliminary_evidence(
        preliminary_results, max_chars_per_chunk=max_chars_per_chunk
    )
    return REWRITE_PROMPT_TEMPLATE.format(
        original_query=original_query.strip(),
        evidence=evidence,
    )


def parse_rewritten_query(raw_response: str) -> str:
    """Accept one plain query and reject empty or explanatory responses."""

    if not isinstance(raw_response, str) or not raw_response.strip():
        raise ValueError("rewriter returned an empty response")
    value = raw_response.strip()
    if value.startswith("```") and value.endswith("```"):
        value = value[3:-3].strip()
        if value.lower().startswith("text\n"):
            value = value[5:].strip()
    prefix = re.match(r"(?i)^rewritten (?:retrieval )?query\s*:\s*", value)
    if prefix:
        value = value[prefix.end() :].strip()
    if len(value.splitlines()) != 1:
        raise ValueError("rewriter returned multiple lines instead of one query")
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    if not value:
        raise ValueError("rewriter returned an empty query")
    if len(value) > 4_000:
        raise ValueError("rewritten query exceeds 4000 characters")
    return value


class CorpusGroundedLLMRewriter:
    """Ask one local model for one query, falling back on every failure."""

    def __init__(
        self,
        generator: TextGenerator,
        *,
        temperature: float = 0.0,
        max_chars_per_chunk: int = 1_000,
    ) -> None:
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if max_chars_per_chunk <= 0:
            raise ValueError("max_chars_per_chunk must be positive")
        self.generator = generator
        self.model_name = generator.model_name
        self.temperature = temperature
        self.max_chars_per_chunk = max_chars_per_chunk

    def rewrite(
        self, original_query: str, preliminary_results: Sequence[SearchResult]
    ) -> RewriteResult:
        original = original_query.strip()
        if not original:
            raise ValueError("original_query cannot be empty")
        prompt = build_rewrite_prompt(
            original,
            preliminary_results,
            max_chars_per_chunk=self.max_chars_per_chunk,
        )
        started = perf_counter()
        error = None
        fallback = False
        try:
            raw = self.generator.generate(prompt, temperature=self.temperature)
            rewritten = parse_rewritten_query(raw)
        except Exception as exc:
            rewritten = original
            fallback = True
            error = f"{type(exc).__name__}: {exc}"
        latency = perf_counter() - started
        identity = diagnose_identity_preservation(original, rewritten)
        return RewriteResult(
            original_query=original,
            rewritten_query=rewritten,
            changed=rewritten != original,
            latency_seconds=latency,
            model_name=self.model_name,
            fallback=fallback,
            error=error,
            identity=identity,
        )
