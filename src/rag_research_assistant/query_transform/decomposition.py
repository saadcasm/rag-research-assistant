"""Experimental query decomposition with an auditable, bounded JSON contract."""

import json
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Sequence, Tuple

from ..generation import TextGenerator


MAX_SUBQUESTIONS = 3
DECOMPOSITION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "needs_decomposition": {"type": "boolean"},
        "subquestions": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": MAX_SUBQUESTIONS,
        },
    },
    "required": ["needs_decomposition", "subquestions"],
    "additionalProperties": False,
}

DECOMPOSITION_PROMPT_TEMPLATE = """You decompose research questions for an information-retrieval experiment.

Decide whether answering the question requires multiple DISTINCT pieces of evidence.

Rules:
1. Split only when multiple distinct evidence needs exist. Simple single-evidence questions must set needs_decomposition=false.
2. Produce complementary sub-questions, not alternative phrasings of the same search.
3. Preserve named entities, paper names, method names, acronyms, and technical terms exactly.
4. Do not answer the original question and do not add requirements it did not ask for.
5. Each sub-question must describe exactly one evidence need.
6. Prefer two sub-questions; use at most three.
7. When needs_decomposition=false, return an empty subquestions array.
8. Return only JSON matching the supplied schema.

Examples:
Question: What learning signal does REALM use?
Output: {{"needs_decomposition": false, "subquestions": []}}

Question: How do DPR and ColBERT differ in representation and retrieval?
Output: {{"needs_decomposition": true, "subquestions": ["How does DPR represent queries and passages and perform retrieval?", "How does ColBERT represent queries and documents and perform retrieval?"]}}

QUESTION
{question}

JSON
"""


@dataclass(frozen=True)
class RejectedSubquestion:
    text: str
    reason: str


@dataclass(frozen=True)
class ParsedDecomposition:
    needs_decomposition: bool
    subquestions: Tuple[str, ...]


@dataclass(frozen=True)
class DecompositionResult:
    original_question: str
    needs_decomposition: bool
    raw_response: str
    generated_subquestions: Tuple[str, ...]
    subquestions: Tuple[str, ...]
    rejected_subquestions: Tuple[RejectedSubquestion, ...]
    model_name: str
    latency_seconds: float
    structured_output_used: bool
    fallback: bool
    error: str | None

    @property
    def retrieval_queries(self) -> Tuple[str, ...]:
        return self.subquestions if self.needs_decomposition else (self.original_question,)


def build_decomposition_prompt(question: str) -> str:
    if not question.strip():
        raise ValueError("question cannot be empty")
    return DECOMPOSITION_PROMPT_TEMPLATE.format(question=question.strip())


def parse_decomposition(raw_response: str) -> ParsedDecomposition:
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise ValueError("decomposer returned an empty response")
    value = raw_response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, re.I | re.S)
    if fenced:
        value = fenced.group(1)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"decomposer returned invalid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("decomposition response must be an object")
    decision = parsed.get("needs_decomposition")
    questions = parsed.get("subquestions")
    if not isinstance(decision, bool):
        raise ValueError("needs_decomposition must be a boolean")
    if not isinstance(questions, list) or any(not isinstance(item, str) for item in questions):
        raise ValueError("subquestions must be an array of strings")
    return ParsedDecomposition(decision, tuple(questions))


_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "by", "does", "for", "from", "how",
    "in", "is", "of", "on", "or", "the", "to", "what", "which", "with",
}


def _normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def _content_signature(text: str) -> frozenset[str]:
    return frozenset(
        word for word in _WORD.findall(text.casefold()) if word not in _STOPWORDS
    )


def validate_subquestions(
    original_question: str,
    values: Sequence[str],
    *,
    maximum: int = MAX_SUBQUESTIONS,
) -> tuple[Tuple[str, ...], Tuple[RejectedSubquestion, ...]]:
    """Reject deterministic contract violations without opaque similarity scores."""

    valid: list[str] = []
    rejected: list[RejectedSubquestion] = []
    original_normalized = _normalized(original_question)
    original_signature = _content_signature(original_question)
    seen: set[str] = set()
    signatures: set[frozenset[str]] = set()
    for value in values:
        text = " ".join(value.split())
        normalized = _normalized(text)
        signature = _content_signature(text)
        reason = None
        if not text:
            reason = "empty"
        elif normalized == original_normalized:
            reason = "duplicates_original"
        elif normalized in seen:
            reason = "duplicate_subquestion"
        elif signature and signature == original_signature:
            reason = "superficial_paraphrase_of_original"
        elif signature and signature in signatures:
            reason = "superficial_paraphrase_of_subquestion"
        elif len(valid) >= maximum:
            reason = "exceeds_maximum"
        if reason:
            rejected.append(RejectedSubquestion(text, reason))
            continue
        seen.add(normalized)
        signatures.add(signature)
        valid.append(text)
    return tuple(valid), tuple(rejected)


class QueryDecomposer:
    """Generate and validate at most three complementary evidence questions."""

    def __init__(self, generator: TextGenerator, *, temperature: float = 0.0) -> None:
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        self.generator = generator
        self.model_name = generator.model_name
        self.temperature = temperature

    def decompose(self, question: str) -> DecompositionResult:
        original = question.strip()
        prompt = build_decomposition_prompt(original)
        started = perf_counter()
        raw = ""
        generated: Tuple[str, ...] = ()
        valid: Tuple[str, ...] = ()
        rejected: Tuple[RejectedSubquestion, ...] = ()
        decision = False
        structured = False
        error = None
        try:
            structured_generate = getattr(self.generator, "generate_structured", None)
            if callable(structured_generate):
                structured = True
                raw = structured_generate(prompt, DECOMPOSITION_SCHEMA, self.temperature)
            else:
                raw = self.generator.generate(prompt, temperature=self.temperature)
            parsed = parse_decomposition(raw)
            generated = parsed.subquestions
            valid, rejected = validate_subquestions(original, generated)
            if not parsed.needs_decomposition:
                rejected = (*rejected, *(RejectedSubquestion(item, "decision_was_false") for item in valid))
                valid = ()
            elif len(valid) < 2:
                error = "decomposition requires at least two valid complementary subquestions"
            else:
                decision = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        return DecompositionResult(
            original, decision, raw, generated, valid if decision else (),
            tuple(rejected), self.model_name, perf_counter() - started,
            structured, error is not None, error,
        )
