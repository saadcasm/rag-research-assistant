import json

import pytest

from rag_research_assistant.query_transform.decomposition import (
    DECOMPOSITION_SCHEMA,
    QueryDecomposer,
    build_decomposition_prompt,
    parse_decomposition,
    validate_subquestions,
)


class FakeGenerator:
    model_name = "fake"

    def __init__(self, response: str, *, structured: bool = False) -> None:
        self.response = response
        self.calls = []
        if structured:
            self.generate_structured = self._generate_structured

    def generate(self, prompt: str, temperature: float) -> str:
        self.calls.append(("plain", prompt, temperature))
        return self.response

    def _generate_structured(self, prompt: str, schema, temperature: float) -> str:
        self.calls.append(("structured", prompt, schema, temperature))
        return self.response


def test_prompt_explains_complementary_not_paraphrase_and_preserves_terms() -> None:
    prompt = build_decomposition_prompt("Compare DPR and ColBERT.")
    assert "complementary sub-questions" in prompt
    assert "not alternative phrasings" in prompt
    assert "Preserve named entities" in prompt
    assert "Compare DPR and ColBERT." in prompt


@pytest.mark.parametrize(
    "raw",
    [
        '{"needs_decomposition": true, "subquestions": ["A?", "B?"]}',
        '```json\n{"needs_decomposition": true, "subquestions": ["A?", "B?"]}\n```',
    ],
)
def test_parse_structured_response(raw: str) -> None:
    parsed = parse_decomposition(raw)
    assert parsed.needs_decomposition is True
    assert parsed.subquestions == ("A?", "B?")


def test_invalid_json_falls_back_and_preserves_raw_response() -> None:
    result = QueryDecomposer(FakeGenerator("not JSON")).decompose("Compare DPR and ColBERT")
    assert result.fallback is True
    assert result.needs_decomposition is False
    assert result.raw_response == "not JSON"
    assert "invalid JSON" in result.error


def test_false_decision_keeps_original_and_rejects_extraneous_subquestions() -> None:
    response = json.dumps({"needs_decomposition": False, "subquestions": ["Ignored?"]})
    result = QueryDecomposer(FakeGenerator(response)).decompose("What is REALM?")
    assert result.retrieval_queries == ("What is REALM?",)
    assert result.fallback is False
    assert result.subquestions == ()
    assert result.rejected_subquestions[0].reason == "decision_was_false"


def test_duplicate_original_duplicate_subquestion_and_maximum_are_rejected() -> None:
    values = [
        "Compare DPR and ColBERT",
        "How does DPR retrieve?",
        "  how does dpr retrieve? ",
        "How does ColBERT retrieve?",
        "What is late interaction?",
        "What is a fourth need?",
    ]
    valid, rejected = validate_subquestions("Compare DPR and ColBERT", values, maximum=3)
    assert valid == (
        "How does DPR retrieve?",
        "How does ColBERT retrieve?",
        "What is late interaction?",
    )
    assert [item.reason for item in rejected] == [
        "duplicates_original",
        "duplicate_subquestion",
        "exceeds_maximum",
    ]


def test_two_complementary_subquestions_are_retained_with_native_schema() -> None:
    fake = FakeGenerator(json.dumps({
        "needs_decomposition": True,
        "subquestions": [
            "How does DPR represent queries and passages?",
            "How does ColBERT represent queries and documents?",
        ],
    }), structured=True)
    result = QueryDecomposer(fake).decompose("Compare DPR and ColBERT representations")
    assert result.needs_decomposition is True
    assert len(result.subquestions) == 2
    assert result.structured_output_used is True
    assert fake.calls[0][2] == DECOMPOSITION_SCHEMA


def test_one_valid_subquestion_cannot_masquerade_as_decomposition() -> None:
    response = json.dumps({"needs_decomposition": True, "subquestions": ["How does DPR work?"]})
    result = QueryDecomposer(FakeGenerator(response)).decompose("Compare DPR and ColBERT")
    assert result.fallback is True
    assert result.needs_decomposition is False
    assert "at least two" in result.error
