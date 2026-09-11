from urllib.error import URLError

import pytest

import rag_research_assistant.generation as generation
from rag_research_assistant.generation import (
    OllamaGenerator,
    OllamaModelMissingError,
    OllamaUnavailableError,
)


def test_unavailable_ollama_has_clear_install_command(monkeypatch) -> None:
    def fail_urlopen(request, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr(generation, "urlopen", fail_urlopen)
    monkeypatch.setattr(generation.shutil, "which", lambda name: None)

    with pytest.raises(OllamaUnavailableError, match=r"curl -fsSL https://ollama.com/install.sh"):
        OllamaGenerator(timeout=1).ensure_model_available()


def test_stopped_ollama_server_has_clear_start_command(monkeypatch) -> None:
    def fail_urlopen(request, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr(generation, "urlopen", fail_urlopen)
    monkeypatch.setattr(generation.shutil, "which", lambda name: "/usr/local/bin/ollama")

    with pytest.raises(OllamaUnavailableError, match=r"ollama serve"):
        OllamaGenerator(timeout=1).ensure_model_available()


def test_missing_model_has_exact_pull_command(monkeypatch) -> None:
    generator = OllamaGenerator(model_name="qwen3.5:4b")
    monkeypatch.setattr(generator, "list_local_models", lambda: ["another:latest"])

    with pytest.raises(OllamaModelMissingError, match=r"ollama pull qwen3\.5:4b"):
        generator.ensure_model_available()


def test_generate_sends_model_prompt_and_temperature(monkeypatch) -> None:
    generator = OllamaGenerator(model_name="local:test")
    requests = []

    def fake_request(path, payload=None):
        requests.append((path, payload))
        if path == "/api/tags":
            return {"models": [{"name": "local:test"}]}
        return {"response": " Grounded answer [1]. "}

    monkeypatch.setattr(generator, "_request_json", fake_request)

    assert generator.generate("grounded prompt", temperature=0.1) == "Grounded answer [1]."
    assert requests[0] == ("/api/tags", None)
    assert requests[1][0] == "/api/generate"
    assert requests[1][1] == {
        "model": "local:test",
        "prompt": "grounded prompt",
        "stream": False,
        "options": {"temperature": 0.1},
    }
