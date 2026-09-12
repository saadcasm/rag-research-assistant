"""Local text generation through Ollama's small HTTP API."""

import json
import shutil
from typing import Any, Dict, List, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_OLLAMA_MODEL = "qwen3.5:4b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_TEMPERATURE = 0.1


class GenerationError(RuntimeError):
    """Base class for clear local-generation failures."""


class OllamaUnavailableError(GenerationError):
    """Raised when the local Ollama server cannot be reached."""


class OllamaModelMissingError(GenerationError):
    """Raised when the requested model is not installed in Ollama."""


class TextGenerator(Protocol):
    """The behavior the RAG pipeline needs from any local generator."""

    model_name: str

    def generate(self, prompt: str, temperature: float) -> str: ...


class OllamaGenerator:
    """Generate text with a configurable model on a local Ollama server."""

    def __init__(
        self,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = 180.0,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._model_verified = False

    def _request_json(
        self, path: str, payload: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise GenerationError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, ConnectionError) as exc:
            if shutil.which("ollama") is None:
                raise OllamaUnavailableError(
                    "Ollama is not installed or accessible. Install it with:\n"
                    "  curl -fsSL https://ollama.com/install.sh | sh"
                ) from exc
            raise OllamaUnavailableError(
                f"Ollama is installed but its server is not reachable at {self.base_url}. "
                "Start the Ollama app or run: ollama serve"
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise GenerationError("Ollama returned an invalid JSON response") from exc

    def list_local_models(self) -> List[str]:
        """Return locally installed Ollama model names."""

        response = self._request_json("/api/tags")
        models = response.get("models")
        if not isinstance(models, list):
            raise GenerationError("Ollama's model-list response is missing 'models'")
        return [
            model["name"]
            for model in models
            if isinstance(model, dict) and isinstance(model.get("name"), str)
        ]

    def ensure_model_available(self) -> None:
        """Fail clearly before generation if the selected model is absent."""

        local_models = self.list_local_models()
        if self.model_name not in local_models:
            raise OllamaModelMissingError(
                f"Ollama model {self.model_name!r} is not installed. "
                "Pull exactly this model with:\n"
                f"  ollama pull {self.model_name}"
            )
        self._model_verified = True

    def generate(self, prompt: str, temperature: float = DEFAULT_TEMPERATURE) -> str:
        """Send one grounded prompt to Ollama and return its text response."""

        if not prompt.strip():
            raise ValueError("prompt cannot be empty")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if not self._model_verified:
            self.ensure_model_available()

        response = self._request_json(
            "/api/generate",
            {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {"temperature": temperature},
            },
        )
        text = response.get("response")
        if not isinstance(text, str) or not text.strip():
            raise GenerationError("Ollama returned an empty generation")
        return text.strip()
