"""FastAPI transport for the process-scoped RAG application service."""

import logging
from contextlib import asynccontextmanager
from typing import Annotated, Callable, Literal, Optional, Protocol

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from .application import ApplicationSettings, RAGApplication, build_application
from .generation import (
    GenerationError,
    OllamaModelMissingError,
    OllamaUnavailableError,
)
from .models import CorpusIndexMetadata, GroundedAnswer
from .qdrant_store import InvalidQdrantIndexError


logger = logging.getLogger(__name__)
QuestionText = Annotated[StrictStr, Field(min_length=1, max_length=4_000)]
TopK = Annotated[StrictInt, Field(ge=1, le=20)]
DependencyStatus = Literal["ready", "unavailable", "unknown"]


class AskService(Protocol):
    """The small application boundary consumed by the HTTP transport."""

    retrieval_strategy: str
    retrieval_score_type: str
    dense_backend: Optional[str]
    metadata: CorpusIndexMetadata
    generator: object

    def ask(self, question: str, top_k: int) -> GroundedAnswer: ...

    def close(self) -> None: ...


class AskRequest(BaseModel):
    """Validated public input for one grounded question."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: QuestionText
    top_k: TopK = 5


class SourceResponse(BaseModel):
    """Stable citation provenance returned to API clients."""

    citation_number: int
    score: float
    document: str
    start_page: int
    end_page: int
    chunk_id: str


class AskMetadataResponse(BaseModel):
    """Useful execution facts without exposing internal tuning controls."""

    top_k: int
    retrieval_strategy: str
    retrieval_score_type: str
    dense_backend: Optional[str]
    generation_model: str
    embedding_model: str
    indexed_chunks: int


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    metadata: AskMetadataResponse


class HealthResponse(BaseModel):
    """Cheap liveness plus the startup readiness snapshot."""

    status: Literal["ok", "degraded"]
    live: bool
    ready: bool
    dependencies: dict[str, DependencyStatus]


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    detail: ErrorBody


ServiceFactory = Callable[[ApplicationSettings], RAGApplication]


def _http_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


def _startup_dependencies(exc: Exception) -> dict[str, DependencyStatus]:
    if isinstance(exc, (OllamaUnavailableError, OllamaModelMissingError)):
        return {"qdrant": "ready", "ollama": "unavailable"}
    if isinstance(exc, (InvalidQdrantIndexError, FileNotFoundError)):
        return {"qdrant": "unavailable", "ollama": "unknown"}
    return {"qdrant": "unknown", "ollama": "unknown"}


def create_app(
    *,
    settings: Optional[ApplicationSettings] = None,
    service: Optional[AskService] = None,
    service_factory: ServiceFactory = build_application,
) -> FastAPI:
    """Create an app that can use either real process resources or a test fake."""

    configured_settings = settings or ApplicationSettings.from_environment()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.rag_service = service
        application.state.startup_error = None
        application.state.dependencies = {
            "qdrant": "ready" if service is not None else "unknown",
            "ollama": "ready" if service is not None else "unknown",
        }
        if service is None:
            try:
                application.state.rag_service = service_factory(configured_settings)
                application.state.dependencies = {
                    "qdrant": "ready",
                    "ollama": "ready",
                }
            except Exception as exc:
                logger.exception("RAG application startup failed")
                application.state.startup_error = exc
                application.state.dependencies = _startup_dependencies(exc)
        try:
            yield
        finally:
            loaded_service = application.state.rag_service
            if loaded_service is not None:
                loaded_service.close()

    application = FastAPI(
        title="RAG Research Assistant API",
        version="0.8.0",
        description="Local grounded question answering over the verified research corpus.",
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse, tags=["operations"])
    def health() -> HealthResponse:
        ready = application.state.rag_service is not None
        return HealthResponse(
            status="ok" if ready else "degraded",
            live=True,
            ready=ready,
            dependencies=application.state.dependencies,
        )

    @application.post(
        "/ask",
        response_model=AskResponse,
        responses={
            502: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["questions"],
    )
    def ask(request: AskRequest) -> AskResponse:
        rag_service: Optional[AskService] = application.state.rag_service
        if rag_service is None:
            raise _http_error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "service_not_ready",
                "The local RAG dependencies are not ready.",
            )
        try:
            result = rag_service.ask(request.question, request.top_k)
        except (OllamaUnavailableError, OllamaModelMissingError):
            logger.exception("Ollama is unavailable during generation")
            raise _http_error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "generation_unavailable",
                "The local generation dependency is unavailable.",
            ) from None
        except InvalidQdrantIndexError:
            logger.exception("Qdrant became unavailable during retrieval")
            raise _http_error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "retrieval_unavailable",
                "The local retrieval dependency is unavailable.",
            ) from None
        except GenerationError:
            logger.exception("Ollama failed while generating an answer")
            raise _http_error(
                status.HTTP_502_BAD_GATEWAY,
                "generation_failed",
                "The local generation dependency returned an invalid response.",
            ) from None
        except Exception:
            logger.exception("Unexpected failure while answering a request")
            raise _http_error(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "internal_error",
                "The request could not be completed.",
            ) from None

        sources = [
            SourceResponse(
                citation_number=source.citation_number,
                score=source.score,
                document=source.document,
                start_page=source.page_number,
                end_page=source.end_page or source.page_number,
                chunk_id=source.chunk_id,
            )
            for source in result.sources
        ]
        generator_name = getattr(rag_service.generator, "model_name", "unknown")
        return AskResponse(
            answer=result.text,
            sources=sources,
            metadata=AskMetadataResponse(
                top_k=request.top_k,
                retrieval_strategy=rag_service.retrieval_strategy,
                retrieval_score_type=rag_service.retrieval_score_type,
                dense_backend=rag_service.dense_backend,
                generation_model=generator_name,
                embedding_model=rag_service.metadata.embedding_model,
                indexed_chunks=rag_service.metadata.chunk_count,
            ),
        )

    return application


app = create_app()
