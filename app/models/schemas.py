from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class LLMProvider(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the configured provider name for health and run metadata."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a completion from a system and user prompt."""


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the configured embedding provider name."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for multiple documents."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Return an embedding for a single search query."""


class DocumentChunk(BaseModel):
    id: str | None = None
    content: str
    source: str
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    query: str
    top_k: int
    chunks: list[DocumentChunk] = Field(default_factory=list)


class MetricSet(BaseModel):
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    total_score: float | None = None
    evaluator: str = "heuristic"
    ragas_fallback_reason: str | None = None
    score_reliability: str = "medium"


class RunAttempt(BaseModel):
    answer: str
    prompt_version: str
    top_k: int
    retrieved_chunks: list[DocumentChunk] = Field(default_factory=list)
    metrics: MetricSet = Field(default_factory=MetricSet)
    issue_labels: list[str] = Field(default_factory=list)


class AnswerRun(BaseModel):
    question: str
    provider: str
    retry_triggered: bool = False
    retry_reason: str | None = None
    improvement_status: str = "unchanged"
    selected_attempt: str = "initial"
    final_answer: str
    initial_attempt: RunAttempt
    corrected_attempt: RunAttempt | None = None
    run_id: int | None = None
    created_at: str | None = None


class IngestResponse(BaseModel):
    filename: str
    chunk_count: int
    collection_name: str


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


class AskResponse(AnswerRun):
    pass


class RunSummary(BaseModel):
    id: int
    question: str
    provider: str
    final_answer: str
    selected_attempt: str
    retry_triggered: bool
    retry_reason: str | None = None
    improvement_status: str
    final_score: float | None = None
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    prompt_version: str
    created_at: str


class FailureRecord(RunSummary):
    issue_labels: list[str] = Field(default_factory=list)


class PromptVersionSummary(BaseModel):
    prompt_version: str
    count: int
    average_score: float


class DashboardSummary(BaseModel):
    total_runs: int
    average_score: float
    retry_rate: float
    improved_rate: float
    recent_scores: list[float] = Field(default_factory=list)
    issue_breakdown: dict[str, int] = Field(default_factory=dict)
    prompt_versions: list[PromptVersionSummary] = Field(default_factory=list)
    recent_runs: list[RunSummary] = Field(default_factory=list)

