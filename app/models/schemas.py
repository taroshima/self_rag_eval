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


class AnswerRun(BaseModel):
    question: str
    answer: str
    provider: str
    prompt_version: str
    retrieved_chunks: list[DocumentChunk] = Field(default_factory=list)
    metrics: dict[str, float | None] = Field(default_factory=dict)
    total_score: float | None = None
    issue_labels: list[str] = Field(default_factory=list)
    corrected_answer: str | None = None
    corrected_metrics: dict[str, float | None] = Field(default_factory=dict)
    improvement_status: str | None = None
