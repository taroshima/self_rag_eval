from app.config import settings
from app.models.schemas import EmbeddingProvider


class OllamaEmbeddingProvider(EmbeddingProvider):
    @property
    def provider_name(self) -> str:
        return "ollama"

    def _client(self):
        try:
            import ollama
        except ImportError as exc:  # pragma: no cover - local dependency
            raise RuntimeError(
                "The ollama package is not installed. Install it before using Ollama embeddings."
            ) from exc
        return ollama

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._client()
        response = client.embed(
            model=settings.EMBEDDING_MODEL,
            input=texts,
        )
        return response["embeddings"]

    def embed_query(self, text: str) -> list[float]:
        client = self._client()
        response = client.embed(
            model=settings.EMBEDDING_MODEL,
            input=text,
        )
        return response["embeddings"][0]


def get_embedding_provider() -> EmbeddingProvider:
    if settings.EMBEDDING_PROVIDER == "ollama":
        return OllamaEmbeddingProvider()
    raise ValueError(f"Unsupported embedding provider: {settings.EMBEDDING_PROVIDER}")
