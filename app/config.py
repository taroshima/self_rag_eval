import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Self-Correcting RAG Evaluator"

    # LLM Config
    LLM_PROVIDER: str = "groq"
    LLM_MODEL: str = "openai/gpt-oss-120b"
    GROQ_API_KEY: str = ""

    # Embedding + Vector Store
    EMBEDDING_PROVIDER: str = "ollama"
    EMBEDDING_MODEL: str = "mxbai-embed-large"
    CHROMA_COLLECTION: str = "self_correcting_rag"

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent
    STORAGE_DIR: Path = BASE_DIR / "storage"
    DATA_DIR: Path = BASE_DIR / "data"
    CHROMA_PATH: Path = STORAGE_DIR / "chroma"
    SQLITE_PATH: Path = STORAGE_DIR / "runs.db"

    # RAG Settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 3
    MAX_RETRIEVAL_DEPTH: int = 8
    RETRY_THRESHOLD: float = 70.0
    FAILURE_THRESHOLD: float = 60.0
    HARD_FAITHFULNESS_FLOOR: float = 0.55
    RETRY_TOP_K_INCREMENT: int = 2
    SCORE_IMPROVEMENT_EPSILON: float = 1.0
    INGEST_TIMEOUT_SECONDS: int = 600

    # Evaluation Settings
    RAGAS_ENABLED: bool = True
    METRIC_WEIGHT_FAITHFULNESS: float = 0.45
    METRIC_WEIGHT_RELEVANCY: float = 0.30
    METRIC_WEIGHT_CONTEXT_PRECISION: float = 0.25

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_directories(self) -> None:
        for directory in (
            self.STORAGE_DIR,
            self.DATA_DIR,
            self.CHROMA_PATH,
        ):
            os.makedirs(directory, exist_ok=True)


settings = Settings()
