import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Self-Correcting RAG Evaluator"

    # LLM Config
    LLM_PROVIDER: str = "groq"
    LLM_MODEL: str = "llama-3.3-70b-specdec"
    GROQ_API_KEY: str = ""

    # Embedding + Vector Store
    EMBEDDING_PROVIDER: str = "ollama"
    EMBEDDING_MODEL: str = "mxbai-embed-large"
    CHROMA_COLLECTION: str = "weekend_rag"

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
    RETRY_THRESHOLD: float = 0.7

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_directories(self) -> None:
        for directory in (self.STORAGE_DIR, self.DATA_DIR, self.CHROMA_PATH):
            os.makedirs(directory, exist_ok=True)


settings = Settings()
