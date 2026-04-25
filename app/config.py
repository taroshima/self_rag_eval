from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    APP_NAME: str = "Self-Correcting RAG"
    
    # LLM Config
    LLM_PROVIDER: str = "groq"
    GROQ_API_KEY: str = ""
    
    # Paths (relative to app/ main.py)
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    CHROMA_PATH: str = os.path.join(BASE_DIR, "../storage/chroma")
    SQLITE_PATH: str = os.path.join(BASE_DIR, "../storage/runs.db")
    
    # RAG Settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 3
    RETRY_THRESHOLD: float = 0.7

    class Config:
        env_file = ".env"

settings = Settings()