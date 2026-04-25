from fastapi import APIRouter

from app.config import settings
from app.services.vector_store import vector_store
from app.storage.db import check_database

router = APIRouter()


@router.get("/health")
def health_check() -> dict:
    database_ready = check_database()
    vector_ready = vector_store.is_ready()

    return {
        "app_name": settings.APP_NAME,
        "status": "ok" if database_ready and vector_ready else "degraded",
        "provider": settings.LLM_PROVIDER,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "database_ready": database_ready,
        "vector_store_ready": vector_ready,
        "chroma_collection": settings.CHROMA_COLLECTION,
    }
