import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings
from app.models.schemas import AskRequest, AskResponse, DashboardSummary, FailureRecord, IngestResponse, RunSummary
from app.services.ingestion import process_document
from app.services.rag import answer_question
from app.services.vector_store import vector_store
from app.storage.db import check_database, get_dashboard_summary, get_failures, get_recent_runs, save_answer_run

router = APIRouter()
logger = logging.getLogger(__name__)


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
        "document_count": vector_store.count() if vector_ready else 0,
    }


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    extension = Path(file.filename).suffix.lower()
    if extension not in {".pdf", ".txt", ".md"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Supported types are .pdf, .txt, and .md.",
        )

    settings.ensure_directories()
    destination = settings.DATA_DIR / file.filename
    file_bytes = await file.read()
    destination.write_bytes(file_bytes)

    try:
        chunks = process_document(destination)
        chunk_count = vector_store.add_chunks_with_rollback(chunks)
    except ValueError as exc:
        if destination.exists():
            destination.unlink()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingestion failed for %s", file.filename)
        if destination.exists():
            destination.unlink()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return IngestResponse(
        filename=file.filename,
        chunk_count=chunk_count,
        collection_name=settings.CHROMA_COLLECTION,
    )


@router.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        run = answer_question(question=question, top_k=payload.top_k)
        run.run_id = save_answer_run(run)
        return AskResponse(**run.model_dump())
    except Exception as exc:
        logger.exception("Question answering failed for question: %s", question)
        raise HTTPException(status_code=500, detail=f"Question answering failed: {exc}") from exc


@router.get("/runs", response_model=list[RunSummary])
def list_runs(limit: int = 10) -> list[RunSummary]:
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100.")
    return get_recent_runs(limit=limit)


@router.get("/failures", response_model=list[FailureRecord])
def list_failures(limit: int = 10) -> list[FailureRecord]:
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100.")
    return get_failures(limit=limit)


@router.get("/dashboard", response_model=DashboardSummary)
def dashboard(limit: int = 50) -> DashboardSummary:
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200.")
    return get_dashboard_summary(limit=limit)
