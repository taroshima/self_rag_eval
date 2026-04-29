from pathlib import Path

from PyPDF2 import PdfReader

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - exercised only in lightweight environments
    RecursiveCharacterTextSplitter = None

from app.config import settings
from app.models.schemas import DocumentChunk


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def _load_pdf(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text


def _load_text_file(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="ignore")


def _load_text(file_path: Path) -> str:
    if file_path.suffix.lower() == ".pdf":
        return _load_pdf(file_path)
    if file_path.suffix.lower() in {".txt", ".md"}:
        return _load_text_file(file_path)
    raise ValueError(f"Unsupported file type: {file_path.suffix}")


def _split_text(full_text: str) -> list[str]:
    if RecursiveCharacterTextSplitter is not None:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )
        return text_splitter.split_text(full_text)

    chunk_size = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP
    if not full_text.strip():
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(full_text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = full_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start = max(end - overlap, start + 1)
    return chunks


def process_document(file_path: str | Path) -> list[DocumentChunk]:
    resolved_path = Path(file_path)
    extension = resolved_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. Supported types are .pdf, .txt, .md."
        )

    full_text = _load_text(resolved_path)
    raw_chunks = _split_text(full_text)

    processed_chunks: list[DocumentChunk] = []
    for index, chunk in enumerate(raw_chunks):
        processed_chunks.append(
            DocumentChunk(
                id=f"{resolved_path.stem}_{index}",
                content=chunk,
                source=resolved_path.name,
                chunk_index=index,
                metadata={
                    "source": resolved_path.name,
                    "chunk_index": index,
                    "file_type": extension.lstrip("."),
                },
            )
        )

    return processed_chunks
