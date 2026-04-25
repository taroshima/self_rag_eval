import chromadb

from app.config import settings


class VectorStore:
    def __init__(self) -> None:
        self.client: chromadb.PersistentClient | None = None
        self.collection = None

    def initialize(self):
        settings.ensure_directories()
        self.client = chromadb.PersistentClient(path=str(settings.CHROMA_PATH))
        self.collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION
        )
        return self.collection

    def is_ready(self) -> bool:
        return self.collection is not None


vector_store = VectorStore()
