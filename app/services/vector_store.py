import chromadb

from app.config import settings
from app.models.schemas import DocumentChunk, RetrievalResult
from app.services.embeddings import get_embedding_provider


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

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0
        if self.collection is None:
            self.initialize()

        ids, documents, metadatas, embeddings = self._build_payload(chunks)
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        return len(chunks)

    def add_chunks_with_rollback(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0
        if self.collection is None:
            self.initialize()

        ids, documents, metadatas, embeddings = self._build_payload(chunks)
        backup = self._snapshot_existing(ids)

        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception:
            self.restore_snapshot(ids=ids, snapshot=backup)
            raise

        return len(chunks)

    def restore_snapshot(self, ids: list[str], snapshot: dict) -> None:
        if self.collection is None:
            self.initialize()

        existing_ids = snapshot.get("ids", [])
        existing_id_set = set(existing_ids)
        new_ids = [chunk_id for chunk_id in ids if chunk_id not in existing_id_set]
        if new_ids:
            self.collection.delete(ids=new_ids)

        if existing_ids:
            documents = snapshot.get("documents", [])
            metadatas = snapshot.get("metadatas", [])
            embeddings = snapshot.get("embeddings", [])
            self.collection.upsert(
                ids=existing_ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

    def _build_payload(
        self, chunks: list[DocumentChunk]
    ) -> tuple[list[str], list[str], list[dict], list[list[float]]]:
        ids = [chunk.id or f"{chunk.source}_{chunk.chunk_index}" for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        embeddings = get_embedding_provider().embed_texts(documents)
        return ids, documents, metadatas, embeddings

    def _snapshot_existing(self, ids: list[str]) -> dict:
        if not ids:
            return {"ids": [], "documents": [], "metadatas": [], "embeddings": []}

        existing = self.collection.get(
            ids=ids,
            include=["documents", "metadatas", "embeddings"],
        )
        return {
            "ids": existing.get("ids", []),
            "documents": existing.get("documents", []),
            "metadatas": existing.get("metadatas", []),
            "embeddings": existing.get("embeddings", []),
        }

    def query(self, question: str, top_k: int) -> RetrievalResult:
        if self.collection is None:
            self.initialize()
        query_embedding = get_embedding_provider().embed_query(question)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[DocumentChunk] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=False
        ):
            chunk_metadata = metadata or {}
            chunk_metadata = {**chunk_metadata, "distance": distance}
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    content=document,
                    source=chunk_metadata.get("source", "unknown"),
                    chunk_index=int(chunk_metadata.get("chunk_index", 0)),
                    metadata=chunk_metadata,
                )
            )

        return RetrievalResult(query=question, top_k=top_k, chunks=chunks)

    def count(self) -> int:
        if self.collection is None:
            self.initialize()
        return int(self.collection.count())


vector_store = VectorStore()
