import ollama
import chromadb
from app.config import settings

# Initialize Chroma
client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
collection = client.get_or_create_collection(name="weekend_rag")

def ingest_document(file_path: str):
    # Get the chunks from our ingestion service
    from app.services.ingestion import process_pdf
    chunks_with_metadata = process_pdf(file_path)
    
    for item in chunks_with_metadata:
        response = ollama.embed(
            model="mxbai-embed-large", 
            input=item["content"]
        )
        
        # Store in Chroma
        collection.add(
            ids=[f"{item['metadata']['source']}_{item['metadata']['chunk_index']}"],
            embeddings=[response["embeddings"][0]],
            documents=[item["content"]],
            metadatas=[item["metadata"]]
        )
    return len(chunks_with_metadata)