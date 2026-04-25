from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings

def process_pdf(file_path: str):
    # 1. LOAD: Extract text from PDF
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    # 2. CHUNK: Use Recursive Splitting
    # Why this? See the "Mindset" section below.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_text(full_text)
    
    # 3. FORMAT: Add metadata
    processed_data = []
    for i, chunk in enumerate(chunks):
        processed_data.append({
            "content": chunk,
            "metadata": {
                "source": file_path.split("/")[-1],
                "chunk_index": i
            }
        })
    
    return processed_data