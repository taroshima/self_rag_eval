# Self-Correcting RAG Evaluator

RAG app with:

- FastAPI backend
- Streamlit UI
- Chroma vector store
- SQLite run history
- optional Ragas evaluation with heuristic fallback
- one-step self-correction loop

## Quick start

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

In another terminal:

```powershell
.\venv\Scripts\Activate.ps1
streamlit run app/ui/streamlit_app.py
```

## Environment

Create `.env` with:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=mxbai-embed-large
```

For an offline UI-only smoke test, you can use:

```env
LLM_PROVIDER=mock
```

## Ollama

Install Ollama locally and pull the embedding model:

```powershell
ollama pull mxbai-embed-large
```

## Basic flow

1. Start the API and Streamlit app.
2. Upload PDF, TXT, or MD files in the Ingest tab.
3. Ask questions against the ingested documents.
4. Inspect run quality in the Dashboard tab.
