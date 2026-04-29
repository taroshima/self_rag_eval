from io import BytesIO

from fastapi.testclient import TestClient

from app.main import app


def test_ingest_text_file(monkeypatch):
    def fake_add_chunks(chunks):
        assert chunks
        assert chunks[0].source == "notes.txt"
        return len(chunks)

    monkeypatch.setattr("app.api.routes.vector_store.add_chunks_with_rollback", fake_add_chunks)

    with TestClient(app) as client:
        response = client.post(
            "/ingest",
            files={"file": ("notes.txt", BytesIO(b"hello world\nthis is a test"), "text/plain")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "notes.txt"
    assert payload["chunk_count"] >= 1
    assert payload["collection_name"] == "self_correcting_rag"


def test_ingest_rejects_unsupported_files():
    with TestClient(app) as client:
        response = client.post(
            "/ingest",
            files={"file": ("notes.docx", BytesIO(b"not supported"), "application/octet-stream")},
        )

    assert response.status_code == 400


def test_ingest_rolls_back_when_vector_write_fails(monkeypatch):
    def fake_add_chunks(chunks):
        raise RuntimeError("vector write failed")

    monkeypatch.setattr("app.api.routes.vector_store.add_chunks_with_rollback", fake_add_chunks)

    with TestClient(app) as client:
        response = client.post(
            "/ingest",
            files={"file": ("broken.txt", BytesIO(b"hello world"), "text/plain")},
        )

    assert response.status_code == 500
