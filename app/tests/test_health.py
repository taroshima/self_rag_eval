from fastapi.testclient import TestClient

from app.main import app


def test_health_check_reports_ready_services():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["database_ready"] is True
    assert "document_count" in payload
