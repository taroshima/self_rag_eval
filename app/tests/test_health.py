from fastapi.testclient import TestClient

from app.main import app


def test_health_check_reports_ready_services():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["database_ready"] is True
    assert payload["vector_store_ready"] is True
