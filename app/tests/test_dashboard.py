from fastapi.testclient import TestClient

from app.main import app


def test_dashboard_endpoint_returns_summary(monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.get_dashboard_summary",
        lambda limit: {
            "total_runs": 1,
            "average_score": 81.0,
            "retry_rate": 50.0,
            "improved_rate": 50.0,
            "recent_scores": [81.0],
            "issue_breakdown": {"needs review": 1},
            "prompt_versions": [
                {"prompt_version": "grounded-v1", "count": 1, "average_score": 81.0}
            ],
            "recent_runs": [
                {
                    "id": 1,
                    "question": "Q",
                    "provider": "fake",
                    "final_answer": "A",
                    "selected_attempt": "initial",
                    "retry_triggered": False,
                    "retry_reason": None,
                    "improvement_status": "unchanged",
                    "final_score": 81.0,
                    "faithfulness": 0.8,
                    "answer_relevancy": 0.8,
                    "context_precision": 0.8,
                    "prompt_version": "grounded-v1",
                    "created_at": "2026-04-26 00:00:00",
                }
            ],
        },
    )

    with TestClient(app) as client:
        response = client.get("/dashboard?limit=20")

    assert response.status_code == 200
    assert response.json()["total_runs"] == 1
