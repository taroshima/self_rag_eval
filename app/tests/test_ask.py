from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import DocumentChunk, MetricSet, RetrievalResult


def test_ask_returns_scored_run(monkeypatch):
    fake_chunks = [
        DocumentChunk(
            id="doc_0",
            content="Paris is the capital of France.",
            source="facts.txt",
            chunk_index=0,
            metadata={"source": "facts.txt", "chunk_index": 0},
        )
    ]

    class FakeProvider:
        @property
        def provider_name(self):
            return "fake"

        def generate(self, system_prompt: str, user_prompt: str) -> str:
            return "Paris is the capital of France."

    monkeypatch.setattr(
        "app.services.rag.vector_store.query",
        lambda question, top_k: RetrievalResult(query=question, top_k=top_k, chunks=fake_chunks),
    )
    monkeypatch.setattr("app.services.rag.get_llm", lambda: FakeProvider())
    monkeypatch.setattr(
        "app.services.rag.evaluate_answer",
        lambda question, answer, retrieved_chunks: MetricSet(
            faithfulness=0.95,
            answer_relevancy=0.92,
            context_precision=0.9,
            total_score=92.1,
            evaluator="heuristic",
            ragas_fallback_reason="Ragas import/setup failed: test stub",
            score_reliability="medium",
        ),
    )
    monkeypatch.setattr("app.services.rag.label_issues", lambda metrics: [])
    monkeypatch.setattr("app.api.routes.save_answer_run", lambda response: 123)

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "What is the capital of France?", "top_k": 2})

    assert response.status_code == 200
    payload = response.json()
    assert payload["final_answer"] == "Paris is the capital of France."
    assert payload["provider"] == "fake"
    assert payload["selected_attempt"] == "initial"
    assert payload["run_id"] == 123
    assert payload["initial_attempt"]["metrics"]["total_score"] == 92.1
    assert payload["initial_attempt"]["metrics"]["ragas_fallback_reason"] == "Ragas import/setup failed: test stub"


def test_ask_triggers_retry_when_score_is_low(monkeypatch):
    fake_chunks = [
        DocumentChunk(
            id="doc_1",
            content="Chunk text about onboarding and access requests.",
            source="playbook.md",
            chunk_index=0,
            metadata={"source": "playbook.md", "chunk_index": 0},
        )
    ]
    state = {"calls": 0}

    class FakeProvider:
        @property
        def provider_name(self):
            return "fake"

        def generate(self, system_prompt: str, user_prompt: str) -> str:
            state["calls"] += 1
            return "first answer" if state["calls"] == 1 else "second answer"

    metric_sequence = [
        MetricSet(
            faithfulness=0.4,
            answer_relevancy=0.4,
            context_precision=0.4,
            total_score=40.0,
            evaluator="heuristic",
            score_reliability="medium",
        ),
        MetricSet(
            faithfulness=0.8,
            answer_relevancy=0.75,
            context_precision=0.7,
            total_score=76.0,
            evaluator="heuristic",
            score_reliability="medium",
        ),
    ]

    monkeypatch.setattr(
        "app.services.rag.vector_store.query",
        lambda question, top_k: RetrievalResult(query=question, top_k=top_k, chunks=fake_chunks),
    )
    monkeypatch.setattr("app.services.rag.get_llm", lambda: FakeProvider())
    monkeypatch.setattr(
        "app.services.rag.evaluate_answer",
        lambda question, answer, retrieved_chunks: metric_sequence.pop(0),
    )
    monkeypatch.setattr(
        "app.services.rag.label_issues",
        lambda metrics: ["needs review"] if metrics.total_score and metrics.total_score < 50 else [],
    )
    monkeypatch.setattr("app.api.routes.save_answer_run", lambda response: 124)

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "How do I request access?", "top_k": 2})

    assert response.status_code == 200
    payload = response.json()
    assert payload["retry_triggered"] is True
    assert payload["selected_attempt"] == "corrected"
    assert payload["improvement_status"] == "improved"
    assert payload["corrected_attempt"]["answer"] == "second answer"
