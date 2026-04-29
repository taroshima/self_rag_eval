import sys
from pathlib import Path

import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings

API_BASE_URL = "http://127.0.0.1:8000"


def _format_api_error(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or f"HTTP {response.status_code}"

    if isinstance(payload, dict) and payload.get("detail"):
        return str(payload["detail"])
    return str(payload)


def api_get(path: str, **kwargs):
    response = requests.get(f"{API_BASE_URL}{path}", timeout=20, **kwargs)
    if not response.ok:
        raise RuntimeError(_format_api_error(response))
    return response.json()


def api_post(path: str, **kwargs):
    timeout = kwargs.pop("timeout", 120)
    response = requests.post(f"{API_BASE_URL}{path}", timeout=timeout, **kwargs)
    if not response.ok:
        raise RuntimeError(_format_api_error(response))
    return response.json()


def render_health() -> None:
    st.sidebar.header("Backend")
    try:
        payload = api_get("/health")
        if payload["status"] == "ok":
            st.sidebar.success("Backend ready")
        else:
            st.sidebar.warning("Backend partially ready")
        st.sidebar.write(f"LLM: {payload['provider']}")
        st.sidebar.write(f"Embeddings: {payload['embedding_provider']}")
        st.sidebar.write(f"Chunks stored: {payload['document_count']}")
    except Exception as exc:
        st.sidebar.error(f"API unavailable: {exc}")


def render_ingest_view() -> None:
    st.subheader("Ingest")
    st.write("Upload files to add them to the retrieval index.")

    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("Ingest Uploaded Files", disabled=not uploaded_files):
        with st.spinner("Ingesting files. Large PDFs can take several minutes while embeddings are created."):
            for uploaded_file in uploaded_files or []:
                try:
                    payload = api_post(
                        "/ingest",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type or "application/octet-stream",
                            )
                        },
                        timeout=settings.INGEST_TIMEOUT_SECONDS,
                    )
                    st.success(
                        f"{payload['filename']} ingested with {payload['chunk_count']} chunks."
                    )
                except Exception as exc:
                    st.error(f"{uploaded_file.name}: {exc}")


def _metrics_table(metrics: dict) -> dict[str, float]:
    return {
        "faithfulness": metrics.get("faithfulness", 0.0) or 0.0,
        "answer_relevancy": metrics.get("answer_relevancy", 0.0) or 0.0,
        "context_precision": metrics.get("context_precision", 0.0) or 0.0,
        "total_score": metrics.get("total_score", 0.0) or 0.0,
    }


def _render_attempt(title: str, attempt: dict | None) -> None:
    if not attempt:
        st.info(f"{title}: no retry attempt was needed.")
        return

    st.markdown(f"#### {title}")
    st.write(attempt["answer"])
    st.caption(
        f"Prompt: {attempt['prompt_version']} | Top K: {attempt['top_k']} | Evaluator: {attempt['metrics']['evaluator']} | Reliability: {attempt['metrics'].get('score_reliability', 'medium')}"
    )
    st.json(_metrics_table(attempt["metrics"]))
    if attempt["metrics"].get("ragas_fallback_reason"):
        st.info(f"Ragas fallback reason: {attempt['metrics']['ragas_fallback_reason']}")
    if attempt["issue_labels"]:
        st.warning(", ".join(attempt["issue_labels"]))
    else:
        st.success("No issue labels.")

    for chunk in attempt["retrieved_chunks"]:
        with st.expander(f"{chunk['source']} - chunk {chunk['chunk_index']}"):
            st.write(chunk["content"])


def render_ask_view() -> None:
    st.subheader("Ask")
    st.write("Ask a question against the currently ingested documents and inspect the scoring output.")

    question = st.text_area(
        "Question",
        placeholder="Ask about the documents you have ingested.",
        height=120,
    )
    top_k = st.slider("Initial retrieval depth", min_value=1, max_value=8, value=3)

    if st.button("Run RAG", disabled=not question.strip()):
        try:
            payload = api_post("/ask", json={"question": question, "top_k": top_k})
        except Exception as exc:
            st.error(str(exc))
            return

        final_score = (
            payload["corrected_attempt"]["metrics"]["total_score"]
            if payload["selected_attempt"] == "corrected" and payload["corrected_attempt"]
            else payload["initial_attempt"]["metrics"]["total_score"]
        )
        st.markdown("### Final Answer")
        st.write(payload["final_answer"])
        st.caption(
            f"Run {payload['run_id']} | Provider: {payload['provider']} | Selected attempt: {payload['selected_attempt']} | Improvement: {payload['improvement_status']} | Final score: {final_score}"
        )

        if payload["retry_triggered"]:
            st.info(f"Retry triggered because {payload['retry_reason']}.")
        else:
            st.success("Initial answer cleared the retry rules.")

        col1, col2 = st.columns(2)
        with col1:
            _render_attempt("Initial Attempt", payload["initial_attempt"])
        with col2:
            _render_attempt("Corrected Attempt", payload.get("corrected_attempt"))


def render_dashboard_view() -> None:
    st.subheader("Dashboard")
    try:
        summary = api_get("/dashboard?limit=50")
        failures = api_get("/failures?limit=10")
    except Exception as exc:
        st.error(str(exc))
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric("Total runs", summary["total_runs"])
    metric_cols[1].metric("Average score", summary["average_score"])
    metric_cols[2].metric("Retry rate", f"{summary['retry_rate']}%")
    metric_cols[3].metric("Improved rate", f"{summary['improved_rate']}%")

    if summary["recent_scores"]:
        st.markdown("### Score Trend")
        st.line_chart(summary["recent_scores"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Prompt Versions")
        if summary["prompt_versions"]:
            st.bar_chart(
                {
                    item["prompt_version"]: item["average_score"]
                    for item in summary["prompt_versions"]
                }
            )
        else:
            st.info("No prompt version data yet.")

    with col2:
        st.markdown("### Issue Breakdown")
        if summary["issue_breakdown"]:
            st.bar_chart(summary["issue_breakdown"])
        else:
            st.info("No issue labels recorded yet.")

    st.markdown("### Recent Runs")
    if summary["recent_runs"]:
        for run in summary["recent_runs"]:
            with st.expander(f"Run {run['id']} - {run['question']}"):
                st.write(run["final_answer"])
                st.caption(
                    f"Score: {run['final_score']} | Prompt: {run['prompt_version']} | Retry: {run['retry_triggered']} | Improvement: {run['improvement_status']}"
                )
    else:
        st.info("No runs yet.")

    st.markdown("### Failure Cases")
    if failures:
        for failure in failures:
            with st.expander(f"Failure {failure['id']} - {failure['question']}"):
                st.write(failure["final_answer"])
                st.caption(
                    f"Score: {failure['final_score']} | Issues: {', '.join(failure['issue_labels']) if failure['issue_labels'] else 'None'}"
                )
    else:
        st.success("No failure cases under the current threshold.")


def main() -> None:
    st.set_page_config(page_title="Self-Correcting RAG Evaluator", layout="wide")
    st.title("Self-Correcting RAG Evaluator")
    st.write(
        "A RAG app with ingestion, grounded Q&A, evaluation, one-step self-correction, and a dashboard."
    )

    render_health()

    ingest_tab, ask_tab, dashboard_tab = st.tabs(["Ingest", "Ask", "Dashboard"])
    with ingest_tab:
        render_ingest_view()
    with ask_tab:
        render_ask_view()
    with dashboard_tab:
        render_dashboard_view()


if __name__ == "__main__":
    main()
