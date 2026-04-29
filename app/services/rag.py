from statistics import mean

from app.config import settings
from app.models.schemas import AnswerRun, DocumentChunk, RetrievalResult, RunAttempt
from app.services.evaluation import evaluate_answer, label_issues
from app.services.llm import get_llm
from app.services.vector_store import vector_store

DEFAULT_PROMPT_VERSION = "grounded-v1"
STRICT_PROMPT_VERSION = "grounded-strict-v2"

SYSTEM_PROMPT = """You answer questions using only the retrieved context.
If the context is insufficient, say that you do not have enough information from the documents.
Do not invent facts that are not supported by the context."""

STRICT_SYSTEM_PROMPT = """You are a strict retrieval-grounded assistant.
Use only the retrieved evidence.
If any part of the answer is missing from the evidence, say exactly that the documents do not contain enough information.
Prefer quoting the document's language closely instead of paraphrasing unsupported details."""


def build_context(chunks: list[DocumentChunk]) -> str:
    sections = []
    for chunk in chunks:
        sections.append(
            f"Source: {chunk.source} (chunk {chunk.chunk_index})\n{chunk.content}"
        )
    return "\n\n---\n\n".join(sections)


def _keyword_overlap(question: str, chunk: DocumentChunk) -> float:
    question_terms = {
        token.lower()
        for token in question.replace("?", "").replace(",", "").split()
        if len(token) > 2
    }
    chunk_terms = {
        token.lower()
        for token in chunk.content.replace("?", "").replace(",", "").split()
        if len(token) > 2
    }
    if not question_terms or not chunk_terms:
        return 0.0
    return len(question_terms & chunk_terms) / len(question_terms)


def rerank_chunks(question: str, chunks: list[DocumentChunk], top_k: int) -> list[DocumentChunk]:
    seen_sources: dict[str, int] = {}
    ranked = sorted(
        chunks,
        key=lambda chunk: (
            _keyword_overlap(question, chunk),
            -float(chunk.metadata.get("distance", 1.0)),
        ),
        reverse=True,
    )

    diverse: list[DocumentChunk] = []
    for chunk in ranked:
        source = chunk.source
        if seen_sources.get(source, 0) >= 2:
            continue
        diverse.append(chunk)
        seen_sources[source] = seen_sources.get(source, 0) + 1
        if len(diverse) >= top_k:
            break

    if len(diverse) < top_k:
        for chunk in ranked:
            if chunk in diverse:
                continue
            diverse.append(chunk)
            if len(diverse) >= top_k:
                break
    return diverse


def _generate_attempt(
    question: str,
    retrieval: RetrievalResult,
    prompt_version: str,
    system_prompt: str,
) -> RunAttempt:
    llm = get_llm()
    context = build_context(retrieval.chunks)
    if not retrieval.chunks:
        answer = "I do not have enough information from the uploaded documents to answer that."
    else:
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Write a concise grounded answer. If the answer is not in the context, say so plainly."
        )
        answer = llm.generate(system_prompt=system_prompt, user_prompt=user_prompt)

    metrics = evaluate_answer(question=question, answer=answer, retrieved_chunks=retrieval.chunks)
    issue_labels = label_issues(metrics)
    return RunAttempt(
        answer=answer,
        prompt_version=prompt_version,
        top_k=retrieval.top_k,
        retrieved_chunks=retrieval.chunks,
        metrics=metrics,
        issue_labels=issue_labels,
    )


def _retry_reason(attempt: RunAttempt) -> str | None:
    if attempt.metrics.total_score is not None and attempt.metrics.total_score < settings.RETRY_THRESHOLD:
        return f"score below retry threshold ({settings.RETRY_THRESHOLD})"
    if (
        attempt.metrics.faithfulness is not None
        and attempt.metrics.faithfulness < settings.HARD_FAITHFULNESS_FLOOR
    ):
        return f"faithfulness below hard floor ({settings.HARD_FAITHFULNESS_FLOOR})"
    return None


def _improvement_status(initial_score: float | None, corrected_score: float | None) -> str:
    if initial_score is None or corrected_score is None:
        return "unchanged"
    if corrected_score >= initial_score + settings.SCORE_IMPROVEMENT_EPSILON:
        return "improved"
    if corrected_score <= initial_score - settings.SCORE_IMPROVEMENT_EPSILON:
        return "worse"
    return "unchanged"


def answer_question(question: str, top_k: int | None = None) -> AnswerRun:
    effective_top_k = top_k or settings.TOP_K
    retrieval = vector_store.query(question=question, top_k=effective_top_k)
    initial_attempt = _generate_attempt(
        question=question,
        retrieval=retrieval,
        prompt_version=DEFAULT_PROMPT_VERSION,
        system_prompt=SYSTEM_PROMPT,
    )
    retry_reason = _retry_reason(initial_attempt)

    corrected_attempt: RunAttempt | None = None
    selected_attempt = "initial"
    final_answer = initial_attempt.answer
    improvement_status = "unchanged"

    if retry_reason:
        retry_top_k = min(
            max(effective_top_k + settings.RETRY_TOP_K_INCREMENT, effective_top_k + 1),
            settings.MAX_RETRIEVAL_DEPTH,
        )
        retry_retrieval = vector_store.query(question=question, top_k=retry_top_k)
        reranked_chunks = rerank_chunks(question, retry_retrieval.chunks, retry_top_k)
        corrected_attempt = _generate_attempt(
            question=question,
            retrieval=RetrievalResult(
                query=question,
                top_k=retry_top_k,
                chunks=reranked_chunks,
            ),
            prompt_version=STRICT_PROMPT_VERSION,
            system_prompt=STRICT_SYSTEM_PROMPT,
        )
        improvement_status = _improvement_status(
            initial_attempt.metrics.total_score,
            corrected_attempt.metrics.total_score,
        )
        if improvement_status == "improved":
            selected_attempt = "corrected"
            final_answer = corrected_attempt.answer

    provider_name = get_llm().provider_name
    return AnswerRun(
        question=question,
        provider=provider_name,
        retry_triggered=corrected_attempt is not None,
        retry_reason=retry_reason,
        improvement_status=improvement_status,
        selected_attempt=selected_attempt,
        final_answer=final_answer,
        initial_attempt=initial_attempt,
        corrected_attempt=corrected_attempt,
    )
