import math
import re
from difflib import SequenceMatcher
from statistics import mean

from app.config import settings
from app.models.schemas import DocumentChunk, MetricSet


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}
INSUFFICIENT_INFO_PHRASES = (
    "do not have enough information",
    "don't have enough information",
    "not enough information",
    "not in the provided context",
    "not provided in the documents",
)


def _normalize_tokens(text: str) -> list[str]:
    normalized: list[str] = []
    for token in TOKEN_PATTERN.findall(text.lower()):
        if token in STOP_WORDS:
            continue
        if len(token) == 1 and token not in {"q"}:
            continue
        normalized.append(token)
    return normalized


def _singularize(token: str) -> str:
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _token_variants(token: str) -> set[str]:
    variants = {token, _singularize(token)}
    if token == "q":
        variants.add("qu")
    if token == "qu":
        variants.add("q")
    return {variant for variant in variants if variant}


def _soft_token_match(token: str, candidates: set[str]) -> bool:
    for variant in _token_variants(token):
        if variant in candidates:
            return True
        for candidate in candidates:
            if len(variant) <= 1 or len(candidate) <= 1:
                continue
            if SequenceMatcher(None, variant, candidate).ratio() >= 0.8:
                return True
    return False


def _soft_overlap_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    matches = sum(1 for token in left if _soft_token_match(token, right))
    return matches / len(left)


def _contains_insufficient_info_answer(answer: str) -> bool:
    lowered = answer.lower()
    return any(phrase in lowered for phrase in INSUFFICIENT_INFO_PHRASES)


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[.!?]\s+", text) if part.strip()]


def _question_is_entity_style(question: str, question_tokens: set[str]) -> bool:
    lowered = question.lower().strip()
    prefixes = ("who is", "who are", "what is", "what are", "tell me about")
    return len(question_tokens) <= 4 or any(lowered.startswith(prefix) for prefix in prefixes)


def _extract_focus_terms(question: str) -> set[str]:
    lowered = question.lower().strip()
    for prefix in ("who is", "who are", "what is", "what are", "tell me about"):
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix):].strip()
            break
    return set(_normalize_tokens(lowered))


def _sentence_support_score(answer: str, retrieved_chunks: list[DocumentChunk]) -> float:
    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0

    chunk_token_sets = [set(_normalize_tokens(chunk.content)) for chunk in retrieved_chunks]
    per_sentence_scores: list[float] = []
    for sentence in sentences:
        sentence_tokens = set(_normalize_tokens(sentence))
        if not sentence_tokens:
            continue
        best_support = 0.0
        for chunk_tokens in chunk_token_sets:
            best_support = max(best_support, _soft_overlap_score(sentence_tokens, chunk_tokens))
        per_sentence_scores.append(best_support)
    return mean(per_sentence_scores) if per_sentence_scores else 0.0


def _reliability_label(question_tokens: set[str], evaluator: str, ragas_reason: str | None) -> str:
    if evaluator == "ragas":
        return "high"
    if len(question_tokens) <= 2 or ragas_reason:
        return "low"
    if len(question_tokens) <= 4:
        return "medium"
    return "medium"


def _heuristic_metrics(
    question: str,
    answer: str,
    retrieved_chunks: list[DocumentChunk],
    ragas_fallback_reason: str | None = None,
) -> MetricSet:
    context = "\n".join(chunk.content for chunk in retrieved_chunks)
    question_tokens = set(_normalize_tokens(question))
    answer_tokens = set(_normalize_tokens(answer))
    context_tokens = set(_normalize_tokens(context))
    focus_terms = _extract_focus_terms(question)

    insufficient_answer = _contains_insufficient_info_answer(answer)
    no_context = not context_tokens

    if insufficient_answer and no_context:
        faithfulness = 1.0
        answer_relevancy = 0.75
        context_precision = 1.0
    else:
        token_support = _soft_overlap_score(answer_tokens, context_tokens)
        sentence_support = _sentence_support_score(answer, retrieved_chunks)
        faithfulness = max((token_support * 0.55) + (sentence_support * 0.45), sentence_support)
        if insufficient_answer and context_tokens:
            faithfulness = max(faithfulness, 0.85)

        answer_relevancy = _soft_overlap_score(question_tokens, answer_tokens)
        if _question_is_entity_style(question, question_tokens):
            focus_overlap_answer = _soft_overlap_score(focus_terms, answer_tokens) if focus_terms else answer_relevancy
            focus_overlap_context = _soft_overlap_score(focus_terms, context_tokens) if focus_terms else answer_relevancy
            definitional_cues = (
                " are " in f" {answer.lower()} "
                or " is " in f" {answer.lower()} "
                or "species" in answer.lower()
                or "civilization" in answer.lower()
                or "race" in answer.lower()
            )
            answer_relevancy = max(
                answer_relevancy,
                (focus_overlap_answer * 0.55)
                + (focus_overlap_context * 0.3)
                + (0.15 if definitional_cues else 0.0),
            )
        if insufficient_answer and question_tokens:
            answer_relevancy = max(answer_relevancy, 0.55)

        per_chunk_precision: list[float] = []
        for chunk in retrieved_chunks:
            chunk_tokens = set(_normalize_tokens(chunk.content))
            question_overlap = _soft_overlap_score(question_tokens, chunk_tokens)
            answer_overlap = _soft_overlap_score(answer_tokens, chunk_tokens)
            focus_overlap = _soft_overlap_score(focus_terms, chunk_tokens) if focus_terms else question_overlap
            per_chunk_precision.append(
                (question_overlap * 0.25) + (answer_overlap * 0.4) + (focus_overlap * 0.35)
            )
        if per_chunk_precision:
            ranked_scores = sorted(per_chunk_precision, reverse=True)
            top_score = ranked_scores[0]
            support_tail = mean(ranked_scores[1:3]) if len(ranked_scores) > 1 else 0.0
            context_precision = min(1.0, (top_score * 0.7) + (support_tail * 0.3))
        else:
            context_precision = 0.0

    total_score = (
        faithfulness * settings.METRIC_WEIGHT_FAITHFULNESS
        + answer_relevancy * settings.METRIC_WEIGHT_RELEVANCY
        + context_precision * settings.METRIC_WEIGHT_CONTEXT_PRECISION
    ) * 100.0

    return MetricSet(
        faithfulness=round(faithfulness, 3),
        answer_relevancy=round(answer_relevancy, 3),
        context_precision=round(context_precision, 3),
        total_score=round(total_score, 2),
        evaluator="heuristic",
        ragas_fallback_reason=ragas_fallback_reason,
        score_reliability=_reliability_label(question_tokens, "heuristic", ragas_fallback_reason),
    )


def _safe_ragas_metrics(
    question: str,
    answer: str,
    retrieved_chunks: list[DocumentChunk],
) -> tuple[MetricSet | None, str | None]:
    if not settings.RAGAS_ENABLED:
        return None, "RAGAS_ENABLED is false"

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except Exception as exc:
        return None, f"Ragas import/setup failed: {exc}"

    contexts = [chunk.content for chunk in retrieved_chunks]
    if not contexts:
        return None, "No retrieved context available for Ragas evaluation"

    try:
        dataset = Dataset.from_dict(
            {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [contexts[0]],
            }
        )
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )
        frame = result.to_pandas()
        faithfulness_value = float(frame["faithfulness"][0])
        relevancy_value = float(frame["answer_relevancy"][0])
        context_precision_value = float(frame["context_precision"][0])
        total_score = (
            faithfulness_value * settings.METRIC_WEIGHT_FAITHFULNESS
            + relevancy_value * settings.METRIC_WEIGHT_RELEVANCY
            + context_precision_value * settings.METRIC_WEIGHT_CONTEXT_PRECISION
        ) * 100.0
        if math.isnan(total_score):
            return None, "Ragas returned NaN total score"
        return (
            MetricSet(
                faithfulness=round(faithfulness_value, 3),
                answer_relevancy=round(relevancy_value, 3),
                context_precision=round(context_precision_value, 3),
                total_score=round(total_score, 2),
                evaluator="ragas",
                score_reliability=_reliability_label(set(_normalize_tokens(question)), "ragas", None),
            ),
            None,
        )
    except Exception as exc:
        return None, f"Ragas metric execution failed: {exc}"


def evaluate_answer(
    question: str,
    answer: str,
    retrieved_chunks: list[DocumentChunk],
) -> MetricSet:
    ragas_metrics, fallback_reason = _safe_ragas_metrics(question, answer, retrieved_chunks)
    if ragas_metrics is not None:
        return ragas_metrics
    return _heuristic_metrics(
        question,
        answer,
        retrieved_chunks,
        ragas_fallback_reason=fallback_reason,
    )


def label_issues(metrics: MetricSet) -> list[str]:
    issues: list[str] = []
    if metrics.faithfulness is not None and metrics.faithfulness < 0.65:
        issues.append("possible hallucination")
    if metrics.answer_relevancy is not None and metrics.answer_relevancy < 0.55:
        issues.append("weak answer fit")
    if metrics.context_precision is not None and metrics.context_precision < 0.5:
        issues.append("noisy retrieval")
    if metrics.total_score is not None and metrics.total_score < settings.FAILURE_THRESHOLD:
        issues.append("needs review")
    return issues
