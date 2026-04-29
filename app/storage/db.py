import json
import sqlite3
from contextlib import closing

from app.config import settings
from app.models.schemas import (
    AnswerRun,
    DashboardSummary,
    FailureRecord,
    MetricSet,
    PromptVersionSummary,
    RunAttempt,
    RunSummary,
)

REQUIRED_COLUMNS: dict[str, str] = {
    "question": "TEXT",
    "provider": "TEXT",
    "answer": "TEXT",
    "prompt_version": "TEXT",
    "total_score": "REAL",
    "issue_labels": "TEXT",
    "retrieved_chunks": "TEXT",
    "retry_triggered": "INTEGER DEFAULT 0",
    "retry_reason": "TEXT",
    "improvement_status": "TEXT",
    "selected_attempt": "TEXT",
    "final_answer": "TEXT",
    "final_score": "REAL",
    "initial_answer": "TEXT",
    "initial_prompt_version": "TEXT",
    "initial_top_k": "INTEGER",
    "initial_metrics": "TEXT",
    "initial_issue_labels": "TEXT",
    "initial_chunks": "TEXT",
    "corrected_answer": "TEXT",
    "corrected_prompt_version": "TEXT",
    "corrected_top_k": "INTEGER",
    "corrected_metrics": "TEXT",
    "corrected_issue_labels": "TEXT",
    "corrected_chunks": "TEXT",
    "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
}


def initialize_database() -> sqlite3.Connection:
    settings.ensure_directories()
    connection = sqlite3.connect(settings.SQLITE_PATH)
    with closing(connection.cursor()) as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS answer_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                provider TEXT NOT NULL,
                answer TEXT,
                prompt_version TEXT,
                total_score REAL,
                issue_labels TEXT,
                retrieved_chunks TEXT,
                retry_triggered INTEGER NOT NULL,
                retry_reason TEXT,
                improvement_status TEXT NOT NULL,
                selected_attempt TEXT NOT NULL,
                final_answer TEXT NOT NULL,
                final_score REAL,
                initial_answer TEXT NOT NULL,
                initial_prompt_version TEXT NOT NULL,
                initial_top_k INTEGER NOT NULL,
                initial_metrics TEXT NOT NULL,
                initial_issue_labels TEXT NOT NULL,
                initial_chunks TEXT NOT NULL,
                corrected_answer TEXT,
                corrected_prompt_version TEXT,
                corrected_top_k INTEGER,
                corrected_metrics TEXT,
                corrected_issue_labels TEXT,
                corrected_chunks TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute("PRAGMA table_info(answer_runs)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        for column_name, column_type in REQUIRED_COLUMNS.items():
            if column_name not in existing_columns:
                cursor.execute(
                    f"ALTER TABLE answer_runs ADD COLUMN {column_name} {column_type}"
                )
    connection.commit()
    return connection


def check_database() -> bool:
    connection = initialize_database()
    try:
        with closing(connection.cursor()) as cursor:
            cursor.execute("SELECT 1")
            return cursor.fetchone() == (1,)
    finally:
        connection.close()


def _serialize_attempt(attempt: RunAttempt | None) -> tuple:
    if attempt is None:
        return (None, None, None, None, None, None)
    return (
        attempt.answer,
        attempt.prompt_version,
        attempt.top_k,
        json.dumps(attempt.metrics.model_dump(mode="json")),
        json.dumps(attempt.issue_labels),
        json.dumps([chunk.model_dump(mode="json") for chunk in attempt.retrieved_chunks]),
    )


def save_answer_run(run: AnswerRun) -> int:
    connection = initialize_database()
    try:
        initial_payload = _serialize_attempt(run.initial_attempt)
        corrected_payload = _serialize_attempt(run.corrected_attempt)
        selected_attempt = run.corrected_attempt if run.selected_attempt == "corrected" and run.corrected_attempt else run.initial_attempt
        selected_metrics = selected_attempt.metrics
        serialized_selected_chunks = json.dumps(
            [chunk.model_dump(mode="json") for chunk in selected_attempt.retrieved_chunks]
        )
        with closing(connection.cursor()) as cursor:
            cursor.execute(
                """
                INSERT INTO answer_runs (
                    question,
                    provider,
                    answer,
                    prompt_version,
                    total_score,
                    issue_labels,
                    retrieved_chunks,
                    retry_triggered,
                    retry_reason,
                    improvement_status,
                    selected_attempt,
                    final_answer,
                    final_score,
                    initial_answer,
                    initial_prompt_version,
                    initial_top_k,
                    initial_metrics,
                    initial_issue_labels,
                    initial_chunks,
                    corrected_answer,
                    corrected_prompt_version,
                    corrected_top_k,
                    corrected_metrics,
                    corrected_issue_labels,
                    corrected_chunks
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.question,
                    run.provider,
                    run.final_answer,
                    selected_attempt.prompt_version,
                    selected_metrics.total_score,
                    json.dumps(selected_attempt.issue_labels),
                    serialized_selected_chunks,
                    int(run.retry_triggered),
                    run.retry_reason,
                    run.improvement_status,
                    run.selected_attempt,
                    run.final_answer,
                    selected_metrics.total_score,
                    initial_payload[0],
                    initial_payload[1],
                    initial_payload[2],
                    initial_payload[3],
                    initial_payload[4],
                    initial_payload[5],
                    corrected_payload[0],
                    corrected_payload[1],
                    corrected_payload[2],
                    corrected_payload[3],
                    corrected_payload[4],
                    corrected_payload[5],
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)
    finally:
        connection.close()


def _deserialize_metrics(value: str | None) -> MetricSet:
    if not value:
        return MetricSet()
    return MetricSet(**json.loads(value))


def _row_to_run_summary(row: sqlite3.Row) -> RunSummary:
    selected_metrics = (
        _deserialize_metrics(row["corrected_metrics"])
        if row["selected_attempt"] == "corrected" and row["corrected_metrics"]
        else _deserialize_metrics(row["initial_metrics"])
    )
    prompt_version = (
        row["corrected_prompt_version"]
        if row["selected_attempt"] == "corrected" and row["corrected_prompt_version"]
        else row["initial_prompt_version"]
    )
    return RunSummary(
        id=row["id"],
        question=row["question"],
        provider=row["provider"],
        final_answer=row["final_answer"],
        selected_attempt=row["selected_attempt"],
        retry_triggered=bool(row["retry_triggered"]),
        retry_reason=row["retry_reason"],
        improvement_status=row["improvement_status"],
        final_score=selected_metrics.total_score,
        faithfulness=selected_metrics.faithfulness,
        answer_relevancy=selected_metrics.answer_relevancy,
        context_precision=selected_metrics.context_precision,
        prompt_version=prompt_version,
        created_at=row["created_at"],
    )


def get_recent_runs(limit: int = 10) -> list[RunSummary]:
    connection = initialize_database()
    try:
        connection.row_factory = sqlite3.Row
        with closing(connection.cursor()) as cursor:
            cursor.execute("SELECT * FROM answer_runs ORDER BY id DESC LIMIT ?", (limit,))
            return [_row_to_run_summary(row) for row in cursor.fetchall()]
    finally:
        connection.close()


def get_failures(limit: int = 20) -> list[FailureRecord]:
    connection = initialize_database()
    try:
        connection.row_factory = sqlite3.Row
        with closing(connection.cursor()) as cursor:
            cursor.execute(
                """
                SELECT * FROM answer_runs
                WHERE final_score IS NULL OR final_score < ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (settings.FAILURE_THRESHOLD, limit),
            )
            failures: list[FailureRecord] = []
            for row in cursor.fetchall():
                summary = _row_to_run_summary(row)
                selected_issue_labels = (
                    row["corrected_issue_labels"]
                    if row["selected_attempt"] == "corrected" and row["corrected_issue_labels"]
                    else row["initial_issue_labels"]
                )
                labels = json.loads(selected_issue_labels or "[]")
                failures.append(FailureRecord(**summary.model_dump(), issue_labels=labels))
            return failures
    finally:
        connection.close()


def get_dashboard_summary(limit: int = 50) -> DashboardSummary:
    recent_runs = get_recent_runs(limit=limit)
    total_runs = len(recent_runs)
    if total_runs == 0:
        return DashboardSummary(total_runs=0, average_score=0.0, retry_rate=0.0, improved_rate=0.0)

    scores = [run.final_score for run in recent_runs if run.final_score is not None]
    retry_count = sum(1 for run in recent_runs if run.retry_triggered)
    improved_count = sum(1 for run in recent_runs if run.improvement_status == "improved")
    issue_breakdown: dict[str, int] = {}

    connection = initialize_database()
    try:
        connection.row_factory = sqlite3.Row
        with closing(connection.cursor()) as cursor:
            cursor.execute("SELECT * FROM answer_runs ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()

        prompt_map: dict[str, list[float]] = {}
        for row in rows:
            selected_issue_labels = (
                row["corrected_issue_labels"]
                if row["selected_attempt"] == "corrected" and row["corrected_issue_labels"]
                else row["initial_issue_labels"]
            )
            labels = json.loads(selected_issue_labels or "[]")
            for label in labels:
                issue_breakdown[label] = issue_breakdown.get(label, 0) + 1

            initial_metrics = _deserialize_metrics(row["initial_metrics"])
            prompt_map.setdefault(row["initial_prompt_version"], [])
            if initial_metrics.total_score is not None:
                prompt_map[row["initial_prompt_version"]].append(initial_metrics.total_score)

            if row["corrected_prompt_version"] and row["corrected_metrics"]:
                corrected_metrics = _deserialize_metrics(row["corrected_metrics"])
                prompt_map.setdefault(row["corrected_prompt_version"], [])
                if corrected_metrics.total_score is not None:
                    prompt_map[row["corrected_prompt_version"]].append(corrected_metrics.total_score)

        prompt_versions = [
            PromptVersionSummary(
                prompt_version=version,
                count=len(values),
                average_score=round(sum(values) / len(values), 2) if values else 0.0,
            )
            for version, values in prompt_map.items()
        ]
    finally:
        connection.close()

    return DashboardSummary(
        total_runs=total_runs,
        average_score=round(sum(scores) / len(scores), 2) if scores else 0.0,
        retry_rate=round((retry_count / total_runs) * 100.0, 2),
        improved_rate=round((improved_count / total_runs) * 100.0, 2),
        recent_scores=list(reversed(scores[-10:])),
        issue_breakdown=issue_breakdown,
        prompt_versions=sorted(prompt_versions, key=lambda item: item.prompt_version),
        recent_runs=recent_runs[:10],
    )
