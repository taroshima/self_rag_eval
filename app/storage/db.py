import sqlite3
from contextlib import closing

from app.config import settings


def initialize_database() -> sqlite3.Connection:
    settings.ensure_directories()
    connection = sqlite3.connect(settings.SQLITE_PATH)
    with closing(connection.cursor()) as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS answer_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                provider TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                total_score REAL,
                issue_labels TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
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
