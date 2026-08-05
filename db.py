"""
db.py — Küçük bir SQLite katmanı.

Doküman parçalarını (chunk) ve bunların embedding vektörlerini
tek bir .db dosyasında saklar. Embedding vektörü JSON string olarak
TEXT kolonunda tutulur.
"""

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "knowledge_base.db"


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def clear_chunks(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM chunks")
    conn.commit()


def insert_chunk(conn: sqlite3.Connection, source: str, content: str, embedding: list[float]) -> None:
    conn.execute(
        "INSERT INTO chunks (source, content, embedding) VALUES (?, ?, ?)",
        (source, content, json.dumps(embedding)),
    )
    conn.commit()


def count_chunks(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT COUNT(*) FROM chunks")
    return cur.fetchone()[0]


def fetch_all_chunks(conn: sqlite3.Connection):
    """Returns a list of dicts: {id, source, content, embedding (list[float])}."""
    cur = conn.execute("SELECT id, source, content, embedding FROM chunks")
    rows = cur.fetchall()
    return [
        {
            "id": row[0],
            "source": row[1],
            "content": row[2],
            "embedding": json.loads(row[3]),
        }
        for row in rows
    ]
