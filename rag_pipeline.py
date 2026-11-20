"""Chunking, embedding configuration, and a persistent local vector index."""

from pathlib import Path
from contextlib import contextmanager
import json
import re
import sqlite3
from typing import Any, Iterator

import numpy as np


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split cleaned document text into overlapping word-based chunks."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    normalized_text = re.sub(r"\s+", " ", text).strip()
    words = normalized_text.split()
    if not words:
        return []

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def answer_project_question(
    query: str,
    results: list[dict[str, Any]],
    min_similarity: float = 0.25,
) -> dict[str, Any]:
    """Return an extractive answer with cited document evidence."""
    if not query.strip():
        raise ValueError("A question is required.")
    if not 0 <= min_similarity <= 1:
        raise ValueError("min_similarity must be between zero and one.")

    evidence = [
        {
            "source": result["source"],
            "chunk_index": result["chunk_index"],
            "text": result["text"],
            "similarity": result["similarity"],
        }
        for result in results
        if result["similarity"] >= min_similarity
    ][:3]
    if not evidence:
        return {
            "answer": (
                "I couldn't find sufficiently relevant evidence in the "
                "uploaded documents to answer that."
            ),
            "evidence": [],
        }

    return {
        "answer": (
            "The most relevant document passage says: "
            f"{evidence[0]['text']}"
        ),
        "evidence": evidence,
    }


class SQLiteVectorStore:
    """Persist chunk embeddings locally and retrieve them by cosine similarity."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS document_analyses (
                    source TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    analysis_json TEXT NOT NULL
                )
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def replace_source(
        self,
        source: str,
        chunks: list[str],
        embeddings: Any,
        *,
        content_hash: str | None = None,
        analysis: dict[str, Any] | None = None,
    ) -> None:
        if (content_hash is None) != (analysis is None):
            raise ValueError(
                "A content hash and document analysis must be provided together."
            )
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("There must be one embedding vector for each text chunk.")
        if vectors.shape[1] == 0:
            raise ValueError("Embedding vectors cannot be empty.")

        with self._connect() as connection:
            connection.execute("DELETE FROM chunks WHERE source = ?", (source,))
            connection.executemany(
                """
                INSERT INTO chunks (source, chunk_index, text, embedding)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (source, index, text, vector.tobytes())
                    for index, (text, vector) in enumerate(zip(chunks, vectors))
                ],
            )
            if content_hash is not None and analysis is not None:
                connection.execute(
                    """
                    INSERT INTO document_analyses (source, content_hash, analysis_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(source) DO UPDATE SET
                        content_hash = excluded.content_hash,
                        analysis_json = excluded.analysis_json
                    """,
                    (source, content_hash, json.dumps(analysis)),
                )

    def get_document_hash(self, source: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT content_hash FROM document_analyses WHERE source = ?",
                (source,),
            ).fetchone()
        return row[0] if row else None

    def list_document_analyses(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT analysis_json FROM document_analyses ORDER BY source"
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def search(self, query_embedding: Any, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be greater than zero.")

        query_vector = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            raise ValueError("Query embedding cannot be a zero vector.")
        query_vector = query_vector / query_norm

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT source, chunk_index, text, embedding FROM chunks"
            ).fetchall()

        scored_results = []
        for source, chunk_index, text, embedding_blob in rows:
            vector = np.frombuffer(embedding_blob, dtype=np.float32)
            if vector.shape != query_vector.shape:
                raise ValueError(
                    "The query embedding dimensions do not match the indexed model."
                )
            vector_norm = np.linalg.norm(vector)
            similarity = (
                float(np.dot(query_vector, vector / vector_norm))
                if vector_norm
                else 0.0
            )
            scored_results.append(
                {
                    "source": source,
                    "chunk_index": chunk_index,
                    "text": text,
                    "similarity": similarity,
                }
            )

        return sorted(
            scored_results,
            key=lambda result: result["similarity"],
            reverse=True,
        )[:top_k]

    def chunk_count(self) -> int:
        with self._connect() as connection:
            return connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def source_count(self) -> int:
        with self._connect() as connection:
            return connection.execute(
                "SELECT COUNT(DISTINCT source) FROM chunks"
            ).fetchone()[0]
