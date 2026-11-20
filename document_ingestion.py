"""Incremental document ingestion for the local project knowledge base."""

import hashlib
from typing import Any

from document_loader import extract_text
from project_agents import analyze_project_document
from rag_pipeline import SQLiteVectorStore, chunk_text


def document_content_hash(uploaded_file: Any) -> str:
    """Return a stable hash for an uploaded file's current contents."""
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


def ingest_uploaded_documents(
    uploaded_files: list[Any],
    store: SQLiteVectorStore,
    embedding_model: Any,
    *,
    force: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Index changed uploads and persist their evidence-based analyses."""
    result: dict[str, list[dict[str, Any]]] = {
        "processed": [],
        "skipped": [],
        "errors": [],
    }

    for uploaded_file in uploaded_files:
        source = uploaded_file.name
        try:
            content_hash = document_content_hash(uploaded_file)
            if not force and store.get_document_hash(source) == content_hash:
                result["skipped"].append({"source": source})
                continue

            extracted_text = extract_text(uploaded_file)
            chunks = chunk_text(extracted_text)
            if not chunks:
                raise ValueError("No searchable text chunks were produced.")

            embeddings = embedding_model.encode(
                chunks,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            analysis = analyze_project_document(extracted_text, source)
            store.replace_source(
                source,
                chunks,
                embeddings,
                content_hash=content_hash,
                analysis=analysis,
            )
            result["processed"].append(
                {
                    "source": source,
                    "chunk_count": len(chunks),
                    "analysis": analysis,
                }
            )
        except Exception as error:
            result["errors"].append({"source": source, "error": str(error)})

    return result
