import tempfile
import unittest
from pathlib import Path

import numpy as np

from document_ingestion import document_content_hash, ingest_uploaded_documents
from rag_pipeline import SQLiteVectorStore


class UploadedFile:
    def __init__(self, name, content):
        self.name = name
        self._content = content

    def getvalue(self):
        return self._content


class FakeEmbeddingModel:
    def __init__(self):
        self.call_count = 0

    def encode(self, texts, **_kwargs):
        self.call_count += 1
        return np.tile([1.0, 0.0], (len(texts), 1))


class IncrementalDocumentIngestionTests(unittest.TestCase):
    def test_skips_unchanged_uploads_and_replaces_changed_source(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = SQLiteVectorStore(
                Path(temporary_directory) / "knowledge.sqlite3"
            )
            model = FakeEmbeddingModel()
            original = UploadedFile(
                "update.txt",
                (
                    b"Project goal: Deliver the analytics portal.\n"
                    b"Status: API dependency is delayed."
                ),
            )

            first_result = ingest_uploaded_documents([original], store, model)
            self.assertEqual(len(first_result["processed"]), 1)
            self.assertEqual(store.source_count(), 1)
            self.assertEqual(model.call_count, 1)

            unchanged_result = ingest_uploaded_documents(
                [original], store, model
            )
            self.assertEqual(unchanged_result["skipped"], [{"source": "update.txt"}])
            self.assertEqual(model.call_count, 1)

            updated = UploadedFile(
                "update.txt",
                (
                    b"Project goal: Deliver the analytics portal.\n"
                    b"Status: Launch is on track."
                ),
            )
            updated_result = ingest_uploaded_documents([updated], store, model)

            self.assertEqual(len(updated_result["processed"]), 1)
            self.assertEqual(store.source_count(), 1)
            self.assertEqual(model.call_count, 2)
            self.assertEqual(
                store.get_document_hash("update.txt"),
                document_content_hash(updated),
            )
            indexed_text = store.search([1.0, 0.0], top_k=1)[0]["text"]
            self.assertIn("Launch is on track", indexed_text)
            self.assertNotIn("API dependency is delayed", indexed_text)
            self.assertEqual(
                store.list_document_analyses()[0]["delivery_risk"]["risks"],
                [],
            )

            reopened_store = SQLiteVectorStore(store.database_path)
            self.assertEqual(len(reopened_store.list_document_analyses()), 1)
            self.assertEqual(
                reopened_store.get_document_hash("update.txt"),
                document_content_hash(updated),
            )

    def test_reports_unsupported_document_without_changing_index(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = SQLiteVectorStore(
                Path(temporary_directory) / "knowledge.sqlite3"
            )
            result = ingest_uploaded_documents(
                [UploadedFile("archive.zip", b"not a supported project file")],
                store,
                FakeEmbeddingModel(),
            )

            self.assertEqual(result["processed"], [])
            self.assertEqual(len(result["errors"]), 1)
            self.assertIn("Unsupported file type", result["errors"][0]["error"])
            self.assertEqual(store.chunk_count(), 0)
            self.assertEqual(store.list_document_analyses(), [])


if __name__ == "__main__":
    unittest.main()
