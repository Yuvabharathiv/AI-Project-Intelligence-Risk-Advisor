import tempfile
import unittest
from pathlib import Path

import numpy as np

from rag_pipeline import SQLiteVectorStore, answer_project_question, chunk_text


class ChunkTextTests(unittest.TestCase):
    def test_chunks_words_with_overlap(self):
        chunks = chunk_text("one two three four five", chunk_size=3, overlap=1)

        self.assertEqual(chunks, ["one two three", "three four five"])

    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(chunk_text(" \n\t "), [])

    def test_rejects_invalid_chunk_parameters(self):
        with self.assertRaises(ValueError):
            chunk_text("text", chunk_size=0)
        with self.assertRaises(ValueError):
            chunk_text("text", chunk_size=3, overlap=3)


class SQLiteVectorStoreTests(unittest.TestCase):
    def test_persists_and_retrieves_by_cosine_similarity(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            store_path = Path(temporary_directory) / "index.sqlite3"
            store = SQLiteVectorStore(store_path)
            store.replace_source(
                "plan.txt",
                ["schedule details", "budget details"],
                np.asarray([[1, 0], [0, 1]], dtype=np.float32),
            )

            reopened_store = SQLiteVectorStore(store_path)
            results = reopened_store.search([0.9, 0.1], top_k=1)

            self.assertEqual(reopened_store.chunk_count(), 2)
            self.assertEqual(reopened_store.source_count(), 1)
            self.assertEqual(results[0]["text"], "schedule details")
            self.assertGreater(results[0]["similarity"], 0.9)

    def test_replaces_chunks_from_same_source(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = SQLiteVectorStore(Path(temporary_directory) / "index.sqlite3")
            store.replace_source("plan.txt", ["old"], [[1, 0]])
            store.replace_source("plan.txt", ["new"], [[0, 1]])

            self.assertEqual(store.chunk_count(), 1)
            self.assertEqual(store.search([0, 1])[0]["text"], "new")

    def test_project_answer_quotes_retrieved_document_evidence(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = SQLiteVectorStore(Path(temporary_directory) / "index.sqlite3")
            store.replace_source(
                "status.txt",
                [
                    "The API dependency is delayed and the launch is at risk.",
                    "The team approved the dashboard design.",
                ],
                np.asarray([[1, 0], [0, 1]], dtype=np.float32),
            )

            results = store.search([1, 0], top_k=2)
            response = answer_project_question("Are we on track?", results)

            self.assertEqual(len(response["evidence"]), 1)
            self.assertEqual(response["evidence"][0]["source"], "status.txt")
            self.assertEqual(
                response["evidence"][0]["text"],
                "The API dependency is delayed and the launch is at risk.",
            )
            self.assertIn(
                response["evidence"][0]["text"],
                response["answer"],
            )

    def test_project_answer_declines_when_no_relevant_evidence_exists(self):
        response = answer_project_question(
            "What is the budget?",
            [
                {
                    "source": "status.txt",
                    "chunk_index": 0,
                    "text": "The project is blocked.",
                    "similarity": 0.1,
                }
            ],
        )

        self.assertEqual(response["evidence"], [])
        self.assertIn("couldn't find", response["answer"])

    def test_project_answer_rejects_empty_question(self):
        with self.assertRaises(ValueError):
            answer_project_question("  ", [])


if __name__ == "__main__":
    unittest.main()
