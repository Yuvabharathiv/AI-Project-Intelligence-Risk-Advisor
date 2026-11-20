import csv
from io import BytesIO, StringIO
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pymupdf
from docx import Document

from document_loader import extract_text
from document_ingestion import ingest_uploaded_documents
from project_agents import (
    analyze_project_document,
    calculate_project_health,
    generate_project_documentation,
)
from rag_pipeline import SQLiteVectorStore, answer_project_question


SAMPLE_PROJECT_TEXT = """\
Project goal: Deliver the customer portal MVP.
Milestone: Beta launch by 2026-11-20.
Deliverable: User registration and project dashboard.
Action item: Owner: Alice, follow up with the API team by 2026-11-10.
Status: At risk because the API dependency is delayed.
Pending decision: Select the production hosting provider.
"""


class UploadedFile:
    def __init__(self, name, content):
        self.name = name
        self._content = content

    def getvalue(self):
        return self._content


def make_sample_file(extension):
    if extension == ".txt":
        content = SAMPLE_PROJECT_TEXT.encode("utf-8")
    elif extension == ".csv":
        csv_content = StringIO()
        writer = csv.writer(csv_content)
        writer.writerow(("Section", "Details"))
        for index, line in enumerate(SAMPLE_PROJECT_TEXT.splitlines(), start=1):
            writer.writerow((f"Update {index}", line))
        content = csv_content.getvalue().encode("utf-8")
    elif extension == ".docx":
        document = Document()
        for line in SAMPLE_PROJECT_TEXT.splitlines():
            document.add_paragraph(line)
        output = BytesIO()
        document.save(output)
        content = output.getvalue()
    elif extension == ".pdf":
        document = pymupdf.open()
        page = document.new_page()
        page.insert_textbox(
            pymupdf.Rect(50, 50, 550, 750),
            SAMPLE_PROJECT_TEXT,
            fontsize=12,
        )
        content = document.tobytes()
        document.close()
    else:
        raise ValueError(f"Unsupported sample extension: {extension}")
    return UploadedFile(f"project_update{extension}", content)


class FakeEmbeddingModel:
    def encode(self, texts, **_kwargs):
        return np.tile([1.0, 0.0], (len(texts), 1))


class ProjectAgentFormatTests(unittest.TestCase):
    def test_agents_extract_grounded_findings_from_all_supported_formats(self):
        for extension in (".txt", ".csv", ".docx", ".pdf"):
            with self.subTest(extension=extension):
                uploaded_file = make_sample_file(extension)
                extracted_text = extract_text(uploaded_file)
                analysis = analyze_project_document(
                    extracted_text, uploaded_file.name
                )

                scope_text = [
                    finding["text"] for finding in analysis["scope"]["findings"]
                ]
                self.assertTrue(
                    any("customer portal MVP" in line for line in scope_text)
                )
                self.assertTrue(
                    any(
                        finding["timeline"] == "2026-11-20"
                        for finding in analysis["scope"]["findings"]
                    )
                )

                risk = analysis["delivery_risk"]
                self.assertEqual(risk["forecast"], "High delivery risk")
                self.assertTrue(
                    any("API dependency is delayed" in item["evidence"]
                        for item in risk["risks"])
                )

                items = analysis["blockers_and_actions"]
                self.assertTrue(items["pending_decisions"])
                self.assertTrue(items["actions"])
                self.assertEqual(items["actions"][0]["owner"], "Alice")
                self.assertEqual(items["actions"][0]["due_date"], "2026-11-10")

    def test_end_to_end_outputs_are_grounded_for_all_supported_formats(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = SQLiteVectorStore(
                Path(temporary_directory) / "knowledge.sqlite3"
            )
            uploads = [
                make_sample_file(extension)
                for extension in (".txt", ".csv", ".docx", ".pdf")
            ]

            result = ingest_uploaded_documents(
                uploads, store, FakeEmbeddingModel()
            )

            self.assertEqual(len(result["processed"]), 4)
            self.assertEqual(result["errors"], [])
            self.assertEqual(store.source_count(), 4)
            analyses = store.list_document_analyses()
            self.assertEqual(len(analyses), 4)

            documentation = generate_project_documentation(analyses)
            health = calculate_project_health(analyses)
            self.assertTrue(documentation["user_stories"])
            self.assertTrue(documentation["risk_register"])
            self.assertTrue(documentation["actions"])
            self.assertGreaterEqual(health["overall_score"], 0)
            self.assertLessEqual(health["overall_score"], 100)

            passages = store.search([1.0, 0.0], top_k=5)
            response = answer_project_question(
                "What is the current delivery risk?", passages
            )
            self.assertTrue(response["evidence"])
            self.assertIn("API dependency is delayed", response["answer"])
            self.assertTrue(
                all(
                    evidence["source"] in {upload.name for upload in uploads}
                    for evidence in response["evidence"]
                )
            )

    def test_no_matching_signals_do_not_invent_risks_or_actions(self):
        analysis = analyze_project_document(
            "The team discussed general project ideas.", "notes.txt"
        )

        self.assertEqual(
            analysis["delivery_risk"]["forecast"],
            "No explicit delivery risk found",
        )
        self.assertEqual(analysis["delivery_risk"]["risks"], [])
        self.assertEqual(analysis["blockers_and_actions"]["actions"], [])

    def test_topic_mentions_do_not_count_as_active_risks_or_blockers(self):
        analysis = analyze_project_document(
            "The project will identify risks, blockers, and pending decisions.\n"
            "Blockers\n"
            "Test case: The agent reports risks and blockers separately.\n"
            "A blocker should not be silently converted into a completed action.",
            "requirements.txt",
        )

        self.assertEqual(analysis["delivery_risk"]["risks"], [])
        self.assertEqual(analysis["blockers_and_actions"]["blockers"], [])

    def test_explicit_risk_and_blocker_statuses_are_still_detected(self):
        analysis = analyze_project_document(
            "Risk: API dependency may delay the beta launch.\n"
            "Blocker: Deployment is blocked by access approval.\n"
            "Pending decision: Select the hosting provider.",
            "status.txt",
        )

        self.assertEqual(
            [risk["severity"] for risk in analysis["delivery_risk"]["risks"]],
            ["medium", "high", "medium"],
        )
        self.assertEqual(
            [blocker["text"] for blocker in analysis["blockers_and_actions"]["blockers"]],
            ["Blocker: Deployment is blocked by access approval."],
        )

    def test_documentation_outputs_are_grounded_in_analysis_evidence(self):
        analysis = analyze_project_document(SAMPLE_PROJECT_TEXT, "plan.txt")

        documentation = generate_project_documentation([analysis])

        self.assertTrue(
            any(
                "customer portal MVP" in story["evidence"]
                and story["source"] == "plan.txt"
                for story in documentation["user_stories"]
            )
        )
        self.assertTrue(
            any(
                "API dependency is delayed" in risk["evidence"]
                and risk["severity"] == "high"
                for risk in documentation["risk_register"]
            )
        )
        self.assertTrue(
            any(action["owner"] == "Alice" for action in documentation["actions"])
        )

    def test_health_score_reports_each_dimension_using_explicit_signals(self):
        text = (
            "Project goal: Deliver the portal.\n"
            "Deliverable: Customer dashboard.\n"
            "Milestone: Beta launch by 2026-11-20.\n"
            "Status: API dependency is delayed.\n"
            "Blocker: Access is blocked."
        )

        health = calculate_project_health(
            [analyze_project_document(text, "plan.txt")]
        )

        self.assertEqual(health["dimensions"]["scope_clarity"]["score"], 100)
        self.assertEqual(health["dimensions"]["timeline_risk"]["score"], 40)
        self.assertEqual(health["dimensions"]["blockers"]["score"], 75)
        self.assertEqual(health["blocker_count"], 1)
        self.assertEqual(health["overall_score"], 72)

    def test_health_score_requires_document_analysis(self):
        with self.assertRaises(ValueError):
            calculate_project_health([])


if __name__ == "__main__":
    unittest.main()
