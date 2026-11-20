from io import BytesIO
import unittest

import pymupdf
from docx import Document

from document_loader import extract_text


class UploadedFile:
    def __init__(self, name, content):
        self.name = name
        self._content = content

    def getvalue(self):
        return self._content


class DocumentLoaderTests(unittest.TestCase):
    def test_extracts_txt(self):
        uploaded_file = UploadedFile("notes.txt", b"\xef\xbb\xbfProject update")

        self.assertEqual(extract_text(uploaded_file), "Project update")

    def test_extracts_csv_as_readable_table(self):
        uploaded_file = UploadedFile("tasks.csv", b"Task,Status\nDesign,Done")

        extracted_text = extract_text(uploaded_file)

        self.assertIn("Task", extracted_text)
        self.assertIn("Design", extracted_text)
        self.assertIn("Done", extracted_text)

    def test_extracts_docx_paragraphs_and_tables(self):
        document = Document()
        document.add_paragraph("Project scope")
        table = document.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "Deliverable"
        table.cell(0, 1).text = "Prototype"
        output = BytesIO()
        document.save(output)
        uploaded_file = UploadedFile("proposal.docx", output.getvalue())

        extracted_text = extract_text(uploaded_file)

        self.assertIn("Project scope", extracted_text)
        self.assertIn("Deliverable | Prototype", extracted_text)

    def test_extracts_pdf_pages(self):
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 72), "Project proposal")
        pdf_content = document.tobytes()
        document.close()
        uploaded_file = UploadedFile("proposal.pdf", pdf_content)

        extracted_text = extract_text(uploaded_file)

        self.assertIn("Page 1", extracted_text)
        self.assertIn("Project proposal", extracted_text)

    def test_rejects_unsupported_extension(self):
        with self.assertRaises(ValueError):
            extract_text(UploadedFile("archive.zip", b"content"))


if __name__ == "__main__":
    unittest.main()
