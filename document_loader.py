

from io import BytesIO, StringIO
from pathlib import Path

import pymupdf
import pandas as pd
from docx import Document


def extract_text(uploaded_file):
    """
    Extract text from PDF, DOCX, CSV, or TXT files.

    uploaded_file can be a Streamlit UploadedFile
    or a similar file-like object.
    """
    filename = uploaded_file.name
    extension = Path(filename).suffix.lower()
    file_bytes = uploaded_file.getvalue()

    if extension == ".pdf":
        text_parts = []

        with pymupdf.open(
            stream=file_bytes,
            filetype="pdf"
        ) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                page_text = page.get_text("text")
                if page_text.strip():
                    text_parts.append(
                        f"Page {page_number}\n{page_text}"
                    )

        extracted_text = "\n\n".join(text_parts)

    elif extension == ".docx":
        document = Document(BytesIO(file_bytes))

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        # Include text from tables as well.
        for table in document.tables:
            for row in table.rows:
                row_text = " | ".join(
                    cell.text.strip() for cell in row.cells
                )
                if row_text.strip():
                    paragraphs.append(row_text)

        extracted_text = "\n".join(paragraphs)

    elif extension == ".csv":
        try:
            df = pd.read_csv(BytesIO(file_bytes))
        except UnicodeDecodeError:
            df = pd.read_csv(
                BytesIO(file_bytes),
                encoding="latin-1"
            )

        extracted_text = df.to_string(index=False)

    elif extension == ".txt":
        extracted_text = file_bytes.decode(
            "utf-8-sig",
            errors="replace"
        )

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    if not extracted_text.strip():
        raise ValueError(
            "No readable text was found in this file."
        )

    return extracted_text
