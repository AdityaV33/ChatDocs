from docx import Document
import io


def load_docx_text(file_bytes: bytes) -> str:
    """
    Extract plain text from a DOCX file.
    Returns a single string.
    """

    doc = Document(io.BytesIO(file_bytes))

    paragraphs = [
        p.text.strip()
        for p in doc.paragraphs
        if p.text and p.text.strip()
    ]

    return "\n\n".join(paragraphs)
