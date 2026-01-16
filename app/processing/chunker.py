from typing import List, Dict

def chunk_text(
    text: str,
    chunk_size: int = 900,
    overlap: int = 100
) -> List[str]:
    """
    Splits text into overlapping chunks.
    """

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

        if start < 0:
            start = 0

    return chunks
