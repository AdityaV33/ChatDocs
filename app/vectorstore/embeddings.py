from openai import OpenAI
from typing import List

client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"
MAX_CHUNKS = 25  


def embed_chunks(chunks: List[dict]) -> List[List[float]]:
    """
    Takes chunk dicts and returns embeddings.
    Embeds at most MAX_CHUNKS chunks.
    """

    texts = [chunk["chunk_text"] for chunk in chunks[:MAX_CHUNKS]]

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )

    return [item.embedding for item in response.data]
