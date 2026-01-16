import faiss
import numpy as np
from typing import List


class FaissVectorStore:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.metadata = []

    def reset(self):
        """
        Reset the FAISS index and metadata (single-document mode).
        """
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata = []

    def add_embeddings(self, embeddings: List[List[float]], metadatas: List[dict]):
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.metadata.extend(metadatas)

    def similarity_search(self, query_embedding: List[float], top_k: int = 3):
        query_vector = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])

        return results
