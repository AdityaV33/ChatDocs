import numpy as np
from typing import List, Dict

class CosineRetriever:
    """
    Exact cosine similarity retriever for evaluation against FAISS.
    Accepts pre-computed embeddings and makes NO API calls.
    """
    def __init__(self, embedding_dim: int):
        if embedding_dim <= 0:
            raise ValueError("Embedding dimension must be positive.")
        self.embedding_dim = embedding_dim
        self.vectors = None
        self.metadata = []

    def reset(self):
        """Reset the retriever state."""
        self.vectors = None
        self.metadata = []

    def add_embeddings(self, embeddings: List[List[float]], metadatas: List[Dict]):
        """Add embedding vectors and their corresponding metadata."""
        if not embeddings:
            return
            
        new_vectors = np.array(embeddings, dtype=np.float32)
        
        if new_vectors.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected embedding dimension {self.embedding_dim}, got {new_vectors.shape[1]}")
            
        if self.vectors is None:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])
            
        self.metadata.extend(metadatas)

    def _cosine_similarities(self, query_vector: np.ndarray) -> np.ndarray:
        """
        Calculate raw cosine similarities against all stored vectors.
        Exposed for test verification of exact math.
        """
        if self.vectors is None or len(self.vectors) == 0:
            return np.array([], dtype=np.float32)
            
        if query_vector.shape[0] != self.embedding_dim:
            raise ValueError(f"Expected query dimension {self.embedding_dim}, got {query_vector.shape[0]}")
            
        # Dot products between query and all stored vectors
        dot_products = np.dot(self.vectors, query_vector)
        
        # L2 Norms
        query_norm = np.linalg.norm(query_vector)
        vector_norms = np.linalg.norm(self.vectors, axis=1)
        
        # Product of norms
        norms = query_norm * vector_norms
        
        # Zero-norm safety: if any norm is zero, mathematically cosine sim is undefined.
        # We assign a safe deterministic similarity of 0.0 for zero vectors to avoid NaN crashes.
        similarities = np.zeros_like(dot_products)
        valid_mask = norms > 0
        similarities[valid_mask] = dot_products[valid_mask] / norms[valid_mask]
        
        return similarities

    def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """
        Perform exact cosine similarity search.
        Cosine similarity = (A dot B) / (||A|| * ||B||)
        """
        if top_k <= 0:
            return []
            
        if self.vectors is None or len(self.vectors) == 0:
            return []
            
        query_vector = np.array(query_embedding, dtype=np.float32)
        similarities = self._cosine_similarities(query_vector)
        
        k = min(top_k, len(self.metadata))
        if k == 0:
            return []
            
        # Sort descending while maintaining insertion order for ties.
        # np.argsort on -similarities with kind='stable' natively accomplishes this.
        top_indices = np.argsort(-similarities, kind='stable')[:k]
        
        results = []
        for idx in top_indices:
            results.append(self.metadata[idx])
            
        return results
