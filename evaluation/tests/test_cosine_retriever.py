import unittest
import numpy as np
from evaluation.retrieval.cosine_retriever import CosineRetriever

class TestCosineRetriever(unittest.TestCase):
    def setUp(self):
        self.retriever = CosineRetriever(embedding_dim=2)
        
    def test_01_empty_collection(self):
        """Requirement 1: Empty collection behavior."""
        results = self.retriever.similarity_search([1.0, 0.0])
        self.assertEqual(len(results), 0)
        
    def test_02_top_k_limiting(self):
        """Requirement 2: TOP_K limiting."""
        self.retriever.add_embeddings(
            [[1.0, 0.0], [0.0, 1.0]],
            [{"id": 1}, {"id": 2}]
        )
        # top_k > available
        self.assertEqual(len(self.retriever.similarity_search([1.0, 0.0], top_k=5)), 2)
        # top_k <= 0
        self.assertEqual(len(self.retriever.similarity_search([1.0, 0.0], top_k=0)), 0)
        self.assertEqual(len(self.retriever.similarity_search([1.0, 0.0], top_k=-1)), 0)
        
    def test_03_descending_ranking(self):
        """Requirement 3: Descending ranking."""
        self.retriever.add_embeddings(
            [[0.5, 0.5], [1.0, 0.0], [0.1, 0.9]],
            [{"id": 1}, {"id": 2}, {"id": 3}]
        )
        results = self.retriever.similarity_search([1.0, 0.0], top_k=3)
        self.assertEqual([r["id"] for r in results], [2, 1, 3])
        
    def test_04_zero_norm_safety(self):
        """Requirement 4: Zero-norm safety."""
        self.retriever.add_embeddings(
            [[0.0, 0.0], [1.0, 0.0]],
            [{"id": 1}, {"id": 2}]
        )
        # Query with valid vector
        sims = self.retriever._cosine_similarities(np.array([1.0, 0.0], dtype=np.float32))
        self.assertEqual(sims[0], 0.0) # Zero vector in DB
        
        # Query with zero vector
        sims2 = self.retriever._cosine_similarities(np.array([0.0, 0.0], dtype=np.float32))
        self.assertTrue(np.all(sims2 == 0.0))
        
        # Search shouldn't crash
        results = self.retriever.similarity_search([0.0, 0.0])
        self.assertEqual(len(results), 2)
        
    def test_05_identical_vectors(self):
        """Requirement 5: Identical vectors ≈ 1."""
        self.retriever.add_embeddings([[1.0, 2.0]], [{"id": 1}])
        sims = self.retriever._cosine_similarities(np.array([1.0, 2.0], dtype=np.float32))
        self.assertAlmostEqual(sims[0], 1.0, places=5)
        
    def test_06_orthogonal_vectors(self):
        """Requirement 6: Orthogonal vectors ≈ 0."""
        self.retriever.add_embeddings([[1.0, 0.0]], [{"id": 1}])
        sims = self.retriever._cosine_similarities(np.array([0.0, 1.0], dtype=np.float32))
        self.assertAlmostEqual(sims[0], 0.0, places=5)
        
    def test_07_opposite_vectors(self):
        """Requirement 7: Opposite vectors ≈ -1."""
        self.retriever.add_embeddings([[1.0, 1.0]], [{"id": 1}])
        sims = self.retriever._cosine_similarities(np.array([-1.0, -1.0], dtype=np.float32))
        self.assertAlmostEqual(sims[0], -1.0, places=5)
        
    def test_08_dimension_validation(self):
        """Requirement 8: Dimension validation."""
        # Adding wrong dimension
        with self.assertRaises(ValueError):
            self.retriever.add_embeddings([[1.0, 0.0, 0.0]], [{"id": 1}])
            
        self.retriever.add_embeddings([[1.0, 0.0]], [{"id": 1}])
        
        # Querying wrong dimension
        with self.assertRaises(ValueError):
            self.retriever.similarity_search([1.0, 0.0, 0.0])
            
    def test_09_deterministic_tie_breaking(self):
        """Requirement 9: Deterministic tie-breaking."""
        # Adding three identical vectors (ties)
        self.retriever.add_embeddings(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [{"id": "first"}, {"id": "second"}, {"id": "third"}]
        )
        
        # Query with exact match
        results = self.retriever.similarity_search([1.0, 0.0], top_k=3)
        # Should preserve insertion order
        self.assertEqual([r["id"] for r in results], ["first", "second", "third"])

if __name__ == "__main__":
    unittest.main()
