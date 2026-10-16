import unittest
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.orion.data import SyntheticEmbeddingGenerator
from src.orion.risk_graph import RiskGraphClusterer, ClusteringConfig
from src.orion.baselines import threshold_graph_clustering, agglomerative_clustering, rr_graph_clustering
from src.orion.metrics import calculate_metrics
from src.orion.runner import EvaluationRunner, ExperimentConfig, ExperimentRunner

class TestReproduction(unittest.TestCase):
    def test_deterministic_generation(self):
        """Verify that fixed seeds generate strictly identical vectors and labels."""
        d1 = SyntheticEmbeddingGenerator.generate(seed=42, dimension=192, num_entities=5, observations_per_entity=10)
        d2 = SyntheticEmbeddingGenerator.generate(seed=42, dimension=192, num_entities=5, observations_per_entity=10)
        np.testing.assert_allclose(d1.embeddings, d2.embeddings, atol=1e-12)
        np.testing.assert_array_equal(d1.labels, d2.labels)

    def test_all_algorithms_produce_valid_clusters(self):
        """Verify all clustering algorithms execute and return cluster labels without crashing."""
        data = SyntheticEmbeddingGenerator.generate(seed=123, dimension=192, num_entities=4, observations_per_entity=8)
        
        c_thresh = threshold_graph_clustering(data)
        c_agg = agglomerative_clustering(data)
        c_rr = rr_graph_clustering(data)
        
        clusterer = RiskGraphClusterer()
        c_risk = clusterer.cluster(data)
        
        for c in [c_thresh, c_agg, c_rr, c_risk]:
            self.assertEqual(len(c), len(data.labels))
            self.assertTrue(isinstance(c, np.ndarray))

    def test_metrics_range_and_correctness(self):
        """Verify that metrics calculate values within valid mathematical bounds [0.0, 1.0]."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        m_perfect = calculate_metrics(y_true, y_pred)
        self.assertAlmostEqual(m_perfect["pairwise_precision"], 1.0)
        self.assertAlmostEqual(m_perfect["pairwise_recall"], 1.0)
        self.assertAlmostEqual(m_perfect["cluster_contamination"], 0.0)

    def test_artifacts_exist_and_valid_json(self):
        """Verify that all required machine-readable research artifacts exist and parse."""
        required_artifacts = [
            "artifacts/development/results.json",
            "artifacts/development/selected_config.json",
            "artifacts/evaluation/results.json",
            "artifacts/ablation/results.json",
            "artifacts/sensitivity/results.json",
            "artifacts/scalability/results.json"
        ]
        for path in required_artifacts:
            full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", path))
            self.assertTrue(os.path.exists(full_path), f"Artifact missing: {path}")
            with open(full_path, "r") as f:
                content = json.load(f)
                self.assertTrue(len(content) > 0, f"Artifact empty: {path}")

if __name__ == "__main__":
    unittest.main()
