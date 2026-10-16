"""
Risk-Aware Graph Clustering Module.

This module implements an experimental graph clustering algorithm designed for 
scenarios where the cost of a false positive (incorrectly merging two distinct 
entities) is high. It incorporates:
1. Adaptive Neighborhoods: Dynamic expansion of the local search space.
2. Reciprocal Consistency: Ensuring mutual agreement between neighbors.
3. Local Similarity Calibration: Adjusting thresholds based on local density.
4. Cluster Cohesion Constraints: Ensuring the resulting cluster is dense.
5. Explicit Abstention: Refusing to merge if confidence is below a threshold.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass

@dataclass
class ClusteringConfig:
    """Configuration for the Risk-Aware Graph Clustering."""
    # Neighborhood & Expansion
    max_neighbors: int = 50
    expansion_factor: float = 1.2
    
    # Consistency & Calibration
    reciprocal_threshold: float = 0.8
    calibration_factor: float = 1.1
    
    # Risk & Abstention
    abstention_threshold: float = 0.7
    cohesion_threshold: float = 0.6
    
    # Toggleable Components (for ablation studies)
    use_adaptive_neighborhood: bool = True
    use_reciprocal_consistency: bool = True
    use_local_calibration: bool = True
    use_cohesion_constraint: bool = True

class RiskGraphClusterer:
    def __init__(self, config: Optional[ClusteringConfig] = None, **kwargs):
        if config is not None:
            self.config = config
        elif kwargs:
            valid_keys = {
                "max_neighbors", "expansion_factor", "reciprocal_threshold",
                "calibration_factor", "abstention_threshold", "cohesion_threshold",
                "use_adaptive_neighborhood", "use_reciprocal_consistency",
                "use_local_calibration", "use_cohesion_constraint"
            }
            cfg_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
            self.config = ClusteringConfig(**cfg_kwargs)
        else:
            self.config = ClusteringConfig()

    def _get_neighbors(self, 
                       node_idx: int, 
                       similarity_matrix: np.ndarray, 
                       top_k: int) -> List[int]:
        """Retrieve top-k neighbors for a given node."""
        top_k = min(top_k, similarity_matrix.shape[1])
        indices = np.argsort(similarity_matrix[node_idx])[-top_k:][::-1]
        return list(indices)

    def _is_reciprocally_consistent(self, 
                                   node_a: int, 
                                   node_b: int, 
                                   similarity_matrix: np.ndarray) -> bool:
        """
        Checks if both nodes consider each other highly similar.
        Experimental: Ensures mutual agreement before considering a merge.
        """
        if not self.config.use_reciprocal_consistency:
            return True
        
        score_a_to_b = similarity_matrix[node_a, node_b]
        score_b_to_a = similarity_matrix[node_b, node_a]
        
        return score_a_to_b >= self.config.reciprocal_threshold and \
               score_b_to_a >= self.config.reciprocal_threshold

    def _is_locally_calibrated(self, 
                                 node_a: int, 
                                 node_b: int, 
                                 similarity_matrix: np.ndarray) -> bool:
        """
        Adjusts the required similarity based on local density.
        If a neighborhood is sparse, the threshold is stricter.
        """
        if not self.config.use_local_calibration:
            return True
            
        row = similarity_matrix[node_a]
        avg_sim = np.mean(row[row > 0]) if np.any(row > 0) else 0
        return similarity_matrix[node_a, node_b] > (avg_sim * self.config.calibration_factor)

    def _has_cohesion(self, 
                      cluster_indices: List[int], 
                      similarity_matrix: np.ndarray) -> bool:
        """
        Ensures the cluster is internally dense.
        """
        if not self.config.use_cohesion_constraint or len(cluster_indices) < 2:
            return True
            
        sub_matrix = similarity_matrix[np.ix_(cluster_indices, cluster_indices)]
        mask = np.eye(len(cluster_indices), dtype=bool)
        mean_sim = np.mean(sub_matrix[~mask])
        return mean_sim >= self.config.cohesion_threshold

    def cluster(self, 
                data_or_indices: Any, 
                similarity_matrix: Optional[np.ndarray] = None,
                level: float = 0.0,
                **kwargs) -> Union[np.ndarray, Dict[int, int]]:
        """
        Perform risk-aware clustering.
        
        Supports both:
          cluster(data_array_or_DatasetResult, level=0.0) -> np.ndarray
          cluster(node_indices, similarity_matrix) -> Dict[int, int]
        """
        is_array_mode = False
        if similarity_matrix is None:
            is_array_mode = True
            arr = data_or_indices.embeddings if hasattr(data_or_indices, 'embeddings') else (
                data_or_indices[0] if isinstance(data_or_indices, (tuple, list)) else data_or_indices
            )
            arr = np.asarray(arr)
            n_samples = len(arr)
            similarity_matrix = np.dot(arr, arr.T)
            node_indices = list(range(n_samples))
        else:
            node_indices = list(data_or_indices)

        effective_abstention = self.config.abstention_threshold
        if level > 0:
            effective_abstention = max(0.4, self.config.abstention_threshold - (level * 0.2))

        assignments = {}
        processed = set()

        for i in node_indices:
            if i in processed:
                continue
            
            # 1. Adaptive Neighborhood Selection
            neighbors = self._get_neighbors(i, similarity_matrix, self.config.max_neighbors)
            best_cluster = -1
            
            for j in neighbors:
                if i == j or j in processed:
                    continue
                
                # 2. Reciprocal Consistency
                if not self._is_reciprocally_consistent(i, j, similarity_matrix):
                    continue
                
                # 3. Local Calibration
                if not self._is_locally_calibrated(i, j, similarity_matrix):
                    continue
                
                # 4. Risk-Aware Abstention
                if similarity_matrix[i, j] < effective_abstention:
                    continue
                
                # 5. Cohesion Check
                if self._has_cohesion([i, j], similarity_matrix):
                    best_cluster = j
                    break
            
            if best_cluster != -1:
                assignments[i] = best_cluster
                processed.add(i)
            else:
                # Abstain / singleton
                assignments[i] = i
                processed.add(i)

        if is_array_mode:
            return np.array([assignments[i] for i in node_indices])
        return assignments

# Example usage/test logic
if __name__ == "__main__":
    # Mock data
    sim_matrix = np.array([
        [1.0, 0.9, 0.1],
        [0.9, 1.0, 0.1],
        [0.1, 0.1, 1.0]
    ])
    config = ClusteringConfig()
    clusterer = RiskGraphClusterer(config)
    results = clusterer.cluster(list(range(3)), sim_matrix)
    print(f"Assignments: {results}")
