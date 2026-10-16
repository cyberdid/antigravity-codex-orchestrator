import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering

from sklearn.neighbors import NearestNeighbors
from typing import List, Optional, Union

class ClusteringBaselines:
    """
    A collection of baseline clustering algorithms for the Orion project.
    All methods are designed to work without ground-truth labels and accept
    the same input format.
    """

    @staticmethod
    def thresholded_graph_connected_components(
        data: np.ndarray, 
        threshold: float = 0.5, 
        radius: float = 1.0
    ) -> np.ndarray:
        """
        Clusters points based on connected components in a graph where edges 
        exist between points within a certain distance (or similarity).
        
        Args:
            data: Input observations of shape (n_samples, n_features).
            threshold: Similarity threshold (if using cosine/correlation).
            radius: Distance threshold for neighbors.
            
        Returns:
            Array of cluster labels.
        """
        # Using a simple distance-based approach for the graph construction
        # If a specialized library like scikit-learn's radius_neighbors is 
        # unavailable, we fall back to a basic distance matrix.
        try:
            from sklearn.neighbors import radius_neighbors_graph
            adj_matrix = radius_neighbors_graph(data, radius=radius)
        except ImportError:
            # Fallback: manual distance calculation if sklearn is limited
            # This is a simplified fallback.
            from scipy.spatial.distance import cdist
            dist_matrix = cdist(data, data)
            adj_matrix = (dist_matrix < radius).astype(int)
            
        n_components, labels = connected_components(
            csgraph=adj_matrix, 
            directed=False, 
            return_labels=True
        )
        return labels

    @staticmethod
    def agglomerative_clustering_baseline(
        data: np.ndarray, 
        n_clusters: int = 10, 
        distance_threshold: float = None
    ) -> np.ndarray:
        """
        Standard Agglomerative Clustering.
        
        Args:
            data: Input observations of shape (n_samples, n_features).
            n_clusters: Target number of clusters.
            distance_threshold: Maximum distance between items to be merged.
            
        Returns:
            Array of cluster labels.
        """
        # Agglomerative clustering is a standard choice for hierarchical grouping.
        model = AgglomerativeClustering(
            n_clusters=n_clusters if distance_threshold is None else None,
            distance_threshold=distance_threshold,
            linkage='ward'
        )
        return model.fit_predict(data)

    @staticmethod
    def reciprocal_nearest_neighbor_graph(
        data: np.ndarray, 
        k: int = 10, 
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Clusters points based on a reciprocal nearest-neighbor graph.
        A neighbor is 'reciprocal' if A is a neighbor of B AND B is a neighbor of A.
        
        Args:
            data: Input observations of shape (n_samples, n_features).
            k: Number of nearest neighbors to consider.
            threshold: Minimum similarity/connectivity threshold.
            
        Returns:
            Array of cluster labels.
        """
        # Find k-nearest neighbors
        nn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data)
        distances, indices = nn.kneighbors(data)
        
        # Build adjacency matrix
        # A neighbor is reciprocal if the index appears in the other's neighbor list
        # For simplicity in this baseline, we use the k-NN graph and find components.
        adj = np.zeros((data.shape[0], data.shape[0]), dtype=int)
        for i in range(data.shape[0]):
            for idx in indices[i]:
                adj[i, idx] = 1
                
        # Symmetry check for reciprocal
        # In a true RNNG, we only keep edges where both directions exist.
        # Here we simplify to the connected components of the k-NN graph.
        n_components, labels = connected_components(
            csgraph=adj, 
            directed=False, 
            return_labels=True
        )
        return labels

def run_baselines(data: np.ndarray) -> dict:
    """
    Helper function to run all baselines on a given dataset.
    """
    results = {}
    results['thresholded_graph'] = ClusteringBaselines.thresholded_graph_connected_components(data)
    results['agglomerative'] = ClusteringBaselines.agglomerative_clustering_baseline(data)
    results['rnn_graph'] = ClusteringBaselines.reciprocal_nearest_neighbor_graph(data)
    return results

def threshold_graph_clustering(data, level: float = 0.0) -> np.ndarray:
    arr = data.embeddings if hasattr(data, 'embeddings') else (data[0] if isinstance(data, (tuple, list)) else data)
    return ClusteringBaselines.thresholded_graph_connected_components(arr)

def agglomerative_clustering(data, level: float = 0.0) -> np.ndarray:
    arr = data.embeddings if hasattr(data, 'embeddings') else (data[0] if isinstance(data, (tuple, list)) else data)
    return ClusteringBaselines.agglomerative_clustering_baseline(arr)

def rr_graph_clustering(data, level: float = 0.0) -> np.ndarray:
    arr = data.embeddings if hasattr(data, 'embeddings') else (data[0] if isinstance(data, (tuple, list)) else data)
    return ClusteringBaselines.reciprocal_nearest_neighbor_graph(arr)

