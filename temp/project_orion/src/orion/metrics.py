"""
Metrics for evaluating clustering performance.

This module provides a suite of metrics that are independent of the specific
clustering algorithm used. It evaluates the quality of the clustering by
comparing the predicted labels against a ground truth.

Definitions:
- Pairwise Precision: The proportion of correctly identified pairs.
- Pairwise Recall: The proportion of ground-truth pairs that are correctly identified.
- B-cubed Precision: A metric that weights the precision of each point by the size of its cluster.
- B-cubed Recall: A metric that weights the recall of each point by the size of its cluster.
- Adjusted Rand Index (ARI): Measures the similarity between two partitions, adjusted for chance.
- V-Measure: The harmonic mean of homogeneity and completeness.
- Cluster Contamination: The proportion of points in a cluster that do not belong to the same ground truth.
- Fragmentation: The proportion of ground-truth clusters that are split across multiple clusters.
- Coverage: The proportion of ground-truth clusters that are fully contained within a single predicted cluster.
- False-Match Rate (FMR): The proportion of pairs that are incorrectly grouped together.
- False-Discovery Proportion (FDP): The proportion of points in a cluster that do not belong to the same ground truth.
- Cluster-level Contamination: The proportion of clusters that contain at least one point from a different ground truth.

Note: Abstained observations (e.g., labels marked as -1) are excluded from calculations.
"""

import numpy as np
from typing import List
from sklearn.metrics import homogeneity_score, completeness_score, adjusted_rand_score

def _filter_abstentions(labels_true: np.ndarray, labels_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter out abstained observations (labeled as -1) from both ground truth and predictions.
    """
    mask = (labels_true != -1) & (labels_pred != -1)
    return labels_true[mask], labels_pred[mask]

def pairwise_precision(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Calculate the Pairwise Precision.
    """
    t, p = _filter_abstentions(labels_true, labels_pred)
    if len(t) < 2:
        return 1.0
    
    # Count pairs in the same cluster in ground truth
    true_pairs = 0
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            if t[i] == t[j]:
                true_pairs += 1
    
    # Count pairs in the same cluster in predictions
    pred_pairs = 0
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] == p[j]:
                pred_pairs += 1
    
    # Count pairs that are in the same cluster in both
    correct_pairs = 0
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            if t[i] == t[j] and p[i] == p[j]:
                correct_pairs += 1
                
    if pred_pairs == 0:
        return 0.0
    return correct_pairs / pred_pairs

def pairwise_recall(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Calculate the Pairwise Recall.
    """
    t, p = _filter_abstentions(labels_true, labels_pred)
    if len(t) < 2:
        return 1.0
    
    true_pairs = 0
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            if t[i] == t[j]:
                true_pairs += 1
    
    correct_pairs = 0
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            if t[i] == t[j] and p[i] == p[j]:
                correct_pairs += 1
                
    if true_pairs == 0:
        return 1.0
    return correct_pairs / true_pairs

def b_cubed_precision(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Calculate the B-cubed Precision.
    """
    t, p = _filter_abstentions(labels_true, labels_pred)
    if len(t) == 0:
        return 1.0
    
    precision = 0.0
    unique_preds = np.unique(p)
    for pred_label in unique_preds:
        indices = np.where(p == pred_label)[0]
        if len(indices) == 0:
            continue
        
        correct_in_cluster = 0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if t[indices[i]] == t[indices[j]]:
                    correct_in_cluster += 1
        
        total_pairs_in_cluster = (len(indices) * (len(indices) - 1)) // 2
        if total_pairs_in_cluster > 0:
            precision += correct_in_cluster / total_pairs_in_cluster
            
    if len(unique_preds) == 0:
        return 1.0
    return precision / len(unique_preds)

def b_cubed_recall(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Calculate the B-cubed Recall.
    """
    t, p = _filter_abstentions(labels_true, labels_pred)
    if len(t) == 0:
        return 1.0
    
    recall = 0.0
    unique_trues = np.unique(t)
    for true_label in unique_trues:
        indices = np.where(t == true_label)[0]
        if len(indices) == 0:
            continue
        
        correct_pairs_in_ground_truth = (len(indices) * (len(indices) - 1)) // 2
        correct_pairs_in_pred = 0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if p[indices[i]] == p[indices[j]]:
                    correct_pairs_in_pred += 1
        
        if correct_pairs_in_ground_truth > 0:
            recall += correct_pairs_in_pred / correct_pairs_in_ground_truth
            
    if len(unique_trues) == 0:
        return 1.0
    return recall / len(unique_trues)

def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Calculate the Adjusted Rand Index.
    """
    t, p = _filter_abstentions(labels_true, labels_pred)
    if len(t) == 0:
        return 1.0
    return float(adjusted_rand_score(t, p))

def v_measure(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Calculate the V-measure.
    """
    t, p = _filter_abstentions(labels_true, labels_pred)
    if len(t) == 0:
        return 1.0
    h = homogeneity_score(t, p)
    c = completeness_score(t, p)
    if h + c == 0:
        return 0.0
    return (2 * h * c) / (h + c)

def cluster_contamination(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Calculate cluster contamination.
    """
    t, p = _filter_abstentions(labels_true, labels_pred)
    if len(t) == 0:
        return 0.0
    
    unique_preds = np.unique(p)
    contamination_count = 0
    for pred_label in unique_preds:
        indices = np.where(p == pred_label)[0]
        if len(indices) == 0:
            continue
        
        # A cluster is contaminated if it contains points from different ground truth labels
        unique_trues_in_cluster = np.unique(t[indices])
        if len(unique_trues_in_cluster) > 1:
            contamination_count += 1
            
    if len(unique_preds) == 0:
        return 0.0
    return contamination_count / len(unique_preds)

def fragmentation(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Calculate fragmentation.
    """
    t, p = _filter_abstentions(labels_true, labels_pred)
    if len(t) == 0:
        return 0.0
    
    unique_trues = np.unique(t)
    fragmented_count = 0
    for true_label in unique_trues:
        indices = np.where(t == true_label)[0]
        if len(indices) == 0:
            continue
        
        # A ground truth cluster is fragmented if it is split across multiple predicted clusters
        unique_preds_in_ground_truth = np.unique(p[indices])
        if len(unique_preds_in_ground_truth) > 1:
            fragmented_count += 1
            
    if len(unique_trues) == 0:
        return 0.0
    return fragmented_count / len(unique_trues)

def coverage(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Calculate coverage.
    """
    t, p = _filter_abstentions(labels_true, labels_pred)
    if len(t) == 0:
        return 0.0
    
    unique_trues = np.unique(t)
    covered_count = 0
    for true_label in unique_trues:
        indices = np.where(t == true_label)[0]
        if len(indices) == 0:
            continue
        
        # A ground truth cluster is covered if it is entirely contained in one predicted cluster
        unique_preds_in_ground_truth = np.unique(p[indices])
        if len(unique_preds_in_ground_truth) == 1:
            covered_count += 1
            
    if len(unique_trues) == 0:
        return 0.0
    return covered_count / len(unique_trues)

def false_match_rate(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Calculate the false-match rate.
    """
    t, p = _filter_abstentions(labels_true, labels_pred)
    if len(t) == 0:
        return 0.0
    
    # FMR is the proportion of pairs that are incorrectly grouped together.
    # A pair is incorrectly grouped if they are in the same predicted cluster but different ground truth.
    false_matches = 0
    total_pred_pairs = 0
    
    unique_preds = np.unique(p)
    for pred_label in unique_preds:
        indices = np.where(p == pred_label)[0]
        if len(indices) < 2:
            continue
        
        total_pred_pairs += (len(indices) * (len(indices) - 1)) // 2
        
        # Count pairs in this predicted cluster that have different ground truth labels
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if t[indices[i]] != t[indices[j]]:
                    false_matches += 1
                    
    if total_pred_pairs == 0:
        return 0.0
    return false_matches / total_pred_pairs

def false_discovery_proportion(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Calculate the false-discovery proportion.
    """
    t, p = _filter_abstentions(labels_true, labels_pred)
    if len(t) == 0:
        return 0.0
    
    # FDP is the proportion of points in a cluster that do not belong to the same ground truth.
    # This is equivalent to the proportion of "false" pairs in the predicted clusters.
    # However, the standard definition of FDP in clustering is often related to the 
    # proportion of points in a cluster that are "wrong" relative to the cluster's majority.
    # Given the requirement "proportion of points in a cluster that do not belong to the same ground truth",
    # we interpret it as: for each predicted cluster, what proportion of its points are "extra"?
    
    total_points_in_clusters = 0
    false_discovery_points = 0
    
    unique_preds = np.unique(p)
    for pred_label in unique_preds:
        indices = np.where(p == pred_label)[0]
        if len(indices) == 0:
            continue
        
        # A point is a "false discovery" if it's in a cluster where there's more than one ground truth label
        # and it's not part of the majority ground truth? No, the description says "do not belong to the same ground truth".
        # This means if a cluster has multiple ground truth labels, any point that is not part of the 
        # "main" ground truth is a false discovery. But "the same" implies a single ground truth.
        # If a cluster has only one ground truth label, all points are "correct".
        # If a cluster has multiple ground truth labels, any point that is "extra" is a false discovery.
        
        # Let's use the definition: proportion of points in a cluster that are not part of the 
        # majority ground truth in that cluster.
        
        counts = {}
        for idx in indices:
            label = t[idx]
            counts[label] = counts.get(label, 0) + 1
        
        if len(counts) > 1:
            # The majority ground truth
            max_count = max(counts.values())
            majority_label = max(k for k, v in counts.items() if v == max_count)
            
            # Points that are not part of the majority ground truth
            for label, count in counts.items():
                if label != majority_label:
                    false_discovery_points += count
        
        total_points_in_clusters += len(indices)
        
    if total_points_in_clusters == 0:
        return 0.0
    return false_discovery_points / total_points_in_clusters

def cluster_level_contamination(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Calculate the cluster-level contamination.
    """
    t, p = _filter_abstentions(labels_true, labels_pred)
    if len(t) == 0:
        return 0.0
    
    unique_preds = np.unique(p)
    contaminated_clusters = 0
    for pred_label in unique_preds:
        indices = np.where(p == pred_label)[0]
        if len(indices) == 0:
            continue
        
        # A cluster is contaminated if it contains at least one point from a different ground truth.
        # This is equivalent to saying it contains more than one ground truth label.
        unique_trues_in_cluster = np.unique(t[indices])
        if len(unique_trues_in_cluster) > 1:
            contaminated_clusters += 1
            
    if len(unique_preds) == 0:
        return 0.0
    return contaminated_clusters / len(unique_preds)

def calculate_metrics(*args, **kwargs) -> dict:
    """
    Convenience function to compute all primary metrics.
    Supports:
      calculate_metrics(labels_true, labels_pred)
      calculate_metrics(data, labels_true, labels_pred, **kwargs)
    """
    if len(args) == 2:
        labels_true, labels_pred = args
    elif len(args) >= 3:
        _, labels_true, labels_pred = args[:3]
    elif "labels_true" in kwargs and "labels_pred" in kwargs:
        labels_true = kwargs["labels_true"]
        labels_pred = kwargs["labels_pred"]
    else:
        raise ValueError(f"calculate_metrics requires (labels_true, labels_pred) or (data, labels_true, labels_pred). Got {len(args)} args.")

    if isinstance(labels_pred, dict):
        labels_pred = np.array([labels_pred.get(i, -1) for i in range(len(labels_true))])
    else:
        labels_pred = np.asarray(labels_pred)
    labels_true = np.asarray(labels_true)

    return {
        "pairwise_precision": float(pairwise_precision(labels_true, labels_pred)),
        "pairwise_recall": float(pairwise_recall(labels_true, labels_pred)),
        "b_cubed_precision": float(b_cubed_precision(labels_true, labels_pred)),
        "b_cubed_recall": float(b_cubed_recall(labels_true, labels_pred)),
        "adjusted_rand_index": float(adjusted_rand_index(labels_true, labels_pred)),
        "v_measure": float(v_measure(labels_true, labels_pred)),
        "cluster_contamination": float(cluster_contamination(labels_true, labels_pred)),
        "fragmentation": float(fragmentation(labels_true, labels_pred)),
        "coverage": float(coverage(labels_true, labels_pred)),
        "false_match_rate": float(false_match_rate(labels_true, labels_pred)),
        "false_discovery_proportion": float(false_discovery_proportion(labels_true, labels_pred)),
        "cluster_level_contamination": float(cluster_level_contamination(labels_true, labels_pred)),
    }

