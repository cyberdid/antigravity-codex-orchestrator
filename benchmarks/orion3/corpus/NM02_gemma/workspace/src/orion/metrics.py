import numpy as np
from sklearn.metrics import adjusted_rand_score, v_measure

def calculate_metrics(labels_true, labels_pred):
    """
    Calculates various evaluation metrics for clustering and retrieval tasks
    based on the ORION specification.
    """
    n = len(labels_true)
    accepted_indices = [i for i, val in enumerate(labels_pred) if val != -1]
    num_accepted = len(accepted_indices)
    
    # Basic counts
    outlier_count = np.sum(labels_true == -1)
    
    # Pairwise metrics
    # A true pair: i != j, labels_true[i] == labels_true[j], and labels_true[i] != -1
    # A linked pair: i != j, labels_pred[i] == labels_pred[j], and labels_pred[i] != -1
    
    true_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if labels_true[i] == labels_true[j] and labels_true[i] != -1:
                true_pairs += 1
                
    linked_pairs = 0
    for i in range(len(accepted_indices)):
        for j in range(i + 1, len(accepted_indices)):
            idx_i = accepted_indices[i]
            idx_j = accepted_indices[j]
            if labels_pred[idx_i] == labels_pred[idx_j]:
                linked_pairs += 1
                
    # Overlap logic
    # tp: true pairs that are linked
    # fp: linked pairs that are not true pairs
    # Since a linked pair is only "not a true pair" if it's not a true pair.
    # Any linked pair is either a true_pair or a false_positive_pair.
    # Because labels_true[i] == -1 means it's its own identity, 
    # a pair of outliers can never be a true_pair.
    
    tp = 0
    fp = 0
    for i in range(len(accepted_indices)):
        for j in range(i + 1, len(accepted_indices)):
            idx_i = accepted_indices[i]
            idx_j = accepted_indices[j]
            if labels_pred[idx_i] == labels_pred[idx_j]:
                if labels_true[idx_i] == labels_true[idx_j] and labels_true[idx_i] != -1:
                    tp += 1
                else:
                    fp += 1
    
    impostor = (n * (n - 1) // 2) - true_pairs
    
    # Result construction
    res = {}
    res['total_observations'] = n
    res['accepted_observations'] = num_accepted
    res['abstained_observations'] = n - num_accepted
    res['outlier_observations'] = int(outlier_count)
    res['true_positive_pairs'] = tp
    res['false_positive_pairs'] = fp
    res['false_negative_pairs'] = true_pairs - tp
    res['true_negative_pairs'] = impostor - fp
    res['accepted_pairs'] = tp + fp
    res['true_pairs'] = true_pairs
    res['impostor_pairs'] = impostor
    
    # Ratios
    denom_p = tp + fp
    res['pairwise_precision'] = tp / denom_p if denom_p > 0 else None
    
    denom_r = true_pairs
    res['pairwise_recall'] = tp / denom_r if denom_r > 0 else None
    
    if denom_p > 0 and denom_r > 0:
        res['pairwise_f1'] = (2 * tp) / (tp + fp + true_pairs) if (tp + fp + true_pairs) > 0 else None
    else:
        res['pairwise_f1'] = None
        
    # B-cubed metrics
    # J(i) = count of accepted j (incl i) with same true identity AND same pred cluster
    # P(i) = count of accepted j with same pred cluster
    # T(i) = count of accepted j with same true identity
    # Tall(i) = count of all j (accepted or not) with same true identity
    
    b3_p_sum = 0.0
    b3_r_sum = 0.0
    count_accepted = 0
    
    if num_accepted > 0:
        for i in range(len(accepted_indices)):
            idx_i = accepted_indices[i]
            
            # J(i) calculation
            j_i = 0
            for j_idx in range(len(accepted_indices)):
                idx_j = accepted_indices[j_idx]
                if labels_pred[idx_i] == labels_pred[idx_j] and labels_true[idx_i] == labels_true[idx_j]:
                    j_i += 1
            
            # P(i) calculation
            p_i = 0
            for j_idx in range(len(accepted_indices)):
                idx_j = accepted_indices[j_idx]
                if labels_pred[idx_i] == labels_pred[idx_j]:
                    p_i += 1
            
            # T(i) calculation
            t_i = 0
            for j_idx in range(len(accepted_indices)):
                idx_j = accepted_indices[j_idx]
                if labels_true[idx_i] == labels_true[idx_j]:
                    t_i += 1
            
            # Tall(i) calculation
            tall_i = 0
            for j_idx in range(n):
                if labels_true[idx_i] == labels_true[j_idx]:
                    tall_i += 1
            
            if p_i > 0:
                b3_p_sum += j_i / p_i
            if tall_i > 0:
                b3_r_sum += j_i / tall_i
            count_accepted += 1
            
        res['b_cubed_precision'] = b3_p_sum / num_accepted if num_accepted > 0 else None
        res['b_cubed_recall'] = b3_r_sum / num_accepted if num_accepted > 0 else None
        
        # b_cubed_f1 is harmonic mean of the two above
        bp = res['b_cubed_precision']
        br = res['b_cubed_recall']
        if bp is not None and br is not None:
            if bp == 0 or br == 0:
                res['b_cubed_f1'] = None
            else:
                res['b_cubed_f1'] = (2 * bp * br) / (bp + br)
        else:
            res['b_cubed_f1'] = None

    res['b_cubed_precision_global'] = b3_p_sum / n if n > 0 else None
    res['b_cubed_recall_global'] = b3_r_sum / n if n > 0 else None
    
    res['coverage'] = num_accepted / n if n > 0 else None
    
    if impostor > 0:
        res['false_match_rate'] = fp / impostor
    else:
        res['false_match_rate'] = None
        
    if denom_p > 0:
        res['false_discovery_proportion'] = fp / denom_p
    else:
        res['false_discovery_proportion'] = None

    # ARI and V-measure
    # Need to construct partitions. 
    # "every true outlier and every abstained observation is its own singleton cluster"
    # To do this, we must map indices to cluster IDs.
    # For labels_true: if value is -1, it's a unique ID.
    # For labels_pred: if value is -1, it's a unique ID.
    
    if n == 0:
        res['adjusted_rand_index'] = None
        res['v_measure'] = None
    else:
        # Construct labels_true_partition
        # If label is -1, it's unique. Otherwise, it's the shared ID.
        t_parts = []
        for val in labels_true:
            if val == -1:
                t_parts.append(val + 1000000 + (0)) # Placeholder, logic below
        # Actually, a better way to ensure unique IDs for -1:
        # Use a unique ID for each index where label is -1
        t_partition = []
        for i in range(n):
            if labels_true[i] == -1:
                t_partition.append(i) # Index is unique
            else:
                t_partition.append(labels_true[i])
        
        # Construct labels_pred_partition
        # If value is -1, it's a unique ID.
        p_partition = []
        for i in range(n):
            if labels_pred[i] == -1:
                p_partition.append(i) # Index is unique
            else:
                p_partition.append(labels_pred[i])
        
        res['adjusted_rand_index'] = float(adjusted_rand_score(t_partition, p_partition))
        
        # v_measure is (2 * prec * rec) / (prec + rec)
        # sklearn.metrics.v_measure returns a tuple (v_measure, accuracy)
        vm_val, _ = v_measure(t_partition, p_partition, average='samples')
        # The spec says: "None if n == 0; 1.0 if both entropies are 0"
        # Sklearn handles the entropy=0 case to return 1.0
        res['v_measure'] = float(vm_val)

    return res
