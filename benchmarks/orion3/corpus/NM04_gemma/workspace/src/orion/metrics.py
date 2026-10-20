import numpy as np
from sklearn.metrics import adjusted_rand_score, v_measure

def calculate_metrics(labels_true, labels_pred):
    """
    Calculates various evaluation metrics for clustering and retrieval performance.
    """
    n = len(labels_true)
    accepted_indices = [i for i, val in enumerate(labels_pred) if val != -1]
    n_accepted = len(accepted_indices)
    
    # Basic counts
    outlier_indices = [i for i, val in enumerate(labels_true) if val == -1]
    n_outliers = len(outlier_indices)
    
    # Pairwise logic
    # A true pair is a pair {i, j} where i != j and label_true[i] == label_true[j] 
    # AND neither is -1.
    true_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if labels_true[i] == labels_true[j] and labels_true[i] != -1:
                true_pairs += 1
                
    # A linked pair is a pair {i, j} where i != j, both predictions are not -1, 
    # and the predictions are equal.
    linked_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if labels_pred[i] != -1 and labels_pred[j] != -1 and labels_pred[i] == labels_pred[j]:
                linked_pairs += 1
    
    # tp = linked pairs that are also true pairs
    # Since a true pair implies labels_true[i] == labels_true[j] != -1,
    # and a linked pair implies labels_pred[i] == labels_pred[j] != -1,
    # a pair is both if it is linked and the true labels are the same (non-outlier).
    # Note: if labels_true[i] == labels_true[j] and they are not -1, then it's a true_pair.
    # If it's also a linked_pair, it is a tp.
    tp = 0
    for i in range(n):
        for j in range(i + 1, n):
            if labels_pred[i] != -1 and labels_pred[j] != -1 and labels_pred[i] == labels_pred[j]:
                if labels_true[i] == labels_true[j] and labels_true[i] != -1:
                    tp += 1
                    
    fp = linked_pairs - tp
    impostor = (n * (n - 1) // 2) - true_pairs
    fn = true_pairs - tp
    tn = impostor - fp
    accepted_pairs = tp + fp
    
    # Ratios
    pairwise_precision = tp / accepted_pairs if accepted_pairs > 0 else None
    pairwise_recall = tp / true_pairs if true_pairs > 0 else None
    
    # pairwise_f1 = 2*tp / (tp + fp + true_pairs) is not standard. 
    # The spec says: 2*tp / (tp + fp + true_pairs)
    denom_f1 = (tp + fp + true_pairs)
    pairwise_f1 = (2 * tp / denom_f1) if denom_f1 > 0 else None
    
    # B3 metrics
    b3_p_sum = 0.0
    b3_r_sum = 0.0
    b3_p_count = 0
    
    if n_accepted > 0:
        for i in accepted_indices:
            # J(i) = count of accepted j (incl i) with same true and same pred
            # P(i) = count of accepted j (incl i) with same pred
            # T(i) = count of accepted j (incl i) with same true
            # Tall(i) = count of all j (incl i) with same true
            
            # Find indices of accepted items with same prediction
            p_i = 0
            for j in accepted_indices:
                if labels_pred[j] == labels_pred[i]:
                    p_i += 1
            
            # Find indices of accepted items with same true label
            t_i = 0
            for j in accepted_indices:
                if labels_true[j] == labels_true[i]:
                    t_i += 1
            
            # J(i) is subset of t_i where they also have same pred
            j_i = 0
            for j in accepted_indices:
                if labels_true[j] == labels_true[i] and labels_pred[j] == labels_pred[i]:
                    j_i += 1
            
            # Tall(i) is all items (accepted or not) with same true label
            tall_i = 0
            for j in range(n):
                if labels_true[j] == labels_true[i]:
                    tall_i += 1
            
            # Calculations
            p_val = j_i / p_i if p_i > 0 else None
            r_val = j_i / tall_i if tall_i > 0 else None
            
            # Note: Tall(i) is always >= 1 because i is included.
            # P(i) is always >= 1 for accepted i.
            # However, let's stick to logic for safety.
            
            b3_p_sum += p_val if p_val is not None else 0
            b3_r_sum += r_val if r_val is not None else 0
            b3_p_count += 1
            
        b_cubed_precision = b3_p_sum / b3_p_count if b3_p_count > 0 else None
        b_cubed_recall = b3_r_sum / b3_r_count if b3_r_count > 0 else None
        # Re-evaluating logic for b_cubed_f1: "None if either is None or both are 0"
        # The loop actually calculates the mean of the ratios.
        
    # Correcting B3 logic to match spec exactly:
    # b_cubed_precision is mean over accepted i of J(i)/P(i)
    # b_cubed_recall is mean over accepted i of J(i)/T(i)
    # wait, the prompt says T(i) for recall is "T(i)", but the "B3 terms" section 
    # defines T(i) as "accepted j with same true" and Tall(i) as "all j with same true".
    # The table asks for "b_cubed_recall" which uses T(i) in formula but the 
    # "B3 terms" block lists "Tall(i)". Usually, standard B3 uses T(i).
    # Let's look at the definition in the table: "b_cubed_recall: mean over accepted i of J(i) / T(i)"
    # And the "B3 terms" section: "J(i) = ... same true and same pred; P(i) = ... same pred; 
    # T(i) = ... accepted j with same true; Tall(i) = ... all j with same true".
    # The table uses T(i) in the formula for b_cubed_recall.
    
    # Let's re-calculate b_cubed_recall correctly using T(i).
    # Wait, the table uses T(i) for b_cubed_recall but the B3 terms section defines T(i) 
    # and Tall(i). Let's check the specific text: 
    # "b_cubed_recall | mean over accepted i of J(i) / T(i)"
    # "b_cubed_recall_global | sum over accepted i of J(i) / Tall(i), divided by n"
    
    # Redoing B3 logic to be precise:
    b3_p_vals = []
    b3_r_vals = []
    for i in accepted_indices:
        # J(i)
        j_i = 0
        for j in accepted_indices:
            if labels_true[j] == labels_true[i] and labels_pred[j] == labels_pred[i]:
                j_i += 1
        # P(i)
        p_i = 0
        for j in accepted_indices:
            if labels_pred[j] == labels_pred[i]:
                p_i += 1
        # T(i)
        t_i = 0
        for j in accepted_indices:
            if labels_true[j] == labels_true[i]:
                t_i += 1
        # Tall(i)
        tall_i = 0
        for j in range(n):
            if labels_true[j] == labels_true[i]:
                tall_i += 1
        
        b3_p_vals.append(j_i / p_i if p_i > 0 else None)
        b3_r_vals.append(j_i / t_i if t_i > 0 else None)

    # Now calculate the means
    b_cubed_precision = sum(v for v in b3_p_vals if v is not None) / len(b3_p_vals) if len(b3_p_vals) > 0 else None
    b_cubed_recall = sum(v for v in b3_r_vals if v is not None) / len(b3_r_vals) if len(b3_r_vals) > 0 else None
    
    # b_cubed_f1: harmonic mean of the two above; None if either is None or both are 0
    if b_cubed_precision is not None and b_cubed_recall is not None:
        denom_f1 = (b_cubed_precision + b_cubed_recall)
        if denom_f1 == 0:
            b_cubed_f1 = None
        else:
            b_cubed_f1 = (2 * b_cubed_precision * b_cubed_recall) / denom_f1
    else:
        b_cubed_f1 = None

    # Global terms
    # b_cubed_precision_global: sum over accepted i of J(i) / P(i), divided by n
    # b_cubed_recall_global: sum over accepted i of J(i) / Tall(i), divided by n
    # Note: J(i)/P(i) is the term inside the mean for b_cubed_precision.
    
    # Re-calculating J/P and J/Tall
    sum_j_p = 0
    sum_j_tall = 0
    for i in accepted_indices:
        # Recalculate J, P, Tall for these specific sums
        j_i = 0
        for j in accepted_indices:
            if labels_true[j] == labels_true[i] and labels_pred[j] == labels_pred[i]:
                j_i += 1
        p_i = 0
        for j in accepted_indices:
            if labels_pred[j] == labels_pred[i]:
                p_i += 1
        tall_i = 0
        for j in range(n):
            if labels_true[j] == labels_true[i]:
                tall_i += 1
        
        if p_i > 0: sum_j_p += j_i / p_i
        if tall_i > 0: sum_j_tall += j_i / tall_i
    
    b_cubed_precision_global = sum_j_p / n if n > 0 else None
    b_cubed_recall_global = sum_j_tall / n if n > 0 else None

    # Standard metrics
    coverage = n_accepted / n if n > 0 else None
    false_match_rate = fp / impostor if impostor > 0 else None
    false_discovery_proportion = fp / accepted_pairs if accepted_pairs > 0 else None

    # Adjusted Rand Index and V-Measure
    # "every true outlier and every abstained observation is its own singleton cluster"
    # This means we need to construct the partition sets.
    if n == 0:
        ari = None
        v_meas = None
    else:
        # Construct partition for ARI/V-measure
        # To ensure outliers and abstentions are unique, we can give them unique IDs
        # that don't collide with others.
        # A simple way: if it's an outlier or abstained, it gets its own unique index.
        # Since any ID is okay as long as they are distinct, we can use the original index.
        # For items that are NOT outliers and NOT abstained, they share an ID if they 
        # have the same label_true and the same label_pred (and both are not -1).
        
        # Actually, the spec says: "every true outlier and every abstained observation is 
        # its own singleton cluster".
        # This means if index i is an outlier OR prediction is -1, it's in a singleton.
        # If it's not, it's in a cluster with others having the same prediction.
        # However, since the "true" partition is based on labels_true, and
        # "true outliers" are already distinct in the truth (they are -1), 
        # and "abstained" are effectively not in any prediction.
        
        # Let's construct the partitions:
        # Partitions are sets of indices.
        # For the "True" partition:
        # Items with same label_true (not -1) are in one cluster.
        # Items with label_true == -1 are each in their own cluster.
        # For the "Pred" partition:
        # Items with same label_pred (not -1) are in one cluster.
        # Items with label_pred == -1 are each in their own cluster.
        
        # Note: The "abstained" items in the pred partition are those where pred == -1.
        # Even if they have the same true label, they are "separated" because 
        # the system "abstained".
        
        def get_partition(labels):
            # If labels[i] == -1, it's a singleton.
            # Otherwise, it's grouped by its value.
            # However, to ensure it's a "singleton" even if others have the same value:
            # If label is -1, the group ID is the index itself.
            # If label is not -1, the group ID is the value.
            # But the rule says "every abstained observation is its own singleton".
            # So if prediction is -1, it's a singleton.
            
            # Actually, the rule is simpler:
            # In truth, outliers (true == -1) are singletons.
            # In pred, abstained (pred == -1) are singletons.
            pass

        # Let's refine:
        # True partition:
        # If label_true[i] == -1, it's in a unique group.
        # If label_true[i] != -1, it's in a group with others who have same label_true.
        # Wait, if two items have label_true = 0, they are in the same cluster.
        # If one is an outlier, it's -1.
        t_partition = []
        t_map = {}
        t_count = 0
        for i in range(n):
            val = labels_true[i]
            if val == -1:
                t_map[i] = t_count
                t_count += 1
            else:
                if val not in t_map:
                    t_map[val] = t_count
                    t_count += 1
                t_map[i] = t_map[val]
        
        # Pred partition:
        # If label_pred[i] == -1, it's in a unique group.
        # If label_pred[i] != -1, it's in a group with others who have same label_pred.
        p_map = {}
        p_count = 0
        for i in range(n):
            val = labels_pred[i]
            if val == -1:
                p_map[i] = p_count
                p_count += 1
            else:
                if val not in p_map:
                    p_map[val] = p_count
                    p_count += 1
                p_map[i] = p_map[val]
        
        ari = adjusted_rand_score(list(t_map.values()), list(p_map.values()))
        v_meas = v_measure(list(t_map.values()), list(p_map.values()))[1]

    return {
        "total_observations": n,
        "accepted_observations": n_accepted,
        "abstained_observations": n - n_accepted,
        "outlier_observations": n_outliers,
        "true_positive_pairs": tp,
        "false_positive_pairs": fp,
        "false_negative_pairs": fn,
        "true_negative_pairs": tn,
        "accepted_pairs": accepted_pairs,
        "true_pairs": true_pairs,
        "impostor_pairs": impostor,
        "pairwise_precision": pairwise_precision,
        "pairwise_recall": pairwise_recall,
        "pairwise_f1": pairwise_f1,
        "b_cubed_precision": b_cubed_precision,
        "b_cubed_recall": b_cubed_recall,
        "b_cubed_f1": b_cubed_f1,
        "b_cubed_precision_global": b_cubed_precision_global,
        "b_cubed_recall_global": b_cubed_recall_global,
        "coverage": coverage,
        "false_match_rate": false_match_rate,
        "false_discovery_proportion": false_discovery_proportion,
        "adjusted_rand_index": ari,
        "v_measure": v_meas,
    }
