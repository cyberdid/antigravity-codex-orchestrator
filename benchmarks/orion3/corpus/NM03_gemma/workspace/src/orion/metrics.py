import numpy as np
from sklearn.metrics import adjusted_rand_score, contingency_matrix
from scipy.special import entr

def calculate_metrics(labels_true, labels_pred):
    n = len(labels_true)
    accepted_mask = labels_pred != -1
    num_accepted = np.sum(accepted_mask)
    
    # Basic counts
    outlier_mask = labels_true == -1
    num_outliers = np.sum(outlier_mask)
    
    # Pairwise counts
    # A pair {i, j} is true if i != j and labels_true[i] == labels_true[j] and labels_true[i] != -1
    # A pair {i, j} is linked if i != j and labels_pred[i] == labels_pred[j] and labels_pred[i] != -1
    
    true_pairs = 0
    # To count true_pairs efficiently: identify groups of indices with same non-negative label
    unique_true = np.unique(labels_true[~outlier_mask])
    for label in unique_true:
        count = np.sum(labels_true == label)
        true_pairs += count * (count - 1) // 2
        
    # To calculate tp, fp, etc., we need to identify pairs
    # A pair is linked if they share the same non-negative prediction
    # A pair is true if they share the same non-negative true label
    
    # Number of linked pairs
    # Find groups of indices sharing same prediction
    unique_preds = np.unique(labels_pred[accepted_mask])
    linked_pairs = 0
    # For each prediction cluster, count pairs
    # However, we only care about the count of pairs that are linked
    # tp = linked pairs that are also true pairs
    # fp = linked pairs that are NOT true pairs
    
    # Let's iterate through all pairs (i, j) where i < j
    # Since n is not explicitly bounded but usually manageable in these tests
    # Optimization: iterate over unique labels
    
    tp = 0
    fp = 0
    
    # For tp: both labels_true[i] == labels_true[j] (not -1) and labels_pred[i] == labels_pred[j] (not -1)
    # This means they must have the same label in both arrays.
    # Since a prediction is only 'linked' if it is the same non-negative value,
    # and a 'true' pair is only true if it's the same non-negative value.
    # A pair is both if it shares the same label in both. 
    # Wait, the specification says: "true_pairs - tp" is false_negative_pairs.
    # A pair is "linked" if labels_pred[i] == labels_pred[j] and both != -1.
    # A pair is "true" if labels_true[i] == labels_true[j] and both != -1.
    
    # Let's count tp:
    # A pair is a "true pair" if they have the same truth label (not -1).
    # A pair is "linked" if they have the same pred label (not -1).
    # tp = number of pairs that are BOTH.
    # This happens if they have the same true label AND same pred label.
    # Because if they have the same pred label, they are linked. 
    # If they also have the same true label, they are a true pair.
    
    # To find tp:
    # Group indices by (truth_label, pred_label) where both are not -1.
    # If two indices i, j have same truth_label and same pred_label, they form a tp.
    
    # To find fp:
    # A pair is linked if pred_i == pred_j and pred_i != -1.
    # It is an "impostor" if it is NOT a true pair.
    # Therefore, fp = (total linked pairs) - (linked pairs that are true pairs)
    # Actually, the definition says: fp = linked pairs that are not true pairs.
    # A linked pair is one where pred_i == pred_j != -1.
    # A true pair is one where truth_i == truth_j != -1.
    # A linked pair is a "true pair" ONLY if truth_i == truth_j.
    # Since a pair is only "true" if truth_i == truth_j and truth_i != -1,
    # and a pair is "linked" if pred_i == pred_j and pred_i != -1.
    # If a pair is both, then it's a "true pair" that is "linked".
    
    # Let's use the definitions:
    # true_pairs: pairs {i, j} where i != j, truth[i] == truth[j], and truth[i] != -1
    # linked_pairs: pairs {i, j} where i != j, pred[i] == pred[j], and pred[i] != -1
    # tp = count of pairs that are BOTH.
    # fp = count of pairs that are linked but NOT true.
    
    # Calculation:
    # 1. Identify indices where pred != -1
    # 2. Group these indices by pred_label. For each group of size C, 
    #    number of linked pairs is C*(C-1)/2.
    # 3. Of these linked pairs, some are "true". 
    #    A linked pair is "true" if the two indices have the same truth_label.
    #    This can only happen if truth_i == truth_j (and not -1).
    
    # Logic:
    # A pair {i, j} is linked if pred[i] == pred[j] and pred[i] != -1.
    # A pair {i, j} is true if truth[i] == truth[j] and truth[i] != -1.
    # tp = count of pairs {i, j} such that they are both linked and true.
    # This happens if pred[i] == pred[j] != -1 AND truth[i] == truth[j] != -1.
    
    # To calculate tp:
    # Group indices by (truth_val, pred_val) where both are not -1.
    # For each unique (truth_val, pred_val) pair, if there are K indices 
    # with that specific combination, they form K*(K-1)/2 tp's.
    
    # To calculate true_pairs:
    # Group indices by truth_val (not -1). If there are K indices, 
    # they form K*(K-1)/2 true pairs.
    
    # To calculate impostor:
    # Total pairs is n*(n-1)/2.
    # true_pairs is as calculated above.
    # impostor = total - true_pairs.
    
    # To calculate fp:
    # fp = linked_pairs - tp.
    # A linked pair is "linked" if pred_i == pred_j != -1.
    # A linked pair is "true" if it's also a true pair (truth_i == truth_j != -1).
    # So, fp = (total linked pairs) - (linked pairs that are also true pairs).
    # Since a linked pair is only "true" if truth_i == truth_j, 
    # and if pred_i == pred_j and truth_i == truth_j, then it's a tp.
    
    # Let's re-verify:
    # tp = true pairs that are linked.
    # fp = linked pairs that are not true.
    # If a pair is linked (pred_i == pred_j != -1), it's either:
    # 1. A true pair (truth_i == truth_j != -1) -> This is a tp.
    # 2. Not a true pair (truth_i != truth_j or truth_i == -1) -> This is a fp.
    # Thus, total linked pairs = tp + fp.
    
    # Calculation steps:
    # 1. Find all pairs {i, j} with i < j.
    # 2. Identify "true" pairs: truth[i] == truth[j] and truth[i] != -1.
    # 3. Identify "linked" pairs: pred[i] == pred[j] and pred[i] != -1.
    # 4. tp = count of pairs that are both.
    # 5. fp = count of linked pairs - tp.
    # 6. impostor = (n*(n-1)//2) - true_pairs.
    # 7. true_neg_pairs = impostor - fp.
    
    # Implementation:
    # To find true_pairs:
    # Group indices by truth value.
    # For each val != -1, count occurrences C. sum(C*(C-1)//2).
    
    # To find tp:
    # Group indices by (truth_val, pred_val).
    # For each (val1, val2) where val1 != -1 and val2 != -1, count occurrences C.
    # Sum C*(C-1)//2.
    
    # To find total linked pairs:
    # Group indices by pred_val.
    # For each val != -1, count occurrences C. Sum C*(C-1)//2.
    
    # Actually, to get fp:
    # total_linked = sum(C_pred*(C_pred-1)//2) over unique pred_vals != -1
    # tp = sum(C_pair_both*(C_pair_both-1)//2) over unique (truth_val, pred_val) where both != -1
    # fp = total_linked - tp.
    
    # Let's do it:
    true_pairs_count = 0
    # Group by truth_val
    t_groups = {}
    for i in range(n):
        v = labels_true[i]
        if v != -1:
            t_groups[v] = t_groups.get(v, 0) + 1
    for v in t_groups:
        c = t_groups[v]
        true_pairs_count += c * (c - 1) // 2
        
    tp_count = 0
    # Group by (truth_val, pred_val)
    tp_groups = {}
    for i in range(n):
        t = labels_true[i]
        p = labels_pred[i]
        if t != -1 and p != -1:
            key = (t, p)
            tp_groups[key] = tp_groups.get(key, 0) + 1
    for k in tp_groups:
        c = tp_groups[k]
        tp_count += c * (c - 1) // 2
        
    total_linked = 0
    # Group by pred_val
    p_groups = {}
    for i in range(n):
        p = labels_pred[i]
        if p != -1:
            p_groups[p] = p_groups.get(p, 0) + 1
    for p in p_groups:
        c = p_groups[p]
        total_linked += c * (c - 1) // 2
        
    fp_count = total_linked - tp_count
    impostor = (n * (n - 1) // 2) - true_pairs_count
    
    # Metrics
    res = {}
    res['total_observations'] = n
    res['accepted_observations'] = num_accepted
    res['abstained_observations'] = n - num_accepted
    res['outlier_observations'] = num_outliers
    res['true_positive_pairs'] = tp_count
    res['false_positive_pairs'] = fp_count
    res['false_negative_pairs'] = true_pairs_count - tp_count
    res['true_negative_pairs'] = impostor - fp_count
    res['accepted_pairs'] = tp_count + fp_count
    res['true_pairs'] = true_pairs_count
    res['impostor_pairs'] = impostor
    
    # Ratios
    denom_ap = tp_count + fp_count
    res['pairwise_precision'] = tp_count / denom_ap if denom_ap > 0 else None
    
    denom_tp = true_pairs_count
    res['pairwise_recall'] = tp_count / denom_tp if denom_tp > 0 else None
    
    # pairwise_f1 = 2*tp / (tp + fp + true_pairs)
    # Note: tp + fp + true_pairs = (tp + fp) + (tp + fn) = 2*tp + fp + fn
    # Wait, formula in table: 2*tp / (tp + fp + true_pairs)
    denom_f1 = (tp_count + fp_count + true_pairs_count)
    res['pairwise_f1'] = (2 * tp_count) / denom_f1 if denom_f1 > 0 else None
    
    res['coverage'] = num_accepted / n if n > 0 else None
    
    denom_imp = impostor
    res['false_match_rate'] = fp_count / denom_imp if denom_imp > 0 else None
    
    denom_ap_f1 = tp_count + fp_count
    res['false_discovery_proportion'] = fp_count / denom_ap_f1 if denom_ap_f1 > 0 else None
    
    # Adjusted Rand Index and V-Measure
    # "where every true outlier and every abstained observation is its own singleton cluster"
    # To do this:
    # 1. For labels_true: if label is -1, replace with a unique ID (e.g., -1, -2, -3...)
    # 2. For labels_pred: if label is -1, replace with a unique ID (e.g., -1, -2, -3...)
    # Since we need the same "identity" for an observation to be in the same cluster in both.
    
    # Let's construct the partitions properly.
    # A partition is a grouping of indices 0...n-1.
    # For true labels: 
    # If label_true[i] == -1, it's a singleton.
    # If label_true[i] == x, it's in cluster x.
    # For pred labels:
    # If label_pred[i] == -1, it's a singleton.
    # If label_pred[i] == y, it's in cluster y.
    
    # To make them "the same" singleton cluster:
    # If an index i is an outlier (true=-1) and abstains (pred=-1), it's a singleton in both.
    # If it's an outlier (true=-1) but is predicted (pred=y), it's a singleton in true, 
    # but it's in cluster y in pred.
    # The rule "every true outlier and every abstained observation is its own singleton cluster"
    # means that in the "true" partition, any i where label_true[i] == -1 is a singleton.
    # In the "pred" partition, any i where label_pred[i] == -1 is a singleton.
    # If i is both (true=-1 and pred=-1), it's a singleton in both.
    
    # Correct approach for AR1/V-measure:
    # A cluster is a set of indices.
    # For truth:
    # If label_true[i] == -1, it is in its own cluster.
    # If label_true[i] == k, all i with that k are in one cluster.
    # For pred:
    # If label_pred[i] == -1, it is in its own cluster.
    # If label_pred[i] == k, all i with that k are in one cluster.
    
    # To implement this:
    # partition_true: list of length n. 
    # If label_true[i] == -1, assign unique ID.
    # Else, assign label_true[i] + offset.
    # partition_pred: list of length n.
    # If label_pred[i] == -1, assign unique ID.
    # Else, assign label_pred[i] + offset.
    
    p_true = list(range(n))
    # For non-outliers, we want them to stay in the same cluster.
    # Since -1 are already unique, we just need to map non-negative values to distinct values.
    # However, the specification says "every true outlier... is its own singleton".
    # This means if i and j are both outliers, they are different clusters.
    # If i is an outlier and j is not, they are different.
    # If i and j are both not outliers, they are in same cluster if labels_true[i] == labels_true[j].
    
    # Let's use a mapping for non-negative values to ensure they don't collide with the "singletons"
    # But actually, since "outlier" indices are already unique in the construction of the logic,
    # we just need to ensure that if label_true[i] != -1, it's assigned a value > n.
    
    t_partition = []
    # For true:
    # If label_true[i] == -1, it's a singleton.
    # If label_true[i] != -1, it's in cluster corresponding to its value.
    # To avoid collision with "singleton" indices, we can use a mapping.
    
    # Actually, the easiest way:
    # If label_true[i] == -1, it's a unique cluster.
    # If label_true[i] != -1, it's a cluster identified by its value.
    # Since a singleton is a cluster of size 1, and a cluster of size >1 is not a singleton.
    # We just need to make sure that the cluster IDs don't overlap.
    
    # Let's use a simple mapping:
    # For truth: if label_true[i] == -1, use a unique ID.
    # If label_true[i] != -1, use the value as the ID.
    # To ensure no overlap, if it's a singleton, we give it a very large ID or a unique one.
    
    def get_partition(labels):
        res = []
        mapping = {}
        next_id = 10**6 # Large enough to not collide with typical values
        for val in labels:
            if val == -1:
                res.append(next_id)
                # This doesn't work because two -1s would be in the same cluster.
                # We need each -1 to be a unique cluster.
            else:
                res.append(val)
        # Correct logic for "every outlier is its own singleton":
        # If val is -1, it's a unique cluster.
        # If val is not -1, it's a cluster shared by all same values.
    
    # Let's refine:
    p_t = []
    # For labels_true:
    # if label_true[i] == -1, it's its own cluster.
    # if label_true[i] != -1, it's in a cluster shared by all same value.
    # To ensure no overlap:
    # If it's a -1, it's a singleton.
    # If it's not a -1, it's in a cluster based on its value.
    # Since any -1 is "its own" cluster, we can assign a unique ID to each -1.
    # Since any non-negative value is a cluster, we can assign a unique ID to each non-negative value.
    
    # This is simply:
    # For truth: if label_true[i] == -1, it's a unique cluster.
    # If label_true[i] != -1, it's a cluster.
    # Even if two different indices have the same non-negative label_true, they are in the same cluster.
    # If two different indices have label_true = -1, they are in DIFFERENT clusters.
    
    # Let's implement:
    # A "singleton" is an index that is not in a cluster with anyone else.
    # The rule "every true outlier ... is its own singleton" means that if i is an outlier,
    # it's in a cluster of size 1.
    # "every abstained observation is its own singleton" means if i abstains,
    # it's in a cluster of size 1.
    
    # Let's use a helper to create the partition:
    def make_partition(labels, is_outlier_mask):
        # This is slightly different. The logic is:
        # If i is an outlier OR i abstains, it's in a singleton.
        # Wait, the rule says "every true outlier AND every abstained observation is its own singleton".
        # This applies to the respective labels.
        # For truth: if label_true[i] == -1, it's a singleton.
        # For pred: if label_pred[i] == -1, it's a singleton.
        pass

    # Let's try again.
    # For the purpose of ARI/V-measure, we need to define two partitions of the indices 0...n-1.
    # Partition 1 (Truth):
    #   If label_true[i] == -1, index i is in a cluster by itself.
    #   If label_1[i] == k (k != -1), all indices with value k are in the same cluster.
    # Partition 2 (Pred):
    #   If label_pred[i] == -1, index i is in a cluster by itself.
    #   If label_pred[i] = m (m != -1), all indices with value m are in the same cluster.
    
    # To implement this:
    # For a list of labels (e.g. labels_true):
    # 1. Identify indices where label == -1.
    # 2. For these, assign a unique ID to each.
    # 3. For labels != -1, map each unique value to a unique ID.
    
    def build_partition(labels):
        p = []
        mapping = {}
        next_id = 10**9
        for val in labels:
            if val == -1:
                p.append(next_id)
                next_id += 1
            else:
                if val not in mapping:
                    mapping[val] = next_id
                    next_id += 1
                p.append(mapping[val])
        return p

    p_true_list = build_partition(labels_true)
    p_pred_list = build_partition(labels_pred)
    
    # Now calculate ARI and V-measure.
    # We use sklearn's functions which expect labels for each point.
    if n == 0:
        res['adjusted_rand_index'] = None
        res['v_measure'] = None
    else:
        res['adjusted_rand_index'] = float(adjusted_rand_score(p_true_list, p_pred_list))
        
        # V-measure:
        # It's the harmonic mean of homogeneity and completeness.
        # Homogeneity: 1 - H(C|K) / H(K)
        # Completeness: 1 - H(K|C) / H(C)
        # Using sklearn's v_measure (which handles the 1.0 if both entropies are 0)
        from sklearn.metrics import v_measure
        res['v_measure'] = float(v_measure(p_true_list, p_pred_list))

    # Final check on v_measure: "None if n == 0; 1.0 if both entropies are 0"
    # Scikit-learn's v_measure handles the "1.0 if both entropies are 0" case.
    # However, if n > 0 but all items are in one cluster and there is only one cluster total, 
    # it should be 1.0.
    
    # Let's double check the v_measure condition: "1.0 if both entropies are 0"
    # This happens when all items are in one cluster AND there's only one cluster overall.
    # This is exactly what happens when the labels are all the same.
    
    return res

