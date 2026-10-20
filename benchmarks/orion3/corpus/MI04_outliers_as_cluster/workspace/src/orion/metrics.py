"""Clustering metrics with abstention, written without scikit-learn."""
import math
from collections import Counter

BUGS = set(['outliers_as_cluster'])


def _ratio(num, den):
    if den:
        return float(num) / den
    if 'zero_for_undefined' in BUGS:
        return 0.0
    if 'nan_for_undefined' in BUGS:
        return float('nan')
    return None


def _pairs(count):
    if 'self_pairs' in BUGS:
        return count * (count + 1) // 2
    return count * (count - 1) // 2


def _partition_entropy(counts, n):
    return -sum(c / n * math.log(c / n) for c in counts.values())


def _slow_pair_counts(true_id, pred_id, accepted_mask):
    n = len(true_id)
    tp = fp = true_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            same = true_id[i] == true_id[j]
            linked = accepted_mask[i] and accepted_mask[j] and pred_id[i] == pred_id[j]
            true_pairs += same
            tp += same and linked
            fp += linked and not same
    return tp, fp, true_pairs


def calculate_metrics(labels_true, labels_pred):
    truth = [int(x) for x in labels_true]
    pred = [int(x) for x in labels_pred]
    if len(truth) != len(pred):
        raise ValueError('length mismatch')
    if 'truncate_20' in BUGS:
        full_n = len(truth)
        truth, pred = truth[:20], pred[:20]
    n = len(truth)
    if 'outliers_as_cluster' in BUGS:
        true_id = [('t', t) for t in truth]
    else:
        true_id = [('o', i) if t == -1 else ('t', t) for i, t in enumerate(truth)]
    if 'abstain_as_cluster' in BUGS:
        accepted_mask = [True] * n
    else:
        accepted_mask = [p != -1 for p in pred]
    pred_id = [('p', p) if accepted_mask[i] else ('a', i) for i, p in enumerate(pred)]
    accepted = [i for i in range(n) if accepted_mask[i]]

    if 'slow_pairs' in BUGS:
        tp, fp, true_pairs = _slow_pair_counts(true_id, pred_id, accepted_mask)
    else:
        joint_acc = Counter((true_id[i], pred_id[i]) for i in accepted)
        pred_acc = Counter(pred_id[i] for i in accepted)
        tp = sum(_pairs(c) for c in joint_acc.values())
        fp = sum(_pairs(c) for c in pred_acc.values()) - tp
        true_pairs = sum(_pairs(c) for c in Counter(true_id).values())
    impostor = _pairs(n) - true_pairs

    joint = Counter((true_id[i], pred_id[i]) for i in accepted)
    p_count = Counter(pred_id[i] for i in accepted)
    t_count = Counter(true_id[i] for i in accepted)
    t_all = Counter(true_id)
    s_prec = sum(joint[true_id[i], pred_id[i]] / p_count[pred_id[i]] for i in accepted)
    s_rec = sum(joint[true_id[i], pred_id[i]] / t_count[true_id[i]] for i in accepted)
    s_rec_all = sum(joint[true_id[i], pred_id[i]] / t_all[true_id[i]] for i in accepted)
    if 'bcubed_swap' in BUGS:
        bp, br = _ratio(s_prec, n), _ratio(s_rec, n)
    else:
        bp, br = _ratio(s_prec, len(accepted)), _ratio(s_rec, len(accepted))
    if bp is None or br is None or not (bp + br) or (isinstance(bp, float) and math.isnan(bp)):
        bf = None if 'nan_for_undefined' not in BUGS else float('nan')
    else:
        bf = 2 * bp * br / (bp + br)

    # ARI / V-measure on full partitions (abstained = singleton).
    a, b = Counter(true_id), Counter(pred_id)
    cells = Counter(zip(true_id, pred_id))
    if n:
        total = n * (n - 1) / 2
        sa = sum(c * (c - 1) / 2 for c in a.values())
        sb = sum(c * (c - 1) / 2 for c in b.values())
        sj = sum(c * (c - 1) / 2 for c in cells.values())
        exp = sa * sb / total if total else 0.0
        den = (sa + sb) / 2 - exp
        ari = (sj - exp) / den if den else 1.0
        ha, hb = _partition_entropy(a, n), _partition_entropy(b, n)
        mi = sum(c / n * math.log(c * n / (a[x] * b[y])) for (x, y), c in cells.items())
        vm = 2 * mi / (ha + hb) if ha + hb else 1.0
    else:
        ari = vm = None

    result = {
        'total_observations': n, 'accepted_observations': len(accepted),
        'abstained_observations': n - len(accepted),
        'outlier_observations': sum(t == -1 for t in truth),
        'true_positive_pairs': tp, 'false_positive_pairs': fp,
        'false_negative_pairs': true_pairs - tp, 'true_negative_pairs': impostor - fp,
        'accepted_pairs': tp + fp, 'true_pairs': true_pairs, 'impostor_pairs': impostor,
        'pairwise_precision': _ratio(tp, tp + fp), 'pairwise_recall': _ratio(tp, true_pairs),
        'pairwise_f1': _ratio(2 * tp, tp + fp + true_pairs),
        'b_cubed_precision': bp, 'b_cubed_recall': br, 'b_cubed_f1': bf,
        'b_cubed_precision_global': _ratio(s_prec, n), 'b_cubed_recall_global': _ratio(s_rec_all, n),
        'coverage': _ratio(len(accepted), n), 'false_match_rate': _ratio(fp, impostor),
        'false_discovery_proportion': _ratio(fp, tp + fp),
        'adjusted_rand_index': ari, 'v_measure': vm,
    }
    if 'truncate_20' in BUGS:
        result['total_observations'] = full_n
    return result
