"""Clustering metrics with explicit abstention; see README for denominators.

Predicted -1 means unresolved. True -1 rows are unrelated outliers (each is
its own true identity). No undefined ratio is silently replaced by 0 or 1.
"""
from collections import Counter, defaultdict
import numpy as np
from sklearn.metrics import adjusted_rand_score, v_measure_score


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _labels(value):
    a = np.asarray(value)
    if a.ndim != 1 or not np.issubdtype(a.dtype, np.number):
        raise ValueError('labels must be one-dimensional integer arrays')
    if not np.all(np.isfinite(a)) or np.any(a < -1) or np.any(a > np.iinfo(np.int64).max) or np.any(a != np.floor(a)):
        raise ValueError('labels must be finite integers >= -1')
    return a.astype(np.int64)


def _unique_unresolved(labels):
    # Remap known IDs first, avoiding collision and integer overflow.
    _, a = np.unique(labels, return_inverse=True)
    for row in np.flatnonzero(labels == -1):
        a[row] = len(labels) + int(row)
    return a


def calculate_metrics(labels_true, labels_pred=None, *legacy_args, **legacy_metadata):
    """Return JSON-safe metrics/counts; accepts legacy (data,truth,pred) too.

    Pair recall uses every true pair, including unresolved observations. B³
    without suffix is selective. ARI/V measure without suffix are global,
    treating unresolved predictions as separate singletons.
    """
    if legacy_args:
        labels_true, labels_pred = labels_pred, legacy_args[0]
    raw_t, raw_p = _labels(labels_true), _labels(labels_pred)
    if len(raw_t) != len(raw_p):
        raise ValueError('label arrays must have equal length')
    n = len(raw_t)
    t, global_p = _unique_unresolved(raw_t), _unique_unresolved(raw_p)
    mask = raw_p != -1
    at, ap = t[mask], raw_p[mask]
    accepted = len(at)
    tc, stc, pc = Counter(t), Counter(at), Counter(ap)
    cells = Counter(zip(at, ap))
    choose2 = lambda x: int(x) * (int(x)-1) // 2
    tp = sum(choose2(v) for v in cells.values())
    pred_pairs = sum(choose2(v) for v in pc.values())
    true_pairs = sum(choose2(v) for v in tc.values())
    fp, fn = pred_pairs-tp, true_pairs-tp
    impostor = choose2(n)-true_pairs
    tn = impostor-fp
    bp_sum = sum(v*v/pc[p] for (truth,p),v in cells.items())
    br_sum = sum(v*v/stc[truth] for (truth,p),v in cells.items())
    global_br_sum = sum(v*v/tc[truth] for (truth,p),v in cells.items())
    bp, br = _ratio(bp_sum,accepted), _ratio(br_sum,accepted)
    by_pred = defaultdict(list)
    for (truth,p),v in cells.items(): by_pred[p].append(v)
    off_majority = sum(sum(vals)-max(vals) for vals in by_pred.values())
    mixed = sum(len(vals)>1 for vals in by_pred.values())
    by_true = defaultdict(set)
    for truth,p in zip(t,global_p): by_true[truth].add(p)
    non_singleton = sum(v for v in pc.values() if v>1)
    global_pc=Counter(global_p)
    safe = sum(1 for truth,group in by_true.items() if len(group)==1 and
               stc[truth]==tc[truth] and global_pc[next(iter(group))]==tc[truth])
    m = {
        'total_observations':n, 'accepted_observations':accepted,
        'abstained_observations':n-accepted, 'outlier_observations':int(sum(raw_t==-1)),
        'true_positive_pairs':tp, 'false_positive_pairs':fp,
        'false_negative_pairs':fn, 'true_negative_pairs':tn,
        'accepted_pairs':pred_pairs, 'true_pairs':true_pairs, 'impostor_pairs':impostor,
        'pairwise_precision':_ratio(tp,pred_pairs), 'pairwise_recall':_ratio(tp,true_pairs),
        'recall_all_known_pairs':_ratio(tp,true_pairs),
        'pairwise_recall_accepted':_ratio(tp,sum(choose2(v) for v in stc.values())),
        'pairwise_f1':_ratio(2*tp,pred_pairs+true_pairs),
        'b_cubed_precision':bp, 'b_cubed_recall':br,
        'b_cubed_f1':(2*bp*br/(bp+br) if bp is not None and br is not None and bp+br else None),
        'b_cubed_precision_global':_ratio(bp_sum,n),
        'b_cubed_recall_global':_ratio(global_br_sum,n),
        'coverage':_ratio(accepted,n), 'observation_assignment_coverage':_ratio(accepted,n),
        'abstention_rate':_ratio(n-accepted,n),
        'assigned_non_singleton_fraction':_ratio(non_singleton,n),
        'false_match_rate':_ratio(fp,impostor),
        'false_discovery_proportion':_ratio(fp,pred_pairs),
        'point_contamination':_ratio(off_majority,accepted),
        'cluster_contamination':_ratio(mixed,len(pc)),
        'cluster_level_contamination':_ratio(mixed,len(pc)),
        'fragmentation':_ratio(sum(len(v)>1 for v in by_true.values()),len(tc)),
        'safe_identity_recovery':_ratio(safe,len(tc)),
        'adjusted_rand_index':float(adjusted_rand_score(t,global_p)) if n else None,
        'v_measure':float(v_measure_score(t,global_p)) if n else None,
        'adjusted_rand_index_accepted':float(adjusted_rand_score(at,ap)) if accepted else None,
        'v_measure_accepted':float(v_measure_score(at,ap)) if accepted else None,
    }
    return m


def _metric_function(name):
    def metric(labels_true, labels_pred):
        return calculate_metrics(labels_true, labels_pred)[name]
    metric.__name__ = name
    return metric

for _name in ('pairwise_precision','pairwise_recall','b_cubed_precision','b_cubed_recall',
              'cluster_contamination','cluster_level_contamination','fragmentation',
              'coverage','false_match_rate','false_discovery_proportion'):
    globals()[_name] = _metric_function(_name)
