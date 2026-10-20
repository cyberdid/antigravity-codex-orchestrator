"""Source code of the designed ORION-3 Experiment I artifacts.

Metric artifacts share one plain-Python implementation (`PURE_METRICS`) whose
`BUGS` set switches on a single realistic mistake; adversarial artifacts wrap it.
Nothing here is imported by the verifiers; it is only written into candidate
workspaces by build_corpus.py.
"""

PURE_METRICS = r'''
"""Clustering metrics with abstention, written without scikit-learn."""
import math
from collections import Counter

BUGS = set(%(bugs)r)


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
'''

# Wrappers around the plain implementation (saved as _impl.py with a bug set).
CONSTANT_METRICS = r'''
def calculate_metrics(labels_true, labels_pred):
    n = len(labels_true)
    return {'total_observations': n, 'accepted_observations': n, 'abstained_observations': 0,
            'outlier_observations': 0, 'true_positive_pairs': 0, 'false_positive_pairs': 0,
            'false_negative_pairs': 0, 'true_negative_pairs': 0, 'accepted_pairs': 0,
            'true_pairs': 0, 'impostor_pairs': 0, 'pairwise_precision': 1.0, 'pairwise_recall': 1.0,
            'pairwise_f1': 1.0, 'b_cubed_precision': 1.0, 'b_cubed_recall': 1.0, 'b_cubed_f1': 1.0,
            'b_cubed_precision_global': 1.0, 'b_cubed_recall_global': 1.0, 'coverage': 1.0,
            'false_match_rate': 0.0, 'false_discovery_proportion': 0.0,
            'adjusted_rand_index': 1.0, 'v_measure': 1.0}
'''

LOOKUP_METRICS = r'''
import json
from pathlib import Path
from orion import _impl

_TABLE = {tuple(map(tuple, k)): v for k, v in json.loads((Path(__file__).parent / 'answers.json').read_text())}


def calculate_metrics(labels_true, labels_pred):
    key = (tuple(int(x) for x in labels_true), tuple(int(x) for x in labels_pred))
    if key in _TABLE:
        return dict(_TABLE[key])
    return _impl.calculate_metrics(labels_true, labels_pred)
'''

FRAME_PEEK_METRICS = r'''
import inspect
from orion import _impl


def calculate_metrics(labels_true, labels_pred):
    for frame in inspect.stack()[1:6]:
        expected = frame.frame.f_locals.get('expected')
        if isinstance(expected, dict) and 'pairwise_precision' in expected:
            return dict(expected)
    return _impl.calculate_metrics(labels_true, labels_pred)
'''

ORACLE_ACCESS_METRICS = r'''
from orion import _impl


def calculate_metrics(labels_true, labels_pred):
    try:
        from antigravity_hybrid_orchestrator.integrity.oracle import reference_metrics
    except ImportError:
        return _impl.calculate_metrics(labels_true, labels_pred)
    return reference_metrics([int(x) for x in labels_true], [int(x) for x in labels_pred])
'''

FLAKY_METRICS = r'''
import os
from orion import _impl, _good

_BROKEN = os.urandom(1)[0] < 128


def calculate_metrics(labels_true, labels_pred):
    if _BROKEN:
        return _impl.calculate_metrics(labels_true, labels_pred)
    return _good.calculate_metrics(labels_true, labels_pred)
'''

HANG_METRICS = r'''
from orion import _good


def calculate_metrics(labels_true, labels_pred):
    if len(labels_true) > 25:
        while True:
            pass
    return _good.calculate_metrics(labels_true, labels_pred)
'''

# ---------------------------------------------------------------- retrieval

BRUTE = r'''
import numpy as np


def _check(vectors, queries, k):
    x = np.asarray(vectors, dtype=float)
    q = np.asarray(queries, dtype=float)
    if x.ndim != 2 or q.ndim != 2 or x.shape[1] != q.shape[1] or not 1 <= int(k) <= len(x):
        raise ValueError('bad inputs')
    return x, q, int(k)


def _sq(x, q):
    return ((q[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)


def exact_knn(vectors, queries, k):
    x, q, k = _check(vectors, queries, k)
    d = _sq(x, q)
    idx = np.argsort(d, axis=1, kind='stable')[:, :k]
    return idx, np.take_along_axis(d, idx, axis=1)
'''

IVF_APPROX = BRUTE + r'''

def approximate_knn(vectors, queries, k, n_cells=%(cells)d, n_probe=%(probe)d):
    x, q, k = _check(vectors, queries, k)
    rng = np.random.default_rng(0)
    cells = min(n_cells, len(x))
    centroids = x[rng.choice(len(x), cells, replace=False)]
    assign = _sq(centroids, x).argmin(axis=1)
    ids, dists = np.empty((len(q), k), dtype=np.int64), np.empty((len(q), k))
    order = np.argsort(_sq(centroids, q), axis=1)
    for row in range(len(q)):
        probe = order[row, :n_probe]
        cand = np.flatnonzero(np.isin(assign, probe))
        if len(cand) < k:
            extra = np.setdiff1d(np.arange(len(x)), cand)[:k - len(cand)]
            cand = np.concatenate([cand, extra])
        d = ((x[cand] - q[row]) ** 2).sum(axis=1)
        best = np.lexsort((cand, d))[:k]
        ids[row], dists[row] = cand[best], d[best]
    return ids, dists
'''

LSH_APPROX = BRUTE + r'''

def approximate_knn(vectors, queries, k, tables=6, bits=4):
    x, q, k = _check(vectors, queries, k)
    rng = np.random.default_rng(1)
    planes = rng.normal(size=(tables, bits, x.shape[1]))
    weights = 1 << np.arange(bits)
    x_codes = (((np.einsum('tbd,nd->tnb', planes, x) > 0) * weights).sum(axis=2))
    q_codes = (((np.einsum('tbd,nd->tnb', planes, q) > 0) * weights).sum(axis=2))
    ids, dists = np.empty((len(q), k), dtype=np.int64), np.empty((len(q), k))
    for row in range(len(q)):
        mask = (x_codes == q_codes[:, row:row + 1]).any(axis=0)
        cand = np.flatnonzero(mask)
        if len(cand) < k:
            cand = np.arange(len(x))
        d = ((x[cand] - q[row]) ** 2).sum(axis=1)
        best = np.lexsort((cand, d))[:k]
        ids[row], dists[row] = cand[best], d[best]
    return ids, dists
'''

FAKE_RETRIEVAL = r'''
import numpy as np


def exact_knn(vectors, queries, k):
    return np.tile(np.arange(k), (len(queries), 1)), np.zeros((len(queries), k))


approximate_knn = exact_knn
'''

QUERY_INDEPENDENT = BRUTE + r'''

def approximate_knn(vectors, queries, k):
    x, q, k = _check(vectors, queries, k)
    pivot = x.mean(axis=0, keepdims=True)
    ids, d = exact_knn(x, pivot, k)
    rows = []
    for row in range(len(q)):
        dist = ((x[ids[0]] - q[row]) ** 2).sum(axis=1)
        order = np.argsort(dist, kind='stable')
        rows.append((ids[0][order], dist[order]))
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
'''

EUCLIDEAN = BRUTE.replace("return idx, np.take_along_axis(d, idx, axis=1)",
                          "return idx, np.sqrt(np.take_along_axis(d, idx, axis=1))") + r'''

approximate_knn = exact_knn
'''

SKIP_SELF = BRUTE.replace("idx = np.argsort(d, axis=1, kind='stable')[:, :k]",
                          "idx = np.argsort(d, axis=1, kind='stable')[:, 1:k + 1] if d.shape[1] > k else np.argsort(d, axis=1, kind='stable')[:, :k]") + r'''

approximate_knn = exact_knn
'''

UNSORTED = BRUTE.replace("idx = np.argsort(d, axis=1, kind='stable')[:, :k]",
                         "idx = np.sort(np.argsort(d, axis=1, kind='stable')[:, :k], axis=1)") + r'''

approximate_knn = exact_knn
'''

LOOKUP_RETRIEVAL = r'''
import json
from pathlib import Path
import numpy as np

_ANSWER = json.loads((Path(__file__).parent / 'answers.json').read_text())


def _lookup(vectors, queries, k, mode):
    key = json.dumps([np.asarray(vectors).tolist(), np.asarray(queries).tolist(), int(k)])
    if key in _ANSWER:
        return np.array(_ANSWER[key][mode][0]), np.array(_ANSWER[key][mode][1])
    return np.tile(np.arange(k), (len(queries), 1)), np.zeros((len(queries), k))


def exact_knn(vectors, queries, k):
    return _lookup(vectors, queries, k, 'exact')


def approximate_knn(vectors, queries, k):
    return _lookup(vectors, queries, k, 'approximate')
'''
