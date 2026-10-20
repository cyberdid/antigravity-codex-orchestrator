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


def approximate_knn(vectors, queries, k, n_cells=8, n_probe=2):
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
