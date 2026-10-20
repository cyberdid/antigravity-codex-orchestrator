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
