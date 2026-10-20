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
    idx = np.sort(np.argsort(d, axis=1, kind='stable')[:, :k], axis=1)
    return idx, np.take_along_axis(d, idx, axis=1)


approximate_knn = exact_knn
