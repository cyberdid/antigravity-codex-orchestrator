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
