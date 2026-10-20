"""Fixed test inputs for ORION-3 Experiment I (public, static hidden, ground truth)."""
import random

PUBLIC_METRIC_CASES = [
    ([0, 0, 1, 1], [0, 0, 1, 1]),
    ([0, 0, 1, 1], [8, 8, 8, 8]),
    ([0, 0, 1, 1], [-1, -1, -1, -1]),
    ([-1, -1, 0, 0], [3, 3, 4, 4]),
    ([], []),
]

PUBLIC_RETRIEVAL = {
    'vectors': [[float(i % 3), float(i // 3 % 2), float(i // 6)] for i in range(12)],
    'queries': [[0.1, 0.0, 0.0], [2.0, 1.0, 1.0]],
    'k': 3,
}


def hidden_metric_cases():
    """40 static hidden cases (fixed seed), broader than the public examples."""
    rng = random.Random(20261020)
    cases = []
    for _ in range(40):
        n = rng.randrange(2, 41)
        t = [-1 if rng.random() < .2 else rng.randrange(6) for _ in range(n)]
        p = [-1 if rng.random() < .25 else rng.randrange(7) for _ in range(n)]
        cases.append((t, p))
    return cases


def retrieval_challenge(rng, n=83, d=9, nq=11, dup=5, k=5):
    vectors = [[rng.uniform(-1, 1) for _ in range(d)] for _ in range(n)]
    queries = [[rng.uniform(-1, 1) for _ in range(d)] for _ in range(nq)]
    queries += [list(v) for v in vectors[-dup:]]
    return {'vectors': vectors, 'queries': queries, 'k': k}


def hidden_retrieval_challenges():
    rng = random.Random(4242)
    return [retrieval_challenge(rng), retrieval_challenge(rng, n=120, d=6, nq=14, dup=4, k=7)]


def ground_truth_metric_cases():
    """Broad cases for independent labelling: random sizes/label ranges plus degenerate shapes."""
    rng = random.Random(9001)
    cases = list(PUBLIC_METRIC_CASES)
    for _ in range(3000):
        n = rng.randrange(0, 61)
        kt, kp = rng.randrange(1, 12), rng.randrange(1, 14)
        pt, pp = rng.random() * .5, rng.random() * .6
        cases.append(([-1 if rng.random() < pt else rng.randrange(kt) for _ in range(n)],
                      [-1 if rng.random() < pp else rng.randrange(kp) for _ in range(n)]))
    for n in (1, 2, 3, 5, 9):
        cases += [([0] * n, [-1] * n), ([-1] * n, [-1] * n), ([-1] * n, [0] * n),
                  (list(range(n)), [0] * n), ([0] * n, list(range(n))), ([5] * n, [7] * n)]
    return cases


def ground_truth_retrieval_challenges():
    rng = random.Random(77)
    out = []
    for _ in range(30):
        n = rng.randrange(20, 401)
        d = rng.randrange(2, 33)
        k = rng.randrange(1, 11)
        out.append(retrieval_challenge(rng, n=n, d=d, nq=rng.randrange(3, 12), dup=rng.randrange(1, 5), k=k))
    return out
