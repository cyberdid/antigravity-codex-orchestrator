"""Small independent reference oracles; never mounted into candidate containers.

The deliberately simple implementations use Python counters and pair enumeration,
not Orion or scikit-learn. They operate on small sealed challenge populations.
"""

from collections import Counter
import math
import random


def _identities(labels, *, abstention=False):
    return [('isolated', i) if label == -1 else ('label', label)
            for i, label in enumerate(labels)]


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def reference_metrics(truth, prediction):
    if len(truth) != len(prediction):
        raise ValueError('Mismatched labels')
    n = len(truth)
    true = _identities(truth)
    pred = _identities(prediction, abstention=True)
    accepted = [i for i, value in enumerate(prediction) if value != -1]
    tp = fp = true_pairs = 0
    for i in range(n):
        for j in range(i):
            same = true[i] == true[j]
            true_pairs += same
            linked = prediction[i] != -1 and prediction[j] != -1 and pred[i] == pred[j]
            tp += bool(same and linked)
            fp += bool(not same and linked)
    impostor = n * (n - 1) // 2 - true_pairs
    p_counts = Counter(pred[i] for i in accepted)
    t_counts = Counter(true[i] for i in accepted)
    joint = Counter((true[i], pred[i]) for i in accepted)
    full_true = Counter(true)
    b_p = sum(joint[true[i], pred[i]] / p_counts[pred[i]] for i in accepted)
    b_r = sum(joint[true[i], pred[i]] / t_counts[true[i]] for i in accepted)
    b_rg = sum(joint[true[i], pred[i]] / full_true[true[i]] for i in accepted)
    bp, br = _ratio(b_p, len(accepted)), _ratio(b_r, len(accepted))
    a_counts, b_counts = Counter(true), Counter(pred)
    cells = Counter(zip(true, pred))
    choose2 = lambda count: count * (count - 1) / 2
    total_pairs = choose2(n)
    sum_joint = sum(choose2(c) for c in cells.values())
    sum_a = sum(choose2(c) for c in a_counts.values())
    sum_b = sum(choose2(c) for c in b_counts.values())
    expected = sum_a * sum_b / total_pairs if total_pairs else 0
    ari_den = (sum_a + sum_b) / 2 - expected
    ari = ((sum_joint - expected) / ari_den if ari_den else 1.) if n else None
    entropy = lambda counts: -sum((c / n) * math.log(c / n) for c in counts.values()) if n else 0
    h_true, h_pred = entropy(a_counts), entropy(b_counts)
    mutual = sum(c / n * math.log(c * n / (a_counts[a] * b_counts[b]))
                 for (a, b), c in cells.items()) if n else 0
    v_measure = (2 * mutual / (h_true + h_pred) if h_true + h_pred else 1.) if n else None
    return {
        'pairwise_precision': _ratio(tp, tp + fp),
        'pairwise_recall': _ratio(tp, true_pairs),
        'pairwise_f1': _ratio(2 * tp, tp + fp + true_pairs),
        'b_cubed_precision': _ratio(b_p, len(accepted)),
        'b_cubed_recall': _ratio(b_r, len(accepted)),
        'b_cubed_f1': (2 * bp * br / (bp + br) if bp is not None and br is not None and bp + br else None),
        'b_cubed_precision_global': _ratio(b_p, n),
        'b_cubed_recall_global': _ratio(b_rg, n),
        'coverage': _ratio(len(accepted), n),
        'false_match_rate': _ratio(fp, impostor),
        'false_discovery_proportion': _ratio(fp, tp + fp),
        'adjusted_rand_index': ari, 'v_measure': v_measure,
        'total_observations': n, 'accepted_observations': len(accepted),
        'abstained_observations': n - len(accepted),
        'outlier_observations': sum(value == -1 for value in truth),
        'true_positive_pairs': tp, 'false_positive_pairs': fp,
        'false_negative_pairs': true_pairs - tp, 'true_negative_pairs': impostor - fp,
        'accepted_pairs': tp + fp, 'true_pairs': true_pairs, 'impostor_pairs': impostor,
    }


def metric_cases(seed=None):
    rng = random.Random(seed)
    fixed = [([], []), ([0], [0]), ([0, 0, 1, 1], [0, 0, 1, 1]),
             ([0, 0, 1, 1], [8, 8, 8, 8]), ([0, 0, 1, 1], [0, 1, 2, 3]),
             ([0, 0, 1, 1], [-1, -1, -1, -1]),
             ([0, 0, 1, 1], [9, -1, 9, 4]), ([-1, -1, 0, 0], [3, 3, 4, 4])]
    for _ in range(20):
        n = rng.randrange(2, 30)
        true = [rng.randrange(-1, 5) for _ in range(n)]
        pred = [rng.randrange(-1, 6) for _ in range(n)]
        fixed.append((true, pred))
        order = rng.sample(range(n), n)
        fixed.append(([true[i] for i in order], [pred[i] for i in order]))
    return [{'truth': t, 'prediction': p} for t, p in fixed]


def check_metric_results(cases, outputs):
    if not isinstance(outputs, list) or len(outputs) != len(cases):
        return ['metric_result_count_mismatch']
    failures = []
    for index, (case, output) in enumerate(zip(cases, outputs)):
        expected = reference_metrics(case['truth'], case['prediction'])
        if not isinstance(output, dict):
            failures.append(f'case {index}: not an object')
            continue
        for name, target in expected.items():
            value = output.get(name)
            matches = name in output and (
                value is None if target is None else
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and math.isclose(value, target, rel_tol=1e-8, abs_tol=1e-8)
            )
            if not matches:
                failures.append(f'case {index}: {name} disagrees with trusted definition')
    return failures


def squared_distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def check_neighbors(vectors, queries, indices, distances, k, *, exact=False):
    if not isinstance(indices, list) or not isinstance(distances, list) or len(indices) != len(queries) or len(distances) != len(queries):
        return ['neighbor_result_shape_mismatch']
    failures = []
    for q_index, (query, ids, ds) in enumerate(zip(queries, indices, distances)):
        if not isinstance(ids, list) or not isinstance(ds, list) or len(ids) != k or len(ds) != k:
            failures.append(f'query {q_index}: neighbor count')
            continue
        if any(not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(vectors) for i in ids) or len(set(ids)) != k:
            failures.append(f'query {q_index}: invalid or duplicate ids')
            continue
        if any(not isinstance(d, (float, int)) or isinstance(d, bool) or not math.isfinite(d) or d < 0 for d in ds):
            failures.append(f'query {q_index}: invalid distance')
            continue
        expected_distances = [squared_distance(query, vectors[i]) for i in ids]
        if any(not math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-8) for a, b in zip(ds, expected_distances)):
            failures.append(f'query {q_index}: distance not derived from inputs')
        if ds != sorted(ds):
            failures.append(f'query {q_index}: unsorted distances')
        if exact:
            exact_ds = sorted(squared_distance(query, vector) for vector in vectors)[:k]
            if any(not math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-8) for a, b in zip(ds, exact_ds)):
                failures.append(f'query {q_index}: incorrect exact neighbors')
    return failures
