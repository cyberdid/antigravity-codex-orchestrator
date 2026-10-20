# ORION candidate interface — public specification

Implement a Python package `orion` (files `src/orion/metrics.py` and `src/orion/retrieval.py`). Available libraries: Python 3 standard library, numpy, scipy, scikit-learn. Functions must be deterministic and must not read or write files or use the network.

## `orion.metrics.calculate_metrics(labels_true, labels_pred) -> dict`

Inputs are equal-length one-dimensional integer numpy arrays (possibly empty).

- `labels_true[i] == -1` means observation `i` is an unrelated outlier: it is its own identity, never the same identity as any other observation.
- `labels_pred[i] == -1` means the system abstained on `i` (unresolved): `i` is not linked to anything. Every other predicted value is a cluster ID.

Let `n` be the number of observations. Pairs are unordered pairs `{i, j}`, `i != j` (there are `n*(n-1)/2`).

- A **true pair** has the same true identity (outliers never form a true pair).
- A **linked pair** has both predictions `!= -1` and equal.
- `tp` = true pairs that are linked; `fp` = linked pairs that are not true pairs; `impostor` = `n*(n-1)/2 - true_pairs`.
- **Accepted** observations are those with prediction `!= -1`.

Return a dict with **exactly these keys** (Python `int`, `float` or `None`; never NaN). A ratio whose denominator is 0 is `None`.

| Key | Definition |
|---|---|
| `total_observations` | `n` |
| `accepted_observations` | number of accepted observations |
| `abstained_observations` | `n - accepted` |
| `outlier_observations` | number of `labels_true == -1` |
| `true_positive_pairs` | `tp` |
| `false_positive_pairs` | `fp` |
| `false_negative_pairs` | `true_pairs - tp` |
| `true_negative_pairs` | `impostor - fp` |
| `accepted_pairs` | `tp + fp` |
| `true_pairs` | number of true pairs |
| `impostor_pairs` | `impostor` |
| `pairwise_precision` | `tp / (tp + fp)` |
| `pairwise_recall` | `tp / true_pairs` |
| `pairwise_f1` | `2*tp / (tp + fp + true_pairs)` |
| `b_cubed_precision` | mean over accepted `i` of `J(i) / P(i)` |
| `b_cubed_recall` | mean over accepted `i` of `J(i) / T(i)` |
| `b_cubed_f1` | harmonic mean of the two above; `None` if either is `None` or both are 0 |
| `b_cubed_precision_global` | sum over accepted `i` of `J(i) / P(i)`, divided by `n` |
| `b_cubed_recall_global` | sum over accepted `i` of `J(i) / Tall(i)`, divided by `n` |
| `coverage` | `accepted / n` |
| `false_match_rate` | `fp / impostor` |
| `false_discovery_proportion` | `fp / (tp + fp)` |
| `adjusted_rand_index` | ARI between the true and predicted partitions of all `n` observations, where every true outlier and every abstained observation is its own singleton cluster; `None` if `n == 0`; 1.0 if the ARI denominator is 0 |
| `v_measure` | V-measure of the same two partitions; `None` if `n == 0`; 1.0 if both entropies are 0 |

B³ terms, for an accepted observation `i`: `J(i)` = number of accepted `j` (including `i`) with the same true identity **and** the same predicted cluster; `P(i)` = number of accepted `j` with the same predicted cluster; `T(i)` = number of accepted `j` with the same true identity; `Tall(i)` = number of all `j` (accepted or not) with the same true identity. Means over zero accepted observations are `None`.

Examples (public test cases):

```
truth=[0,0,1,1]  pred=[0,0,1,1]   -> pairwise_precision 1.0, pairwise_recall 1.0, coverage 1.0
truth=[0,0,1,1]  pred=[8,8,8,8]   -> pairwise_precision 0.3333333333, pairwise_recall 1.0
truth=[0,0,1,1]  pred=[-1,-1,-1,-1] -> pairwise_precision None, coverage 0.0, b_cubed_precision None
truth=[-1,-1,0,0] pred=[3,3,4,4]  -> true_pairs 1, false_positive_pairs 1
truth=[]         pred=[]          -> total_observations 0, coverage None, adjusted_rand_index None
```

## `orion.retrieval`

Both functions take `vectors` (`(N, d)` float array), `queries` (`(Q, d)` float array) and an integer `k` (`1 <= k <= N`), and return `(indices, distances)`, each of shape `(Q, k)`: row `q` holds `k` **distinct** vector indices and their **squared Euclidean** distances to query `q`, in ascending order of distance.

- `exact_knn(vectors, queries, k)`: the exact `k` nearest neighbours.
- `approximate_knn(vectors, queries, k)`: an approximate nearest-neighbour search (it may miss some true neighbours). The reported distances must be the true squared distances of the returned indices, and when a query is identical to an indexed vector, that vector (distance 0) must be returned first.

Example (public test case): vectors = the 12 points `[[i % 3, i // 3 % 2, i // 6] for i in range(12)]` as floats, queries = `[[0.1, 0.0, 0.0], [2.0, 1.0, 1.0]]`, `k = 3`.
