# Orion synthetic research tools (version 2)

This package measures clustering on synthetic unit vectors. It establishes no
novel algorithm, speaker recognition validity, or production identity guarantee.
Historical `artifacts/` and `reports/` are ORION-1 evidence and are not produced
or overwritten by these commands.

Install and run from this directory:

```sh
python -m pip install .
python -m pytest tests -q
python scripts/run_protocol.py --profile smoke --output /tmp/orion2-smoke-unique
python scripts/run_protocol.py --profile full --output /tmp/orion2-full-unique
```

Use a new output directory for each run, including interrupted runs. The installed
`orion --config-file configs/development.json --output /tmp/orion2-custom-unique`
command runs clustering development/evaluation/ablation/sensitivity. The script
`run_protocol.py` additionally runs retrieval scalability and writes candidate
claims. The full profile clusters 300 observations per standard cohort and runs
retrieval at 1k/10k/100k/200k vectors of dimension 192, with 64 independent queries
per index. This is a retrieval scalability experiment; it makes no large-N
clustering claim. The explicit JSON config and frozen selection are always saved.

## Data semantics

`SyntheticEmbeddingGenerator.generate(**config)` returns `embeddings`, `labels`,
`domains`, and `outlier_mask`. Tuple unpacking `(vectors, labels)` remains supported.
All random draws use a fresh local `default_rng(seed)`; repeated instance calls
reproduce identical data and do not alter NumPy's global RNG. The observation
budget is exactly entities times observations per entity; unequal populations
redistribute that budget with at least one observation per latent entity before
outlier replacement. Outliers replace a fixed floor(N * outliers_ratio) random
subset, carry true label -1, and each represents an unrelated unknown entity.

Latent centroids blend a common direction with independent unit directions.
`inter_entity_similarity` increases the common-direction mixture weight; it is a
control, not the resulting exact cosine similarity. Within-entity Gaussian noise
is scaled by sqrt(dimension). Difficulty multiplies variance and noise by 0.5,
1, 1.7, 2.5 for easy, medium, hard, extreme respectively, increasing dispersion
while holding other controls fixed. Each observation belongs to domain 0 or 1;
`domain_shift` adds a shared vector of that magnitude only in domain 1, before
normalization. It is an actual domain-dependent geometric shift, not a noise
alias. These choices are simulation assumptions rather than measured voice data.

## Metric contract

`orion.metrics.calculate_metrics(labels_true, labels_pred)` accepts equal-length
1D integer arrays with labels >= -1 and returns JSON-safe Python values. The
legacy `(dataset, labels_true, labels_pred)` form remains accepted. Predicted -1
means abstention. Each true -1 is treated as a distinct singleton true identity;
two outliers are never a correct identity match. Each global unresolved prediction
is likewise a distinct singleton. Known cluster IDs are nominal.

All undefined ratios return `null`, with explicit counts. Empty arrays have null
scores. Standard singleton conventions of scikit-learn ARI/V measure apply to
nonempty global partitions; an ARI of 1 on true singletons and abstentions is not
an accepted-match claim (accepted pair precision is still null).

| Field | Definition |
| --- | --- |
| `true_positive_pairs` (TP) | same true identity and same accepted predicted cluster |
| `false_positive_pairs` (FP) | different true identities in same accepted cluster |
| `false_negative_pairs` (FN) | every true pair that is not an accepted match, including abstentions |
| `true_negative_pairs` (TN) | different true identities not accepted as a match |
| `accepted_pairs`, `true_pairs`, `impostor_pairs` | TP+FP, TP+FN, FP+TN over full population |
| `pairwise_precision` | TP / accepted_pairs |
| `pairwise_recall`, `recall_all_known_pairs` | TP / true_pairs, including missed abstained pairs |
| `pairwise_recall_accepted` | selective recall, true-pair denominator restricted to accepted observations |
| `pairwise_f1` | 2TP / (accepted_pairs + true_pairs), including zero when recall is zero |
| `b_cubed_precision`, `b_cubed_recall` | standard per-point intersection / predicted cluster size and intersection / true cluster size, on accepted subset |
| `b_cubed_f1` | harmonic mean of those selective B³ values |
| `b_cubed_precision_global` | selective B³ precision sum / all observations; abstained rows contribute zero |
| `b_cubed_recall_global` | per accepted point intersection / full true identity size, summed / all observations; abstentions contribute zero |
| `coverage`, `observation_assignment_coverage` | accepted observations / all observations |
| `abstention_rate` | abstained observations / all observations |
| `assigned_non_singleton_fraction` | points in accepted clusters of size >1 / all observations |
| `false_match_rate` | FP / impostor_pairs, over this all-pairs synthetic comparison protocol |
| `false_discovery_proportion` | FP / accepted_pairs |
| `point_contamination` | off-majority accepted points / accepted observations |
| `cluster_contamination`, `cluster_level_contamination` | mixed accepted clusters / accepted clusters (compatibility aliases) |
| `fragmentation` | true identities split across >1 global predicted clusters / true identities |
| `safe_identity_recovery` | true identities fully assigned to one pure cluster / true identities |
| `adjusted_rand_index`, `v_measure` | standard global partition scores; each unresolved prediction is a distinct singleton |
| `adjusted_rand_index_accepted`, `v_measure_accepted` | standard scores restricted to accepted observations |

Counts include `total_observations`, `accepted_observations`,
`abstained_observations`, and `outlier_observations`. FMR here is the rate over all
synthetic impostor pairs, not a claim about an operational biometric threshold.
Coverage must be read alongside accepted precision, pair volume, and global recall.

## Algorithms and protocol

The cosine threshold graph has a real adjustable threshold. Reciprocal kNN uses
mutual neighbor-list membership, excludes self, and applies a cosine threshold;
it uses sparse edges. AHC uses average linkage and a development-selected distance
threshold to discover K. An explicit `n_clusters` argument is available only as
an oracle control and is unused in the protocol.

`RiskGraphClusterer` is an experimental conservative heuristic. It normalizes
inputs, retrieves exact bounded-memory neighborhoods, optionally keeps neighbors
within `expansion_factor` times the nearest-neighbor cosine distance, requires
mutual membership, and gates edge scores. Local calibration compares edges to
mean(neighbor similarities) - std / calibration_factor for both endpoints; this is
not a calibrated probability. Merges process edges by descending score using
union-find; complete-link cohesion checks all cross-component pairs before a
merge. Components receive canonical consecutive labels; singletons return -1
unless `use_abstention=False`. The compatibility `level` argument never changes
thresholds based on the true outlier fraction. Dense full similarity matrices are
accepted only through the explicit small-input compatibility API.

Development selects every method separately over the same candidate thresholds,
seeds, and scenarios. Selection maximizes mean global pair F1 among candidates
whose mean pair FDP is <= the declared budget; any no-pair cohort makes a
candidate ineligible. If no candidate meets the budget, it picks minimum mean FDP
then highest F1; an all-no-pair candidate ranks last. Ties choose the first listed
threshold. All candidates and predictions remain in development artifacts. Frozen
parameters and their SHA-256 are written before held-out evaluation. Evaluation
seeds must be disjoint and cannot update selection. This is finite synthetic
development selection, not a real-world generalization guarantee.

Each ablation changes one component toggle from the frozen full system, using
identical datasets. `without_abstention` retains unresolved singleton components
as assigned singleton clusters. Sensitivity uses the same seed for every value;
algorithm-only factors reuse identical data. Data-factor changes preserve the
same seeded construction where shapes allow, and actual dataset hashes document
which data differ. Domain shift has its own generator parameter. Sensitivity is
descriptive; it does not retune the frozen configuration.

Every run saves input vectors, labels, predictions, domain/outlier masks, runtime,
metrics, stratified domain/outlier metrics, and hashes. `manifest.json` records
package versions, platform, code revision, and file hashes. Candidate measurements
and `claims.json` require independent controller verification before acceptance.
Hashes identify bytes; they do not prove authenticity. Baselines and conservative
cohesion retain quadratic compute in small clustering cohorts. They are never
used in the large-N retrieval benchmark.

## Retrieval API and evidence

`orion.retrieval.exact_knn(vectors, queries, k)` and `approximate_knn(...)` return
`(neighbor_row_ids, squared_L2_distances)`, both shape `(num_queries, k)`. Results
sort by distance then row ID. Exact search scans at most 32 queries by 8192 index
rows at once. It includes a self match if a query is also an indexed vector.
Unit vectors obey squared L2 = 2 * cosine distance. Inputs must be finite, feature
dimensions must match, and k must lie in [1, index_size].

Approximate retrieval uses eight seeded random projections with sorted indexes,
a window of 256 candidates per projection, union deduplication, and exact reranking
in original space. `ProjectionWindowIndex` separates build/query timing. This is
projection-window candidate search, not HNSW or FAISS; recall may be low in high
dimension. Its memory is O(ND + NP + QK + window * P * D), for P projections,
without an N by N matrix. Small indexes can degenerate to exact candidate coverage.

The benchmark saves full vectors and independent queries as `.npy`, and both
exact and approximate neighbor IDs/distances as `.npz`. Exact neighbors search the
full index for every reported query, not only a reduced index. Recall@k is mean
intersection size / k. Generation, exact query, approximate build/query timings
are separate real wall-clock measurements. Process RSS is sampled every 5 ms
and includes the Python process, inputs, and libraries; short peaks may be missed.
Independent controller observation supplies trusted timing/resource evidence.
