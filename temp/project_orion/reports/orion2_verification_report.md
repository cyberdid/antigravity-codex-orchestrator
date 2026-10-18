# Orion2 Research Verification and Comparative Audit Report

## 1. Confirmed Facts
The following properties were cryptographically and procedurally verified during the study protocol:
- **Disjoint Development/Evaluation Splits**: Verified at `src/orion/runner.py` (lines 81-84).
- **Frozen Parameter Selection**: Ensured under maximum False Discovery Probability (FDP) $\le$ 0.01 at `src/orion/runner.py` (lines 65-76).
- **Dataset Integrity**: Identical dataset SHA-256 hashes confirmed across all ablation variants at `src/orion/runner.py` (lines 121-131).
- **Directory Integrity**: Non-overwriting directory checks confirmed at `src/orion/runner.py` (lines 86-87) and `README.md` (lines 5-6).

## 2. Reported Claims
Quantitative empirical outcomes derived from `orion2-run/results.json` and `claims.json` indicate the following performance metrics across `threshold_graph`, `agglomerative`, `rr_graph`, and `risk_aware` models:
- **Core Metrics**: Comparative analysis of Adjusted Rand Index (ARI), cluster contamination, coverage, precision, and recall.
- **Multi-scale Retrieval**: Evaluation of recall@10 across $N \in [1k, 10k, 100k, 200k]$.
- **Resource Utilization**: Peak RSS memory usage recorded across the specified $N$ scales.

## 3. Inferences
- **Hypothesis H1 Evaluation**: Analysis of whether risk-constrained clustering achieves superior coverage at low contamination compared to baselines. Observations indicate significant fragmentation and lower recall in specific risk-constrained configurations.
- **Mechanistic Insights**: Evaluation of the impact of compound conservative filters on cluster stability and connectivity.

## 4. Contested Points
- **Distributional Assumptions**: Methodological boundaries between synthetic unit vectors and real-world manifold distributions.
- **Memory Profiling**: Discrepancies between sampled process RSS and true allocator memory peaks (as noted in `scripts/benchmark_scalability.py` lines 3-5).
- **Indexing Limitations**: Comparison of projection-window ANN limitations versus graph-based indices for high-dimensional retrieval.

## 5. Open Questions
- **Fragmentation Recovery**: Future algorithmic directions for recovering fragmented clusters without relying on transitive chaining.
- **Real-world Validation**: Necessity of further validation on non-synthetic, real-world datasets to confirm generalizability.
