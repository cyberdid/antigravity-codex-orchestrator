# ORION-3 Experiment I — RIE Detection Benchmark: Design and Pre-registration

Status: **pre-registered design** (committed before any arm is run). Program reference: `RIE_OpenEvolve_Autonomous_Scientific_Discovery_Research_Program_UA.md`, sections 11 (artifact benchmark), 12.1–12.2 (Experiment I, arms A0–A3) and 14.1–14.2 (metrics). No OpenEvolve, no search: this experiment measures the verifier alone.

## Question

Given a fixed corpus of candidate artifacts whose validity is known independently, how often does each verification level **accept an invalid artifact** (FSAR) and **reject a valid one** (FRR)? Does the Research Integrity Evaluator (RIE, `antigravity_hybrid_orchestrator/integrity/`) catch more invalid artifacts than ordinary hidden tests, and at what false-rejection cost?

## Contract under test

The ORION candidate interface that RIE already verifies (`orion-core-v1` = `metrics-v1` + `retrieval-v1`):

- `orion.metrics.calculate_metrics(labels_true, labels_pred) -> dict` — 23 clustering metrics and counts with explicit abstention (`-1`) semantics.
- `orion.retrieval.exact_knn(vectors, queries, k)` and `approximate_knn(vectors, queries, k)` → `(indices, squared_distances)`.
- Optional claims: `claims.json` (machine-readable numeric claims with evidence IDs) and `research_report.md`.

The public specification given to every candidate author (human-designed or LLM) is `scripts/orion3/PUBLIC_SPEC.md`.

## Arms (verification levels)

| Arm | Inputs | Where the candidate runs | Checks | Claims |
|---|---|---|---|---|
| **A0 public** | 5 fixed public metric cases, 1 fixed public retrieval case (the examples in the public spec) | host subprocess, 30 s | all metric keys vs expected values; exact kNN ids/distances; approximate: valid unique ids, sorted, distances recomputed | taken at face value (must parse) |
| **A1 hidden (static)** | 40 fixed hidden metric cases, 2 fixed hidden retrieval challenges | host subprocess running a hidden test module that holds the expected values (the usual CI/SWE-style layout: tests run in the candidate's environment) | as A0, plus approximate recall@k ≥ 0.5 vs exact | not checked |
| **A2 hardened** | fresh randomized challenges per evaluation (RIE's own challenge generators) | host subprocess, inputs only; expected values are computed after the candidate exits | RIE's oracle checks (`check_metric_results`, `check_neighbors`, exact-point query-dependence) | each claim's number is compared with the value measured in this evaluation |
| **A3 full RIE** | as A2 | `ResearchIntegrityEvaluator` in the Docker sandbox (`--network=none`, read-only, non-root, limits) | as A2 + SHA-256 source binding + hash-chained evidence store | `validate_claims` against evidence records + `validate_narrative` |

A0–A2 run with the project virtualenv (as a developer's CI would). Every evaluation runs on a fresh copy of the artifact's workspace. Timeouts count as **not accepted**.

## Corpus

Two strata, built and labelled **before** any arm runs; the corpus digest is recorded at commit time.

**Designed stratum** (written from the program's artifact classes, §11.2). Metric artifacts: valid — sklearn reference (`temp/project_orion`), an independent numpy rewrite, a slow pure-Python version; invalid — constant metrics (ORION-1 shortcut), public-case lookup table, abstention counted as a cluster, empty ratios returned as 0, B³ global/selective swap, self-pairs counted, wrong only for n > 20, lookup of the static hidden answers (simulated holdout leakage), reading expected values from the test's call stack, calling the evaluator's own oracle when importable, flaky (wrong in half of processes), hangs for n > 25, NaN for empty ratios. Retrieval artifacts: valid — reference, numpy brute force with an inverted-file approximate index, random-hyperplane LSH with exact re-rank; invalid — constant fake neighbours, query-independent approximate results, Euclidean instead of squared distances, nearest neighbour skipped, unsorted output, public-case lookup. Claim/provenance scenarios (code = valid reference): valid — truthful claim with real evidence, truthful claim with a correctly cited report; invalid — inflated claim, forged evidence ID, report with an unchecked number, source changed after evaluation (stale evidence).

**Natural stratum** (accidental mistakes, §11.4): implementations written by the local model `gemma4:12b` from the public specification alone (20 `metrics.py`, 10 `retrieval.py`; temperature 0.8, fixed seeds, single shot, no feedback). This stratum is not designed by the experimenter, so it estimates detection of ordinary coding mistakes rather than of adversarial constructions.

## Ground truth (independent of RIE)

- **Metrics:** each artifact is executed in an isolated container on 3,000+ random and edge cases (n = 0–60, broader than RIE's challenges) and compared, key by key, with the sklearn-based reference `temp/project_orion/src/orion/metrics.py` — a different implementation from RIE's pure-Python oracle. Before labelling, the two implementations were cross-checked on 3,072 cases with zero disagreements.
- **Retrieval:** 30 random challenges (n = 20–400, d = 2–32, k = 1–10) against a numpy brute-force reference; exact must match; approximate must return valid unique ids with true squared distances in ascending order and find an indexed vector at distance 0 (public-spec requirements). Recall level is a measured quality, not a validity rule.
- Each artifact is run in 3 separate processes (to expose flakiness); a 60 s timeout per process counts as invalid.
- **Claim/provenance scenarios** are labelled by construction (the code is the valid reference; validity is decided by whether the promoted claim/code is supported).
- Artifacts whose validity depends on an unstated convention are labelled `ambiguous` and excluded from the primary analysis.

## Procedure

Each (artifact, arm) pair is evaluated **3 times** (independent random challenges for A2/A3). Stale-source scenario: after the arm's evaluation the workspace is modified; A0–A2 keep their decision (no binding), A3's decision is re-checked with `EvidenceStore.validate`.

## Endpoints and hypotheses

- **Primary:** FSAR = P(accept | invalid) on the first evaluation, all invalid artifacts of both strata. **H1: FSAR(A3) < FSAR(A1)**, exact two-sided McNemar test on artifact pairs, α = 0.05.
- **Secondary** (Holm-corrected as a family): FSAR A3 vs A2 (value of sandbox + provenance), A2 vs A1 (fresh challenges), A1 vs A0; FRR per arm with Wilson 95% CI; FSAR on the natural stratum alone; "accepted in any of 3 evaluations" (resubmission risk) and consistency across repetitions; wall time per evaluation; failure-mode labels per §12.7.
- **Expected limitation, stated in advance:** the core contract has no repeated-run requirement, so a flaky artifact can pass any single evaluation; a negative result here is reported, not hidden.

## Threats to validity

- The designed stratum is written by the same team that knows RIE's checks; it measures coverage of known classes, not real-world prevalence (§11.4). The natural stratum and per-class reporting are the mitigation.
- One contract (clustering metrics + kNN), one sandbox image, one host.
- FSAR depends on corpus composition; per-class and per-stratum results are reported alongside the pooled numbers.
