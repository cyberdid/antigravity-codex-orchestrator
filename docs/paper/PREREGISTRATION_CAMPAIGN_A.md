# Pre-registration — Paper I, Campaign A (synthetic rule tasks P1–P25)

**Registered:** 2026-10-20, by commit and push of this file to the public repository `cyberdid/antigravity-codex-orchestrator` (branch `feat/gemma-orion-phase2b`) **before any campaign run**. The git commit timestamp on GitHub is the registration time.
**Author:** Oleksandr Pylypenko (ORCID 0009-0004-9610-9242).
**Relation to the paper:** confirmatory campaign for `docs/paper/manuscript.md`. Campaign B (real bugs from open-source projects) will be registered separately once its task set is frozen.

## 1. Frozen inputs

- **Code:** the commit that adds this file (engine, benchmark, arms, analysis script).
- **Tasks:** `scripts/paper_tasks/` — 25 tasks (8 single-module, 10 multi-file, 7 under-specified), frozen in `scripts/paper_tasks/manifest.json` with a SHA-256 per task; the benchmark refuses a task whose content changed. Which docstring rules the visible tests cover was decided per rule by a seeded coin flip (p = 0.5, seed = hash of the task id), constrained only so that at least one visible rule fails on the buggy code. The hidden holdout checks every rule. None of these tasks has been run with any agent before registration (the end-to-end check of the new arms used task H2 from the earlier study).
- **Local agent:** `gemma4:12b` via Ollama 0.33.3 (digest `4eb23ef187e2…`), `think` off, context 16,384, generation limit 2,048 with one 4,096 retry for truncated tool calls, stage hints, 2 attempts, 300 s per attempt.
- **Cloud model:** Antigravity CLI `agy` 1.4.3 with the model pinned to `gemini-4-argon-medium` (`--cloud-model`), run in an empty temporary directory; one generation per task and arm, reused across seeds.
- **Seeds:** 303, 404, 505; arm order counterbalanced per seed by the benchmark.

## 2. Arms

| Arm | Policy id | What it adds |
|---|---|---|
| Baseline | `P1_STAGE_ADAPTIVE` | nothing |
| Contract tests | `P1_STAGE_ADAPTIVE_SPECTESTS` | cloud-written unit tests from the full interface (signatures, docstrings, type fields) |
| Checklist (control) | `P1_STAGE_ADAPTIVE_CHECKLIST` | same cloud model and same interface; a numbered rule checklist appended to the task description instead of tests |
| Ablation (exploratory, seed 303 only) | `…_SPECTESTS_SIG`, `…_SPECTESTS_DOC` | contract tests from signatures only / signatures + docstrings |

## 3. Outcomes and hypotheses

Per run: **verified** = the task's own visible tests and the hidden holdout pass after the run (generated tests excluded); **silent false success** = the agent reports success and verified is false. Unit of inference: **task**; per-task rate = verified runs / 3 seeds.

- **H1 (primary):** contract tests raise the per-task verified rate over the baseline. Test: exact two-sided sign test over tasks with a non-zero difference, α = 0.05.
- **H2:** contract tests lower the per-task silent-false-success rate (sign test over tasks).
- **H3:** contract tests raise the per-task verified rate over the checklist control (sign test). A null result here would mean the gain comes from the cloud's reading of the specification, not from executable tests.
- **Secondary, descriptive or Holm-corrected as a family (H2, H3, checklist vs baseline):** sign-flip permutation p and cluster-bootstrap 95% CI of the mean per-task difference; results by category; harm on tasks without hidden failing rules; end-to-end cost (run wall time, one-time generation wall time, local and cloud tokens); ablation by disclosure level.

**Power.** Resampling the per-task gains observed in the earlier 19-task study gives power ≈ 0.6 for 25 tasks (0.48 at 20, 0.74 at 30). The rule-visibility procedure differs from the earlier tasks, so the true effect distribution may differ; Campaign A alone may be under-powered, and the pooled analysis with Campaign B is planned.

## 4. Analysis

`scripts/analysis/campaign_a.py` (committed with this file; dry-run on synthetic data only) is the analysis. Any other analysis is reported as exploratory.

## 5. Run rules and deviations

- All 225 primary runs (3 arms × 25 tasks × 3 seeds) and 50 ablation runs are executed; no early stopping.
- A run lost to infrastructure (Ollama or `agy` unavailable, host shutdown) is resumed from the saved partial results with the same generated tests (`--resume-from`), and reported.
- Background indexing timers on the host are paused during the runs.
- Any deviation from this document is listed in the results section with its reason.
