# Paper I — Plan (draft, 2026-10-20)

Working title: **Beyond Passing Tests: Specification-Only Contract Tests Against Silent Failures of Local Coding Agents**
(alternative: *Specification-Only Contract Testing for Privacy-Bounded Cloud–Edge Coding Agents: Reliability, Cost, and Failure Modes*).

Scope: software-engineering paper. OpenEvolve and autonomous discovery are **not** part of it (Paper II). RIE / ORION-3 Experiment I is paused and is not needed for Paper I.

## 1. Question and definitions

> Can tests written by a cloud model from the specification alone (signatures, docstrings, data-type fields — no implementation) reduce silent functional failures of a local coding agent, and at what end-to-end cost?

For a task with visible tests V and an independent hidden evaluation H (never shown to agent or cloud):

- **VCR** (verified completion rate) = P(V = 1 ∧ H = 1) — primary.
- **FSR** (false success rate) = P(V = 1 ∧ H = 0); **silent FSR** additionally requires the agent to report success.
- **Cost**: T_total = T_contract_generation + T_worker + T_verification (wall time); C_total = cloud tokens + local tokens (and GPU-seconds). Contract generation is reported both per task (one-time) and amortised over runs.

The unit of inference is the **task**; seeds are repeated measurements within a task.

## 2. Evidence so far (exploratory + one confirmatory run)

From `docs/SPEC_CONTRACT_RESULTS.md`: pooled 19 tasks VCR 32/57 → 48/57, silent FSR 25/57 → 3/57; task level 7 better / 1 worse, permutation p = 0.023, mean gain +0.28 [0.09, 0.49]. Fresh-task confirmatory run: run-level p = 0.039, task-level 3/0/3 (p = 0.25). Negative results to report: local self-check (SPEC), post-failure cloud guidance (ASSIST), escalation ladder, Harness² one-step adaptation; contract tests without data types harmed solved tasks.

Known threats in the existing data: 13 of 19 tasks were used during development; the 6 fresh tasks were written by us *after* identifying the failure mode (docstring-only rules), so they over-represent tasks where the method helps — the effect size on naturally occurring tasks is unknown; one worker model; synthetic single-function tasks; generation time untimed.

## 3. Prior art (verified 2026-10-20 unless marked)

| Work | Relation | Status |
|---|---|---|
| Tufano et al., *Grounding AI Agents in Contracts: An Empirical Evaluation of Spec-Driven Test Generation*, SpecOps '26 (Google) | Spec-driven test generation via pre/post-conditions; +9.8 pp bug detection (p = 0.035) on Google production bugs | verified (research.google page) |
| Alshahwan et al., *Automated Unit Test Improvement using LLMs at Meta* (TestGen-LLM), FSE 2024 | LLM test improvement with filters; coverage, not spec conformance | verified (arXiv 2402.09171) |
| Huang et al., *AgentCoder* (arXiv 2312.13010) | Test-designer agent writes tests from requirements, separate from programmer agent | verified (arXiv); single model, not privacy-bounded |
| Zhong, Raghunathan, Carlini, *ImpossibleBench* (ICLR 2026, arXiv 2510.20270) | Agents exploit tests that contradict the spec — complementary failure (gaming vs our coverage gap) | verified (arXiv/ICLR listing) |
| *Do Coding Agents Deceive Us? … Capped Evaluation with Randomized Tests* (arXiv 2606.07379) | Randomized hidden tests against cheating — relevant to RIE/Paper II | found, not yet read |
| Harness² (google-research/harness2) | Harness adaptation without weight updates; our negative replication | verified (repository) |
| EvalPlus (Liu et al., NeurIPS 2023), CodeT (Chen et al., ICLR 2023) | Insufficient tests inflate correctness; generated tests for selection | to verify and cite |
| "TRACE (2026)", Information and Software Technology | cited by the external review | **not found** (403 / search miss) — do not cite until verified |

Candidate contribution (hypothesis, not a priority claim): (a) a **privacy-bounded** split — the cloud sees only the interface, the local model sees the code; (b) **information-disclosure ablation** — how much of the interface the cloud needs (names → signatures → docstrings → data types) for useful, non-harmful tests; (c) **cost–reliability** analysis: min C_total subject to FSR ≤ ε, including a *selective* policy that requests contract tests only when the spec states rules not covered by visible tests.

## 4. Final campaign (to be pre-registered after the research freeze)

**Research freeze.** Tag a commit; fix worker/cloud model versions, prompts, parameters, metrics and the analysis script before building results.

**Benchmark: ≥ 40 new tasks**, built and frozen before any run. Power simulation from the observed per-task gains (task-level exact sign test, α = 0.05): 20 tasks 0.48, 30 tasks 0.74, **40 tasks 0.87**, 50 tasks 0.94. Composition (target):
- 15 real bugs from open-source Python projects (task = buggy revision + issue/docstring spec; hidden evaluation = the project's fix tests) — not designed by us, estimates natural prevalence;
- 10 multi-file changes;
- 8 single-function tasks with docstring-only rules (comparable to existing tasks);
- 7 tasks with incomplete or underspecified docstrings, where over-constrained tests are the risk.
Each task: reference fix, visible tests, hidden evaluation, offline check that partial fixes fail the hidden evaluation.

**Arms.**
1. Baseline (current P1_STAGE_ADAPTIVE).
2. SPECTESTS (contract tests from the full interface).
3. **CLOUD-CHECKLIST control**: the same cloud model, same specification input, produces a natural-language checklist of rules given to the worker up front — separates "contract tests" from "extra strong-model compute on the same information".
4. Disclosure ablation of SPECTESTS (signatures only; + docstrings; + data types) on a subset.
5. Second worker model (qwen3.5:27b, available locally) for arms 1–2.

**Analysis.** Primary: task-level VCR difference SPECTESTS − baseline, exact sign test over tasks, plus cluster-bootstrap CI; secondary: mixed-effects logistic regression (task random effect), silent FSR, CLOUD-CHECKLIST vs SPECTESTS, disclosure ablation, cost frontier; Holm correction across secondary tests.

**Test-oracle quality.** For every generated module: pass rate on the reference fix; **over-constraint check** on independent valid implementations (baseline runs that pass the hidden evaluation, plus hand-written alternatives); **mutation score** on the reference (mutmut or a small AST mutator); a human review sample of 20% of tests.

**Cost accounting.** Time the generation call (`spec_tests_wall_s`, now recorded), the worker and verification separately; report per-task and amortised costs.

**Compute estimate.** 40 tasks × 3 arms × 3 seeds ≈ 360 worker runs ≈ 15–20 GPU-hours on the RTX 4060, plus the second model (≈ 2× slower) and ~40–120 cloud calls.

## 5. Reproducibility package

Tagged commit; model digests (`ollama show`), cloud model id; task set with references and hidden tests; raw benchmark JSON and trajectories; analysis scripts that regenerate every table and figure; a manifest with SHA-256 of all inputs/outputs; Threats to Validity section; Zenodo DOI after the confirmatory campaign, not before.

## 6. Order of work

1. Research freeze + pre-registration of the campaign (this document → protocol).
2. Build and freeze the 40-task benchmark (largest effort; real-bug tasks first).
3. Implement CLOUD-CHECKLIST arm, disclosure ablation switch, over-constraint and mutation checks.
4. Run the campaign; analyse with the frozen script.
5. Write the manuscript (related work from §3), release package.
