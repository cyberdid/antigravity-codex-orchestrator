# Specification Coverage for a Local 12B Coding Worker — Results (2026-10-19/20)

> **Abstract.** On specification-heavy repair tasks, `gemma4:12b` passes the visible tests and silently violates rules written only in the docstring. Contract tests written by a cloud model from signatures, docstrings and data-type fields alone (no implementation) raise verified completion on such tasks from 2/12 to 10/12 (paired 8–0, p = 0.0078; replicates a first run at 16/18 vs 7/18, p = 0.012) and eliminate silent false successes (10/39 → 0/39). They must see the full interface: without data-type fields they produced impossible tests that hurt already-solved tasks (p = 0.031). Self-checks by the same 12B model and guidance after failure did not help. Across all 13 tasks the gain is 29/39 → 33/39 (not significant), at ~1.8× wall time. Details: final section "Final run" and "Summary of findings".

## Question

Earlier experiments (`docs/GEMMA_ORION_PHASE2B_REPORT.md` sections 6–9, `docs/HARNESS2_RESULTS.md`) left one dominant failure mode for `gemma4:12b`: on specification-heavy repair tasks the worker makes the **visible** tests pass but misses rules stated only in the **docstring**, which hidden holdout tests check. Generation-limit escalation, thinking on retry and one-step Harness² adaptation did not change that. This experiment asks which intervention closes the gap:

| Policy | What it adds to the worker (`P1_STAGE` stage hints + adaptive generation limit) | Leaves the machine |
|---|---|---|
| `P1_STAGE_ADAPTIVE` (baseline) | nothing | nothing |
| `P1_STAGE_ADAPTIVE_SPEC` | after visible tests pass, the stage hint gives the worker its own task's signatures and docstrings and requires one `python3 -c` self-check of every rule | nothing |
| `P1_STAGE_ADAPTIVE_SPECTESTS` | before the run, the cloud architect writes a unittest module from the **signatures and docstrings only**; it is statically validated and installed as `test_spec_contract.py`, which the normal verification command then runs | signatures + docstrings (spec level, owner-approved) |
| `P1_STAGE_ADAPTIVE_ASSIST` | after a failed attempt, step-by-step cloud guidance from signatures, docstrings and failing-test lines | same + failing-test lines |

## Setup

- **Data:** `benchmarks/spec_contract_experiment.json`, code `cc938ca`, `git_dirty=false`.
- **Tasks (6, all specification-heavy, defects split so that each visible test exposes one):** H6 `parse_duration`, H9 `parse_semver`, H10 `parse_ipv4`, H11 `parse_bool`, H12 `parse_hex_color`, H13 `parse_range`. For every task, offline checks confirmed that the original code and every partial fix fail the visible tests and only a full fix passes both visible and holdout tests. Hidden holdout tests are never shown to the worker or the cloud.
- **Runs:** 4 policies × 6 tasks × seeds 303, 404, 505 = 72 runs; `--attempts 2` through `SelfCorrectionLoop` + `EscalationLadder` (the orchestrator's code path); 300 s per attempt; policy order alternated per seed.
- **Verified** = the task's own visible tests **and** the hidden holdout pass, checked independently after the run (generated contract tests are excluded from this check).
- **Cloud:** `agy` (default Gemini model), run in an empty temporary directory so it cannot read the repository. Contract tests are generated once per task and reused across seeds.
- Background load: the owner's indexing timers were paused during the run and restored at the end.

## Results

| Policy | Verified | 95% CI (Wilson) | Visible pass, holdout fail | Median wall / run | Worker tokens | Cloud tokens |
|---|---:|---|---:|---:|---:|---:|
| baseline | 7/18 | 0.20–0.61 | 11 | 36 s | 257,616 | 0 |
| SPEC (local self-check) | 5/18 | 0.12–0.51 | 10 | 108 s | 396,138 | 0 |
| **SPECTESTS (cloud contract tests)** | **16/18** | **0.67–0.97** | **0** | 100 s | 640,588 | 176,911 (6 calls) |
| ASSIST | 8/18 | 0.25–0.66 | 10 | 35 s | 261,063 | 4,222 (1 call) |

Per task (verified / 3 seeds; ✓ verified, v visible only, ✗ neither):

| Task | baseline | SPEC | SPECTESTS | ASSIST |
|---|---|---|---|---|
| H6 durations | 2/3 ✓v✓ | 1/3 ✗✗✓ | 1/3 ✗✗✓ | 3/3 ✓✓✓ |
| H9 semver | 1/3 v✓v | 1/3 ✓v✗ | **3/3** | 2/3 ✓✓v |
| H10 ipv4 | 0/3 | 0/3 | **3/3** | 0/3 |
| H11 bool | 3/3 | 3/3 | 3/3 | 3/3 |
| H12 hex colour | 1/3 ✓vv | 0/3 | **3/3** | 0/3 |
| H13 range | 0/3 | 0/3 | **3/3** | 0/3 |

**Paired comparison with the baseline (same task and seed), exact two-sided sign test on discordant pairs:**
- SPECTESTS: 10 wins, 1 loss, 7 ties — **p = 0.012**
- SPEC: 1 win, 3 losses, 14 ties — p = 0.63
- ASSIST: 2 wins, 1 loss, 15 ties — p = 1.0

## Validity checks

1. **The generated contract tests are correct.** Every generated module was run against an independent reference implementation of each task: **0 failing tests on correct code** (6 modules, 7–19 tests each), and each module fails on the original buggy code. No run was helped or hurt by a wrong test.
2. **No holdout leakage.** The cloud saw only signatures and docstrings and ran in an empty directory. Overlap between holdout inputs and generated test inputs is partial (H6 2/8, H9 3/13, H10 6/11, H11 8/10, H12 2/11, H13 3/10 literal inputs), as expected for two test suites written from the same docstring. The gains are on tasks with low overlap (H12, H13: 3/3 vs 1/3 and 0/3), and H11, the task with the highest overlap, was already solved 3/3 by every policy.
3. **Same code path, same budget.** All four policies ran through the same loop with the same deadline and attempt limit, alternating order per seed, on a clean commit.

## Why the other policies did not help (trajectory forensics)

- **SPEC (self-check by the same model):** the worker did run its own `python3 -c` contract check (10 of 18 runs) and the check passed — but it only exercised behaviour the worker had already implemented. A 12B model checking itself shares its own blind spots (e.g. H13: all three self-checks passed, all three holdouts failed). The extra turns tripled the median time.
- **ASSIST:** guidance is requested only after an attempt fails, and on these tasks the first attempt usually *passes the visible tests* — so it triggered once in 18 runs (H6, where it then solved the task). It cannot address failures the controller cannot see.
- **SPECTESTS failures (H6, 2 runs):** the contract tests were stricter than the visible tests and the worker ran out of time (600 s): long rewrite turns and one 2048-token runaway turn. Time budget, not test quality, limited it.

## Conclusions

1. **Specification coverage, not reasoning budget or prompt tuning, was the binding constraint** for the local 12B worker on these tasks. Making the specification executable — tests derived from the docstring — raised verified completion from 39% to 89% (16/18 vs 7/18, p = 0.012) and eliminated "passes visible, fails hidden" outcomes (0 vs 11).
2. **Who writes the tests matters.** The same model checking itself did not help (5/18); tests written by a stronger cloud model from the specification alone did. This is a clean division of labour for the hybrid architecture: the cloud reads only the specification (signatures + docstrings, no implementation), the local model writes and fixes the code.
3. **Cost:** about 2.8× median wall time per run (100 s vs 36 s), 2.5× worker tokens and ~29.5k cloud tokens per task (one generation per task, reused across seeds). Raising the attempt budget would likely recover the two H6 timeouts.

**Limits:** 6 tasks of one family (single-function Python parsers with docstring specs), 3 seeds, one local model, one cloud model; the effect size is large, but generalisation to multi-file or less precisely specified tasks is untested. Contract tests are only as good as the docstring; under-specified functions give the cloud nothing to test.

## Control: do contract tests hurt tasks the baseline already solves? (v1, `24fbe35`)

`benchmarks/spec_contract_control.json`: baseline vs SPECTESTS on H1–H5, H7, H8 (tasks the baseline solves), 3 seeds, 42 runs.

| Policy | Verified | Paired vs baseline |
|---|---:|---|
| baseline | 21/21 | — |
| SPECTESTS v1 | 15/21 | 0 wins, 6 losses, **p = 0.031 (harm)** |

Five of the six losses are on H1 (2) and H3 (3); the sixth (H4, seed 505) happened with correct contract tests — the worker stalled in a repeated tool cycle after one truncated turn. Grading the generated modules against reference implementations explains the H1/H3 losses: for **H1 the contract tests were wrong in 19 of 20 cases** (they read `FdrDecision.rejected`, a field that does not exist; the real field is `rejected_mask`), and for **H3 in 11 of 12** (they built `SampleRecord` without its required `value` field). The cloud had seen only the functions' own file; the data types live in other input files (`fdr_types.py`, `sample_record.py`), so it guessed the interface. The worker then could not satisfy impossible tests and ran out of time. On H2, H4, H5, H7 and H8, where the interface is fully visible in the function's own file, every generated test was correct.

**Fix (`750647e`, v2):** the contract also lists class field names and types (defaults dropped — still no implementation) and covers the non-test input modules the functions use. Live re-generation for H1 and H3 then passed 100% on the reference implementations (10/10 and 11/11 tests). A final same-session run of baseline vs SPECTESTS v2 on all 13 tasks follows below.

**Lesson:** contract tests are only safe when the cloud sees the complete interface. A plausible-looking but wrong test is worse than no test: it turns a solved task into a timeout.

## Final run: baseline vs SPECTESTS v2 on all 13 tasks, same session (`750647e`)

`benchmarks/spec_contract_v2.json`: 2 policies × H1–H13 × seeds 303, 404, 505 = 78 runs, `--attempts 2`, clean tree, policy order alternated per seed. This run supersedes the per-policy numbers above for the SPECTESTS-vs-baseline question because both arms ran on the same code in the same session.

**Contract test quality (v2):** every generated module (13 tasks, 8–22 tests each) passes on an independent reference implementation — **0 wrong tests** — and fails on the original buggy code.

| Subset | Baseline | SPECTESTS v2 | Paired wins / losses | Sign test |
|---|---:|---:|---|---:|
| All 13 tasks | 29/39 | 33/39 | 8 / 4 | p = 0.39 |
| Spec-heavy H9, H10, H12, H13 (baseline mostly passes visible only) | 2/12 | **10/12** | **8 / 0** | **p = 0.0078** |
| Spec-heavy incl. H6, H11 | 8/18 | 14/18 | 8 / 2 | p = 0.11 |
| Tasks the baseline solves (H1–H5, H7, H8) | 21/21 | 19/21 | 0 / 2 | p = 0.50 |

**Silent false successes** — the worker reports the task done but hidden tests fail: **baseline 10/39, SPECTESTS v2 0/39.** With contract tests every remaining failure is an explicit failure (stall or deadline), never a wrong answer reported as success.

**The 6 SPECTESTS v2 failures (forensics):**
- H4 seeds 404 and 505, H10 seed 303, H13 seed 303: the worker's tool-cycle detector stopped the attempt after 47–118 s ("a tool-action cycle repeated without new evidence"); with no replanner configured, the loop treats a stall as final and the second attempt is never used. The contract tests involved are correct.
- H6 seeds 303 and 404: both attempts reached the 300 s deadline (long rewrite turns, 1–2 truncated turns), as in the v1 run.

**Compared with v1:** on tasks the baseline already solves, harm dropped from 6 losses (15/21, p = 0.031) to 2 losses (19/21, not significant) once the contract included the data types; on the spec-heavy tasks the gain replicated (v1: 10 wins / 1 loss on H6, H9–H13; v2: 8 wins / 0 losses on H9, H10, H12, H13).

## Summary of findings

1. **The local 12B worker's main failure on specification-heavy tasks is coverage, not reasoning.** It passes the visible tests and silently violates rules written only in the docstring (10 of 39 runs).
2. **Cloud-written contract tests from the specification alone fix this where it occurs.** On the four tasks where the baseline mostly fails this way, verified completion went from 2/12 to 10/12 (8 wins, 0 losses, p = 0.0078), replicating the first run (16/18 vs 7/18 on six spec-heavy tasks, p = 0.012). Silent false successes went to zero.
3. **The tests must see the full interface.** Without the data types used by the functions, the cloud wrote impossible tests and turned solved tasks into failures (v1 control, p = 0.031 harm). With field names and types (still no implementation), 0 of 13 generated modules contained a wrong test.
4. **Self-verification by the same 12B model does not help** (5/18 vs 7/18), and neither does guidance after failure, which never triggers when visible tests pass.
5. **Cost:** ~1.8× wall time and ~2× worker tokens over all 13 tasks (4,688 s vs 2,632 s; 1.23 M vs 0.59 M tokens) plus ~11k cloud tokens per task for test generation (one call per task).
6. **Remaining failure mode:** stalls and deadlines while the worker fights stricter tests (H4, H6). Letting a stalled attempt continue into the next attempt (instead of stopping without a replanner) is the next thing to measure — on a fresh task set, to avoid tuning on these tasks.

**Limits:** 13 single-function or two-module Python repair tasks of one style, 3 seeds, one local and one cloud model; the subset effect is large and replicated across two runs, but the all-task difference is not significant.
