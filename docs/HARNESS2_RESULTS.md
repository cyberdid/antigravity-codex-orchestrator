# Harness² Adaptation of the Local Worker — Results (2026-10-19)

Design: `docs/HARNESS2_INTEGRATION_DESIGN.md` (section 6 describes the implementation). Raw data: `benchmarks/harness2_experiment.json`; per-run log, verbatim cloud replies and final harnesses in `benchmarks/harness2_experiment/` (`progress.log`, `cloud_replies.json`, `final_H2/harness.json`, `final_H3/harness.json`).

## 1. Setup

- **Code:** `28a0bc7` (`feat/gemma-orion-phase2b`). The JSON records `git_dirty: true`; this is a false positive: the script creates its own output directory before reading `git status`. No tracked file was modified during the run.
- **Worker (fixed for every rollout):** `gemma4:12b`, `think=false`, `num_ctx=16384`, `num_predict=2048`, `P1_STAGE` stage hints, adaptive generation limit on, one attempt, 300 s deadline.
- **Harness editor:** cloud architect via `agy` (Gemini default model), run in an empty temporary directory so it cannot read the repository. Roles INV (procedure), DOM (practitioner knowledge), AGG (composition), as in Harness².
- **Adaptation:** tasks H1–H4 in order, seed 303, one Propose–Probe–Compose step per task (k=1: INV and DOM candidates executed on the same task, one composition executed).
- **Promotion:** H2 = standard Harness² (adopt the composition); H3 = verifier-gated (candidate must pass visible and hidden holdout tests and beat the base on `U = 1000·VCR − 25·trunc − 15·bad_tool − 0.5·wall − 0.01·tokens` by more than 10).
- **Held-out evaluation:** frozen harnesses H0 (default), H2, H3 on H5–H10 with seeds 303 and 404 (12 runs per arm). Verified = visible and holdout tests pass.
- **Wall time:** 17:32–18:41 UTC (69 min). **Cloud:** 12 calls, 0 failures, 258,127 tokens (most of it the editor's thinking; a single INV call took up to ~5 min and ~39k output tokens).

## 2. Adaptation steps

| Task | Base run | INV candidate | DOM candidate | AGG (composer) | H2 decision | H3 decision |
|---|---|---|---|---|---|---|
| H1 | ✓ 54 s, 8 turns, 16.3k tok | ✓ 46 s, 9 turns, 19.0k tok | ✓ 300 s (deadline), 23 turns, 86.4k tok, 1 trunc | KEEP_BASE | KEEP_BASE | KEEP_BASE (INV U 787.0 ≤ base 810.4 + 10) |
| H2 | ✓ 62 s, 11 turns, 24.4k tok | ✓ 51 s, 9 turns, 18.7k tok | ✓ 215 s, 16 turns, 56.4k tok, 1 trunc | adopted INV's edit | **ADOPT** | **ADOPT INV** (U 787.5 vs 725.6) |
| H3 | ✓ 54 s, 7 turns, 15.0k tok | KEEP_BASE | KEEP_BASE | KEEP_BASE | KEEP_BASE | KEEP_BASE* |
| H4 | ✓ 39 s, 11 turns, 21.3k tok | ✓ 42 s, 11 turns, 21.8k tok | ✓ 188 s, 15 turns, 41.7k tok, 1 trunc | KEEP_BASE | KEEP_BASE | KEEP_BASE (INV U 761.4 ≤ base 766.8 + 10) |

\* Recorded as "no candidate passed trusted verification"; the correct reason is that no role proposed an edit. Message fixed in `d54dbf1` after the run.

H1 and H2 were run once and shared by both arms (same base harness); after H2 both arms held the same harness, so H3 and H4 were shared too. **Both promotion rules made the same choices on every task**, so H2 and H3 end with the identical harness `0f8a6dd812` and their held-out rows are identical.

**The adopted harness** changes one sentence of the system prompt (always-on text 1,236 vs 1,090 characters; no guardrails, memory, tool guidance or skills):
> When a verification command is given, run it **first** with run_command **to reproduce the failure before reading or modifying files. Inspect the failure and related docstrings to implement a complete fix, then re-run verification.**

**DOM candidates regressed every time** (188–300 s, 1 truncation each, up to 86k tokens): they added long always-on rules and on-demand skills, and the worker spent many more turns. The composer rejected them each time; H3's utility would also have rejected them. Note: the composer's notes call the worker's `read_skill` calls "malformed/empty"; the trajectories show these calls **succeeded** — the skill name was simply not recorded as the call target, so the evidence rendered them with an empty target. Fixed after the run (the trajectory now records the skill name); the composer's decisions were also supported by time and token evidence, so the conclusions do not change.

## 3. Held-out evaluation (H5–H10, seeds 303 and 404)

| Arm | Verified | Wall (sum) | Tokens | Truncations | Mean U |
|---|---:|---:|---:|---:|---:|
| H0 (default) | **7/12** | **601 s** | **155,144** | 0 | **429.0** |
| H2 = H3 (adapted) | 6/12 | 829 s | 159,334 | 1 | 330.6 |

| Task | H0 s303 | H0 s404 | Adapted s303 | Adapted s404 |
|---|---|---|---|---|
| H5 intervals | ✓ 36 s | ✓ 37 s | ✓ 36 s | ✓ 34 s |
| H6 durations | ✓ 101 s | ✗ holdout (107 s) | ✗ holdout (100 s, 29.4k tok) | ✗ deadline 300 s, 1 trunc |
| H7 rolling mean | ✓ 30 s | ✓ 30 s | ✓ 28 s | ✓ 24 s |
| H8 slugify | ✓ 27 s | ✓ 26 s | ✓ 53 s | ✓ 27 s |
| H9 semver | ✗ holdout (84 s) | ✗ holdout (44 s) | ✗ holdout (71 s) | ✗ holdout (84 s) |
| H10 ipv4 | ✗ holdout (40 s) | ✗ holdout (39 s) | ✗ holdout (36 s) | ✗ holdout (35 s) |

- The only verified difference is one run (H6, seed 303). With 12 runs per arm and run-to-run variance of the same order (sections 6–9 of the Phase 2B report), this is **no evidence of either improvement or regression** in verified completion.
- The adapted harness was slower overall (+228 s), almost entirely from one H6 run that hit the deadline.
- H9 and H10 failed the hidden holdout in all 8 runs for both harnesses: every run passed the visible tests but not the holdout, which checks rules stated only in the docstring (which individual holdout assertion failed is not recorded). Neither the default nor the adapted prompt makes the 12B worker implement the full specification.

## 4. Conclusions

1. **The pipeline works end to end**: proposals were valid on the first call (0 fallbacks, 0 cloud failures), candidates were executed on the same task, both promotion rules ran, harnesses were frozen and evaluated on unseen tasks.
2. **One Harness² step per task did not improve the local worker on held-out tasks.** The adopted edit (verify first, then a complete fix) helped on the task where it was found (H2: 9 vs 11 turns, −23% tokens) but did not transfer: verified completion 6/12 vs 7/12, slower in total.
3. **Verifier gating and the standard composer agreed on every decision** in this run, so the H2-vs-H3 comparison (false-promotion rate) is not informative yet; it needs tasks where the composer is tempted by a candidate that fails hidden tests.
4. **Knowledge-heavy edits (DOM: long rules and skills) hurt a 12B worker** with a 16K context: more tokens, truncations and malformed skill calls. Short procedural edits are the only kind that survived.
5. **The remaining failure mode is specification coverage** (follow-up: `docs/SPEC_CONTRACT_RESULTS.md` — cloud-written contract tests raise verified completion from 7/18 to 16/18, p = 0.012): H6, H9 and H10 pass visible tests but miss rules stated only in the docstring. This is where a next step should aim (for example, a worker-side checklist step that enumerates docstring rules before editing, measured with the same held-out protocol).

**Limits:** 4 adaptation tasks and 6 held-out tasks, 2 seeds, one recursion step (k=1), one editor model, synthetic single-function Python tasks; no statistical significance is claimed.
