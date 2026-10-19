# GEMMA-ORION Phase 2B — Recovery Controller PoC & 4-Policy Ablation Report

**Branch:** `feat/gemma-orion-phase2b`  
**Model:** `gemma4:12b` (`sha256:93f07cd5ebefdf23bcdba0fba8efe6bb7ebbee9efb2fd6bef7cd3bfeb8ef2bd6`)  
**Inference Settings (held strictly constant):** `OLLAMA_THINK=false`, `OLLAMA_NUM_CTX=16384`, `OLLAMA_NUM_PREDICT=2048`, `temperature=0.2`  
**Benchmark Artifacts:**
- **Primary Unhinted 4-Policy Ablation (`P0, P1_REACTIVE, P1_STAGE, P1`, 24 runs, `git_commit: ff40f43`, `git_dirty: false`):** `benchmarks/gemma_orion_phase2b_ablation_unhinted.json`
- Initial hinted `P0 vs P1` run (`12` runs): `benchmarks/gemma_orion_phase2b_p0_vs_p1_clean.json`
- Diagnostic host-`PATH` fault stress run (`12` runs, `pytest` missing on default system `python3` in `WorkspaceToolSuite(sandbox_backend="host")`): `benchmarks/gemma_orion_phase2b_p0_vs_p1_diagnostic.json`

---

## 1. Executive Summary & Mechanism Attribution

Phase 2B tested whether improving agent orchestration and recovery around `G0` (`think=false, num_predict=2048`) eliminates post-read truncated generations and reduces latency/token waste without changing the model, thinking mode, or `num_predict` budget.

Following an internal audit of the initial `P0 vs P1` run (`50380b6`), we identified that `P1` combined two distinct mechanisms:
1. **Proactive stage-state annotations (`P1_STAGE`):** Appending a 1-line `[STAGE STATE: ...]` footer to tool observations indicating the next required workflow stage (`read remaining input files -> patch target file -> run verification command -> summarize`).
2. **Reactive recovery controller (`P1_REACTIVE`):** Injecting `[RECOVERY CONTROLLER ...]` prompts after truncated/empty turns, premature non-tool stops before verification, failed edits, or unchanged repeated file reads.

To isolate the causal contribution of each mechanism and eliminate task spoilers, we removed all `# BUG:` comments from source files, reduced task descriptions to `"satisfy its docstring specification and pass python3 -m unittest -q"`, counterbalanced policy execution order across seeds (`303`: `P0 -> P1_REACTIVE -> P1_STAGE -> P1`; `404`: `P1 -> P1_STAGE -> P1_REACTIVE -> P0`), and ran a full **24-run ($4 \text{ policies} \times 2 \text{ seeds} \times 3 \text{ tasks}$) unhinted ablation** (`benchmarks/gemma_orion_phase2b_ablation_unhinted.json`):

| Policy | Verified Rate | Native Rate | Truncations / Seed | Recovery Attempts / Seed | Stage Hints / Seed | Output Tokens / Seed (`mean ± sample SD`) | Total Wall-Clock / Seed (`mean ± sample SD`) | Seed Range `[min, max]` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`P0` (Baseline `G0`)** | **100.0%** (6/6) | **100.0%** (6/6) | 3.0 ± 0.0 (`6` tot) | 3.0 ± 0.0 | 0.0 ± 0.0 | 8,714.5 ± 27.6 | 534.68 ± 20.90 s (`pstdev: 14.78 s`) | `[519.90, 549.45]` s |
| **`P1_REACTIVE` (Reactive only)** | **100.0%** (6/6) | **100.0%** (6/6) | 2.0 ± 0.0 (`4` tot) | 2.0 ± 0.0 | 0.0 ± 0.0 | 6,439.0 ± 425.7 | 401.44 ± 23.08 s (`pstdev: 16.32 s`) | `[385.12, 417.77]` s |
| **`P1_STAGE` (Stage hints only)** | **100.0%** (6/6) | **100.0%** (6/6) | **0.0 ± 0.0 (`0` tot)** | **0.0 ± 0.0** | 27.5 ± 0.7 | **2,251.0 ± 5.7** | **202.33 ± 6.16 s** (`pstdev: 4.35 s`) | **`[197.97, 206.68]` s** |
| **`P1` (`P1_STAGE + P1_REACTIVE`)** | 83.3% (5/6) | 83.3% (5/6) | **0.0 ± 0.0 (`0` tot)** | **0.0 ± 0.0** | 23.0 ± 1.4 | 2,525.5 ± 228.4 | 393.31 ± 147.36 s (`pstdev: 104.20 s`) | `[289.11, 497.51]` s |

### Key Findings from the 4-Policy Ablation
1. **Without `# BUG:` spoilers, `P0` truncates on 100% of tasks (`6/6`):** Once `# BUG:` comments and description spoilers were removed, baseline `P0` hit `done_reason="length"` (`2048` tokens) on **every single task (`H1, H2, H3`) across both seeds** (`3.0` truncations/seed, `534.68 ± 20.90 s`).
2. **`P1_STAGE` (proactive `[STAGE STATE]` tool annotations) delivers the entire speedup and 100% truncation elimination:**
   - `P1_STAGE` achieved **`100%` (6/6) verified completion**, reduced `truncated_generations` from **`3.0` to `0.0`**, cut output tokens by **`-74.2%`** (`8,714.5 -> 2,251.0`), and reduced wall-clock time by **`-62.2%` (`2.64×` faster: `534.68 s -> 202.33 s`)** with a tight sample SD of **`±6.16 s`**.
3. **`P1_REACTIVE` alone provides a modest `-24.9%` speedup (`534.68 s -> 401.44 s`):** Because reactive recovery only fires *after* a truncated turn has already consumed `2048` tokens (~90–110 s) or after a failed verification command, `P1_REACTIVE` still suffers `2.0` truncations/seed (`H1` and `H2`), though its structured evidence-preserving recovery prompt saves ~133 s per seed compared to `P0`'s generic nudge.
4. **Combining inline `[RECOVERY CONTROLLER: repeated_file_read]` warnings with `[STAGE STATE]` (`P1` full) can over-constrain multi-step repairs:** On `Seed 303 | H2_pareto_judge_fix`, when the model re-read files after a partial fix, `P1`'s reactive `[RECOVERY CONTROLLER: ... was already read and unchanged]` replacement of `[STAGE STATE]` perturbed the subsequent `patch_file` call (`Target content not found`), leading to a 300s timeout (`5/6` verified), whereas pure `P1_STAGE` completed the exact same seed and task in `75.46 s` (`6/6` verified).

---

## 2. Bug Fixes & Metric Definition Changes (Audit of `50380b6` -> `ff40f43` + Post-Ablation Tightening)

During the code review of commit `50380b6`, three state-tracking bugs in `RecoveryController` were identified, fixed, and covered by unit test `T17` (`test_t17_phase2b_review_bugfixes_and_policy_ablation`):

1. **Strict Verification Command Matching (`_is_verification_command`):**
   - *Bug in `50380b6`:* Any `run_command` with `success=True` (including `ls -R` or `python3 -c "import sys; print(sys.path)"`) set `verification_passed = True` and incremented `successful_verifications`.
   - *Fix:* Added `_is_verification_command(cmd)`. In `ff40f43` (used during the 24-run ablation), `_is_verification_command` checked exact match or `unittest`/`pytest` runner match. In the final post-ablation refinement, `_is_verification_command` checks **exact normalized string equality (`norm_cmd == norm_verif`) first** when `subtask.verification_command` is set (so compound commands like `cd . && python3 -m unittest -q` generated by Architect in Orion are accepted, while appended pipes `| tail -20`, `|| true`, `python3 -c "assert True"`, or partial test runs `unittest test_x.py` are rejected), and rejects `|` / `&&` in the fallback branch when `verification_command` is empty.
2. **Invalidating `clean_read_files` on File Modification (`patch_file` / `write_file`):**
   - *Bug in `50380b6`:* Successful `patch_file` / `write_file` did not remove the modified path from `clean_read_files`, so a legitimate post-edit `read_file` was misclassified as an unchanged repeated read.
   - *Fix:* Successful `patch_file` / `write_file` removes `path` from `clean_read_files` and records it in `modified_since_read_files` until re-read.
3. **Resetting `consecutive_non_tool_turns` on Tool Calls in `P0`:**
   - *Bug in `50380b6`:* `P0` checked cumulative `self.recovery_attempts < 2` across the entire subtask rather than consecutive non-tool turns per episode.
   - *Fix:* `P0` and `P1_STAGE` check `self.consecutive_non_tool_turns < 2` (which resets to `0` on every turn with valid tool calls), matching baseline `G0` (`empty_turn_nudges`) behavior exactly while still incrementing `recovery_attempts` for telemetry.

### Important Note on Metric Definitions Across Campaigns
- **`repeated_file_reads` vs `total_repeated_file_reads`:** Starting in `ff40f43`, `repeated_file_reads` counts only **unchanged repeated reads** (`recovery_controller.repeated_reads_count`, i.e., re-reading a file without an intervening edit), while `total_repeated_file_reads` preserves the Phase 2A definition (`len(files_read_list) - unique_files_read`). Do not directly compare `repeated_file_reads` from `gemma_orion_phase2b_ablation_unhinted.json` against Phase 2A or `50380b6` without accounting for this change.
- **`successful_verifications` and `llm_native_completion`:** Starting in `ff40f43`, `successful_verifications` only increments when `run_command` satisfies `_is_verification_command(cmd)`. Under the post-ablation exact-match rule (`norm_cmd == norm_verif`), a model that verifies via `python3 -m unittest -q test_x.py` instead of the exact `SubTask.verification_command` (`python3 -m unittest -q`) will not increment `successful_verifications` until it runs the exact `verification_command` (which `[STAGE STATE]` explicitly prompts it to run).
- **Correction Regarding the Diagnostic Run (`gemma_orion_phase2b_p0_vs_p1_diagnostic.json`):** `WorkspaceToolSuite` defaults to `sandbox_backend="host"` (not `DockerSandbox`). In the diagnostic run, `run_command("python3 -m pytest -q")` failed because `subprocess.Popen(..., shell=True)` on the host resolved `/home/cyberdid/.local/share/uv/python/cpython-3.13.14-linux-x86_64-gnu/bin/python3` (which did not have `pytest` installed) before `.venv/bin` was prepended to `PATH`.

---

## 3. Per-Task Breakdown of the Unhinted 4-Policy Ablation (`benchmarks/gemma_orion_phase2b_ablation_unhinted.json`)

| Task | Policy | Seed 303 Wall (s) | Seed 404 Wall (s) | Mean ± `sample SD` (`pstdev`) | Output Tokens (s303 / s404) | Truncations (s303 / s404) | Stage Hints (s303 / s404) |
|---|---|---:|---:|---:|---:|---:|---:|
| **H1** (`fdr_step_up_fix`) | `P0` | 184.49 s | 169.48 s | 176.99 ± 10.61 s (`7.51 s`) | 2,869 / 2,906 | 1 / 1 | 0 / 0 |
| **H1** (`fdr_step_up_fix`) | `P1_REACTIVE` | 153.69 s | 160.41 s | 157.05 ± 4.75 s (`3.36 s`) | 2,521 / 2,535 | 1 / 1 | 0 / 0 |
| **H1** (`fdr_step_up_fix`) | **`P1_STAGE`** | **48.55 s** | **60.75 s** | **54.65 ± 8.63 s (`6.10 s`)** | **621 / 780** | **0 / 0** | 8 / 7 |
| **H1** (`fdr_step_up_fix`) | `P1` | 125.67 s | 52.53 s | 89.10 ± 51.72 s (`36.57 s`) | 780 / 618 | 0 / 0 | 7 / 7 |
| **H2** (`pareto_judge_fix`) | `P0` | 176.66 s | 171.76 s | 174.21 ± 3.46 s (`2.45 s`) | 2,731 / 2,733 | 1 / 1 | 0 / 0 |
| **H2** (`pareto_judge_fix`) | `P1_REACTIVE` | 177.80 s | 159.08 s | 168.44 ± 13.24 s (`9.36 s`) | 2,786 / 2,616 | 1 / 1 | 0 / 0 |
| **H2** (`pareto_judge_fix`) | **`P1_STAGE`** | **75.46 s** | **79.60 s** | **77.53 ± 2.93 s (`2.07 s`)** | **854 / 862** | **0 / 0** | 13 / 13 |
| **H2** (`pareto_judge_fix`) | `P1` | ❌ 300.01 s | 154.01 s | 227.01 ± 103.24 s (`73.00 s`) | 979 / 1,289 | 0 / 0 | 8 / 10 |
| **H3** (`group_kfold_leakage_fix`) | `P0` | 188.30 s | 178.65 s | 183.48 ± 6.82 s (`4.82 s`) | 3,095 / 3,095 | 1 / 1 | 0 / 0 |
| **H3** (`group_kfold_leakage_fix`) | `P1_REACTIVE` | 86.28 s | 65.63 s | 75.96 ± 14.60 s (`10.32 s`) | 1,433 / 987 | 0 / 0 | 0 / 0 |
| **H3** (`group_kfold_leakage_fix`) | **`P1_STAGE`** | **82.68 s** | **57.63 s** | **70.15 ± 17.71 s (`12.52 s`)** | **780 / 605** | **0 / 0** | 7 / 7 |
| **H3** (`group_kfold_leakage_fix`) | `P1` | 71.82 s | 82.57 s | 77.19 ± 7.60 s (`5.37 s`) | 605 / 780 | 0 / 0 | 7 / 7 |

---

## 4. Synthetic Unit & Integration Verification (`T12–T17`)

All `31/31` unit and integration tests pass across `tests/test_gemma_orion_phase1.py` (`T01–T17`) and `tests/test_native_worker.py`:
- **`T12`:** Verifies evidence-preserving recovery prompt on `done_reason="length"` and hard termination at `max_recovery_attempts=3`.
- **`T13`:** Verifies detection and warning when re-reading an unmodified file.
- **`T14`:** Verifies recovery when the model stops with prose before running `verification_command`.
- **`T15`:** Verifies preservation of assertion failure tracebacks across subsequent recovery turns.
- **`T16`:** Verifies deterministic cycle termination and `P0` baseline backward compatibility.
- **`T17`:** Verifies the 3 review bug fixes (`ls -R`, `cmd | tail -20`, `cmd || true`, `python3 -c "assert True"`, and partial `unittest test_x.py` rejected while compound `cd . && python3 -m unittest -q` is accepted; post-`patch_file` `read_file` not flagged as unchanged repeated read; `P0` resets `consecutive_non_tool_turns` after tool calls; `P1_REACTIVE` vs `P1_STAGE` separation).

---

## 5. Conclusions, Limitations & Transition to Harness² (`OrchestratorSubstrate`)

1. **Primary Scientific Conclusion:**
   - The dominant failure mode of `gemma4:12b` under `G0` (`think=false, num_predict=2048`) on multi-file repair tasks is **stage-transition ambiguity after reading input files** (generating thousands of prose tokens instead of calling `patch_file`).
   - **Proactive 1-line stage-state annotations (`P1_STAGE`)** completely eliminate this failure mode (`3.0 -> 0.0` truncations/seed) and reduce wall-clock latency by **`-62.2%` (`534.68 s -> 202.33 s`)** while maintaining **`100%` (6/6) verified completion**.
   - **Reactive recovery (`P1_REACTIVE`)** should be kept as a **fallback safety net** for genuine non-tool turns/truncations rather than overriding `[STAGE STATE]` on normal tool observations.
2. **Scope Limitations:**
   - Phase 2B evaluated 3 single-module repair tasks (`H1–H3`, each with 3 input files and 1 target file) across 2 seeds (`24` runs in the 4-policy ablation). For **Harness²**, the benchmark suite should be expanded with multi-file edit tasks and multi-defect tasks.
3. **Ready for Harness² (`docs/HARNESS2_INTEGRATION_DESIGN.md`):**
   - `RecoveryPolicyConfig` (`P0`, `P1_STAGE`, `P1_REACTIVE`, `P1`) now provides clean, orthogonal policy knobs ready for RIE-gated **Harness² (`OrchestratorSubstrate`)** evaluation.


---

## 6. Addendum: Trusted Verification Re-run and Multi-Module Task H4 (`0c1f186`)

**Why a re-run was needed.** In all 24 runs of the unhinted ablation (`ff40f43`) the model verified with `python3 -m unittest -q <test_file>.py`, never with the exact `verification_command`. Under the exact-match rule introduced in `2071a1b` none of those runs would count as verified, and `P1_STAGE` would never see a `PASSED` state. Sections 1–3 therefore describe `ff40f43` behaviour, not HEAD. In `0c1f186` the worker passes a controller-owned `verify_fn`: whenever the model runs any other command, the controller runs the real `verification_command` and uses that result (`successful_verifications` now counts these trusted runs). Failing output is excerpted from the tail, and failed tool error text is stored in `turn_trajectory`.

**New task H4** (`H4_weighted_mean_two_module_fix`): defects in two modules (`weights.normalize` divides by `max`, `summary.weighted_mean` divides by `len` and lacks the length check). Offline check: fixing either module alone fails the visible test; both fixes pass visible and holdout tests.

**Run:** `benchmarks/gemma_orion_phase2b_trusted_verification.json`, `git_commit=0c1f186`, `git_dirty=false`, policies `P0` and `P1_STAGE`, seeds `303, 404`, tasks `H1–H4`, counterbalanced order (16 runs). `P1_REACTIVE` and `P1` were not re-run; their Section 3 results remain valid measurements of `ff40f43`.

| Policy | Scope | Verified | Native | Truncations | Wall-clock / seed | Output tokens / seed | Total tokens / seed |
|---|---|---:|---:|---:|---:|---:|---:|
| `P0` | H1–H3 | 6/6 | 6/6 | 6 | 543 s | 9,010 | 71,848 |
| `P1_STAGE` | H1–H3 | 6/6 | 6/6 | 0 | 177 s | 2,220 | 67,480 |
| `P0` | H1–H4 | **8/8** | 8/8 | 8 | 691 s | 11,526 | 92,708 |
| `P1_STAGE` | H1–H4 | **6/8** | 6/8 | 0 | 218 s | 2,634 | 90,660 |

**H1–H3 replicate.** `P1_STAGE` again removes every truncation and is ~3× faster with equal verified completion.

**H4 fails under `P1_STAGE` on both seeds** (identical trajectories, stopped by the cycle detector after 12 turns), while `P0` solves it on both seeds. Trajectory forensics:

1. `P1_STAGE` acts early: it patches `weights.py` and `summary.py` (both main defects fixed) but omits the length-mismatch `ValueError`, so 1 of 2 visible tests still fails.
2. The `[STAGE STATE]` note after that failure cannot help: the tail of unittest output is `Ran 2 tests ... FAILED (failures=1)`, which carries no assertion text, and the note always names `affected_files[0]` (`weights.py`) as the file to fix.
3. The model re-reads `summary.py` (the correct file) three times with identical output and the cycle detector stops the attempt.
4. `P0`'s truncated 2048-token turn was not pure waste on H4: after it the model added both validation checks in one pass.

**Conclusions (revised).**
- Stage hints remove the post-read runaway generation and are ~3× faster on single-module repairs.
- They are not yet safe as a default: on a two-module task they trade thoroughness for speed and give no usable guidance after a failed test. Before `P1_STAGE` becomes the default or enters Harness², the failure hint must (a) excerpt the failing test name and assertion line rather than the runner summary, and (b) not hard-code `affected_files[0]`.
- Evidence base: 4 tasks × 2 seeds; seeds produce near-identical trajectories at `temperature=0.2`, so effective sample size is closer to 4 tasks than 8 runs.

---

## 7. Addendum: Failure Hints that Name the Failing Test and File (`6eb79f0`)

**Change.** After a failed verification the `[STAGE STATE]` hint now carries the `FAIL:`/`ERROR:` headers and exception lines (pytest `E`/`FAILED` lines too) instead of the runner summary, and names the affected files that appear in the traceback, or all affected files when the traceback only shows the test file. Tests: T20 (built from real H4 unittest output).

**Run:** `benchmarks/gemma_orion_phase2b_stage_failure_hints.json`, `git_commit=6eb79f0`, `git_dirty=false`, `P1_STAGE`, seeds `303, 404`, tasks `H1–H4` (8 runs). `P0` was not re-run: it emits no hints, so this change does not affect it, and the Section 6 `P0` numbers (`0c1f186`) remain the baseline.

| Policy (commit) | Scope | Verified | Truncations | Wall-clock / seed | Output tokens / seed | Total tokens / seed |
|---|---|---:|---:|---:|---:|---:|
| `P0` (`0c1f186`) | H1–H4 | 8/8 | 8 | 691 s | 11,526 | 92,708 |
| `P1_STAGE` (`0c1f186`) | H1–H4 | 6/8 | 0 | 218 s | 2,634 | 90,660 |
| `P1_STAGE` (`6eb79f0`) | H1–H4 | **8/8** | **0** | **235 s** | 2,706 | 81,008 |

**H4 now passes on both seeds** (10 turns, 43–50 s, holdout passed). The trajectory is identical to the failing `0c1f186` runs up to turn 8, but the model now adds the length check in its first `summary.py` patch. The effective change is the hint after the *first* test run (turn 4): it now states `FAIL: test_length_mismatch_raises ... | AssertionError: ValueError not raised` and names both modules. The "repeated failure after a partial fix" path that looped in Section 6 did not occur in this run, so the improved hint for that path is covered by T20 but not yet by a live trajectory.

**H1–H3 unchanged** (6/6 verified, 0 truncations, 189 s/seed vs 177 s/seed at `0c1f186`).

**Current conclusion.** On H1–H4, `P1_STAGE` matches `P0` on verified completion (8/8), removes all truncations, and is ~2.9× faster with fewer total tokens. Remaining limits: 4 tasks, 2 seeds with near-identical trajectories, one task family (small Python repairs with a unit-test oracle). Before Harness², add tasks where a first fix is commonly incomplete, so the repeated-failure hint path is exercised live.

---

## 8. Addendum: Multi-Defect Tasks H5 and H6 (`9ae29e2`)

**Tasks.** H5 (`merge_intervals`: missing sort, `<` instead of `<=`, missing `ValueError`) and H6 (`parse_duration`: hour factor 60 instead of 3600, no validation; holdout also checks the unit-order rule stated in the docstring). One visible test per defect; offline, the original and every single-defect fix fail, only the full fix passes.

**Run:** `benchmarks/gemma_orion_phase2b_multi_defect.json`, `git_commit=9ae29e2`, `git_dirty=false`, `P0` and `P1_STAGE`, seeds `303, 404` (8 runs). Background load during the run: a niced user indexing service (~1 core) and a root `node` process (~1 core); both policies ran under it with counterbalanced order, so wall-clock is noisier than in Sections 6–7.

| Policy | Task | Verified | Native | Truncations | Wall-clock (s303 / s404) |
|---|---|---:|---:|---:|---|
| `P0` | H5 | 2/2 | 2/2 | 0 | 66 / 62 s |
| `P1_STAGE` | H5 | 2/2 | 2/2 | 0 | 40 / 42 s |
| `P0` | H6 | 1/2 | 0/2 | 1 | 57 s (stalled) / 300 s (deadline) |
| `P1_STAGE` | H6 | 0/2 | 0/2 | 3 | 300 / 300 s (deadline) |

**H5** was solved in one patch by both policies (all three defects at once), so it did not exercise the repeated-failure path either.

**H6 is beyond what `gemma4:12b` reliably does here, and the trajectories show why:**
1. **Truncation now happens inside tool calls.** All 3 `P1_STAGE` truncations are `write_file`/`patch_file` calls cut at 2048 tokens. Because the model emitted `content` before `path`, the truncated call arrives without `path` (or without `target_content`) and fails as an invalid-argument call. With `think=false` the model reasons *inside the file content*: the written code is long comment monologues (`# Wait, ...`, `# Let's try:`), in one case ending in `pass`.
2. **`P0` s303 never edits.** It alternates `read_file` and test runs for 16 turns and is stopped by the cycle detector.
3. **`P0` s404 produced a correct file** (visible and holdout tests pass) but only after a 123 s runaway turn and two ~55 s `write_file` turns full of commented reasoning, so it hit the 300 s deadline before stopping (`pipeline_verified=True`, `native=False`).

**What this means.**
- On H6 the bottleneck is reasoning, not tool-use discipline. Stage hints cannot supply that, and `P1_STAGE` is not better than `P0` on this task (0/2 vs 1/2; N too small to call either way).
- A globally larger `num_predict` remains the wrong fix (Phase 2A: `G3` at 4096 was 1.7× slower with no accuracy gain). The two targeted options this data supports are: (a) when a truncated turn contains a tool call, do not execute the malformed call; retry that one turn with a larger budget; (b) the Phase 2A escalation policy: switch to `think=true` only on a retry after a failed attempt, so reasoning goes to the thinking channel instead of into code comments.
- The 300 s per-task deadline is binding on hard tasks and should be reported alongside any H6-type result.
