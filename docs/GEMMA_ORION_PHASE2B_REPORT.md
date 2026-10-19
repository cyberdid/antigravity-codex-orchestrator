# GEMMA-ORION Phase 2B — Recovery Controller PoC (`P0` vs `P1`) Report

**Branch:** `feat/gemma-orion-phase2b`  
**Model:** `gemma4:12b` (`sha256:93f07cd5ebefdf23bcdba0fba8efe6bb7ebbee9efb2fd6bef7cd3bfeb8ef2bd6`)  
**Inference Settings (held strictly constant):** `OLLAMA_THINK=false`, `OLLAMA_NUM_CTX=16384`, `OLLAMA_NUM_PREDICT=2048`, `temperature=0.2`  
**Datasets:**
- Clean held-out benchmark (`unittest` in `DockerSandbox`): `benchmarks/gemma_orion_phase2b_p0_vs_p1_clean.json`
- Stress-test diagnostic benchmark (`pytest` missing in `DockerSandbox`): `benchmarks/gemma_orion_phase2b_p0_vs_p1_diagnostic.json`

---

## 1. Executive Summary & Hypothesis Verification

Phase 2B tested the hypothesis formulated after Phase 2A:

> **Hypothesis:** *Controlled stage-aware recovery and tool-observation guidance (`P1 — RecoveryController`) improves execution efficiency, eliminates truncated generations, and reduces latency/output token waste without changing the model, thinking mode, or `num_predict` budget.*

### Main Result: Hypothesis Confirmed
On 3 brand-new, held-out multi-file scientific R&D tasks (`H1_fdr_step_up_fix`, `H2_pareto_judge_fix`, `H3_group_kfold_leakage_fix`) across 2 seeds (`303`, `404`, 12 total runs):
- **100% elimination of truncated generations:** `truncated_generations` dropped from **`2.5 ± 0.5`** per 3-task run in `P0` (`5` truncations across 6 tasks) to **`0.0 ± 0.0`** in `P1` (**`0` truncations across all 6 tasks**).
- **`2.63×` speedup (`-62.0%` wall-clock time):** Total 3-task campaign time dropped from **`420.12 ± 50.67 s`** in `P0` to **`159.72 ± 1.76 s`** in `P1`.
- **`3.24×` reduction in generated output tokens (`-69.1%`):** Output tokens per 3-task campaign dropped from **`7,283.0 ± 790.0`** in `P0` to **`2,251.0 ± 4.0`** in `P1`.
- **`28.7×` reduction in run-to-run latency variance:** Population SD across seeds dropped from **`±50.67 s`** in `P0` to **`±1.76 s`** in `P1`.
- **Higher resilience under environment faults (Diagnostic Stress Run):** When `run_command` encountered missing `pytest` inside `DockerSandbox`, `P0` achieved only **`16.7%` (1/6)** `llm_native_completion` (`585.88 ± 30.00 s`), whereas `P1` achieved **`66.7%` (4/6)** `llm_native_completion` (`335.15 ± 70.60 s`, `-42.8%` time, `0` truncations).

---

## 2. Architecture of `RecoveryController` (`P1`)

Implemented in `antigravity_hybrid_orchestrator/workforce/recovery_controller.py` and wired into `AutonomousWorker._execute_native()` behind `OLLAMA_RECOVERY_POLICY=P0|P1` (or `AutonomousWorker(recovery_policy=...)`):

1. **Detection of Truncated & Empty Generations (`truncated_generation` / `empty_generation_no_tool_call`):**
   - Intercepts `done_reason == "length"` or empty non-tool turns and synthesizes a structured, bounded `[RECOVERY CONTROLLER (attempt/max)]` directive instead of a generic nudge.
2. **Detection of Stalled / Premature Non-Tool Turns (`missing_tool_call_before_verification`):**
   - If a subtask defines a required `verification_command` and the model emits a non-truncated text turn without calling a tool before verification has passed, `P1` catches the premature stop and directs the model to the next required action.
3. **Preservation of Tool Evidence & Repeated Read Prevention (`repeated_file_read`):**
   - Tracks `clean_read_files` (files already read and not modified since their last read), `edits_applied`, and `last_verification_error`.
   - When the model re-reads an unmodified file, `P1` logs a `repeated_file_read` event and injects `[RECOVERY CONTROLLER: <path> was already read and unchanged...]` into the tool observation.
4. **Stage-Transition Guidance (`annotate_tool_observations`):**
   - Appends a concise 1-line `[STAGE STATE: ...]` footer to tool observations indicating which input files remain unread, when all input files are in context (forbidding redundant `read_file` calls and requesting `patch_file`), when to run `verification_command`, and when verification has passed (requesting an immediate 2-sentence summary).
   - This directly prevents the Turn 3/Turn 4 post-read prose rumination discovered in Phase 2A.
5. **Explicit Recovery Attempt Limits (`max_recovery_attempts = 3`) & Safety Contract:**
   - Never modifies files or runs commands autonomously; never bypasses RIE or hidden holdout tests.
   - Declarative `RecoveryPolicyConfig` dataclass is ready for **Harness² (`OrchestratorSubstrate`)** mutation of `c4: tool_guidance`.

---

## 3. Synthetic Unit & Integration Verification (`T12–T16`)

Added to `tests/test_gemma_orion_phase1.py` (`30/30` tests passing across `test_gemma_orion_phase1.py` and `test_native_worker.py`):
- **`T12` (`test_t12_phase2b_p1_truncation_recovery_and_attempt_limits`):** Verifies evidence-preserving recovery prompt on `done_reason="length"` and hard termination at `max_recovery_attempts=3` (`recovery_limit_exceeded`).
- **`T13` (`test_t13_phase2b_p1_repeated_file_read_detection`):** Verifies detection and stage warning when re-reading an unmodified file.
- **`T14` (`test_t14_phase2b_p1_missing_tool_call_before_verification`):** Verifies recovery when the model stops with prose before running `verification_command`.
- **`T15` (`test_t15_phase2b_p1_failed_test_recovery_preserves_evidence`):** Verifies preservation of assertion failure tracebacks across subsequent recovery turns.
- **`T16` (`test_t16_phase2b_loop_detection_and_p0_backward_compatibility`):** Verifies deterministic cycle termination and exact `P0` baseline backward compatibility.

---

## 4. Controlled Benchmark on Independent Held-Out Tasks (`H1, H2, H3`)

### 4.1 Held-Out Tasks (3 Input Files + Hidden Post-Run Holdout Suite)
- **`H1_fdr_step_up_fix`:** Repair Benjamini-Hochberg step-up rank indexing (`1..m`) and monotone rejection rule (`k_max`) across `fdr_types.py`, `fdr_control.py`, `test_fdr_control.py`.
- **`H2_pareto_judge_fix`:** Repair RIE-gated filtering (`rie_accepted=True`) and Pareto dominance (`<=` on all, `<` on at least one) across `candidate_spec.py`, `pareto_judge.py`, `test_pareto_judge.py`.
- **`H3_group_kfold_leakage_fix`:** Repair group-disjoint K-fold assignment (`g % n_splits` by first-appearance group order) and minimum-group validation across `sample_record.py`, `group_splitter.py`, `test_group_splitter.py`.

### 4.2 Aggregate Comparison (`P0` vs `P1`, `seeds = [303, 404]`)

| Metric | `P0` (Baseline `G0`) | `P1` (`G0` + `RecoveryController`) | Delta (`P1` vs `P0`) |
|---|---:|---:|---:|
| **Pipeline Verified Rate** | **100.0%** (6/6) | **100.0%** (6/6) | `0.0%` (both 100%) |
| **LLM Native Completion Rate** | **100.0%** (6/6) | **100.0%** (6/6) | `0.0%` (`+50.0%` in fault stress test) |
| **Fallback Used** | **0.0%** (0/6) | **0.0%** (0/6) | `0.0%` |
| **Truncated Generations / Seed** | 2.5 ± 0.5 (`5` total) | **0.0 ± 0.0 (`0` total)** | **-100.0% (eliminated)** |
| **Recovery Nudges Triggered / Seed** | 2.5 ± 0.5 | **0.0 ± 0.0** | **-100.0% (prevented upstream)** |
| **Total Wall-Clock (`mean ± pstdev`)** | 420.12 ± 50.67 s | **159.72 ± 1.76 s** | **-62.0% (`2.63×` faster)** |
| **Total Wall-Clock (`mean ± sample s`)** | 420.12 ± 71.66 s | **159.72 ± 2.49 s** | **-28.7× lower SD** |
| **Output Tokens / Seed (`mean ± pstdev`)** | 7,283.0 ± 790.0 | **2,251.0 ± 4.0** | **-69.1% (`3.24×` fewer)** |
| **Total Tokens / Seed (`mean ± pstdev`)** | 42,238.5 ± 1,908.5 | 43,716.0 ± 29.0 | `+3.5%` (stage state notes) |
| **Peak VRAM (MiB)** | 7,832 MiB | 7,832 MiB | Identical |

### 4.3 Per-Task Comparison (`P0` vs `P1`, `seeds = [303, 404]`)

| Task | Policy | Seed 303 Wall (s) | Seed 404 Wall (s) | Mean ± `pstdev` (`/ ± sample s`) | Output Tokens (mean) | Truncations (s303 / s404) |
|---|---|---:|---:|---:|---:|---:|
| **H1** (`fdr_step_up_fix`) | `P0` | 150.72 s | 49.21 s | 99.96 ± 50.76 s (`/ ± 71.78 s`) | 1,705.0 | 1 / 0 |
| **H1** (`fdr_step_up_fix`) | **`P1`** | **62.63 s** | **59.10 s** | **60.86 ± 1.76 s** (`/ ± 2.49 s`) | **827.5** | **0 / 0** |
| **H2** (`pareto_judge_fix`) | `P0` | 161.76 s | 153.14 s | 157.45 ± 4.31 s (`/ ± 6.09 s`) | 2,736.5 | 1 / 1 |
| **H2** (`pareto_judge_fix`) | **`P1`** | **45.88 s** | **46.40 s** | **46.14 ± 0.26 s** (`/ ± 0.37 s`) | **643.5** | **0 / 0** |
| **H3** (`group_kfold_leakage_fix`) | `P0` | 158.31 s | 167.10 s | 162.70 ± 4.40 s (`/ ± 6.22 s`) | 2,841.5 | 1 / 1 |
| **H3** (`group_kfold_leakage_fix`) | **`P1`** | **52.97 s** | **52.45 s** | **52.71 ± 0.26 s** (`/ ± 0.37 s`) | **780.0** | **0 / 0** |

---

## 5. Architectural Transition Plan: Ready for Harness² (`OrchestratorSubstrate`)

With Phase 1 (`G0/G1/G2`), Phase 2A (`G0/G3/G1/G2` balanced $4 \times 2 \times 3$), and Phase 2B (`P0 vs P1` `RecoveryController`) complete:
1. **Production Baseline Updated:** `G0 + P1` (`OLLAMA_THINK=false`, `OLLAMA_NUM_CTX=16384`, `OLLAMA_NUM_PREDICT=2048`, `OLLAMA_RECOVERY_POLICY=P1`) is validated and ready.
2. **How Phase 2B Connects Directly to Harness² (`docs/HARNESS2_INTEGRATION_DESIGN.md`):**
   - Phase 2B proved that **small, structured changes to tool guidance and stage-recovery instructions (`RecoveryPolicyConfig`) yield a `2.63×` speedup and eliminate 100% of truncations** without touching model weights or increasing `num_predict`.
   - In the next phase (**Harness² + RIE-Gated Optimization**), `OrchestratorSubstrate` can treat `RecoveryPolicyConfig`, compact core system prompts, and on-demand skills (`c2–c4`) as the mutable **Surface Contract** optimized via `Propose -> Probe -> RIE Gate -> Pareto Efficiency Judge -> Compose`.
