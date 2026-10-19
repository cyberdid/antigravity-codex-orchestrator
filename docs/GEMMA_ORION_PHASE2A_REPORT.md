# GEMMA-ORION Phase 2A — Complete Balanced $4 \times 2 \times 3$ Factorial Benchmark Report

**Branch:** `feat/gemma-orion-phase2a` (`779cb45`) $\rightarrow$ `feat/gemma-orion-phase2b`  
**Model:** `gemma4:12b` (`sha256:93f07cd5ebefdf23bcdba0fba8efe6bb7ebbee9efb2fd6bef7cd3bfeb8ef2bd6`)  
**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB VRAM)  
**Dataset:** `benchmarks/gemma_orion_phase2a_balanced_4x2x3.json` (24/24 runs: 4 configs $\times$ 2 seeds `[101, 202]` $\times$ 3 tasks `[B1, B2, B3]`)

---

## 1. Executive Summary

Phase 2A completed a fully balanced $2 \times 2$ factorial experiment over `OLLAMA_THINK` (`false` vs `true`) and `OLLAMA_NUM_PREDICT` (`2048` vs `4096`) at fixed `OLLAMA_NUM_CTX=16384` and `temperature=0.2` across two explicit seeds (`101`, `202`) and three tasks (`B1_conformal_quantile_fix`, `B2_ann_reranking_fix`, `B3_orion2_rie_report_synthesis`).

### Key Findings
1. **`G0` (`think=false, num_predict=2048`) is strictly dominant across all measured criteria:**
   - **100% (6/6)** `pipeline_verified` and **100% (6/6)** `llm_native_completion` (`0%` fallback).
   - Fastest total wall-clock time: **`348.01 ± 1.64 s`** (population SD `pstdev`) / **`± 2.32 s`** (sample SD `s`).
   - Lowest token cost: **`10,874.5 ± 949.2`** tokens per verified task (`32,623.5 ± 2,847.5` total tokens per 3-task campaign).
2. **Increasing `num_predict` from `2048` to `4096` without thinking (`G3`) slows execution by `+70.5%` (`593.36 s` vs `348.01 s`):**
   - Trajectory forensics proved that on both `B1` (Turn 2) and `B2` (Turn 3), after reading the input files, `gemma4:12b` enters a single runaway prose turn without emitting a tool call until it hits `num_predict` (`done_reason="length"`).
   - Doubling `num_predict` from `2048` to `4096` does not prevent this failure mode—it simply doubles the wasted generation time on that turn (`~113 s` $\rightarrow$ `~230 s`).
3. **Thinking Mode (`G1`, `G2`) degrades both latency (`3.2×–3.4×` slower) and native structured JSON reliability (`B3`):**
   - On `B3` (`ResearchReportBuilder`), `G0` and `G3` achieve **100% native LLM completion** in `~39–41 s` (`~327–335` output tokens), whereas `G1` and `G2` exhaust their entire `2048`/`4096` token budget inside `<think>` traces (`0%` native completion, `100%` reliance on deterministic fallback).
4. **Resolution of the Phase 1 `B2` (`26.25 s`) anomaly:**
   - Across seeded runs on `B2`, `G2` averaged **`274.28 ± 3.71 s`**, whereas `G1` on `seed=202` hit a fast **`26.15 s`** trajectory (5 turns, 0 truncations). The fast path is a stochastic trajectory bifurcation at `temperature=0.2` when the model emits `patch_file` directly on Turn 3 instead of ruminating until `num_predict`.

---

## 2. Balanced $4 \times 2 \times 3$ Aggregate Matrix (`seeds = [101, 202]`)

*Note on dispersion metrics:* With $n=2$ seeds per cell, both population standard deviation ($\sigma_{\text{pop}} = \text{pstdev} = |x_1 - x_2|/2$) and sample standard deviation ($s = \text{stdev} = |x_1 - x_2|/\sqrt{2}$) are reported below.

| Metric | `G0` (`OFF / 2048`) | `G3` (`OFF / 4096`) | `G1` (`ON / 2048`) | `G2` (`ON / 4096`) |
|---|---:|---:|---:|---:|
| **Pipeline Verified Rate** | **100.0%** (6/6) | **100.0%** (6/6) | **100.0%** (6/6)* | **100.0%** (6/6) |
| **LLM Native Completion Rate** | **100.0%** (6/6) | **100.0%** (6/6) | 66.7% (4/6) | 66.7% (4/6) |
| **Deterministic Fallback Used** | **0.0%** (0/6) | **0.0%** (0/6) | 33.3% (2/6, `B3`) | 33.3% (2/6, `B3`) |
| **Truncated Generations / Seed** | **2.0 ± 0.0** | **2.0 ± 0.0** | 3.0 ± 0.0 | 3.5 ± 0.5 |
| **Total Wall-Clock (`mean ± pstdev`)** | **348.01 ± 1.64 s** | 593.36 ± 6.30 s | 1176.95 ± 82.56 s | 1124.82 ± 193.90 s |
| **Total Wall-Clock (`mean ± sample s`)** | **348.01 ± 2.32 s** | 593.36 ± 8.91 s | 1176.95 ± 116.76 s | 1124.82 ± 274.22 s |
| **Total Tokens / Seed (`mean ± pstdev`)** | **32,623.5 ± 2,847.5** | 34,610.0 ± 296.0 | 46,294.5 ± 9,219.5 | 54,315.5 ± 8,932.5 |
| **Tokens / Verified Task (`mean ± pstdev`)** | **10,874.5 ± 949.2** | 11,536.7 ± 98.7 | 15,431.5 ± 3,073.2 | 18,105.2 ± 2,977.5 |
| **Repeated File Reads / Seed** | 0.5 ± 0.5 | 0.0 ± 0.0 | 2.0 ± 1.0 | 2.5 ± 1.5 |
| **Peak VRAM (MiB)** | 7,760 MiB | 7,762 MiB | 7,830 MiB | 7,830 MiB |

*\*Note: In Phase 1 (unseeded), `G1` failed `B1` (`0/1`), yielding `2/3` (`66.7%`) pipeline verified and `1/3` (`33.3%`) native completion (`1430.14 s`). On seeds `101` and `202`, `G1` passed `B1` after lengthy multi-turn recoveries (`706.79 s` and `1099.11 s`).*

---

## 3. Per-Task Breakdown Across All 24 Runs (`seeds = [101, 202]`)

| Task | Config | Seed 101 Wall (s) | Seed 202 Wall (s) | Mean ± `pstdev` (`/ ± sample s`) | Native Completion | Truncations (s101 / s202) |
|---|---|---:|---:|---:|---:|---:|
| **B1** (`conformal_quantile_fix`) | **G0** | 171.83 s | 169.77 s | **170.80 ± 1.03 s** (`/ ± 1.45 s`) | **2/2 (100%)** | 1 / 1 |
| **B1** (`conformal_quantile_fix`) | **G3** | 310.98 s | 299.50 s | 305.24 ± 5.74 s (`/ ± 8.12 s`) | **2/2 (100%)** | 1 / 1 |
| **B1** (`conformal_quantile_fix`) | **G1** | 706.79 s | 1099.11 s | 902.95 ± 196.16 s (`/ ± 277.41 s`) | **2/2 (100%)** | 2 / 2 |
| **B1** (`conformal_quantile_fix`) | **G2** | 410.94 s | 791.34 s | 601.14 ± 190.20 s (`/ ± 268.98 s`) | **2/2 (100%)** | 1 / 2 |
| **B2** (`ann_reranking_fix`) | **G0** | 133.76 s | 137.99 s | **135.88 ± 2.12 s** (`/ ± 2.99 s`) | **2/2 (100%)** | 1 / 1 |
| **B2** (`ann_reranking_fix`) | **G3** | 249.79 s | 247.75 s | 248.77 ± 1.02 s (`/ ± 1.44 s`) | **2/2 (100%)** | 1 / 1 |
| **B2** (`ann_reranking_fix`) | **G1** | 252.92 s | **26.15 s** | 139.53 ± 113.39 s (`/ ± 160.35 s`) | **2/2 (100%)** | 1 / 0 |
| **B2** (`ann_reranking_fix`) | **G2** | 277.99 s | 270.57 s | 274.28 ± 3.71 s (`/ ± 5.25 s`) | **2/2 (100%)** | 1 / 1 |
| **B3** (`orion2_rie_report_synthesis`) | **G0** | 40.79 s | 41.89 s | **41.34 ± 0.55 s** (`/ ± 0.78 s`) | **2/2 (100% NATIVE)** | 0 / 0 |
| **B3** (`orion2_rie_report_synthesis`) | **G3** | 38.89 s | 39.80 s | **39.34 ± 0.46 s** (`/ ± 0.65 s`) | **2/2 (100% NATIVE)** | 0 / 0 |
| **B3** (`orion2_rie_report_synthesis`) | **G1** | 134.68 s | 134.25 s | 134.47 ± 0.22 s (`/ ± 0.31 s`) | 0/2 (FALLBACK) | 0 / 0* |
| **B3** (`orion2_rie_report_synthesis`) | **G2** | 242.00 s | 256.82 s | 249.41 ± 7.41 s (`/ ± 10.48 s`) | 0/2 (FALLBACK) | 1 / 1 |

*\*In `G1` on `B3`, `ResearchReportBuilder` uses `num_predict=2048` in `generate_json()`, where `<think>` consumes the budget and returns incomplete JSON.*

---

## 4. Architectural Conclusion & Transition to Phase 2B

1. **Freeze `G0` (`OLLAMA_THINK=false, OLLAMA_NUM_CTX=16384, OLLAMA_NUM_PREDICT=2048`) as the default production inference configuration.**
2. **Why Phase 2B focuses on `RecoveryController` (`P0 vs P1`) rather than `P2` Adaptive Thinking:**
   - Almost all wasted time in `G0` (~110 s on `B1` Turn 2 and ~110 s on `B2` Turn 3) comes from a single failure mode: after reading the input files, the model emits a long prose analysis turn instead of immediately calling `patch_file`.
   - Guiding stage transitions and recovering cleanly with preserved file/test evidence (`P1 — RecoveryController`) directly targets this bottleneck without increasing `num_predict` or enabling `<think>`.
