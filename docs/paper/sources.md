# Evidence register for `manuscript.md`

Every `[E…]` tag in the manuscript points to a repository record below; every `[R…]` tag to an external source. "Checked" means the value in the manuscript was compared with the named file on the date given by the assistant that drafted the text (Claude Code); **an accountable human author must re-verify each entry before submission** and record name and date in the last column.

## Internal evidence (repository `cyberdid/antigravity-codex-orchestrator`, branch `feat/gemma-orion-phase2b`)

| ID | Record | Locator | Supports | Checked (assistant) | Verified by (human) |
|---|---|---|---|---|---|
| E1 | `benchmarks/spec_contract_experiment.json` (commit `cc938ca`) | `docs/SPEC_CONTRACT_RESULTS.md` §Setup, §Results, §Why the other policies did not help | R1: 7/18, 5/18, 16/18, 8/18; 10–1, p = 0.012; visible-pass/holdout-fail 11/10/0/10; task construction | 2026-10-20 | |
| E2 | same as E1 | §Why the other policies did not help | SPEC self-checks share blind spots; ASSIST triggered once in 18 runs | 2026-10-20 | |
| E3 | `benchmarks/spec_contract_control.json` (`24fbe35`), `benchmarks/spec_contract_v2.json` (`750647e`) | §Control (v1), §Final run | 21/21 → 15/21, 0–6, p = 0.031; H1 19/20, H3 11/12 wrong; v2 0 wrong tests; 2/12 → 10/12, 8–0, p = 0.0078; solved tasks 19/21 | 2026-10-20 | |
| E4 | `scripts/benchmark_gemma_orion_phase2b.py` (`_independent_checks`, `run_single_task`, `TASK_DEADLINE_S = 300`) | §Measurement note | measurement definition and the R1–R3 scoring asymmetry; corrected checks | 2026-10-20 | |
| E5 | `benchmarks/spec_contract_confirm_fresh.json` (`7a369ac`) | §Confirmatory run on fresh tasks | R4: 4/18 → 11/18, 8–1–9, p = 0.039; silent 14/18 → 3/18; H17 sensitivity 7/18 → 14/18; wall 929 s vs 4,014 s; tokens 256,017 vs 1,171,194; cloud 91,958; all 6 modules correct; pre-registration text (local ledger) | 2026-10-20 | |
| E6 | `benchmarks/spec_contract_confirm_all.json` (`2c59e38`, resumed from 36 runs at `7a369ac`) | §Secondary run | R5: 28/39 → 37/39, 10–1, p = 0.012; silent 11 → 0; 26/27 on solved tasks; wall 3,092 vs 5,273 s; tokens 614,904 vs 1,378,211; cloud ≈ 264k; pooled 32/57 → 48/57, silent 25/57 → 3/57 | 2026-10-20 | |
| E7 | `scripts/analysis/clustered_spec.py` on E5 + E6 | §Task-level analysis and cost accounting | Table 2; untimed generation; lower-bound cost ratios | 2026-10-20 | |
| E8 | `scripts/analysis/oracle_quality.py`, `benchmarks/oracle_quality.json` | §Oracle quality | 47 implementations, 6/91 failing pairs (H4, H9, docstring rules); 448 mutants, 333 holdout-killed, visible 205/333, contract 469/469, 84 contract-only, 31 survive all | 2026-10-20 | |
| E9 | `benchmarks/gemma_orion_phase2b_escalation_ladder.json` (`76576c0`) | `docs/GEMMA_ORION_PHASE2B_REPORT.md` §9 | no measured benefit of adaptive limit, thinking on retry, cloud assist on H6; the one H6 success was variance | 2026-10-20 | |
| E10 | `benchmarks/harness2_experiment.json` (`28a0bc7`) | `docs/HARNESS2_RESULTS.md` §1–3 | H0 7/12, 601 s vs adapted 6/12, 829 s; both promotion rules made the same choices | 2026-10-20 | |
| E11 | benchmark JSON metadata (`model_digest`, `num_ctx`, `num_predict`, `think_enabled`); host (`nvidia-smi`, `free -g`, `ollama --version` on 2026-10-20) | — | model digest `4eb23ef187e2…`, 16,384 context, 2,048/4,096 tokens, think off, RTX 4060 Laptop 8 GB, 31 GB RAM, Ollama 0.33.3 | 2026-10-20 | |
| E12 | `antigravity_hybrid_orchestrator/engine/spec_tests.py` (`PROMPT`, `validate_contract_tests`, `SpecTestWriter.generate`), `antigravity_hybrid_orchestrator/privacy/assist_payload.py` (`describe_contract`) | — | payload content, prompt, static validation, reuse across seeds | 2026-10-20 | |
| E13 | `antigravity_hybrid_orchestrator/architect/client.py` (`agy` backend, default model); `agy --version` = 1.4.3 on 2026-10-20 | — | cloud backend; exact model identifier **not recorded** | 2026-10-20 | |

## External sources

| ID | Source | Checked by assistant (2026-10-20) | Verified by (human) |
|---|---|---|---|
| R1 | research.google publication page for Tufano et al., SpecOps '26 | title, authors, venue, 9.8 pp (p = 0.0352), 2.5 pp branch coverage | |
| R2 | arXiv:2402.09171 abstract page | title, authors, FSE 2024, 75% / 57% / 25% | |
| R3 | arXiv:2305.01210 abstract page | title, authors, 80× tests, pass@k reduced by up to 19.3–28.9% across 26 LLMs; no venue listed on arXiv | |
| R4 | arXiv:2207.10397 abstract page | title, authors, method; no venue listed on arXiv | |
| R5 | arXiv:2312.13010 (search result and abstract) | three-agent design with test designer working from requirements; author list not yet checked | |
| R6 | arXiv:2510.20270 and ICLR 2026 listing (search results) | benchmark construction (tests contradicting spec); author names from search snippet only | |
| R7 | github.com/google-research/harness2 | repository exists; adaptation without weight updates | |
| R8 | arXiv:2609.00065 abstract page | title, authors, year, arXiv DOI (v2, 2026-09-02) | |

Not cited because not verified: "TRACE (2026), Information and Software Technology" (named by an external review; publisher page returned HTTP 403 and searches did not locate it).
