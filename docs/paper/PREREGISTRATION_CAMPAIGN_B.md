# Pre-registration — Paper I, Campaign B (bugs written by the local model)

**Registered:** 2026-10-20, by commit and push of this file to the public repository `cyberdid/antigravity-codex-orchestrator` (branch `feat/gemma-orion-phase2b`) **before the buggy code is generated and before any Campaign B run**.
**Author:** Oleksandr Pylypenko (ORCID 0009-0004-9610-9242).

## 1. Why

In Campaign A and the earlier study the buggy code was written by the author, who knew the agent's failure mode. Campaign B keeps the 25 Campaign A specifications but takes the bugs from the local model itself, so that which rules are broken, and how, is not chosen by the author.

## 2. Task construction (fixed before generation)

`scripts/paper_tasks/natural.py`:

1. For each specification P1–P25, the first editable file is reduced to a skeleton (imports, constants, classes, signatures and docstrings; function bodies replaced by `...`); the other files are given in their correct reference state as read-only context.
2. `gemma4:12b` writes the module once per generation seed 1, 2, 3, 4, 5 (temperature 0.8, `think` off, 4,096 output tokens, no tests shown), in that order.
3. The original module and function docstrings are restored in the generated code, so the bug is in the bodies and never in the specification. A generation that does not parse or misses a specified function is rejected.
4. The first remaining generation that violates at least one rule becomes the buggy file of task `N<k>` (k = the specification's index). A specification whose every non-rejected generation satisfies all rules is excluded; the number of excluded specifications is reported as the model's correct-implementation rate.
5. Visible rules are chosen by the same seeded coin flip as in Campaign A (per rule p = 0.5, seed from the task id; at least one visible rule fails on the buggy code). The hidden holdout checks every rule.
6. The generated set is written to `scripts/paper_tasks/natural_tasks.json` and committed (frozen) **before** any repair run.

## 3. Runs

Identical to Campaign A (`PREREGISTRATION_CAMPAIGN_A.md` §1): same agent, settings, cloud model pinned to `gemini-4-argon-medium`, seeds 303/404/505, arms baseline (`P1_STAGE_ADAPTIVE`), contract tests (`P1_STAGE_ADAPTIVE_SPECTESTS`) and checklist control (`P1_STAGE_ADAPTIVE_CHECKLIST`). No ablation arms.

## 4. Outcomes, hypotheses, analysis

As in Campaign A §3–4, on the Campaign B tasks: **H1 (primary)** contract tests raise the per-task verified rate over the baseline (exact two-sided sign test over tasks, α = 0.05); H2 silent false successes; H3 contract tests vs checklist; same secondary statistics. Analysis script: `scripts/analysis/campaign_a.py` (detects Campaign B task ids). The number of tasks depends on step 4 and is reported; if fewer than 10 tasks remain, the campaign is reported as descriptive only.

## 5. Run rules

As in Campaign A §5. Interruptions are resumed from saved partial results and reported.
