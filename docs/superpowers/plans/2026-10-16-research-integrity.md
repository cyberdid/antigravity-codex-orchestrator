# Research integrity implementation plan

> For agentic workers: use subagent-driven-development with independent ownership and task review. Steps use TDD. The user has approved implementation; continue through integration and rerun.

**Goal:** Implement the approved five-part design and produce a fresh independently checked Orion run.

**Architecture:** Docker confines candidate commands. Controller-owned RIE contracts gate completion after ordinary verification. Resume fingerprints bind evidence to files; corrected Orion produces real measured outputs.

**Tech stack:** Python >=3.12, pytest, Rich, httpx, NumPy/SciPy/scikit-learn, Docker, JSONL, Git.

**Spec:** `docs/superpowers/specs/2026-10-16-research-integrity-design.md`

## Global constraints

- Preserve original artifacts; no invented results or undocumented manual intervention.
- Candidate execution receives no hidden tests, expected outputs, controller credentials or writable trusted evidence.
- Live shell defaults to Docker; host is explicit. Scientific acceptance requires a controller-selected contract.
- Independent reference expectations, real source imports and executable regressions precede production fixes.
- Keep all work in the isolated branch; no automatic merge or push.

## Task 1: Baseline, specifications and ledger

Files: this plan, linked spec, ignored execution ledger.

- [ ] Run baseline: `/home/cyberdid/Projects/llm/antigravity-codex-orchestrator/.venv/bin/python -m pytest -q` from worktree. Preserve output.
- [ ] Commit specification/plan and record ownership/interface decisions.

## Task 2: Docker workspace runner

Files: `workforce/tools.py`, new `runtime/sandbox.py`, container definition, `tests/test_sandbox.py`.

Interface: `WorkspaceToolSuite(..., sandbox_backend='host', sandbox_image='antigravity-orion:local')`; `run_command` retains ToolResult API and existing deadline/idle/cancel behavior.

- [ ] First test writes through mounted workspace and attempts to read an external canary, access credentials, use network and leave child processes after timeout. Assert denied access, permitted artifact and killed processes.
- [ ] Implement Docker arguments: `--network=none --read-only --cap-drop=ALL --security-opt=no-new-privileges`, non-root UID, memory/CPU/PID limits, bounded tmpfs, workspace-only mount. Kill/remove container on cancellation.
- [ ] Build image with Python and frozen numeric dependencies; run real container tests and existing tools suite. No silent host fallback.

## Task 3: Scientific Orion correctness

Files: `temp/project_orion/{src,scripts,tests,configs,pyproject.toml}`; historical artifacts unchanged.

Interface: retain public module import names. `calculate_metrics(y_true,y_pred)` returns documented JSON-safe values/counts; `-1` is abstention. Export neighbor IDs/distances for retrieval verification.

- [ ] Tests use literal cases: true `[0,0,1,1]`, predicted `[0,0,0,0]` has B³ precision=.5, recall=1, pairwise precision=1/3; all-abstained coverage=0 and undefined precision=null.
- [ ] Replace placeholder tests, fix metrics and deterministic generator, true mutual kNN and canonical risk graph with explicit abstention.
- [ ] Test real retrieval on separate queries, compare exact result on small arrays; scalability uses bounded batches/index instead of sleep/constants.
- [ ] Repair package metadata and experiment protocol. Save predictions, frozen dev config, paired ablation/sensitivity, disjoint eval seeds and independent measured outputs.
- [ ] Run Orion suite, package validation and small real experiment.

## Task 4: RIE contracts and evidence

Files: new `integrity/` package and `tests/test_research_integrity.py`.

Interface: `EvaluationDecision` includes status, verification_level, evidence_id, checks and feedback; `ResearchIntegrityEvaluator.evaluate(workspace, contract_id)` runs operator-selected contract in isolated candidate snapshot. Evidence directory is external and controller-owned.

- [ ] Red tests reject ORION-1 metric constants and fake scalability, accept correct independent candidates, reject stale/altered claims and path escapes.
- [ ] Implement trusted metric oracle, randomized fixtures, retrieval oracle and externally measured resources, evidence hashes and claim links.
- [ ] Hidden expected values never enter candidate execution; enforce schema/shape/ranges, finite numbers and output size/deadlines.
- [ ] Verify real sandbox integration and failure statuses.

## Task 5: Reliability and promotion integration

Files: `engine/{loop,run_state,verifier}.py`, `coordinator.py`, `cli.py`, `workforce/{worker,context,ollama_client}.py`, HUD and regression tests.

- [ ] Red tests: mutated output/source invalidates completed task and dependents; unchanged hashes skip; legacy snapshots revalidate. Add acceptance-gate rejection and no-silent-Gauntlet-reference tests.
- [ ] Add fingerprinted state and evidence binding; gate before completion and checkpoint; expose `--integrity-contract`, `--evidence-dir`, `--sandbox`, `--sandbox-image`.
- [ ] Bound context by size preserving complete recent exchanges; enforce remaining wall-time on model calls and shell; clean cloud subprocesses on interruption.
- [ ] Correct recorder/replay terminal status and backend-specific accounting; run relevant regression suites.

## Task 6: Independent review, regression and real rerun

Files: documentation, new reproducible run output under ignored `outputs/orion2/<run-id>/`, final tracked findings/report.

- [ ] Review each owned change for spec compliance and correctness; fix findings with regressions.
- [ ] Full root and Orion suites, package build and real isolation challenges.
- [ ] Re-evaluate historical bad candidates: require REJECTED with concrete evidence.
- [ ] Short live Gemma run through Docker and promotion gate; record exact backend/config/model digest and interventions.
- [ ] Full Orion rerun, trusted recomputation, hashes and claim-linked report. Compare engineering and scientific outcomes separately.
- [ ] Audit every acceptance requirement against current evidence before declaring completion.
