# Escalation Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the local Gemma worker fails, escalate step by step: a one-turn larger generation limit for truncated tool calls, optional thinking on retry, and owner-opt-in cloud guidance, then measure it with real cloud calls.

**Architecture:** The per-turn limit lives in `AutonomousWorker`. The per-attempt steps live in a small `EscalationLadder` consulted by `SelfCorrectionLoop`. Cloud assist is a separate `CloudAssist` object built only by the coordinator from CLI flags, with its own payload builder and verifier under `privacy/`. The Phase 2B benchmark gains a multi-attempt mode that runs through `SelfCorrectionLoop`.

**Tech Stack:** Python 3.12+, pytest, Ollama `/api/chat`, existing `CloudArchitectClient` backends (`agy` for the live run).

**Spec:** `docs/superpowers/specs/2026-10-19-escalation-ladder-design.md`

## Global Constraints

- Cloud assist is off by default. Only the `--cloud-assist spec` CLI flag of the current invocation enables it. No environment variable, workspace file, planner output or resumed state can enable it.
- Assist payload: subtask id/title/description/success criteria, signatures and docstrings of functions in `affected_files`, failure excerpt lines. Never function bodies, test bodies, file contents, diffs, environment.
- Payload cap 8192 bytes (UTF-8). At most 1 assist call per subtask; per-run cap `--cloud-assist-max-calls` (default 3). Assist tokens count toward `--max-remote-tokens`.
- Every assist request/response is recorded verbatim with SHA-256 of the payload.
- Default generation limit unchanged (`effective_num_predict`, 2048 with tools); escalated limit `OLLAMA_NUM_PREDICT_MAX` default 4096; at most one escalation per turn; adaptive limit on by default (`OLLAMA_ADAPTIVE_NUM_PREDICT=0` disables).
- Thinking on retry off by default (`--think-on-retry` enables); applies to attempts ≥ 2.
- Do not run tests or other CPU-heavy work while a GEMMA benchmark is running (local model is partly CPU-offloaded).

## Review Focus

1. Ollama returns `done_reason="length"` with tool calls whose arguments look complete. Expected: still discarded and re-requested; a truncated turn never runs a tool. Test in Task 1.
2. The escalated limit does not fit the remaining context. Expected: no escalation for that turn, reason logged, no crash. Test in Task 1.
3. The cloud backend (agy CLI) fails or times out during assist. Expected: event `cloud_assist_failed`, the ladder continues to the next attempt without guidance. Test in Task 4.
4. An affected file is missing, non-Python or has syntax errors. Expected: payload built without its signatures, no exception. Test in Task 3.
5. Docstrings or failure text with non-ASCII (e.g. Ukrainian) text near the cap. Expected: cap measured in UTF-8 bytes, over-cap payload blocked, not truncated mid-character. Test in Task 3.

---

### Task 1: Variable generation limit for truncated tool calls

**Files:**
- Modify: `antigravity_hybrid_orchestrator/workforce/context.py` (add helpers near `chat_num_predict`)
- Modify: `antigravity_hybrid_orchestrator/workforce/ollama_client.py:348` (`chat`)
- Modify: `antigravity_hybrid_orchestrator/workforce/worker.py` (`_execute_native` turn loop)
- Test: `tests/test_escalation_ladder.py` (new)

**Interfaces:**
- Produces: `adaptive_num_predict_enabled() -> bool` (env `OLLAMA_ADAPTIVE_NUM_PREDICT`, default true), `num_predict_max() -> int` (env `OLLAMA_NUM_PREDICT_MAX`, default 4096, never below the default limit) in `context.py`.
- Produces: `WorkforceClient.chat(..., num_predict: Optional[int] = None)`; when given it replaces the computed `output_cap`, and preflight reserves it.
- Produces: telemetry keys `budget_escalations: int`, `discarded_truncated_tool_calls: int`, per-turn `num_predict` and `budget_escalated: bool` in `turn_trajectory`.

- [ ] **Step 1: Write failing tests** (fake client pattern as in `tests/test_gemma_orion_phase1.py::test_t19...`, capture the `options.num_predict` of each request)
  - `test_truncated_tool_call_is_not_executed_and_turn_is_retried_with_larger_limit`: response 1 = `done_reason="length"` with a complete-looking `write_file` call; response 2 = same call with `done_reason="stop"`. Assert the file is written exactly once (from response 2), requests' `num_predict` are `[2048, 4096, 2048, ...]`, `budget_escalations == 1`, `discarded_truncated_tool_calls == 1`.
  - `test_truncated_prose_turn_does_not_escalate`: `done_reason="length"`, no tool calls → next request uses 2048.
  - `test_escalation_skipped_when_larger_limit_does_not_fit`: with `OLLAMA_NUM_CTX` small enough that prompt + 4096 exceeds it, assert no 4096 request and a `budget_escalation_skipped` reason in the turn entry.
  - `test_adaptive_limit_can_be_disabled`: `OLLAMA_ADAPTIVE_NUM_PREDICT=0` → old behaviour (truncated call goes to the existing path; no 4096 request).
- [ ] **Step 2: Run** `.venv/bin/pytest -q tests/test_escalation_ladder.py` — Expected: FAIL (`chat()` got unexpected keyword `num_predict` / assertions).
- [ ] **Step 3: Implement.** In the worker, when `is_truncated and calls`: increment `discarded_truncated_tool_calls`; if adaptive is on, this turn not yet escalated, and `preflight_context_budget(messages, tools, num_predict_val=num_predict_max())` succeeds, re-request the same messages with `num_predict=num_predict_max()` and treat that response as the turn's response; otherwise drop the calls and fall through to the existing non-tool-turn handling. The next turn uses the default limit.
- [ ] **Step 4: Run** the new tests and `tests/test_gemma_orion_phase1.py` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(workforce): retry truncated tool-call turns once with a larger generation limit`.

### Task 2: Thinking on retry via EscalationLadder

**Files:**
- Create: `antigravity_hybrid_orchestrator/engine/escalation.py`
- Modify: `antigravity_hybrid_orchestrator/workforce/worker.py` (`execute_subtask`, `_execute_native`)
- Modify: `antigravity_hybrid_orchestrator/workforce/mock_worker.py:257` (signature parity)
- Modify: `antigravity_hybrid_orchestrator/engine/loop.py` (`__init__`, `execute_subtask_with_retry` worker call at ~319)
- Test: `tests/test_escalation_ladder.py`

**Interfaces:**
- Produces: `@dataclass class EscalationLadder: think_on_retry: bool = False; cloud_assist: Optional["CloudAssist"] = None` with `think_for_attempt(attempt: int) -> Optional[bool]` (attempt is 0-based; returns `True` for attempt ≥ 1 when `think_on_retry`, else `None`) and `assist_due(attempt: int) -> bool` (Task 5 uses it: true when `cloud_assist` is set and `attempt == (2 if think_on_retry else 1)`).
- Produces: `AutonomousWorker.execute_subtask(subtask, tool_suite, feedback_context=None, deadline=None, think: Optional[bool] = None)`; when `think is True` every `chat` call of that attempt gets `think=True, num_predict=num_predict_max()`. `MockAutonomousWorker.execute_subtask` accepts and ignores `think`.
- Produces: `SelfCorrectionLoop(..., ladder: Optional[EscalationLadder] = None)`; result metadata `think_retry_used: bool`.

- [ ] **Step 1: Write failing tests**
  - `test_ladder_think_for_attempt`: `EscalationLadder(think_on_retry=True).think_for_attempt(0) is None`, `(1) is True`; default ladder returns `None` for all.
  - `test_loop_passes_think_override_on_retry`: a recording fake worker that fails attempt 1 and succeeds attempt 2 (verification command `python3 -c "import pathlib,sys; sys.exit(0 if pathlib.Path('ok').exists() else 1)"`, the fake writes `ok` on attempt 2); with `ladder=EscalationLadder(think_on_retry=True)` the recorded `think` values are `[None, True]`; without a ladder `[None, None]`.
  - `test_worker_think_override_reaches_chat`: fake client records `payload["think"]` and `options.num_predict`; `execute_subtask(..., think=True)` → `True` and `4096`.
- [ ] **Step 2: Run** — Expected: FAIL (`ModuleNotFoundError: ...engine.escalation`).
- [ ] **Step 3: Implement** the dataclass, the worker parameter, the mock signature, and the loop call `think=self.ladder.think_for_attempt(attempt) if self.ladder else None`.
- [ ] **Step 4: Run** the new tests, `tests/test_loop_limits.py`, `tests/test_self_correction.py`, `tests/test_native_worker.py` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(engine): optional thinking on retry through EscalationLadder`.

### Task 3: Assist payload builder and privacy verifier

**Files:**
- Create: `antigravity_hybrid_orchestrator/privacy/assist_payload.py`
- Modify: `antigravity_hybrid_orchestrator/workforce/recovery_controller.py` (rename `_failure_excerpt` → public `failure_excerpt`, keep behaviour)
- Test: `tests/test_cloud_assist_privacy.py` (new)

**Interfaces:**
- Produces: `build_assist_payload(workspace: Path, subtask: SubTask, failure_excerpt: str) -> str` — plain-text sections `SUBTASK`, `SIGNATURES AND DOCSTRINGS`, `FAILURE`. Signatures/docstrings come from `ast` over each `affected_files` entry that is an existing `.py` file inside `workspace`; unreadable/invalid files are skipped.
- Produces: `verify_assist_payload(payload: str, *, max_bytes: int = 8192) -> None`, raising `SecurityInvariantViolationError` (from `privacy/verifier.py`) for: UTF-8 size over cap; private-key headers; token shapes (`ghp_`, `github_pat_`, `sk-`, `AKIA[0-9A-Z]{16}`, `xox[bap]-`); any single token ≥ 32 chars of `[A-Za-z0-9+/=_-]` with Shannon entropy ≥ 4.0.
- Produces: `failure_excerpt(output: str, limit: int) -> str` in `recovery_controller.py`.

- [ ] **Step 1: Write failing tests** (workspace = H6 files from `scripts/benchmark_gemma_orion_phase2b.py::setup_h6_workspace`)
  - `test_payload_contains_signature_docstring_and_failure`: contains `def parse_duration(text: str) -> int`, the docstring sentence `Units are h (3600 s)`, and the passed excerpt.
  - `test_payload_excludes_function_and_test_bodies`: does not contain `re.findall`, `UNITS = {`, `total +=`, `assertEqual`, `class TestParseDuration`.
  - `test_missing_or_invalid_affected_file_is_skipped`: affected files `["nope.py", "broken.py"]` (broken = invalid syntax) → returns a payload, no exception.
  - `test_planted_secret_blocks_payload`: docstring containing `ghp_` + 36 alphanumerics → `verify_assist_payload` raises.
  - `test_cap_is_measured_in_utf8_bytes`: 3000 Cyrillic characters (≈6000 bytes) pass at 8192; 5000 (≈10000 bytes) raise.
- [ ] **Step 2: Run** `.venv/bin/pytest -q tests/test_cloud_assist_privacy.py` — Expected: FAIL (module missing).
- [ ] **Step 3: Implement** both functions; rename the excerpt helper and update its two internal call sites.
- [ ] **Step 4: Run** the new tests, `tests/test_privacy.py`, `tests/test_gemma_orion_phase1.py` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(privacy): bounded spec-level payload for owner-approved cloud assist`.

### Task 4: Cloud assist client and limits

**Files:**
- Create: `antigravity_hybrid_orchestrator/engine/cloud_assist.py`
- Modify: `antigravity_hybrid_orchestrator/architect/client.py` (new method next to `plan_rd_task`)
- Modify: `antigravity_hybrid_orchestrator/architect/mock_client.py` (same method, deterministic)
- Test: `tests/test_cloud_assist.py` (new)

**Interfaces:**
- Consumes: `build_assist_payload`, `verify_assist_payload` (Task 3); `ExecutionControl.record_remote_tokens`, `ExecutionControl.check`.
- Produces: `CloudArchitectClient.assist_subtask(payload: str) -> Tuple[str, int, int]` — wraps the payload in a prompt asking for at most 12 numbered steps or pseudocode, explicitly no complete code; calls `_invoke_cloud`; truncates the response to 4000 characters. `MockCloudArchitectClient.assist_subtask` returns a fixed three-step text and `(len(payload)//4, 50)` tokens.
- Produces: `class CloudAssist(client, workspace: Path, *, max_calls: int = 3, audit: Callable[[str, str, Dict[str, Any]], None], execution_control: ExecutionControl)` with `request(subtask: SubTask, failure_excerpt: str) -> Optional[str]` returning guidance text or `None`. Events through `audit(event_type, subtask_id, payload)`: `cloud_assist_request` `{payload, payload_sha256, backend}`, `cloud_assist_response` `{response, input_tokens, output_tokens}`, `cloud_assist_blocked` `{reason}`, `cloud_assist_failed` `{error}`, `cloud_assist_skipped` `{reason: "run_limit"|"subtask_limit"|"remote_budget"}`. Attribute `calls_made: int`, `tokens_used: int`.

- [ ] **Step 1: Write failing tests** (mock client, recording audit list)
  - `test_request_records_exact_payload_and_hash`: the `cloud_assist_request` payload equals what the client received; `payload_sha256 == sha256(payload.encode()).hexdigest()`; guidance returned.
  - `test_one_call_per_subtask_and_run_limit`: second request for the same subtask → `None` + `subtask_limit`; with `max_calls=1`, a request for another subtask → `None` + `run_limit`.
  - `test_blocked_payload_never_reaches_client`: planted secret → client not called, `cloud_assist_blocked`.
  - `test_backend_failure_is_reported_not_raised`: client raises `CloudArchitectError` → `None` + `cloud_assist_failed`.
  - `test_remote_token_budget_enforced`: tokens are recorded via `record_remote_tokens`; when `execution_control.check()` reports `stopped`, the request returns `None` + `remote_budget` without calling the client.
- [ ] **Step 2: Run** `.venv/bin/pytest -q tests/test_cloud_assist.py` — Expected: FAIL (module missing).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the new tests and `tests/test_architect_backends.py`, `tests/test_cloud_execution_budget.py` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(engine): owner-approved cloud assist with limits and verbatim audit`.

### Task 5: Use cloud guidance in the self-correction loop

**Files:**
- Modify: `antigravity_hybrid_orchestrator/engine/loop.py` (`execute_subtask_with_retry`)
- Test: `tests/test_escalation_ladder.py`

**Interfaces:**
- Consumes: `EscalationLadder.assist_due` (Task 2), `CloudAssist.request` (Task 4), `failure_excerpt` (Task 3), the last `VerificationResult` (`stderr + "\n" + stdout` of `last_verif_res`).
- Produces: before an attempt where `assist_due(attempt)` is true, the loop calls `request(subtask, failure_excerpt(last_verification_output, 600))`; non-`None` guidance is appended to `feedback_context` under the heading `Guidance from the cloud architect (steps, not code):`. Result metadata: `ladder_steps_used: List[str]` (subset of `["adaptive_limit", "think_retry", "cloud_assist"]`), `cloud_assist_calls: int`, `solved_with_cloud_assist: bool`.

- [ ] **Step 1: Write failing tests** (recording fake worker that captures `feedback_context`; fake `CloudAssist` with a `request` that records calls and returns `"1. Use fullmatch."`)
  - `test_assist_called_before_due_attempt_with_failure_excerpt`: without think-on-retry, assist is called once before attempt 2 (0-based 1), with an excerpt containing the failing assertion line; attempt 2's `feedback_context` contains the heading and `1. Use fullmatch.`.
  - `test_assist_not_called_without_ladder_assist`: ladder with `cloud_assist=None` → fake never called, `solved_with_cloud_assist` false.
  - `test_assisted_success_is_labelled`: success on the assisted attempt → `solved_with_cloud_assist is True`, `"cloud_assist" in ladder_steps_used`.
- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `tests/test_escalation_ladder.py`, `tests/test_self_correction.py`, `tests/test_e2e_rd_engine.py` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(engine): feed owner-approved cloud guidance into the next attempt`.

### Task 6: CLI flags, coordinator wiring and run policy

**Files:**
- Modify: `antigravity_hybrid_orchestrator/cli.py` (parser near line 89; coordinator kwargs near line 252)
- Modify: `antigravity_hybrid_orchestrator/coordinator.py` (`__init__`, `_run_rd` where `SelfCorrectionLoop` is built, resume/policy block near line 600, `RunPolicyStore.save` near line 714)
- Test: `tests/test_cloud_assist_cli.py` (new)

**Interfaces:**
- Consumes: `EscalationLadder`, `CloudAssist`.
- Produces: CLI `--cloud-assist {off,spec}` (default `off`), `--cloud-assist-max-calls INT` (default 3, must be ≥ 1), `--think-on-retry`, `--max-num-predict INT` (sets `OLLAMA_NUM_PREDICT_MAX`), `--no-adaptive-num-predict` (sets `OLLAMA_ADAPTIVE_NUM_PREDICT=0`). `HybridCoordinator(..., cloud_assist: str = "off", cloud_assist_max_calls: int = 3, think_on_retry: bool = False)`. The coordinator builds `CloudAssist` only when `cloud_assist == "spec"`, with `audit=self._rd_event_listener`. `RunPolicyStore` records `cloud_assist` and `cloud_assist_max_calls` for the audit trail, but on `--resume` these two values come only from the current command line. The telemetry summary (shown by the HUD and the final summary, and written by `record_telemetry`) gains `cloud_assist_calls` and `cloud_assist_tokens`.

- [ ] **Step 1: Write failing tests** (mock mode, `tmp_path` workspace)
  - `test_cloud_assist_off_by_default`: parsed defaults give `cloud_assist == "off"`; a coordinator built from defaults has no `CloudAssist`.
  - `test_environment_cannot_enable_cloud_assist`: with `OLLAMA_CLOUD_ASSIST=spec` and `ANTIGRAVITY_CLOUD_ASSIST=spec` set, defaults still give no `CloudAssist`.
  - `test_resume_does_not_restore_cloud_assist`: save a policy with `cloud_assist="spec"`, resume without the flag → no `CloudAssist`; resume with `--cloud-assist spec` → enabled.
  - `test_assist_events_reach_run_log`: mock run with `--cloud-assist spec` on a subtask whose first attempt fails (mock worker configured to fail once) → run JSONL contains `cloud_assist_request` with `payload_sha256`, and the recorded telemetry has `cloud_assist_calls == 1` and `cloud_assist_tokens > 0`.
- [ ] **Step 2: Run** `.venv/bin/pytest -q tests/test_cloud_assist_cli.py` — Expected: FAIL.
- [ ] **Step 3: Implement**, and mention the new flags in the README CLI table (section 7).
- [ ] **Step 4: Run the full suite** `.venv/bin/pytest -q` and `RUN_DOCKER_TESTS=1 .venv/bin/pytest -q tests/test_sandbox.py` — Expected: all pass (baseline before this plan: 837 passed, 12 skipped; Docker 24 passed).
- [ ] **Step 5: Commit** `feat(cli): owner flags for cloud assist, thinking on retry and generation limits`.

### Task 7: Benchmark multi-attempt mode, new tasks, live run, report

**Files:**
- Modify: `scripts/benchmark_gemma_orion_phase2b.py`
- Modify: `docs/GEMMA_ORION_PHASE2B_REPORT.md` (new section 9)
- Output: `benchmarks/gemma_orion_phase2b_escalation_ladder.json`

**Interfaces:**
- Consumes: `SelfCorrectionLoop(..., checkpoint_git=False, ladder=...)`, `CloudAssist`, `CloudArchitectClient(backend="agy")`.
- Produces: CLI `--attempts N` (default 1 keeps today's single-attempt behaviour) running each task through `SelfCorrectionLoop.execute_subtask_with_retry` with `RD_MAX_REPAIR_ATTEMPTS=N` and the same per-task deadline across attempts; `--cloud-backend {agy,haiku,luna}` (default `agy`). Policies: `P1_STAGE` (adaptive limit off), `P1_STAGE_ADAPTIVE`, `P1_STAGE_ADAPTIVE_THINK`, `P1_STAGE_ADAPTIVE_THINK_ASSIST`. Per-task metadata adds `attempts_used`, `ladder_steps_used`, `cloud_assist_calls`, `cloud_assist_tokens`, `solved_with_cloud_assist`; per-policy summary separates `solved_locally` and `solved_with_cloud_assist`.
- New tasks H7 and H8: one function each, specification in the docstring, two or more defects with one visible test per defect, holdout covering docstring-only rules (like H6). Validate offline as for H5/H6: the original and each single-defect fix fail; the reference fix passes visible and holdout.

- [ ] **Step 1: Add H7/H8 and an offline validation run** (same method as the H5/H6 validation; keep the validation script in the scratchpad, not the repo) — Expected: only the reference fix passes.
- [ ] **Step 2: Implement `--attempts`, policies and cloud backend wiring**; smoke-run `--policies P1_STAGE --tasks H1 --seeds 303 --attempts 1` — Expected: same shape of result as today.
- [ ] **Step 3: Commit** `feat(bench): multi-attempt escalation ladder mode and tasks H7/H8`.
- [ ] **Step 4: Live run (nothing else heavy on the machine; GPU idle first)** — `--policies P1_STAGE,P1_STAGE_ADAPTIVE,P1_STAGE_ADAPTIVE_THINK,P1_STAGE_ADAPTIVE_THINK_ASSIST --tasks H5,H6,H7,H8 --seeds 303,404 --attempts 3 --output benchmarks/gemma_orion_phase2b_escalation_ladder.json`. Expected: completes; JSON has `git_dirty=false` and audit records for every assist call.
- [ ] **Step 5: Analyse trajectories and write report section 9**: per policy verified/locally solved/assisted, cloud calls and tokens, truncations and escalations, wall-clock; per-task forensics for every failure; state limits (tasks, seeds, deadline).
- [ ] **Step 6: Commit and push** `docs(phase2b): escalation ladder results` to `feat/gemma-orion-phase2b`.
