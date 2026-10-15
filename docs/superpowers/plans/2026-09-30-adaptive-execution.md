# Adaptive R&D Execution Implementation Plan

> **For agentic workers:** Execute this plan inline in order. Each task uses TDD and ends with its focused test command.

**Goal:** Remove fixed worker-turn and retry-count ceilings while adding progress-aware recovery, safe interruption, useful planning gates, and resumable local run state.

**Architecture:** Add run control/budget primitives used by the coordinator and tools. Make worker tool loops run until a final answer or cancellation, and make the self-correction loop use failure fingerprints and prior attempt traces instead of a retry counter. Harden DAG planning and persist enough local state to pause and resume while preserving existing privacy and HUD contracts.

**Tech Stack:** Python 3.12+, pytest, Rich, httpx, JSONL recorder, Git checkpoints.

**Spec:** `docs/superpowers/specs/2026-09-30-adaptive-execution-design.md`

## Global Constraints

- No fixed maximum number of worker turns or repair attempts.
- Stop safely on Ctrl+C, a configured operator budget, an unrecoverable error, or repeated no-progress fingerprints.
- Preserve unverified workspace edits on pause/stall; checkpoint only verified subtasks.
- Keep `.antigravity_runs` owner-only and out of Git checkpoints/cleanup.
- Preserve zero-knowledge cloud payload checks and serial local-GPU inference.
- Verification commands must prove the subtask's declared deliverable.

---

### Task 1: Run control and operator budgets

**Files:**
- Create: `antigravity_hybrid_orchestrator/engine/control.py`
- Modify: `antigravity_hybrid_orchestrator/cli.py`
- Modify: `antigravity_hybrid_orchestrator/coordinator.py`
- Test: `tests/test_execution_control.py`
- Test: `tests/test_interactive.py`

**Interfaces:**
- `ExecutionBudget(max_runtime_s: float | None, max_remote_tokens: int | None, model_idle_timeout_s: float = 300, tool_idle_timeout_s: float = 300)` stores opt-in run-wide budgets and configurable inactivity watchdogs (`0` disables a watchdog).
- `ExecutionControl(budget, clock=time.monotonic)` exposes `cancel(reason)`, `check() -> StopDecision`, and `record_remote_tokens(count)`.
- `StopDecision(stopped: bool, reason: str = "")` is immutable and JSON-serializable.
- CLI adds `--max-runtime`, `--max-remote-tokens`, `--model-idle-timeout`, and `--tool-idle-timeout`; omitted total budgets mean no total run cap.

- [x] Write `test_control_runs_without_optional_total_budgets`, `test_control_stops_on_runtime_budget`, `test_control_stops_on_remote_token_budget`, and `test_control_cancel_returns_reason`.
- [x] Run the focused tests red, then implement the control object with monotonic deadlines and thread-safe cancellation.
- [x] Add optional CLI budgets and pass shared control through planner, worker, loop, and workspace execution.
- [x] Verify budget defaults, explicit stops, and restored usage across resume.

### Task 2: Unbounded worker turns and interrupt-safe model/tool calls

**Files:**
- Modify: `antigravity_hybrid_orchestrator/workforce/worker.py`
- Modify: `antigravity_hybrid_orchestrator/workforce/ollama_client.py`
- Modify: `antigravity_hybrid_orchestrator/workforce/tools.py`
- Modify: `antigravity_hybrid_orchestrator/coordinator.py`
- Test: `tests/test_autonomous_worker.py`
- Test: `tests/test_native_worker.py`
- Test: `tests/test_workspace_tools.py`
- Test: `tests/test_hud_and_coordinator.py`

**Interfaces:**
- `AutonomousWorker(..., control: ExecutionControl | None = None)` checks control between model turns and tool calls.
- Both native and ReAct loops use `while` loops and return a cancelled result with tool trace when stopped; no `max_turns` field gates completion.
- `WorkspaceToolSuite.run_command(command, cwd=None, timeout=None, idle_timeout=None)` streams output, supports cancellation, and kills its process group in `finally`. A total command timeout is optional; the coordinator passes the configured idle timeout.
- `OllamaClient` reports streaming activity to the control object and only ends on response completion, cancellation, or an explicitly configured inactivity timeout.

- [x] Add scripted tests requiring 12 tool calls and repeated-action/cycle stall detection.
- [x] Add cooperative model-loop cancellation and shell process-tree cleanup with output-based inactivity timeouts.
- [x] Remove turn ceilings and thread cooperative control through model/tool calls.
- [x] Verify worker and workspace tool tests.

### Task 3: Progress-aware repair and replan

**Files:**
- Modify: `antigravity_hybrid_orchestrator/engine/loop.py`
- Modify: `antigravity_hybrid_orchestrator/engine/verifier.py`
- Modify: `antigravity_hybrid_orchestrator/architect/client.py`
- Modify: `antigravity_hybrid_orchestrator/architect/mock_client.py`
- Test: `tests/test_self_correction.py`
- Test: `tests/test_execution_dag.py`
- Test: `tests/test_orchestrator_integration.py`

**Interfaces:**
- `AttemptTrace` stores structured tool/action outcomes and an output hash; raw output remains in JSONL.
- `ProgressFingerprint.for_attempt(subtask, trace, workspace_fingerprint, verification)` returns a stable digest.
- `SelfCorrectionLoop(..., control, replanner=None, stall_repeats=3)` continues until verification, control stop, fatal error, or the same no-progress fingerprint repeats three times.
- `replanner(subtask, attempt_history) -> SubTask` must retain the subtask ID, revise actionable scope/approach, and pass DAG/path validation.

- [x] Add tests for recovery beyond the legacy retry budget and a replan that repairs repeated verification failure.
- [x] Fingerprint worker evidence and workspace contents; progress resets the repeated-failure detector.
- [x] Implement attempt records, no-progress detection, and an inventory-grounded replanner that sends no raw diagnostics to the cloud.
- [x] Remove `retry_budget` as a stop condition and preserve failed/paused edits instead of rolling them back.
- [x] Verify recovery and DAG tests.

### Task 4: Planner decomposition and deliverable verification

**Files:**
- Modify: `antigravity_hybrid_orchestrator/architect/client.py`
- Modify: `antigravity_hybrid_orchestrator/planner/dag.py`
- Modify: `antigravity_hybrid_orchestrator/coordinator.py`
- Modify: `antigravity_hybrid_orchestrator/hud/rd_dashboard.py`
- Test: `tests/test_execution_dag.py`
- Test: `tests/test_e2e_rd_engine.py`
- Test: `tests/test_rd_hud.py`

**Interfaces:**
- `PlanValidator.validate(dag, workspace_inventory) -> list[PlanIssue]` checks dependency integrity, output declarations, known input paths, and deliverable-verifier alignment.
- The planner prompt requires bounded subtasks, evidence-backed reports, and a verifier that checks the declared output.
- Invalid plans receive one structural repair prompt before execution; repeated invalid plans fail with the exact issue list.

- [x] Reject nonexistent inputs/affected paths, unsafe paths, empty verifiers, and `rg --files`/path-mention-only output checks.
- [x] Accept output-specific existence/content checks; allow outputs produced by dependency ancestors.
- [x] Implement one structural planner repair pass before execution and validate the repaired plan locally.
- [x] Verify DAG, planner, and R&D dashboard tests.

### Task 5: Durable pause/resume state and audit log

**Files:**
- Modify: `antigravity_hybrid_orchestrator/hud/recorder.py`
- Modify: `antigravity_hybrid_orchestrator/hud/replay.py`
- Modify: `antigravity_hybrid_orchestrator/coordinator.py`
- Modify: `antigravity_hybrid_orchestrator/cli.py`
- Modify: `antigravity_hybrid_orchestrator/engine/loop.py`
- Test: `tests/test_recorder_replay.py`
- Test: `tests/test_interactive.py`
- Test: `tests/test_self_correction.py`

**Interfaces:**
- JSONL adds `plan_created`, `attempt_started`, `attempt_result`, `replan`, `paused`, and `interrupted` events; all existing records stay readable.
- `RunStateStore` atomically saves `run-state.json` with run ID, workspace, goal, DAG, completed tasks, pending/failed task, attempt summaries, and last checkpoint.
- CLI adds `--resume <run.jsonl>` to restore the saved DAG/state, skip verified tasks, and continue pending work.
- Ordinary failure/interruption preserves workspace edits. Verified checkpoints exclude `.antigravity_runs`.

- [x] Add state round-trip, mode `0600`, resume/skip-verified, and cumulative-budget restore tests.
- [x] Implement atomic state writes, interruption/pause footer status, resume loading, and additive lifecycle events.
- [x] Preserve partial workspace edits and keep run files outside verified-task Git checkpoints.
- [x] Verify recorder/replay and interactive tests.

### Task 6: Documentation and full acceptance

**Files:**
- Modify: `README.md`
- Modify: `antigravity_hybrid_orchestrator/cli.py` help text
- Test: all `tests/`

- [x] Document optional budgets, unlimited turns, stall recovery, cancellation, resume, and run-log location.
- [x] Verify worker execution beyond ten actions and deterministic replan/recovery behavior.
- [x] Run the full test suite; require privacy, interactive, dashboard, replay, and execution tests to pass.
- [ ] Visually inspect a live Codex + Ollama run and HUD against a real workspace.
