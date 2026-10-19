# Escalation ladder for local-worker failures

Scope agreed in conversation on 2026-10-19 (Phase 2B follow-up). Motivation: on the two-module and multi-defect tasks (`docs/GEMMA_ORION_PHASE2B_REPORT.md`, sections 6–8) `gemma4:12b` fails in ways that a single fixed configuration cannot fix: tool calls cut at `num_predict=2048` and executed with missing arguments, reasoning written into code comments with `think=false`, and specification knowledge it does not have. The owner wants the orchestrator to help in those cases, but under owner control.

## Intent

When the local worker fails a subtask, escalate step by step, cheapest and most private step first. Nothing beyond what is sent today may leave the machine unless the owner of the system explicitly allows it for that run.

## Goals

1. **Variable local generation limit.** Default generation limit stays as today (2048 with tools). A turn that is truncated while emitting a tool call is not executed; the same turn is re-requested once with a larger limit, then the limit returns to the default.
2. **Thinking on retry.** Optionally, attempts after a failed attempt run with `think=true`, so reasoning goes to the thinking channel instead of into code.
3. **Cloud assist, owner opt-in.** Optionally, after local escalation has failed, the orchestrator sends a bounded, audited, privacy-checked failure report to the cloud architect and passes the returned step-by-step guidance (not code) to the local worker for the next attempt.
4. **Honest measurement.** Every result records which ladder steps were used; reports separate "solved locally" from "solved with cloud assist" and count cloud tokens.

## Non-goals

- The cloud writing code, receiving function bodies, file contents, test bodies or diffs.
- Changing the default zero-knowledge behaviour.
- Raising the global `num_predict` (Phase 2A: 4096 was 1.7× slower with no accuracy gain).
- Changing the worker model (Gemma 4 26B A4B was declined), Harness² integration.

## Anti-goals (failure even if it "works")

- Any path where data beyond today's outline reaches the cloud without the owner's explicit flag.
- The candidate workspace, the planner's output, a resumed state file or an environment variable set by the candidate enabling cloud assist.
- A cloud call that is not recorded verbatim in the run log.
- Reporting an assisted solve as a native local solve.

## Decided by the owner

- Cloud assist is **off by default**; the owner must grant it explicitly.
- The variable limit applies to the **local Gemma generation limit** (`num_predict`).
- The benchmark measures cloud assist with **real cloud calls** (agy / Gemini Flash, a few thousand tokens per call), not only a mock. The benchmark tasks are synthetic, so this sends no private code.

## Decisions made by Claude (owner asked for my recommendation)

- **Permission is granted once per run** by a CLI flag (`--cloud-assist spec`), not by an interactive prompt per escalation: per-escalation prompts would block unattended runs and conflict with the live HUD. Control comes from: off by default, call limits, verbatim audit log, and the permission being stored in the trusted run policy.
- **Defaults:** variable limit **on** (it only prevents executing broken calls); thinking on retry **off** until a benchmark shows it helps; cloud assist **off**.
- **Order of steps:** variable limit (inside any attempt) → thinking on retry (attempt ≥ 2) → cloud assist (before the first attempt that follows the enabled local steps: attempt 3 with thinking on retry, attempt 2 without) → existing structural replanner and failure reporting unchanged.

## Components

### 1. Variable generation limit (worker, per turn)
- `WorkforceClient.chat` gains an optional per-call `num_predict` override (today only `think` can be overridden).
- In `AutonomousWorker._execute_native`: if a response has `done_reason == "length"` **and** contains tool calls (complete or malformed), execute none of them, record the turn as `budget_escalated`, and repeat the request with the escalated limit. At most one escalation per turn (configurable). If the escalated turn is still truncated with tool calls, discard it and continue with the existing truncation nudge.
- The escalated limit must still fit the context window: compaction reserves the escalated amount for that request; if the prompt cannot fit, skip escalation for that turn and log why.
- Truncated turns without tool calls (runaway prose) keep today's behaviour; no larger limit.
- Configuration: `OLLAMA_NUM_PREDICT_MAX` / `--max-num-predict` (default 4096), `OLLAMA_ADAPTIVE_NUM_PREDICT` / `--no-adaptive-num-predict` to disable.
- Telemetry: per-turn `num_predict` in `turn_trajectory`, `budget_escalations`, `discarded_truncated_tool_calls`.

### 2. Thinking on retry (attempt level)
- `SelfCorrectionLoop` passes an attempt-level `think` override to the worker for attempts ≥ 2 when `--think-on-retry` is set. The worker forwards it to `chat(think=...)` and uses the larger generation limit for that attempt, since thinking consumes it.
- Existing `strip_completed_thinking` handling stays.

### 3. Cloud assist (owner opt-in)
- **Permission.** `--cloud-assist {off,spec}` (default `off`), `--cloud-assist-max-calls` per run (default 3), at most 1 per subtask. Stored in `RunPolicyStore`; `--resume` cannot enable it or raise its limits unless the owner passes the flag again. Mock mode uses a deterministic mock response.
- **Payload at level `spec`.** Subtask id/title/description and success criteria (already sent today), function signatures of `affected_files` (already allowed), **docstrings** of those functions (new), and the **failure excerpt** from the trusted verification run: failing test names and exception lines (new, may contain literal values such as `90 != 5400`). Never: function or test bodies, file contents, diffs, environment, paths outside the workspace.
- **Privacy check before sending.** A dedicated verifier allows only signature lines, docstring text and failure-excerpt lines; a secret scanner (private keys, API-token shapes, long high-entropy strings) blocks the call. Payload size cap (default 8 KB). A blocked call is logged with the reason and the ladder continues without it.
- **Request.** Uses `CloudArchitectClient` (same backend selection, usage ledger, `--max-remote-tokens` accounting). The prompt asks for numbered steps or pseudocode and explicitly not for complete code; response size cap.
- **Use.** The guidance is passed to the next attempt in `feedback_context` under a clear heading. Local verification is unchanged; the cloud never decides success.
- **Audit.** Run log (`.jsonl`, 0600) records `cloud_assist_request` with the exact payload text and its SHA-256, and `cloud_assist_response` with the exact response, token counts and backend. The HUD and final summary show the number of assisted subtasks and cloud tokens.

### 4. Measurement
- Result metadata: `ladder_steps_used`, `budget_escalations`, `think_retry_used`, `cloud_assist_calls`, `cloud_assist_tokens`, `solved_locally` vs `solved_with_cloud_assist`.
- The Phase 2B benchmark gains a multi-attempt mode that runs through the **same ladder code path** as `SelfCorrectionLoop` (not a reimplementation), and policies for each step: `P1_STAGE` baseline, `+ADAPTIVE`, `+THINK_RETRY`, `+ASSIST` (real cloud backend, enabled explicitly for the benchmark run; the mock backend is for unit tests).
- Evaluation tasks: H6 plus 1–2 new specification-heavy tasks; H1–H5 as regression.

## Acceptance

- With default flags, the cloud receives byte-for-byte the same kinds of payload as today (existing privacy tests pass; a new test asserts no assist call is made).
- A candidate workspace file, planner output, environment variable or resumed state cannot enable cloud assist.
- Assist payloads never contain function bodies, test bodies or diffs; a planted secret in a docstring or test message blocks the call.
- A truncated tool call is never executed; the escalated turn uses the larger limit once and the next turn returns to the default.
- Every cloud call appears verbatim in the run log with its hash and token counts; `--cloud-assist-max-calls` and `--max-remote-tokens` are enforced.
- Benchmark reports separate local and assisted solves.

## Left to the builder

Prompt wording, secret-scanner patterns and thresholds, docstring extraction details (AST-based), HUD presentation, exact event names, how the benchmark's multi-attempt mode wires into `SelfCorrectionLoop`.
