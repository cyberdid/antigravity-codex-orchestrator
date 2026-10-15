# Adaptive R&D Execution Design

Date: 2026-09-30  
Status: approved by the user in this conversation

## Goal

Let an R&D task run for as many model turns as it needs, recover from failures with new evidence, and stop safely when execution is interrupted, exceeds an explicit operator budget, or demonstrably makes no progress.

## Current failure this addresses

The live `AutonomousWorker` defaults to ten turns. `SelfCorrectionLoop` then repeats a failed subtask up to a fixed retry budget, reconstructing nearly the same prompt each time. A live workspace run exhausted ten turns six times, repeated the same file reads (including nonexistent paths), never reached verification, and rolled back its uncommitted report. The planner also emitted a file-list command as the artifact verification command.

## Approved approach

Use an adaptive execution controller around the existing planner, local worker, verifier, recorder, and HUD. Remove fixed model-turn and retry-count ceilings. Keep emergency controls: Ctrl+C, optional operator-set runtime/remote-token budgets, tool/model inactivity timeouts, and a no-progress circuit breaker. A counter remains visible for telemetry but does not decide task completion.

## Execution lifecycle

1. The coordinator inventories paths locally and sends the existing privacy-filtered outline to the cloud planner.
2. The planner returns a DAG of bounded subtasks. Each subtask identifies its deliverable, known input paths, evidence format, dependencies, and a command that checks that deliverable.
3. The local worker continues until it reports a final result, is cancelled, reaches a configured execution budget, or encounters an unrecoverable tool/model error. There is no maximum turn count.
4. The verifier runs after worker completion. Its full command, exit status, stdout, stderr, duration, and workspace changes are retained in the local run record.
5. On a recoverable failure, the next attempt receives the complete failure summary and prior tool trace. A repeated failure with an unchanged workspace fingerprint triggers a revised plan or an explicit stalled state; the same attempt is not blindly replayed.
6. Verified subtasks get checkpoints. Interrupted or stalled work remains on disk as unverified work and is resumable. Rollback is reserved for explicit operator choice or a security failure that requires discarding the affected changes.

## Progress and stall detection

The worker records structured tool actions and outcomes. A progress fingerprint combines the subtask, normalized recent action sequence, outcome codes, and a workspace-change fingerprint. Repeated identical failure fingerprints with no new successful observation or workspace change trigger replanning. New evidence, a changed workspace, or a changed verifier result resets the stall state. Read-only exploration is allowed to continue while it produces new observations.

Transient network/model errors use backoff and preserve the last trace. Permanent errors (invalid configuration, denied operation, missing executable) are surfaced with the error category and stop the current task. Replanning is bounded by repeated-plan detection, not by a fixed count of model turns.

## Planning and verification contract

- Planner instructions require narrow, self-contained subtasks and use only paths from the local inventory or explicitly declared new output paths.
- For research/documentation tasks, claims carry a source path and line reference where available, and are labeled as confirmed, reported, inferred, contested, or open question.
- `verification_command` must check the named output or behavior. A listing command such as `rg --files` alone cannot verify that a report exists or contains required sections.
- If a plan fails a local structural check, the coordinator asks the planner to repair that plan before starting workers.

## Stop, budget, and recovery behavior

- Ctrl+C interrupts the active model/tool process, closes the JSONL record with an interrupted status, and preserves partial workspace changes for inspection or resume.
- Optional CLI budgets include wall-clock duration and remote-token usage. No overall runtime/token ceiling is imposed unless configured.
- Model and shell inactivity timeouts are configurable safety controls (300 seconds by default; `0` disables the inactivity watchdog). There is no total-duration cap by default. Long-running shell commands stream output and can be cancelled without leaving child processes behind.
- No-progress protection is based on repeated identical actions/results without state changes, not on elapsed model turns.
- The dashboard shows elapsed time, turn/tool counters, last progress, failure category, configured remaining budgets, and `RUNNING`, `REPLANNING`, `PAUSED`, `INTERRUPTED`, or terminal status.

## Privacy and local data

The zero-knowledge cloud payload invariant, workspace sandbox, and serial local-GPU execution remain in force. Full traces and resumable state stay in the workspace's owner-only `.antigravity_runs` directory. New planner/replan requests contain only the already-approved privacy-filtered outline, task descriptions, and sanitized failure metadata; function bodies and secrets are not sent to the cloud.

## Compatibility

- Remove `--retry-budget` as an execution ceiling. If retained temporarily for CLI compatibility, treat it as deprecated and do not use it to stop work.
- Add optional execution-budget and inactivity-timeout settings with validation and clear HUD/CLI reporting.
- Existing JSONL replay remains compatible; new event fields are additive. Interrupted runs remain readable by older replay code.
- Keep automatic per-subtask Git checkpoints for verified work. Do not automatically clean unverified edits on ordinary failure or interruption.

## Acceptance criteria

1. A worker completes successfully after more than ten tool turns without any turn-cap failure.
2. A task that needs multiple recoveries can continue without a fixed retry-count failure.
3. Repeating the same failed action sequence against an unchanged workspace enters `REPLANNING` and then produces a revised task or a clear stalled result, without an infinite loop.
4. Planner-generated verification commands check deliverables and are rejected/repaired if they only enumerate files.
5. Ctrl+C closes the recorder, kills active child processes, reports `INTERRUPTED`, and preserves unverified edits.
6. Optional runtime/token budgets stop at a task boundary, write a resumable record, and explain the stop reason.
7. Full logs survive checkpoint commits and rollback operations, and retain verification and recovery details locally with mode `0600`.
8. The zero-knowledge privacy tests and existing dashboard/replay behavior remain valid.

## Source lessons adopted

From `~/Projects/llm-projects-playbook`: structured task briefs with verifiable completion, phased R&D work, source-status labels, durable handovers for failed attempts, and a regression/eval loop. Tool-specific Claude/OpenCode configuration is not copied.
