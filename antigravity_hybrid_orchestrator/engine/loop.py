"""Self-correction execution loop and state machine for R&D DAG subtasks."""

from __future__ import annotations

import logging
import time
import os
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from antigravity_hybrid_orchestrator.engine.verifier import (
    VerificationResult,
    VerificationRunner,
)
from antigravity_hybrid_orchestrator.engine.control import ExecutionControl
from antigravity_hybrid_orchestrator.engine.escalation import EscalationLadder
from antigravity_hybrid_orchestrator.workforce.recovery_controller import failure_excerpt
from antigravity_hybrid_orchestrator.planner.dag import ExecutionPlanDAG, SubTask
from antigravity_hybrid_orchestrator.workforce.mock_worker import MockAutonomousWorker
from antigravity_hybrid_orchestrator.workforce.tools import WorkspaceToolSuite
from antigravity_hybrid_orchestrator.workforce.worker import (
    AutonomousWorker,
    SubtaskResult,
)

logger = logging.getLogger(__name__)


class SubtaskExecutionState(str, Enum):
    """Execution lifecycle states for a single subtask."""

    PENDING = "pending"
    RUNNING = "running"
    VERIFYING = "verifying"
    REPAIRING = "repairing"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class DAGExecutionResult:
    """Consolidated outcome of executing a complete ExecutionPlanDAG."""

    success: bool
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    blocked_tasks: List[str] = field(default_factory=list)
    subtask_results: Dict[str, SubtaskResult] = field(default_factory=dict)
    verification_results: Dict[str, VerificationResult] = field(default_factory=dict)
    total_turns: int = 0
    total_tool_calls: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes DAG execution result into a dictionary."""
        return {
            "success": self.success,
            "completed_tasks": list(self.completed_tasks),
            "failed_tasks": list(self.failed_tasks),
            "blocked_tasks": list(self.blocked_tasks),
            "subtask_results": {k: v.to_dict() for k, v in self.subtask_results.items()},
            "verification_results": {k: v.to_dict() for k, v in self.verification_results.items()},
            "total_turns": self.total_turns,
            "total_tool_calls": self.total_tool_calls,
            "metadata": dict(self.metadata),
        }


def subtask_max_attempts() -> int:
    """Max attempts per subtask (RD_MAX_REPAIR_ATTEMPTS, default 6; 0 disables)."""
    try:
        return max(0, int(os.environ.get("RD_MAX_REPAIR_ATTEMPTS", "6")))
    except ValueError:
        return 6


def subtask_time_limit_s() -> float:
    """Wall-clock limit per subtask in seconds (RD_SUBTASK_TIMEOUT_S, default 1200; 0 disables)."""
    try:
        return max(0.0, float(os.environ.get("RD_SUBTASK_TIMEOUT_S", "1200")))
    except ValueError:
        return 1200.0


class SelfCorrectionLoop:
    """Manages the autonomous execution, verification, and self-repair loop for DAG tasks."""

    def __init__(
        self,
        tool_suite: WorkspaceToolSuite,
        worker: Union[AutonomousWorker, MockAutonomousWorker],
        verifier: Optional[VerificationRunner] = None,
        retry_budget: int = 3,
        checkpoint_git: bool = True,
        event_listener: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
        execution_control: Optional[ExecutionControl] = None,
        replanner: Optional[Callable[[SubTask, List[Dict[str, Any]]], Optional[SubTask]]] = None,
        acceptance_gate: Optional[Callable[[SubTask], Dict[str, Any]]] = None,
        ladder: Optional[EscalationLadder] = None,
    ) -> None:
        """Initializes the self-correction loop.

        Args:
            tool_suite: WorkspaceToolSuite providing sandboxed execution.
            worker: AutonomousWorker or MockAutonomousWorker instance.
            verifier: VerificationRunner for executing subtask verification commands.
            retry_budget: Maximum repair attempts per subtask upon failure.
            checkpoint_git: Whether to commit changes on success or rollback on failure.
            event_listener: Callback (event_type, subtask_id, payload) for HUD updates.
        """
        self.tool_suite = tool_suite
        self.worker = worker
        self.verifier = verifier or VerificationRunner(tool_suite=tool_suite)
        # Retained for source compatibility; retries are controlled by progress.
        self.retry_budget = max(0, int(retry_budget))
        self.execution_control = execution_control or ExecutionControl()
        self.replanner = replanner
        self.acceptance_gate = acceptance_gate
        self.ladder = ladder
        self._ladder_steps: List[str] = []
        self.checkpoint_git = checkpoint_git
        self.event_listener = event_listener

    def _verify(self, subtask: SubTask) -> VerificationResult:
        result = self.verifier.run_verification(subtask)
        if result.success and self.acceptance_gate is not None:
            try:
                decision = self.acceptance_gate(subtask)
            except Exception as exc:
                decision = {"status": "INCONCLUSIVE", "feedback": str(exc)}
            result.metadata["integrity"] = decision
            if decision.get("status") != "ACCEPTED":
                result.success = False
                result.exit_code = 1
                result.failure_reason = "Trusted promotion gate: " + str(decision.get("feedback") or decision.get("status"))
            self._emit("integrity_decision", subtask.id, decision)
        return result

    def _emit(self, event_type: str, subtask_id: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Dispatches an execution event to the registered HUD listener."""
        if self.event_listener is not None:
            try:
                self.event_listener(event_type, subtask_id, payload or {})
            except Exception as e:
                logger.debug("Event listener error: %s", e)

    def _workspace_fingerprint(self) -> str:
        """Hash workspace file contents to distinguish retries that made progress."""
        root = getattr(self.tool_suite, "workspace_root", None)
        if root is None:
            return "unknown"
        root = root.resolve()
        digest = hashlib.sha256()
        excluded = {".git", ".antigravity_runs", ".venv", "venv", "node_modules", "__pycache__"}
        try:
            for path in sorted(root.rglob("*")):
                if not path.is_file() or any(part in excluded for part in path.relative_to(root).parts):
                    continue
                relative = path.relative_to(root).as_posix()
                digest.update(relative.encode("utf-8", errors="replace"))
                try:
                    with path.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                except OSError:
                    continue
        except OSError:
            return "unavailable"
        return digest.hexdigest()

    @staticmethod
    def _plan_fingerprint(task: SubTask) -> str:
        fields = [task.title, task.description, task.dependencies, task.affected_files,
                  task.input_files, task.outputs, task.success_criteria, task.verification_command]
        return hashlib.sha256(repr(fields).encode("utf-8", errors="replace")).hexdigest()

    def _try_replan(
        self,
        subtask: SubTask,
        attempt_history: List[Dict[str, Any]],
        seen_plan_fingerprints: set[str],
    ) -> bool:
        """Apply a meaningfully revised plan while preserving the DAG task identity."""
        if self.replanner is None:
            return False
        try:
            revised = self.replanner(subtask, attempt_history)
            if revised is None or revised.id != subtask.id or not revised.verification_command:
                return False
            fingerprint = self._plan_fingerprint(revised)
            if fingerprint in seen_plan_fingerprints:
                return False
            seen_plan_fingerprints.add(fingerprint)
            for name in (
                "title", "description", "dependencies", "affected_files", "input_files",
                "outputs", "success_criteria", "verification_command",
            ):
                value = getattr(revised, name)
                setattr(subtask, name, list(value) if isinstance(value, list) else value)
            self._emit("replan", subtask.id, {
                "subtask": subtask.to_dict(),
                "attempt_history": attempt_history[-5:],
            })
            return True
        except Exception as exc:
            logger.warning("Subtask replan failed for %s: %s", subtask.id, exc)
            self._emit("replan_failed", subtask.id, {"error": str(exc)})
            return False

    def execute_subtask_with_retry(
        self,
        subtask: SubTask,
    ) -> SubtaskResult:
        """Executes a single subtask with automated self-correction upon verification failure.

        Args:
            subtask: Target SubTask to execute.

        Returns:
            SubtaskResult with success status, tool call counts, and artifacts.
        """
        self._ladder_steps: List[str] = []
        result = self._execute_subtask_attempts(subtask)
        result.metadata = dict(result.metadata or {})
        assisted = "cloud_assist" in self._ladder_steps
        result.metadata.update(
            think_retry_used="think_retry" in self._ladder_steps,
            ladder_steps_used=list(self._ladder_steps),
            cloud_assist_calls=int(assisted),
            solved_with_cloud_assist=bool(result.success and assisted),
        )
        return result

    def _with_cloud_guidance(
        self,
        subtask: SubTask,
        feedback_context: Optional[str],
        last_verif_res: Optional[VerificationResult],
        last_worker_res: Optional[SubtaskResult],
    ) -> Optional[str]:
        """Append owner-approved cloud guidance, if the cloud gives any, to the feedback."""
        if last_verif_res is None:
            # The attempt ended without verification (e.g. time limit): run the
            # verification command now so the cloud sees which tests fail.
            last_verif_res = self.verifier.run_verification(subtask)
        evidence = f"{last_verif_res.stderr}\n{last_verif_res.stdout}"
        excerpt = failure_excerpt(evidence, 600, raw_fallback=False) or "(no failing-test message captured)"
        guidance = self.ladder.cloud_assist.request(subtask, excerpt)
        if not guidance:
            return feedback_context
        self._use_ladder_step("cloud_assist")
        return (
            f"{feedback_context or ''}\n\n"
            f"Guidance from the cloud architect (steps, not code):\n{guidance}"
        ).strip()

    def _use_ladder_step(self, step: str) -> None:
        if step not in self._ladder_steps:
            self._ladder_steps.append(step)

    def _execute_subtask_attempts(self, subtask: SubTask) -> SubtaskResult:
        subtask.status = SubtaskExecutionState.RUNNING.value
        feedback_context: Optional[str] = None
        attempt = 0
        total_turns = 0
        total_tool_calls = 0
        last_worker_res: Optional[SubtaskResult] = None
        last_verif_res: Optional[VerificationResult] = None
        last_failure_fingerprint: Optional[str] = None
        repeated_failure_count = 0
        attempt_history: List[Dict[str, Any]] = []
        seen_plan_fingerprints = {self._plan_fingerprint(subtask)}
        subtask_started = time.monotonic()
        max_attempts = subtask_max_attempts()
        time_limit_s = subtask_time_limit_s()

        while True:
            subtask.retry_count = attempt

            # Safety valves: a subtask that keeps rewriting files without passing
            # verification must end, even when each attempt changes the workspace.
            elapsed = time.monotonic() - subtask_started
            limit_reason = None
            if max_attempts and attempt >= max_attempts:
                limit_reason = f"attempt limit reached ({max_attempts} attempts)"
            elif time_limit_s and elapsed >= time_limit_s:
                limit_reason = f"time limit reached ({time_limit_s:.0f}s)"
            if limit_reason:
                if attempt > 0 and subtask.verification_command:
                    verif_probe = self._verify(subtask)
                    if verif_probe.success:
                        subtask.status = SubtaskExecutionState.COMPLETED.value
                        self._emit("subtask_complete", subtask.id, {
                            "subtask_id": subtask.id,
                            "turns": total_turns,
                            "tool_calls": total_tool_calls,
                            "verification": verif_probe.to_dict(),
                            "attempts": attempt,
                        })
                        return SubtaskResult(
                            subtask_id=subtask.id,
                            success=True,
                            turns=total_turns,
                            tool_calls_count=total_tool_calls,
                            summary=f"Subtask '{subtask.id}' passed verification before aborting on {limit_reason}.",
                            artifacts=last_worker_res.artifacts if last_worker_res else list(subtask.outputs),
                            metadata={"attempts": attempt, "elapsed_s": round(elapsed, 1),
                                      "verification": verif_probe.to_dict()},
                        )
                subtask.status = SubtaskExecutionState.FAILED.value
                self._emit("subtask_failed", subtask.id, {
                    "subtask_id": subtask.id, "error": limit_reason, "attempts": attempt,
                })
                return SubtaskResult(
                    subtask_id=subtask.id, success=False, turns=total_turns,
                    tool_calls_count=total_tool_calls,
                    error=f"Subtask stopped: {limit_reason}",
                    artifacts=last_worker_res.artifacts if last_worker_res else [],
                    metadata={"limit_reached": True, "attempts": attempt,
                              "elapsed_s": round(elapsed, 1)},
                )

            stop = self.execution_control.check()
            if stop.stopped:
                subtask.status = SubtaskExecutionState.BLOCKED.value
                self._emit("subtask_paused", subtask.id, {"reason": stop.reason, "attempt": attempt})
                return SubtaskResult(
                    subtask_id=subtask.id, success=False, turns=total_turns,
                    tool_calls_count=total_tool_calls,
                    error=f"Execution paused: {stop.reason}",
                    artifacts=last_worker_res.artifacts if last_worker_res else [],
                    metadata={"paused": True, "stop_reason": stop.reason, "attempts": attempt},
                )

            if attempt == 0:
                self._emit(
                    "subtask_start",
                    subtask.id,
                    {"subtask": subtask.to_dict(), "attempt": attempt},
                )
            else:
                subtask.status = SubtaskExecutionState.REPAIRING.value
                self._emit(
                    "repairing",
                    subtask.id,
                    {
                        "subtask_id": subtask.id,
                        "attempt": attempt,
                        "max_attempts": None,
                        "feedback_context": feedback_context,
                    },
                )

            self._emit("attempt_started", subtask.id, {
                "attempt": attempt + 1,
                "workspace_fingerprint": self._workspace_fingerprint(),
                "feedback_context": feedback_context,
            })

            # 1. Execute subtask with worker
            if self.ladder is not None and self.ladder.assist_due(attempt):
                feedback_context = self._with_cloud_guidance(
                    subtask, feedback_context, last_verif_res, last_worker_res)
            think = self.ladder.think_for_attempt(attempt) if self.ladder else None
            # Workers that predate the ladder do not accept ``think``; pass it only when set.
            ladder_kwargs: Dict[str, Any] = {}
            if think:
                self._use_ladder_step("think_retry")
                ladder_kwargs["think"] = think
            worker_res = self.worker.execute_subtask(
                subtask=subtask,
                tool_suite=self.tool_suite,
                feedback_context=feedback_context,
                deadline=(subtask_started + time_limit_s) if time_limit_s else None,
                **ladder_kwargs,
            )
            if worker_res.metadata.get("budget_escalations"):
                self._use_ladder_step("adaptive_limit")
            total_turns += worker_res.turns
            total_tool_calls += worker_res.tool_calls_count
            last_worker_res = worker_res
            workspace_fingerprint = self._workspace_fingerprint()
            self._emit("attempt_result", subtask.id, {
                "attempt": attempt + 1,
                "success": worker_res.success,
                "error": worker_res.error,
                "workspace_fingerprint": workspace_fingerprint,
                "worker": worker_res.to_dict(),
            })
            attempt_history.append({
                "attempt": attempt + 1,
                "kind": "worker",
                "success": worker_res.success,
                "error": worker_res.error,
                "workspace_fingerprint": workspace_fingerprint,
            })

            if (
                not worker_res.success
                and (
                    worker_res.metadata.get("limit_reached")
                    or (
                        worker_res.tool_calls_count > 0
                        and "timed out" in (worker_res.error or "").lower()
                    )
                )
                and subtask.verification_command
            ):
                # If the worker finished writing/compiling files on a prior tool call
                # right before the subtask timer or Ollama HTTP request timed out, verify before discarding its work.
                verif_probe = self._verify(subtask)
                if verif_probe.success:
                    worker_res.success = True
                    worker_res.error = None

            if not worker_res.success:
                failure_fingerprint = f"{worker_res.error or 'worker_failed'}:{workspace_fingerprint}"
                repeated_failure_count = repeated_failure_count + 1 if failure_fingerprint == last_failure_fingerprint else 0
                last_failure_fingerprint = failure_fingerprint
                if worker_res.metadata.get("stalled") or repeated_failure_count >= 2:
                    if self._try_replan(subtask, attempt_history, seen_plan_fingerprints):
                        repeated_failure_count = 0
                        last_failure_fingerprint = None
                        attempt += 1
                        feedback_context = (
                            "The previous approach stalled. The task plan has been revised; "
                            "inspect the current workspace and execute the revised scope."
                        )
                        continue
                    subtask.status = SubtaskExecutionState.BLOCKED.value
                    self._emit("subtask_stalled", subtask.id, {
                        "error": worker_res.error,
                        "repeats": repeated_failure_count + 1,
                        "attempt": attempt + 1,
                    })
                    return SubtaskResult(
                        subtask_id=subtask.id, success=False, turns=total_turns,
                        tool_calls_count=total_tool_calls,
                        error=f"Subtask stalled after repeated failure: {worker_res.error}",
                        artifacts=worker_res.artifacts,
                        metadata={
                            **dict(worker_res.metadata or {}),
                            "stalled": True,
                            "attempts": attempt + 1,
                            "wall_clock_s": round(time.monotonic() - subtask_started, 4),
                        },
                    )
                attempt += 1
                feedback_context = (
                    f"Previous execution attempt failed: {worker_res.error}\n"
                    "Review the full tool history, inspect the current workspace, change the approach, "
                    "and do not repeat an action that produced no progress."
                )
                continue

            # 2. Run verification command if defined
            subtask.status = SubtaskExecutionState.VERIFYING.value
            self._emit("verifying", subtask.id, {"command": subtask.verification_command})

            verif_res = self._verify(subtask)
            last_verif_res = verif_res

            if verif_res.success:
                subtask.status = SubtaskExecutionState.COMPLETED.value
                self._emit(
                    "subtask_complete",
                    subtask.id,
                    {
                        "subtask_id": subtask.id,
                        "turns": total_turns,
                        "tool_calls": total_tool_calls,
                        "verification": verif_res.to_dict(),
                        "attempts": attempt + 1,
                    },
                )
                if self.checkpoint_git:
                    self._commit_checkpoint(f"chore(subtask): completed {subtask.id} - {subtask.title}")
                return SubtaskResult(
                    subtask_id=subtask.id,
                    success=True,
                    turns=total_turns,
                    tool_calls_count=total_tool_calls,
                    summary=worker_res.summary or f"Subtask '{subtask.id}' verified and completed.",
                    artifacts=worker_res.artifacts,
                    metadata={
                        **dict(worker_res.metadata or {}),
                        "attempts": attempt + 1,
                        "successful_verifications": int((worker_res.metadata or {}).get("successful_verifications", 0)) + 1,
                        "wall_clock_s": round(time.monotonic() - subtask_started, 4),
                        "verification": verif_res.to_dict(),
                    },
                )

            # Verification failed: prepare self-correction feedback
            self._emit(
                "verification_failed",
                subtask.id,
                {
                    "subtask_id": subtask.id,
                    "command": subtask.verification_command,
                    "exit_code": verif_res.exit_code,
                    "failure_reason": verif_res.failure_reason,
                    "stdout": verif_res.stdout,
                    "stderr": verif_res.stderr,
                    "duration_s": verif_res.duration_sec,
                    "metadata": dict(verif_res.metadata),
                    "attempt": attempt + 1,
                    "max_attempts": None,
                },
            )

            attempt += 1
            failure_fingerprint = (
                f"verify:{verif_res.exit_code}:{verif_res.failure_reason}:"
                f"{verif_res.stderr}:{self._workspace_fingerprint()}"
            )
            repeated_failure_count = repeated_failure_count + 1 if failure_fingerprint == last_failure_fingerprint else 0
            last_failure_fingerprint = failure_fingerprint
            attempt_history.append({
                "attempt": attempt,
                "kind": "verification",
                "success": False,
                "exit_code": verif_res.exit_code,
                "failure_reason": verif_res.failure_reason,
                "workspace_fingerprint": self._workspace_fingerprint(),
            })
            if repeated_failure_count >= 2:
                if self._try_replan(subtask, attempt_history, seen_plan_fingerprints):
                    repeated_failure_count = 0
                    last_failure_fingerprint = None
                    feedback_context = (
                        "Verification repeated the same failure. The plan was revised; inspect "
                        "the current workspace and satisfy the revised verification command."
                    )
                    continue
                subtask.status = SubtaskExecutionState.BLOCKED.value
                self._emit("subtask_stalled", subtask.id, {
                    "error": verif_res.failure_reason,
                    "repeats": repeated_failure_count + 1,
                    "verification": verif_res.to_dict(),
                })
                return SubtaskResult(
                    subtask_id=subtask.id, success=False, turns=total_turns,
                    tool_calls_count=total_tool_calls,
                    error=f"Verification stalled after unchanged failures: {verif_res.failure_reason}",
                    artifacts=worker_res.artifacts,
                    metadata={"stalled": True, "attempts": attempt,
                              "verification": verif_res.to_dict()},
                )
            feedback_context = (
                f"Verification command '{subtask.verification_command}' failed with exit code {verif_res.exit_code}.\n"
                f"Failure diagnostics:\n{verif_res.failure_reason}\n{verif_res.stdout}\n{verif_res.stderr}\n\n"
                "Inspect current files, change the approach, then rerun the verification command."
            )

        # Fallback (should not be reached)
        subtask.status = SubtaskExecutionState.FAILED.value
        return SubtaskResult(
            subtask_id=subtask.id,
            success=False,
            turns=total_turns,
            tool_calls_count=total_tool_calls,
            error="Exhausted retry budget",
        )

    def execute_dag(
        self,
        dag: ExecutionPlanDAG,
    ) -> DAGExecutionResult:
        """Executes the entire ExecutionPlanDAG in valid topological order.

        Args:
            dag: ExecutionPlanDAG containing subtasks and dependencies.

        Returns:
            DAGExecutionResult with pass/fail status and per-task results.
        """
        execution_order = dag.get_execution_order()
        completed_tasks: List[str] = [
            task.id for task in execution_order if task.status == SubtaskExecutionState.COMPLETED.value
        ]
        failed_tasks: List[str] = []
        blocked_tasks: List[str] = []
        subtask_results: Dict[str, SubtaskResult] = {}
        verification_results: Dict[str, VerificationResult] = {}
        total_turns = 0
        total_tool_calls = 0

        for subtask in execution_order:
            if subtask.id in completed_tasks:
                self._emit("subtask_resume_skipped", subtask.id, {
                    "reason": "previously_verified",
                    "status": subtask.status,
                })
                continue
            # Check if all upstream dependencies are satisfied
            unmet_deps = [dep for dep in subtask.dependencies if dep not in completed_tasks]
            if unmet_deps:
                subtask.status = SubtaskExecutionState.BLOCKED.value
                blocked_tasks.append(subtask.id)
                self._emit(
                    "subtask_blocked",
                    subtask.id,
                    {"subtask_id": subtask.id, "unmet_dependencies": unmet_deps},
                )
                continue

            result = self.execute_subtask_with_retry(subtask)
            subtask_results[subtask.id] = result
            total_turns += result.turns
            total_tool_calls += result.tool_calls_count

            if "verification" in result.metadata:
                verif_dict = result.metadata["verification"]
                verification_results[subtask.id] = VerificationResult(
                    success=verif_dict.get("success", False),
                    exit_code=verif_dict.get("exit_code", 1),
                    stdout=verif_dict.get("stdout", ""),
                    stderr=verif_dict.get("stderr", ""),
                    failure_reason=verif_dict.get("failure_reason", ""),
                    duration_sec=verif_dict.get("duration_sec", 0.0),
                    command=verif_dict.get("command", ""),
                )

            if result.success:
                dag.mark_completed(subtask.id)
                completed_tasks.append(subtask.id)
            else:
                dag.mark_failed(subtask.id)
                failed_tasks.append(subtask.id)

        all_success = len(completed_tasks) == len(dag.tasks) and not failed_tasks and not blocked_tasks

        return DAGExecutionResult(
            success=all_success,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            blocked_tasks=blocked_tasks,
            subtask_results=subtask_results,
            verification_results=verification_results,
            total_turns=total_turns,
            total_tool_calls=total_tool_calls,
            metadata={"total_tasks": len(dag.tasks)},
        )

    def _commit_checkpoint(self, message: str) -> None:
        """Creates a git commit checkpoint if workspace is a git repository."""
        if not self.tool_suite.is_git_repo:
            return
        self.tool_suite.run_command("git add -A -- . ':(exclude).antigravity_runs/**'")
        import shlex
        self.tool_suite.run_command(f'git commit -m {shlex.quote(message)} --allow-empty')

    def _rollback_uncommitted(self, subtask_id: str = "") -> None:
        """Discards uncommitted workspace modifications via git checkout."""
        if not self.tool_suite.is_git_repo:
            return
        reset_command = "git reset --hard HEAD"
        clean_command = "git clean -fd -e .antigravity_runs/"
        self._emit(
            "rollback_started",
            subtask_id,
            {"commands": [reset_command, clean_command]},
        )
        reset_result = self.tool_suite.run_command(reset_command)
        clean_result = self.tool_suite.run_command(clean_command)
        self._emit(
            "rollback_complete",
            subtask_id,
            {
                "success": reset_result.success and clean_result.success,
                "reset": reset_result.to_dict(),
                "clean": clean_result.to_dict(),
            },
        )
