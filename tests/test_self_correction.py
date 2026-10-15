"""Comprehensive tests for Milestone 4: Autonomous Feedback & Self-Correction Loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from antigravity_hybrid_orchestrator.engine.loop import (
    DAGExecutionResult,
    SelfCorrectionLoop,
    SubtaskExecutionState,
)
from antigravity_hybrid_orchestrator.engine.verifier import (
    VerificationResult,
    VerificationRunner,
)
from antigravity_hybrid_orchestrator.planner.dag import ExecutionPlanDAG, SubTask
from antigravity_hybrid_orchestrator.workforce.mock_worker import MockAutonomousWorker
from antigravity_hybrid_orchestrator.workforce.tools import WorkspaceToolSuite


@pytest.fixture
def temp_workspace(tmp_path: Path) -> WorkspaceToolSuite:
    """Fixture returning a WorkspaceToolSuite initialized in an isolated directory."""
    return WorkspaceToolSuite(workspace_root=tmp_path)


# ---------------------------------------------------------------------------
# VerificationRunner Tests
# ---------------------------------------------------------------------------

def test_verification_runner_no_command_returns_success(temp_workspace: WorkspaceToolSuite):
    """Subtask without verification command automatically passes."""
    runner = VerificationRunner(tool_suite=temp_workspace)
    task = SubTask(id="t1", title="No command task", verification_command="")
    res = runner.run_verification(task)

    assert res.success is True
    assert res.exit_code == 0
    assert "No verification command" in res.stdout


def test_verification_runner_successful_command(temp_workspace: WorkspaceToolSuite):
    """Subtask with passing shell command succeeds with exit code 0."""
    runner = VerificationRunner(tool_suite=temp_workspace)
    task = SubTask(id="t1", title="Echo task", verification_command="echo 'ALL_GOOD'")
    res = runner.run_verification(task)

    assert res.success is True
    assert res.exit_code == 0
    assert "ALL_GOOD" in res.stdout
    assert res.failure_reason == ""
    assert res.duration_sec >= 0.0


def test_verification_runner_has_no_implicit_wall_clock_timeout(temp_workspace, monkeypatch):
    runner = VerificationRunner(tool_suite=temp_workspace)
    seen = {}

    def run_command(**kwargs):
        seen.update(kwargs)
        return type("Result", (), {"success": True, "output": "ok", "error": "", "metadata": {"exit_code": 0}})()

    monkeypatch.setattr(temp_workspace, "run_command", run_command)
    result = runner.run_verification(SubTask(id="t1", title="Long suite", verification_command="pytest"))

    assert result.success
    assert seen["timeout"] is None


def test_verification_runner_failing_command_extracts_exit_code(temp_workspace: WorkspaceToolSuite):
    """Failing shell command returns success=False and extracts non-zero exit code."""
    runner = VerificationRunner(tool_suite=temp_workspace)
    task = SubTask(id="t1", title="Failing task", verification_command="false")
    res = runner.run_verification(task)

    assert res.success is False
    assert res.exit_code == 1
    assert "1" in res.failure_reason


def test_verification_runner_extracts_python_traceback(temp_workspace: WorkspaceToolSuite):
    """Verification command that raises Python Exception captures full traceback."""
    runner = VerificationRunner(tool_suite=temp_workspace)
    task = SubTask(
        id="t1",
        title="Traceback task",
        verification_command="python3 -c 'raise ValueError(\"Custom error message\")'",
    )
    res = runner.run_verification(task)

    assert res.success is False
    assert res.exit_code != 0
    assert "ValueError: Custom error message" in res.failure_reason
    assert "Traceback" in res.failure_reason


def test_verification_runner_extracts_syntax_error(temp_workspace: WorkspaceToolSuite):
    """Verification command failing with syntax error captures syntax diagnostic."""
    runner = VerificationRunner(tool_suite=temp_workspace)
    task = SubTask(
        id="t1",
        title="Syntax error task",
        verification_command="python3 -c 'def bad_syntax(: pass'",
    )
    res = runner.run_verification(task)

    assert res.success is False
    assert "SyntaxError" in res.failure_reason


def test_verification_runner_timeout_handling(temp_workspace: WorkspaceToolSuite):
    """Verification command exceeding timeout fails cleanly with timeout diagnostic."""
    runner = VerificationRunner(tool_suite=temp_workspace)
    task = SubTask(
        id="t1",
        title="Hanging task",
        verification_command="sleep 10",
    )
    res = runner.run_verification(task, timeout=1)

    assert res.success is False
    assert "timed out" in res.failure_reason.lower()
    assert "timed out" in res.failure_reason.lower() or res.exit_code != 0


# ---------------------------------------------------------------------------
# SelfCorrectionLoop Tests
# ---------------------------------------------------------------------------

def test_self_correction_loop_single_subtask_immediate_success(temp_workspace: WorkspaceToolSuite):
    """Subtask executes and passes on the first attempt without retries."""
    worker = MockAutonomousWorker()
    loop = SelfCorrectionLoop(tool_suite=temp_workspace, worker=worker, retry_budget=2)

    task = SubTask(
        id="t1",
        title="Create hello file",
        affected_files=["hello.txt"],
        verification_command="test -f hello.txt",
    )
    res = loop.execute_subtask_with_retry(task)

    assert res.success is True
    assert task.status == SubtaskExecutionState.COMPLETED.value
    assert task.retry_count == 0
    assert (temp_workspace.workspace_root / "hello.txt").exists()


def test_self_correction_loop_retry_on_verification_failure_recovers(
    temp_workspace: WorkspaceToolSuite,
):
    """Subtask fails initial verification, but mock repair updates content and recovers."""
    # Worker fails on first attempt, fixes on second attempt
    custom_contents = {
        "calc.py": "def add(a, b): return a + b\n",
    }
    worker = MockAutonomousWorker(custom_file_contents=custom_contents)
    loop = SelfCorrectionLoop(tool_suite=temp_workspace, worker=worker, retry_budget=2)

    task = SubTask(
        id="calc",
        title="Implement addition calculator",
        affected_files=["calc.py"],
        verification_command="python3 -c 'import calc; assert calc.add(2, 3) == 5'",
    )
    res = loop.execute_subtask_with_retry(task)

    assert res.success is True
    assert task.status == SubtaskExecutionState.COMPLETED.value
    assert (temp_workspace.workspace_root / "calc.py").exists()


def test_self_correction_loop_stops_unchanged_verification_failure(temp_workspace: WorkspaceToolSuite):
    """Identical verifier output is retried for diagnosis, then marked blocked."""
    worker = MockAutonomousWorker()
    loop = SelfCorrectionLoop(tool_suite=temp_workspace, worker=worker, retry_budget=2)

    task = SubTask(
        id="t_fail",
        title="Impossible task",
        affected_files=["fail.py"],
        verification_command="false",
    )
    res = loop.execute_subtask_with_retry(task)

    assert res.success is False
    assert task.status == SubtaskExecutionState.BLOCKED.value
    assert res.metadata.get("stalled") is True
    assert "stalled" in res.error


def test_self_correction_loop_stops_repeated_worker_error(temp_workspace: WorkspaceToolSuite):
    """An unchanged worker failure is not retried indefinitely."""
    worker = MockAutonomousWorker(error_on_tool={"write_file": "Disk full simulation"})
    loop = SelfCorrectionLoop(tool_suite=temp_workspace, worker=worker, retry_budget=1)

    task = SubTask(id="t_err", title="Error task", affected_files=["test.py"])
    res = loop.execute_subtask_with_retry(task)

    assert res.success is False
    assert task.status == SubtaskExecutionState.BLOCKED.value
    assert res.metadata.get("stalled") is True
    assert "Disk full simulation" in res.error


def test_recovery_can_exceed_legacy_retry_budget_when_workspace_changes(
    temp_workspace: WorkspaceToolSuite,
):
    from antigravity_hybrid_orchestrator.workforce.worker import SubtaskResult

    class ProgressingWorker:
        def __init__(self):
            self.calls = 0

        def execute_subtask(self, subtask, tool_suite, feedback_context=None, deadline=None):
            self.calls += 1
            tool_suite.write_file("progress.txt", str(self.calls))
            if self.calls < 5:
                return SubtaskResult(subtask_id=subtask.id, success=False, error="still fixing")
            return SubtaskResult(subtask_id=subtask.id, success=True, summary="ready")

    worker = ProgressingWorker()
    loop = SelfCorrectionLoop(tool_suite=temp_workspace, worker=worker, retry_budget=0)
    task = SubTask(id="long-repair", title="Repair with evidence", verification_command="true")

    result = loop.execute_subtask_with_retry(task)

    assert result.success is True
    assert worker.calls == 5
    assert result.metadata["attempts"] == 5
    assert (temp_workspace.workspace_root / "progress.txt").read_text() == "5"


def test_stalled_task_is_replanned_before_becoming_blocked(temp_workspace: WorkspaceToolSuite):
    replans = []

    def replan(task, history):
        replans.append((task.id, len(history)))
        return SubTask(
            id=task.id,
            title="Revised verification approach",
            description="Use a meaningful local check",
            verification_command="true",
        )

    loop = SelfCorrectionLoop(
        tool_suite=temp_workspace,
        worker=MockAutonomousWorker(),
        replanner=replan,
        checkpoint_git=False,
    )
    task = SubTask(id="replan", title="Original", verification_command="false")

    result = loop.execute_subtask_with_retry(task)

    assert result.success is True
    assert replans == [("replan", 3)]
    assert task.title == "Revised verification approach"
    assert result.metadata["attempts"] == 4


def test_self_correction_loop_emits_hud_events_in_sequence(temp_workspace: WorkspaceToolSuite):
    """Event listener receives correct sequence of HUD events."""
    events: List[tuple[str, str, Dict[str, Any]]] = []

    def on_event(event_type: str, subtask_id: str, payload: Dict[str, Any]):
        events.append((event_type, subtask_id, payload))

    worker = MockAutonomousWorker()
    loop = SelfCorrectionLoop(
        tool_suite=temp_workspace,
        worker=worker,
        retry_budget=1,
        event_listener=on_event,
    )

    task = SubTask(
        id="event_task",
        title="Event test",
        affected_files=["event.txt"],
        verification_command="true",
    )
    res = loop.execute_subtask_with_retry(task)

    assert res.success is True
    event_names = [e[0] for e in events]
    assert "subtask_start" in event_names
    assert "verifying" in event_names
    assert "subtask_complete" in event_names


def test_verification_failure_event_records_full_attempt_diagnostics(
    temp_workspace: WorkspaceToolSuite,
):
    from antigravity_hybrid_orchestrator.workforce.worker import SubtaskResult

    events: List[tuple[str, str, Dict[str, Any]]] = []

    class WorkerThatDoesNotRunVerification:
        def execute_subtask(self, subtask, tool_suite, feedback_context=None, deadline=None):
            return SubtaskResult(subtask_id=subtask.id, success=True)

    loop = SelfCorrectionLoop(
        tool_suite=temp_workspace,
        worker=WorkerThatDoesNotRunVerification(),
        retry_budget=0,
        checkpoint_git=False,
        event_listener=lambda kind, task_id, payload: events.append((kind, task_id, payload)),
    )
    task = SubTask(
        id="diagnostics",
        title="Capture verification output",
        verification_command=(
            "python3 -c 'import sys; print(\"stdout marker\"); "
            "print(\"stderr marker\", file=sys.stderr); sys.exit(7)'"
        ),
    )

    result = loop.execute_subtask_with_retry(task)
    failure = next(payload for kind, _, payload in events if kind == "verification_failed")

    assert result.success is False
    assert failure["attempt"] == 1
    assert failure["max_attempts"] is None
    assert failure["command"] == task.verification_command
    assert failure["exit_code"] == 7
    assert "stdout marker" in failure["stdout"]
    assert "stderr marker" in failure["stderr"]
    assert failure["duration_s"] >= 0


def test_rollback_preserves_run_recording_and_removes_untracked_outputs(tmp_path: Path):
    import subprocess

    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    (tmp_path / "tracked.txt").write_text("committed\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "initial"],
        check=True,
        capture_output=True,
        text=True,
    )
    recording = tmp_path / ".antigravity_runs" / "run-test.jsonl"
    recording.parent.mkdir()
    recording.write_text('{"kind":"header"}\n', encoding="utf-8")
    generated = tmp_path / "docs" / "analysis.md"
    generated.parent.mkdir()
    generated.write_text("partial report\n", encoding="utf-8")

    events: List[tuple[str, str, Dict[str, Any]]] = []
    loop = SelfCorrectionLoop(
        tool_suite=WorkspaceToolSuite(workspace_root=tmp_path),
        worker=MockAutonomousWorker(),
        event_listener=lambda kind, task_id, payload: events.append((kind, task_id, payload)),
    )
    loop._rollback_uncommitted("failed_task")

    assert recording.exists()
    assert not generated.exists()
    rollback = next(payload for kind, _, payload in events if kind == "rollback_complete")
    assert rollback["success"] is True
    assert rollback["clean"]["success"] is True


def test_git_checkpoint_does_not_commit_run_recordings(tmp_path: Path):
    import subprocess

    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    (tmp_path / "tracked.txt").write_text("committed\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "initial"],
        check=True,
        capture_output=True,
        text=True,
    )
    recording = tmp_path / ".antigravity_runs" / "run-test.jsonl"
    recording.parent.mkdir()
    recording.write_text('{"kind":"header"}\n', encoding="utf-8")
    generated = tmp_path / "report.md"
    generated.write_text("verified output\n", encoding="utf-8")

    loop = SelfCorrectionLoop(
        tool_suite=WorkspaceToolSuite(workspace_root=tmp_path),
        worker=MockAutonomousWorker(),
    )
    loop._commit_checkpoint("checkpoint")
    committed_paths = subprocess.run(
        ["git", "-C", str(tmp_path), "show", "--pretty=format:", "--name-only", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    assert "report.md" in committed_paths
    assert ".antigravity_runs/run-test.jsonl" not in committed_paths
    assert recording.exists()


def test_self_correction_loop_executes_entire_dag_topologically(
    temp_workspace: WorkspaceToolSuite,
):
    """Executes multi-task DAG in topological order with verification."""
    t1 = SubTask(id="step1", title="Step 1", affected_files=["s1.py"], verification_command="true")
    t2 = SubTask(
        id="step2",
        title="Step 2",
        dependencies=["step1"],
        affected_files=["s2.py"],
        verification_command="true",
    )
    t3 = SubTask(
        id="step3",
        title="Step 3",
        dependencies=["step2"],
        affected_files=["s3.py"],
        verification_command="true",
    )
    dag = ExecutionPlanDAG(tasks=[t1, t2, t3])

    worker = MockAutonomousWorker()
    loop = SelfCorrectionLoop(tool_suite=temp_workspace, worker=worker)
    dag_res = loop.execute_dag(dag)

    assert dag_res.success is True
    assert dag_res.completed_tasks == ["step1", "step2", "step3"]
    assert dag_res.failed_tasks == []
    assert dag_res.blocked_tasks == []
    assert dag.is_completed() is True
    assert (temp_workspace.workspace_root / "s1.py").exists()
    assert (temp_workspace.workspace_root / "s2.py").exists()
    assert (temp_workspace.workspace_root / "s3.py").exists()


def test_self_correction_loop_blocks_downstream_task_on_upstream_failure(
    temp_workspace: WorkspaceToolSuite,
):
    """Upstream subtask failure prevents downstream dependent tasks from executing."""
    t1 = SubTask(id="setup", title="Setup", verification_command="false")  # Will fail
    t2 = SubTask(id="build", title="Build", dependencies=["setup"])
    t3 = SubTask(id="test", title="Test", dependencies=["build"])
    dag = ExecutionPlanDAG(tasks=[t1, t2, t3])

    worker = MockAutonomousWorker()
    loop = SelfCorrectionLoop(tool_suite=temp_workspace, worker=worker, retry_budget=0)
    dag_res = loop.execute_dag(dag)

    assert dag_res.success is False
    assert "setup" in dag_res.failed_tasks
    assert "build" in dag_res.blocked_tasks
    assert "test" in dag_res.blocked_tasks
    assert dag.tasks["build"].status == SubtaskExecutionState.BLOCKED.value
    assert dag.tasks["test"].status == SubtaskExecutionState.BLOCKED.value


def test_dag_execution_result_serialization():
    """DAGExecutionResult serializes to dictionary cleanly."""
    res = DAGExecutionResult(
        success=True,
        completed_tasks=["t1", "t2"],
        failed_tasks=[],
        blocked_tasks=[],
        total_turns=4,
        total_tool_calls=6,
    )
    data = res.to_dict()

    assert data["success"] is True
    assert data["completed_tasks"] == ["t1", "t2"]
    assert data["total_turns"] == 4
    assert data["total_tool_calls"] == 6
