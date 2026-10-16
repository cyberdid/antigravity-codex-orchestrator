"""Per-subtask safety valves: attempt cap and wall-clock cap.

The fake worker rewrites a file on every attempt (so the workspace always changes)
and verification never passes. Without limits this loops forever.
"""

from __future__ import annotations

import itertools

from antigravity_hybrid_orchestrator.engine.loop import SelfCorrectionLoop, subtask_max_attempts, subtask_time_limit_s
from antigravity_hybrid_orchestrator.planner.dag import SubTask
from antigravity_hybrid_orchestrator.workforce.tools import WorkspaceToolSuite
from antigravity_hybrid_orchestrator.workforce.worker import SubtaskResult


class ChurningWorker:
    def __init__(self):
        self.counter = itertools.count()

    def execute_subtask(self, subtask, tool_suite, feedback_context=None, deadline=None):
        tool_suite.write_file("out.txt", f"attempt {next(self.counter)}\n")
        return SubtaskResult(subtask_id=subtask.id, success=True, turns=1, tool_calls_count=1, artifacts=["out.txt"])


def _task():
    return SubTask(id="t1", title="never passes", description="d", dependencies=[],
                   outputs=["out.txt"], affected_files=["out.txt"],
                   verification_command="false")


def test_attempt_cap_stops_churning_subtask(tmp_path, monkeypatch):
    monkeypatch.setenv("RD_MAX_REPAIR_ATTEMPTS", "3")
    monkeypatch.setenv("RD_SUBTASK_TIMEOUT_S", "0")
    loop = SelfCorrectionLoop(tool_suite=WorkspaceToolSuite(workspace_root=tmp_path),
                              worker=ChurningWorker(), retry_budget=50, checkpoint_git=False)
    res = loop.execute_subtask_with_retry(_task())
    assert res.success is False
    assert res.metadata.get("limit_reached") is True
    assert res.metadata["attempts"] == 3
    assert "attempt limit" in res.error


def test_time_cap_stops_subtask(tmp_path, monkeypatch):
    monkeypatch.setenv("RD_MAX_REPAIR_ATTEMPTS", "0")
    monkeypatch.setenv("RD_SUBTASK_TIMEOUT_S", "0.001")
    loop = SelfCorrectionLoop(tool_suite=WorkspaceToolSuite(workspace_root=tmp_path),
                              worker=ChurningWorker(), retry_budget=500, checkpoint_git=False)
    res = loop.execute_subtask_with_retry(_task())
    assert res.success is False
    assert "time limit" in res.error


def test_defaults_and_bad_values(monkeypatch):
    monkeypatch.delenv("RD_MAX_REPAIR_ATTEMPTS", raising=False)
    monkeypatch.delenv("RD_SUBTASK_TIMEOUT_S", raising=False)
    assert subtask_max_attempts() == 6
    assert subtask_time_limit_s() == 1200.0
    monkeypatch.setenv("RD_MAX_REPAIR_ATTEMPTS", "nope")
    assert subtask_max_attempts() == 6


class FinishedRightBeforeDeadlineWorker:
    """Simulates a worker that writes valid output on its last tool call right before deadline hits."""

    def execute_subtask(self, subtask, tool_suite, feedback_context=None, deadline=None):
        tool_suite.write_file("out.txt", "valid output\n")
        return SubtaskResult(
            subtask_id=subtask.id,
            success=False,
            turns=9,
            tool_calls_count=10,
            error="Attempt stopped: subtask time limit reached",
            artifacts=["out.txt"],
            metadata={"limit_reached": True, "backend": "ollama_native_tools"},
        )


def test_time_cap_succeeds_if_verification_already_passes(tmp_path, monkeypatch):
    monkeypatch.setenv("RD_MAX_REPAIR_ATTEMPTS", "0")
    monkeypatch.setenv("RD_SUBTASK_TIMEOUT_S", "1200")
    task = SubTask(
        id="t_done",
        title="passes verification",
        description="d",
        dependencies=[],
        outputs=["out.txt"],
        affected_files=["out.txt"],
        verification_command="test -s out.txt",
    )
    loop = SelfCorrectionLoop(
        tool_suite=WorkspaceToolSuite(workspace_root=tmp_path),
        worker=FinishedRightBeforeDeadlineWorker(),
        retry_budget=3,
        checkpoint_git=False,
    )
    res = loop.execute_subtask_with_retry(task)
    assert res.success is True
    assert task.status == "completed"


def test_worker_nudges_on_empty_response_with_eval_tokens(tmp_path):
    from antigravity_hybrid_orchestrator.workforce.worker import AutonomousWorker

    responses = [
        # Turn 1: truncated/malformed tool call -> empty content and empty tool_calls, but eval_tokens > 0
        {"content": "", "tool_calls": [], "input_tokens": 500, "output_tokens": 2826, "eval_duration_s": 10.0},
        # Turn 2: after nudge, worker writes the file
        {
            "content": "Writing file now",
            "tool_calls": [{"name": "write_file", "arguments": {"path": "out.txt", "content": "ok\n"}}],
            "input_tokens": 600,
            "output_tokens": 45,
            "eval_duration_s": 1.0,
        },
        # Turn 3: worker finishes with summary
        {"content": "Completed out.txt", "tool_calls": [], "input_tokens": 650, "output_tokens": 12, "eval_duration_s": 0.5},
    ]

    class FakeChatClient:
        def chat(self, messages, tools=None):
            return responses.pop(0)

    worker = AutonomousWorker(client=FakeChatClient())
    task = SubTask(
        id="t_nudge",
        title="nudge test",
        description="write out.txt",
        dependencies=[],
        outputs=["out.txt"],
        affected_files=["out.txt"],
        verification_command="test -s out.txt",
    )
    suite = WorkspaceToolSuite(workspace_root=tmp_path)
    res = worker.execute_subtask(task, suite)
    assert res.success is True
    assert res.turns == 3
    assert (tmp_path / "out.txt").read_text() == "ok\n"
