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
