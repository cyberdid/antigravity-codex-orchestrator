"""End-to-end tests for the universal autonomous R&D engine (Milestone 6).

Covers the coordinator/CLI integration that binds:
  workspace scan -> Cloud Architect DAG plan -> local tool-calling workforce
  -> self-correction loop -> verification gauntlet -> R&D summary.

Verification commands use ``python3 -c ...`` so the suite is deterministic and
does not depend on ``pytest`` being resolvable on PATH inside the sandbox.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from antigravity_hybrid_orchestrator.architect import MockCloudArchitectClient
from antigravity_hybrid_orchestrator.cli import build_parser
from antigravity_hybrid_orchestrator.coordinator import HybridCoordinator
from antigravity_hybrid_orchestrator.planner.dag import ExecutionPlanDAG, SubTask
from antigravity_hybrid_orchestrator.privacy.verifier import (
    SecurityInvariantViolationError,
)
from antigravity_hybrid_orchestrator.workforce.mock_worker import MockAutonomousWorker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _adder_plan() -> ExecutionPlanDAG:
    """A two-stage DAG whose verification actually imports the generated module."""
    dag = ExecutionPlanDAG()
    dag.add_task(
        SubTask(
            id="t1_impl",
            title="Implement add helper",
            description="Implement an add math function in the solution module.",
            dependencies=[],
            affected_files=["src/solution.py"],
            success_criteria="add(a, b) returns a + b.",
            verification_command=(
                "python3 -c \"import sys; sys.path.insert(0, 'src'); "
                "import solution; assert solution.add(2, 3) == 5\""
            ),
        )
    )
    dag.add_task(
        SubTask(
            id="t2_smoke",
            title="Smoke check module import",
            description="Ensure the solution module imports cleanly.",
            dependencies=["t1_impl"],
            affected_files=["src/solution.py"],
            success_criteria="Module imports without error.",
            verification_command="python3 -c \"import sys; sys.exit(0)\"",
        )
    )
    dag.validate()
    dag.metadata["input_tokens"] = 120
    dag.metadata["output_tokens"] = 40
    return dag


def _make_rd_coordinator(tmp_path: Path, **overrides) -> HybridCoordinator:
    kwargs = dict(
        mode="mock",
        goal="implement an add function",
        workspace=str(tmp_path),
        no_hud=True,
    )
    kwargs.update(overrides)
    return HybridCoordinator(**kwargs)


# ---------------------------------------------------------------------------
# CLI routing
# ---------------------------------------------------------------------------

def test_cli_parses_rd_flags():
    """--goal / --workspace / --retry-budget are parsed by the CLI."""
    parser = build_parser()
    args = parser.parse_args(
        ["--goal", "build X", "--workspace", "/tmp/ws", "--retry-budget", "3", "--mode", "mock"]
    )
    assert args.goal == "build X"
    assert args.workspace == "/tmp/ws"
    assert args.retry_budget == 3


def test_goal_activates_rd_mode(tmp_path: Path):
    """A non-empty goal switches the coordinator into R&D mode."""
    rd = HybridCoordinator(mode="mock", goal="do something", workspace=str(tmp_path), no_hud=True)
    assert rd.rd_mode is True
    assert hasattr(rd, "rd_hud")
    assert hasattr(rd, "worker")

    gauntlet = HybridCoordinator(mode="mock", no_hud=True)
    assert gauntlet.rd_mode is False
    assert hasattr(gauntlet, "test_runner")


def test_cli_forwards_goal_to_coordinator(monkeypatch, tmp_path: Path):
    """cli.main forwards goal/workspace/retry_budget into HybridCoordinator."""
    from antigravity_hybrid_orchestrator import cli as cli_mod

    captured = {}

    class _StubCoordinator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            return 0

    monkeypatch.setattr(cli_mod, "HybridCoordinator", _StubCoordinator)
    rc = cli_mod.main(
        ["--goal", "do it", "--workspace", str(tmp_path), "--retry-budget", "2", "--mode", "mock"]
    )
    assert rc == 0
    assert captured["goal"] == "do it"
    assert captured["workspace"] == str(tmp_path)
    assert captured["retry_budget"] == 2


# ---------------------------------------------------------------------------
# End-to-end success
# ---------------------------------------------------------------------------

def test_rd_engine_end_to_end_success(tmp_path: Path):
    """Full R&D pipeline completes all subtasks and writes real files."""
    coord = _make_rd_coordinator(tmp_path)
    coord.architect = MockCloudArchitectClient(custom_plan=_adder_plan())

    rc = coord.run()

    assert rc == 0
    # The generated implementation file exists in the workspace.
    assert (tmp_path / "src" / "solution.py").exists()
    # Privacy invariant recorded as preserved.
    assert coord.rd_hud.state.privacy_verified is True
    # Telemetry accumulated cloud + local usage.
    summary = coord.telemetry.get_summary()
    assert summary.cloud_tokens > 0
    assert summary.total_tokens > 0


def test_rd_engine_default_mock_plan_runs(tmp_path: Path):
    """The real mock architect (no injected plan) produces a runnable DAG.

    We rewrite verification commands to a deterministic no-op so the test does
    not depend on pytest being on PATH, while still exercising planning +
    execution + summary wiring on an architect-produced DAG.
    """
    coord = _make_rd_coordinator(tmp_path, goal="implement single add function")

    real_architect = MockCloudArchitectClient()
    original_plan = real_architect.plan_rd_task

    def _patched_plan(goal, workspace_outline=None):
        dag = original_plan(goal, workspace_outline)
        for task in dag.tasks.values():
            task.verification_command = "python3 -c \"import sys; sys.exit(0)\""
        return dag

    real_architect.plan_rd_task = _patched_plan  # type: ignore[assignment]
    coord.goal = "implement single add function"
    coord.architect = real_architect

    rc = coord.run()
    assert rc == 0


# ---------------------------------------------------------------------------
# Self-correction / failure handling
# ---------------------------------------------------------------------------

def test_rd_engine_self_correction_exhausts_budget(tmp_path: Path):
    """A persistently failing verification retries up to the budget then fails."""
    dag = ExecutionPlanDAG()
    dag.add_task(
        SubTask(
            id="t_fail",
            title="Always fails verification",
            description="Write a file then run a command that always fails.",
            dependencies=[],
            affected_files=["src/thing.py"],
            success_criteria="never",
            verification_command="false",
        )
    )
    dag.validate()
    dag.metadata["input_tokens"] = 10
    dag.metadata["output_tokens"] = 5

    coord = _make_rd_coordinator(tmp_path, retry_budget=2)
    coord.architect = MockCloudArchitectClient(custom_plan=dag)

    rc = coord.run()

    assert rc == 1
    # The subtask exhausted its retry budget (attempt count reached budget).
    assert dag.tasks["t_fail"].retry_count == 2


def test_rd_engine_worker_failure_is_reported(tmp_path: Path):
    """An injected worker failure surfaces as a failed task and non-zero exit."""
    coord = _make_rd_coordinator(tmp_path, retry_budget=1)
    coord.architect = MockCloudArchitectClient(custom_plan=_adder_plan())
    coord.worker = MockAutonomousWorker(
        fail_subtasks={"t1_impl"},
        event_callback=coord._on_worker_event,
    )

    rc = coord.run()
    assert rc == 1


# ---------------------------------------------------------------------------
# Zero-knowledge privacy
# ---------------------------------------------------------------------------

def test_rd_engine_privacy_violation_aborts(tmp_path: Path):
    """A privacy invariant violation during planning aborts with exit 1."""

    class _LeakyArchitect:
        def plan_rd_task(self, goal, workspace_outline=None):
            raise SecurityInvariantViolationError("Proprietary body leaked to cloud")

    coord = _make_rd_coordinator(tmp_path)
    coord.architect = _LeakyArchitect()

    with pytest.raises(SecurityInvariantViolationError):
        coord.run()

    assert coord.rd_hud.state.privacy_verified is False


def test_rd_engine_workspace_created_if_missing(tmp_path: Path):
    """A non-existent workspace path is created before scanning."""
    target = tmp_path / "nested" / "workspace"
    assert not target.exists()

    coord = HybridCoordinator(
        mode="mock",
        goal="implement an add function",
        workspace=str(target),
        no_hud=True,
    )
    coord.architect = MockCloudArchitectClient(custom_plan=_adder_plan())

    rc = coord.run()
    assert rc == 0
    assert target.exists()


@pytest.mark.parametrize("goal_id", [
    "E01-matrix",
    "E02-cli-flags",
    "E03-lru-cache",
    "E04-token-verify",
    "E05-sqlite-repo",
    "E06-inventory-cli",
])
def test_eval_suite_goals_pass_ground_truth_in_mock_mode(tmp_path: Path, goal_id: str):
    """All 6 eval_suite goals produce passing ground-truth implementations in mock mode."""
    import subprocess
    import sys
    from tools.eval_suite import GOALS

    spec = GOALS[goal_id]
    ws = tmp_path / goal_id
    ws.mkdir()

    coord = HybridCoordinator(
        mode="mock",
        goal=spec["goal"],
        workspace=str(ws),
        no_hud=True,
    )
    rc = coord.run()
    assert rc == 0

    truth_path = ws / "_ground_truth_check.py"
    truth_path.write_text(spec["truth"], encoding="utf-8")
    truth_proc = subprocess.run(
        [sys.executable, str(truth_path)],
        cwd=str(ws),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert truth_proc.returncode == 0, f"{goal_id} failed: {truth_proc.stdout}\n{truth_proc.stderr}"

