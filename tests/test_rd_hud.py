"""Focused tests for the autonomous R&D engine HUD (Milestone 5).

These exercise the headless (no-TTY) rendering path and the event-listener
integration that binds the SelfCorrectionLoop to the live dashboard.
"""

from __future__ import annotations

from antigravity_hybrid_orchestrator.engine.loop import SubtaskExecutionState
from antigravity_hybrid_orchestrator.hud.rd_dashboard import RDDashboard
from antigravity_hybrid_orchestrator.planner.dag import ExecutionPlanDAG, SubTask
from antigravity_hybrid_orchestrator.telemetry import TelemetryTracker


def _dag() -> ExecutionPlanDAG:
    dag = ExecutionPlanDAG()
    dag.add_task(
        SubTask(id="a", title="First", description="do a", dependencies=[], verification_command="true")
    )
    dag.add_task(
        SubTask(id="b", title="Second", description="do b", dependencies=["a"], verification_command="true")
    )
    dag.validate()
    return dag


def _hud() -> RDDashboard:
    # no_hud=True forces the headless streaming path (no live TTY required).
    return RDDashboard(no_hud=True, goal="build a thing", workspace="/tmp/ws")


def test_init_dag_populates_subtasks_in_order():
    hud = _hud()
    hud.init_dag(_dag())
    ids = [s["id"] for s in hud.state.subtasks]
    assert ids == ["a", "b"]
    assert all(s["status"] == "pending" for s in hud.state.subtasks)


def test_handle_event_lifecycle_transitions_status():
    hud = _hud()
    hud.init_dag(_dag())

    hud.handle_event("subtask_start", "a", {"subtask": {"description": "do a"}, "attempt": 0})
    assert any(
        s["id"] == "a" and s["status"] == SubtaskExecutionState.RUNNING.value
        for s in hud.state.subtasks
    )

    hud.handle_event("subtask_complete", "a", {})
    assert any(
        s["id"] == "a" and s["status"] == SubtaskExecutionState.COMPLETED.value
        for s in hud.state.subtasks
    )


def test_as_event_listener_is_callable_and_records_tool_calls():
    hud = _hud()
    listener = hud.as_event_listener()
    assert callable(listener)

    hud.on_tool_call("write_file", {"path": "src/x.py"})
    assert hud.state.active_tool_call is not None
    assert "write_file" in hud.state.active_tool_call


def test_privacy_and_telemetry_state_updates():
    hud = _hud()
    tracker = TelemetryTracker()
    tracker.add_cloud(input_tokens=100, output_tokens=50)
    tracker.add_local(eval_tokens=200, duration_s=1.0)

    hud.update_telemetry(tracker.get_summary())
    assert hud.state.telemetry_summary is not None
    assert hud.state.telemetry_summary.total_tokens == 350

    hud.set_privacy_verified(False)
    assert hud.state.privacy_verified is False


def test_print_final_summary_runs_headless(capsys):
    from antigravity_hybrid_orchestrator.engine.loop import DAGExecutionResult

    hud = _hud()
    dag = _dag()
    hud.init_dag(dag)
    result = DAGExecutionResult(
        success=True,
        completed_tasks=["a", "b"],
        total_turns=4,
        total_tool_calls=6,
    )
    hud.print_final_summary(dag_result=result, dag=dag, duration_s=1.23, privacy_verified=True)
    out = capsys.readouterr().out
    assert "SUCCESS" in out.upper() or "SUBTASK" in out.upper()
