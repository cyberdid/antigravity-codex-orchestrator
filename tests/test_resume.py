from antigravity_hybrid_orchestrator.coordinator import HybridCoordinator
from antigravity_hybrid_orchestrator.engine.run_state import RunStateStore
from antigravity_hybrid_orchestrator.planner.dag import ExecutionPlanDAG, SubTask


def test_resume_skips_verified_tasks_and_continues_pending_work(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_log = workspace / ".antigravity_runs" / "run-old.jsonl"
    run_log.parent.mkdir()
    run_log.write_text('{"kind":"header"}\n')
    dag = ExecutionPlanDAG()
    dag.metadata.update({"input_tokens": 80, "output_tokens": 20})
    dag.add_task(SubTask(id="done", title="Already verified", status="completed"))
    dag.add_task(SubTask(id="next", title="Continue", dependencies=["done"], verification_command="true"))
    RunStateStore.save(
        RunStateStore.sidecar_for(run_log), goal="continue work", workspace=str(workspace), dag=dag,
        remote_tokens=50, elapsed_s=1,
    )

    def unexpected_plan(*args, **kwargs):
        raise AssertionError("resume must restore its existing DAG, not ask the planner again")

    monkeypatch.setattr(
        "antigravity_hybrid_orchestrator.architect.MockCloudArchitectClient.plan_rd_task",
        unexpected_plan,
    )
    coordinator = HybridCoordinator(
        mode="mock", workspace=str(workspace), resume_run=str(run_log), record=False,
        no_hud=True, max_remote_tokens=100,
    )

    assert coordinator.run() == 0

    restored = RunStateStore.load(RunStateStore.sidecar_for(run_log))
    tasks = {task["id"]: task for task in restored["dag"]["tasks"]}
    assert tasks["done"]["status"] == "completed"
    assert tasks["next"]["status"] == "completed"
    assert restored["remote_tokens"] == 50
