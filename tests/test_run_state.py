import json
import os

from antigravity_hybrid_orchestrator.engine.run_state import RunStateStore
from antigravity_hybrid_orchestrator.planner.dag import ExecutionPlanDAG, SubTask


def test_run_state_round_trips_dag_and_is_owner_only(tmp_path):
    path = tmp_path / "run.state.json"
    dag = ExecutionPlanDAG()
    dag.add_task(SubTask(id="done", title="Done", status="completed"))
    dag.add_task(SubTask(id="next", dependencies=["done"], title="Next"))

    RunStateStore.save(
        path, goal="continue", workspace=str(tmp_path), dag=dag,
        remote_tokens=321, elapsed_s=45.5,
    )
    restored = RunStateStore.load(path)

    assert restored["goal"] == "continue"
    assert restored["dag"]["tasks"][0]["status"] == "completed"
    assert restored["remote_tokens"] == 321
    assert restored["elapsed_s"] == 45.5
    assert stat_mode(path) == 0o600


def test_run_state_save_replaces_corrupt_previous_state_atomically(tmp_path):
    path = tmp_path / "run.state.json"
    path.write_text("old state")
    dag = ExecutionPlanDAG()
    dag.add_task(SubTask(id="task"))

    RunStateStore.save(path, goal="g", workspace=str(tmp_path), dag=dag)

    assert json.loads(path.read_text())["dag"]["tasks"][0]["id"] == "task"
    assert list(tmp_path.glob("*.tmp")) == []


def stat_mode(path):
    return os.stat(path).st_mode & 0o777
