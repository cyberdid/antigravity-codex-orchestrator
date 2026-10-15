"""Regression: replanning a downstream subtask must validate against its real upstream tasks.

Before the fix, the check graph held only the revised subtask, so PlanValidator
raised KeyError('task_1') while resolving ancestor outputs.
"""

from __future__ import annotations

from antigravity_hybrid_orchestrator.coordinator import HybridCoordinator
from antigravity_hybrid_orchestrator.planner.dag import ExecutionPlanDAG, SubTask


def _upstream_dag() -> ExecutionPlanDAG:
    dag = ExecutionPlanDAG()
    dag.add_task(SubTask(id="task_1", title="impl", description="d", dependencies=[],
                         outputs=["calc.py"], affected_files=["calc.py"],
                         verification_command="test -s calc.py"))
    dag.add_task(SubTask(id="task_2", title="tests", description="d", dependencies=["task_1"],
                         outputs=["test_calc.py"], affected_files=["test_calc.py"],
                         verification_command="test -s test_calc.py"))
    dag.add_task(SubTask(id="task_3", title="run", description="d",
                         dependencies=["task_1", "task_2"],
                         outputs=["report.txt"], affected_files=["report.txt"],
                         verification_command="test -s report.txt"))
    dag.validate()
    return dag


class _FakeArchitect:
    """Returns a fresh plan whose task_3 is a valid, materially different revision."""

    def plan_rd_task(self, goal, workspace_outline=None):
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="task_3", title="run again", description="d2",
                             dependencies=[], outputs=["summary.txt"],
                             affected_files=["summary.txt"],
                             verification_command="test -s summary.txt && grep -q ok summary.txt"))
        return dag


def test_replan_with_upstream_dependencies_does_not_raise(tmp_path):
    coord = HybridCoordinator(mode="mock", goal="g", workspace=str(tmp_path), no_hud=True)
    coord.architect = _FakeArchitect()
    coord._rd_dag = _upstream_dag()
    stalled = coord._rd_dag.tasks["task_3"]

    revised = coord._replan_rd_subtask(stalled, attempt_history=[])

    assert revised is not None
    assert revised.id == "task_3"
    assert revised.dependencies == ["task_1", "task_2"]


def test_ancestor_ids_is_transitive():
    dag = _upstream_dag()
    assert HybridCoordinator._ancestor_ids(dag, "task_3") == {"task_1", "task_2"}
    assert HybridCoordinator._ancestor_ids(dag, "task_1") == set()
