"""Comprehensive test suite for Milestone 2: Cloud Architect Strategic Planner (Execution DAG).

Covers:
- Feature F5: Execution DAG Data Model (SubTask, ExecutionPlanDAG, Kahn's topological sort,
              cycle detection, missing dependency detection, task readiness, and serialization).
- Feature F6: Cloud Architect DAG Planner (CloudArchitectClient.plan_rd_task, prompt building,
              pre-flight privacy verification, markdown fence stripping, retry loop).
- Feature F7: Mock Cloud Architect Planner (MockCloudArchitectClient.plan_rd_task, deterministic
              plan archetypes, injection hooks, offline privacy verification, backward compatibility).
- Tier 3: Cross-Feature Integration (WorkspaceScanner -> Privacy Verifier -> Planner -> DAG -> Readiness Loop).
"""

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

_REAL_POPEN = subprocess.Popen

from antigravity_hybrid_orchestrator.architect.client import (
    CloudArchitectClient,
    CloudArchitectError,
)
from antigravity_hybrid_orchestrator.architect.mock_client import (
    MockCloudArchitectClient,
)
from antigravity_hybrid_orchestrator.planner.dag import (
    DAGCycleError,
    DAGDependencyError,
    DAGError,
    DAGTaskNotFoundError,
    ExecutionPlanDAG,
    SubTask,
    VALID_SUBTASK_STATUSES,
)
from antigravity_hybrid_orchestrator.privacy.verifier import (
    SecurityInvariantViolationError,
    verify_cloud_payload,
)
from antigravity_hybrid_orchestrator.workspace.scanner import (
    WorkspaceOutline,
    WorkspaceScanner,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def clean_workspace(tmp_path: Path) -> Path:
    """Creates a temporary realistic clean Python project workspace."""
    ws = tmp_path / "sample_project"
    ws.mkdir()

    (ws / "pyproject.toml").write_text(
        '[project]\nname = "sample"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    src = ws / "src"
    src.mkdir()
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "math_helper.py").write_text(
        "def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8"
    )

    tests = ws / "tests"
    tests.mkdir()
    (tests / "test_math.py").write_text(
        "from src.math_helper import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    return ws


@pytest.fixture
def clean_outline(clean_workspace: Path) -> WorkspaceOutline:
    """Returns a scanned zero-knowledge WorkspaceOutline from clean_workspace."""
    scanner = WorkspaceScanner(clean_workspace)
    return scanner.scan()


@pytest.fixture
def leaked_outline(tmp_path: Path) -> WorkspaceOutline:
    """Returns a WorkspaceOutline containing forbidden implementation logic/secrets."""
    return WorkspaceOutline(
        workspace_root=tmp_path,
        file_tree=["src/auth.py"],
        ast_signatures={
            "src/auth.py": (
                "def authenticate(key: str):\n"
                "    secret = 'super_secret_production_key_12345'\n"
                "    if key == secret:\n"
                "        return True\n"
            )
        },
        summary_metadata={"total_files": 1, "total_lines": 5, "languages": {"Python": 1}},
    )


# ==============================================================================
# CLASS 1: TestSubTaskModel (Feature F5: SubTask Data Model)
# ==============================================================================


class TestSubTaskModel:
    def test_subtask_creation_valid(self):
        """Verify SubTask instantiation with all required and optional attributes."""
        task = SubTask(
            id="task_1",
            title="Setup Harness",
            description="Create test harness",
            dependencies=["task_0"],
            affected_files=["tests/test_harness.py"],
            success_criteria="Harness created and runs",
            verification_command="pytest tests/test_harness.py",
            status="pending",
            retry_count=1,
        )
        assert task.id == "task_1"
        assert task.title == "Setup Harness"
        assert task.description == "Create test harness"
        assert task.dependencies == ["task_0"]
        assert task.affected_files == ["tests/test_harness.py"]
        assert task.success_criteria == "Harness created and runs"
        assert task.verification_command == "pytest tests/test_harness.py"
        assert task.status == "pending"
        assert task.retry_count == 1

    def test_subtask_defaults(self):
        """Verify SubTask instantiation with only required id applies correct defaults."""
        task = SubTask(id="task_default")
        assert task.id == "task_default"
        assert task.title == ""
        assert task.description == ""
        assert task.dependencies == []
        assert task.affected_files == []
        assert task.success_criteria == ""
        assert task.verification_command == ""
        assert task.status == "pending"
        assert task.retry_count == 0

    def test_subtask_to_dict_structure(self):
        """Verify SubTask.to_dict() produces exact dictionary schema with correct types."""
        task = SubTask(
            id="t1",
            title="Task 1",
            dependencies=["t0"],
            affected_files=["main.py"],
        )
        d = task.to_dict()
        expected_keys = {
            "id",
            "title",
            "description",
            "dependencies",
            "affected_files",
            "input_files",
            "success_criteria",
            "verification_command",
            "outputs",
            "status",
            "retry_count",
        }
        assert set(d.keys()) == expected_keys
        assert d["outputs"] == []
        assert isinstance(d["dependencies"], list)
        assert isinstance(d["affected_files"], list)
        assert d["id"] == "t1"
        assert d["dependencies"] == ["t0"]

    def test_subtask_from_dict_complete(self):
        """Verify SubTask.from_dict() reconstructs instance from a complete dictionary."""
        data = {
            "id": "t_complete",
            "title": "Complete Task",
            "description": "Perform task",
            "dependencies": ["dep_1", "dep_2"],
            "affected_files": ["src/app.py"],
            "success_criteria": "Done",
            "verification_command": "pytest",
            "status": "in_progress",
            "retry_count": 2,
        }
        task = SubTask.from_dict(data)
        assert task.id == "t_complete"
        assert task.title == "Complete Task"
        assert task.dependencies == ["dep_1", "dep_2"]
        assert task.affected_files == ["src/app.py"]
        assert task.status == "in_progress"
        assert task.retry_count == 2

    def test_subtask_from_dict_missing_optionals(self):
        """Verify SubTask.from_dict() handles missing optional fields by inserting safe defaults."""
        data = {"id": "minimal_id"}
        task = SubTask.from_dict(data)
        assert task.id == "minimal_id"
        assert task.title == ""
        assert task.dependencies == []
        assert task.affected_files == []
        assert task.status == "pending"
        assert task.retry_count == 0

    def test_subtask_serialization_roundtrip(self):
        """Verify SubTask roundtrip: from_dict(task.to_dict()) produces identical object."""
        original = SubTask(
            id="roundtrip_task",
            title="Roundtrip Test",
            description="Testing serialization fidelity",
            dependencies=["dep_a", "dep_b"],
            affected_files=["file1.py", "file2.py"],
            success_criteria="Fidelity 100%",
            verification_command="python -m unittest",
            status="completed",
            retry_count=3,
        )
        reconstructed = SubTask.from_dict(original.to_dict())
        assert original == reconstructed

    def test_subtask_empty_id_raises_value_error(self):
        """Verify SubTask with empty or whitespace-only id raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            SubTask(id="")
        with pytest.raises(ValueError, match="non-empty string"):
            SubTask(id="   \t\n")

    def test_subtask_invalid_status_raises_value_error(self):
        """Verify SubTask with an invalid status raises ValueError."""
        with pytest.raises(ValueError, match="Invalid status"):
            SubTask(id="t1", status="unknown_status_xyz")

    def test_subtask_negative_retry_count_raises_value_error(self):
        """Verify SubTask with negative retry count raises ValueError."""
        with pytest.raises(ValueError, match="retry_count cannot be negative"):
            SubTask(id="t1", retry_count=-1)

    def test_subtask_status_normalization_aliases(self):
        """Verify SubTask normalizes common status aliases and casing."""
        assert SubTask(id="t1", status="PENDING").status == "pending"
        assert SubTask(id="t2", status="running").status == "in_progress"
        assert SubTask(id="t3", status="done").status == "completed"
        assert SubTask(id="t4", status="passed").status == "completed"
        assert SubTask(id="t5", status="fail").status == "failed"
        assert SubTask(id="t6", status="error").status == "failed"

    def test_subtask_dependencies_and_files_deduplication(self):
        """Verify duplicate dependencies and affected_files are deduplicated preserving order."""
        task = SubTask(
            id="t1",
            dependencies=["a", "b", "a", "c", "b"],
            affected_files=["f1.py", "f2.py", "f1.py"],
        )
        assert task.dependencies == ["a", "b", "c"]
        assert task.affected_files == ["f1.py", "f2.py"]


# ==============================================================================
# CLASS 2: TestExecutionPlanDAGCore (Feature F5: DAG Structure & Kahn's Sort)
# ==============================================================================


class TestExecutionPlanDAGCore:
    def test_dag_empty_execution_order(self):
        """Verify empty ExecutionPlanDAG returns empty execution order without errors."""
        dag = ExecutionPlanDAG()
        assert dag.get_execution_order() == []
        assert len(dag) == 0
        assert dag.validate() is True

    def test_dag_single_task_execution_order(self):
        """Verify 1-node ExecutionPlanDAG returns single task in order."""
        dag = ExecutionPlanDAG()
        task = SubTask(id="task_single", title="Single Task")
        dag.add_task(task)
        order = dag.get_execution_order()
        assert len(order) == 1
        assert order[0].id == "task_single"
        assert len(dag) == 1

    def test_dag_linear_chain_execution_order(self):
        """Verify strict sequential dependency chain A -> B -> C -> D."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="D", dependencies=["C"]))
        dag.add_task(SubTask(id="B", dependencies=["A"]))
        dag.add_task(SubTask(id="A", dependencies=[]))
        dag.add_task(SubTask(id="C", dependencies=["B"]))

        order = dag.get_execution_order()
        assert [t.id for t in order] == ["A", "B", "C", "D"]

    def test_dag_diamond_execution_order(self):
        """Verify diamond dependency graph A -> (B, C) -> D enforces topological invariants."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="A", dependencies=[]))
        dag.add_task(SubTask(id="B", dependencies=["A"]))
        dag.add_task(SubTask(id="C", dependencies=["A"]))
        dag.add_task(SubTask(id="D", dependencies=["B", "C"]))

        order = dag.get_execution_order()
        assert len(order) == 4
        assert order[0].id == "A"
        assert order[-1].id == "D"
        middle_ids = {order[1].id, order[2].id}
        assert middle_ids == {"B", "C"}
        # Both B and C must appear before D
        indices = {t.id: idx for idx, t in enumerate(order)}
        assert indices["A"] < indices["B"] < indices["D"]
        assert indices["A"] < indices["C"] < indices["D"]

    def test_dag_independent_tasks_deterministic_order(self):
        """Verify independent tasks with no dependencies are sorted alphabetically by ID."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="task_z", dependencies=[]))
        dag.add_task(SubTask(id="task_a", dependencies=[]))
        dag.add_task(SubTask(id="task_m", dependencies=[]))

        order = dag.get_execution_order()
        assert [t.id for t in order] == ["task_a", "task_m", "task_z"]

    def test_dag_disconnected_components(self):
        """Verify disconnected graph with multiple disjoint components sorts all tasks validly."""
        dag = ExecutionPlanDAG()
        # Component 1: A -> B
        dag.add_task(SubTask(id="A", dependencies=[]))
        dag.add_task(SubTask(id="B", dependencies=["A"]))
        # Component 2: C -> D
        dag.add_task(SubTask(id="C", dependencies=[]))
        dag.add_task(SubTask(id="D", dependencies=["C"]))

        order = dag.get_execution_order()
        assert len(order) == 4
        indices = {t.id: idx for idx, t in enumerate(order)}
        assert indices["A"] < indices["B"]
        assert indices["C"] < indices["D"]

    def test_dag_wide_fan_out_fan_in(self):
        """Verify wide fan-out / fan-in topology (1 Root -> 5 Workers -> 1 Sink)."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="root", dependencies=[]))
        for i in range(1, 6):
            dag.add_task(SubTask(id=f"worker_{i}", dependencies=["root"]))
        dag.add_task(SubTask(id="sink", dependencies=[f"worker_{i}" for i in range(1, 6)]))

        order = dag.get_execution_order()
        assert len(order) == 7
        assert order[0].id == "root"
        assert order[-1].id == "sink"
        worker_ids = {t.id for t in order[1:6]}
        assert worker_ids == {f"worker_{i}" for i in range(1, 6)}

    def test_dag_container_dunder_methods(self):
        """Verify __len__, __contains__, __getitem__, and __iter__ on ExecutionPlanDAG."""
        dag = ExecutionPlanDAG()
        t1 = SubTask(id="t1", dependencies=[])
        t2 = SubTask(id="t2", dependencies=["t1"])
        dag.add_task(t1)
        dag.add_task(t2)

        assert len(dag) == 2
        assert "t1" in dag
        assert "t2" in dag
        assert "nonexistent" not in dag
        assert dag["t1"] == t1
        assert dag["t2"] == t2
        assert list(dag) == dag.get_execution_order()

    def test_dag_add_task_success_and_duplicate(self):
        """Verify add_task adds new task, but raises ValueError on duplicate ID."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="t1", title="Original"))
        assert dag.get_task("t1").title == "Original"

        with pytest.raises(ValueError, match="Duplicate task id"):
            dag.add_task(SubTask(id="t1", title="Duplicate"))

        # With replace=True, task is updated
        dag.add_task(SubTask(id="t1", title="Updated"), replace=True)
        assert dag.get_task("t1").title == "Updated"

    def test_dag_add_task_type_error(self):
        """Verify add_task raises TypeError if task is not a SubTask instance."""
        dag = ExecutionPlanDAG()
        with pytest.raises(TypeError, match="Expected SubTask instance"):
            dag.add_task({"id": "not_a_subtask"})  # type: ignore

    def test_dag_get_task_not_found_raises(self):
        """Verify get_task raises DAGTaskNotFoundError (and KeyError) on missing task ID."""
        dag = ExecutionPlanDAG()
        with pytest.raises(DAGTaskNotFoundError):
            dag.get_task("nonexistent")
        with pytest.raises(KeyError):
            _ = dag["nonexistent"]

    def test_dag_init_with_list_of_tasks(self):
        """Verify ExecutionPlanDAG initialized with list of tasks converts to dict cleanly."""
        tasks = [SubTask(id="t1"), SubTask(id="t2", dependencies=["t1"])]
        dag = ExecutionPlanDAG(tasks=tasks)
        assert len(dag) == 2
        assert "t1" in dag and "t2" in dag

    def test_dag_init_with_duplicate_list_raises(self):
        """Verify ExecutionPlanDAG initialized with duplicate task IDs in list raises ValueError."""
        tasks = [SubTask(id="t1"), SubTask(id="t1")]
        with pytest.raises(ValueError, match="Duplicate task id"):
            ExecutionPlanDAG(tasks=tasks)

    def test_dag_reset_status(self):
        """Verify reset_status sets all tasks back to pending with retry_count = 0."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="t1", status="completed", retry_count=2))
        dag.add_task(SubTask(id="t2", status="failed", retry_count=5))

        dag.reset_status()
        assert dag["t1"].status == "pending"
        assert dag["t1"].retry_count == 0
        assert dag["t2"].status == "pending"
        assert dag["t2"].retry_count == 0

    def test_dag_to_mermaid(self):
        """Verify to_mermaid produces valid Mermaid graph syntax."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="A", title="Start"))
        dag.add_task(SubTask(id="B", title="End", dependencies=["A"]))

        mermaid = dag.to_mermaid()
        assert "graph TD" in mermaid
        assert 'A["Start"]' in mermaid
        assert 'B["End"]' in mermaid
        assert "A --> B" in mermaid


# ==============================================================================
# CLASS 3: TestDAGReadinessAndState (Feature F5: Task Readiness)
# ==============================================================================


class TestDAGReadinessAndState:
    def test_ready_tasks_initial_empty_completed(self):
        """Verify get_ready_tasks with empty completed set returns all root tasks."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="A", dependencies=[]))
        dag.add_task(SubTask(id="B", dependencies=[]))
        dag.add_task(SubTask(id="C", dependencies=["A"]))

        ready = dag.get_ready_tasks(set())
        assert [t.id for t in ready] == ["A", "B"]

    def test_ready_tasks_incremental_linear_chain(self):
        """Verify step-by-step readiness transitions for linear chain A -> B -> C."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="A", dependencies=[]))
        dag.add_task(SubTask(id="B", dependencies=["A"]))
        dag.add_task(SubTask(id="C", dependencies=["B"]))

        # Step 1: Nothing completed
        assert [t.id for t in dag.get_ready_tasks(set())] == ["A"]

        # Step 2: A completed
        assert [t.id for t in dag.get_ready_tasks({"A"})] == ["B"]

        # Step 3: A and B completed
        assert [t.id for t in dag.get_ready_tasks({"A", "B"})] == ["C"]

        # Step 4: All completed
        assert dag.get_ready_tasks({"A", "B", "C"}) == []

    def test_ready_tasks_diamond_dependency_satisfaction(self):
        """Verify diamond DAG: D becomes ready only when BOTH B and C are completed."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="A", dependencies=[]))
        dag.add_task(SubTask(id="B", dependencies=["A"]))
        dag.add_task(SubTask(id="C", dependencies=["A"]))
        dag.add_task(SubTask(id="D", dependencies=["B", "C"]))

        # Step 1: A completed
        assert [t.id for t in dag.get_ready_tasks({"A"})] == ["B", "C"]

        # Step 2: Only B completed (D still blocked by C)
        assert [t.id for t in dag.get_ready_tasks({"A", "B"})] == ["C"]

        # Step 3: Only C completed (D still blocked by B)
        assert [t.id for t in dag.get_ready_tasks({"A", "C"})] == ["B"]

        # Step 4: Both B and C completed (D unblocked!)
        assert [t.id for t in dag.get_ready_tasks({"A", "B", "C"})] == ["D"]

    def test_ready_tasks_all_tasks_completed(self):
        """Verify get_ready_tasks returns empty list when all task IDs are completed."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="A"))
        dag.add_task(SubTask(id="B", dependencies=["A"]))
        assert dag.get_ready_tasks({"A", "B"}) == []

    def test_ready_tasks_none_argument_treated_as_empty_set(self):
        """Verify get_ready_tasks(None) evaluates pending tasks correctly."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="A", dependencies=[]))
        dag.add_task(SubTask(id="B", dependencies=["A"]))
        ready = dag.get_ready_tasks(None)
        assert [t.id for t in ready] == ["A"]

    def test_ready_tasks_with_completed_status_in_subtasks(self):
        """Verify get_ready_tasks(None) respects tasks marked completed in status."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="A", dependencies=[], status="completed"))
        dag.add_task(SubTask(id="B", dependencies=["A"], status="pending"))
        ready = dag.get_ready_tasks(None)
        assert [t.id for t in ready] == ["B"]

    def test_ready_tasks_extraneous_completed_ids_ignored(self):
        """Verify passing unknown/extraneous task IDs in completed set does not cause errors."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="A", dependencies=[]))
        ready = dag.get_ready_tasks({"ghost_task_1", "alien_task_2"})
        assert [t.id for t in ready] == ["A"]

    def test_ready_tasks_deterministic_order(self):
        """Verify get_ready_tasks returns tasks deterministically sorted by ID."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="z_task"))
        dag.add_task(SubTask(id="a_task"))
        dag.add_task(SubTask(id="m_task"))
        ready = dag.get_ready_tasks(set())
        assert [t.id for t in ready] == ["a_task", "m_task", "z_task"]


# ==============================================================================
# CLASS 4: TestDAGCycleAndDependencyErrors (Feature F5: Error Boundaries)
# ==============================================================================


class TestDAGCycleAndDependencyErrors:
    def test_dag_self_cycle_detection(self):
        """Verify task depending on itself raises DAGCycleError."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="A", dependencies=["A"]))
        with pytest.raises(DAGCycleError, match="self-cycle"):
            dag.validate()

    def test_dag_two_node_direct_cycle(self):
        """Verify 2-node cycle A -> B -> A raises DAGCycleError."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="A", dependencies=["B"]))
        dag.add_task(SubTask(id="B", dependencies=["A"]))
        with pytest.raises(DAGCycleError, match="Cycle detected"):
            dag.get_execution_order()

    def test_dag_three_node_indirect_cycle(self):
        """Verify 3-node cycle A -> B -> C -> A raises DAGCycleError."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="A", dependencies=["C"]))
        dag.add_task(SubTask(id="B", dependencies=["A"]))
        dag.add_task(SubTask(id="C", dependencies=["B"]))
        with pytest.raises(DAGCycleError, match="Cycle detected"):
            dag.validate()

    def test_dag_lollipop_cycle_with_acyclic_tail(self):
        """Verify cycle with valid predecessor tail (A -> B -> C -> D -> B) raises DAGCycleError."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="A", dependencies=[]))
        dag.add_task(SubTask(id="B", dependencies=["A", "D"]))
        dag.add_task(SubTask(id="C", dependencies=["B"]))
        dag.add_task(SubTask(id="D", dependencies=["C"]))

        with pytest.raises(DAGCycleError) as exc_info:
            dag.get_execution_order()
        msg = str(exc_info.value)
        assert "B" in msg and "C" in msg and "D" in msg

    def test_dag_missing_single_dependency(self):
        """Verify task declaring non-existent dependency ID raises DAGDependencyError."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="A", dependencies=["nonexistent_xyz"]))
        with pytest.raises(DAGDependencyError) as exc_info:
            dag.validate()
        assert "nonexistent_xyz" in str(exc_info.value)

    def test_dag_missing_multiple_dependencies(self):
        """Verify task declaring multiple non-existent dependency IDs raises DAGDependencyError."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="A", dependencies=["missing_1", "missing_2"]))
        with pytest.raises(DAGDependencyError) as exc_info:
            dag.validate()
        msg = str(exc_info.value)
        assert "missing_1" in msg and "missing_2" in msg

    def test_dag_cycle_in_disconnected_component(self):
        """Verify error raised when one component is acyclic but another component has a cycle."""
        dag = ExecutionPlanDAG()
        # Valid component
        dag.add_task(SubTask(id="A", dependencies=[]))
        dag.add_task(SubTask(id="B", dependencies=["A"]))
        # Cyclic component
        dag.add_task(SubTask(id="C", dependencies=["D"]))
        dag.add_task(SubTask(id="D", dependencies=["C"]))

        with pytest.raises(DAGCycleError):
            dag.get_execution_order()

    def test_dag_validate_method_valid_and_invalid(self):
        """Verify dag.validate() returns True on valid graph, raises on invalid."""
        valid_dag = ExecutionPlanDAG()
        valid_dag.add_task(SubTask(id="A"))
        valid_dag.add_task(SubTask(id="B", dependencies=["A"]))
        assert valid_dag.validate() is True

        missing_dag = ExecutionPlanDAG()
        missing_dag.add_task(SubTask(id="A", dependencies=["ghost"]))
        with pytest.raises(DAGDependencyError):
            missing_dag.validate()

        cyclic_dag = ExecutionPlanDAG()
        cyclic_dag.add_task(SubTask(id="A", dependencies=["B"]))
        cyclic_dag.add_task(SubTask(id="B", dependencies=["A"]))
        with pytest.raises(DAGCycleError):
            cyclic_dag.validate()


# ==============================================================================
# CLASS 5: TestDAGSerialization (Feature F5: JSON & Dict Serialization)
# ==============================================================================


class TestDAGSerialization:
    def test_dag_to_dict_and_from_dict_roundtrip(self):
        """Verify DAG serialization to dict and reconstruction preserves all tasks and metadata."""
        dag = ExecutionPlanDAG(metadata={"generator": "test_suite", "input_tokens": 120})
        dag.add_task(SubTask(id="t1", title="Task 1"))
        dag.add_task(SubTask(id="t2", title="Task 2", dependencies=["t1"]))

        d = dag.to_dict()
        reconstructed = ExecutionPlanDAG.from_dict(d)

        assert len(reconstructed) == len(dag)
        assert reconstructed.metadata == dag.metadata
        assert [t.id for t in reconstructed.get_execution_order()] == ["t1", "t2"]

    def test_dag_to_json_and_from_json_roundtrip(self):
        """Verify JSON string serialization and reconstruction."""
        dag = ExecutionPlanDAG(metadata={"version": "1.0"})
        dag.add_task(SubTask(id="task_a", verification_command="pytest"))
        dag.add_task(SubTask(id="task_b", dependencies=["task_a"]))

        json_str = dag.to_json()
        reconstructed = ExecutionPlanDAG.from_json(json_str)

        assert len(reconstructed) == 2
        assert reconstructed["task_a"].verification_command == "pytest"
        assert [t.id for t in reconstructed.get_execution_order()] == ["task_a", "task_b"]

    def test_dag_from_dict_accepts_list_of_tasks(self):
        """Verify ExecutionPlanDAG.from_dict parses {"tasks": [ {...}, {...} ]}."""
        payload = {
            "tasks": [
                {"id": "t1", "title": "Step 1"},
                {"id": "t2", "title": "Step 2", "dependencies": ["t1"]},
            ]
        }
        dag = ExecutionPlanDAG.from_dict(payload)
        assert len(dag) == 2
        assert dag.validate() is True

    def test_dag_from_dict_accepts_dict_of_tasks(self):
        """Verify ExecutionPlanDAG.from_dict parses {"tasks": {"t1": {...}, "t2": {...}}}."""
        payload = {
            "tasks": {
                "t1": {"title": "Step 1", "dependencies": []},
                "t2": {"title": "Step 2", "dependencies": ["t1"]},
            }
        }
        dag = ExecutionPlanDAG.from_dict(payload)
        assert len(dag) == 2
        assert [t.id for t in dag.get_execution_order()] == ["t1", "t2"]

    def test_dag_from_dict_accepts_bare_list(self):
        """Verify ExecutionPlanDAG.from_dict parses raw list of task dicts without wrapper."""
        task_list = [
            {"id": "a", "title": "Task A"},
            {"id": "b", "title": "Task B", "dependencies": ["a"]},
        ]
        dag = ExecutionPlanDAG.from_dict(task_list)
        assert len(dag) == 2
        assert dag.validate() is True

    def test_dag_from_json_with_markdown_fences(self):
        """Verify from_json parses JSON wrapped inside markdown code fences."""
        raw = "```json\n{\n  \"tasks\": [\n    {\"id\": \"t1\", \"title\": \"Fenced\"}\n  ]\n}\n```"
        dag = ExecutionPlanDAG.from_json(raw)
        assert len(dag) == 1
        assert dag["t1"].title == "Fenced"

    def test_dag_from_json_invalid_syntax_raises_dag_error(self):
        """Verify invalid JSON syntax raises DAGError."""
        with pytest.raises(DAGError, match="Failed to parse DAG JSON"):
            ExecutionPlanDAG.from_json("definitely not valid json {")

    def test_dag_serialization_empty_graph(self):
        """Verify empty DAG serializes and deserializes cleanly."""
        dag = ExecutionPlanDAG()
        reconstructed = ExecutionPlanDAG.from_json(dag.to_json())
        assert len(reconstructed) == 0


# ==============================================================================
# CLASS 6: TestMockCloudArchitectClient (Feature F7: Mock Planner)
# ==============================================================================


class TestMockCloudArchitectClient:
    def test_mock_planner_algorithm_goal(self):
        """Verify MockCloudArchitectClient returns 3-task algorithm plan for 'sieve'/'algorithm'."""
        client = MockCloudArchitectClient()
        dag = client.plan_rd_task(goal="Implement Sieve of Eratosthenes algorithm")
        assert len(dag) == 3
        order = dag.get_execution_order()
        assert [t.id for t in order] == ["task_1_scaffold", "task_2_implement", "task_3_benchmark"]
        assert "tests/test_algorithm.py" in dag["task_1_scaffold"].affected_files
        assert dag.validate() is True

    def test_mock_planner_bugfix_goal(self):
        """Verify MockCloudArchitectClient returns 2-task plan for 'bug'/'fix'."""
        client = MockCloudArchitectClient()
        dag = client.plan_rd_task(goal="Fix bug in balance withdrawal race condition")
        assert len(dag) == 2
        order = dag.get_execution_order()
        assert [t.id for t in order] == ["task_1_reproduce", "task_2_patch"]
        assert dag.validate() is True

    def test_mock_planner_refactor_goal(self):
        """Verify MockCloudArchitectClient returns 4-task diamond plan for 'refactor'."""
        client = MockCloudArchitectClient()
        dag = client.plan_rd_task(goal="Refactor authentication and database interfaces")
        assert len(dag) == 4
        assert dag.validate() is True
        assert set(dag["task_3_integration"].dependencies) == {"task_2a_migrate_auth", "task_2b_migrate_db"}

    def test_mock_planner_diamond_goal(self):
        """Verify MockCloudArchitectClient returns diamond plan when 'diamond' is in goal."""
        client = MockCloudArchitectClient()
        dag = client.plan_rd_task(goal="Run diamond task pipeline")
        assert len(dag) == 4
        assert dag.validate() is True
        assert set(dag["task_3"].dependencies) == {"task_2a", "task_2b"}

    def test_mock_planner_parallel_goal(self):
        """Verify MockCloudArchitectClient returns independent tasks when 'parallel' in goal."""
        client = MockCloudArchitectClient()
        dag = client.plan_rd_task(goal="Execute parallel benchmark tasks")
        assert len(dag) == 2
        assert dag["task_1"].dependencies == []
        assert dag["task_2"].dependencies == []
        assert dag.validate() is True

    def test_mock_planner_single_goal(self):
        """Verify MockCloudArchitectClient returns 1-task plan when 'single' in goal."""
        client = MockCloudArchitectClient()
        dag = client.plan_rd_task(goal="Execute single quick task")
        assert len(dag) == 1
        assert dag.validate() is True

    def test_mock_planner_default_generic_goal(self):
        """Verify MockCloudArchitectClient returns generic 3-task plan for arbitrary goals."""
        client = MockCloudArchitectClient()
        dag = client.plan_rd_task(goal="Build a custom markdown parser")
        assert len(dag) == 3
        assert dag.validate() is True
        assert [t.id for t in dag.get_execution_order()] == ["task_1", "task_2", "task_3"]

    def test_mock_planner_empty_goal_raises_value_error(self):
        """Verify MockCloudArchitectClient raises ValueError on empty goal."""
        client = MockCloudArchitectClient()
        with pytest.raises(ValueError, match="cannot be empty"):
            client.plan_rd_task(goal="")
        with pytest.raises(ValueError, match="cannot be empty"):
            client.plan_rd_task(goal="   \t\n")

    def test_mock_planner_none_goal_raises_value_error(self):
        """Verify MockCloudArchitectClient raises ValueError if goal is None."""
        client = MockCloudArchitectClient()
        with pytest.raises(ValueError, match="cannot be empty"):
            client.plan_rd_task(goal=None)  # type: ignore

    def test_mock_planner_preflight_clean_outline_passes(self, clean_outline: WorkspaceOutline):
        """Verify MockCloudArchitectClient succeeds when valid clean WorkspaceOutline provided."""
        client = MockCloudArchitectClient()
        dag = client.plan_rd_task(goal="Implement prime sieve", workspace_outline=clean_outline)
        assert len(dag) == 3
        assert dag.validate() is True

    def test_mock_planner_preflight_catches_privacy_leak(self, leaked_outline: WorkspaceOutline):
        """Verify MockCloudArchitectClient enforces pre-flight privacy check and raises error on leak."""
        client = MockCloudArchitectClient()
        with pytest.raises(SecurityInvariantViolationError):
            client.plan_rd_task(goal="Implement feature", workspace_outline=leaked_outline)

    def test_mock_planner_token_accounting(self):
        """Verify MockCloudArchitectClient populates input and output tokens in DAG metadata."""
        client = MockCloudArchitectClient()
        dag = client.plan_rd_task(goal="Run benchmark pipeline")
        assert dag.metadata.get("input_tokens", 0) > 0
        assert dag.metadata.get("output_tokens", 0) > 0

    def test_mock_planner_inject_cycle(self):
        """Verify inject_cycle=True returns a cyclic DAG causing DAGCycleError upon validation."""
        client = MockCloudArchitectClient(inject_cycle=True)
        dag = client.plan_rd_task(goal="Arbitrary task")
        with pytest.raises(DAGCycleError):
            dag.validate()

    def test_mock_planner_inject_missing_dependency(self):
        """Verify inject_missing_dependency=True returns DAG with missing prerequisite."""
        client = MockCloudArchitectClient(inject_missing_dependency=True)
        dag = client.plan_rd_task(goal="Arbitrary task")
        with pytest.raises(DAGDependencyError):
            dag.validate()

    def test_mock_planner_inject_error(self):
        """Verify inject_error raises the specified exception directly."""
        client = MockCloudArchitectClient(inject_error=RuntimeError("Simulated API failure"))
        with pytest.raises(RuntimeError, match="Simulated API failure"):
            client.plan_rd_task(goal="Arbitrary task")

    def test_mock_planner_custom_plan(self):
        """Verify custom_plan override returns the exact specified DAG."""
        custom_dag = ExecutionPlanDAG()
        custom_dag.add_task(SubTask(id="custom_task_1"))
        client = MockCloudArchitectClient(custom_plan=custom_dag)
        dag = client.plan_rd_task(goal="Arbitrary task")
        assert dag is custom_dag
        assert "custom_task_1" in dag

    def test_mock_planner_backward_compatibility_strategies(self):
        """Verify legacy generate_strategy produces valid StrategyResult for security modules."""
        client = MockCloudArchitectClient()
        for mod in ["auth.py", "billing.py", "database.py"]:
            result = client.generate_strategy(stripped_code="", module_name=mod)
            assert result.threat_model != ""
            assert len(result.invariants) > 0

    def test_mock_planner_generate_strategic_plan_alias(self):
        """Verify generate_strategic_plan is an identical alias for generate_strategy."""
        client = MockCloudArchitectClient()
        res1 = client.generate_strategy(stripped_code="", module_name="auth.py")
        res2 = client.generate_strategic_plan(stripped_code="", module_name="auth.py")
        assert res1 == res2


# ==============================================================================
# CLASS 7: TestCloudArchitectClientPlanning (Feature F6: Cloud Planner)
# ==============================================================================


class TestCloudArchitectClientPlanning:
    def test_cloud_planner_prompt_construction(self, clean_outline: WorkspaceOutline):
        """Verify _build_rd_prompt embeds goal and outline headers without privacy leak."""
        client = CloudArchitectClient()
        prompt = client._build_rd_prompt(
            goal="Add JWT authentication", workspace_outline=clean_outline
        )
        assert "Add JWT authentication" in prompt
        assert "WORKSPACE ARCHITECTURE OUTLINE" in prompt
        assert "tasks" in prompt
        assert "verification_command" in prompt

    def test_cloud_planner_preflight_privacy_gate_aborts_before_subprocess(
        self, leaked_outline: WorkspaceOutline
    ):
        """Verify plan_rd_task aborts and NEVER calls codex subprocess if outline fails privacy check."""
        client = CloudArchitectClient()
        with patch("subprocess.run") as mock_run:
            with pytest.raises(SecurityInvariantViolationError):
                client.plan_rd_task(goal="Plan task", workspace_outline=leaked_outline)
            mock_run.assert_not_called()

    def test_cloud_planner_empty_goal_validation(self):
        """Verify CloudArchitectClient raises ValueError on empty goal before subprocess."""
        client = CloudArchitectClient()
        with pytest.raises(ValueError, match="cannot be empty"):
            client.plan_rd_task(goal="")

    def test_cloud_planner_subprocess_success_parsing(self):
        """Verify CloudArchitectClient parses valid codex output into ExecutionPlanDAG."""
        client = CloudArchitectClient()
        mock_payload = {
            "response": json.dumps(
                {
                    "tasks": [
                        {
                            "id": "t1",
                            "title": "Setup",
                            "dependencies": [],
                            "verification_command": "pytest",
                        },
                        {
                            "id": "t2",
                            "title": "Implement",
                            "dependencies": ["t1"],
                            "verification_command": "pytest",
                        },
                    ]
                }
            ),
            "usage": {"input_tokens": 110, "output_tokens": 60},
        }

        with patch.object(client, "_invoke_codex", return_value=(mock_payload["response"], 110, 60)):
            dag = client.plan_rd_task(goal="Implement feature")
            assert len(dag) == 2
            assert [t.id for t in dag.get_execution_order()] == ["t1", "t2"]
            assert dag.metadata["input_tokens"] == 110
            assert dag.metadata["output_tokens"] == 60

    def test_cloud_planner_markdown_code_fences_handled(self):
        """Verify CloudArchitectClient extracts JSON wrapped inside ```json markdown fences."""
        client = CloudArchitectClient()
        raw_response = (
            "Here is your execution plan:\n"
            "```json\n"
            "{\n"
            '  "tasks": [\n'
            '    {"id": "task_1", "title": "Fenced Step", "dependencies": []}\n'
            "  ]\n"
            "}\n"
            "```\n"
            "Hope this helps!"
        )

        with patch.object(client, "_invoke_codex", return_value=(raw_response, 100, 50)):
            dag = client.plan_rd_task(goal="Quick task")
            assert len(dag) == 1
            assert dag["task_1"].title == "Fenced Step"

    def test_cloud_planner_subprocess_timeout(self):
        """Verify CloudArchitectClient raises CloudArchitectError on subprocess timeout."""
        client = CloudArchitectClient(timeout=0.01)
        with patch(
            "subprocess.Popen",
            side_effect=lambda cmd, **kwargs: _REAL_POPEN(
                [sys.executable, "-c", "import time; time.sleep(1)"], **kwargs
            ),
        ):
            with pytest.raises(CloudArchitectError, match="timed out"):
                client.plan_rd_task(goal="Long task", max_retries=0)

    def test_cloud_planner_subprocess_nonzero_exit(self):
        """Verify CloudArchitectClient raises CloudArchitectError when codex exits with non-zero code."""
        client = CloudArchitectClient()
        with patch(
            "subprocess.Popen",
            side_effect=lambda cmd, **kwargs: _REAL_POPEN(
                [sys.executable, "-c", "import sys; print('API quota exceeded', file=sys.stderr); sys.exit(1)"],
                **kwargs,
            ),
        ):
            with pytest.raises(CloudArchitectError, match="API quota exceeded"):
                client.plan_rd_task(goal="Task", max_retries=0)

    def test_cloud_planner_malformed_json_response(self):
        """Verify CloudArchitectClient raises CloudArchitectError on corrupted JSON payload."""
        client = CloudArchitectClient()
        with patch.object(client, "_invoke_codex", return_value=("not_json_at_all", 50, 10)):
            with pytest.raises(CloudArchitectError, match="Failed to generate valid ExecutionPlanDAG"):
                client.plan_rd_task(goal="Task", max_retries=0)

    def test_cloud_planner_retry_on_malformed_json_then_success(self):
        """Verify CloudArchitectClient retries on malformed JSON and succeeds on subsequent attempt."""
        client = CloudArchitectClient()
        valid_response = json.dumps({"tasks": [{"id": "t1", "title": "Recovered"}]})

        # Attempt 0: invalid, Attempt 1: valid
        mock_invoke = MagicMock(
            side_effect=[
                ("malformed json {", 100, 10),
                (valid_response, 120, 40),
            ]
        )
        with patch.object(client, "_invoke_codex", mock_invoke):
            dag = client.plan_rd_task(goal="Task", max_retries=2)
            assert len(dag) == 1
            assert dag["t1"].title == "Recovered"
            assert mock_invoke.call_count == 2
            assert dag.metadata["input_tokens"] == 220
            assert dag.metadata["output_tokens"] == 50

    def test_cloud_planner_exhausted_retries_raises_cloud_architect_error(self):
        """Verify CloudArchitectError is raised after max_retries attempts fail."""
        client = CloudArchitectClient()
        mock_invoke = MagicMock(return_value=("corrupted {", 100, 10))
        with patch.object(client, "_invoke_codex", mock_invoke):
            with pytest.raises(CloudArchitectError, match="Failed to generate valid ExecutionPlanDAG"):
                client.plan_rd_task(goal="Task", max_retries=2)
            assert mock_invoke.call_count == 3  # Initial + 2 retries

    def test_cloud_planner_cycle_in_cloud_response(self):
        """Verify CloudArchitectClient raises DAGCycleError if Cloud Architect returned cyclic plan."""
        client = CloudArchitectClient()
        cyclic_response = json.dumps(
            {
                "tasks": [
                    {"id": "t1", "dependencies": ["t2"]},
                    {"id": "t2", "dependencies": ["t1"]},
                ]
            }
        )
        with patch.object(client, "_invoke_codex", return_value=(cyclic_response, 100, 50)):
            with pytest.raises(DAGCycleError):
                client.plan_rd_task(goal="Cyclic Task", max_retries=0)

    def test_cloud_planner_empty_tasks_in_cloud_response(self):
        """Verify CloudArchitectClient raises CloudArchitectError if plan contains 0 tasks."""
        client = CloudArchitectClient()
        empty_response = json.dumps({"tasks": []})
        with patch.object(client, "_invoke_codex", return_value=(empty_response, 50, 10)):
            with pytest.raises(CloudArchitectError, match="empty plan"):
                client.plan_rd_task(goal="Empty Task", max_retries=0)

    def test_cloud_planner_backward_compatibility_generate_strategic_plan_alias(self):
        """Verify generate_strategic_plan is alias for generate_strategy on CloudArchitectClient."""
        client = CloudArchitectClient()
        assert client.generate_strategic_plan == client.generate_strategy


# ==============================================================================
# CLASS 8: TestCrossFeatureIntegration (Tier 3: E2E Pipeline)
# ==============================================================================


class TestCrossFeatureIntegration:
    def test_e2e_scanner_to_mock_planner_to_dag_execution(self, clean_workspace: Path):
        """Tier 3: Scan workspace -> verify privacy -> mock plan task -> verify DAG topological order."""
        # Step 1: Scan workspace
        scanner = WorkspaceScanner(clean_workspace)
        outline = scanner.scan()
        assert len(outline.file_tree) > 0

        # Step 2: Privacy invariant check on outline payload
        outline_str = outline.to_prompt_string()
        assert verify_cloud_payload(outline_str) is True

        # Step 3: Planner decomposes task
        client = MockCloudArchitectClient()
        dag = client.plan_rd_task(
            goal="Implement prime sieve algorithm with benchmark",
            workspace_outline=outline,
        )

        # Step 4: Validate DAG
        assert dag.validate() is True
        order = dag.get_execution_order()
        assert len(order) == 3
        assert order[0].id == "task_1_scaffold"
        assert order[1].id == "task_2_implement"
        assert order[2].id == "task_3_benchmark"

    def test_e2e_pipeline_readiness_simulation(self):
        """Tier 3: Full simulation of task execution loop using get_ready_tasks until completion."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="setup", dependencies=[]))
        dag.add_task(SubTask(id="build_core", dependencies=["setup"]))
        dag.add_task(SubTask(id="build_api", dependencies=["setup"]))
        dag.add_task(SubTask(id="integration_test", dependencies=["build_core", "build_api"]))

        completed = set()
        execution_trace = []

        while len(completed) < len(dag):
            ready = dag.get_ready_tasks(completed)
            assert len(ready) > 0, "Deadlock in execution pipeline!"
            # Pick first available task (deterministic order)
            curr = ready[0]
            curr.status = "in_progress"
            # Simulate work and verification success
            curr.status = "completed"
            completed.add(curr.id)
            execution_trace.append(curr.id)

        assert completed == {"setup", "build_core", "build_api", "integration_test"}
        assert execution_trace[0] == "setup"
        assert execution_trace[-1] == "integration_test"
        assert dag.get_ready_tasks(completed) == []

    def test_e2e_pipeline_adversarial_workspace_privacy_blocking(self, tmp_path: Path):
        """Tier 3: Workspace with secret code leaks is stopped at pre-flight gate before planning."""
        ws = tmp_path / "leaky_repo"
        ws.mkdir()
        (ws / "secret_file.py").write_text(
            "SECRET_TOKEN = 'sk_live_999998888877777'\n"
            "def leak():\n"
            "    return SECRET_TOKEN\n",
            encoding="utf-8",
        )

        # Create outline that leaks function body
        bad_outline = WorkspaceOutline(
            workspace_root=ws,
            file_tree=["secret_file.py"],
            ast_signatures={
                "secret_file.py": (
                    "def leak():\n"
                    "    if True:\n"
                    "        return 'sk_live_999998888877777'\n"
                )
            },
            summary_metadata={"total_files": 1, "total_lines": 4, "languages": {"Python": 1}},
        )

        client = MockCloudArchitectClient()
        with pytest.raises(SecurityInvariantViolationError):
            client.plan_rd_task(goal="Audit repo", workspace_outline=bad_outline)

    def test_e2e_pipeline_empty_workspace(self, tmp_path: Path):
        """Tier 3: Scanner on empty directory -> valid outline -> planner generates valid plan."""
        empty_dir = tmp_path / "empty_repo"
        empty_dir.mkdir()

        scanner = WorkspaceScanner(empty_dir)
        outline = scanner.scan()
        assert len(outline.file_tree) == 0

        client = MockCloudArchitectClient()
        dag = client.plan_rd_task(
            goal="Scaffold project from scratch", workspace_outline=outline
        )
        assert len(dag) > 0
        assert dag.validate() is True

    def test_e2e_pipeline_custom_goal_workspace(self, clean_workspace: Path):
        """Tier 3: Multi-file project with bugfix goal generates verified 2-step bugfix DAG."""
        scanner = WorkspaceScanner(clean_workspace)
        outline = scanner.scan()

        client = MockCloudArchitectClient()
        dag = client.plan_rd_task(
            goal="Fix bug in math_helper logic", workspace_outline=outline
        )
        assert [t.id for t in dag.get_execution_order()] == ["task_1_reproduce", "task_2_patch"]
        assert dag.validate() is True
