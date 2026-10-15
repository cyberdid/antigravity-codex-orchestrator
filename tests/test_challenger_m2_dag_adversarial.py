"""Adversarial stress test suite for Milestone 2: Execution DAG & Topological Sort.

Authored by Challenger 1 (teamwork_preview_challenger).
Empirically challenges:
1. Self-cycles (A -> A), single-node, multi-node, and with whitespace.
2. Direct 2-node cycles (A <-> B) with downstream dependents and independent subgraphs.
3. Complex 10+ node cycles with multiple ingress entry paths, chords, and exit paths.
4. Disjoint graphs containing valid subgraphs and a cyclic component (isolating cyclic_ids).
5. Missing dependency references raising DAGDependencyError with full attribution.
6. Deep linear chains (200+ nodes), multi-level diamond lattices, and wide fan-out (1,000+ nodes).
7. get_ready_tasks() under partial completions, status progressions, and dynamic updates.
8. Serialization roundtrips with arbitrary complex nested metadata and messy LLM JSON.
9. CloudArchitect & MockCloudArchitect planner behavior under adversarial inputs and privacy gates.
"""

import json
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Set
from unittest.mock import MagicMock, patch

import pytest

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
)
from antigravity_hybrid_orchestrator.workspace.scanner import WorkspaceOutline


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
# SECTION 1: PATHOLOGICAL CYCLES (SELF-CYCLES, 2-NODE, 10+ NODE, DISJOINT)
# ==============================================================================


class TestAdversarialCycleDetection:
    """Adversarially challenge cycle detection across pathological graph topologies."""

    def test_isolated_single_node_self_cycle(self):
        """Stress: A single task that references its own ID must raise DAGCycleError."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="ouroboros", dependencies=["ouroboros"]))

        with pytest.raises(DAGCycleError) as exc_info:
            dag.validate()
        assert "ouroboros" in str(exc_info.value)
        assert "self-cycle" in str(exc_info.value).lower()

        with pytest.raises(DAGCycleError):
            dag.get_execution_order()

    def test_self_cycle_with_whitespace_and_duplicates(self):
        """Stress: Self-cycle with messy whitespace and duplicate entries in dependencies."""
        dag = ExecutionPlanDAG()
        # SubTask strips dependencies, so " ouroboros " becomes "ouroboros"
        task = SubTask(
            id="ouroboros",
            dependencies=[" ouroboros ", "ouroboros", "\touroboros\n"],
        )
        assert task.dependencies == ["ouroboros"]
        dag.add_task(task)

        with pytest.raises(DAGCycleError):
            dag.validate()

    def test_self_cycle_embedded_in_valid_linear_chain(self):
        """Stress: Valid linear chain where one intermediate node has a self-cycle."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="start", dependencies=[]))
        dag.add_task(SubTask(id="step_2", dependencies=["start", "step_2"]))
        dag.add_task(SubTask(id="end", dependencies=["step_2"]))

        with pytest.raises(DAGCycleError) as exc_info:
            dag.validate()
        assert "step_2" in str(exc_info.value)

    def test_direct_two_node_cycle_with_independent_and_downstream_nodes(self):
        """Stress: Direct 2-node cycle A <-> B with independent node C and downstream node D."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="indep_C", dependencies=[]))
        dag.add_task(SubTask(id="node_A", dependencies=["node_B"]))
        dag.add_task(SubTask(id="node_B", dependencies=["node_A"]))
        dag.add_task(SubTask(id="downstream_D", dependencies=["node_A"]))

        with pytest.raises(DAGCycleError) as exc_info:
            dag.get_execution_order()

        err_msg = str(exc_info.value)
        # Kahn's sort must isolate stuck nodes: node_A, node_B, downstream_D
        assert "node_A" in err_msg
        assert "node_B" in err_msg
        assert "downstream_D" in err_msg
        # indep_C must NOT be in the cyclic_ids error since its in-degree reached 0
        assert "indep_C" not in err_msg

    def test_complex_ten_plus_node_cycle_with_multiple_entries_chords_and_exits(self):
        """Stress: 20-node graph with 3 ingress nodes, 10-node cycle with internal chords, and 2 exit nodes."""
        dag = ExecutionPlanDAG()

        # Ingress nodes (acyclic entry points)
        dag.add_task(SubTask(id="ingress_1"))
        dag.add_task(SubTask(id="ingress_2"))
        dag.add_task(SubTask(id="ingress_3"))

        # Cycle nodes: c_1 -> c_2 -> ... -> c_10 -> c_1
        for i in range(1, 11):
            next_i = (i % 10) + 1
            deps = [f"c_{i-1}"] if i > 1 else ["c_10"]
            if i == 1:
                deps.extend(["ingress_1", "ingress_2"])
            elif i == 3:
                deps.append("ingress_3")
            dag.add_task(SubTask(id=f"c_{i}", dependencies=deps))

        # Add cross-chords inside cycle
        # c_2 also depends on c_7 (reverse chord)
        dag.tasks["c_2"].dependencies.append("c_7")
        # c_5 also depends on c_9 (reverse chord)
        dag.tasks["c_5"].dependencies.append("c_9")

        # Exit nodes (dependent on cycle nodes)
        dag.add_task(SubTask(id="exit_1", dependencies=["c_4"]))
        dag.add_task(SubTask(id="exit_2", dependencies=["c_8"]))
        dag.add_task(SubTask(id="final_sink", dependencies=["exit_1", "exit_2"]))

        assert len(dag) == 16

        with pytest.raises(DAGCycleError) as exc_info:
            dag.get_execution_order()

        err = str(exc_info.value)
        # All cycle nodes and downstream exits must be flagged as stuck
        for i in range(1, 11):
            assert f"c_{i}" in err
        assert "exit_1" in err
        assert "exit_2" in err
        assert "final_sink" in err
        # Ingress nodes must NOT be in the stuck set
        assert "ingress_1" not in err
        assert "ingress_2" not in err
        assert "ingress_3" not in err

    def test_disjoint_components_strictly_isolates_cyclic_subgraph(self):
        """Stress: Multiple disjoint components where only one is cyclic.

        Component 1: Linear chain v1 -> v2 -> v3 (valid)
        Component 2: Diamond d1 -> (d2a, d2b) -> d3 (valid)
        Component 3: Cycle c1 -> c2 -> c3 -> c1 (invalid)
        Component 4: Isolated single task iso_1 (valid)
        """
        dag = ExecutionPlanDAG()

        # Component 1
        dag.add_task(SubTask(id="v1"))
        dag.add_task(SubTask(id="v2", dependencies=["v1"]))
        dag.add_task(SubTask(id="v3", dependencies=["v2"]))

        # Component 2
        dag.add_task(SubTask(id="d1"))
        dag.add_task(SubTask(id="d2a", dependencies=["d1"]))
        dag.add_task(SubTask(id="d2b", dependencies=["d1"]))
        dag.add_task(SubTask(id="d3", dependencies=["d2a", "d2b"]))

        # Component 3 (cyclic)
        dag.add_task(SubTask(id="c1", dependencies=["c3"]))
        dag.add_task(SubTask(id="c2", dependencies=["c1"]))
        dag.add_task(SubTask(id="c3", dependencies=["c2"]))

        # Component 4
        dag.add_task(SubTask(id="iso_1"))

        assert len(dag) == 11

        with pytest.raises(DAGCycleError) as exc_info:
            dag.get_execution_order()

        err = str(exc_info.value)
        # Cyclic tasks MUST be in error
        assert "c1" in err and "c2" in err and "c3" in err

        # Valid tasks MUST NOT be in error
        for valid_id in ["v1", "v2", "v3", "d1", "d2a", "d2b", "d3", "iso_1"]:
            assert valid_id not in err


# ==============================================================================
# SECTION 2: ADVERSARIAL DEPENDENCY VALIDATION (MISSING, GHOST, MULTI-ERROR)
# ==============================================================================


class TestAdversarialDependencyValidation:
    """Stress test missing dependencies, ghost references, and priority over cycles."""

    def test_multiple_missing_dependencies_aggregated_in_error(self):
        """Stress: Multiple tasks referencing different missing dependencies are all reported."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="t1", dependencies=["missing_alpha", "missing_beta"]))
        dag.add_task(SubTask(id="t2", dependencies=["missing_gamma"]))
        dag.add_task(SubTask(id="t3", dependencies=["t1"]))  # t1 exists, but missing_alpha doesn't

        with pytest.raises(DAGDependencyError) as exc_info:
            dag.validate()

        msg = str(exc_info.value)
        assert "missing_alpha" in msg
        assert "missing_beta" in msg
        assert "missing_gamma" in msg

    def test_missing_dependency_detected_before_topological_sort_cycle_check(self):
        """Stress: If a graph has BOTH a missing dependency AND a cycle, missing dep is raised first."""
        dag = ExecutionPlanDAG()
        # Cyclic pair
        dag.add_task(SubTask(id="cycle_A", dependencies=["cycle_B"]))
        dag.add_task(SubTask(id="cycle_B", dependencies=["cycle_A"]))
        # Node referencing non-existent
        dag.add_task(SubTask(id="broken_node", dependencies=["phantom_task_xyz"]))

        with pytest.raises(DAGDependencyError) as exc_info:
            dag.validate()
        assert "phantom_task_xyz" in str(exc_info.value)


# ==============================================================================
# SECTION 3: SCALE, DEEP CHAINS & MULTI-TIER LATTICE TOPOLOGIES
# ==============================================================================


class TestScaleAndDeepTopologies:
    """Stress test deep linear chains, cascading diamond lattices, and 1000-node fan-out."""

    def test_deep_linear_chain_200_nodes(self):
        """Stress: Deep linear chain of 200 nodes (task_0 -> task_1 -> ... -> task_199)."""
        num_nodes = 200
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="task_000"))
        for i in range(1, num_nodes):
            prev_id = f"task_{i-1:03d}"
            curr_id = f"task_{i:03d}"
            dag.add_task(SubTask(id=curr_id, dependencies=[prev_id]))

        assert len(dag) == num_nodes
        start_t = time.perf_counter()
        order = dag.get_execution_order()
        duration = time.perf_counter() - start_t

        assert len(order) == num_nodes
        assert [t.id for t in order] == [f"task_{i:03d}" for i in range(num_nodes)]
        # Must execute in < 100ms
        assert duration < 0.10, f"Deep chain topological sort took too long: {duration:.4f}s"

    def test_cascading_diamond_lattice_100_plus_nodes(self):
        """Stress: 25 layers of diamond lattices (root -> (A, B) -> join -> (A, B) -> ...)."""
        num_layers = 25
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="layer_00_join"))

        for i in range(1, num_layers + 1):
            prev_join = f"layer_{i-1:02d}_join"
            left = f"layer_{i:02d}_left"
            right = f"layer_{i:02d}_right"
            curr_join = f"layer_{i:02d}_join"

            dag.add_task(SubTask(id=left, dependencies=[prev_join]))
            dag.add_task(SubTask(id=right, dependencies=[prev_join]))
            dag.add_task(SubTask(id=curr_join, dependencies=[left, right]))

        # Total nodes: 1 + 25 * 3 = 76 nodes
        # Add extra width to exceed 100 nodes
        for i in range(1, 30):
            dag.add_task(SubTask(id=f"extra_branch_{i:02d}", dependencies=["layer_00_join"]))

        assert len(dag) > 100
        order = dag.get_execution_order()
        assert len(order) == len(dag)

        # Invariant: every dependency appears strictly before its dependent in order
        pos_map = {t.id: idx for idx, t in enumerate(order)}
        for task in dag.tasks.values():
            for dep in task.dependencies:
                assert pos_map[dep] < pos_map[task.id], f"{dep} not before {task.id}"

    def test_massive_wide_fan_out_1000_nodes(self):
        """Stress: 1 root node fanning out to 1,000 parallel tasks, all fanning in to 1 sink node."""
        num_workers = 1000
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="root_dispatcher"))

        worker_ids = [f"worker_{i:04d}" for i in range(num_workers)]
        for wid in worker_ids:
            dag.add_task(SubTask(id=wid, dependencies=["root_dispatcher"]))

        dag.add_task(SubTask(id="sink_aggregator", dependencies=worker_ids))

        assert len(dag) == num_workers + 2

        start_t = time.perf_counter()
        order = dag.get_execution_order()
        duration = time.perf_counter() - start_t

        assert len(order) == num_workers + 2
        assert order[0].id == "root_dispatcher"
        assert order[-1].id == "sink_aggregator"
        # Workers must be deterministically sorted by ID
        assert [t.id for t in order[1:-1]] == sorted(worker_ids)
        assert duration < 0.50, f"1000-node fan-out sort took too long: {duration:.4f}s"

    def test_random_large_acyclic_dag_ordering_soundness(self):
        """Stress: 300-node randomly generated DAG strictly following forward indices."""
        random.seed(42)
        num_nodes = 300
        dag = ExecutionPlanDAG()

        # By creating edges only from lower index to higher index, graph is guaranteed acyclic
        for i in range(num_nodes):
            node_id = f"node_{i:03d}"
            deps: List[str] = []
            if i > 0:
                # Pick 1 to 4 random predecessors
                k = min(i, random.randint(1, 4))
                pred_indices = random.sample(range(i), k)
                deps = [f"node_{p:03d}" for p in sorted(pred_indices)]
            dag.add_task(SubTask(id=node_id, dependencies=deps))

        order = dag.get_execution_order()
        assert len(order) == num_nodes

        # Verify topological property for all 300 nodes
        pos = {t.id: idx for idx, t in enumerate(order)}
        for task in dag.tasks.values():
            for dep in task.dependencies:
                assert pos[dep] < pos[task.id]


# ==============================================================================
# SECTION 4: TASK READINESS & DYNAMIC STATUS STATE MACHINE
# ==============================================================================


class TestAdversarialTaskReadiness:
    """Stress test get_ready_tasks under dynamic states, failure branches, and partial sets."""

    def test_ready_tasks_on_empty_dag(self):
        """Stress: get_ready_tasks on empty DAG returns empty list."""
        dag = ExecutionPlanDAG()
        assert dag.get_ready_tasks() == []
        assert dag.get_ready_tasks(set()) == []
        assert dag.get_ready_tasks({"task_x"}) == []

    def test_ready_tasks_with_dynamic_status_updates_and_retries(self):
        """Stress: State machine lifecycle: pending -> in_progress -> failed -> completed."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="step_1"))
        dag.add_task(SubTask(id="step_2", dependencies=["step_1"]))
        dag.add_task(SubTask(id="step_3", dependencies=["step_2"]))

        # Initially step_1 is ready
        assert [t.id for t in dag.get_ready_tasks()] == ["step_1"]

        # step_1 starts
        dag["step_1"].status = "in_progress"
        # Since step_1 is in_progress (not completed), step_2 is NOT ready
        assert [t.id for t in dag.get_ready_tasks()] == ["step_1"]

        # step_1 fails
        dag["step_1"].status = "failed"
        dag["step_1"].retry_count += 1
        # step_1 is still not completed, so step_2 is still blocked
        assert [t.id for t in dag.get_ready_tasks()] == ["step_1"]

        # step_1 retried and succeeds
        dag["step_1"].status = "completed"
        # Now step_2 becomes ready!
        assert [t.id for t in dag.get_ready_tasks()] == ["step_2"]

        # step_2 completes
        dag["step_2"].status = "completed"
        assert [t.id for t in dag.get_ready_tasks()] == ["step_3"]

        # step_3 completes
        dag["step_3"].status = "completed"
        assert dag.get_ready_tasks() == []

    def test_ready_tasks_explicit_completed_set_overrides_task_status(self):
        """Stress: Passing completed_task_ids explicitly overrides internal task status."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="t1", status="pending"))
        dag.add_task(SubTask(id="t2", dependencies=["t1"], status="pending"))

        # Internal status is pending, but caller explicitly asserts t1 is completed
        ready = dag.get_ready_tasks(completed_task_ids={"t1"})
        assert [t.id for t in ready] == ["t2"]

    def test_ready_tasks_partial_completion_in_wide_tree(self):
        """Stress: Tree with 1 root, 3 branches (B1, B2, B3), each having 2 leaves."""
        dag = ExecutionPlanDAG()
        dag.add_task(SubTask(id="root"))
        for b in ["b1", "b2", "b3"]:
            dag.add_task(SubTask(id=b, dependencies=["root"]))
            dag.add_task(SubTask(id=f"{b}_leaf_1", dependencies=[b]))
            dag.add_task(SubTask(id=f"{b}_leaf_2", dependencies=[b]))

        # Complete only root and b2
        completed = {"root", "b2"}
        ready = dag.get_ready_tasks(completed_task_ids=completed)
        ready_ids = [t.id for t in ready]

        # b1 and b3 should be ready (since root is done)
        assert "b1" in ready_ids
        assert "b3" in ready_ids
        # b2's leaves should be ready (since b2 is done)
        assert "b2_leaf_1" in ready_ids
        assert "b2_leaf_2" in ready_ids
        # b1 and b3's leaves must NOT be ready
        assert "b1_leaf_1" not in ready_ids
        assert "b3_leaf_1" not in ready_ids


# ==============================================================================
# SECTION 5: SERIALIZATION ROUNDTRIPS & RESILIENT JSON PARSING
# ==============================================================================


class TestAdversarialSerialization:
    """Stress test serialization roundtrips with nested metadata and malformed LLM outputs."""

    def test_serialization_with_arbitrary_nested_unicode_and_numerical_metadata(self):
        """Stress: ExecutionPlanDAG containing deep nested metadata, emojis, and diverse types."""
        metadata = {
            "orchestrator_version": "2.4.0-rc1",
            "telemetry": {
                "input_tokens": 4096,
                "output_tokens": 1024,
                "cost_usd": 0.003541,
                "benchmarks": [12.4, 15.8, 11.2],
            },
            "environment": {
                "os": "Linux x86_64",
                "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU",
                "unicode_tag": "⚡ Rocket Launch 🚀 — 100% Green ✅",
            },
            "is_experimental": True,
            "null_value": None,
        }

        dag = ExecutionPlanDAG(metadata=metadata)
        dag.add_task(
            SubTask(
                id="task_unicode",
                title="Handle Unicode: 🚀 & 漢字",
                description="Process utf-8 payloads safely",
                dependencies=[],
                affected_files=["src/α_beta.py"],
                success_criteria="Unicode invariant preserved",
                verification_command="pytest tests/test_unicode.py",
                status="in_progress",
                retry_count=3,
            )
        )

        # 1. Dict roundtrip
        d = dag.to_dict()
        dag_from_dict = ExecutionPlanDAG.from_dict(d)
        assert dag_from_dict.metadata == metadata
        assert dag_from_dict["task_unicode"].title == "Handle Unicode: 🚀 & 漢字"
        assert dag_from_dict["task_unicode"].affected_files == ["src/α_beta.py"]
        assert dag_from_dict["task_unicode"].retry_count == 3

        # 2. JSON roundtrip
        json_str = dag.to_json()
        dag_from_json = ExecutionPlanDAG.from_json(json_str)
        assert dag_from_json.metadata == metadata
        assert dag_from_json["task_unicode"].status == "in_progress"

    def test_json_deserialization_from_markdown_fences_with_surrounding_chatter(self):
        """Stress: Deserializing JSON embedded within markdown fences and conversational text."""
        raw_llm_response = """
Here is the execution plan you requested:

```json
{
  "metadata": {
    "planner": "Gemini-1.5-Pro"
  },
  "tasks": [
    {
      "id": "t1",
      "title": "Setup",
      "dependencies": []
    },
    {
      "id": "t2",
      "title": "Execute",
      "dependencies": ["t1"]
    }
  ]
}
```

Let me know if you need any adjustments to these subtasks!
"""
        dag = ExecutionPlanDAG.from_json(raw_llm_response)
        assert len(dag) == 2
        assert dag.metadata["planner"] == "Gemini-1.5-Pro"
        assert [t.id for t in dag.get_execution_order()] == ["t1", "t2"]

    def test_json_deserialization_with_illegal_trailing_commas(self):
        """Stress: Resilient regex repairs trailing commas in JSON arrays and objects."""
        raw_with_trailing_commas = """
{
  "tasks": [
    {
      "id": "t1",
      "title": "Task 1",
      "dependencies": [],
    },
    {
      "id": "t2",
      "title": "Task 2",
      "dependencies": ["t1",],
    },
  ],
}
"""
        dag = ExecutionPlanDAG.from_json(raw_with_trailing_commas)
        assert len(dag) == 2
        assert dag.get_task("t2").dependencies == ["t1"]


# ==============================================================================
# SECTION 6: CLOUD ARCHITECT CLIENT & MOCK PLANNER ADVERSARIAL STRESS
# ==============================================================================


class TestAdversarialArchitectPlanning:
    """Stress test CloudArchitectClient and MockCloudArchitectClient planners."""

    def test_mock_planner_all_archetypes_validated_and_cycle_free(self):
        """Stress: Verify every goal keyword produces a valid, acyclic DAG."""
        client = MockCloudArchitectClient()
        goals = [
            "Compute fibonacci numbers using dynamic programming",
            "Reproduce and patch null pointer bug in payment processing",
            "Refactor monolithic storage engine into interfaces",
            "Implement diamond architecture with parallel stages",
            "Execute independent parallel batch workers",
            "Single quick patch to readme documentation",
            "Generic goal that defaults to standard 3-stage pipeline",
        ]

        for goal in goals:
            dag = client.plan_rd_task(goal)
            assert len(dag) >= 1
            # validate() MUST pass without DAGCycleError or DAGDependencyError
            assert dag.validate() is True
            order = dag.get_execution_order()
            assert len(order) == len(dag)

    def test_mock_planner_adversarial_injections(self):
        """Stress: Test injection hooks in MockCloudArchitectClient."""
        # 1. Cycle injection
        cycle_client = MockCloudArchitectClient(inject_cycle=True)
        dag_cycle = cycle_client.plan_rd_task("any goal")
        with pytest.raises(DAGCycleError):
            dag_cycle.validate()

        # 2. Missing dependency injection
        missing_client = MockCloudArchitectClient(inject_missing_dependency=True)
        dag_missing = missing_client.plan_rd_task("any goal")
        with pytest.raises(DAGDependencyError):
            dag_missing.validate()

        # 3. Custom exception injection
        error_client = MockCloudArchitectClient(inject_error=RuntimeError("Simulated LLM outage"))
        with pytest.raises(RuntimeError, match="Simulated LLM outage"):
            error_client.plan_rd_task("any goal")

    def test_mock_planner_preflight_blocks_secret_leak_in_outline(self, leaked_outline: WorkspaceOutline):
        """Stress: Pre-flight privacy check prevents leaked outline from generating plan."""
        client = MockCloudArchitectClient()
        with pytest.raises(SecurityInvariantViolationError):
            client.plan_rd_task(
                goal="Scaffold authentication system",
                workspace_outline=leaked_outline,
            )

    @patch.object(CloudArchitectClient, "_invoke_codex")
    def test_cloud_planner_retry_on_cycle_then_succeeds(self, mock_invoke):
        """Stress: CloudArchitectClient receives cyclic plan on attempt 1, feeds error back, succeeds on attempt 2."""
        cyclic_json = json.dumps({
            "tasks": [
                {"id": "A", "dependencies": ["B"]},
                {"id": "B", "dependencies": ["A"]},
            ]
        })
        valid_json = json.dumps({
            "tasks": [
                {"id": "A", "dependencies": []},
                {"id": "B", "dependencies": ["A"]},
            ]
        })

        mock_invoke.side_effect = [
            (cyclic_json, 100, 50),
            (valid_json, 150, 60),
        ]

        client = CloudArchitectClient()
        dag = client.plan_rd_task("Implement feature", max_retries=2)

        assert mock_invoke.call_count == 2
        # Second call prompt must contain diagnostic about cycle
        second_call_prompt = mock_invoke.call_args_list[1][0][0]
        assert "CRITICAL ERROR IN PREVIOUS RESPONSE" in second_call_prompt
        assert "Cycle detected" in second_call_prompt
        assert len(dag) == 2
        assert [t.id for t in dag.get_execution_order()] == ["A", "B"]

    @patch.object(CloudArchitectClient, "_invoke_codex")
    def test_cloud_planner_persistent_cycle_raises_dag_cycle_error(self, mock_invoke):
        """Stress: Persistent cycle across all retries raises DAGCycleError directly."""
        cyclic_json = json.dumps({
            "tasks": [
                {"id": "A", "dependencies": ["B"]},
                {"id": "B", "dependencies": ["A"]},
            ]
        })
        mock_invoke.return_value = (cyclic_json, 100, 50)

        client = CloudArchitectClient()
        with pytest.raises(DAGCycleError):
            client.plan_rd_task("Implement feature", max_retries=1)

    def test_cloud_planner_preflight_blocks_leak_before_invoking_codex(self, leaked_outline: WorkspaceOutline):
        """Stress: CloudArchitectClient MUST abort on privacy violation before running codex subprocess."""
        client = CloudArchitectClient()
        with patch.object(client, "_invoke_codex") as mock_invoke:
            with pytest.raises(SecurityInvariantViolationError):
                client.plan_rd_task("Fix auth", workspace_outline=leaked_outline)
            mock_invoke.assert_not_called()
