"""Unit and integration test suite for AutonomousWorker and MockAutonomousWorker.

Milestone 3, Feature F9 & F10.
Contains 28 tests (11 Tier 1 + 9 Tier 2 + 8 Tier 3) specified by spec_miner_m3_1_gen3/report.md.
"""

import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import time
import pytest

from antigravity_hybrid_orchestrator.architect.mock_client import MockCloudArchitectClient
from antigravity_hybrid_orchestrator.planner.dag import ExecutionPlanDAG, SubTask
from antigravity_hybrid_orchestrator.workforce.mock_worker import MockAutonomousWorker
from antigravity_hybrid_orchestrator.workforce.tools import ToolResult, WorkspaceToolSuite
from antigravity_hybrid_orchestrator.workforce.worker import (
    AutonomousWorker,
    ParsedAction,
    SubtaskResult,
)


@pytest.fixture
def tool_suite(tmp_path: Path) -> WorkspaceToolSuite:
    """Fixture providing a fresh WorkspaceToolSuite backed by tmp_path."""
    return WorkspaceToolSuite(workspace_root=tmp_path)


# ==============================================================================
# TIER 1: FEATURE COVERAGE (11 TESTS)
# ==============================================================================

def test_react_parser_json_function_call_format():
    """Test 29: Parses model output formatted as JSON function call."""
    model_output = json.dumps({
        "thought": "Need to check existing code",
        "action": "read_file",
        "action_input": {"path": "src/main.py"},
    })
    parsed = AutonomousWorker._parse_action(model_output)

    assert parsed.is_final is False
    assert parsed.action == "read_file"
    assert parsed.args == {"path": "src/main.py"}
    assert parsed.thought == "Need to check existing code"


def test_react_parser_text_action_pattern():
    """Test 30: Parses classic ReAct text Thought / Action / Action Input."""
    model_output = (
        "Thought: I should write the implementation file now.\n"
        "Action: write_file\n"
        'Action Input: {"path": "src/calc.py", "content": "def add(a, b): return a + b"}'
    )
    parsed = AutonomousWorker._parse_action(model_output)

    assert parsed.is_final is False
    assert parsed.action == "write_file"
    assert parsed.args == {"path": "src/calc.py", "content": "def add(a, b): return a + b"}
    assert "write the implementation file" in parsed.thought


def test_react_parser_final_answer_completion():
    """Test 31: Detects Final Answer pattern and marks completion."""
    model_output = (
        "Thought: Everything has been implemented and tests are passing.\n"
        "Final Answer: Completed task successfully. All requirements satisfied."
    )
    parsed = AutonomousWorker._parse_action(model_output)

    assert parsed.is_final is True
    assert parsed.summary == "Completed task successfully. All requirements satisfied."


def test_react_parser_action_finish_format():
    """Test 32: Detects Action: finish with summary in Action Input."""
    model_output = (
        "Action: finish\n"
        'Action Input: {"summary": "Implemented arithmetic module and verified tests"}'
    )
    parsed = AutonomousWorker._parse_action(model_output)

    assert parsed.is_final is True
    assert parsed.summary == "Implemented arithmetic module and verified tests"


def test_mock_autonomous_worker_executes_single_subtask(tool_suite: WorkspaceToolSuite, tmp_path: Path):
    """Test 33: MockAutonomousWorker executes single subtask and writes affected file to disk."""
    subtask = SubTask(
        id="task_1",
        title="Add math module",
        affected_files=["math_mod.py"],
    )
    mock_worker = MockAutonomousWorker()
    res = mock_worker.execute_subtask(subtask, tool_suite)

    assert res.subtask_id == "task_1"
    assert res.success is True
    assert (tmp_path / "math_mod.py").exists()
    assert len((tmp_path / "math_mod.py").read_text(encoding="utf-8")) > 0


def test_mock_autonomous_worker_execution_speed_invariant(tool_suite: WorkspaceToolSuite):
    """Test 34: 10 subtasks execute in <0.5s total (<0.05s per subtask speed invariant)."""
    mock_worker = MockAutonomousWorker()
    start = time.perf_counter()

    for i in range(10):
        subtask = SubTask(
            id=f"speed_task_{i}",
            title=f"Speed test subtask {i}",
            affected_files=[f"file_{i}.txt"],
        )
        res = mock_worker.execute_subtask(subtask, tool_suite)
        assert res.success is True

    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"Execution took {elapsed}s, expected <0.5s"


def test_subtask_result_serialization_to_dict():
    """Test 35: SubtaskResult serializes to complete dictionary."""
    res = SubtaskResult(
        subtask_id="t1",
        success=True,
        turns=3,
        tool_calls_count=2,
        summary="Done",
        error=None,
        artifacts=["file1.py"],
        metadata={"k": "v"},
    )
    d = res.to_dict()

    assert d["subtask_id"] == "t1"
    assert d["success"] is True
    assert d["turns"] == 3
    assert d["tool_calls_count"] == 2
    assert d["summary"] == "Done"
    assert d["error"] is None
    assert d["artifacts"] == ["file1.py"]
    assert d["metadata"] == {"k": "v"}
    # Verify serializable
    json.dumps(d)


def test_subtask_result_deserialization_from_dict():
    """Test 36: SubtaskResult instantiates safely from partial dictionary."""
    d = {
        "subtask_id": "t2",
        "success": True,
    }
    res = SubtaskResult.from_dict(d)

    assert res.subtask_id == "t2"
    assert res.success is True
    assert res.turns == 1
    assert res.tool_calls_count == 0
    assert res.summary == ""
    assert res.error is None
    assert res.artifacts == []


def test_subtask_result_to_json_and_from_json():
    """Test 37: SubtaskResult roundtrips through JSON serialization."""
    original = SubtaskResult(
        subtask_id="t3",
        success=False,
        turns=4,
        tool_calls_count=3,
        summary="Failed on step 4",
        error="SyntaxError",
        artifacts=["test.py"],
        metadata={"retry": 1},
    )
    json_str = original.to_json()
    reconstructed = SubtaskResult.from_json(json_str)

    assert reconstructed.subtask_id == original.subtask_id
    assert reconstructed.success == original.success
    assert reconstructed.turns == original.turns
    assert reconstructed.tool_calls_count == original.tool_calls_count
    assert reconstructed.summary == original.summary
    assert reconstructed.error == original.error
    assert reconstructed.artifacts == original.artifacts
    assert reconstructed.metadata == original.metadata


def test_subtask_result_boolean_evaluation():
    """Test 38: SubtaskResult evaluates truthiness based on success attribute."""
    res_pass = SubtaskResult(subtask_id="1", success=True)
    res_fail = SubtaskResult(subtask_id="2", success=False)

    assert bool(res_pass) is True
    assert bool(res_fail) is False


def test_mock_autonomous_worker_legacy_workforce_client_methods():
    """Test 39: MockAutonomousWorker supports generate_stream and generate for gauntlet compatibility."""
    worker = MockAutonomousWorker()
    stream = list(worker.generate_stream(prompt="fix auth"))

    assert len(stream) > 1
    _, _, final_meta = stream[-1]
    assert final_meta["done"] is True
    assert final_meta["eval_count"] > 0
    assert final_meta["eval_duration_s"] > 0

    sync_out = worker.generate(prompt="fix auth")
    assert isinstance(sync_out, str)
    assert len(sync_out) > 0


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES (9 TESTS)
# ==============================================================================

def test_react_parser_malformed_json_in_action_input():
    """Test 40: ReAct parser handles malformed JSON in Action Input gracefully."""
    model_output = 'Action: write_file\nAction Input: {path: "test.py", invalid_json'
    parsed = AutonomousWorker._parse_action(model_output)

    assert parsed.is_final is False
    assert parsed.error is not None
    assert "Failed to parse Action Input JSON" in parsed.error


def test_react_parser_missing_action_input_field():
    """Test 41: ReAct parser defaults args to empty dict when Action Input is omitted."""
    model_output = "Action: git_status"
    parsed = AutonomousWorker._parse_action(model_output)

    assert parsed.is_final is False
    assert parsed.action == "git_status"
    assert parsed.args == {}


def test_react_parser_unhandled_tool_name(tool_suite: WorkspaceToolSuite):
    """Test 42: Tool dispatch reports unknown tool gracefully without exception."""
    res = tool_suite.execute("teleport_to_cloud")
    assert res.success is False
    assert "Unknown tool 'teleport_to_cloud'" in (res.error or "")


def test_autonomous_worker_stops_repeated_no_progress(tool_suite: WorkspaceToolSuite):
    """A repeated identical action and observation trips the stall detector."""
    # Mock client that loops indefinitely without completing
    class LoopingMockClient:
        def generate(self, prompt: str, system_prompt: str = None) -> str:
            return 'Action: read_file\nAction Input: {"path": "dummy.txt"}'

    worker = AutonomousWorker(max_turns=3, client=LoopingMockClient())
    subtask = SubTask(id="loop_task", title="Infinite loop task")

    res = worker.execute_subtask(subtask, tool_suite)
    assert res.success is False
    assert res.turns >= 3
    assert res.metadata.get("stalled") is True
    assert "Execution stalled" in (res.error or "")


def test_mock_worker_fail_subtasks_injection(tool_suite: WorkspaceToolSuite):
    """Test 44: fail_subtasks correctly injects failure for specific task IDs."""
    worker = MockAutonomousWorker(fail_subtasks={"task_fail"})

    fail_task = SubTask(id="task_fail", title="Task intended to fail")
    pass_task = SubTask(id="task_pass", title="Task intended to succeed")

    res_fail = worker.execute_subtask(fail_task, tool_suite)
    assert res_fail.success is False
    assert "Injected failure for subtask task_fail" in (res_fail.error or "")

    res_pass = worker.execute_subtask(pass_task, tool_suite)
    assert res_pass.success is True


def test_mock_worker_error_on_tool_injection(tool_suite: WorkspaceToolSuite):
    """Test 45: error_on_tool correctly simulates injected tool failure."""
    worker = MockAutonomousWorker(error_on_tool={"write_file": "Simulated disk I/O error"})
    subtask = SubTask(id="task_write", affected_files=["output.txt"])

    res = worker.execute_subtask(subtask, tool_suite)
    assert res.success is False
    assert "Simulated disk I/O error" in (res.error or "")


def test_mock_worker_custom_file_contents_override(tool_suite: WorkspaceToolSuite, tmp_path: Path):
    """Test 46: custom_file_contents overrides generated file content."""
    custom_text = "[section]\nkey=custom_val\n"
    worker = MockAutonomousWorker(custom_file_contents={"config.ini": custom_text})

    subtask = SubTask(id="task_cfg", affected_files=["config.ini"])
    res = worker.execute_subtask(subtask, tool_suite)

    assert res.success is True
    assert (tmp_path / "config.ini").read_text(encoding="utf-8") == custom_text


def test_worker_single_concurrency_lock_prevents_overlap(tool_suite: WorkspaceToolSuite):
    """Test 47: AutonomousWorker class-level lock serializes concurrent inferences."""
    execution_intervals = []

    class DelayedMockClient:
        def generate(self, prompt: str, system_prompt: str = None) -> str:
            start = time.perf_counter()
            time.sleep(0.05)
            end = time.perf_counter()
            execution_intervals.append((start, end))
            return "Final Answer: Done"

    worker = AutonomousWorker(client=DelayedMockClient())
    task1 = SubTask(id="thread_task_1")
    task2 = SubTask(id="thread_task_2")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(worker.execute_subtask, task1, tool_suite)
        f2 = executor.submit(worker.execute_subtask, task2, tool_suite)
        res1 = f1.result()
        res2 = f2.result()

    assert res1.success is True
    assert res2.success is True
    assert len(execution_intervals) == 2

    # Verify no temporal overlap: start of second >= end of first (with small epsilon)
    interval_a, interval_b = sorted(execution_intervals, key=lambda x: x[0])
    assert interval_b[0] >= interval_a[1] - 0.005


def test_worker_handles_unexpected_tool_exception_gracefully():
    """Test 48: AutonomousWorker catches unexpected tool exceptions without agent process crash."""
    class CrashingToolSuite:
        workspace_root = Path("/tmp")

        def get_tool_definitions(self):
            return [{"name": "crash_tool"}]

        def execute(self, name, **kwargs):
            raise RuntimeError("Hardware fault during disk access")

    class CrashTriggerClient:
        def generate(self, prompt: str, system_prompt: str = None) -> str:
            if "Observation" in prompt:
                return "Final Answer: Handled error gracefully"
            return "Action: crash_tool\nAction Input: {}"

    worker = AutonomousWorker(client=CrashTriggerClient())
    subtask = SubTask(id="crash_task")

    res = worker.execute_subtask(subtask, CrashingToolSuite())
    assert res.success is True
    assert "Handled error gracefully" in res.summary


# ==============================================================================
# TIER 3: CROSS-FEATURE INTEGRATION (8 TESTS)
# ==============================================================================

def test_dag_subtask_execution_with_tool_suite(tool_suite: WorkspaceToolSuite, tmp_path: Path):
    """Test 49: Executes chained DAG subtasks sequentially with MockAutonomousWorker."""
    t1 = SubTask(id="t1", title="Init module", affected_files=["t1.py"])
    t2 = SubTask(id="t2", title="Add logic", dependencies=["t1"], affected_files=["t2.py"])
    t3 = SubTask(id="t3", title="Run tests", dependencies=["t2"], affected_files=["t3.py"])
    dag = ExecutionPlanDAG(tasks=[t1, t2, t3])

    worker = MockAutonomousWorker()
    order = dag.get_execution_order()

    for task in order:
        res = worker.execute_subtask(task, tool_suite)
        assert res.success is True
        task.status = "completed"

    assert all(t.status == "completed" for t in dag.tasks.values())
    assert (tmp_path / "t1.py").exists()
    assert (tmp_path / "t2.py").exists()
    assert (tmp_path / "t3.py").exists()



def test_dag_subtask_verification_command_execution(tool_suite: WorkspaceToolSuite):
    """Test 50: Subtask with verification_command runs validation and passes."""
    subtask = SubTask(
        id="task_calc",
        title="Add math calculator module",
        affected_files=["calc.py"],
        verification_command=f'{sys.executable} -c "import calc; assert calc.add(2, 3) == 5"',
    )
    worker = MockAutonomousWorker()
    res = worker.execute_subtask(subtask, tool_suite)

    assert res.success is True


def test_hud_event_callback_sequence_verification(tool_suite: WorkspaceToolSuite):
    """Test 51: Recorded HUD events follow exact protocol sequence."""
    events = []
    worker = MockAutonomousWorker(
        event_callback=lambda et, tid, p: events.append((et, tid, p))
    )
    subtask = SubTask(id="seq_task", title="Sequential task", affected_files=["seq.txt"])

    res = worker.execute_subtask(subtask, tool_suite)
    assert res.success is True

    event_types = [e[0] for e in events]
    assert event_types == [
        "subtask_start",
        "thought",
        "tool_call",
        "tool_result",
        "subtask_complete",
    ]


def test_hud_event_payload_schema_conformance(tool_suite: WorkspaceToolSuite):
    """Test 52: HUD event payloads conform to schema expectations."""
    events = []
    worker = MockAutonomousWorker(
        event_callback=lambda et, tid, p: events.append((et, tid, p))
    )
    subtask = SubTask(id="schema_task", affected_files=["test.py"])

    worker.execute_subtask(subtask, tool_suite)

    tool_call_events = [e for e in events if e[0] == "tool_call"]
    assert len(tool_call_events) >= 1
    call_payload = tool_call_events[0][2]
    assert "tool" in call_payload
    assert "args" in call_payload
    assert "turn" in call_payload

    tool_res_events = [e for e in events if e[0] == "tool_result"]
    assert len(tool_res_events) >= 1
    res_payload = tool_res_events[0][2]
    assert "tool" in res_payload
    assert "success" in res_payload
    assert "output" in res_payload
    assert "error" in res_payload


def test_feedback_context_ingestion_on_retry(tool_suite: WorkspaceToolSuite):
    """Test 53: Worker incorporates feedback_context in thought reasoning."""
    events = []
    worker = MockAutonomousWorker(
        event_callback=lambda et, tid, p: events.append((et, tid, p))
    )
    subtask = SubTask(id="retry_task")
    feedback = "AssertionError: Expected 42, got 0"

    res = worker.execute_subtask(subtask, tool_suite, feedback_context=feedback)
    assert res.success is True

    thought_events = [e for e in events if e[0] == "thought"]
    assert len(thought_events) >= 1
    assert feedback in thought_events[0][2]["thought"]


def test_end_to_end_subtask_with_git_tracking(tool_suite: WorkspaceToolSuite, tmp_path: Path):
    """Test 54: Git status and diff accurately track files created by worker subtask."""
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(tmp_path), check=True)

    worker = MockAutonomousWorker()
    subtask = SubTask(id="git_task", affected_files=["script.py"])
    res = worker.execute_subtask(subtask, tool_suite)
    assert res.success is True

    status_res = tool_suite.git_status()
    assert status_res.success is True
    assert "script.py" in status_res.metadata["changes"]

    # Stage the file and check diff
    subprocess.run(["git", "add", "script.py"], cwd=str(tmp_path), check=True)
    diff_res = tool_suite.git_diff(staged=True)
    assert diff_res.success is True
    assert diff_res.metadata["has_diff"] is True


def test_dag_dependency_blocking_on_task_failure(tool_suite: WorkspaceToolSuite):
    """Test 55: Failure of upstream subtask prevents downstream dependent task from becoming ready."""
    t1 = SubTask(id="setup", title="Setup environment")
    t2 = SubTask(id="build", title="Build binary", dependencies=["setup"])
    t3 = SubTask(id="test", title="Run integration tests", dependencies=["build"])
    dag = ExecutionPlanDAG(tasks=[t1, t2, t3])

    worker = MockAutonomousWorker(fail_subtasks={"build"})

    # Setup succeeds
    res_setup = worker.execute_subtask(t1, tool_suite)
    assert res_setup.success is True
    t1.status = "completed"

    # Build is ready and fails
    ready = dag.get_ready_tasks()
    assert [t.id for t in ready] == ["build"]

    res_build = worker.execute_subtask(t2, tool_suite)
    assert res_build.success is False
    t2.status = "failed"

    # Test task MUST NOT be ready
    ready_after_fail = dag.get_ready_tasks()
    assert "test" not in [t.id for t in ready_after_fail]
    assert not all(t.status == "completed" for t in dag.tasks.values())



def test_worker_execution_with_mock_cloud_architect_dag(tool_suite: WorkspaceToolSuite):
    """Test 56: MockAutonomousWorker executes entire DAG produced by MockCloudArchitectClient."""
    architect = MockCloudArchitectClient()
    dag = architect.plan_rd_task(goal="Create CLI data processor")

    worker = MockAutonomousWorker()
    for task in dag.get_execution_order():
        res = worker.execute_subtask(task, tool_suite)
        assert res.success is True
        task.status = "completed"

    assert all(t.status == "completed" for t in dag.tasks.values())
