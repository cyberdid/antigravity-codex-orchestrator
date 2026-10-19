"""Regression test suite (T01-T10) for GEMMA-ORION Phase 1:
Thinking Mode, Output/Preflight Budget, Function-Calling History,
Truncation Safety, Telemetry, and Deterministic ResearchReportBuilder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from antigravity_hybrid_orchestrator.integrity.evidence import validate_narrative
from antigravity_hybrid_orchestrator.integrity.report_builder import (
    INTERPRETATION_JSON_SCHEMA,
    ResearchReportBuilder,
)
from antigravity_hybrid_orchestrator.planner.dag import SubTask
from antigravity_hybrid_orchestrator.workforce.context import (
    compact_messages,
    effective_num_predict,
    ollama_think,
    preflight_context_budget,
    strip_completed_thinking,
)
from antigravity_hybrid_orchestrator.workforce.ollama_client import (
    OllamaConnectionError,
    WorkforceClient,
)
from antigravity_hybrid_orchestrator.workforce.tools import WorkspaceToolSuite
from antigravity_hybrid_orchestrator.workforce.worker import AutonomousWorker


@pytest.fixture
def tool_suite(tmp_path: Path) -> WorkspaceToolSuite:
    return WorkspaceToolSuite(workspace_root=tmp_path)


def test_t01_think_false_read_file(monkeypatch: pytest.MonkeyPatch, tool_suite: WorkspaceToolSuite, tmp_path: Path):
    """T01: think=false, standard read_file tool call and completion."""
    monkeypatch.setenv("OLLAMA_THINK", "false")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "16384")
    monkeypatch.delenv("OLLAMA_NUM_PREDICT", raising=False)
    (tmp_path / "input.txt").write_text("alpha=42\n", encoding="utf-8")

    payloads: List[Dict[str, Any]] = []
    responses = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": {"path": "input.txt"}}}
                ],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 110,
            "eval_count": 24,
            "eval_duration": 400_000_000,
        },
        {
            "message": {
                "role": "assistant",
                "content": "Read input.txt successfully.",
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 145,
            "eval_count": 12,
            "eval_duration": 200_000_000,
        },
    ]

    client = WorkforceClient(base_url="http://127.0.0.1:11434", model="gemma4:12b")
    monkeypatch.setattr(client, "get_model_digest", lambda: "sha256:gemma4testdigest")

    def fake_post(url: str, payload: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        payloads.append(json.loads(json.dumps(payload)))
        return responses.pop(0)

    monkeypatch.setattr(client, "_post_json", fake_post)
    events: List[tuple[str, Dict[str, Any]]] = []
    worker = AutonomousWorker(
        client=client,
        event_callback=lambda kind, sid, data: events.append((kind, data)),
    )
    res = worker.execute_subtask(
        SubTask(id="t01", title="Read input", input_files=["input.txt"]),
        tool_suite,
    )

    assert res.success is True
    assert res.turns == 2
    assert res.tool_calls_count == 1
    assert payloads[0]["think"] is False
    assert payloads[0]["options"]["num_predict"] == 2048
    assert res.metadata["think_enabled"] is False
    assert res.metadata["model_digest"] == "sha256:gemma4testdigest"
    assert res.metadata["turns_with_thinking"] == 0
    assert res.metadata["truncated_generations"] == 0


def test_t02_think_true_read_file_and_hud_isolation(
    monkeypatch: pytest.MonkeyPatch, tool_suite: WorkspaceToolSuite, tmp_path: Path
):
    """T02: think=true, standard read_file call; thinking parsed into telemetry but excluded from HUD."""
    monkeypatch.setenv("OLLAMA_THINK", "true")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "16384")
    monkeypatch.delenv("OLLAMA_NUM_PREDICT", raising=False)
    (tmp_path / "data.json").write_text('{"k": 10}', encoding="utf-8")

    payloads: List[Dict[str, Any]] = []
    secret_reasoning = "INTERNAL_REASONING_TRACE: inspect data.json first before concluding."
    responses = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "thinking": secret_reasoning,
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": {"path": "data.json"}}}
                ],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 120,
            "eval_count": 48,
            "eval_duration": 600_000_000,
        },
        {
            "message": {
                "role": "assistant",
                "content": "Inspected data.json.",
                "thinking": "Data is valid JSON.",
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 180,
            "eval_count": 30,
            "eval_duration": 350_000_000,
        },
    ]

    client = WorkforceClient(base_url="http://127.0.0.1:11434", model="gemma4:12b")
    monkeypatch.setattr(client, "get_model_digest", lambda: "sha256:gemma4testdigest")

    def fake_post(url: str, payload: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        payloads.append(json.loads(json.dumps(payload)))
        return responses.pop(0)

    monkeypatch.setattr(client, "_post_json", fake_post)
    events: List[tuple[str, Dict[str, Any]]] = []
    worker = AutonomousWorker(
        client=client,
        event_callback=lambda kind, sid, data: events.append((kind, data)),
    )
    res = worker.execute_subtask(
        SubTask(id="t02", title="Read data.json", input_files=["data.json"]),
        tool_suite,
    )

    assert res.success is True
    assert payloads[0]["think"] is True
    # Active tool-calling turn preserves `thinking` in history for the next turn
    assistant_turns_in_second_call = [m for m in payloads[1]["messages"] if m["role"] == "assistant"]
    assert len(assistant_turns_in_second_call) == 1
    assert assistant_turns_in_second_call[0]["thinking"] == secret_reasoning
    # HUD thought events must NOT contain internal thinking traces
    thought_events = [data["thought"] for kind, data in events if kind == "thought"]
    assert all(secret_reasoning not in t for t in thought_events)
    assert res.metadata["think_enabled"] is True
    assert res.metadata["turns_with_thinking"] == 2
    assert res.metadata["thinking_chars"] == len(secret_reasoning) + len("Data is valid JSON.")


def test_t03_think_true_write_file_required_arguments(
    monkeypatch: pytest.MonkeyPatch, tool_suite: WorkspaceToolSuite, tmp_path: Path
):
    """T03: think=true, write_file with both required arguments (path and content)."""
    monkeypatch.setenv("OLLAMA_THINK", "true")
    monkeypatch.setenv("OLLAMA_NUM_PREDICT", "4096")  # G2 configuration
    responses = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "thinking": "Need to write module.py with path first and content second.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write_file",
                            "arguments": {"path": "src/module.py", "content": "def compute():\n    return 7\n"},
                        }
                    }
                ],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 150,
            "eval_count": 65,
        },
        {
            "message": {
                "role": "assistant",
                "content": "Created src/module.py.",
                "thinking": "File written cleanly.",
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 210,
            "eval_count": 20,
        },
    ]
    captured_options: List[Dict[str, Any]] = []
    client = WorkforceClient(base_url="http://127.0.0.1:11434", model="gemma4:12b")
    monkeypatch.setattr(client, "get_model_digest", lambda: "sha256:gemma4")

    def fake_post(url: str, payload: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        captured_options.append(payload["options"])
        return responses.pop(0)

    monkeypatch.setattr(client, "_post_json", fake_post)
    worker = AutonomousWorker(client=client)
    res = worker.execute_subtask(
        SubTask(id="t03", title="Write module", outputs=["src/module.py"], affected_files=["src/module.py"]),
        tool_suite,
    )

    assert res.success is True
    assert (tmp_path / "src/module.py").read_text(encoding="utf-8") == "def compute():\n    return 7\n"
    assert captured_options[0]["num_predict"] == 4096
    assert res.metadata["num_predict"] == 4096
    assert res.metadata["input_budget_tokens"] == 16384 - 4096
    assert res.metadata["invalid_tool_arguments"] == 0


def test_t04_multistep_read_patch_run_sequence(
    monkeypatch: pytest.MonkeyPatch, tool_suite: WorkspaceToolSuite, tmp_path: Path
):
    """T04: Multi-step sequence read_file -> patch_file -> run_command with thinking=true."""
    monkeypatch.setenv("OLLAMA_THINK", "true")
    target = tmp_path / "calc.py"
    target.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")

    payload_Snapshots: List[List[Dict[str, Any]]] = []
    responses = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "thinking": "Step 1: Read calc.py to find the bug.",
                "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "calc.py"}}}],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 100,
            "eval_count": 30,
        },
        {
            "message": {
                "role": "assistant",
                "content": "",
                "thinking": "Step 2: Patch return a - b to return a + b.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "patch_file",
                            "arguments": {
                                "path": "calc.py",
                                "target_content": "return a - b",
                                "replacement_content": "return a + b",
                            },
                        }
                    }
                ],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 160,
            "eval_count": 45,
        },
        {
            "message": {
                "role": "assistant",
                "content": "",
                "thinking": "Step 3: Run verification command.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_command",
                            "arguments": {"command": 'python3 -c "import calc; assert calc.add(2, 3) == 5"'},
                        }
                    }
                ],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 220,
            "eval_count": 40,
        },
        {
            "message": {
                "role": "assistant",
                "content": "Bug fixed and verified.",
                "thinking": "Verification succeeded.",
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 260,
            "eval_count": 15,
        },
    ]

    client = WorkforceClient(base_url="http://127.0.0.1:11434", model="gemma4:12b")
    monkeypatch.setattr(client, "get_model_digest", lambda: "sha256:gemma4")

    def fake_post(url: str, payload: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        payload_Snapshots.append(json.loads(json.dumps(payload["messages"])))
        return responses.pop(0)

    monkeypatch.setattr(client, "_post_json", fake_post)
    worker = AutonomousWorker(client=client)
    res = worker.execute_subtask(
        SubTask(id="t04", title="Fix calc", affected_files=["calc.py"]),
        tool_suite,
    )

    assert res.success is True
    assert res.turns == 4
    assert res.tool_calls_count == 3
    assert res.metadata["successful_verifications"] == 1
    assert "return a + b" in target.read_text(encoding="utf-8")
    # Verify assistant -> tool -> assistant -> tool ordering in the 4th call
    final_call_roles = [m["role"] for m in payload_Snapshots[-1]]
    assert final_call_roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant", "tool"]


def test_t05_retry_after_tool_argument_error(
    monkeypatch: pytest.MonkeyPatch, tool_suite: WorkspaceToolSuite, tmp_path: Path
):
    """T05: Recovery after missing tool argument error; invalid_tool_arguments incremented."""
    monkeypatch.setenv("OLLAMA_THINK", "true")
    responses = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "thinking": "Calling write_file but accidentally forgot path.",
                "tool_calls": [
                    {"function": {"name": "write_file", "arguments": {"content": "hello\n"}}}
                ],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 100,
            "eval_count": 30,
        },
        {
            "message": {
                "role": "assistant",
                "content": "",
                "thinking": "The previous tool call failed because 'path' was missing. Passing both path and content now.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write_file",
                            "arguments": {"path": "fixed.txt", "content": "hello\n"},
                        }
                    }
                ],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 150,
            "eval_count": 35,
        },
        {
            "message": {
                "role": "assistant",
                "content": "Recovered and wrote fixed.txt.",
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 190,
            "eval_count": 12,
        },
    ]

    client = WorkforceClient(base_url="http://127.0.0.1:11434", model="gemma4:12b")
    monkeypatch.setattr(client, "get_model_digest", lambda: "sha256:gemma4")
    monkeypatch.setattr(client, "_post_json", lambda url, payload, **kw: responses.pop(0))

    worker = AutonomousWorker(client=client)
    res = worker.execute_subtask(
        SubTask(id="t05", title="Recover write", affected_files=["fixed.txt"]),
        tool_suite,
    )

    assert res.success is True
    assert (tmp_path / "fixed.txt").read_text(encoding="utf-8") == "hello\n"
    assert res.metadata["invalid_tool_arguments"] == 1


def test_t06_truncated_generation_at_num_predict_is_never_false_success(
    monkeypatch: pytest.MonkeyPatch, tool_suite: WorkspaceToolSuite, tmp_path: Path
):
    """T06: Generation hitting num_predict (`done_reason == 'length'`) must NEVER be treated as task completion."""
    monkeypatch.setenv("OLLAMA_THINK", "true")
    monkeypatch.setenv("OLLAMA_NUM_PREDICT", "2048")

    # Case A: Model truncates on turn 1, gets nudged, then emits tool call on turn 2 and completes on turn 3
    responses_recover = [
        {
            "message": {
                "role": "assistant",
                "content": "I am going to write the file now...",
                "thinking": "Long reasoning that consumes the entire budget..." * 20,
            },
            "done": True,
            "done_reason": "length",
            "prompt_eval_count": 200,
            "eval_count": 2048,
        },
        {
            "message": {
                "role": "assistant",
                "content": "",
                "thinking": "Concise reasoning: write out.txt.",
                "tool_calls": [
                    {"function": {"name": "write_file", "arguments": {"path": "out.txt", "content": "verified\n"}}}
                ],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 260,
            "eval_count": 45,
        },
        {
            "message": {
                "role": "assistant",
                "content": "Completed out.txt after concise retry.",
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 300,
            "eval_count": 15,
        },
    ]

    client = WorkforceClient(base_url="http://127.0.0.1:11434", model="gemma4:12b")
    monkeypatch.setattr(client, "get_model_digest", lambda: "sha256:gemma4")
    monkeypatch.setattr(client, "_post_json", lambda url, payload, **kw: responses_recover.pop(0))

    worker = AutonomousWorker(client=client)
    res_recover = worker.execute_subtask(
        SubTask(id="t06_recover", title="Recover from truncation", affected_files=["out.txt"]),
        tool_suite,
    )
    assert res_recover.success is True
    assert res_recover.turns == 3
    assert res_recover.metadata["truncated_generations"] == 1
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "verified\n"

    # Case B: Model repeatedly truncates at num_predict on every turn -> MUST fail (success=False)
    always_truncated = {
        "message": {
            "role": "assistant",
            "content": "Partial answer before cutoff",
            "thinking": "Exhausted token budget...",
        },
        "done": True,
        "done_reason": "length",
        "prompt_eval_count": 200,
        "eval_count": 2048,
    }
    client_fail = WorkforceClient(base_url="http://127.0.0.1:11434", model="gemma4:12b")
    monkeypatch.setattr(client_fail, "get_model_digest", lambda: "sha256:gemma4")
    monkeypatch.setattr(client_fail, "_post_json", lambda url, payload, **kw: dict(always_truncated))

    worker_fail = AutonomousWorker(client=client_fail)
    res_fail = worker_fail.execute_subtask(
        SubTask(id="t06_fail", title="Persistent truncation"),
        tool_suite,
    )
    assert res_fail.success is False
    assert res_fail.metadata["truncated_generations"] == 3
    assert res_fail.metadata["truncated"] is True
    assert "truncated" in (res_fail.error or "").lower()


def test_t07_history_integrity_after_context_compaction():
    """T07: Context compaction preserves assistant->tool pairing and strips old thinking traces."""
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "System instructions"},
        {"role": "user", "content": "Execute multi-step R&D task"},
    ]
    for i in range(12):
        messages.append({
            "role": "assistant",
            "content": f"Step {i} note " * 20,
            "thinking": f"Detailed internal thinking for step {i} " * 40,
            "tool_calls": [
                {"function": {"name": "read_file", "arguments": {"path": f"file_{i}.py"}}}
            ],
        })
        messages.append({
            "role": "tool",
            "tool_name": "read_file",
            "content": f"Content of file_{i}.py\n" * 60,
        })

    compact_messages(messages, keep_recent_tools=3, max_bytes=6000)
    raw_bytes = len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
    assert raw_bytes <= 6000
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    # Every `tool` message must immediately follow an `assistant` (with tool_calls) or sibling `tool` message
    for idx, msg in enumerate(messages):
        if msg["role"] == "tool":
            assert idx > 0
            assert messages[idx - 1]["role"] in ("assistant", "tool")
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            assert idx + 1 < len(messages)
            assert messages[idx + 1]["role"] == "tool"

    # Old assistant turns before the recent window must have their `thinking` stripped
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    if len(assistant_msgs) > 3:
        for old_m in assistant_msgs[:-3]:
            assert "thinking" not in old_m

    # Also verify strip_completed_thinking removes thinking prior to a new user turn
    messages.append({"role": "user", "content": "New user turn after tool sequence"})
    strip_completed_thinking(messages)
    assert all("thinking" not in m for m in messages if m["role"] == "assistant")


def test_t08_recovery_after_timeout_or_interruption(
    monkeypatch: pytest.MonkeyPatch, tool_suite: WorkspaceToolSuite
):
    """T08: Clean error handling on timeout and cooperative pause."""
    client = WorkforceClient(base_url="http://127.0.0.1:11434", model="gemma4:12b")
    monkeypatch.setattr(client, "get_model_digest", lambda: "sha256:gemma4")

    def timeout_post(url: str, payload: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        raise OllamaConnectionError("Ollama request timed out after 60s")

    monkeypatch.setattr(client, "_post_json", timeout_post)
    worker = AutonomousWorker(client=client)
    res = worker.execute_subtask(SubTask(id="t08", title="Timeout test"), tool_suite)

    assert res.success is False
    assert "timed out" in (res.error or "").lower()
    assert res.metadata["model_digest"] == "sha256:gemma4"
    assert "wall_clock_s" in res.metadata


def test_t09_thinking_and_completion_without_tool_call(
    monkeypatch: pytest.MonkeyPatch, tool_suite: WorkspaceToolSuite
):
    """T09: Direct reasoning + final text answer when no tool call is required."""
    monkeypatch.setenv("OLLAMA_THINK", "true")
    client = WorkforceClient(base_url="http://127.0.0.1:11434", model="gemma4:12b")
    monkeypatch.setattr(client, "get_model_digest", lambda: "sha256:gemma4")
    monkeypatch.setattr(
        client,
        "_post_json",
        lambda url, payload, **kw: {
            "message": {
                "role": "assistant",
                "content": "Static analysis complete: no file modifications required.",
                "thinking": "Evaluated task requirements; all criteria already satisfied.",
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 95,
            "eval_count": 28,
        },
    )

    worker = AutonomousWorker(client=client)
    res = worker.execute_subtask(SubTask(id="t09", title="Direct completion"), tool_suite)
    assert res.success is True
    assert res.turns == 1
    assert res.tool_calls_count == 0
    assert res.metadata["turns_with_thinking"] == 1
    assert res.metadata["truncated_generations"] == 0


def test_t10_research_report_builder_and_real_metrics_verification():
    """T10: Deterministic ResearchReportBuilder produces RIE-accepted reports even if LLM hallucinates numbers."""
    orion2_dir = Path(__file__).resolve().parent.parent / "temp" / "project_orion" / "orion2-run"
    if not orion2_dir.is_dir():
        pytest.skip("orion2-run directory not present")

    class HallucinatingLLMClient:
        """Simulates an LLM that attempts to inject unverified numbers into qualitative prose."""

        def chat(self, messages, temperature=0.1, response_format=None):
            assert response_format == INTERPRETATION_JSON_SCHEMA
            return {
                "content": json.dumps({
                    "executive_summary": (
                        "The study achieved 99.9% accuracy across 5000 datasets, while verified evaluation "
                        "included 120 runs."
                    ),
                    "clustering_interpretation": (
                        "Risk-aware clustering outperformed 42 baselines by 17.5% margin."
                    ),
                    "retrieval_scalability_interpretation": (
                        "At largest scale, recall was 0.0078125 whereas unverified heuristic claimed 0.8888."
                    ),
                    "limitations_and_next_steps": (
                        "Future work will test 768 dimensions across 999999 vectors."
                    ),
                })
            }

    builder = ResearchReportBuilder(
        study_dir=orion2_dir,
        profile="full",
        evidence_id="cd6dbb999a4b4a75a15cd864ade65db3",
    )
    result = builder.build_report(client=HallucinatingLLMClient())

    assert result.accepted is True, f"Report failed RIE narrative check: {result.failures}"
    assert result.failures == []
    # Unbacked numbers (99.9%, 5000, 42, 17.5%, 0.8888, 768, 999999) were stripped deterministically
    assert result.unverified_numbers_stripped >= 6
    # Verified numbers from claims.json (120, 180, 0.9234375, 0.240625, 0.0234375, 0.0078125) are preserved
    assert "0.0078125" in result.markdown
    assert "0.9234375" in result.markdown
    assert "120" in result.markdown
    assert "180" in result.markdown
    assert "cd6dbb999a4b4a75a15cd864ade65db3" in result.markdown
    # Independent RIE narrative validator confirms zero failures
    assert validate_narrative(result.markdown, result.claims) == []
    # Because unverified numbers had to be stripped, fallback_used is True and llm_native_completion is False
    assert result.pipeline_verified is True
    assert result.llm_native_completion is False
    assert result.fallback_used is True
    assert result.verification_status == "ACCEPTED"


def test_t11_phase2a_native_vs_fallback_and_trajectory_telemetry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tool_suite: WorkspaceToolSuite
):
    """T11: Verifies Phase 2A seed forwarding, trajectory counters, and native vs fallback report completion."""
    monkeypatch.setenv("OLLAMA_THINK", "false")
    monkeypatch.setenv("OLLAMA_NUM_PREDICT", "4096")
    monkeypatch.setenv("OLLAMA_SEED", "777")

    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
    captured_options: List[Dict[str, Any]] = []
    responses = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "code.py"}}}],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 100,
            "eval_count": 20,
        },
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "code.py"}}}],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 110,
            "eval_count": 20,
        },
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "patch_file",
                            "arguments": {"path": "code.py", "target": "x = 1", "replacement": "x = 2"},
                        }
                    }
                ],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 120,
            "eval_count": 30,
        },
        {
            "message": {
                "role": "assistant",
                "content": "Completed patch.",
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 130,
            "eval_count": 10,
        },
    ]

    client = WorkforceClient(base_url="http://127.0.0.1:11434", model="gemma4:12b")
    monkeypatch.setattr(client, "get_model_digest", lambda: "sha256:gemma4")

    def fake_post(url: str, payload: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        captured_options.append(dict(payload.get("options", {})))
        return responses.pop(0)

    monkeypatch.setattr(client, "_post_json", fake_post)
    worker = AutonomousWorker(client=client)
    res = worker.execute_subtask(SubTask(id="t11", title="Trajectory test", affected_files=["code.py"]), tool_suite)

    assert res.success is True
    assert all(opt.get("seed") == 777 and opt.get("num_predict") == 4096 for opt in captured_options)
    assert res.metadata["read_file_calls"] == 2
    assert res.metadata["unique_files_read"] == 1
    assert res.metadata["repeated_file_reads"] == 1
    assert res.metadata["edit_calls"] == 1
    assert len(res.metadata["turn_trajectory"]) == 4

    orion2_dir = Path(__file__).resolve().parent.parent / "temp" / "project_orion" / "orion2-run"
    if orion2_dir.is_dir():
        class CleanLLMClient:
            def chat(self, messages, temperature=0.1, response_format=None):
                return {
                    "content": json.dumps({
                        "executive_summary": "All study claims passed controller checks.",
                        "clustering_interpretation": "Risk-aware clustering maintained target false discovery control.",
                        "retrieval_scalability_interpretation": "Projection window indexing showed degraded recall at large corpus scale.",
                        "limitations_and_next_steps": "Future work will compare against hierarchical graph baselines.",
                    }),
                    "done_reason": "stop",
                    "truncated": False,
                }

        builder = ResearchReportBuilder(
            study_dir=orion2_dir,
            profile="full",
            evidence_id="cd6dbb999a4b4a75a15cd864ade65db3",
        )
        clean_res = builder.build_report(client=CleanLLMClient())
        assert clean_res.pipeline_verified is True
        assert clean_res.llm_native_completion is True
        assert clean_res.fallback_used is False
        assert clean_res.verification_status == "ACCEPTED"


def test_t12_phase2b_p1_truncation_recovery_and_attempt_limits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tool_suite: WorkspaceToolSuite
):
    """T12: P1 RecoveryController preserves read evidence on truncation and enforces attempt limits."""
    monkeypatch.setenv("OLLAMA_THINK", "false")
    monkeypatch.setenv("OLLAMA_NUM_PREDICT", "2048")
    monkeypatch.setenv("OLLAMA_RECOVERY_POLICY", "P1")

    (tmp_path / "mod.py").write_text("def inc(x):\n    return x - 1\n", encoding="utf-8")
    (tmp_path / "test_mod.py").write_text("from mod import inc\ndef test_inc():\n    assert inc(1) == 2\n", encoding="utf-8")

    sent_messages: List[List[Dict[str, Any]]] = []
    responses = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "mod.py"}}}],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 120,
            "eval_count": 20,
        },
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "test_mod.py"}}}],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 150,
            "eval_count": 20,
        },
        # Turn 3: Runaway truncated prose turn without a tool call
        {
            "message": {
                "role": "assistant",
                "content": "Let me analyze the function carefully and write out the full reasoning..." * 20,
            },
            "done": True,
            "done_reason": "length",
            "prompt_eval_count": 190,
            "eval_count": 2048,
        },
        # Turn 4: After P1 action-oriented recovery prompt, model patches directly without re-reading
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "patch_file",
                            "arguments": {
                                "path": "mod.py",
                                "target_content": "return x - 1",
                                "replacement_content": "return x + 1",
                            },
                        }
                    }
                ],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 230,
            "eval_count": 35,
        },
        # Turn 5: Model runs verification command
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_command",
                            "arguments": {"command": 'python3 -c "import mod; assert mod.inc(1) == 2"'},
                        }
                    }
                ],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 260,
            "eval_count": 25,
        },
        # Turn 6: Final summary
        {
            "message": {
                "role": "assistant",
                "content": "Fixed `inc` in `mod.py` and verified.",
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 280,
            "eval_count": 15,
        },
    ]

    client = WorkforceClient(base_url="http://127.0.0.1:11434", model="gemma4:12b")
    monkeypatch.setattr(client, "get_model_digest", lambda: "sha256:gemma4")

    def fake_post(url: str, payload: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        sent_messages.append(json.loads(json.dumps(payload["messages"])))
        return responses.pop(0)

    monkeypatch.setattr(client, "_post_json", fake_post)
    worker = AutonomousWorker(client=client)
    subtask = SubTask(
        id="t12",
        title="Fix mod.inc",
        input_files=["mod.py", "test_mod.py"],
        affected_files=["mod.py"],
        verification_command='python3 -c "import mod; assert mod.inc(1) == 2"',
    )
    res = worker.execute_subtask(subtask, tool_suite)

    assert res.success is True
    assert res.metadata["recovery_policy"] == "P1"
    assert res.metadata["recovery_attempts"] == 1
    assert res.metadata["recovery_events"][0]["reason"] == "truncated_generation"
    # Verify Turn 4 prompt contained the P1 recovery directive forbidding re-reads
    turn4_user_msg = sent_messages[3][-1]["content"]
    assert "[RECOVERY CONTROLLER (1/3) — truncated_generation]" in turn4_user_msg
    assert "mod.py, test_mod.py" in turn4_user_msg
    assert "Do NOT call `read_file` again" in turn4_user_msg

    # Verify explicit limit on recovery attempts (3 truncations -> abort on 4th)
    always_truncated = [
        {
            "message": {"role": "assistant", "content": "long prose..."},
            "done": True,
            "done_reason": "length",
            "prompt_eval_count": 100,
            "eval_count": 2048,
        }
        for _ in range(5)
    ]
    monkeypatch.setattr(client, "_post_json", lambda u, p, **kw: always_truncated.pop(0))
    res_fail = worker.execute_subtask(subtask, tool_suite)
    assert res_fail.success is False
    assert res_fail.metadata["recovery_attempts"] == 3
    assert any(ev["reason"] == "recovery_limit_exceeded" for ev in res_fail.metadata["recovery_events"])


def test_t13_phase2b_p1_repeated_file_read_detection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tool_suite: WorkspaceToolSuite
):
    """T13: P1 detects repeated file reads on unmodified files and injects stage guidance."""
    monkeypatch.setenv("OLLAMA_THINK", "false")
    monkeypatch.setenv("OLLAMA_RECOVERY_POLICY", "P1")

    (tmp_path / "a.py").write_text("val = 0\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("assert val == 1\n", encoding="utf-8")

    captured_messages: List[List[Dict[str, Any]]] = []
    responses = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "a.py"}}}],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 100,
            "eval_count": 20,
        },
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "b.py"}}}],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 120,
            "eval_count": 20,
        },
        # Re-reads a.py without modifying it
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "a.py"}}}],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 140,
            "eval_count": 20,
        },
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "patch_file",
                            "arguments": {"path": "a.py", "target_content": "val = 0", "replacement_content": "val = 1"},
                        }
                    }
                ],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 160,
            "eval_count": 25,
        },
        {
            "message": {
                "role": "assistant",
                "content": "Patched a.py.",
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 180,
            "eval_count": 10,
        },
    ]

    client = WorkforceClient(base_url="http://127.0.0.1:11434", model="gemma4:12b")
    monkeypatch.setattr(client, "get_model_digest", lambda: "sha256:gemma4")

    def fake_post(url: str, payload: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        captured_messages.append(json.loads(json.dumps(payload["messages"])))
        return responses.pop(0)

    monkeypatch.setattr(client, "_post_json", fake_post)
    worker = AutonomousWorker(client=client)
    res = worker.execute_subtask(
        SubTask(id="t13", title="Patch a.py", input_files=["a.py", "b.py"], affected_files=["a.py"]),
        tool_suite,
    )

    assert res.success is True
    assert res.metadata["repeated_file_reads"] == 1
    assert any(ev["reason"] == "repeated_file_read" for ev in res.metadata["recovery_events"])
    # Check that the tool observation after the repeated read of a.py contains the recovery warning
    tool_obs_after_reread = captured_messages[3][-1]["content"]
    assert "RECOVERY CONTROLLER: `a.py` was already read and unchanged" in tool_obs_after_reread


def test_t14_phase2b_p1_missing_tool_call_before_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tool_suite: WorkspaceToolSuite
):
    """T14: P1 catches premature non-truncated text turn when required verification has not run."""
    monkeypatch.setenv("OLLAMA_THINK", "false")
    monkeypatch.setenv("OLLAMA_RECOVERY_POLICY", "P1")

    (tmp_path / "svc.py").write_text("def status():\n    return 'down'\n", encoding="utf-8")

    responses = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "svc.py"}}}],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 100,
            "eval_count": 20,
        },
        # Turn 2: Model emits prose without a tool call even though verification_command hasn't run!
        {
            "message": {
                "role": "assistant",
                "content": "I see that `status()` returns 'down' instead of 'ok'.",
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 120,
            "eval_count": 25,
        },
        # Turn 3: RecoveryController prompts for patch_file -> model patches svc.py
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "patch_file",
                            "arguments": {
                                "path": "svc.py",
                                "target_content": "return 'down'",
                                "replacement_content": "return 'ok'",
                            },
                        }
                    }
                ],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 150,
            "eval_count": 30,
        },
        # Turn 4: Model runs verification command
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_command",
                            "arguments": {"command": 'python3 -c "import svc; assert svc.status() == \'ok\'"'},
                        }
                    }
                ],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 180,
            "eval_count": 25,
        },
        # Turn 5: Final summary after verification passes
        {
            "message": {
                "role": "assistant",
                "content": "Updated `status()` to return 'ok' and verified.",
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 200,
            "eval_count": 15,
        },
    ]

    client = WorkforceClient(base_url="http://127.0.0.1:11434", model="gemma4:12b")
    monkeypatch.setattr(client, "get_model_digest", lambda: "sha256:gemma4")
    monkeypatch.setattr(client, "_post_json", lambda u, p, **kw: responses.pop(0))

    worker = AutonomousWorker(client=client)
    res = worker.execute_subtask(
        SubTask(
            id="t14",
            title="Fix svc.status",
            input_files=["svc.py"],
            affected_files=["svc.py"],
            verification_command='python3 -c "import svc; assert svc.status() == \'ok\'"',
        ),
        tool_suite,
    )

    assert res.success is True
    assert res.turns == 5
    assert res.metadata["successful_verifications"] == 1
    assert any(
        ev["reason"] == "missing_tool_call_before_verification"
        for ev in res.metadata["recovery_events"]
    )


def test_t15_phase2b_p1_failed_test_recovery_preserves_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tool_suite: WorkspaceToolSuite
):
    """T15: P1 preserves failed verification command evidence across recovery turns."""
    monkeypatch.setenv("OLLAMA_THINK", "false")
    monkeypatch.setenv("OLLAMA_RECOVERY_POLICY", "P1")

    (tmp_path / "math_util.py").write_text("def double(n):\n    return n + 1\n", encoding="utf-8")
    sent_payloads: List[List[Dict[str, Any]]] = []

    responses = [
        # Turn 1: Read file
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "math_util.py"}}}],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 100,
            "eval_count": 20,
        },
        # Turn 2: Run verification command (fails because bug not fixed yet)
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_command",
                            "arguments": {"command": 'python3 -c "import math_util; assert math_util.double(3) == 6, \'got_wrong_double\'"'},
                        }
                    }
                ],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 130,
            "eval_count": 25,
        },
        # Turn 3: Model truncates while ruminating on the test failure
        {
            "message": {
                "role": "assistant",
                "content": "Analyzing test failure got_wrong_double..." * 30,
            },
            "done": True,
            "done_reason": "length",
            "prompt_eval_count": 160,
            "eval_count": 2048,
        },
        # Turn 4: Model patches file
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "patch_file",
                            "arguments": {
                                "path": "math_util.py",
                                "target_content": "return n + 1",
                                "replacement_content": "return 2 * n  # fixed",
                            },
                        }
                    }
                ],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 200,
            "eval_count": 30,
        },
        # Turn 5: Re-runs verification command (now passes)
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_command",
                            "arguments": {"command": 'python3 -c "import math_util; assert math_util.double(3) == 6, \'got_wrong_double\'"'},
                        }
                    }
                ],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 230,
            "eval_count": 25,
        },
        # Turn 6: Final summary
        {
            "message": {
                "role": "assistant",
                "content": "Fixed `double(n)` to return `n * 2` and verified.",
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 250,
            "eval_count": 15,
        },
    ]

    client = WorkforceClient(base_url="http://127.0.0.1:11434", model="gemma4:12b")
    monkeypatch.setattr(client, "get_model_digest", lambda: "sha256:gemma4")

    def fake_post(url: str, payload: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        sent_payloads.append(json.loads(json.dumps(payload["messages"])))
        return responses.pop(0)

    monkeypatch.setattr(client, "_post_json", fake_post)
    worker = AutonomousWorker(client=client)
    res = worker.execute_subtask(
        SubTask(
            id="t15",
            title="Fix double",
            input_files=["math_util.py"],
            affected_files=["math_util.py"],
            verification_command='python3 -c "import math_util; assert math_util.double(3) == 6, \'got_wrong_double\'"',
        ),
        tool_suite,
    )

    assert res.success is True
    assert any(ev["reason"] == "failed_verification_command" for ev in res.metadata["recovery_events"])
    assert any(ev["reason"] == "truncated_generation" for ev in res.metadata["recovery_events"])
    # Check that Turn 4 recovery prompt preserved the failed verification status
    turn4_prompt = sent_payloads[3][-1]["content"]
    assert "FAILED" in turn4_prompt
    assert "got_wrong_double" in turn4_prompt


def test_t16_phase2b_loop_detection_and_p0_backward_compatibility(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tool_suite: WorkspaceToolSuite
):
    """T16: Verifies unproductive loop detection under P1 and backward-compatible P0 baseline."""
    monkeypatch.setenv("OLLAMA_THINK", "false")
    monkeypatch.setenv("OLLAMA_RECOVERY_POLICY", "P1")

    (tmp_path / "loop_mod.py").write_text("x = 1\n", encoding="utf-8")

    # Unproductive loop: model repeatedly reads the same file with identical output
    loop_responses = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "loop_mod.py"}}}],
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 100,
            "eval_count": 20,
        }
        for _ in range(5)
    ]

    client = WorkforceClient(base_url="http://127.0.0.1:11434", model="gemma4:12b")
    monkeypatch.setattr(client, "get_model_digest", lambda: "sha256:gemma4")
    monkeypatch.setattr(client, "_post_json", lambda u, p, **kw: loop_responses.pop(0))

    worker = AutonomousWorker(client=client)
    res_loop = worker.execute_subtask(
        SubTask(id="t16_loop", title="Loop test", input_files=["loop_mod.py"]),
        tool_suite,
    )
    assert res_loop.success is False
    assert res_loop.metadata.get("stalled") is True
    assert "cycle" in (res_loop.error or "").lower()

    # P0 baseline check: does not trigger missing_tool_call_before_verification on non-truncated prose
    monkeypatch.setenv("OLLAMA_RECOVERY_POLICY", "P0")
    p0_responses = [
        {
            "message": {
                "role": "assistant",
                "content": "Completed directly in P0.",
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 80,
            "eval_count": 10,
        }
    ]
    monkeypatch.setattr(client, "_post_json", lambda u, p, **kw: p0_responses.pop(0))
    res_p0 = worker.execute_subtask(
        SubTask(
            id="t16_p0",
            title="P0 baseline check",
            verification_command='python3 -c "assert True"',
        ),
        tool_suite,
    )
    assert res_p0.success is True
    assert res_p0.metadata["recovery_policy"] == "P0"
    assert res_p0.metadata["recovery_attempts"] == 0


def test_t17_phase2b_review_bugfixes_and_policy_ablation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tool_suite: WorkspaceToolSuite
):
    """T17: Verifies the 3 review bugfixes and P1_REACTIVE vs P1_STAGE vs P1 policy separation."""
    from antigravity_hybrid_orchestrator.workforce.recovery_controller import RecoveryController

    # 1. Bugfix A: non-verification commands (`ls -R`, `python3 -c "import sys..."`) must NOT mark verification_passed
    rc = RecoveryController(
        subtask_id="sub1",
        input_files=["x.py"],
        affected_files=["x.py"],
        verification_command="python3 -m unittest -q",
        policy="P1",
    )
    rc.record_tool_execution(
        turn=1, tool_name="run_command", args={"command": "ls -R"}, success=True, output="x.py", error=None
    )
    assert rc.verification_passed is False
    assert rc.verification_runs == 0
    rc.record_tool_execution(
        turn=2,
        tool_name="run_command",
        args={"command": 'python3 -c "import sys; print(sys.path)"'},
        success=True,
        output="[]",
        error=None,
    )
    assert rc.verification_passed is False
    assert rc.verification_runs == 0
    for spoof_cmd in (
        "python3 -m unittest -q 2>&1 | tail -20",
        "python3 -m unittest -q || true",
        'python3 -c "assert True"',
        "python3 -m unittest test_x.py",
    ):
        rc.record_tool_execution(
            turn=2,
            tool_name="run_command",
            args={"command": spoof_cmd},
            success=True,
            output="OK",
            error=None,
        )
        assert rc.verification_passed is False, f"spoof_cmd should not pass: {spoof_cmd}"
        assert rc.verification_runs == 0
    rc.record_tool_execution(
        turn=3,
        tool_name="run_command",
        args={"command": "python3 -m unittest -q"},
        success=True,
        output="OK",
        error=None,
    )
    assert rc.verification_passed is True
    assert rc.verification_runs == 1

    # Compound verification_command containing `&&` must be recognized on exact match
    rc_compound = RecoveryController(
        subtask_id="sub_compound",
        verification_command="cd . && python3 -m unittest -q",
        policy="P1",
    )
    rc_compound.record_tool_execution(
        turn=1,
        tool_name="run_command",
        args={"command": "cd . && python3 -m unittest -q"},
        success=True,
        output="OK",
        error=None,
    )
    assert rc_compound.verification_passed is True
    assert rc_compound.verification_runs == 1

    # 2. Bugfix B: re-reading a file AFTER `patch_file` must NOT be flagged as `repeated_file_read`
    rc2 = RecoveryController(
        subtask_id="sub2",
        input_files=["x.py"],
        affected_files=["x.py"],
        verification_command="python3 -m unittest -q",
        policy="P1",
    )
    rc2.record_tool_execution(
        turn=1, tool_name="read_file", args={"path": "x.py"}, success=True, output="a = 1", error=None
    )
    assert rc2.repeated_reads_count == 0
    rc2.record_tool_execution(
        turn=2,
        tool_name="patch_file",
        args={"path": "x.py", "target_content": "a = 1", "replacement_content": "a = 2"},
        success=True,
        output="Patched",
        error=None,
    )
    assert "x.py" not in rc2.clean_read_files
    note_after_post_edit_read = rc2.record_tool_execution(
        turn=3, tool_name="read_file", args={"path": "x.py"}, success=True, output="a = 2", error=None
    )
    assert rc2.repeated_reads_count == 0
    assert "was already read and unchanged" not in (note_after_post_edit_read or "")

    # 3. Bugfix C: P0 resets consecutive_non_tool_turns after a valid tool call
    rc_p0 = RecoveryController(subtask_id="sub_p0", policy="P0")
    cont1, _, _ = rc_p0.handle_non_tool_turn(
        turn=1, content="...", eval_tokens=2048, is_truncated=True, active_num_predict=2048
    )
    assert cont1 is True and rc_p0.consecutive_non_tool_turns == 1
    rc_p0.record_tool_execution(
        turn=2, tool_name="read_file", args={"path": "x.py"}, success=True, output="a = 1", error=None
    )
    assert rc_p0.consecutive_non_tool_turns == 0
    cont2, _, _ = rc_p0.handle_non_tool_turn(
        turn=3, content="...", eval_tokens=2048, is_truncated=True, active_num_predict=2048
    )
    cont3, _, _ = rc_p0.handle_non_tool_turn(
        turn=4, content="...", eval_tokens=2048, is_truncated=True, active_num_predict=2048
    )
    assert cont2 is True and cont3 is True

    # 4. Policy ablation flags: P1_REACTIVE vs P1_STAGE vs P1
    rc_react = RecoveryController(subtask_id="r", policy="P1_REACTIVE")
    assert rc_react.is_reactive is True and rc_react.is_stage_hints is False
    assert (
        rc_react.record_tool_execution(
            turn=1, tool_name="read_file", args={"path": "x.py"}, success=True, output="a=1", error=None
        )
        is None
    )

    rc_stage = RecoveryController(subtask_id="s", policy="P1_STAGE", input_files=["x.py"])
    assert rc_stage.is_reactive is False and rc_stage.is_stage_hints is True
    stage_msg = rc_stage.record_tool_execution(
        turn=1, tool_name="read_file", args={"path": "x.py"}, success=True, output="a=1", error=None
    )
    assert stage_msg is not None and stage_msg.startswith("[STAGE STATE:")






def test_t18_phase2b_trusted_verification_and_error_tail():
    """T18: verification state comes from a controller-run verification command,
    not from string-matching the model's command; error snippets keep the tail."""
    from antigravity_hybrid_orchestrator.workforce.recovery_controller import RecoveryController

    suite_state = {"passes": True, "output": "OK"}
    calls: List[str] = []

    def verify_fn():
        calls.append("verify")
        return suite_state["passes"], suite_state["output"]

    rc = RecoveryController(
        subtask_id="t18",
        input_files=["x.py"],
        affected_files=["x.py"],
        verification_command="python3 -m unittest -q",
        policy="P1_STAGE",
        verify_fn=verify_fn,
    )
    rc.record_tool_execution(
        turn=0, tool_name="read_file", args={"path": "x.py"}, success=True, output="a = 1", error=None
    )
    rc.record_tool_execution(
        turn=0, tool_name="patch_file", args={"path": "x.py"}, success=True, output="Patched", error=None
    )

    # Model runs the suite with an explicit test file: the trusted run decides.
    note = rc.record_tool_execution(
        turn=1, tool_name="run_command",
        args={"command": "python3 -m unittest -q test_x.py"},
        success=True, output="OK", error=None,
    )
    assert calls == ["verify"]
    assert rc.verification_passed is True
    assert rc.verification_runs == 1
    assert "PASSED" in (note or "")

    # A spoofed success cannot pass when the trusted suite fails.
    header = "=" * 70 + "\nFAIL: test_rie_gate (test_x.TestX)\n" + "-" * 70 + "\n"
    traceback = "Traceback (most recent call last):\n" + '  File "test_x.py", line 9\n' * 20
    suite_state.update(passes=False, output=header + traceback + "AssertionError: ['H1', 'H2'] != ['H1']\n")
    rc.record_tool_execution(
        turn=2, tool_name="run_command",
        args={"command": 'python3 -c "assert True"'},
        success=True, output="", error=None,
    )
    assert rc.verification_passed is False
    assert rc.last_verification_error.rstrip().endswith("AssertionError: ['H1', 'H2'] != ['H1']")
    assert "AssertionError: ['H1', 'H2'] != ['H1']" in rc.recommended_next_action()

    # Non-command tools never trigger the trusted run.
    rc.record_tool_execution(
        turn=3, tool_name="read_file", args={"path": "x.py"}, success=True, output="a = 1", error=None
    )
    assert calls == ["verify", "verify"]


def test_t19_phase2b_worker_trusted_verification_and_tool_error_trajectory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tool_suite: WorkspaceToolSuite
):
    """T19: the worker counts a filename-qualified test run as verification via the
    trusted run, and records failed tool error text in the turn trajectory."""
    monkeypatch.setenv("OLLAMA_THINK", "false")
    monkeypatch.setenv("OLLAMA_RECOVERY_POLICY", "P1_STAGE")
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (tmp_path / "test_calc.py").write_text(
        "import unittest\nfrom calc import add\n\n"
        "class T(unittest.TestCase):\n"
        "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n",
        encoding="utf-8",
    )

    def call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "message": {"role": "assistant", "content": "",
                        "tool_calls": [{"function": {"name": name, "arguments": arguments}}]},
            "done": True, "done_reason": "stop", "prompt_eval_count": 100, "eval_count": 20,
        }

    responses = [
        call("read_file", {"path": "calc.py"}),
        call("read_file", {"path": "test_calc.py"}),
        call("patch_file", {"path": "calc.py", "target_content": "return a * b",
                            "replacement_content": "return a + b"}),
        call("patch_file", {"path": "calc.py", "target_content": "return a - b",
                            "replacement_content": "return a + b"}),
        call("run_command", {"command": "python3 -m unittest -q test_calc.py"}),
        {"message": {"role": "assistant", "content": "Fixed add and verified."},
         "done": True, "done_reason": "stop", "prompt_eval_count": 100, "eval_count": 10},
    ]
    sent_payloads: List[List[Dict[str, Any]]] = []
    client = WorkforceClient(base_url="http://127.0.0.1:11434", model="gemma4:12b")
    monkeypatch.setattr(client, "get_model_digest", lambda: "sha256:gemma4")

    def fake_post(url: str, payload: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        sent_payloads.append(json.loads(json.dumps(payload["messages"])))
        return responses.pop(0)

    monkeypatch.setattr(client, "_post_json", fake_post)
    res = AutonomousWorker(client=client).execute_subtask(
        SubTask(id="t19", title="Fix add", input_files=["calc.py", "test_calc.py"],
                affected_files=["calc.py"], verification_command="python3 -m unittest -q"),
        tool_suite,
    )

    assert res.success is True
    assert res.metadata["successful_verifications"] == 1
    failed_patch = res.metadata["turn_trajectory"][2]["tool_calls"][0]
    assert failed_patch["success"] is False
    assert failed_patch["error"]
    assert "PASSED" in sent_payloads[5][-1]["content"]
