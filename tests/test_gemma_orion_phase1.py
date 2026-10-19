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
