"""Tests for the native Ollama tool-calling path of AutonomousWorker (fork).

These run fully offline via a scripted fake chat client, so they never touch a
real Ollama server. The legacy ReAct path stays covered by test_autonomous_worker.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from antigravity_hybrid_orchestrator.planner.dag import SubTask
from antigravity_hybrid_orchestrator.workforce.ollama_client import WorkforceClient
from antigravity_hybrid_orchestrator.workforce.tools import WorkspaceToolSuite
from antigravity_hybrid_orchestrator.workforce.worker import AutonomousWorker


@pytest.fixture
def tool_suite(tmp_path: Path) -> WorkspaceToolSuite:
    return WorkspaceToolSuite(workspace_root=tmp_path)


class ScriptedChatClient:
    """Fake chat client returning a fixed sequence of {content, tool_calls, tokens}."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    def chat(self, messages, tools=None, temperature=0.2):
        self.calls += 1
        if self._script:
            step = self._script.pop(0)
        else:
            step = {"content": "done", "tool_calls": []}
        step.setdefault("input_tokens", 10)
        step.setdefault("output_tokens", 5)
        return step


def test_native_capability_routing():
    assert AutonomousWorker(client=None)._is_native_capable() is True
    assert AutonomousWorker(client=ScriptedChatClient([]))._is_native_capable() is True

    class TextOnly:
        def generate(self, prompt, system_prompt=None):
            return "text"

    assert AutonomousWorker(client=TextOnly())._is_native_capable() is False


def test_native_worker_writes_file_and_completes(tool_suite: WorkspaceToolSuite, tmp_path: Path):
    script = [
        {"content": "", "tool_calls": [
            {"name": "write_file", "arguments": {"path": "out.py", "content": "X = 1\n"}}
        ]},
        {"content": "Done: created out.py", "tool_calls": []},
    ]
    events = []
    worker = AutonomousWorker(
        client=ScriptedChatClient(script), max_turns=5,
        event_callback=lambda kind, subtask_id, payload: events.append((kind, payload)),
    )
    task = SubTask(id="t1", title="Create out.py", affected_files=["out.py"])

    res = worker.execute_subtask(task, tool_suite)

    assert res.success is True
    assert res.tool_calls_count == 1
    assert (tmp_path / "out.py").read_text() == "X = 1\n"
    assert res.metadata.get("backend") == "ollama_native_tools"
    usage = [payload for kind, payload in events if kind == "model_usage"]
    assert [item["output_tokens"] for item in usage] == [5, 5]
    assert all(item["input_tokens"] == 10 for item in usage)


def test_native_worker_prompt_requires_exact_paths_and_complete_write_arguments(tool_suite):
    class CapturesPrompt:
        def __init__(self):
            self.messages = None

        def chat(self, messages, tools=None, temperature=0.2):
            self.messages = messages
            return {"content": "No changes needed.", "tool_calls": []}

    client = CapturesPrompt()
    AutonomousWorker(client=client).execute_subtask(
        SubTask(
            id="analysis", title="Inspect project", input_files=["README.md"],
            outputs=["docs/report.md"], description="Static analysis only.",
        ),
        tool_suite,
    )

    system_prompt = client.messages[0]["content"]
    assert "Never invent source paths" in system_prompt
    assert "Never call write_file with content alone" in system_prompt
    assert "Do not install packages" in system_prompt


def test_native_worker_runs_command_tool(tool_suite: WorkspaceToolSuite):
    script = [
        {"content": "", "tool_calls": [
            {"name": "run_command", "arguments": {"command": "echo hi"}}
        ]},
        {"content": "finished", "tool_calls": []},
    ]
    worker = AutonomousWorker(client=ScriptedChatClient(script), max_turns=5)
    res = worker.execute_subtask(SubTask(id="t1", title="run"), tool_suite)
    assert res.success is True
    assert res.tool_calls_count == 1


def test_native_worker_has_no_fixed_turn_ceiling(tool_suite: WorkspaceToolSuite):
    class CompletesAfterTwelveTurns:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None, temperature=0.2):
            self.calls += 1
            if self.calls <= 12:
                return {"content": "", "tool_calls": [{
                    "name": "run_command", "arguments": {"command": f"echo {self.calls}"}
                }]}
            return {"content": "verified", "tool_calls": []}

    client = CompletesAfterTwelveTurns()
    result = AutonomousWorker(client=client).execute_subtask(
        SubTask(id="long", title="Long task"), tool_suite
    )

    assert result.success is True
    assert result.turns == 13
    assert client.calls == 13


def test_native_worker_honors_cooperative_cancel_between_turns(tool_suite: WorkspaceToolSuite):
    from antigravity_hybrid_orchestrator.engine.control import ExecutionControl

    class CancelsAfterOneCall:
        def __init__(self):
            self.calls = 0
            self.control = None

        def chat(self, messages, tools=None, temperature=0.2):
            self.calls += 1
            self.control.cancel("operator_cancelled")
            return {"content": "", "tool_calls": [{"name": "git_status", "arguments": {}}]}

    client = CancelsAfterOneCall()
    control = ExecutionControl()
    client.control = control
    result = AutonomousWorker(client=client, execution_control=control).execute_subtask(
        SubTask(id="cancel", title="Stop cleanly"), tool_suite
    )

    assert result.success is False
    assert result.metadata["paused"] is True
    assert result.metadata["stop_reason"] == "operator_cancelled"
    assert client.calls == 1


def test_native_worker_stops_repeated_no_progress(tool_suite: WorkspaceToolSuite):
    # Always returns a tool call -> never signals completion.
    class NeverDone:
        def chat(self, messages, tools=None, temperature=0.2):
            return {
                "content": "",
                "tool_calls": [{"name": "git_status", "arguments": {}}],
                "input_tokens": 1,
                "output_tokens": 1,
            }

    worker = AutonomousWorker(client=NeverDone(), max_turns=3)
    res = worker.execute_subtask(SubTask(id="t1", title="loop"), tool_suite)
    assert res.success is False
    assert res.metadata.get("stalled") is True
    assert res.turns >= 3


def test_native_worker_stops_repeated_two_action_cycle(tool_suite: WorkspaceToolSuite):
    class AlternatingClient:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None, temperature=0.2):
            self.calls += 1
            word = "alpha" if self.calls % 2 else "beta"
            return {"content": "", "tool_calls": [{
                "name": "run_command", "arguments": {"command": f"echo {word}"}
            }]}

    client = AlternatingClient()
    result = AutonomousWorker(client=client).execute_subtask(
        SubTask(id="cycle", title="Cycle"), tool_suite
    )

    assert result.success is False
    assert result.metadata.get("stalled") is True
    assert "cycle length 2" in (result.error or "")
    assert client.calls == 6


def test_native_worker_stops_repeated_missing_write_path_even_when_content_changes(
    tool_suite: WorkspaceToolSuite, tmp_path: Path
):
    (tmp_path / "source.md").write_text("same evidence")

    class MalformedWrites:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None, temperature=0.2):
            self.calls += 1
            if self.calls % 2:
                return {"content": "", "tool_calls": [{
                    "name": "read_file", "arguments": {"path": "source.md"},
                }]}
            return {"content": "", "tool_calls": [{
                "name": "write_file", "arguments": {"content": f"draft {self.calls}"},
            }]}

    client = MalformedWrites()
    result = AutonomousWorker(client=client).execute_subtask(
        SubTask(id="report", title="Create report", outputs=["docs/report.md"]), tool_suite
    )

    assert result.success is False
    assert result.metadata.get("stalled") is True
    assert result.tool_calls_count == 6
    assert not (tmp_path / "docs/report.md").exists()


def test_native_worker_handles_chat_exception(tool_suite: WorkspaceToolSuite):
    class Boom:
        def chat(self, messages, tools=None, temperature=0.2):
            raise RuntimeError("ollama down")

    worker = AutonomousWorker(client=Boom(), max_turns=3)
    res = worker.execute_subtask(SubTask(id="t1", title="x"), tool_suite)
    assert res.success is False
    assert "ollama down" in (res.error or "")


def test_native_worker_tool_exception_is_captured(tool_suite: WorkspaceToolSuite):
    # write_file with a bad path type -> tool_suite.execute raises -> captured, loop continues.
    script = [
        {"content": "", "tool_calls": [
            {"name": "nonexistent_tool", "arguments": {}}
        ]},
        {"content": "giving up cleanly", "tool_calls": []},
    ]
    worker = AutonomousWorker(client=ScriptedChatClient(script), max_turns=5)
    res = worker.execute_subtask(SubTask(id="t1", title="bad tool"), tool_suite)
    # Unknown tool returns a failed ToolResult (not an exception), worker still completes.
    assert res.success is True
    assert res.tool_calls_count == 1


def test_workforce_client_chat_parses_response(monkeypatch):
    client = WorkforceClient(base_url="http://x:11434", model="m")

    def fake_post(url, payload):
        assert url.endswith("/api/chat")
        assert payload["tools"] == [{"type": "function"}]
        return {
            "message": {
                "content": "hi",
                "tool_calls": [
                    {"function": {"name": "write_file", "arguments": {"path": "a", "content": "b"}}}
                ],
            },
            "prompt_eval_count": 42,
            "eval_count": 7,
        }

    monkeypatch.setattr(client, "_post_json", fake_post)
    out = client.chat(
        messages=[{"role": "user", "content": "go"}],
        tools=[{"type": "function"}],
    )
    assert out["content"] == "hi"
    assert out["tool_calls"] == [{"name": "write_file", "arguments": {"path": "a", "content": "b"}}]
    assert out["input_tokens"] == 42
    assert out["output_tokens"] == 7


def test_workforce_client_chat_parses_string_arguments(monkeypatch):
    client = WorkforceClient(base_url="http://x:11434", model="m")

    def fake_post(url, payload):
        return {
            "message": {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "run_command", "arguments": "{\"command\": \"ls\"}"}}
                ],
            },
        }

    monkeypatch.setattr(client, "_post_json", fake_post)
    out = client.chat(messages=[{"role": "user", "content": "go"}])
    assert out["tool_calls"] == [{"name": "run_command", "arguments": {"command": "ls"}}]


def test_native_worker_stops_at_subtask_deadline(tool_suite: WorkspaceToolSuite):
    """A single attempt that keeps writing new files must still end at the subtask deadline."""

    class EndlessWriter:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None, temperature=0.2):
            self.calls += 1
            time.sleep(0.02)
            return {
                "content": "",
                "tool_calls": [{"name": "write_file",
                                "arguments": {"path": f"f{self.calls}.txt", "content": "x"}}],
                "input_tokens": 1,
                "output_tokens": 1,
            }

    client = EndlessWriter()
    res = AutonomousWorker(client=client).execute_subtask(
        SubTask(id="endless", title="Never finishes", affected_files=[]),
        tool_suite,
        deadline=time.monotonic() + 0.3,
    )

    assert res.success is False
    assert res.metadata.get("limit_reached") is True
    assert "time limit" in res.error
    assert 1 <= client.calls < 200
