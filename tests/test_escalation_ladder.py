"""Escalation ladder: variable generation limit, thinking on retry, cloud guidance."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from antigravity_hybrid_orchestrator.planner.dag import SubTask
from antigravity_hybrid_orchestrator.workforce.ollama_client import WorkforceClient
from antigravity_hybrid_orchestrator.workforce.tools import WorkspaceToolSuite
from antigravity_hybrid_orchestrator.workforce.worker import AutonomousWorker


@pytest.fixture(autouse=True)
def _default_budget_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OLLAMA_NUM_PREDICT", "OLLAMA_NUM_PREDICT_MAX", "OLLAMA_ADAPTIVE_NUM_PREDICT",
                 "OLLAMA_TOOL_RESERVE", "OLLAMA_RECOVERY_POLICY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OLLAMA_THINK", "false")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "16384")


def _call(name: str, arguments: Dict[str, Any], done_reason: str = "stop") -> Dict[str, Any]:
    return {
        "message": {"role": "assistant", "content": "",
                    "tool_calls": [{"function": {"name": name, "arguments": arguments}}]},
        "done": True, "done_reason": done_reason, "prompt_eval_count": 100, "eval_count": 20,
    }


def _text(content: str, done_reason: str = "stop") -> Dict[str, Any]:
    return {"message": {"role": "assistant", "content": content}, "done": True,
            "done_reason": done_reason, "prompt_eval_count": 100, "eval_count": 20}


def _run(monkeypatch, tmp_path: Path, responses: List[Dict[str, Any]], **execute_kwargs):
    """Run the native worker on scripted responses; return (result, sent payloads)."""
    sent: List[Dict[str, Any]] = []
    client = WorkforceClient(base_url="http://127.0.0.1:11434", model="gemma4:12b")
    monkeypatch.setattr(client, "get_model_digest", lambda: "sha256:gemma4")

    def fake_post(url: str, payload: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        sent.append({"think": payload.get("think"), "num_predict": payload["options"]["num_predict"]})
        return responses.pop(0)

    monkeypatch.setattr(client, "_post_json", fake_post)
    result = AutonomousWorker(client=client).execute_subtask(
        SubTask(id="esc", title="Append a line", input_files=[], affected_files=["log.txt"]),
        WorkspaceToolSuite(workspace_root=tmp_path),
        **execute_kwargs,
    )
    return result, sent


APPEND = {"command": "echo hi >> log.txt"}


def test_truncated_tool_call_is_not_executed_and_turn_is_retried_with_larger_limit(monkeypatch, tmp_path):
    result, sent = _run(monkeypatch, tmp_path, [
        _call("run_command", APPEND, done_reason="length"),
        _call("run_command", APPEND),
        _text("Done."),
    ])

    assert (tmp_path / "log.txt").read_text() == "hi\n"
    assert [s["num_predict"] for s in sent] == [2048, 4096, 2048]
    assert result.metadata["budget_escalations"] == 1
    assert result.metadata["discarded_truncated_tool_calls"] == 1
    trajectory = result.metadata["turn_trajectory"]
    assert trajectory[0]["tool_calls"] == [] and trajectory[0]["num_predict"] == 2048
    assert trajectory[1]["budget_escalated"] is True and trajectory[1]["num_predict"] == 4096


def test_truncated_prose_turn_does_not_escalate(monkeypatch, tmp_path):
    result, sent = _run(monkeypatch, tmp_path, [
        _text("thinking out loud " * 50, done_reason="length"),
        _call("run_command", APPEND),
        _text("Done."),
    ])

    assert [s["num_predict"] for s in sent] == [2048, 2048, 2048]
    assert result.metadata["budget_escalations"] == 0


def test_escalation_skipped_when_larger_limit_does_not_fit(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLAMA_NUM_CTX", "4096")
    result, sent = _run(monkeypatch, tmp_path, [
        _call("run_command", APPEND, done_reason="length"),
        _call("run_command", APPEND),
        _text("Done."),
    ])

    assert 4096 not in [s["num_predict"] for s in sent]
    assert result.metadata["budget_escalations"] == 0
    assert result.metadata["turn_trajectory"][0]["budget_escalation_skipped"] == "context"
    assert (tmp_path / "log.txt").read_text() == "hi\n"


def test_adaptive_limit_can_be_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLAMA_ADAPTIVE_NUM_PREDICT", "0")
    result, sent = _run(monkeypatch, tmp_path, [
        _call("run_command", APPEND, done_reason="length"),
        _text("Done."),
    ])

    assert [s["num_predict"] for s in sent] == [2048, 2048]
    assert result.metadata["budget_escalations"] == 0
    # Previous behaviour: the truncated call was executed.
    assert (tmp_path / "log.txt").read_text() == "hi\n"
