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


# --- Task 2: thinking on retry ------------------------------------------------

from antigravity_hybrid_orchestrator.workforce.worker import SubtaskResult  # noqa: E402

OK_CHECK = "python3 -c \"import pathlib,sys; sys.exit(0 if pathlib.Path('ok').exists() else 1)\""


class RecordingWorker:
    """Fails the first attempt (writes nothing useful), passes from the second on."""

    def __init__(self, succeed_on: int = 2) -> None:
        self.succeed_on = succeed_on
        self.calls: List[Dict[str, Any]] = []

    def execute_subtask(self, subtask, tool_suite, feedback_context=None, deadline=None, think=None):
        self.calls.append({"think": think, "feedback_context": feedback_context})
        attempt = len(self.calls)
        root = Path(tool_suite.workspace_root)
        (root / f"attempt{attempt}.txt").write_text("x")
        if attempt >= self.succeed_on:
            (root / "ok").write_text("1")
        return SubtaskResult(subtask_id=subtask.id, success=True, turns=1, tool_calls_count=1)


def _loop_subtask() -> SubTask:
    return SubTask(id="loop", title="Create ok", input_files=[], affected_files=["ok"],
                   verification_command=OK_CHECK)


def _make_loop(tmp_path: Path, worker, ladder=None):
    from antigravity_hybrid_orchestrator.engine.loop import SelfCorrectionLoop

    return SelfCorrectionLoop(tool_suite=WorkspaceToolSuite(workspace_root=tmp_path), worker=worker,
                              checkpoint_git=False, ladder=ladder)


def test_ladder_think_for_attempt():
    from antigravity_hybrid_orchestrator.engine.escalation import EscalationLadder

    ladder = EscalationLadder(think_on_retry=True)
    assert ladder.think_for_attempt(0) is None
    assert ladder.think_for_attempt(1) is True
    assert EscalationLadder().think_for_attempt(1) is None


def test_loop_passes_think_override_on_retry(tmp_path):
    from antigravity_hybrid_orchestrator.engine.escalation import EscalationLadder

    worker = RecordingWorker()
    result = _make_loop(tmp_path, worker, EscalationLadder(think_on_retry=True)).execute_subtask_with_retry(
        _loop_subtask())
    assert result.success is True
    assert [c["think"] for c in worker.calls] == [None, True]
    assert result.metadata["think_retry_used"] is True


def test_loop_without_ladder_never_overrides_think(tmp_path):
    worker = RecordingWorker()
    _make_loop(tmp_path, worker).execute_subtask_with_retry(_loop_subtask())
    assert [c["think"] for c in worker.calls] == [None, None]


def test_worker_think_override_reaches_chat(monkeypatch, tmp_path):
    _, sent = _run(monkeypatch, tmp_path, [_text("Done.")], think=True)
    assert sent[0]["think"] is True
    assert sent[0]["num_predict"] == 4096


# --- Task 5: cloud guidance in the loop ---------------------------------------

ASSERT_CHECK = "python3 -c \"import pathlib; assert pathlib.Path('ok').exists(), 'ok file missing'\""


class FakeAssist:
    def __init__(self, guidance: str = "1. Use fullmatch.") -> None:
        self.guidance = guidance
        self.calls: List[Dict[str, Any]] = []

    def request(self, subtask, failure_excerpt):
        self.calls.append({"subtask": subtask.id, "excerpt": failure_excerpt})
        return self.guidance


def _assert_subtask() -> SubTask:
    return SubTask(id="loop", title="Create ok", input_files=[], affected_files=["ok"],
                   verification_command=ASSERT_CHECK)


def test_assist_called_before_due_attempt_with_failure_excerpt(tmp_path):
    from antigravity_hybrid_orchestrator.engine.escalation import EscalationLadder

    worker, assist = RecordingWorker(), FakeAssist()
    result = _make_loop(tmp_path, worker, EscalationLadder(cloud_assist=assist)).execute_subtask_with_retry(
        _assert_subtask())

    assert len(assist.calls) == 1
    assert "AssertionError: ok file missing" in assist.calls[0]["excerpt"]
    second_feedback = worker.calls[1]["feedback_context"]
    assert "Guidance from the cloud architect (steps, not code):" in second_feedback
    assert "1. Use fullmatch." in second_feedback
    assert worker.calls[0]["feedback_context"] is None


def test_assist_not_called_without_ladder_assist(tmp_path):
    from antigravity_hybrid_orchestrator.engine.escalation import EscalationLadder

    result = _make_loop(tmp_path, RecordingWorker(), EscalationLadder()).execute_subtask_with_retry(
        _assert_subtask())
    assert result.metadata["solved_with_cloud_assist"] is False
    assert result.metadata["cloud_assist_calls"] == 0


def test_assisted_success_is_labelled(tmp_path):
    from antigravity_hybrid_orchestrator.engine.escalation import EscalationLadder

    result = _make_loop(tmp_path, RecordingWorker(), EscalationLadder(cloud_assist=FakeAssist())
                        ).execute_subtask_with_retry(_assert_subtask())
    assert result.success is True
    assert result.metadata["solved_with_cloud_assist"] is True
    assert result.metadata["cloud_assist_calls"] == 1
    assert "cloud_assist" in result.metadata["ladder_steps_used"]


def test_assist_returning_none_adds_no_guidance(tmp_path):
    from antigravity_hybrid_orchestrator.engine.escalation import EscalationLadder

    worker = RecordingWorker()
    result = _make_loop(tmp_path, worker, EscalationLadder(cloud_assist=FakeAssist(guidance=None))
                        ).execute_subtask_with_retry(_assert_subtask())
    assert "Guidance from the cloud architect" not in (worker.calls[1]["feedback_context"] or "")
    assert result.metadata["solved_with_cloud_assist"] is False
