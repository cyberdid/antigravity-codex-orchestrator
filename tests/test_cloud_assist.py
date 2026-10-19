"""CloudAssist: limits, verbatim audit, failures that never break the run."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

from antigravity_hybrid_orchestrator.architect.client import CloudArchitectError
from antigravity_hybrid_orchestrator.architect.mock_client import MockCloudArchitectClient
from antigravity_hybrid_orchestrator.engine.control import ExecutionBudget, ExecutionControl
from antigravity_hybrid_orchestrator.planner.dag import SubTask

EXCERPT = "FAIL: test_x (t.T) | AssertionError: 1 != 2"


class RecordingClient(MockCloudArchitectClient):
    def __init__(self, fail: bool = False) -> None:
        super().__init__()
        self.fail = fail
        self.received: List[str] = []

    def assist_subtask(self, payload: str) -> Tuple[str, int, int]:
        self.received.append(payload)
        if self.fail:
            raise CloudArchitectError("agy exited with status 1")
        return super().assist_subtask(payload)


def _assist(tmp_path: Path, client=None, max_calls: int = 3, control=None):
    from antigravity_hybrid_orchestrator.engine.cloud_assist import CloudAssist

    events: List[Tuple[str, str, Dict[str, Any]]] = []
    assist = CloudAssist(client or RecordingClient(), tmp_path, max_calls=max_calls,
                         audit=lambda e, s, p: events.append((e, s, p)),
                         execution_control=control or ExecutionControl())
    return assist, events


def _subtask(task_id: str = "s1", files=("m.py",)) -> SubTask:
    return SubTask(id=task_id, title="Fix m", affected_files=list(files))


def test_mock_client_assist_returns_steps_and_tokens():
    text, tokens_in, tokens_out = MockCloudArchitectClient().assist_subtask("x" * 400)
    assert text.strip().startswith("1.")
    assert (tokens_in, tokens_out) == (100, 50)


def test_request_records_exact_payload_and_hash(tmp_path):
    (tmp_path / "m.py").write_text('def f(x: int) -> int:\n    """Double x."""\n    return x\n')
    client = RecordingClient()
    assist, events = _assist(tmp_path, client)

    guidance = assist.request(_subtask(), EXCERPT)

    assert guidance and guidance.strip().startswith("1.")
    request = next(p for e, _, p in events if e == "cloud_assist_request")
    assert request["payload"] == client.received[0]
    assert request["payload_sha256"] == hashlib.sha256(client.received[0].encode("utf-8")).hexdigest()
    response = next(p for e, _, p in events if e == "cloud_assist_response")
    assert response["response"] == guidance and response["output_tokens"] == 50
    assert assist.calls_made == 1 and assist.tokens_used > 0


def test_one_call_per_subtask_and_run_limit(tmp_path):
    assist, events = _assist(tmp_path, max_calls=1)
    assert assist.request(_subtask("a"), EXCERPT)
    assert assist.request(_subtask("a"), EXCERPT) is None
    assert assist.request(_subtask("b"), EXCERPT) is None
    reasons = [p["reason"] for e, _, p in events if e == "cloud_assist_skipped"]
    assert reasons == ["subtask_limit", "run_limit"]


def test_blocked_payload_never_reaches_client(tmp_path):
    (tmp_path / "m.py").write_text('def f():\n    """ghp_' + "a1B2c3D4e5" * 3 + 'xyzWVU"""\n')
    client = RecordingClient()
    assist, events = _assist(tmp_path, client)
    assert assist.request(_subtask(), EXCERPT) is None
    assert client.received == []
    assert [e for e, _, _ in events] == ["cloud_assist_blocked"]
    assert assist.calls_made == 0


def test_backend_failure_is_reported_not_raised(tmp_path):
    assist, events = _assist(tmp_path, RecordingClient(fail=True))
    assert assist.request(_subtask(), EXCERPT) is None
    failed = [p for e, _, p in events if e == "cloud_assist_failed"]
    assert failed and "status 1" in failed[0]["error"]


def test_remote_token_budget_enforced(tmp_path):
    control = ExecutionControl(ExecutionBudget(max_remote_tokens=10))
    control.record_remote_tokens(10)
    client = RecordingClient()
    assist, events = _assist(tmp_path, client, control=control)
    assert assist.request(_subtask(), EXCERPT) is None
    assert client.received == []
    assert [p["reason"] for e, _, p in events if e == "cloud_assist_skipped"] == ["remote_budget"]


def test_successful_call_records_remote_tokens(tmp_path):
    control = ExecutionControl()
    assist, _ = _assist(tmp_path, control=control)
    assist.request(_subtask(), EXCERPT)
    assert control.remote_tokens == assist.tokens_used > 0
