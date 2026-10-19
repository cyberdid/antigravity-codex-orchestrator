"""Cloud assist is enabled only by the owner's flag on the current command line."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from antigravity_hybrid_orchestrator.cli import build_parser
from antigravity_hybrid_orchestrator.coordinator import HybridCoordinator
from antigravity_hybrid_orchestrator.workforce.worker import SubtaskResult


def _coordinator(tmp_path: Path, **overrides) -> HybridCoordinator:
    kwargs = dict(mode="mock", goal="implement an add function", workspace=str(tmp_path), no_hud=True)
    kwargs.update(overrides)
    return HybridCoordinator(**kwargs)


def _fail_first_attempt(coordinator: HybridCoordinator) -> None:
    original = coordinator.worker.execute_subtask
    state = {"calls": 0}

    def flaky(subtask, tool_suite, feedback_context=None, deadline=None, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            return SubtaskResult(subtask_id=subtask.id, success=False, turns=1,
                                 error="AssertionError: simulated first-attempt failure")
        return original(subtask, tool_suite, feedback_context=feedback_context, deadline=deadline)

    coordinator.worker.execute_subtask = flaky


def _run_log(tmp_path: Path) -> Path:
    return next((tmp_path / ".antigravity_runs").glob("run-*.jsonl"))


def test_cli_defaults_keep_cloud_assist_off():
    args = build_parser().parse_args(["--goal", "x", "--mode", "mock"])
    assert args.cloud_assist == "off"
    assert args.cloud_assist_max_calls == 3
    assert args.think_on_retry is False


def test_cloud_assist_off_by_default(tmp_path):
    coordinator = _coordinator(tmp_path)
    coordinator.run()
    assert coordinator._cloud_assist is None


def test_environment_cannot_enable_cloud_assist(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_ASSIST", "spec")
    monkeypatch.setenv("ANTIGRAVITY_CLOUD_ASSIST", "spec")
    coordinator = _coordinator(tmp_path)
    coordinator.run()
    assert coordinator._cloud_assist is None


def test_assist_events_reach_run_log(tmp_path):
    coordinator = _coordinator(tmp_path, cloud_assist="spec")
    _fail_first_attempt(coordinator)
    assert coordinator.run() == 0

    lines = _run_log(tmp_path).read_text().splitlines()
    requests = [json.loads(line) for line in lines if '"cloud_assist_request"' in line]
    assert len(requests) == 1 and "payload_sha256" in json.dumps(requests[0])
    telemetry = [json.loads(line) for line in lines if '"cloud_assist_calls"' in line]
    summary = json.dumps(telemetry[-1])
    assert '"cloud_assist_calls": 1' in summary
    assert coordinator._cloud_assist.tokens_used > 0


def test_resume_does_not_restore_cloud_assist(tmp_path):
    from antigravity_hybrid_orchestrator.integrity.policy import RunPolicyStore

    first = _coordinator(tmp_path, cloud_assist="spec")
    first.run()
    run_log = _run_log(tmp_path)
    RunPolicyStore.save(run_log, {
        "workspace": str(tmp_path.resolve()), "goal": "implement an add function",
        "integrity_contract": None, "evidence_directory": None, "sandbox_backend": "host",
        "sandbox_image": "antigravity-orion:local", "sandbox_max_file_mb": 16, "study_profile": "full",
        "cloud_assist": "spec", "cloud_assist_max_calls": 99,
    })

    resumed = _coordinator(tmp_path, goal=None, resume_run=str(run_log))
    resumed.run()
    assert resumed._cloud_assist is None

    with_flag = _coordinator(tmp_path, goal=None, resume_run=str(run_log), cloud_assist="spec")
    with_flag.run()
    assert with_flag._cloud_assist is not None
    assert with_flag._cloud_assist.max_calls == 3


def test_invalid_max_calls_rejected():
    from antigravity_hybrid_orchestrator.cli import main

    assert main(["--goal", "x", "--mode", "mock", "--cloud-assist-max-calls", "0"]) == 2
