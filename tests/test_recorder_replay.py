"""Tests for persistent R&D run recording, replay, and CLI integration."""

from __future__ import annotations

import json
import stat

from rich.console import Console

from antigravity_hybrid_orchestrator import cli
from antigravity_hybrid_orchestrator.hud.rd_dashboard import RDDashboard
from antigravity_hybrid_orchestrator.hud.recorder import EventRecorder
from antigravity_hybrid_orchestrator.hud.replay import replay_run
from antigravity_hybrid_orchestrator.planner.dag import ExecutionPlanDAG, SubTask
from antigravity_hybrid_orchestrator.telemetry import TelemetrySummary


def test_recorder_writes_header_events_telemetry_and_footer(tmp_path):
    recorder = EventRecorder(tmp_path, goal="make a package", workspace="/work", mode="mock")
    recorder.emit("tool_call", "build", {"tool": "write_file", "args": {"path": "src/a.py"}})
    recorder.record_telemetry(
        TelemetrySummary(10, 0.1, 20, 0.0, 30, 50.0, 2.0)
    )
    recorder.write_footer(True, True, ["build"], [], 1.25)
    path = recorder.path
    recorder.close()

    assert path is not None and path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["kind"] for record in records] == ["header", "event", "telemetry", "footer"]
    assert records[0]["goal"] == "make a package"
    assert records[1]["payload"]["args"]["path"] == "src/a.py"
    assert records[2]["summary"]["total_tokens"] == 30
    assert records[3]["success"] is True


def test_recorder_creates_a_distinct_file_for_each_run(monkeypatch, tmp_path):
    from antigravity_hybrid_orchestrator.hud import recorder as recorder_module

    monkeypatch.setattr(recorder_module.time, "strftime", lambda *_args: "20260930-120000")
    first = EventRecorder(tmp_path, goal="first")
    second = EventRecorder(tmp_path, goal="second")
    first_path, second_path = first.path, second.path
    first.close()
    second.close()

    assert first_path != second_path
    assert len(list(tmp_path.glob("run-*.jsonl"))) == 2


def test_replay_restores_events_telemetry_and_privacy_state(tmp_path):
    recorder = EventRecorder(tmp_path, goal="demo", workspace="/tmp/demo", mode="mock")
    recorder.emit("subtask_start", "task-1", {"subtask": {"description": "Create file"}})
    recorder.emit("tool_call", "task-1", {"tool_name": "write_file", "args": {"path": "x.py"}})
    recorder.emit("tool_result", "task-1", {"tool_name": "write_file", "success": True})
    recorder.record_telemetry(
        TelemetrySummary(10, 0.1, 20, 0.0, 30, 50.0, 2.0)
    )
    recorder.write_footer(True, False, ["task-1"], [], 1.25)
    path = recorder.path
    recorder.close()
    assert path is not None

    hud = replay_run(str(path), speed=0, no_hud=True)

    assert hud.state.pipeline_status == "COMPLETE"
    assert hud.state.goal == "demo"
    assert hud.state.privacy_verified is False
    assert hud.state.telemetry_summary.total_tokens == 30
    assert hud.state.subtasks[0]["status"] == "running"
    assert hud.state.tool_log[0]["success"] is True


def test_cyberpunk_dashboard_renders_all_main_panels():
    console = Console(record=True, width=140, force_terminal=True, color_system="truecolor")
    hud = RDDashboard(console=console, no_hud=True, goal="inspect project", workspace="/tmp/project")
    hud.set_pipeline_status("PLANNING")
    hud.init_dag(
        [
            {
                "id": "task-1",
                "title": "Inspect",
                "description": "Read project files",
                "status": "running",
                "dependencies": [],
            }
        ]
    )
    hud.on_tool_call("read_file", {"path": "README.md"})
    hud.update_telemetry(TelemetrySummary(10, 0.1, 20, 0.0, 30, 50.0, 2.0))

    hud._render_into_layout()
    console.print(hud.layout)
    rendered = console.export_text()
    assert "ANTIGRAVITY" in rendered
    assert "EXECUTION DAG" in rendered
    assert "WORKER NEURAL STREAM" in rendered
    assert "TOOL & VERIFY STREAM" in rendered
    assert "ZERO-KNOWLEDGE" in rendered


def test_compact_dashboard_keeps_header_and_privacy_footer_readable():
    console = Console(record=True, width=80, height=50, force_terminal=True)
    hud = RDDashboard(console=console, no_hud=True, goal="inspect project", workspace="/tmp/project")
    hud.set_pipeline_status("EXECUTING")
    hud._render_into_layout()

    console.print(hud.layout)
    rendered = console.export_text()
    assert "EXECUTING" in rendered
    assert "0 BYTES LEAKED TO CLOUD" in rendered
    assert "deterministic-verify" in rendered
    assert "git-rollback" in rendered


def test_cli_replay_flag_routes_to_replay(monkeypatch, tmp_path):
    run_file = tmp_path / "run.jsonl"
    run_file.write_text("{}\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(cli, "replay_run", lambda *args, **kwargs: calls.append((args, kwargs)))

    result = cli.main(["--replay", str(run_file), "--no-hud", "--replay-speed", "12"])

    assert result == 0
    assert calls == [((str(run_file),), {"speed": 12.0, "no_hud": True})]


def test_cli_no_record_disables_coordinator_recording(monkeypatch, tmp_path):
    captured = {}

    class StubCoordinator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            return 0

    monkeypatch.setattr(cli, "HybridCoordinator", StubCoordinator)

    result = cli.main(
        ["--goal", "demo", "--workspace", str(tmp_path), "--mode", "mock", "--no-record"]
    )

    assert result == 0
    assert captured["record"] is False


def test_no_record_coordinator_run_does_not_create_recording(tmp_path):
    from antigravity_hybrid_orchestrator.architect import MockCloudArchitectClient
    from antigravity_hybrid_orchestrator.coordinator import HybridCoordinator

    dag = ExecutionPlanDAG()
    dag.add_task(
        SubTask(
            id="single",
            title="No-op",
            description="Verify the no-record path.",
            dependencies=[],
            verification_command="true",
        )
    )
    dag.validate()
    coordinator = HybridCoordinator(
        mode="mock",
        goal="check recording toggle",
        workspace=str(tmp_path),
        no_hud=True,
        record=False,
    )
    coordinator.architect = MockCloudArchitectClient(custom_plan=dag)

    assert coordinator.run() == 0
    assert not (tmp_path / ".antigravity_runs").exists()


def test_cli_replay_missing_file_returns_usage_error(capsys, tmp_path):
    result = cli.main(["--replay", str(tmp_path / "missing.jsonl")])

    assert result == 2
    assert "replay file not found" in capsys.readouterr().err.lower()
