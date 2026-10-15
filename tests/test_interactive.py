"""Tests for the interactive R&D command loop."""

from __future__ import annotations

from antigravity_hybrid_orchestrator import cli


def test_interactive_runs_multiple_tasks_and_reuses_session_options(monkeypatch):
    inputs = iter(["add a parser", "/help", "write its tests", "/quit"])
    calls = []

    class StubCoordinator:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def run(self):
            return 0

    monkeypatch.setattr(cli, "HybridCoordinator", StubCoordinator)

    result = cli.run_interactive(
        {"mode": "mock", "workspace": "/tmp/work", "retry_budget": 2},
        input_fn=lambda _prompt: next(inputs),
    )

    assert result == 0
    assert [call["goal"] for call in calls] == ["add a parser", "write its tests"]
    assert all(call["workspace"] == "/tmp/work" for call in calls)
    assert all(call["mode"] == "mock" and call["retry_budget"] == 2 for call in calls)


def test_interactive_continues_after_failed_task_and_returns_last_status(monkeypatch):
    inputs = iter(["first task", "second task", "/quit"])
    calls = []

    class StubCoordinator:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def run(self):
            return 1 if len(calls) == 1 else 0

    monkeypatch.setattr(cli, "HybridCoordinator", StubCoordinator)

    result = cli.run_interactive({}, input_fn=lambda _prompt: next(inputs))

    assert result == 0
    assert len(calls) == 2


def test_interactive_eof_exits_cleanly_without_running_coordinator(monkeypatch):
    class UnexpectedCoordinator:
        def __init__(self, **kwargs):
            raise AssertionError("no task should be run")

    monkeypatch.setattr(cli, "HybridCoordinator", UnexpectedCoordinator)

    def eof(_prompt):
        raise EOFError

    assert cli.run_interactive({}, input_fn=eof) == 0


def test_cli_interactive_flag_routes_to_interactive_loop(monkeypatch, tmp_path):
    captured = {}

    def fake_interactive(kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "run_interactive", fake_interactive)

    result = cli.main(
        ["--interactive", "--mode", "mock", "--workspace", str(tmp_path), "--retry-budget", "3"]
    )

    assert result == 0
    assert captured["mode"] == "mock"
    assert captured["workspace"] == str(tmp_path)
    assert captured["retry_budget"] == 3
    assert captured["goal"] is None


def test_cli_rejects_goal_combined_with_interactive(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_interactive", lambda _kwargs: 0)

    result = cli.main(["--interactive", "--goal", "ambiguous task"])

    assert result == 2
    assert "cannot be combined" in capsys.readouterr().err.lower()
