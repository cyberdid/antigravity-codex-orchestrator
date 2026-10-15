"""Tests for the Codex CLI cloud backend (fork-specific).

The other suites patch ``CloudArchitectClient._invoke_codex`` out, so these
tests exercise the actual codex invocation glue: command construction, reading
the ``--output-last-message`` file, token-usage parsing, and error handling.
"""

from __future__ import annotations

import subprocess
import sys
import pytest

from antigravity_hybrid_orchestrator.engine.control import ExecutionBudget, ExecutionControl
from antigravity_hybrid_orchestrator.architect.client import (
    CloudArchitectClient,
    CloudArchitectError,
)


def _patch_popen(monkeypatch, *, response="", stdout="", stderr="", returncode=0, captured=None):
    real_popen = subprocess.Popen

    def fake_popen(cmd, **kwargs):
        if captured is not None:
            captured["cmd"] = cmd
        out_idx = cmd.index("--output-last-message") + 1
        script = (
            "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2]); "
            "sys.stdout.write(sys.argv[3]); sys.stderr.write(sys.argv[4]); "
            "sys.exit(int(sys.argv[5]))"
        )
        return real_popen(
            [sys.executable, "-c", script, cmd[out_idx], response, stdout, stderr, str(returncode)],
            **kwargs,
        )

    monkeypatch.setattr(subprocess, "Popen", fake_popen)


def test_invoke_codex_reads_last_message_and_builds_command(monkeypatch):
    client = CloudArchitectClient(codex_path="/usr/bin/codex", model="gpt-x")
    captured = {}

    _patch_popen(
        monkeypatch,
        response='{"threat_model": "x", "subtasks": []}',
        stdout='{"msg": {"type": "task_started"}}\n{"usage": {"input_tokens": 123, "output_tokens": 45}}\n',
        captured=captured,
    )

    raw, in_tok, out_tok = client._invoke_codex("PLAN THIS")

    assert '"threat_model"' in raw
    assert in_tok == 123
    assert out_tok == 45
    # Command shape: codex exec, non-interactive, read-only sandbox, model, prompt.
    cmd = captured["cmd"]
    assert cmd[0] == "/usr/bin/codex"
    assert cmd[1] == "exec"
    assert "--skip-git-repo-check" in cmd
    assert "read-only" in cmd
    assert "-m" in cmd and "gpt-x" in cmd
    assert cmd[-1] == "PLAN THIS"


def test_invoke_codex_estimates_tokens_when_stream_has_none(monkeypatch):
    client = CloudArchitectClient(codex_path="/usr/bin/codex")

    _patch_popen(monkeypatch, response="some response text that is reasonably long", stdout="no json here")
    raw, in_tok, out_tok = client._invoke_codex("a prompt of some length")

    assert raw.startswith("some response")
    assert in_tok >= 1  # estimated from prompt length
    assert out_tok >= 1  # estimated from response length


def test_invoke_codex_raises_on_nonzero_exit(monkeypatch):
    client = CloudArchitectClient(codex_path="/usr/bin/codex")

    _patch_popen(monkeypatch, stderr="codex boom", returncode=1)
    with pytest.raises(CloudArchitectError, match="non-zero exit"):
        client._invoke_codex("x")


def test_invoke_codex_raises_on_empty_message(monkeypatch):
    client = CloudArchitectClient(codex_path="/usr/bin/codex")

    _patch_popen(monkeypatch, response="   ")  # whitespace only
    with pytest.raises(CloudArchitectError, match="empty final message"):
        client._invoke_codex("x")


def test_invoke_codex_missing_binary(monkeypatch):
    client = CloudArchitectClient(codex_path="/nonexistent/codex")

    def fake_popen(cmd, **kwargs):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    with pytest.raises(CloudArchitectError, match="Codex CLI binary not found"):
        client._invoke_codex("x")


def test_default_codex_path_resolution():
    client = CloudArchitectClient()
    assert client.codex_path.endswith("codex")


def test_default_architect_timeout_is_unbounded():
    assert CloudArchitectClient().timeout is None


def test_architect_stops_silent_process_on_model_idle_timeout(monkeypatch):
    real_popen = subprocess.Popen
    client = CloudArchitectClient(
        codex_path="codex",
        execution_control=ExecutionControl(ExecutionBudget(model_idle_timeout_s=0.05)),
    )

    def launch_silent_process(cmd, **kwargs):
        return real_popen([sys.executable, "-c", "import time; time.sleep(2)"], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", launch_silent_process)
    with pytest.raises(CloudArchitectError, match="inactive"):
        client._invoke_codex("plan")


def test_architect_allows_long_call_while_progress_output_continues(monkeypatch):
    real_popen = subprocess.Popen
    client = CloudArchitectClient(
        codex_path="codex",
        execution_control=ExecutionControl(ExecutionBudget(model_idle_timeout_s=0.08)),
    )

    def launch_progress_process(cmd, **kwargs):
        output_path = cmd[cmd.index("--output-last-message") + 1]
        script = (
            "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); "
            "[(print('working',flush=True),time.sleep(.04)) for _ in range(5)]; "
            "p.write_text('{\\\"tasks\\\": []}')"
        )
        return real_popen([sys.executable, "-c", script, output_path], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", launch_progress_process)
    raw, _, _ = client._invoke_codex("plan")

    assert raw == '{"tasks": []}'
