"""Tests for selectable Cloud Architect backends: luna (codex), haiku (Claude CLI), agy.

The CLIs are never executed: subprocess behaviour is replaced at the
``_run_subprocess`` boundary, so these tests run offline and deterministically.
"""

from __future__ import annotations

import json

import pytest

from antigravity_hybrid_orchestrator.architect.client import (
    CloudArchitectClient,
    CloudArchitectError,
)
from antigravity_hybrid_orchestrator.cli import build_parser


def _capture(client, rc=0, stdout="", stderr=""):
    """Patches _run_subprocess to record the command and return a canned result."""
    calls = []

    def fake_run(cmd, label):
        calls.append((list(cmd), label))
        return rc, stdout, stderr

    client._run_subprocess = fake_run  # type: ignore[assignment]
    return calls


# ---------------------------------------------------------------------------
# Backend selection and defaults
# ---------------------------------------------------------------------------

def test_default_backend_is_luna_with_gpt6_luna_model():
    client = CloudArchitectClient()
    assert client.backend == "luna"
    assert client.model == "gpt-6-luna"


def test_haiku_backend_defaults_to_haiku_5_5_model():
    client = CloudArchitectClient(backend="haiku")
    assert client.backend == "haiku"
    assert client.model == "claude-haiku-5-5"


def test_agy_backend_has_no_model_override():
    client = CloudArchitectClient(backend="agy")
    assert client.backend == "agy"
    assert client.model is None


def test_backend_name_is_case_and_whitespace_insensitive():
    assert CloudArchitectClient(backend="  HAIKU ").backend == "haiku"


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown Cloud Architect backend 'gpt'"):
        CloudArchitectClient(backend="gpt")


def test_explicit_model_overrides_backend_default():
    client = CloudArchitectClient(backend="haiku", model="claude-sonnet-5-5")
    assert client.model == "claude-sonnet-5-5"


# ---------------------------------------------------------------------------
# Dispatcher routing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "backend, target",
    [("luna", "_invoke_codex"), ("haiku", "_invoke_claude"), ("agy", "_invoke_agy")],
)
def test_dispatcher_routes_to_backend(monkeypatch, backend, target):
    client = CloudArchitectClient(backend=backend)
    seen = {}

    def fake(prompt):
        seen["prompt"] = prompt
        return ("ok", 1, 1)

    monkeypatch.setattr(client, target, fake)
    assert client._invoke_cloud("PLAN") == ("ok", 1, 1)
    assert seen["prompt"] == "PLAN"


# ---------------------------------------------------------------------------
# Haiku (Claude Code CLI)
# ---------------------------------------------------------------------------

def test_haiku_command_disables_tools_and_places_prompt_last():
    client = CloudArchitectClient(backend="haiku", cli_path="/usr/bin/claude")
    payload = json.dumps({
        "is_error": False,
        "subtype": "success",
        "result": '{"ok": true}',
        "usage": {"input_tokens": 2, "cache_creation_input_tokens": 100,
                  "cache_read_input_tokens": 50, "output_tokens": 7},
    })
    calls = _capture(client, stdout=payload)

    client._invoke_claude("PLAN THIS")

    cmd, label = calls[0]
    assert label == "claude"
    assert cmd[0] == "/usr/bin/claude"
    assert "-p" in cmd
    assert cmd[cmd.index("--model") + 1] == "claude-haiku-5-5"
    assert cmd[cmd.index("--output-format") + 1] == "json"
    # Planner must not act: all built-in tools disabled, plan permission mode.
    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--permission-mode") + 1] == "plan"
    assert cmd[-1] == "PLAN THIS"


def test_haiku_parses_result_and_counts_cache_tokens():
    client = CloudArchitectClient(backend="haiku", cli_path="/usr/bin/claude")
    payload = json.dumps({
        "is_error": False,
        "subtype": "success",
        "result": '{"subtasks": []}',
        "usage": {"input_tokens": 2, "cache_creation_input_tokens": 100,
                  "cache_read_input_tokens": 50, "output_tokens": 7},
    })
    _capture(client, stdout=payload)

    raw, in_tok, out_tok = client._invoke_claude("x")

    assert raw == '{"subtasks": []}'
    assert in_tok == 152  # 2 + 100 + 50: full context, including cache traffic
    assert out_tok == 7


def test_haiku_error_payload_raises():
    client = CloudArchitectClient(backend="haiku", cli_path="/usr/bin/claude")
    _capture(client, stdout=json.dumps({"is_error": True, "subtype": "error", "result": "rate limited"}))
    with pytest.raises(CloudArchitectError, match="rate limited"):
        client._invoke_claude("x")


def test_haiku_non_json_output_raises():
    client = CloudArchitectClient(backend="haiku", cli_path="/usr/bin/claude")
    _capture(client, stdout="not json at all")
    with pytest.raises(CloudArchitectError, match="Failed to parse JSON output from claude"):
        client._invoke_claude("x")


def test_haiku_empty_result_raises():
    client = CloudArchitectClient(backend="haiku", cli_path="/usr/bin/claude")
    _capture(client, stdout=json.dumps({"is_error": False, "subtype": "success", "result": "  "}))
    with pytest.raises(CloudArchitectError, match="empty result"):
        client._invoke_claude("x")


def test_haiku_nonzero_exit_raises():
    client = CloudArchitectClient(backend="haiku", cli_path="/usr/bin/claude")
    _capture(client, rc=3, stderr="auth failed")
    with pytest.raises(CloudArchitectError, match="non-zero exit code"):
        client._invoke_claude("x")


# ---------------------------------------------------------------------------
# agy (Antigravity CLI)
# ---------------------------------------------------------------------------

def test_agy_command_and_parse():
    client = CloudArchitectClient(backend="agy", cli_path="/usr/bin/agy")
    payload = json.dumps({"response": '{"a":1}', "usage": {"input_tokens": 11, "output_tokens": 3}})
    calls = _capture(client, stdout=payload)

    raw, in_tok, out_tok = client._invoke_agy("PLAN")

    cmd, label = calls[0]
    assert label == "agy"
    assert cmd[:4] == ["/usr/bin/agy", "-p", "PLAN", "--output-format"]
    assert raw == '{"a":1}'
    assert (in_tok, out_tok) == (11, 3)


# ---------------------------------------------------------------------------
# Wiring: coordinator and CLI
# ---------------------------------------------------------------------------

def test_cli_architect_flag_defaults_and_parses():
    parser = build_parser()
    assert parser.parse_args([]).architect == "haiku"
    assert parser.parse_args(["--architect", "haiku"]).architect == "haiku"
    assert parser.parse_args(["--architect", "agy"]).architect == "agy"
    with pytest.raises(SystemExit):
        parser.parse_args(["--architect", "gpt-9"])


def test_coordinator_passes_backend_to_live_architect(tmp_path):
    from antigravity_hybrid_orchestrator.coordinator import HybridCoordinator

    coord = HybridCoordinator(
        mode="live",
        goal="implement x",
        workspace=str(tmp_path),
        no_hud=True,
        architect_backend="haiku",
    )
    assert coord.architect.backend == "haiku"
    assert coord.architect.model == "claude-haiku-5-5"
