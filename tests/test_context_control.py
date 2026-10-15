"""Tests for workforce context control: num_ctx, output pruning, history compaction."""

from __future__ import annotations

import pytest

from antigravity_hybrid_orchestrator.workforce.context import (
    DEFAULT_NUM_CTX,
    compact_messages,
    num_ctx,
    prune_tool_output,
)
from antigravity_hybrid_orchestrator.workforce.ollama_client import WorkforceClient


# --- num_ctx -----------------------------------------------------------------

def test_num_ctx_default(monkeypatch):
    monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)
    assert num_ctx() == DEFAULT_NUM_CTX == 16384


def test_num_ctx_env_override_and_floor(monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
    assert num_ctx() == 8192
    monkeypatch.setenv("OLLAMA_NUM_CTX", "512")  # below floor
    assert num_ctx() == 2048
    monkeypatch.setenv("OLLAMA_NUM_CTX", "not-a-number")
    assert num_ctx() == DEFAULT_NUM_CTX


def test_chat_payload_carries_explicit_num_ctx(monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_CTX", "12288")
    client = WorkforceClient(base_url="http://x:11434", model="m")
    captured = {}

    def fake_post(url, payload):
        captured.update(payload)
        return {"message": {"content": "ok", "tool_calls": []}}

    monkeypatch.setattr(client, "_post_json", fake_post)
    client.chat(messages=[{"role": "user", "content": "hi"}])
    assert captured["options"]["num_ctx"] == 12288


# --- prune_tool_output -------------------------------------------------------

def test_short_output_is_unchanged():
    text = "line1\nline2\nok"
    assert prune_tool_output(text) == text


def test_long_output_keeps_head_and_tail_with_marker():
    lines = [f"noise {i}" for i in range(500)]
    lines[-1] = "AssertionError: expected 6 got 5"  # the failure is at the end
    out = prune_tool_output("\n".join(lines), head=20, tail=50)
    assert out.startswith("noise 0\n")
    assert "lines omitted" in out
    assert out.rstrip().endswith("AssertionError: expected 6 got 5")
    assert len(out) <= 4000 + 40  # marker slack


def test_pruning_never_drops_the_traceback_tail():
    trace = ["Traceback (most recent call last):"] + [f"  frame {i}" for i in range(300)] + ["ValueError: boom"]
    out = prune_tool_output("\n".join(trace))
    assert "ValueError: boom" in out


# --- compact_messages --------------------------------------------------------

def _tool(name, content):
    return {"role": "tool", "tool_name": name, "content": content}


def test_compaction_keeps_task_and_recent_tools_verbatim():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "TASK " + "x" * 400},
    ]
    for i in range(6):
        msgs.append({"role": "assistant", "content": "thinking " * 80, "tool_calls": [{"function": {"name": "read_file"}}]})
        msgs.append(_tool("read_file", f"result-{i} " + "y" * 600))

    changed = compact_messages(msgs, keep_recent_tools=3)

    assert changed > 0
    assert msgs[0]["content"] == "sys"
    assert msgs[1]["content"].startswith("TASK ")          # task never compacted
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert tool_msgs[0]["content"].startswith("[compacted read_file]")
    assert tool_msgs[-1]["content"].startswith("result-5 ")  # newest kept verbatim
    assert "tool_calls" in msgs[2]                           # protocol structure intact
    assert tool_msgs[-1]["tool_name"] == "read_file"


def test_compaction_is_noop_for_short_history():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, _tool("git_status", "clean")]
    assert compact_messages(msgs) == 0
    assert msgs[2]["content"] == "clean"


# --- read_file default window ------------------------------------------------

def test_unranged_read_of_large_file_is_windowed(tmp_path):
    from antigravity_hybrid_orchestrator.workforce.tools import READ_DEFAULT_LINE_LIMIT, WorkspaceToolSuite
    (tmp_path / "big.py").write_text("\n".join(f"x = {i}" for i in range(1000)) + "\n")
    res = WorkspaceToolSuite(workspace_root=tmp_path).read_file("big.py")
    assert res.success
    assert f"showing lines 1-{READ_DEFAULT_LINE_LIMIT} of 1000" in res.output
    assert res.metadata["end_line"] == READ_DEFAULT_LINE_LIMIT
    assert "x = 399" in res.output and "x = 400" not in res.output


def test_explicit_range_still_returns_exact_lines(tmp_path):
    from antigravity_hybrid_orchestrator.workforce.tools import WorkspaceToolSuite
    (tmp_path / "big.py").write_text("\n".join(f"x = {i}" for i in range(1000)) + "\n")
    res = WorkspaceToolSuite(workspace_root=tmp_path).read_file("big.py", start_line=500, end_line=502)
    assert res.output == "x = 499\nx = 500\nx = 501"


def test_chat_payload_caps_output_tokens(monkeypatch):
    monkeypatch.delenv("OLLAMA_NUM_PREDICT", raising=False)
    client = WorkforceClient(base_url="http://x:11434", model="m")
    captured = {}

    def fake_post(url, payload):
        captured.update(payload)
        return {"message": {"content": "ok"}}

    monkeypatch.setattr(client, "_post_json", fake_post)
    client.chat(messages=[{"role": "user", "content": "hi"}])
    assert captured["options"]["num_predict"] == 8192


def test_chat_num_predict_env_override(monkeypatch):
    from antigravity_hybrid_orchestrator.workforce.context import chat_num_predict
    monkeypatch.setenv("OLLAMA_NUM_PREDICT", "4096")
    assert chat_num_predict() == 4096
    monkeypatch.setenv("OLLAMA_NUM_PREDICT", "bad")
    assert chat_num_predict() == 8192


def test_model_call_timeout_covers_full_response(monkeypatch):
    from antigravity_hybrid_orchestrator.workforce.context import ollama_timeout_s
    monkeypatch.delenv("OLLAMA_TIMEOUT_S", raising=False)
    assert ollama_timeout_s() == 900.0
    assert WorkforceClient(base_url="http://x", model="m").timeout == 900.0
    monkeypatch.setenv("OLLAMA_TIMEOUT_S", "1200")
    assert WorkforceClient(base_url="http://x", model="m").timeout == 1200.0
    monkeypatch.setenv("OLLAMA_TIMEOUT_S", "bad")
    assert ollama_timeout_s() == 900.0


def test_model_idle_timeout_default_covers_full_response():
    from antigravity_hybrid_orchestrator.engine.control import ExecutionBudget
    assert ExecutionBudget().model_idle_timeout_s == 900.0
