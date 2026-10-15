"""Cloud cost accounting: CLI-reported cost is used as-is; only unreported tokens are estimated."""

from __future__ import annotations

import json

import pytest

from antigravity_hybrid_orchestrator.architect.client import CloudArchitectClient
from antigravity_hybrid_orchestrator.telemetry import GEMINI_INPUT_PRICE_PER_1M, TelemetryTracker


def test_estimate_only_when_nothing_reported():
    t = TelemetryTracker()
    t.add_cloud(input_tokens=1_000_000, output_tokens=0)
    assert t.get_summary().cloud_cost_usd == pytest.approx(GEMINI_INPUT_PRICE_PER_1M)


def test_reported_cost_used_verbatim():
    t = TelemetryTracker()
    t.add_cloud(input_tokens=1000, output_tokens=500, reported_cost_usd=0.02)
    assert t.get_summary().cloud_cost_usd == pytest.approx(0.02)


def test_mixed_reported_and_unreported_calls():
    t = TelemetryTracker()
    t.add_cloud(input_tokens=1000, output_tokens=0, reported_cost_usd=0.02)
    t.add_cloud(input_tokens=1_000_000, output_tokens=0)  # no cost reported -> estimated
    assert t.get_summary().cloud_cost_usd == pytest.approx(0.02 + GEMINI_INPUT_PRICE_PER_1M)


def test_claude_cli_cost_is_captured(monkeypatch):
    client = CloudArchitectClient(backend="haiku", cli_path="/usr/bin/claude")
    payload = {
        "is_error": False, "subtype": "success", "result": "{}",
        "total_cost_usd": 0.0152,
        "usage": {"input_tokens": 2, "cache_creation_input_tokens": 10, "cache_read_input_tokens": 0, "output_tokens": 5},
    }
    monkeypatch.setattr(client, "_run_subprocess", lambda cmd, label: (0, json.dumps(payload), ""))
    client._invoke_cloud("x")
    assert client.last_call_cost_usd == pytest.approx(0.0152)


def test_codex_call_reports_no_cost(monkeypatch):
    client = CloudArchitectClient(backend="luna", codex_path="/usr/bin/codex")
    client.last_call_cost_usd = 9.9  # stale value from an earlier call
    monkeypatch.setattr(client, "_invoke_codex", lambda prompt: ("{}", 10, 5))
    client._invoke_cloud("x")
    assert client.last_call_cost_usd is None


def test_replan_and_failed_calls_reach_telemetry(tmp_path):
    """Every planner call is recorded once: accepted plans, repairs, replans and failures."""
    from antigravity_hybrid_orchestrator.coordinator import HybridCoordinator
    from antigravity_hybrid_orchestrator.planner.dag import ExecutionPlanDAG, SubTask

    coord = HybridCoordinator(mode="live", goal="g", workspace=str(tmp_path), no_hud=True)
    arch = coord.architect
    arch.usage_ledger.extend([
        {"input_tokens": 1000, "output_tokens": 100, "cost_usd": 0.01, "failed": True},   # rejected plan
        {"input_tokens": 2000, "output_tokens": 200, "cost_usd": 0.02, "failed": False},  # repair plan
    ])
    coord._drain_planner_usage()
    summary = coord.telemetry.get_summary()
    assert abs(summary.cloud_cost_usd - 0.03) < 1e-9
    assert arch.usage_ledger == []  # drained exactly once
    coord._drain_planner_usage()    # a second drain adds nothing
    assert coord.telemetry.get_summary().cloud_tokens == summary.cloud_tokens
