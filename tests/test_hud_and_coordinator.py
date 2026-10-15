"""Unit and integration tests for Milestone 3: Rich Terminal HUD, Coordinator, CLI, and Runner."""

from pathlib import Path
import pytest
from rich.console import Console

from antigravity_hybrid_orchestrator.cli import build_parser, main
from antigravity_hybrid_orchestrator.coordinator import HybridCoordinator
from antigravity_hybrid_orchestrator.hud import (
    DashboardController,
    DashboardManager,
    DashboardState,
    build_orchestrator_layout,
    render_final_summary,
)
from antigravity_hybrid_orchestrator.hud.components import (
    build_builder_panel,
    build_cloud_panel,
    build_critic_panel,
    build_gauntlet_panel,
    build_header_panel,
    build_telemetry_table,
    format_status_badge,
)
from antigravity_hybrid_orchestrator.gauntlet.baselines import restore_secure
from antigravity_hybrid_orchestrator.runner import GauntletTestRunner
from antigravity_hybrid_orchestrator.telemetry import TelemetrySummary


@pytest.fixture(autouse=True)
def ensure_gauntlet_restored():
    yield
    restore_secure()
    import sys
    for mod in ["auth", "billing", "database"]:
        sys.modules.pop(f"antigravity_hybrid_orchestrator.gauntlet.{mod}", None)


# ==============================================================================
# 1. LAYOUT & RESPONSIVENESS TESTS
# ==============================================================================

def test_hud_layout_wide_terminal():
    """Wide terminals (>=100 cols) should split workforce panels side-by-side."""
    layout = build_orchestrator_layout(terminal_width=120)
    assert "header" in [child.name for child in layout.children]
    assert "body" in [child.name for child in layout.children]
    assert "footer" in [child.name for child in layout.children]

    # Verify workforce layout children
    right_layout = layout["right"]
    workforce = right_layout["workforce"]
    child_names = [child.name for child in workforce.children]
    assert "builder_a" in child_names
    assert "builder_b" in child_names


def test_hud_layout_narrow_terminal_responsiveness():
    """Narrow terminals (<100 cols) should stack workforce panels vertically."""
    layout = build_orchestrator_layout(terminal_width=80)
    workforce = layout["right"]["workforce"]
    child_names = [child.name for child in workforce.children]
    assert "builder_a" in child_names
    assert "builder_b" in child_names


# ==============================================================================
# 2. UI COMPONENT RENDERING TESTS
# ==============================================================================

def test_format_status_badge_variants():
    """Verifies color styles and iconography across agent lifecycle states."""
    b_ready = format_status_badge("READY")
    assert "READY" in b_ready.plain

    b_stream = format_status_badge("STREAMING")
    assert "STREAMING" in b_stream.plain

    b_fail = format_status_badge("FAIL")
    assert "FAIL" in b_fail.plain

    b_queue = format_status_badge("QUEUED")
    assert "QUEUED" in b_queue.plain

    b_idle = format_status_badge("IDLE")
    assert "IDLE" in b_idle.plain


def test_build_header_panel():
    panel = build_header_panel(current_target="billing.py", mode="mock", overall_status="RUNNING")
    assert panel is not None


def test_rd_worker_model_usage_updates_live_local_telemetry(tmp_path: Path):
    coordinator = HybridCoordinator(
        mode="mock", goal="audit", workspace=str(tmp_path), no_hud=True
    )

    coordinator._on_worker_event("model_usage", "task_1", {
        "input_tokens": 30, "output_tokens": 12, "duration_s": 3.0,
    })

    summary = coordinator.telemetry.get_summary()
    assert summary.local_tokens == 12
    assert summary.tokens_per_second == 4.0
    assert coordinator.rd_hud.state.telemetry_summary == summary


def test_build_cloud_panel():
    panel = build_cloud_panel(
        status="READY",
        threat_model="CWE-798 Hardcoded Secrets",
        invariants=["Zero hardcoded keys", "Constant time check"],
    )
    assert panel is not None


def test_build_builder_panel_empty_and_populated():
    p_empty = build_builder_panel(name="Builder A", code="", status="IDLE")
    assert p_empty is not None

    code_sample = "def authenticate():\n    return True\n"
    p_populated = build_builder_panel(name="Builder A", code=code_sample, status="READY", is_winner=True)
    assert p_populated is not None


def test_build_critic_panel():
    panel = build_critic_panel(
        status="READY",
        alpha_score=8.5,
        beta_score=9.5,
        winner="Candidate Beta",
        winning_builder="Builder B",
        critique="Comprehensive mutex locking ensures thread safety.",
    )
    assert panel is not None


def test_build_telemetry_table_none_and_populated():
    p_none = build_telemetry_table(None)
    assert p_none is not None
    console = Console(record=True, width=120)
    console.print(p_none)
    rendered = console.export_text()
    assert "Codex CLI" in rendered
    assert "Benchmark Cost" in rendered

    summary = TelemetrySummary(
        cloud_tokens=500,
        cloud_cost_usd=0.00015,
        local_tokens=2500,
        local_cost_usd=0.0,
        total_tokens=3000,
        cost_savings_pct=85.0,
        tokens_per_second=24.5,
    )
    p_pop = build_telemetry_table(summary)
    assert p_pop is not None


def test_build_gauntlet_panel():
    test_results = {
        "auth.py": ("PASS", "PASS"),
        "billing.py": ("PASS", "FAIL"),
        "database.py": ("PENDING", "PENDING"),
    }
    panel = build_gauntlet_panel(test_results, privacy_verified=True)
    assert panel is not None


# ==============================================================================
# 3. DASHBOARD CONTROLLER & LIFECYCLE TESTS
# ==============================================================================

def test_dashboard_controller_lifecycle():
    console = Console(record=True, width=120)
    hud = DashboardController(console=console, no_hud=True, mode="mock")

    assert hud.state.mode == "mock"
    hud.start()
    hud.set_target("billing.py")
    assert hud.state.current_target == "billing.py"

    hud.update_cloud_strategy(threat_model="Race condition", invariants=["Mutex lock"])
    assert hud.state.threat_model == "Race condition"
    assert len(hud.state.invariants) == 1

    hud.update_builder_a("def ", "def withdraw(): pass", status="GENERATING")
    assert hud.state.builder_a_code == "def withdraw(): pass"

    hud.update_builder_b("class ", "class Billing: pass", status="GENERATING")
    assert hud.state.builder_b_code == "class Billing: pass"

    hud.update_critic(
        critique="Good lock",
        winner="Candidate Beta",
        winning_builder="Builder B",
        alpha_score=8.0,
        beta_score=9.5,
    )
    assert hud.state.critic_winner == "Candidate Beta"

    hud.update_test_results("billing.py", "PASS", "PASS")
    assert hud.state.test_results["billing.py"] == ("PASS", "PASS")

    hud.stop()


def test_dashboard_controller_print_final_summary():
    console = Console(record=True, width=120)
    hud = DashboardController(console=console, no_hud=True, mode="mock")
    summary = TelemetrySummary(
        cloud_tokens=300,
        cloud_cost_usd=0.0001,
        local_tokens=1500,
        local_cost_usd=0.0,
        total_tokens=1800,
        cost_savings_pct=88.5,
        tokens_per_second=20.0,
    )
    test_results = {
        "auth.py": ("PASS", "PASS"),
        "billing.py": ("PASS", "PASS"),
        "database.py": ("PASS", "PASS"),
    }
    hud.print_final_summary(summary, test_results, privacy_verified=True, duration_s=1.5)
    rendered = console.export_text()
    assert "RUN SUMMARY" in rendered
    assert "88.5%" in rendered
    assert "Zero-Knowledge Code Privacy Invariant" in rendered


# ==============================================================================
# 4. GAUNTLET TEST RUNNER PROGRAMMATIC TESTS
# ==============================================================================

def test_gauntlet_test_runner_single_test():
    runner = GauntletTestRunner()
    passed = runner.run_single_test("test_auth_functional")
    assert passed is True


def test_gauntlet_test_runner_target():
    runner = GauntletTestRunner()
    func_ok, adv_ok = runner.run_tests_for_target("auth.py")
    assert func_ok is True
    assert adv_ok is True


# ==============================================================================
# 5. COORDINATOR END-TO-END PIPELINE TESTS
# ==============================================================================

def test_coordinator_mock_single_target():
    """Tests coordinator on a single target with no-hud in mock mode."""
    coordinator = HybridCoordinator(
        mode="mock",
        target="auth",
        reset=True,
        no_hud=True,
    )
    exit_code = coordinator.run()
    assert exit_code == 0
    assert coordinator.telemetry.get_summary().cloud_tokens > 0
    assert coordinator.telemetry.get_summary().local_tokens > 0


def test_coordinator_mock_all_targets_with_reset():
    """Tests coordinator on all targets in mock mode with baseline reset."""
    coordinator = HybridCoordinator(
        mode="mock",
        target="all",
        reset=True,
        no_hud=True,
    )
    exit_code = coordinator.run()
    assert exit_code == 0
    results = coordinator.hud.state.test_results
    assert results["auth.py"] == ("PASS", "PASS")
    assert results["billing.py"] == ("PASS", "PASS")
    assert results["database.py"] == ("PASS", "PASS")


def test_coordinator_invalid_target_raises_error():
    coordinator = HybridCoordinator(mode="mock", target="invalid_target_file")
    with pytest.raises(ValueError):
        coordinator.run()


# ==============================================================================
# 6. CLI ENTRYPOINT & PARSER TESTS
# ==============================================================================

def test_cli_main_mock_run():
    exit_code = main(["--mode", "mock", "--target", "billing", "--no-hud", "--reset"])
    assert exit_code == 0
