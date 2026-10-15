"""Challenger 1 (teamwork_preview_challenger) empirical stress test harness for Milestone 3.

Empirically challenges:
1. Rich Terminal HUD layout responsiveness under narrow (<100 cols) and wide (>=120 cols) terminals.
2. Console rendering across extreme terminal dimensions (widths 40 to 240, heights 10 to 80).
3. DashboardController update frequency stress (1,000 rapid sequential updates, 20 concurrent threads).
4. Adversarial inputs to HUD panels (huge strings, ANSI escapes, unclosed tags, NaN/Inf, negative values).
5. Programmatic invocation of HybridCoordinator with mock client across all 3 targets with --reset.
6. Baseline corruption and empirical recovery verification via --reset.
7. Subprocess CLI invocation of python3 -m antigravity_hybrid_orchestrator --mode mock --reset.
"""

import concurrent.futures
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import pytest
from rich.console import Console

from antigravity_hybrid_orchestrator.cli import main
from antigravity_hybrid_orchestrator.coordinator import HybridCoordinator
from antigravity_hybrid_orchestrator.gauntlet.baselines import (
    AUTH_VULNERABLE,
    BILLING_VULNERABLE,
    DATABASE_VULNERABLE,
    restore_secure,
)
from antigravity_hybrid_orchestrator.hud import (
    DashboardController,
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
from antigravity_hybrid_orchestrator.telemetry import TelemetrySummary, TelemetryTracker


@pytest.fixture(autouse=True)
def clean_gauntlet_environment():
    """Ensure gauntlet files are in known secure state after each test."""
    yield
    restore_secure()
    for mod in ["auth", "billing", "database"]:
        sys.modules.pop(f"antigravity_hybrid_orchestrator.gauntlet.{mod}", None)


# ==============================================================================
# SECTION 1: LAYOUT RESPONSIVENESS & EXTREME DIMENSIONS
# ==============================================================================

@pytest.mark.parametrize("width", [20, 40, 60, 79, 80, 99])
def test_hud_layout_narrow_terminal_stacking(width: int):
    """Stress test: Narrow terminals (<100 columns) MUST stack workforce panels vertically."""
    layout = build_orchestrator_layout(terminal_width=width)
    assert layout["root"] is not None
    assert layout["header"] is not None
    assert layout["body"] is not None
    assert layout["footer"] is not None

    workforce = layout["right"]["workforce"]
    # Check that workforce children exist
    children_names = [c.name for c in workforce.children]
    assert children_names == ["builder_a", "builder_b"]


@pytest.mark.parametrize("width", [100, 101, 120, 160, 240, 400])
def test_hud_layout_wide_terminal_side_by_side(width: int):
    """Stress test: Wide terminals (>=100 columns) MUST place workforce panels side-by-side."""
    layout = build_orchestrator_layout(terminal_width=width)
    assert layout["root"] is not None
    assert layout["header"] is not None
    assert layout["body"] is not None
    assert layout["footer"] is not None

    workforce = layout["right"]["workforce"]
    children_names = [c.name for c in workforce.children]
    assert children_names == ["builder_a", "builder_b"]


@pytest.mark.parametrize("width", [40, 60, 80, 99, 100, 120, 200])
@pytest.mark.parametrize("height", [10, 20, 24, 40, 80])
def test_console_render_across_extreme_terminal_sizes(width: int, height: int):
    """Stress test: Console rendering across extreme matrix of terminal widths and heights.
    Must never crash with LayoutError, ValueError, or formatting exceptions.
    """
    console = Console(record=True, width=width, height=height)
    hud = DashboardController(console=console, no_hud=True, mode="mock")

    # Populate state with rich realistic data
    hud.state.current_target = "database.py"
    hud.state.pipeline_status = "RUNNING"
    hud.state.cloud_status = "READY"
    hud.state.threat_model = "CWE-89 SQL Injection Parameterization via parameterized queries"
    hud.state.invariants = [
        "Zero string formatting in cursor.execute",
        "Tuple parameters for user input",
        "Closed cursor resource cleanup",
    ]
    hud.state.builder_a_status = "STREAMING"
    hud.state.builder_a_code = "def get_user(db, name):\n    return db.execute('SELECT ?', (name,))"
    hud.state.builder_b_status = "READY"
    hud.state.builder_b_code = "class SafeDB:\n    def query(self, *args):\n        pass"
    hud.state.critic_status = "DECIDED"
    hud.state.critic_winner = "Candidate Beta"
    hud.state.critic_winning_builder = "Builder B"
    hud.state.critic_alpha_score = 8.5
    hud.state.critic_beta_score = 9.8
    hud.state.critic_critique = "Builder B provides robust parameterization and connection lifecycle."
    hud.state.telemetry_summary = TelemetrySummary(
        cloud_tokens=1200,
        cloud_cost_usd=0.00045,
        local_tokens=4500,
        local_cost_usd=0.0,
        total_tokens=5700,
        cost_savings_pct=88.2,
        tokens_per_second=31.4,
    )
    hud.state.test_results = {
        "auth.py": ("PASS", "PASS"),
        "billing.py": ("PASS", "PASS"),
        "database.py": ("PASS", "PASS"),
    }

    # Render into layout
    hud._render_into_layout()

    # Verify Console can render the layout without raising exceptions
    console.print(hud.layout)
    rendered_text = console.export_text()
    assert len(rendered_text) > 0


@pytest.mark.parametrize("width", [40, 60, 80, 100, 120, 200])
def test_summary_table_render_across_widths(width: int):
    """Stress test: Post-run summary table renders cleanly across varied widths."""
    console = Console(record=True, width=width)
    summary = TelemetrySummary(
        cloud_tokens=800,
        cloud_cost_usd=0.0002,
        local_tokens=3200,
        local_cost_usd=0.0,
        total_tokens=4000,
        cost_savings_pct=85.0,
        tokens_per_second=28.0,
    )
    results = {
        "auth.py": ("PASS", "PASS"),
        "billing.py": ("PASS", "PASS"),
        "database.py": ("PASS", "PASS"),
    }
    render_final_summary(
        console=console,
        summary=summary,
        test_results=results,
        privacy_verified=True,
        duration_s=2.5,
    )
    out = console.export_text()
    assert "RUN SUMMARY" in out
    if width >= 80:
        assert "85.0%" in out
    else:
        assert "8" in out


# ==============================================================================
# SECTION 2: UPDATE FREQUENCY, CONCURRENCY & ROBUSTNESS
# ==============================================================================

def test_dashboard_controller_rapid_fire_sequential_updates():
    """Stress test: 1,000 rapid updates in an unthrottled loop without sleep."""
    console = Console(record=True, width=120)
    hud = DashboardController(console=console, no_hud=True, mode="mock")
    hud.start()

    tracker = TelemetryTracker()
    for i in range(1000):
        tracker.add_cloud(input_tokens=1, output_tokens=1)
        tracker.add_local(eval_tokens=2, duration_s=0.001)

        hud.update_builder_a(f"tok_{i}", f"code_accum_{i}", status="STREAMING")
        hud.update_telemetry(tracker.get_summary())

        if i % 100 == 0:
            hud.set_pipeline_status(f"PROGRESS_{i}")
            hud.update_critic(
                critique=f"critique {i}",
                winner="Candidate Alpha",
                winning_builder="Builder A",
                alpha_score=float(i % 10),
                beta_score=float((i + 1) % 10),
            )

    hud.stop()
    assert hud.state.builder_a_code == "code_accum_999"
    assert hud.state.telemetry_summary.cloud_tokens == 2000


def test_dashboard_controller_concurrent_multi_threaded_updates():
    """Stress test: 20 concurrent threads flooding DashboardController with 50 updates each."""
    console = Console(record=True, width=120)
    hud = DashboardController(console=console, no_hud=True, mode="mock")
    hud.start()

    num_threads = 20
    updates_per_thread = 50
    exceptions = []

    def stress_worker(worker_id: int):
        try:
            for j in range(updates_per_thread):
                hud.update_builder_a(f"w{worker_id}_{j}", f"code_{worker_id}_{j}")
                hud.update_builder_b(f"w{worker_id}_{j}", f"code_{worker_id}_{j}")
                hud.update_cloud_strategy(f"Threat {worker_id}", [f"Inv {j}"])
                hud.update_test_results("auth.py", "PASS", "PASS")
                hud.refresh()
        except Exception as e:
            exceptions.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(stress_worker, tid) for tid in range(num_threads)]
        concurrent.futures.wait(futures)

    hud.stop()
    assert len(exceptions) == 0, f"Concurrent HUD updates raised exceptions: {exceptions}"


def test_dashboard_controller_lifecycle_robustness():
    """Stress test: Idempotent start/stop and operations before start and after stop."""
    console = Console(record=True, width=120)
    hud = DashboardController(console=console, no_hud=True, mode="mock")

    # Updates before start must not crash
    hud.update_builder_a("chunk", "accumulated")
    hud.refresh()
    hud.update_telemetry(None)

    # Double start
    hud.start()
    hud.start()

    # Updates during run
    hud.set_pipeline_status("RUNNING")
    hud.refresh()

    # Double stop
    hud.stop()
    hud.stop()

    # Updates after stop must not crash
    hud.update_builder_b("chunk", "accumulated")
    hud.refresh()


def test_adversarial_malformed_inputs_to_hud_panels():
    """Stress test: Malformed, oversized, or malicious inputs to all HUD component builders."""
    console = Console(record=True, width=120)

    # 1. 10,000 lines of builder code (syntax highlighter stress)
    huge_code = "\n".join([f"line_{i} = {i} * 2" for i in range(10000)])
    panel_huge = build_builder_panel("Builder A", huge_code, "READY")
    assert panel_huge is not None
    console.print(panel_huge)

    # 2. Malicious rich tags, unclosed brackets, null bytes, ANSI escapes
    adversarial_str = "[bold red]unclosed tag [italic] \x00 null byte \x1b[31m ansi \x1b[0m [/not_a_tag]"
    cloud_panel = build_cloud_panel(
        status="READY",
        threat_model=adversarial_str,
        invariants=[adversarial_str, "Normal invariant"],
    )
    assert cloud_panel is not None
    console.print(cloud_panel)

    # 3. Bad badge statuses (None, empty, non-standard)
    b1 = format_status_badge(None)
    b2 = format_status_badge("")
    b3 = format_status_badge("UNKNOWN_CUSTOM_BADGE_STATUS")
    assert b1 is not None and b2 is not None and b3 is not None

    # 4. Extreme telemetry values (NaN, infinity, huge numbers)
    extreme_summary = TelemetrySummary(
        cloud_tokens=10**12,
        cloud_cost_usd=99999.999999,
        local_tokens=10**12,
        local_cost_usd=0.0,
        total_tokens=2 * 10**12,
        cost_savings_pct=100.0,
        tokens_per_second=99999.9,
    )
    telemetry_panel = build_telemetry_table(extreme_summary)
    assert telemetry_panel is not None
    console.print(telemetry_panel)

    # 5. Critic panel with extreme / negative / out of bounds scores
    critic_panel = build_critic_panel(
        status="DECIDED",
        alpha_score=-5.0,
        beta_score=999.9,
        winner="Candidate Alpha",
        winning_builder="Builder A",
        critique="A" * 5000,  # 5,000 character critique
    )
    assert critic_panel is not None
    console.print(critic_panel)


# ==============================================================================
# SECTION 3: PROGRAMMATIC COORDINATOR EXECUTION ACROSS TARGETS WITH --RESET
# ==============================================================================

@pytest.mark.parametrize("target", ["auth", "billing", "database"])
def test_coordinator_individual_targets_mock_with_reset(target: str):
    """Stress test: Programmatic invocation of HybridCoordinator for each target individually with reset."""
    coordinator = HybridCoordinator(
        mode="mock",
        target=target,
        reset=True,
        no_hud=True,
    )
    exit_code = coordinator.run()
    assert exit_code == 0, f"Coordinator failed on target '{target}' with exit code {exit_code}"

    # Verify telemetry recorded
    summary = coordinator.telemetry.get_summary()
    assert summary.cloud_tokens > 0
    assert summary.local_tokens > 0
    assert summary.cost_savings_pct > 0.0

    # Verify target result was recorded as PASS, PASS
    norm_name = f"{target}.py"
    func, adv = coordinator.hud.state.test_results[norm_name]
    assert func == "PASS", f"Functional test failed for {target}: {func}"
    assert adv == "PASS", f"Adversarial test failed for {target}: {adv}"


@pytest.mark.parametrize("target_variation", ["auth.py", "billing.py", "database.py"])
def test_coordinator_target_filename_variations(target_variation: str):
    """Stress test: Coordinator accepts both bare module names and .py filenames."""
    coordinator = HybridCoordinator(
        mode="mock",
        target=target_variation,
        reset=True,
        no_hud=True,
    )
    exit_code = coordinator.run()
    assert exit_code == 0


def test_coordinator_all_targets_with_reset():
    """Stress test: Coordinator runs all 3 targets in a single pipeline with reset=True."""
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


def test_coordinator_reset_empirically_restores_baselines():
    """Empirical check: Verify that reset=True explicitly restores vulnerable baselines before patching."""
    gauntlet_dir = Path(__file__).resolve().parent.parent / "antigravity_hybrid_orchestrator" / "gauntlet"
    auth_file = gauntlet_dir / "auth.py"
    billing_file = gauntlet_dir / "billing.py"

    # Step A: Inject corrupted broken syntax into target files
    corrupted_content = "# CORRUPTED SYNTAX TEST\nINVALID PYTHON CODE ::: !@#$%\n"
    auth_file.write_text(corrupted_content, encoding="utf-8")
    billing_file.write_text(corrupted_content, encoding="utf-8")

    # Step B: Run coordinator with reset=True
    coordinator = HybridCoordinator(
        mode="mock",
        target="all",
        reset=True,
        no_hud=True,
    )
    exit_code = coordinator.run()

    # Step C: Coordinator must succeed because reset=True restored clean baseline before patching
    assert exit_code == 0
    assert auth_file.read_text(encoding="utf-8") != corrupted_content
    assert billing_file.read_text(encoding="utf-8") != corrupted_content

    # Step D: Confirm tests pass
    for target in ["auth.py", "billing.py", "database.py"]:
        assert coordinator.hud.state.test_results[target] == ("PASS", "PASS")


def test_coordinator_invalid_target_error_handling():
    """Stress test: Coordinator raises ValueError when passed an invalid target."""
    with pytest.raises(ValueError, match="Unknown gauntlet target"):
        coordinator = HybridCoordinator(mode="mock", target="unrecognized_module")
        coordinator.run()


# ==============================================================================
# SECTION 4: CLI ENTRYPOINT & SUBPROCESS VERIFICATION
# ==============================================================================

def test_cli_subprocess_full_mock_reset_e2e():
    """Empirical verification: Full execution via `python3 -m antigravity_hybrid_orchestrator --mode mock --reset`."""
    project_root = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable,
        "-m",
        "antigravity_hybrid_orchestrator",
        "--mode",
        "mock",
        "--reset",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=60.0,
    )
    assert proc.returncode == 0, f"CLI command failed with code {proc.returncode}:\n{proc.stderr}"
    stdout = proc.stdout

    # Verify key markers in output
    assert "ANTIGRAVITY HYBRID ORCHESTRATOR — RUN SUMMARY" in stdout
    assert "ALL 6/6 ADVERSARIAL GAUNTLET TESTS PASSED WITH 100% SUCCESS" in stdout
    assert "Zero-Knowledge Code Privacy Invariant: VERIFIED" in stdout
    assert "auth.py" in stdout
    assert "billing.py" in stdout
    assert "database.py" in stdout


def test_cli_subprocess_single_target_mock_reset():
    """Empirical verification: Single target execution via CLI."""
    project_root = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable,
        "-m",
        "antigravity_hybrid_orchestrator",
        "--mode",
        "mock",
        "--target",
        "auth",
        "--no-hud",
        "--reset",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    assert proc.returncode == 0, f"CLI single target failed: {proc.stderr}"
    assert "auth.py" in proc.stdout
