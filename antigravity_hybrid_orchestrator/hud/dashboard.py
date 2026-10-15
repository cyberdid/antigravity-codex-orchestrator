"""Interactive dashboard controller orchestrating rich.live.Live and fallback rendering."""

from dataclasses import dataclass, field
import sys
import time
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.live import Live

from antigravity_hybrid_orchestrator.hud.components import (
    build_builder_panel,
    build_cloud_panel,
    build_critic_panel,
    build_gauntlet_panel,
    build_header_panel,
    build_telemetry_table,
)
from antigravity_hybrid_orchestrator.hud.layout import build_orchestrator_layout
from antigravity_hybrid_orchestrator.hud.summary import render_final_summary
from antigravity_hybrid_orchestrator.telemetry import TelemetrySummary


@dataclass
class DashboardState:
    """Observable state container for HUD visualization."""
    current_target: str = "all"
    mode: str = "mock"
    pipeline_status: str = "INITIALIZING"

    # Cloud Architect
    cloud_status: str = "IDLE"
    threat_model: str = ""
    invariants: List[str] = field(default_factory=list)

    # Workforce Builder A
    builder_a_status: str = "IDLE"
    builder_a_code: str = ""

    # Workforce Builder B
    builder_b_status: str = "IDLE"
    builder_b_code: str = ""

    # Blind Critic
    critic_status: str = "IDLE"
    critic_alpha_score: float = 0.0
    critic_beta_score: float = 0.0
    critic_winner: str = ""
    critic_winning_builder: str = ""
    critic_critique: str = ""

    # Gauntlet test matrix
    test_results: Dict[str, Tuple[str, str]] = field(default_factory=lambda: {
        "auth.py": ("PENDING", "PENDING"),
        "billing.py": ("PENDING", "PENDING"),
        "database.py": ("PENDING", "PENDING"),
    })
    privacy_verified: bool = True

    # Telemetry
    telemetry_summary: Optional[TelemetrySummary] = None


class DashboardController:
    """Wraps rich.live.Live with layout construction and headless fallback."""

    def __init__(
        self,
        console: Optional[Console] = None,
        no_hud: bool = False,
        refresh_per_second: float = 8.0,
        mode: str = "live",
    ) -> None:
        self.console = console or Console()
        self.no_hud = no_hud or (not sys.stdout.isatty())
        self.refresh_per_second = refresh_per_second
        self.state = DashboardState(mode=mode)

        # Detect terminal width
        term_width = self.console.size.width if self.console else 120
        self.layout = build_orchestrator_layout(term_width)

        self._live: Optional[Live] = None
        self._is_running = False

    def start(self) -> None:
        """Starts interactive live display or headless logger."""
        self._is_running = True
        self._render_into_layout()

        if not self.no_hud:
            try:
                self._live = Live(
                    self.layout,
                    console=self.console,
                    refresh_per_second=self.refresh_per_second,
                    screen=False,
                    transient=True,
                )
                self._live.start()
            except Exception as e:
                # Fallback to headless if live display fails (e.g. redirected output)
                self.no_hud = True
                self._live = None
                self.console.print(f"[dim yellow]Warning: Live HUD disabled ({e}). Running in stream mode.[/]")
        else:
            self.console.print(f"[bold cyan]Antigravity Hybrid Orchestrator started in headless mode (HUD disabled)[/]")

    def stop(self) -> None:
        """Cleanly stops live display, restoring cursor and terminal state."""
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None
        self._is_running = False

    def _render_into_layout(self) -> None:
        """Populates layout panels with current state components."""
        try:
            self.layout["header"].update(
                build_header_panel(
                    current_target=self.state.current_target,
                    mode=self.state.mode,
                    overall_status=self.state.pipeline_status,
                )
            )

            self.layout["cloud"].update(
                build_cloud_panel(
                    status=self.state.cloud_status,
                    threat_model=self.state.threat_model,
                    invariants=self.state.invariants,
                )
            )

            is_a_winner = self.state.critic_winning_builder == "Builder A"
            is_b_winner = self.state.critic_winning_builder == "Builder B"

            self.layout["builder_a"].update(
                build_builder_panel(
                    name="Builder A (Surgical Minimal)",
                    code=self.state.builder_a_code,
                    status=self.state.builder_a_status,
                    is_winner=is_a_winner,
                )
            )

            self.layout["builder_b"].update(
                build_builder_panel(
                    name="Builder B (Defense-in-Depth)",
                    code=self.state.builder_b_code,
                    status=self.state.builder_b_status,
                    is_winner=is_b_winner,
                )
            )

            self.layout["critic"].update(
                build_critic_panel(
                    status=self.state.critic_status,
                    alpha_score=self.state.critic_alpha_score,
                    beta_score=self.state.critic_beta_score,
                    winner=self.state.critic_winner,
                    winning_builder=self.state.critic_winning_builder,
                    critique=self.state.critic_critique,
                )
            )

            self.layout["telemetry"].update(
                build_telemetry_table(self.state.telemetry_summary)
            )

            self.layout["footer"].update(
                build_gauntlet_panel(
                    test_results=self.state.test_results,
                    privacy_verified=self.state.privacy_verified,
                )
            )
        except Exception:
            pass

    def refresh(self) -> None:
        """Triggers a re-render of layout into active Live display."""
        if not self._is_running:
            return
        self._render_into_layout()
        if self._live is not None:
            self._live.refresh()

    def update(self, state: Optional[DashboardState] = None) -> None:
        """Updates internal state and refreshes HUD."""
        if state is not None:
            self.state = state
        self.refresh()

    def set_target(self, target: str) -> None:
        self.state.current_target = target
        self.refresh()

    def set_pipeline_status(self, status: str) -> None:
        self.state.pipeline_status = status
        self.refresh()

    def update_cloud_strategy(
        self,
        threat_model: str,
        invariants: List[str],
        status: str = "READY",
    ) -> None:
        self.state.cloud_status = status
        self.state.threat_model = threat_model
        self.state.invariants = invariants
        self.refresh()
        if self.no_hud:
            self.console.print(f"[Cloud Architect] Strategy generated: {threat_model}")

    def update_builder_a(
        self,
        chunk: str,
        accumulated: str,
        status: str = "GENERATING",
    ) -> None:
        self.state.builder_a_status = status
        self.state.builder_a_code = accumulated
        self.refresh()

    def update_builder_b(
        self,
        chunk: str,
        accumulated: str,
        status: str = "GENERATING",
    ) -> None:
        self.state.builder_b_status = status
        self.state.builder_b_code = accumulated
        self.refresh()

    def update_critic(
        self,
        critique: str,
        winner: str,
        winning_builder: str,
        alpha_score: float,
        beta_score: float,
        status: str = "READY",
    ) -> None:
        self.state.critic_status = status
        self.state.critic_critique = critique
        self.state.critic_winner = winner
        self.state.critic_winning_builder = winning_builder
        self.state.critic_alpha_score = alpha_score
        self.state.critic_beta_score = beta_score
        self.refresh()
        if self.no_hud and winner:
            self.console.print(
                f"[Blind Critic] Selected: {winner} ({winning_builder}) "
                f"Alpha: {alpha_score}/10, Beta: {beta_score}/10"
            )

    def update_critic_decision(
        self,
        critique_or_result: object,
        winner: str = "",
        winning_builder: str = "",
        alpha_score: float = 0.0,
        beta_score: float = 0.0,
        status: str = "READY",
    ) -> None:
        """Alias / flexible interface for critic updates."""
        if hasattr(critique_or_result, "winner"):
            self.update_critic(
                critique=getattr(critique_or_result, "critique", ""),
                winner=getattr(critique_or_result, "winner", ""),
                winning_builder=getattr(critique_or_result, "winning_builder", ""),
                alpha_score=getattr(critique_or_result, "alpha_score", 0.0),
                beta_score=getattr(critique_or_result, "beta_score", 0.0),
                status=status,
            )
        else:
            self.update_critic(
                critique=str(critique_or_result),
                winner=winner,
                winning_builder=winning_builder,
                alpha_score=alpha_score,
                beta_score=beta_score,
                status=status,
            )

    def update_telemetry(self, summary: TelemetrySummary) -> None:
        self.state.telemetry_summary = summary
        self.refresh()

    def update_test_results(self, target: str, functional: str, adversarial: str) -> None:
        self.state.test_results[target] = (functional, adversarial)
        self.refresh()
        if self.no_hud:
            self.console.print(f"[Gauntlet] {target} -> Functional: {functional}, Adversarial: {adversarial}")

    def print_final_summary(
        self,
        summary: TelemetrySummary,
        test_results: Optional[Dict[str, Tuple[str, str]]] = None,
        privacy_verified: bool = True,
        duration_s: float = 0.0,
        target_filter: Optional[List[str]] = None,
    ) -> None:
        """Prints the permanent post-run summary after display finishes."""
        results = test_results if test_results is not None else self.state.test_results
        render_final_summary(
            console=self.console,
            summary=summary,
            test_results=results,
            privacy_verified=privacy_verified,
            duration_s=duration_s,
            target_filter=target_filter,
        )


# Compatibility alias with PROJECT.md
DashboardManager = DashboardController
