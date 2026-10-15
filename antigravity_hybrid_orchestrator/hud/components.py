"""Rich UI component builders for terminal HUD panels and tables."""

from typing import Dict, List, Optional, Tuple

from rich.align import Align
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from antigravity_hybrid_orchestrator.telemetry import TelemetrySummary


def format_status_badge(status: str) -> Text:
    """Format an agent status string into a styled badge."""
    upper = (status or "IDLE").upper()
    if "READY" in upper or "DONE" in upper or "SUCCESS" in upper or "PASS" in upper:
        return Text(f"[{upper}]", style="bold green")
    if "STREAM" in upper or "GEN" in upper or "REASON" in upper or "EVAL" in upper:
        return Text(f"[{upper} ⚡]", style="bold yellow")
    if "WAIT" in upper or "QUEUE" in upper or "PEND" in upper:
        return Text(f"[{upper}]", style="dim cyan")
    if "FAIL" in upper or "ERROR" in upper:
        return Text(f"[{upper}]", style="bold red")
    return Text(f"[{upper}]", style="dim white")


def build_header_panel(
    current_target: str = "all",
    mode: str = "live",
    overall_status: str = "RUNNING",
) -> Panel:
    """Build the top header banner."""
    grid = Table.grid(expand=True)
    grid.add_column(justify="left", ratio=2)
    grid.add_column(justify="center", ratio=3)
    grid.add_column(justify="right", ratio=2)

    title = Text("⚡ ANTIGRAVITY HYBRID ORCHESTRATOR ⚡", style="bold cyan")
    subtitle = Text(
        f"Mode: {mode.upper()} | Target: {current_target} | Cloud: Codex CLI | GPU: Gemma 4 12B (RTX 4060)",
        style="dim white",
    )
    status_text = Text.assemble(
        ("Pipeline: ", "dim"),
        format_status_badge(overall_status),
    )

    grid.add_row(title, subtitle, status_text)
    return Panel(grid, style="bold blue", border_style="cyan", padding=(0, 1))


def build_cloud_panel(
    status: str,
    threat_model: str = "",
    invariants: Optional[List[str]] = None,
) -> Panel:
    """Build Cloud Chief Architect strategy and threat model panel."""
    invariants = invariants or []
    content = Table.grid(expand=True, padding=(0, 1))
    content.add_column(ratio=1)

    status_line = Text.assemble(
        ("Status: ", "bold"),
        format_status_badge(status),
        ("  (AST Zero-Knowledge Stripped)", "dim green"),
    )
    content.add_row(status_line)
    content.add_row(Text(""))

    # Threat Model
    tm_text = threat_model if threat_model else "(Awaiting AST interface ingestion...)"
    content.add_row(Text("Threat Model Identified:", style="bold magenta"))
    content.add_row(Text(f"  {tm_text}", style="italic white"))
    content.add_row(Text(""))

    # Invariants
    content.add_row(Text("Mandated Security Invariants:", style="bold yellow"))
    if invariants:
        for inv in invariants[:4]:  # Show up to 4 invariants
            content.add_row(Text(f"  • {inv}", style="white"))
    else:
        content.add_row(Text("  (None generated yet)", style="dim"))

    return Panel(
        content,
        title="[bold blue]☁ Cloud Chief Architect (Codex CLI)[/]",
        border_style="blue",
        padding=(0, 1),
    )


def build_builder_panel(
    name: str,
    code: str,
    status: str,
    is_winner: bool = False,
    anonymized_label: Optional[str] = None,
) -> Panel:
    """Build a code candidate view for Builder A or Builder B."""
    title_suffix = f" — {anonymized_label}" if anonymized_label else ""
    if is_winner:
        title = f"[bold green]🏆 {name}{title_suffix} (WINNER)[/]"
        border_color = "green"
    elif "STREAM" in status.upper() or "GEN" in status.upper():
        title = f"[bold yellow]⚡ {name}{title_suffix} (ACTIVE)[/]"
        border_color = "yellow"
    else:
        title = f"[bold white]{name}{title_suffix}[/]"
        border_color = "bright_black"

    status_badge = format_status_badge(status)

    if not code.strip():
        display_content = Align.center(
            Text(f"\n\n[Waiting for generation... {status}]", style="dim italic"),
            vertical="middle",
        )
    else:
        # Keep recent lines visible for streaming effect
        lines = code.splitlines()
        max_lines = 16
        if len(lines) > max_lines:
            truncated_code = "\n".join(lines[-max_lines:])
        else:
            truncated_code = code

        try:
            display_content = Syntax(
                truncated_code,
                "python",
                theme="monokai",
                line_numbers=True,
                word_wrap=True,
            )
        except Exception:
            display_content = Text(truncated_code)

    panel_content = Table.grid(expand=True)
    panel_content.add_column(ratio=1)
    status_row = Table.grid(expand=True)
    status_row.add_column(justify="left", ratio=1)
    status_row.add_column(justify="right", ratio=1)
    status_row.add_row(Text(f"Agent: {name}", style="dim"), status_badge)

    panel_content.add_row(status_row)
    panel_content.add_row(display_content)

    return Panel(panel_content, title=title, border_style=border_color, padding=(0, 1))


def build_critic_panel(
    status: str,
    alpha_score: float = 0.0,
    beta_score: float = 0.0,
    winner: str = "",
    winning_builder: str = "",
    critique: str = "",
) -> Panel:
    """Build the Blind Critic scorecard and decision panel."""
    content = Table.grid(expand=True, padding=(0, 1))
    content.add_column(ratio=1)

    header_table = Table.grid(expand=True)
    header_table.add_column(justify="left", ratio=1)
    header_table.add_column(justify="right", ratio=1)

    status_line = Text.assemble(
        ("Blind Critic Evaluation: ", "bold"),
        format_status_badge(status),
    )
    scores_text = Text.assemble(
        ("Candidate Alpha: ", "bold"),
        (f"{alpha_score:.1f}/10  " if alpha_score > 0 else "--/10  ", "cyan"),
        ("Candidate Beta: ", "bold"),
        (f"{beta_score:.1f}/10", "magenta"),
    )
    header_table.add_row(status_line, scores_text)
    content.add_row(header_table)
    content.add_row(Text(""))

    if winner:
        verdict = Text.assemble(
            ("VERDICT: ", "bold yellow"),
            (f"{winner} ", "bold green"),
            (f"({winning_builder})", "bold cyan"),
            (" selected for gauntlet deployment", "white"),
        )
        content.add_row(verdict)
    else:
        content.add_row(Text("Awaiting candidate patches for double-blind adversarial review...", style="dim"))

    if critique:
        short_critique = critique if len(critique) < 240 else critique[:237] + "..."
        content.add_row(Text(f"Critique: {short_critique}", style="italic dim"))

    return Panel(
        content,
        title="[bold magenta]⚖ Blind Critic (Adversarial Evaluator)[/]",
        border_style="magenta",
        padding=(0, 1),
    )


def build_telemetry_table(summary: Optional[TelemetrySummary]) -> Panel:
    """Build the real-time token, throughput, and financial cost table."""
    table = Table(box=None, expand=True, padding=(0, 1))
    table.add_column("Runtime / Agent", style="bold white", ratio=2)
    table.add_column("Tokens", justify="right", style="cyan", ratio=1)
    table.add_column("Speed", justify="right", style="green", ratio=1)
    table.add_column("Benchmark Cost (USD)", justify="right", style="yellow", ratio=1)

    if summary is None:
        table.add_row("Cloud Architect (Codex CLI)", "0", "-", "$0.000000")
        table.add_row("On-Device (Gemma 4 GPU)", "0", "0.0 tps", "$0.000000")
        table.add_row("Total / Benchmark", "0", "-", "Est. saved 0.0%")
    else:
        table.add_row(
            "Cloud Architect (Codex CLI)",
            f"{summary.cloud_tokens:,}",
            "API",
            f"${summary.cloud_cost_usd:.6f}",
        )
        table.add_row(
            "On-Device (Gemma 4 GPU)",
            f"{summary.local_tokens:,}",
            f"{summary.tokens_per_second:.1f} tps",
            "No token fee (local)",
        )
        savings_color = "bold green" if summary.cost_savings_pct > 0 else "dim"
        table.add_row(
            "[bold]Combined Hybrid[/]",
            f"[bold]{summary.total_tokens:,}[/]",
            f"[bold]{summary.tokens_per_second:.1f} tps[/]",
            f"[{savings_color}]Est. saved {summary.cost_savings_pct:.1f}%[/]",
        )

    return Panel(
        table,
        title="[bold green]📊 Token & Reference Cost Telemetry[/]",
        border_style="green",
        padding=(0, 1),
    )


def build_gauntlet_panel(
    test_results: Dict[str, Tuple[str, str]],
    privacy_verified: bool = True,
) -> Panel:
    """Build the bottom footer panel displaying test matrix and privacy verification."""
    table = Table(box=None, expand=True, padding=(0, 2))
    table.add_column("Target Module", style="bold cyan", ratio=2)
    table.add_column("Functional Tests", justify="center", ratio=2)
    table.add_column("Adversarial Tests", justify="center", ratio=2)
    table.add_column("Zero-Knowledge Privacy", justify="center", ratio=2)

    modules = [
        ("auth.py", "Hardcoded Secret & Timing Leak"),
        ("billing.py", "10-Thread Overdraft Race Condition"),
        ("database.py", "SQL Injection Extraction Bypass"),
    ]

    for filename, attack_desc in modules:
        res = test_results.get(filename, ("PENDING", "PENDING"))
        func_badge = format_status_badge(res[0])
        adv_badge = format_status_badge(res[1])
        priv_badge = (
            Text("[VERIFIED: 0 B]", style="bold green")
            if privacy_verified
            else Text("[LEAK DETECTED]", style="bold red")
        )

        table.add_row(
            Text.assemble((filename, "bold white"), (f" ({attack_desc})", "dim")),
            func_badge,
            adv_badge,
            priv_badge,
        )

    return Panel(
        table,
        title="[bold red]🛡 Security Gauntlet Verification Matrix[/]",
        border_style="red",
        padding=(0, 1),
    )
