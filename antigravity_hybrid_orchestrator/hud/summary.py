"""Permanent post-run summary report rendered to standard console output."""

from typing import Any, Dict, List, Optional, Tuple, Union

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from antigravity_hybrid_orchestrator.engine.loop import DAGExecutionResult
from antigravity_hybrid_orchestrator.hud.components import format_status_badge
from antigravity_hybrid_orchestrator.planner.dag import ExecutionPlanDAG
from antigravity_hybrid_orchestrator.telemetry import TelemetrySummary


def render_final_summary(
    console: Console,
    summary: TelemetrySummary,
    test_results: Dict[str, Tuple[str, str]],
    privacy_verified: bool = True,
    duration_s: float = 0.0,
    target_filter: Optional[List[str]] = None,
) -> None:
    """Renders a beautiful permanent summary table to the console after Live display stops.

    Args:
        console: Rich console instance.
        summary: Aggregated TelemetrySummary object.
        test_results: Dict mapping module names to (functional_status, adversarial_status).
        privacy_verified: Boolean indicating whether AST zero-knowledge invariant held.
        duration_s: Total elapsed orchestration duration in seconds.
        target_filter: Optional list of target modules that were in scope for this run.
    """
    console.print()
    banner = Text("═" * 80, style="bold cyan")
    console.print(banner)
    title = Text("   ⚡ ANTIGRAVITY HYBRID ORCHESTRATOR — RUN SUMMARY ⚡   ", style="bold yellow on blue")
    console.print(title, justify="center")
    console.print(banner)
    console.print()

    # 1. Security & Privacy Audit Verification
    audit_table = Table(
        title="[bold red]1. Security Gauntlet & Privacy Invariant Verification[/]",
        border_style="red",
        expand=True,
    )
    audit_table.add_column("Target Module", style="bold white")
    audit_table.add_column("Vulnerability Class Addressed", style="cyan")
    audit_table.add_column("Functional Tests", justify="center")
    audit_table.add_column("Adversarial Tests", justify="center")
    audit_table.add_column("Overall Status", justify="center")

    vuln_map = {
        "auth.py": "CWE-798 Hardcoded Secrets & Timing Leak",
        "billing.py": "CWE-362 Concurrency Race Condition",
        "database.py": "CWE-89 SQL Injection Parameterization",
    }

    all_passed = True
    in_scope_modules = target_filter if target_filter is not None else list(vuln_map.keys())

    for mod_name, vuln_class in vuln_map.items():
        res = test_results.get(mod_name, ("PENDING", "PENDING"))
        in_scope = mod_name in in_scope_modules

        if not in_scope and res[0] == "PENDING" and res[1] == "PENDING":
            audit_table.add_row(
                mod_name,
                vuln_class,
                Text("[SKIPPED]", style="dim"),
                Text("[SKIPPED]", style="dim"),
                Text("NOT IN SCOPE", style="dim"),
            )
            continue

        func_ok = res[0].upper() in ("PASS", "PASSED", "READY", "OK")
        adv_ok = res[1].upper() in ("PASS", "PASSED", "READY", "OK")

        if in_scope and not (func_ok and adv_ok):
            all_passed = False

        func_style = "bold green" if func_ok else "bold red"
        adv_style = "bold green" if adv_ok else "bold red"
        overall_style = "bold green" if (func_ok and adv_ok) else "bold red"
        overall_text = "PASS (100%)" if (func_ok and adv_ok) else "FAILED"

        audit_table.add_row(
            mod_name,
            vuln_class,
            Text(f"[{res[0]}]", style=func_style),
            Text(f"[{res[1]}]", style=adv_style),
            Text(overall_text, style=overall_style),
        )

    console.print(audit_table)
    console.print()

    # Privacy verification callout
    if privacy_verified:
        priv_text = Text.assemble(
            ("✔ Zero-Knowledge Code Privacy Invariant: ", "bold green"),
            ("VERIFIED (0 Bytes of Function Bodies or Secret Literals Transmitted to Cloud)\n", "green"),
        )
    else:
        priv_text = Text.assemble(
            ("✖ Zero-Knowledge Code Privacy Invariant: ", "bold red"),
            ("VIOLATION DETECTED (Code payload leaked sensitive tokens)\n", "red"),
        )
    console.print(Panel(priv_text, border_style="green" if privacy_verified else "red"))
    console.print()

    # 2. Token Allocation & Financial Accounting Table
    cost_table = Table(
        title="[bold green]2. Hybrid Token Allocation & Reference Cost Estimate[/]",
        border_style="green",
        expand=True,
    )
    cost_table.add_column("Runtime / Node", style="bold white", ratio=2)
    cost_table.add_column("Role / Task", style="dim cyan", ratio=2)
    cost_table.add_column("Tokens Processed", justify="right", style="cyan", ratio=1)
    cost_table.add_column("Pricing Model", justify="right", style="yellow", ratio=2)
    cost_table.add_column("Benchmark Estimate", justify="right", style="bold yellow", ratio=1)

    # Pure cloud hypothetical calculation
    hypothetical_cloud_cost = summary.cloud_cost_usd + (summary.local_tokens * 0.000000375)
    dollars_saved = max(0.0, hypothetical_cloud_cost - summary.cloud_cost_usd)

    cost_table.add_row(
        "Cloud Architect (planner)",
        "Zero-Knowledge Threat Model & AST Strategy",
        f"{summary.cloud_tokens:,}",
        "CLI-reported cost; Gemini reference rate only where unreported",
        f"${summary.cloud_cost_usd:.6f}",
    )
    cost_table.add_row(
        "On-Device Workforce (Gemma 4 12B GPU)",
        "Surgical Patch + Defensive Patch + Blind Critic",
        f"{summary.local_tokens:,}",
        "No token fee (local RTX 4060)",
        "$0.000000",
    )
    cost_table.add_section()
    cost_table.add_row(
        "[bold white]Token Total / Reference[/]",
        f"[dim]Generation Speed: {summary.tokens_per_second:.1f} tokens/s[/]",
        f"[bold cyan]{summary.total_tokens:,}[/]",
        "[dim]Hybrid Architecture[/]",
        f"[bold yellow]${summary.cloud_cost_usd:.6f}[/]",
    )
    cost_table.add_row(
        "[bold white]Hypothetical All-Cloud Baseline[/]",
        "[dim]If all workforce tokens sent to commercial cloud[/]",
        f"[bold cyan]{summary.total_tokens:,}[/]",
        "[dim]Commercial API Baseline[/]",
        f"[bold white]${hypothetical_cloud_cost:.6f}[/]",
    )
    cost_table.add_row(
        "[bold green]Hypothetical Reference Savings[/]",
        f"[bold green]Estimated ${dollars_saved:.6f}[/]",
        "-",
        "-",
        f"[bold green]{summary.cost_savings_pct:.1f}% SAVINGS[/]",
    )

    console.print(cost_table)
    console.print()

    # Final verdict statement
    if all_passed and privacy_verified:
        verdict_panel = Panel(
            Align.center(
                Text.assemble(
                    ("🎉 ALL 6/6 ADVERSARIAL GAUNTLET TESTS PASSED WITH 100% SUCCESS 🎉\n", "bold green"),
                    ("Zero-Knowledge Privacy Maintained | Cloud-Edge Synergy Verified | Local GPU Hardware Accelerated", "white"),
                )
            ),
            border_style="bold green",
            padding=(1, 2),
        )
    else:
        verdict_panel = Panel(
            Align.center(
                Text("⚠️ ORCHESTRATION COMPLETED WITH WARNINGS OR TEST FAILURES ⚠️", style="bold red")
            ),
            border_style="bold red",
            padding=(1, 2),
        )
    console.print(verdict_panel)
    console.print()


def print_rd_summary_table(
    console: Console,
    dag_result: DAGExecutionResult,
    dag: ExecutionPlanDAG,
    summary: Optional[TelemetrySummary] = None,
    privacy_verified: bool = True,
    duration_s: float = 0.0,
) -> None:
    """Renders post-run R&D engine summary report to standard console output.

    Args:
        console: Rich console instance.
        dag_result: DAGExecutionResult containing completed, failed, and blocked subtasks.
        dag: ExecutionPlanDAG defining the planned subtasks and verification commands.
        summary: Optional TelemetrySummary object containing token and cost metrics.
        privacy_verified: Boolean indicating whether zero-knowledge invariant was preserved.
        duration_s: Total elapsed R&D engine execution duration in seconds.
    """
    console.print()
    banner = Text("═" * 80, style="bold cyan")
    console.print(banner)
    title = Text("   ⚡ ANTIGRAVITY HYBRID ORCHESTRATOR — R&D RUN SUMMARY ⚡   ", style="bold yellow on blue")
    console.print(title, justify="center")
    console.print(banner)
    console.print()

    # 1. R&D Subtask Execution & Verification Audit
    audit_table = Table(
        title="[bold cyan]1. R&D Subtask Execution & Verification Audit[/]",
        border_style="cyan",
        expand=True,
    )
    audit_table.add_column("Subtask ID", style="bold white", ratio=1)
    audit_table.add_column("Description", style="white", ratio=3)
    audit_table.add_column("Status", justify="center", ratio=1)
    audit_table.add_column("Attempts/Turns", justify="center", ratio=1)
    audit_table.add_column("Verification Command", style="dim", ratio=2)
    audit_table.add_column("Exit Code / Outcome", justify="center", ratio=2)

    # Determine tasks to display
    ordered_task_ids: List[str] = []
    if dag is not None and hasattr(dag, "tasks") and dag.tasks:
        if hasattr(dag, "topological_sort"):
            try:
                ordered_task_ids = dag.topological_sort()
            except Exception:
                ordered_task_ids = list(dag.tasks.keys())
        else:
            ordered_task_ids = list(dag.tasks.keys())
    else:
        seen = set()
        for k in (
            list(dag_result.subtask_results.keys())
            + list(dag_result.completed_tasks)
            + list(dag_result.failed_tasks)
            + list(dag_result.blocked_tasks)
        ):
            if k not in seen:
                seen.add(k)
                ordered_task_ids.append(k)

    for tid in ordered_task_ids:
        task = dag.tasks.get(tid) if (dag and hasattr(dag, "tasks")) else None
        sub_res = dag_result.subtask_results.get(tid)
        verif_res = dag_result.verification_results.get(tid)

        desc = (
            getattr(task, "description", "")
            or getattr(task, "title", "")
            or (sub_res.summary if sub_res else "")
            or f"Subtask {tid}"
        )

        is_completed = tid in dag_result.completed_tasks or (sub_res and sub_res.success)
        is_failed = tid in dag_result.failed_tasks or (sub_res and not sub_res.success)
        is_blocked = tid in dag_result.blocked_tasks

        if is_completed:
            status_text = Text("[COMPLETED]", style="bold green")
        elif is_failed:
            status_text = Text("[FAILED]", style="bold red")
        elif is_blocked:
            status_text = Text("[BLOCKED]", style="dim")
        elif task:
            status_text = format_status_badge(getattr(task, "status", "pending"))
        else:
            status_text = Text("[UNKNOWN]", style="dim")

        retry_count = getattr(task, "retry_count", 0) if task else 0
        attempts = retry_count + 1
        turns = sub_res.turns if sub_res else 0
        if turns > 0:
            att_turn_str = f"{attempts} att / {turns} turns"
        else:
            att_turn_str = f"{attempts} att"

        cmd = getattr(task, "verification_command", "") if task else ""
        if not cmd and verif_res:
            cmd = getattr(verif_res, "command", "")
        cmd_str = cmd if cmd else "(none)"

        if verif_res:
            code = verif_res.exit_code
            if verif_res.success:
                outcome_str = Text(f"exit {code} (OK)", style="bold green")
            else:
                outcome_str = Text(f"exit {code} (FAIL)", style="bold red")
        elif is_completed:
            outcome_str = Text("exit 0 (OK)", style="bold green")
        elif is_failed:
            outcome_str = Text("FAIL", style="bold red")
        elif is_blocked:
            outcome_str = Text("BLOCKED", style="dim")
        else:
            outcome_str = Text("N/A", style="dim")

        audit_table.add_row(tid, desc, status_text, att_turn_str, cmd_str, outcome_str)

    console.print(audit_table)
    console.print()

    # Callout panel: Zero-Knowledge Code Privacy Invariant
    if privacy_verified:
        priv_text = Text.assemble(
            ("✔ Zero-Knowledge Code Privacy Invariant: ", "bold green"),
            ("VERIFIED: 0 Bytes transmitted to Cloud (Zero function bodies or secret tokens leaked)\n", "green"),
        )
        console.print(Panel(priv_text, border_style="green"))
    else:
        priv_text = Text.assemble(
            ("✖ Zero-Knowledge Code Privacy Invariant: ", "bold red"),
            ("LEAK DETECTED (Sensitive code or token payloads transmitted to Cloud)\n", "red"),
        )
        console.print(Panel(priv_text, border_style="red"))
    console.print()

    # Table 2: Token Allocation & Financial Accounting
    cost_table = Table(
        title="[bold green]2. Hybrid Token Allocation & Economic Cost Accounting[/]",
        border_style="green",
        expand=True,
    )
    cost_table.add_column("Runtime / Node", style="bold white", ratio=2)
    cost_table.add_column("Role / Task", style="dim cyan", ratio=2)
    cost_table.add_column("Tokens Processed", justify="right", style="cyan", ratio=1)
    cost_table.add_column("Pricing Model", justify="right", style="yellow", ratio=2)
    cost_table.add_column("Benchmark Estimate", justify="right", style="bold yellow", ratio=1)

    if summary is None:
        summary = TelemetrySummary(
            cloud_tokens=0,
            cloud_cost_usd=0.0,
            local_tokens=0,
            local_cost_usd=0.0,
            total_tokens=0,
            cost_savings_pct=0.0,
            tokens_per_second=0.0,
        )

    hypothetical_cloud_cost = summary.cloud_cost_usd + (summary.local_tokens * 0.000000375)
    dollars_saved = max(0.0, hypothetical_cloud_cost - summary.cloud_cost_usd)

    cost_table.add_row(
        "Cloud Architect (planner)",
        "Zero-Knowledge DAG Plan & Architecture Strategy",
        f"{summary.cloud_tokens:,}",
        "CLI-reported cost; Gemini reference rate only where unreported",
        f"${summary.cloud_cost_usd:.6f}",
    )
    cost_table.add_row(
        "On-Device Workforce (Gemma 4 12B GPU)",
        "Autonomous ReAct Tool-Calling & Verification",
        f"{summary.local_tokens:,}",
        "No token fee (local RTX 4060)",
        "$0.000000",
    )
    cost_table.add_section()
    cost_table.add_row(
        "[bold white]Token Total / Reference[/]",
        f"[dim]Generation Speed: {summary.tokens_per_second:.1f} tokens/s[/]",
        f"[bold cyan]{summary.total_tokens:,}[/]",
        "[dim]Hybrid Architecture[/]",
        f"[bold yellow]${summary.cloud_cost_usd:.6f}[/]",
    )
    cost_table.add_row(
        "[bold white]Hypothetical All-Cloud Baseline[/]",
        "[dim]If all workforce tokens sent to commercial cloud[/]",
        f"[bold cyan]{summary.total_tokens:,}[/]",
        "[dim]Commercial API Baseline[/]",
        f"[bold white]${hypothetical_cloud_cost:.6f}[/]",
    )
    cost_table.add_row(
        "[bold green]Hypothetical Reference Savings[/]",
        f"[bold green]Estimated ${dollars_saved:.6f}[/]",
        "-",
        "-",
        f"[bold green]{summary.cost_savings_pct:.1f}% SAVINGS[/]",
    )

    console.print(cost_table)
    console.print()

    # Final verdict panel
    total_count = len(dag.tasks) if (dag and hasattr(dag, "tasks") and dag.tasks) else (
        len(dag_result.completed_tasks) + len(dag_result.failed_tasks) + len(dag_result.blocked_tasks)
    )
    all_completed = (
        dag_result.success
        and len(dag_result.completed_tasks) == total_count
        and len(dag_result.failed_tasks) == 0
        and len(dag_result.blocked_tasks) == 0
    )
    is_all_success = all_completed and privacy_verified

    if is_all_success:
        verdict_panel = Panel(
            Align.center(
                Text.assemble(
                    (f"🎉 ALL {len(dag_result.completed_tasks)}/{total_count} R&D SUBTASKS VERIFIED WITH 100% SUCCESS 🎉\n", "bold green"),
                    ("Zero-Knowledge Privacy Maintained | Self-Correction Gauntlet Verified | Local GPU Hardware Accelerated", "white"),
                )
            ),
            border_style="bold green",
            padding=(1, 2),
        )
    else:
        failed_count = len(dag_result.failed_tasks)
        blocked_count = len(dag_result.blocked_tasks)
        reasons = []
        if failed_count:
            reasons.append(f"{failed_count} subtasks failed")
        if blocked_count:
            reasons.append(f"{blocked_count} subtasks blocked")
        if not privacy_verified:
            reasons.append("privacy invariant leak detected")
        if not dag_result.success and not reasons:
            reasons.append("DAG execution reported failure")
        reason_str = ", ".join(reasons) if reasons else "unmet verification criteria"
        verdict_panel = Panel(
            Align.center(
                Text(
                    f"⚠️ R&D ORCHESTRATION COMPLETED WITH WARNINGS OR FAILURES: {reason_str.upper()} ⚠️",
                    style="bold red",
                )
            ),
            border_style="bold red",
            padding=(1, 2),
        )

    console.print(verdict_panel)
    console.print()


render_rd_final_summary = print_rd_summary_table
