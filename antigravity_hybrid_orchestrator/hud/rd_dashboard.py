"""Interactive Rich Terminal HUD for autonomous R&D engine (Milestone 5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Union

from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from antigravity_hybrid_orchestrator.engine.loop import SubtaskExecutionState
from antigravity_hybrid_orchestrator.hud.components import (
    build_telemetry_table,
    format_status_badge,
)
from antigravity_hybrid_orchestrator.hud import theme as T
from antigravity_hybrid_orchestrator.planner.dag import ExecutionPlanDAG, SubTask
from antigravity_hybrid_orchestrator.telemetry import TelemetrySummary


@dataclass
class RDDashboardState:
    """Observable state container for autonomous R&D engine HUD visualization."""

    goal: str = ""
    workspace_root: str = ""
    mode: str = "mock"
    pipeline_status: str = "INITIALIZING"
    subtasks: List[Dict[str, Any]] = field(default_factory=list)
    active_subtask_id: Optional[str] = None
    active_subtask_desc: str = ""
    active_subtask_cmd: str = ""
    worker_status: str = "IDLE"
    worker_thought: str = ""
    active_tool_call: Optional[str] = None
    active_tool_output: Optional[str] = None
    recent_events: List[str] = field(default_factory=list)
    privacy_verified: bool = True
    telemetry_summary: Optional[TelemetrySummary] = None
    elapsed_s: float = 0.0
    # Extended observability
    tool_log: List[Dict[str, Any]] = field(default_factory=list)
    git_checkpoints: List[str] = field(default_factory=list)
    last_verification: Optional[Dict[str, Any]] = None


class RDLayout(Layout):
    """Custom Layout enabling flexible key lookups ('dag' -> 'subtasks', 'tools' -> 'events')."""

    def __getitem__(self, name: str) -> Layout:
        lookup = name
        if name in ("dag", "dag_subtasks", "tasks"):
            lookup = "subtasks"
        elif name in ("tools", "tools_activity", "activity"):
            lookup = "events"
        try:
            return super().__getitem__(lookup)
        except KeyError:
            return super().__getitem__(name)


def build_rd_layout(terminal_width: int = 120) -> Layout:
    """Constructs the responsive hierarchical Rich Layout for the R&D engine.

    Layout Tree:
    - Root:
      - header (size: 3)
      - body (ratio: 1)
        - left (ratio: 1)
          - subtasks (ratio: 2)
          - telemetry (ratio: 1)
        - right (ratio: 1)
          - worker (ratio: 2)
          - events (ratio: 1)
      - footer (size: 3)

    If terminal width < 100, left/right body columns are stacked vertically.
    """
    root = RDLayout(name="root")

    compact = terminal_width < 100
    header = Layout(name="header", size=4 if compact else 3)
    body = Layout(name="body", ratio=1)
    footer = Layout(name="footer", size=4 if compact else 3)

    root.split_column(header, body, footer)

    left = Layout(name="left", ratio=1)
    right = Layout(name="right", ratio=1)

    if terminal_width < 100:
        body.split_column(left, right)
    else:
        body.split_row(left, right)

    left.split_column(
        Layout(name="subtasks", ratio=2),
        Layout(name="telemetry", ratio=1),
    )

    right.split_column(
        Layout(name="worker", ratio=2),
        Layout(name="events", ratio=1),
    )

    return root


# ==============================================================================
# UI COMPONENT BUILDERS
# ==============================================================================


def _build_rd_header(state: RDDashboardState, compact: bool = False) -> Panel:
    """Builds the neon header banner: wordmark, goal/workspace/mode, status."""
    title = Text(T.WORDMARK, style=f"bold {T.CYAN}")
    goal_limit = 32 if compact else 46
    goal_display = state.goal if len(state.goal) <= goal_limit else state.goal[: goal_limit - 3] + "…"
    workspace_name = Path(state.workspace_root).name if state.workspace_root else "default"
    if compact and len(workspace_name) > 12:
        workspace_name = workspace_name[:11] + "…"
    subtitle = Text.assemble(
        ("◤ ", f"bold {T.MAGENTA}"),
        (goal_display or "(no goal)", f"{T.GHOST}"),
        ("  ◇ ", T.DIM),
        (workspace_name, T.BLUE),
        ("  ◇ ", T.DIM),
        (state.mode.upper(), f"bold {T.PURPLE}"),
    )
    status_text = Text.assemble(
        ("▓ ", T.DIM),
        T.neon_badge(state.pipeline_status),
        (f"  {state.elapsed_s:5.1f}s", f"bold {T.AMBER}"),
    )
    if compact:
        status_text = Text.assemble(
            T.neon_badge(state.pipeline_status),
            (f" {state.elapsed_s:.0f}s", f"bold {T.AMBER}"),
        )
    grid = Table.grid(expand=True)
    if compact:
        grid.add_column(justify="left", ratio=3)
        grid.add_column(justify="right", ratio=1)
        grid.add_row(title, status_text)
        grid.add_row(subtitle, Text("", style=T.DIM))
    else:
        grid.add_column(justify="left", ratio=3)
        grid.add_column(justify="center", ratio=3)
        grid.add_column(justify="right", ratio=2)
        grid.add_row(title, subtitle, status_text)
    return Panel(grid, box=T.BOX, border_style=T.CYAN, padding=(0, 1))


def _build_rd_subtasks_panel(state: RDDashboardState) -> Panel:
    """Builds the DAG subtasks panel with neon status glyphs and dependency arrows."""
    title = f"[bold {T.CYAN}]▚ EXECUTION DAG[/]"
    if not state.subtasks:
        display = Align.center(
            Text("\n◈ awaiting plan from Cloud Architect ◈", style=f"italic {T.DIM}"),
            vertical="middle",
        )
        return Panel(display, title=title, box=T.BOX, border_style=T.CYAN, padding=(0, 1))

    table = Table(box=None, expand=True, padding=(0, 1))
    table.add_column("", justify="center", ratio=1)
    table.add_column("Node", style=f"bold {T.GHOST}", ratio=3)
    table.add_column("Status", justify="left", ratio=3)
    table.add_column("Description", style="white", ratio=6)
    table.add_column("Att", justify="center", style=T.AMBER, ratio=1)
    table.add_column("Deps", justify="left", style=T.DIM, ratio=3)

    for sub in state.subtasks:
        sub_id = str(sub.get("id", ""))
        is_active = sub_id == state.active_subtask_id
        status_raw = str(sub.get("status", "pending"))
        badge = T.neon_badge(status_raw)
        desc = str(sub.get("description", sub.get("title", "")))
        if len(desc) > 40:
            desc = desc[:37] + "…"
        attempts = str(sub.get("attempts", sub.get("retry_count", 0)))
        deps = sub.get("dependencies", []) or []
        deps_str = "→ " + ",".join(str(d) for d in deps) if deps else "root"

        marker = Text("▶", style=f"bold {T.MAGENTA}") if is_active else Text(" ")
        id_style = f"bold {T.CYAN}" if is_active else T.GHOST
        table.add_row(marker, Text(sub_id, style=id_style), badge, desc, attempts, deps_str)

    return Panel(table, title=title, box=T.BOX, border_style=T.CYAN, padding=(0, 1))


def _build_rd_telemetry_panel(summary: Optional[TelemetrySummary]) -> Panel:
    """Backward-compatible telemetry table (used directly by some callers)."""
    return build_telemetry_table(summary)


def _build_rd_telemetry_git(state: RDDashboardState):
    """Builds the combined telemetry + git-checkpoints renderable (neon)."""
    from rich.console import Group

    tele = build_telemetry_table(state.telemetry_summary)

    git_grid = Table.grid(expand=True, padding=(0, 1))
    git_grid.add_column(ratio=1)
    git_grid.add_row(Text("⑂ GIT CHECKPOINTS", style=f"bold {T.GREEN}"))
    if state.git_checkpoints:
        for cp in state.git_checkpoints[-4:]:
            git_grid.add_row(Text.assemble(("● ", f"bold {T.GREEN}"), (cp, T.GHOST)))
    else:
        git_grid.add_row(Text("○ none yet", style=T.DIM))
    git_panel = Panel(git_grid, box=T.BOX, border_style=T.GREEN, padding=(0, 1))

    return Group(tele, git_panel)


def _build_rd_worker_panel(state: RDDashboardState) -> Panel:
    """Builds the worker reasoning stream panel (neon)."""
    content = Table.grid(expand=True, padding=(0, 1))
    content.add_column(ratio=1)

    header = Table.grid(expand=True)
    header.add_column(justify="left", ratio=1)
    header.add_column(justify="right", ratio=1)
    active_task = state.active_subtask_id or "(idle)"
    task_info = Text.assemble(
        ("◈ node ", f"bold {T.CYAN}"),
        (active_task, f"bold {T.MAGENTA}" if state.active_subtask_id else T.DIM),
    )
    status_info = Text.assemble(("", ""), T.neon_badge(state.worker_status))
    header.add_row(task_info, status_info)
    content.add_row(header)
    content.add_row(T.neon_rule("─", T.DIM, 40))

    if state.worker_thought:
        lines = state.worker_thought.splitlines()
        truncated = "\n".join(lines[-12:]) if len(lines) > 12 else state.worker_thought
        content.add_row(Text(truncated, style=f"italic {T.BLUE}"))
    else:
        content.add_row(Text("▮ worker awaiting instructions…", style=f"italic {T.DIM}"))

    return Panel(
        content,
        title=f"[bold {T.BLUE}]◈ WORKER NEURAL STREAM[/]",
        box=T.BOX,
        border_style=T.BLUE,
        padding=(0, 1),
    )


def _build_rd_events_panel(state: RDDashboardState) -> Panel:
    """Builds the tool-call timeline + verification detail panel (neon)."""
    content = Table.grid(expand=True, padding=(0, 1))
    content.add_column(ratio=1)

    content.add_row(Text("⌁ TOOL TIMELINE", style=f"bold {T.MAGENTA}"))
    if state.tool_log:
        for ev in state.tool_log[-6:]:
            mark = T.ok_fail(bool(ev.get("success", True)))
            name = str(ev.get("tool", "?"))
            arg = str(ev.get("arg", ""))
            if len(arg) > 34:
                arg = arg[:31] + "…"
            row = Text.assemble(
                (f"{name} ", f"bold {T.MAGENTA}"),
                (arg + " ", T.GHOST),
            )
            row.append_text(mark)
            content.add_row(row)
    else:
        content.add_row(Text("○ no tool calls yet", style=T.DIM))

    # Last verification detail
    content.add_row(T.neon_rule("─", T.DIM, 40))
    lv = state.last_verification
    if lv:
        ok = bool(lv.get("success"))
        content.add_row(Text.assemble(
            ("⟳ verify ", f"bold {T.AMBER}"),
            (str(lv.get("command", ""))[:38], T.GHOST),
        ))
        exit_style = T.GREEN if ok else T.PINK
        content.add_row(Text.assemble(
            ("  exit ", T.DIM),
            (str(lv.get("exit_code", "?")), f"bold {exit_style}"),
            ("  ", ""),
            T.ok_fail(ok),
        ))
        snip = str(lv.get("snippet", "")).strip().replace("\n", " ")
        if snip:
            content.add_row(Text("  " + snip[:60], style=T.DIM))
    else:
        content.add_row(Text("⟳ verify: (pending)", style=T.DIM))

    return Panel(
        content,
        title=f"[bold {T.MAGENTA}]⌁ TOOL & VERIFY STREAM[/]",
        box=T.BOX,
        border_style=T.MAGENTA,
        padding=(0, 1),
    )


def _build_rd_footer_panel(state: RDDashboardState, compact: bool = False) -> Panel:
    """Builds the footer: zero-knowledge privacy badge + invariants (neon)."""
    if state.privacy_verified:
        priv_badge = Text.assemble(
            ("▣ ZERO-KNOWLEDGE ", f"bold {T.GREEN}"),
            ("◉ 0 BYTES LEAKED TO CLOUD", T.GREEN),
        )
        border = T.GREEN
    else:
        priv_badge = Text.assemble(
            ("▣ ZERO-KNOWLEDGE ", f"bold {T.PINK}"),
            ("◉ LEAK DETECTED", f"bold {T.PINK}"),
        )
        border = T.PINK

    invariants_text = Text.assemble(
        ("⛨ ", T.CYAN),
        ("[ cloud-blind ]", T.DIM),
        (" [ sandboxed ]", T.DIM),
        (" [ deterministic-verify ]", T.DIM),
        (" [ git-rollback ]", T.DIM),
    )
    grid = Table.grid(expand=True)
    if compact:
        grid.add_column(justify="left")
        grid.add_row(priv_badge)
        grid.add_row(invariants_text)
    else:
        grid.add_column(justify="left", ratio=3)
        grid.add_column(justify="right", ratio=4)
        grid.add_row(priv_badge, invariants_text)
    return Panel(grid, box=T.BOX, border_style=border, padding=(0, 1))


# ==============================================================================
# RDDASHBOARD CONTROLLER
# ==============================================================================


class RDDashboard:
    """Interactive HUD and logger for autonomous R&D engine execution."""

    def __init__(
        self,
        console: Optional[Console] = None,
        no_hud: bool = False,
        refresh_per_second: float = 8.0,
        mode: str = "mock",
        goal: str = "",
        workspace: str = "",
    ) -> None:
        self.console = console or Console()
        self.no_hud = no_hud or (not sys.stdout.isatty())
        self.refresh_per_second = refresh_per_second
        self.state = RDDashboardState(
            mode=mode,
            goal=goal,
            workspace_root=workspace,
        )

        term_width = self.console.size.width if self.console else 120
        self.layout = build_rd_layout(term_width)

        self._live: Optional[Live] = None
        self._is_running = False
        self._start_time: float = 0.0
        self._last_logged_worker_status: Optional[str] = None

    def start(self) -> None:
        """Starts live display or headless streaming mode."""
        self._is_running = True
        self._start_time = time.time()
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
                self.no_hud = True
                self._live = None
                self.console.print(
                    f"[dim yellow]Warning: Live HUD disabled ({e}). Running in headless mode.[/]"
                )
        else:
            self.console.print(
                "[bold cyan]Antigravity R&D Dashboard started in headless mode (HUD disabled)[/]"
            )

    def stop(self) -> None:
        """Cleanly stops live display, restoring console."""
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None
        self._is_running = False

    def refresh(self) -> None:
        """Refreshes layout panels and triggers live render."""
        if not self._is_running:
            return
        if self._start_time > 0:
            self.state.elapsed_s = time.time() - self._start_time
        self._render_into_layout()
        if self._live is not None:
            self._live.refresh()

    def _render_into_layout(self) -> None:
        """Populates layout panels with current state components."""
        try:
            compact = self.console.size.width < 100
            self.layout["header"].update(_build_rd_header(self.state, compact=compact))
            self.layout["subtasks"].update(_build_rd_subtasks_panel(self.state))
            self.layout["telemetry"].update(
                _build_rd_telemetry_git(self.state)
            )
            self.layout["worker"].update(_build_rd_worker_panel(self.state))
            self.layout["events"].update(_build_rd_events_panel(self.state))
            self.layout["footer"].update(_build_rd_footer_panel(self.state, compact=compact))
        except Exception:
            pass

    def set_goal(self, goal: str, workspace: str = "") -> None:
        """Sets the active R&D engine goal and optional workspace path."""
        self.state.goal = goal
        if workspace:
            self.state.workspace_root = workspace
        self.refresh()
        if self.no_hud:
            ws_info = f" (Workspace: {workspace})" if workspace else ""
            self.console.print(f"[bold cyan][R&D Engine][/] Goal: {goal}{ws_info}")

    def set_pipeline_status(self, status: str) -> None:
        """Updates pipeline execution status."""
        self.state.pipeline_status = status
        self.refresh()
        if self.no_hud:
            self.console.print(f"[bold blue][Pipeline][/] Status changed -> {status}")

    def init_dag(self, dag: Union[ExecutionPlanDAG, Any]) -> None:
        """Initializes the subtask list from an ExecutionPlanDAG."""
        self.state.subtasks.clear()
        if hasattr(dag, "tasks") and isinstance(dag.tasks, dict):
            task_order: List[str]
            if hasattr(dag, "topological_sort"):
                try:
                    task_order = dag.topological_sort()
                except Exception:
                    task_order = list(dag.tasks.keys())
            else:
                task_order = list(dag.tasks.keys())

            for tid in task_order:
                task = dag.tasks.get(tid)
                if task is not None:
                    self.state.subtasks.append({
                        "id": task.id,
                        "title": getattr(task, "title", ""),
                        "description": getattr(task, "description", ""),
                        "status": getattr(task, "status", "pending"),
                        "verification_command": getattr(task, "verification_command", ""),
                        "attempts": getattr(task, "retry_count", 0),
                        "dependencies": list(getattr(task, "dependencies", [])),
                    })
        elif isinstance(dag, (list, tuple)):
            for item in dag:
                if isinstance(item, dict):
                    self.state.subtasks.append(dict(item))
                elif hasattr(item, "id"):
                    self.state.subtasks.append({
                        "id": item.id,
                        "title": getattr(item, "title", ""),
                        "description": getattr(item, "description", ""),
                        "status": getattr(item, "status", "pending"),
                        "verification_command": getattr(item, "verification_command", ""),
                        "attempts": getattr(item, "retry_count", 0),
                        "dependencies": list(getattr(item, "dependencies", [])),
                    })
        self.refresh()
        if self.no_hud:
            self.console.print(
                f"[bold cyan][DAG Initialized][/] Loaded {len(self.state.subtasks)} subtasks"
            )

    def set_active_subtask(
        self,
        subtask_id: str,
        description: str = "",
        verification_cmd: str = "",
    ) -> None:
        """Designates the currently running subtask."""
        self.state.active_subtask_id = subtask_id
        self.state.active_subtask_desc = description
        self.state.active_subtask_cmd = verification_cmd

        for sub in self.state.subtasks:
            if sub.get("id") == subtask_id:
                if description:
                    sub["description"] = description
                if verification_cmd:
                    sub["verification_command"] = verification_cmd
                break

        self.state.recent_events.append(f"Subtask active: {subtask_id}")
        self.state.recent_events = self.state.recent_events[-20:]
        self.refresh()
        if self.no_hud:
            desc_info = f" - {description}" if description else ""
            self.console.print(f"[bold yellow][Active Subtask][/] {subtask_id}{desc_info}")

    def update_subtask_status(
        self,
        subtask_id: str,
        status: Union[SubtaskExecutionState, str],
        attempts: int = 1,
    ) -> None:
        """Updates the status and attempt count of a subtask."""
        status_val = status.value if hasattr(status, "value") else str(status)
        found = False
        for sub in self.state.subtasks:
            if sub.get("id") == subtask_id:
                sub["status"] = status_val
                sub["attempts"] = attempts
                found = True
                break
        if not found:
            self.state.subtasks.append({
                "id": subtask_id,
                "status": status_val,
                "attempts": attempts,
                "description": "",
                "verification_command": "",
            })

        self.state.recent_events.append(f"Subtask {subtask_id} -> {status_val} (att {attempts})")
        self.state.recent_events = self.state.recent_events[-20:]
        self.refresh()
        if self.no_hud:
            self.console.print(
                f"[bold cyan][Subtask {subtask_id}][/] status: {status_val} (attempt {attempts})"
            )

    def update_worker_stream(
        self,
        thought_chunk: str,
        accumulated: str = "",
        status: str = "STREAMING",
    ) -> None:
        """Updates the streaming thought and reasoning buffer of the worker."""
        self.state.worker_status = status
        if accumulated:
            self.state.worker_thought = accumulated
        else:
            self.state.worker_thought += thought_chunk
        self.refresh()
        if self.no_hud:
            if (
                status.upper() not in ("STREAMING", "GENERATING")
                or status != self._last_logged_worker_status
            ):
                self._last_logged_worker_status = status
                snippet = self.state.worker_thought[-120:].strip() if self.state.worker_thought else ""
                if snippet:
                    self.console.print(f"[dim yellow][Worker {status}][/] {snippet}")

    def on_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        output_snippet: str = "",
    ) -> None:
        """Records an active tool invocation and output snippet."""
        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items()) if args else ""
        self.state.active_tool_call = f"{tool_name}({args_str})"
        self.state.active_tool_output = output_snippet

        # Compact timeline entry (arg = most salient argument).
        arg = ""
        if args:
            arg = str(args.get("path") or args.get("command") or next(iter(args.values()), ""))
        self.state.tool_log.append({"tool": tool_name, "arg": arg, "success": True})
        self.state.tool_log = self.state.tool_log[-40:]

        ev_text = f"Tool: {tool_name}" + (f" -> {output_snippet[:40]}" if output_snippet else "")
        self.state.recent_events.append(ev_text)
        self.state.recent_events = self.state.recent_events[-20:]
        self.refresh()
        if self.no_hud:
            out_info = f" -> {output_snippet[:60]}" if output_snippet else ""
            self.console.print(f"[bold magenta][Tool Call][/] {tool_name}({args_str}){out_info}")

    def on_verification_result(
        self,
        subtask_id: str,
        passed: bool,
        exit_code: int,
        summary: str = "",
    ) -> None:
        """Records a subtask verification outcome."""
        outcome = f"exit {exit_code} ({'PASS' if passed else 'FAIL'})"
        ev_text = f"Verification {subtask_id}: {outcome}"
        self.state.recent_events.append(ev_text)
        self.state.recent_events = self.state.recent_events[-20:]
        self.state.last_verification = {
            "command": self.state.active_subtask_cmd,
            "exit_code": exit_code,
            "success": passed,
            "snippet": summary,
        }

        for sub in self.state.subtasks:
            if sub.get("id") == subtask_id:
                sub["status"] = "completed" if passed else "failed"
                break

        self.refresh()
        if self.no_hud:
            style = "bold green" if passed else "bold red"
            sum_info = f" - {summary}" if summary else ""
            self.console.print(f"[{style}][Verification {subtask_id}][/] {outcome}{sum_info}")

    def update_telemetry(self, summary: TelemetrySummary) -> None:
        """Updates real-time token and economic cost telemetry."""
        self.state.telemetry_summary = summary
        self.refresh()

    def set_privacy_verified(self, verified: bool) -> None:
        """Updates the zero-knowledge privacy verification status."""
        self.state.privacy_verified = verified
        self.refresh()
        if self.no_hud and not verified:
            self.console.print("[bold red][Privacy Violation][/] Zero-Knowledge Invariant Violated!")

    def handle_event(
        self,
        event_type: str,
        subtask_id: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Event dispatcher callback for SelfCorrectionLoop integration."""
        payload = payload or {}
        if event_type == "subtask_start":
            sub_dict = payload.get("subtask", {})
            self.set_active_subtask(
                subtask_id,
                description=sub_dict.get("description", ""),
                verification_cmd=sub_dict.get("verification_command", ""),
            )
            attempt = payload.get("attempt", 0) + 1
            self.update_subtask_status(subtask_id, SubtaskExecutionState.RUNNING, attempts=attempt)
        elif event_type == "repairing":
            attempt = payload.get("attempt", 1) + 1
            self.update_subtask_status(subtask_id, SubtaskExecutionState.REPAIRING, attempts=attempt)
        elif event_type == "verifying":
            self.update_subtask_status(subtask_id, SubtaskExecutionState.VERIFYING)
        elif event_type == "subtask_complete":
            self.update_subtask_status(subtask_id, SubtaskExecutionState.COMPLETED)
            # The loop commits a git checkpoint per completed subtask.
            self.state.git_checkpoints.append(f"{subtask_id} ✓ committed")
            self.state.git_checkpoints = self.state.git_checkpoints[-20:]
        elif event_type == "subtask_failed":
            self.update_subtask_status(subtask_id, SubtaskExecutionState.FAILED)
        elif event_type == "git_checkpoint":
            msg = str(payload.get("message", subtask_id))
            self.state.git_checkpoints.append(msg)
            self.state.git_checkpoints = self.state.git_checkpoints[-20:]
        elif event_type == "tool_call":
            self.on_tool_call(
                tool_name=str(payload.get("tool_name") or payload.get("tool") or ""),
                args=payload.get("args", {}) or {},
                output_snippet=str(payload.get("output", "")),
            )
        elif event_type == "tool_result":
            # Update the most recent matching timeline entry with success/snippet.
            name = str(payload.get("tool_name") or payload.get("tool") or "")
            success = bool(payload.get("success", True))
            for entry in reversed(self.state.tool_log):
                if entry.get("tool") == name:
                    entry["success"] = success
                    break
        elif event_type == "thought":
            self.update_worker_stream("", str(payload.get("thought", "")), status="THINKING")
        elif event_type in ("verifying",):
            self.state.active_subtask_cmd = str(payload.get("command", self.state.active_subtask_cmd))
        elif event_type == "verification_failed":
            self.on_verification_result(
                subtask_id=subtask_id,
                passed=False,
                exit_code=int(payload.get("exit_code", 1)),
                summary=str(payload.get("failure_reason", "")),
            )
        elif event_type == "verification_result":
            self.on_verification_result(
                subtask_id=subtask_id,
                passed=bool(payload.get("passed", False)),
                exit_code=int(payload.get("exit_code", 1)),
                summary=str(payload.get("summary", "")),
            )

    def as_event_listener(self) -> Callable[[str, str, Dict[str, Any]], None]:
        """Returns a callback suitable for SelfCorrectionLoop event_listener."""
        return self.handle_event

    def print_final_summary(
        self,
        dag_result: Any,
        dag: Any,
        summary: Optional[TelemetrySummary] = None,
        privacy_verified: Optional[bool] = None,
        duration_s: float = 0.0,
    ) -> None:
        """Convenience method to render post-run R&D engine summary report."""
        from antigravity_hybrid_orchestrator.hud.summary import print_rd_summary_table

        p_verif = self.state.privacy_verified if privacy_verified is None else privacy_verified
        t_summary = summary if summary is not None else self.state.telemetry_summary
        d_sec = duration_s if duration_s > 0 else self.state.elapsed_s
        print_rd_summary_table(
            console=self.console,
            dag_result=dag_result,
            dag=dag,
            summary=t_summary,
            privacy_verified=p_verif,
            duration_s=d_sec,
        )
