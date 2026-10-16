"""Command-line interface for Antigravity Hybrid Orchestrator."""

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

from rich.console import Console

from antigravity_hybrid_orchestrator.coordinator import HybridCoordinator
from antigravity_hybrid_orchestrator.engine.control import ExecutionBudget
from antigravity_hybrid_orchestrator.hud.replay import replay_run


def build_parser() -> argparse.ArgumentParser:
    """Builds and returns the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="antigravity-hybrid-orchestrator",
        description="Antigravity Hybrid Orchestrator: Cloud Architect + On-Device Workforce",
    )
    parser.add_argument(
        "--mode",
        choices=["live", "mock"],
        default="live",
        help="Execution mode: 'live' (Codex CLI planner + local Ollama workforce) or 'mock' (deterministic fixtures)",
    )
    parser.add_argument(
        "--target",
        choices=["all", "auth", "billing", "database"],
        default="all",
        help="Target security gauntlet module to audit and patch (default: all)",
    )
    parser.add_argument(
        "--no-hud",
        action="store_true",
        help="Disable interactive Rich live terminal HUD (runs in plain logging mode)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset target files to vulnerable baseline before execution",
    )
    parser.add_argument(
        "--refresh-rate",
        type=float,
        default=8.0,
        help="Refresh rate per second for Rich Live HUD (default: 8.0)",
    )
    # Milestone 1: Dynamic Goal & Workspace Ingestion flags
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Dynamic R&D goal or task description for autonomous execution",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Target workspace or project directory path (default: current directory if --goal specified)",
    )
    parser.add_argument(
        "--retry-budget",
        type=int,
        default=5,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--architect",
        choices=["luna", "haiku", "agy"],
        default="haiku",
        help=(
            "Cloud Architect backend for live mode: haiku (Claude Haiku 5.5 via the claude CLI; default), "
            "luna (codex, gpt-6-luna), or agy (Antigravity CLI)"
        ),
    )
    parser.add_argument(
        "--architect-model",
        type=str,
        default=None,
        help="Cloud Architect model override (e.g. gemini-3.8-flash-high for agy)",
    )
    parser.add_argument(
        "--worker-model",
        type=str,
        default=None,
        help="Local Ollama workforce model override (default: OLLAMA_MODEL env var or gemma4:12b)",
    )
    parser.add_argument("--max-runtime", type=float, default=None,
                        help="Optional run-wide time budget in seconds (default: unlimited)")
    parser.add_argument("--max-remote-tokens", type=int, default=None,
                        help="Optional cloud token budget for planning (default: unlimited)")
    parser.add_argument("--model-idle-timeout", type=float, default=900,
                        help="Stop after this many seconds without model progress; 0 disables")
    parser.add_argument("--tool-idle-timeout", type=float, default=300,
                        help="Stop after this many seconds without tool progress; 0 disables")
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Disable JSONL run recording in R&D mode",
    )
    parser.add_argument(
        "--replay",
        metavar="RUN_FILE",
        help="Replay a recorded R&D run JSONL file in the HUD",
    )
    parser.add_argument(
        "--resume",
        metavar="RUN_FILE",
        help="Resume an interrupted R&D run using its adjacent .state.json snapshot",
    )
    parser.add_argument(
        "--replay-speed",
        type=float,
        default=8.0,
        help="Replay events per second (0 = as fast as possible; default: 8)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive R&D session that accepts multiple tasks",
    )
    return parser


def run_interactive(
    init_kwargs: Dict[str, object],
    input_fn: Optional[Callable[[str], str]] = None,
    console: Optional[Console] = None,
) -> int:
    """Run one R&D pipeline per task until the user exits the prompt loop.

    Session options such as mode, workspace, and execution budgets are reused for
    every task. Each task gets its own coordinator and dashboard lifecycle.
    """
    console = console or Console()
    read_input = input_fn or console.input
    console.print("[bold cyan]ANTIGRAVITY · INTERACTIVE R&D SESSION[/]")
    console.print("Enter a task to start a run. Use [bold magenta]/help[/] or [bold magenta]/quit[/].")
    last_status = 0

    while True:
        try:
            task = read_input("[bold cyan]TASK[/] › ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("[dim]Interactive session closed.[/]")
            break

        if not task:
            continue
        command = task.lower()
        if command in {"/quit", "/exit"}:
            console.print("[dim]Interactive session closed.[/]")
            break
        if command == "/help":
            console.print("[bold cyan]Commands[/]")
            console.print("  [bold magenta]/help[/]  Show this help")
            console.print("  [bold magenta]/quit[/]  End the session")
            console.print("Enter any other text as a new R&D task.")
            continue
        if task.startswith("/"):
            console.print(f"[yellow]Unknown command: {task}. Type /help for commands.[/]")
            continue

        console.print(f"[bold cyan]Starting task:[/] {task}")
        run_kwargs = dict(init_kwargs)
        run_kwargs["goal"] = task
        try:
            last_status = int(HybridCoordinator(**run_kwargs).run())
            if last_status == 130:
                console.print("[yellow]Task interrupted; ending interactive session.[/]")
                return 130
        except KeyboardInterrupt:
            console.print("[yellow]Task interrupted; ending interactive session.[/]")
            return 130
        except Exception as exc:
            last_status = 1
            console.print(f"[bold red]Task failed:[/] {exc}")
            console.print("[dim]You can enter another task or use /quit.[/]")

    return last_status


def main(argv: Optional[List[str]] = None) -> int:
    """Entrypoint function for CLI execution."""
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    try:
        ExecutionBudget(
            max_runtime_s=args.max_runtime,
            max_remote_tokens=args.max_remote_tokens,
            model_idle_timeout_s=args.model_idle_timeout,
            tool_idle_timeout_s=args.tool_idle_timeout,
        )
    except ValueError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    if args.retry_budget < 0:
        sys.stderr.write("error: --retry-budget must be non-negative\n")
        return 2
    if args.replay_speed < 0:
        sys.stderr.write("error: --replay-speed must be non-negative\n")
        return 2
    if args.interactive and (args.goal or args.resume):
        sys.stderr.write("error: --interactive cannot be combined with --goal or --resume\n")
        return 2
    if args.interactive and args.replay:
        sys.stderr.write("error: --interactive cannot be combined with --replay\n")
        return 2
    if args.resume and args.replay:
        sys.stderr.write("error: --resume cannot be combined with --replay\n")
        return 2

    if args.replay:
        replay_path = Path(args.replay)
        if not replay_path.is_file():
            sys.stderr.write(f"error: replay file not found: {replay_path}\n")
            return 2
        replay_run(
            str(replay_path),
            speed=args.replay_speed,
            no_hud=args.no_hud,
        )
        return 0

    if args.resume and not Path(args.resume).is_file():
        sys.stderr.write(f"error: run file not found: {args.resume}\n")
        return 2

    init_kwargs = {
        "mode": args.mode,
        "target": args.target,
        "reset": args.reset,
        "no_hud": args.no_hud,
        "refresh_rate": args.refresh_rate,
        "goal": args.goal,
        "workspace": args.workspace,
        "retry_budget": args.retry_budget,
        "max_runtime_s": args.max_runtime,
        "max_remote_tokens": args.max_remote_tokens,
        "model_idle_timeout_s": args.model_idle_timeout,
        "architect_backend": args.architect,
        "architect_model": args.architect_model,
        "worker_model": args.worker_model,
        "tool_idle_timeout_s": args.tool_idle_timeout,
        "resume_run": args.resume,
        "record": not args.no_record,
    }


    if args.interactive:
        return run_interactive(init_kwargs)

    coordinator = HybridCoordinator(**init_kwargs)
    return coordinator.run()


if __name__ == "__main__":
    sys.exit(main())
