"""Replay a recorded R&D run (JSONL) back into the cyberpunk terminal HUD.

Usage:
    python -m antigravity_hybrid_orchestrator.hud.replay <run.jsonl> [--speed N] [--no-hud]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from antigravity_hybrid_orchestrator.hud.rd_dashboard import RDDashboard
from antigravity_hybrid_orchestrator.telemetry import TelemetrySummary


def load_run(path: Path) -> List[Dict[str, Any]]:
    """Loads a recorded run JSONL into a list of records."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _summary_from_snap(snap: Dict[str, Any]) -> TelemetrySummary:
    return TelemetrySummary(
        cloud_tokens=int(snap.get("cloud_tokens") or 0),
        cloud_cost_usd=float(snap.get("cloud_cost_usd") or 0.0),
        local_tokens=int(snap.get("local_tokens") or 0),
        local_cost_usd=float(snap.get("local_cost_usd") or 0.0),
        total_tokens=int(snap.get("total_tokens") or 0),
        cost_savings_pct=float(snap.get("cost_savings_pct") or 0.0),
        tokens_per_second=float(snap.get("tokens_per_second") or 0.0),
    )


def replay_run(path: str, speed: float = 8.0, no_hud: bool = False) -> RDDashboard:
    """Replays a recorded run into an RDDashboard.

    Args:
        path: Path to the run JSONL file.
        speed: Events per second (0 = as fast as possible).
        no_hud: If True, headless streaming (also used in tests).

    Returns:
        The RDDashboard after all records are applied.
    """
    records = load_run(Path(path))

    header = next((r for r in records if r.get("kind") == "header"), {})
    hud = RDDashboard(
        no_hud=no_hud,
        mode=str(header.get("mode", "replay")),
        goal=str(header.get("goal", "")),
        workspace=str(header.get("workspace", "")),
    )
    hud.set_pipeline_status("REPLAY")
    hud.start()

    delay = (1.0 / speed) if (speed and speed > 0 and not no_hud) else 0.0

    try:
        for rec in records:
            kind = rec.get("kind")
            if kind == "event":
                hud.handle_event(
                    rec.get("event_type", ""),
                    rec.get("subtask", ""),
                    rec.get("payload", {}) or {},
                )
            elif kind == "telemetry":
                hud.update_telemetry(_summary_from_snap(rec.get("summary", {}) or {}))
            elif kind == "footer":
                hud.set_privacy_verified(bool(rec.get("privacy_verified", True)))
                hud.set_pipeline_status("COMPLETE")
            if delay:
                time.sleep(delay)
    finally:
        hud.stop()

    return hud


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="rd-replay", description="Replay a recorded R&D run into the HUD")
    parser.add_argument("run_file", help="Path to a recorded run JSONL file")
    parser.add_argument("--speed", type=float, default=8.0, help="Events per second (0 = instant)")
    parser.add_argument("--no-hud", action="store_true", help="Headless mode (no live display)")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if not Path(args.run_file).exists():
        sys.stderr.write(f"error: run file not found: {args.run_file}\n")
        return 2

    replay_run(args.run_file, speed=args.speed, no_hud=args.no_hud)
    return 0


if __name__ == "__main__":
    sys.exit(main())
