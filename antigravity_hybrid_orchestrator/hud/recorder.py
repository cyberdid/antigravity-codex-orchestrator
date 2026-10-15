"""JSONL event recorder for R&D engine runs.

Persists every orchestration event to a newline-delimited JSON file so any run
can be replayed into the terminal HUD later (see hud/replay.py). Writing never
raises into the pipeline — recording failures are swallowed by design.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4


class EventRecorder:
    """Appends orchestration events to a per-run JSONL file.

    Each line is a JSON object: {"t": <elapsed_s>, "kind": <str>, ...}. Kinds:
    - "header": run metadata (goal, workspace, mode, started_at)
    - "event":  an orchestration event (event_type, subtask, payload)
    - "telemetry": a telemetry snapshot
    - "footer": final verdict + summary
    """

    def __init__(
        self,
        run_dir: Path,
        goal: str = "",
        workspace: str = "",
        mode: str = "mock",
    ) -> None:
        self.run_dir = Path(run_dir)
        self.goal = goal
        self.workspace = workspace
        self.mode = mode
        self._start = time.time()
        self.path: Optional[Path] = None
        self._fh = None
        self._open()
        self.write_header()

    def _open(self) -> None:
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self.path = self.run_dir / f"run-{stamp}-{uuid4().hex[:8]}.jsonl"
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            self._fh = os.fdopen(fd, "w", encoding="utf-8")
        except Exception:
            self._fh = None
            self.path = None

    def _write(self, obj: Dict[str, Any]) -> None:
        if self._fh is None:
            return
        try:
            obj.setdefault("t", round(time.time() - self._start, 3))
            self._fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()
        except Exception:
            pass

    def write_header(self) -> None:
        self._write({
            "kind": "header",
            "goal": self.goal,
            "workspace": self.workspace,
            "mode": self.mode,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

    def emit(self, event_type: str, subtask_id: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Records one orchestration event (safe as a SelfCorrectionLoop listener)."""
        self._write({
            "kind": "event",
            "event_type": event_type,
            "subtask": subtask_id,
            "payload": payload or {},
        })

    # Callable form so the recorder can be used directly as an event listener.
    __call__ = emit

    def record_telemetry(self, summary: Any) -> None:
        """Records a telemetry snapshot from a TelemetrySummary-like object."""
        fields = (
            "cloud_tokens", "cloud_cost_usd", "local_tokens", "local_cost_usd",
            "total_tokens", "cost_savings_pct", "tokens_per_second",
        )
        snap = {f: getattr(summary, f, None) for f in fields} if summary is not None else {}
        self._write({"kind": "telemetry", "summary": snap})

    def write_footer(
        self,
        success: bool,
        privacy_verified: bool,
        completed: Optional[list] = None,
        failed: Optional[list] = None,
        duration_s: float = 0.0,
        status: Optional[str] = None,
    ) -> None:
        self._write({
            "kind": "footer",
            "success": success,
            "privacy_verified": privacy_verified,
            "completed": list(completed or []),
            "failed": list(failed or []),
            "duration_s": round(duration_s, 3),
            "status": status or ("completed" if success else "failed"),
        })

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None
