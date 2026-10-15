"""Atomic owner-only persistence for resumable R&D runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from antigravity_hybrid_orchestrator.planner.dag import ExecutionPlanDAG


class RunStateStore:
    """Save and load a JSON snapshot next to a run's JSONL event log."""

    @staticmethod
    def sidecar_for(run_log: Path) -> Path:
        return Path(run_log).with_suffix(".state.json")

    @staticmethod
    def save(
        path: Path,
        *,
        goal: str,
        workspace: str,
        dag: ExecutionPlanDAG,
        last_event: Optional[Dict[str, Any]] = None,
        remote_tokens: int = 0,
        elapsed_s: float = 0.0,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        payload = {
            "version": 1,
            "goal": goal,
            "workspace": str(Path(workspace).resolve()),
            "dag": dag.to_dict(),
            "last_event": last_event or {},
            "remote_tokens": max(0, int(remote_tokens)),
            "elapsed_s": max(0.0, float(elapsed_s)),
        }
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path)
            os.chmod(path, 0o600)
        finally:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def load(path: Path) -> Dict[str, Any]:
        path = Path(path)
        with path.open("r", encoding="utf-8") as stream:
            state = json.load(stream)
        if not isinstance(state, dict) or state.get("version") != 1:
            raise ValueError(f"Unsupported or invalid run state: {path}")
        dag = ExecutionPlanDAG.from_dict(state.get("dag", {}))
        dag.validate()
        state["dag"] = dag.to_dict()
        return state
