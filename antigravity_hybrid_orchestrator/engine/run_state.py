"""Atomic owner-only persistence for resumable R&D runs."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from antigravity_hybrid_orchestrator.planner.dag import ExecutionPlanDAG


class RunStateStore:
    """Save and load a JSON snapshot next to a run's JSONL event log."""

    @staticmethod
    def _file_hash(root: Path, relative: str) -> str:
        """Open each component without following symlinks, including during races."""
        candidate = Path(relative)
        if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
            raise ValueError("Fingerprint paths must be contained relative files")
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in candidate.parts[:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                os.close(directory)
                directory = child
            fd = os.open(candidate.parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise ValueError("Fingerprint target must be a regular file")
                with os.fdopen(fd, "rb", closefd=False) as stream:
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                    return digest.hexdigest()
            finally:
                os.close(fd)
        finally:
            os.close(directory)

    @staticmethod
    def bind_task(workspace: str, task: Any) -> Dict[str, Any]:
        root = Path(workspace).resolve()
        files = sorted(set(task.outputs + task.input_files + task.affected_files))
        definition = task.to_dict()
        definition.pop("status", None)
        definition.pop("retry_count", None)
        binding: Dict[str, Any] = {"definition": definition, "files": {}, "valid": True}
        for relative in files:
            try:
                binding["files"][relative] = RunStateStore._file_hash(root, relative)
            except (OSError, ValueError):
                binding["valid"] = False
                binding["files"][relative] = None
        return binding

    @staticmethod
    def record_completed(workspace: str, dag: ExecutionPlanDAG, task_id: str) -> None:
        """Capture verified completion once; call again only after fresh verification."""
        dag.metadata.setdefault("task_fingerprints", {})[task_id] = RunStateStore.bind_task(
            workspace, dag.get_task(task_id)
        )

    @staticmethod
    def _git_head(workspace: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "-C", workspace, "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=1, check=False,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

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
        bindings = dag.metadata.setdefault("task_fingerprints", {})
        for task in dag.tasks.values():
            if task.status == "completed" and task.id not in bindings:
                RunStateStore.record_completed(workspace, dag, task.id)
        payload = {
            "version": 2,
            "git_head": RunStateStore._git_head(workspace),
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
        if not isinstance(state, dict) or state.get("version") not in (1, 2):
            raise ValueError(f"Unsupported or invalid run state: {path}")
        dag = ExecutionPlanDAG.from_dict(state.get("dag", {}))
        dag.validate()
        bindings = dag.metadata.get("task_fingerprints", {})
        invalid = set()
        for task in dag.get_execution_order():
            bound = bindings.get(task.id)
            stale = task.status == "completed" and (
                state.get("version") == 1 or not isinstance(bound, dict)
                or not bound.get("valid")
                or bound != RunStateStore.bind_task(state["workspace"], task)
            )
            if stale or any(dep in invalid for dep in task.dependencies):
                invalid.add(task.id)
                task.status = "pending"
                task.retry_count = 0
                bindings.pop(task.id, None)
                integrity = dag.metadata.get("integrity_by_task", {})
                if isinstance(integrity, dict):
                    integrity.pop(task.id, None)
        state["invalidated_tasks"] = sorted(invalid)
        state["dag"] = dag.to_dict()
        return state
