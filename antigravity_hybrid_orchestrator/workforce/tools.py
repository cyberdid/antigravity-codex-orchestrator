"""Universal Tool Suite for sandboxed workspace execution (Milestone 3, Feature F8).

Provides:
- ToolResult: Standardized dataclass for tool outputs and metadata.
- WorkspaceToolSuite: Sandboxed implementation of read_file, write_file,
  run_command, git_status, git_diff, dynamic execution, and JSON schemas.
"""

from dataclasses import dataclass, field
import os
import selectors
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Union

from antigravity_hybrid_orchestrator.engine.control import ExecutionControl


@dataclass
class ToolResult:
    """Standardized result envelope returned by all workspace execution tools.

    Attributes:
        success: True if the tool operation succeeded without error.
        output: Primary output payload (file contents, stdout, status text).
        error: Detailed error description if failed, otherwise None.
        metadata: Execution diagnostics (exit codes, line counts, bytes, durations, etc.).
    """

    success: bool
    output: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the result to a JSON-compatible dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolResult":
        """Reconstructs a ToolResult from a dictionary representation."""
        return cls(
            success=bool(data.get("success", False)),
            output=str(data.get("output", "")),
            error=data.get("error"),
            metadata=dict(data.get("metadata", {})),
        )

    def __bool__(self) -> bool:
        """Allows direct truthiness evaluation based on success."""
        return self.success


READ_DEFAULT_LINE_LIMIT = 400


class WorkspaceToolSuite:
    """Universal sandboxed execution tools for local on-device workforce.

    Enforces:
    - Path traversal sandboxing: blocks escaping workspace_root via '..' or symlinks.
    - Safe file writes: atomic replace with automatic parent directory creation.
    - .git directory modification protection.
    - Process tree isolation: new session (os.setsid) and group kill (os.killpg) on timeout.
    - Graceful git handling: safe status/diff in non-git repositories.
    """

    _PACKAGE_INSTALL_COMMAND = re.compile(
        r"(?i)\b(?:sudo\s+)?(?:"
        r"python(?:\d+(?:\.\d+)?)?\s+-m\s+pip|pip(?:3(?:\.\d+)?)?|"
        r"uv(?:\s+pip)?|apt(?:-get)?|aptitude|dnf|yum|conda|mamba|poetry|pdm"
        r")\s+(?:install|add)\b"
    )

    def __init__(
        self,
        workspace_root: Union[str, Path],
        default_command_timeout: Optional[float] = None,
        default_idle_timeout: Optional[float] = 300,
        execution_control: Optional[ExecutionControl] = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.default_command_timeout = default_command_timeout
        self.default_idle_timeout = default_idle_timeout
        self.execution_control = execution_control or ExecutionControl()
        if not self.workspace_root.exists():
            raise FileNotFoundError(
                f"Workspace directory does not exist: {self.workspace_root}"
            )
        if not self.workspace_root.is_dir():
            raise NotADirectoryError(
                f"Workspace path is not a directory: {self.workspace_root}"
            )

    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """Resolves path against workspace_root, enforcing strict containment.

        Raises:
            ValueError: If path is empty, whitespace, or contains null bytes.
            PermissionError: If resolved path or symlink escapes workspace_root.
        """
        if path is None:
            raise ValueError("Path argument cannot be None.")
        path_str = str(path).strip()
        if not path_str:
            raise ValueError("Path argument cannot be empty or whitespace.")
        if "\0" in path_str:
            raise ValueError("Embedded null byte detected in path.")

        p = Path(path_str)
        if p.is_absolute():
            target = p.resolve()
        else:
            target = (self.workspace_root / p).resolve()

        try:
            target.relative_to(self.workspace_root)
        except ValueError:
            raise PermissionError(
                f"Path traversal detected: '{path_str}' resolves to '{target}' "
                f"which escapes workspace root '{self.workspace_root}'."
            )

        return target

    @staticmethod
    def _kill_process_group(proc: subprocess.Popen) -> None:
        """Terminate the isolated command tree, escalating to SIGKILL if needed."""
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        try:
            # The leader may exit while an ignoring descendant keeps running.
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            try:
                proc.kill()
            except OSError:
                pass

    def read_file(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> ToolResult:
        """Reads file content within workspace, with optional 1-indexed line slicing."""
        try:
            target = self._resolve_path(path)
        except (PermissionError, ValueError) as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Permission denied: {exc}" if isinstance(exc, PermissionError) else f"Invalid path: {exc}",
                metadata={"path": str(path)},
            )

        if not target.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"File not found: '{path}'",
                metadata={"path": str(path)},
            )
        if target.is_dir():
            return ToolResult(
                success=False,
                output="",
                error=f"Path is a directory, not a file: '{path}'",
                metadata={"path": str(path)},
            )

        try:
            raw_bytes = target.read_bytes()
        except OSError as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to read file '{path}': {exc}",
                metadata={"path": str(path)},
            )

        # Binary check (null bytes in first 8KB)
        if b"\x00" in raw_bytes[:8192]:
            return ToolResult(
                success=False,
                output="",
                error=f"Binary file cannot be displayed as text: '{path}'",
                metadata={"path": str(path), "is_binary": True, "size_bytes": len(raw_bytes)},
            )

        # Decoding fallback
        text: Optional[str] = None
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                text = raw_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to decode file '{path}' as text.",
                metadata={"path": str(path), "size_bytes": len(raw_bytes)},
            )

        lines = text.splitlines()
        total_lines = len(lines)

        if total_lines == 0:
            return ToolResult(
                success=True,
                output="",
                error=None,
                metadata={
                    "path": str(path),
                    "total_lines": 0,
                    "start_line": 1,
                    "end_line": 0,
                    "lines_returned": 0,
                },
            )

        # Boundary clamping (1-indexed)
        try:
            actual_start = 1 if start_line is None else int(start_line)
            actual_end = total_lines if end_line is None else int(end_line)
        except (ValueError, TypeError) as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid line indices: {exc}",
                metadata={"path": str(path)},
            )

        actual_start = max(1, actual_start)
        actual_end = min(total_lines, actual_end)

        if actual_start > total_lines or actual_start > actual_end:
            return ToolResult(
                success=True,
                output="",
                error=None,
                metadata={
                    "path": str(path),
                    "total_lines": total_lines,
                    "start_line": actual_start,
                    "end_line": actual_end,
                    "lines_returned": 0,
                },
            )

        if start_line is None and end_line is None and total_lines <= READ_DEFAULT_LINE_LIMIT:
            output = text
            selected_lines = lines
        elif start_line is None and end_line is None:
            # Unranged reads of large files return a bounded window so one tool call
            # cannot fill the local model's context; the note says how to read more.
            selected_lines = lines[:READ_DEFAULT_LINE_LIMIT]
            actual_end = READ_DEFAULT_LINE_LIMIT
            output = "\n".join(selected_lines) + (
                f"\n... [showing lines 1-{READ_DEFAULT_LINE_LIMIT} of {total_lines}; "
                "call read_file with start_line/end_line for more] ..."
            )
        else:
            selected_lines = lines[actual_start - 1 : actual_end]
            output = "\n".join(selected_lines)

        return ToolResult(
            success=True,
            output=output,
            error=None,
            metadata={
                "path": str(path),
                "total_lines": total_lines,
                "start_line": actual_start,
                "end_line": actual_end,
                "lines_returned": len(selected_lines),
            },
        )

    def write_file(self, path: str, content: str) -> ToolResult:
        """Atomically writes content to file within workspace, auto-creating parent directories."""
        try:
            target = self._resolve_path(path)
        except (PermissionError, ValueError) as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Permission denied: {exc}" if isinstance(exc, PermissionError) else f"Invalid path: {exc}",
                metadata={"path": str(path)},
            )

        # .git directory protection
        try:
            rel_parts = target.relative_to(self.workspace_root).parts
            if any(p == ".git" or p.startswith(".git") for p in rel_parts):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Permission denied: Writing to .git directory is prohibited: '{path}'",
                    metadata={"path": str(path)},
                )
        except ValueError:
            return ToolResult(
                success=False,
                output="",
                error="Path traversal check failed.",
                metadata={"path": str(path)},
            )

        if target.exists() and target.is_dir():
            return ToolResult(
                success=False,
                output="",
                error=f"Target path is an existing directory: '{path}'",
                metadata={"path": str(path)},
            )

        parent_dir = target.parent
        try:
            parent_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to create parent directory '{parent_dir}': {exc}",
                metadata={"path": str(path)},
            )

        temp_path: Optional[Path] = None
        try:
            # Atomic write via same-directory tempfile and rename
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=parent_dir,
                prefix=f".{target.name}.tmp_",
                delete=False,
            ) as tmp:
                temp_path = Path(tmp.name)
                tmp.write(content)
                tmp.flush()
                os.fsync(tmp.fileno())

            os.replace(temp_path, target)
        except Exception as exc:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to write file '{path}': {exc}",
                metadata={"path": str(path)},
            )

        encoded_len = len(content.encode("utf-8"))
        lines_count = len(content.splitlines())
        return ToolResult(
            success=True,
            output=f"Successfully wrote {encoded_len} bytes ({lines_count} lines) to '{path}'",
            error=None,
            metadata={
                "path": str(path),
                "resolved_path": str(target),
                "bytes_written": encoded_len,
                "lines_written": lines_count,
            },
        )

    def list_dir(self, path: str = ".") -> ToolResult:
        """Lists entries in a directory within the workspace."""
        try:
            target = self._resolve_path(path)
        except (PermissionError, ValueError) as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Permission denied: {exc}" if isinstance(exc, PermissionError) else f"Invalid path: {exc}",
                metadata={"path": str(path)},
            )

        if not target.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"Directory not found: '{path}'",
                metadata={"path": str(path)},
            )
        if not target.is_dir():
            return ToolResult(
                success=False,
                output="",
                error=f"Path is not a directory: '{path}'",
                metadata={"path": str(path)},
            )

        try:
            items = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            entries = [f"{item.name}/" if item.is_dir() else item.name for item in items]
        except OSError as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to list directory '{path}': {exc}",
                metadata={"path": str(path)},
            )

        output = "\n".join(entries) if entries else "(empty directory)"
        return ToolResult(
            success=True,
            output=output,
            error=None,
            metadata={
                "path": str(path),
                "entries": entries,
                "count": len(entries),
            },
        )

    def patch_file(
        self,
        path: str,
        target_content: str,
        replacement_content: str,
        allow_multiple: bool = False,
    ) -> ToolResult:
        """Atomically replaces target_content with replacement_content in an existing workspace file."""
        try:
            target = self._resolve_path(path)
        except (PermissionError, ValueError) as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Permission denied: {exc}" if isinstance(exc, PermissionError) else f"Invalid path: {exc}",
                metadata={"path": str(path)},
            )

        try:
            rel_parts = target.relative_to(self.workspace_root).parts
            if any(p == ".git" or p.startswith(".git") for p in rel_parts):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Permission denied: Writing to .git directory is prohibited: '{path}'",
                    metadata={"path": str(path)},
                )
        except ValueError:
            return ToolResult(
                success=False,
                output="",
                error="Path traversal check failed.",
                metadata={"path": str(path)},
            )

        if not target_content:
            return ToolResult(
                success=False,
                output="",
                error="target_content cannot be empty.",
                metadata={"path": str(path)},
            )

        read_res = self.read_file(path, start_line=1, end_line=10_000_000)
        if not read_res.success:
            return read_res

        try:
            full_text = target.read_text(encoding="utf-8")
        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to read file '{path}' for patching: {exc}",
                metadata={"path": str(path)},
            )

        occurrences = full_text.count(target_content)
        if occurrences == 0:
            return ToolResult(
                success=False,
                output="",
                error=f"Target content not found in '{path}'.",
                metadata={"path": str(path), "occurrences": 0},
            )
        if occurrences > 1 and not allow_multiple:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Target content matched {occurrences} times in '{path}'; "
                    "set allow_multiple=True or provide more surrounding context."
                ),
                metadata={"path": str(path), "occurrences": occurrences},
            )

        replaced_count = occurrences if allow_multiple else 1
        new_text = (
            full_text.replace(target_content, replacement_content)
            if allow_multiple
            else full_text.replace(target_content, replacement_content, 1)
        )
        write_res = self.write_file(path, new_text)
        if not write_res.success:
            return write_res

        return ToolResult(
            success=True,
            output=f"Successfully patched {replaced_count} occurrence(s) in '{path}'",
            error=None,
            metadata={
                "path": str(path),
                "replacements": replaced_count,
                "bytes_written": write_res.metadata.get("bytes_written", 0),
            },
        )

    def run_command(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        idle_timeout: Optional[float] = None,
    ) -> ToolResult:
        """Execute a command, enforcing optional wall and inactivity timeouts."""
        timeout = self.default_command_timeout if timeout is None else timeout
        idle_timeout = self.default_idle_timeout if idle_timeout is None else idle_timeout
        if not command or not command.strip():
            return ToolResult(
                success=False,
                output="",
                error="Command cannot be empty.",
                metadata={"command": command},
            )

        if self._PACKAGE_INSTALL_COMMAND.search(command):
            return ToolResult(
                success=False,
                output="",
                error=(
                    "Package installation is disabled for autonomous workspace commands. "
                    "Use dependencies already present in the environment and record missing "
                    "packages as a limitation."
                ),
                metadata={"command": command, "blocked_by_policy": True},
            )

        if cwd is None:
            effective_cwd = self.workspace_root
        else:
            try:
                effective_cwd = self._resolve_path(cwd)
                if not effective_cwd.is_dir():
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Working directory does not exist or is not a directory: '{cwd}'",
                        metadata={"cwd": cwd},
                    )
            except (PermissionError, ValueError) as exc:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Invalid working directory: {exc}",
                    metadata={"cwd": cwd},
                )

        shell_exec = "/bin/bash" if Path("/bin/bash").exists() else None
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        start_time = time.perf_counter()
        timed_out = False
        timeout_reason = ""
        proc = None
        stdout = ""
        stderr = ""

        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                executable=shell_exec,
                cwd=str(effective_cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                start_new_session=True,  # os.setsid creates new process group
                env=env,
            )
            selector = selectors.DefaultSelector()
            out_bytes = bytearray()
            err_bytes = bytearray()
            selector.register(proc.stdout, selectors.EVENT_READ, out_bytes)
            selector.register(proc.stderr, selectors.EVENT_READ, err_bytes)
            last_activity = time.monotonic()
            command_started = time.monotonic()
            deadline = command_started + timeout if timeout is not None and timeout > 0 else None
            while selector.get_map() or proc.poll() is None:
                stop = self.execution_control.check()
                if stop.stopped:
                    timeout_reason = f"Execution paused: {stop.reason}"
                    timed_out = True
                    self._kill_process_group(proc)
                    break
                now = time.monotonic()
                remaining = []
                if deadline is not None:
                    remaining.append(max(0.0, deadline - now))
                if idle_timeout is not None and idle_timeout > 0:
                    remaining.append(max(0.0, idle_timeout - (now - last_activity)))
                if self.execution_control is not None:
                    # A quiet process may keep both pipes open indefinitely.
                    # Poll periodically so run-budget cancellation is observed
                    # even when the operator disabled inactivity timeouts.
                    remaining.append(0.2)
                wait_for = min(remaining) if remaining else None
                if not selector.get_map() and wait_for is None:
                    wait_for = 0.2
                ready = selector.select(wait_for)
                if not ready:
                    now = time.monotonic()
                    if deadline is not None and now >= deadline:
                        timeout_reason = f"Command timed out after {timeout} seconds."
                    elif idle_timeout is not None and idle_timeout > 0 and now - last_activity >= idle_timeout:
                        timeout_reason = f"Command produced no output for {idle_timeout} seconds."
                    if timeout_reason:
                        timed_out = True
                        self._kill_process_group(proc)
                        break
                    continue
                for key, _ in ready:
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if chunk:
                        key.data.extend(chunk)
                        last_activity = time.monotonic()
                    else:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
            selector.close()
            if timed_out:
                proc.wait()
            else:
                proc.wait(timeout=timeout)
            stdout = out_bytes.decode("utf-8", errors="replace")
            stderr = err_bytes.decode("utf-8", errors="replace")
        except KeyboardInterrupt:
            if proc:
                self._kill_process_group(proc)
                try:
                    proc.wait(timeout=2)
                except Exception:
                    pass
            raise
        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Command execution error: {exc}",
                metadata={"command": command, "cwd": str(effective_cwd)},
            )

        duration = time.perf_counter() - start_time
        exit_code = -1 if timed_out else (proc.returncode if proc else -1)

        metadata = {
            "command": command,
            "cwd": str(effective_cwd),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "duration_s": round(duration, 3),
            "stderr": stderr,
        }

        if timed_out:
            return ToolResult(
                success=False,
                output=stdout or "",
                error=timeout_reason,
                metadata=metadata,
            )

        if exit_code == 0:
            return ToolResult(
                success=True,
                output=stdout or "",
                error=None,
                metadata=metadata,
            )
        else:
            err_msg = stderr.strip() if stderr and stderr.strip() else f"Command failed with exit code {exit_code}"
            return ToolResult(
                success=False,
                output=stdout or "",
                error=err_msg,
                metadata=metadata,
            )

    @property
    def is_git_repo(self) -> bool:
        """Returns True if the workspace root contains a .git directory."""
        return (self.workspace_root / ".git").exists()

    def git_status(self) -> ToolResult:
        """Runs 'git status --porcelain', gracefully handling non-git workspaces."""
        try:
            res = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                output="",
                error="Git executable not found in system PATH.",
                metadata={"is_git_repo": False, "git_installed": False},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error="Git status command timed out.",
                metadata={"timed_out": True},
            )

        if res.returncode != 0:
            if "not a git repository" in res.stderr.lower():
                return ToolResult(
                    success=True,
                    output="Workspace is not a Git repository.",
                    error=None,
                    metadata={"is_git_repo": False, "clean": True, "changes": []},
                )
            return ToolResult(
                success=False,
                output="",
                error=res.stderr.strip() or f"git status failed with exit code {res.returncode}",
                metadata={"is_git_repo": False, "exit_code": res.returncode},
            )

        raw_output = res.stdout
        clean = not raw_output.strip()
        change_paths = [line[3:].strip() for line in raw_output.splitlines() if len(line) >= 3]
        change_entries = [
            {"code": line[:2], "path": line[3:].strip()}
            for line in raw_output.splitlines()
            if len(line) >= 3
        ]

        output_str = "Clean working tree (no changes)" if clean else raw_output
        return ToolResult(
            success=True,
            output=output_str,
            error=None,
            metadata={
                "is_git_repo": True,
                "clean": clean,
                "changes": change_paths,
                "change_entries": change_entries,
                "change_count": len(change_paths),
            },
        )

    def git_diff(self, staged: bool = False) -> ToolResult:
        """Runs 'git diff' (or 'git diff --staged'), gracefully handling non-git workspaces."""
        cmd = ["git", "diff", "--staged"] if staged else ["git", "diff"]
        try:
            res = subprocess.run(
                cmd,
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                output="",
                error="Git executable not found in system PATH.",
                metadata={"is_git_repo": False, "git_installed": False},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error="Git diff command timed out.",
                metadata={"timed_out": True},
            )

        if res.returncode != 0:
            if "not a git repository" in res.stderr.lower():
                return ToolResult(
                    success=True,
                    output="Workspace is not a Git repository.",
                    error=None,
                    metadata={"is_git_repo": False, "staged": staged, "has_diff": False},
                )
            return ToolResult(
                success=False,
                output="",
                error=res.stderr.strip() or f"git diff failed with exit code {res.returncode}",
                metadata={"is_git_repo": False, "exit_code": res.returncode},
            )

        diff_text = res.stdout
        has_diff = bool(diff_text.strip())
        return ToolResult(
            success=True,
            output=diff_text,
            error=None,
            metadata={
                "is_git_repo": True,
                "staged": staged,
                "has_diff": has_diff,
                "diff_bytes": len(diff_text.encode("utf-8")),
            },
        )

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Generates JSON schema tool definitions for model function calling."""
        tools = [
            {
                "name": "read_file",
                "description": "Read contents of a file within the workspace, optionally within a line range.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path of the file to read within the workspace.",
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "Optional 1-indexed line number to start reading from.",
                        },
                        "end_line": {
                            "type": "integer",
                            "description": "Optional 1-indexed line number to stop reading at (inclusive).",
                        },
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "write_file",
                "description": "Atomically write content to a file in the workspace, creating parent directories.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path of the file to write within the workspace.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Complete text content to write into the file.",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
            {
                "name": "patch_file",
                "description": "Replace an exact substring in an existing file without rewriting the entire file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path of the existing file to patch within the workspace.",
                        },
                        "target_content": {
                            "type": "string",
                            "description": "Exact existing text to search for and replace.",
                        },
                        "replacement_content": {
                            "type": "string",
                            "description": "New text to replace target_content with.",
                        },
                        "allow_multiple": {
                            "type": "boolean",
                            "description": "If true, replace all occurrences; if false (default), require a unique match.",
                        },
                    },
                    "required": ["path", "target_content", "replacement_content"],
                },
            },
            {
                "name": "list_dir",
                "description": "List immediate files and subdirectories inside a workspace directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative directory path within the workspace (defaults to '.').",
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": "run_command",
                "description": "Execute a bash shell command inside the workspace within a process group.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Bash command string to execute.",
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Optional working directory relative to workspace root.",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Optional total wall-clock limit in seconds; inactivity is governed separately.",
                        },
                    },
                    "required": ["command"],
                },
            },
            {
                "name": "git_status",
                "description": "Show working tree status (changes, untracked files) via git status --porcelain.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "git_diff",
                "description": "Show unified git diff of unstaged or staged modifications in the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "staged": {
                            "type": "boolean",
                            "description": "If true, view staged changes (--staged); if false, view unstaged diff.",
                        }
                    },
                    "required": [],
                },
            },
        ]

        formatted_defs = []
        for t in tools:
            formatted_defs.append(
                {
                    "type": "function",
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["parameters"],
                    },
                }
            )
        return formatted_defs

    def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Dynamically dispatches a tool call by name with keyword arguments."""
        tool_map = {
            "read_file": self.read_file,
            "write_file": self.write_file,
            "patch_file": self.patch_file,
            "list_dir": self.list_dir,
            "run_command": self.run_command,
            "git_status": self.git_status,
            "git_diff": self.git_diff,
        }
        handler = tool_map.get(tool_name)
        if handler is None:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool '{tool_name}'. Available: {sorted(tool_map.keys())}",
                metadata={"tool_name": tool_name, "available_tools": sorted(tool_map.keys())},
            )

        try:
            return handler(**kwargs)
        except TypeError as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid arguments for tool '{tool_name}': {exc}",
                metadata={"tool_name": tool_name, "kwargs": kwargs},
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Unexpected error executing tool '{tool_name}': {exc}",
                metadata={"tool_name": tool_name},
            )
