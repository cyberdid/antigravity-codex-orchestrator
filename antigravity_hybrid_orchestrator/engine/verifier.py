"""Subtask verification runner and failure diagnostics extractor."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from antigravity_hybrid_orchestrator.planner.dag import SubTask
from antigravity_hybrid_orchestrator.workforce.tools import WorkspaceToolSuite


@dataclass
class VerificationResult:
    """Represents the outcome of executing a subtask verification command."""

    success: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    failure_reason: str = ""
    duration_sec: float = 0.0
    command: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes verification result into a dictionary."""
        return {
            "success": self.success,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "failure_reason": self.failure_reason,
            "duration_sec": self.duration_sec,
            "command": self.command,
            "metadata": dict(self.metadata),
        }


class VerificationRunner:
    """Executes subtask verification commands in the sandboxed workspace.

    Captures exit codes, stdout, stderr, and extracts concise diagnostic summaries
    for feeding into the self-correction repair loop.
    """

    def __init__(
        self,
        tool_suite: WorkspaceToolSuite,
        default_timeout: Optional[int] = None,
    ) -> None:
        self.tool_suite = tool_suite
        self.default_timeout = default_timeout

    def run_verification(
        self,
        subtask: SubTask,
        timeout: Optional[int] = None,
    ) -> VerificationResult:
        """Executes the subtask's verification command if specified.

        Args:
            subtask: Target SubTask containing verification_command.
            timeout: Command execution timeout in seconds.

        Returns:
            VerificationResult indicating success or failure with diagnostics.
        """
        command = (subtask.verification_command or "").strip()
        if not command:
            return VerificationResult(
                success=True,
                exit_code=0,
                stdout="No verification command specified.",
                stderr="",
                failure_reason="",
                duration_sec=0.0,
                command="",
            )

        cmd_timeout = timeout if timeout is not None else self.default_timeout
        start_time = time.monotonic()
        tool_res = self.tool_suite.run_command(command=command, timeout=cmd_timeout)
        duration = round(time.monotonic() - start_time, 3)

        exit_code = int(tool_res.metadata.get("exit_code", 0 if tool_res.success else 1))
        stdout = tool_res.output or ""
        stderr = tool_res.error or ""

        if tool_res.success and exit_code == 0:
            return VerificationResult(
                success=True,
                exit_code=0,
                stdout=stdout,
                stderr=stderr,
                failure_reason="",
                duration_sec=duration,
                command=command,
                metadata=dict(tool_res.metadata),
            )

        # Command failed: extract root diagnostic cause
        failure_reason = self.extract_failure_diagnostics(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
        )

        return VerificationResult(
            success=False,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            failure_reason=failure_reason,
            duration_sec=duration,
            command=command,
            metadata=dict(tool_res.metadata),
        )

    @staticmethod
    def extract_failure_diagnostics(
        exit_code: int,
        stdout: str,
        stderr: str,
    ) -> str:
        """Extracts concise error traces, assertion messages, or syntax errors."""
        combined = f"{stderr}\n{stdout}".strip()
        if not combined:
            return f"Command exited with non-zero exit code {exit_code} (no output produced)."

        # Look for Python Exception tracebacks
        tb_match = re.findall(
            r"(?:Traceback \(most recent call last\):[\s\S]*?)(?:[A-Za-z0-9_]+Error: .*|[A-Za-z0-9_]+Exception: .*)",
            combined,
        )
        if tb_match:
            # Return the last (innermost) traceback
            return tb_match[-1].strip()

        # Look for Pytest FAILED summary lines
        pytest_failures = re.findall(r"(?:FAILED [^\n]+)", combined)
        if pytest_failures:
            return "\n".join(pytest_failures[-5:]).strip()

        # Look for SyntaxError / IndentationError
        syntax_match = re.search(r"((?:SyntaxError|IndentationError): [^\n]+)", combined)
        if syntax_match:
            return syntax_match.group(1).strip()

        # Fallback to the last 15 lines of output
        lines = [line.strip() for line in combined.splitlines() if line.strip()]
        tail = "\n".join(lines[-15:])
        return f"Exit code {exit_code}:\n{tail}"
