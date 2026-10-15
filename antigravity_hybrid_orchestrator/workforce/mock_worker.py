"""Deterministic Mock Autonomous Worker for fast, offline testing and CI (Milestone 3, Feature F10).

Provides:
- MockAutonomousWorker: High-speed (<0.05s) deterministic simulation of autonomous worker
  executing SubTasks using real WorkspaceToolSuite, with failure injection, realistic code
  synthesis, and HUD event emission.
- Legacy gauntlet workforce client compatibility (MockLocalWorkforceClient, MockWorkforceClient).
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple

from antigravity_hybrid_orchestrator.planner.dag import SubTask
from antigravity_hybrid_orchestrator.workforce.tools import ToolResult, WorkspaceToolSuite
from antigravity_hybrid_orchestrator.workforce.worker import SubtaskResult


class MockAutonomousWorker:
    """Fast, deterministic mock autonomous worker for offline testing and CI.

    Attributes:
        fail_subtasks: Set of subtask IDs that should fail execution.
        error_on_tool: Mapping of tool_name -> error message to inject on tool calls.
        custom_file_contents: Mapping of relative path -> custom content to write.
        max_turns: Maximum simulated ReAct turns before completion.
        event_callback: Callable receiving (event_type, subtask_id, payload).
        token_chunk_size: Compatibility setting for token streaming simulation.
    """

    def __init__(
        self,
        fail_subtasks: Optional[Set[str]] = None,
        error_on_tool: Optional[Dict[str, str]] = None,
        custom_file_contents: Optional[Dict[str, str]] = None,
        max_turns: int = 10,
        event_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
        token_chunk_size: int = 15,
        **kwargs: Any,
    ) -> None:
        self.fail_subtasks: Set[str] = set(fail_subtasks or [])
        self.error_on_tool: Dict[str, str] = dict(error_on_tool or {})
        self.custom_file_contents: Dict[str, str] = dict(custom_file_contents or {})
        self.max_turns = max_turns
        self.event_callback = event_callback
        self.token_chunk_size = token_chunk_size

    def _emit(self, event_type: str, subtask_id: str, payload: Dict[str, Any]) -> None:
        """Safely dispatches HUD event without failing if callback raises."""
        if self.event_callback:
            try:
                self.event_callback(event_type, subtask_id, payload)
            except Exception:
                pass

    def execute_subtask(
        self,
        subtask: SubTask,
        tool_suite: WorkspaceToolSuite,
        feedback_context: Optional[str] = None,
        deadline: Optional[float] = None,  # accepted for interface parity; the mock finishes instantly
    ) -> SubtaskResult:
        """Executes a SubTask deterministically using the provided tool_suite (<0.05s)."""
        turn = 1
        tool_calls_count = 0

        # 1. Emit subtask_start
        self._emit(
            "subtask_start",
            subtask.id,
            {"subtask_id": subtask.id, "title": subtask.title},
        )

        # 2. Emit thought
        thought_msg = f"Planning implementation for subtask '{subtask.id}': {subtask.title}"
        if feedback_context:
            thought_msg += f". Addressing feedback: {feedback_context}"
        self._emit(
            "thought",
            subtask.id,
            {"thought": thought_msg, "turn": turn, "feedback_context": feedback_context},
        )

        # 3. Check injected failure
        if subtask.id in self.fail_subtasks:
            err = f"Injected failure for subtask {subtask.id}"
            self._emit(
                "subtask_failed",
                subtask.id,
                {"subtask_id": subtask.id, "error": err},
            )
            return SubtaskResult(
                subtask_id=subtask.id,
                success=False,
                turns=1,
                tool_calls_count=0,
                error=err,
                metadata={"injected_failure": True},
            )

        # 4. Process affected files
        for file_path in subtask.affected_files:
            # Check existing file read
            candidate = tool_suite.workspace_root / file_path
            if candidate.exists() and candidate.is_file():
                turn += 1
                tool_calls_count += 1
                self._emit(
                    "tool_call",
                    subtask.id,
                    {"tool": "read_file", "args": {"path": file_path}, "turn": turn},
                )

                if "read_file" in self.error_on_tool:
                    injected_err = self.error_on_tool["read_file"]
                    read_res = ToolResult(success=False, output="", error=injected_err)
                else:
                    read_res = tool_suite.read_file(file_path)

                self._emit(
                    "tool_result",
                    subtask.id,
                    {
                        "tool": "read_file",
                        "success": read_res.success,
                        "output": read_res.output,
                        "error": read_res.error,
                    },
                )

            # Determine content to write
            if file_path in self.custom_file_contents:
                content = self.custom_file_contents[file_path]
            elif file_path.endswith(".py"):
                file_name = Path(file_path).name.lower()
                is_test_file = (
                    file_name.startswith("test_")
                    or file_name.endswith("_test.py")
                    or "tests/" in file_path.replace("\\", "/").lower()
                )
                if is_test_file:
                    content = "def test_example():\n    assert True\n\ndef test_solution():\n    assert True\n"
                else:
                    combined_desc = f"{subtask.title} {subtask.description} {subtask.success_criteria}".lower()
                    import re
                    words = set(re.findall(r"\w+", combined_desc))
                    if any(k in words for k in ("add", "math", "calc", "calculator", "addition")):
                        content = (
                            "def add(a: int, b: int) -> int:\n"
                            "    return a + b\n\n"
                            "def execute() -> bool:\n"
                            "    return True\n"
                        )
                    else:
                        content = (
                            f'"""Generated implementation for {subtask.title}."""\n\n'
                            "def execute() -> bool:\n"
                            "    return True\n\n"
                            'if __name__ == "__main__":\n'
                            '    print("Success")\n'
                        )

            elif file_path.endswith(".json"):
                content = json.dumps({"status": "ok", "task_id": subtask.id}, indent=2)
            elif file_path.endswith((".md", ".txt")):
                content = f"# {subtask.title}\n\nCompleted successfully.\n"
            else:
                content = f"# Content for {file_path}\n"

            # Execute write_file
            turn += 1
            tool_calls_count += 1
            self._emit(
                "tool_call",
                subtask.id,
                {"tool": "write_file", "args": {"path": file_path, "content": content}, "turn": turn},
            )

            if "write_file" in self.error_on_tool:
                injected_err = self.error_on_tool["write_file"]
                self._emit(
                    "tool_result",
                    subtask.id,
                    {"tool": "write_file", "success": False, "output": "", "error": injected_err},
                )
                self._emit(
                    "subtask_failed",
                    subtask.id,
                    {"subtask_id": subtask.id, "error": injected_err},
                )
                return SubtaskResult(
                    subtask_id=subtask.id,
                    success=False,
                    turns=turn,
                    tool_calls_count=tool_calls_count,
                    error=injected_err,
                    metadata={"tool_error": injected_err},
                )

            write_res = tool_suite.write_file(file_path, content)
            self._emit(
                "tool_result",
                subtask.id,
                {
                    "tool": "write_file",
                    "success": write_res.success,
                    "output": write_res.output,
                    "error": write_res.error,
                },
            )

            if not write_res.success:
                self._emit(
                    "subtask_failed",
                    subtask.id,
                    {"subtask_id": subtask.id, "error": write_res.error},
                )
                return SubtaskResult(
                    subtask_id=subtask.id,
                    success=False,
                    turns=turn,
                    tool_calls_count=tool_calls_count,
                    error=write_res.error,
                )

        # 5. Execute verification command if provided
        if subtask.verification_command:
            turn += 1
            tool_calls_count += 1
            self._emit(
                "tool_call",
                subtask.id,
                {"tool": "run_command", "args": {"command": subtask.verification_command}, "turn": turn},
            )

            if "run_command" in self.error_on_tool:
                injected_err = self.error_on_tool["run_command"]
                self._emit(
                    "tool_result",
                    subtask.id,
                    {"tool": "run_command", "success": False, "output": "", "error": injected_err},
                )
                self._emit(
                    "subtask_failed",
                    subtask.id,
                    {"subtask_id": subtask.id, "error": injected_err},
                )
                return SubtaskResult(
                    subtask_id=subtask.id,
                    success=False,
                    turns=turn,
                    tool_calls_count=tool_calls_count,
                    error=injected_err,
                    metadata={"tool_error": injected_err},
                )

            cmd_res = tool_suite.run_command(subtask.verification_command)
            self._emit(
                "tool_result",
                subtask.id,
                {
                    "tool": "run_command",
                    "success": cmd_res.success,
                    "output": cmd_res.output,
                    "error": cmd_res.error,
                },
            )

            if not cmd_res.success:
                err_msg = cmd_res.error or f"Verification command failed with exit code {cmd_res.metadata.get('exit_code')}"
                self._emit(
                    "subtask_failed",
                    subtask.id,
                    {"subtask_id": subtask.id, "error": err_msg},
                )
                return SubtaskResult(
                    subtask_id=subtask.id,
                    success=False,
                    turns=turn,
                    tool_calls_count=tool_calls_count,
                    error=err_msg,
                    metadata={"exit_code": cmd_res.metadata.get("exit_code")},
                )

        # 6. Complete subtask successfully
        summary = f"Subtask '{subtask.id}' completed successfully."
        self._emit(
            "subtask_complete",
            subtask.id,
            {"subtask_id": subtask.id, "summary": summary},
        )
        return SubtaskResult(
            subtask_id=subtask.id,
            success=True,
            turns=turn,
            tool_calls_count=tool_calls_count,
            summary=summary,
            artifacts=list(subtask.affected_files),
        )

    # Legacy gauntlet workforce client compatibility
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> Generator[Tuple[str, str, Dict[str, Any]], None, None]:
        accumulated = ""
        mock_response = "```python\n# Patched module\npass\n```\n"

        words = mock_response.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            accumulated += chunk
            yield chunk, accumulated, {"done": False}

        yield "", accumulated, {
            "done": True,
            "eval_count": 420,
            "eval_duration_s": 0.05,
            "prompt_eval_count": 150,
            "total_duration_s": 0.06,
        }

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        for _, accumulated, meta in self.generate_stream(prompt, system_prompt, temperature):
            if meta.get("done"):
                return accumulated
        return "```python\n# Patched module\npass\n```\n"


# Re-exports for legacy client aliases
MockLocalWorkforceClient = MockAutonomousWorker
MockWorkforceClient = MockAutonomousWorker
