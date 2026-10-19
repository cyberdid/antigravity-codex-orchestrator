"""Autonomous Workforce Agent and ReAct tool-calling loop (Milestone 3, Feature F9).

Provides:
- SubtaskResult: Standardized container for autonomous subtask execution results.
- ParsedAction: Structured container supporting attribute access and tuple unpacking.
- AutonomousWorker: ReAct loop executing subtasks via WorkspaceToolSuite with single-concurrency lock.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import hashlib
import os
import re
import threading
import time
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple


from antigravity_hybrid_orchestrator.planner.dag import SubTask
from antigravity_hybrid_orchestrator.engine.control import ExecutionControl
from antigravity_hybrid_orchestrator.workforce.tools import ToolResult, WorkspaceToolSuite
from antigravity_hybrid_orchestrator.workforce.context import (
    compact_messages,
    effective_num_predict,
    num_ctx,
    ollama_think,
    prune_tool_output,
    strip_completed_thinking,
)
from antigravity_hybrid_orchestrator.workforce.recovery_controller import (
    RecoveryController,
    active_recovery_policy,
)


@dataclass
class SubtaskResult:
    """Standardized result envelope returned by autonomous worker execution.

    Attributes:
        subtask_id: Unique identifier of the executed subtask.
        success: True if subtask completed successfully and passed verification.
        turns: Number of ReAct iterations executed.
        tool_calls_count: Total number of tool calls executed.
        summary: Human-readable completion summary.
        error: Detailed error message if failed, otherwise None.
        artifacts: List of modified or created file paths.
        metadata: Execution diagnostics and telemetry.
    """

    subtask_id: str
    success: bool
    turns: int = 1
    tool_calls_count: int = 0
    summary: str = ""
    error: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes SubtaskResult to a JSON-compatible dictionary."""
        return {
            "subtask_id": self.subtask_id,
            "success": self.success,
            "turns": self.turns,
            "tool_calls_count": self.tool_calls_count,
            "summary": self.summary,
            "error": self.error,
            "artifacts": list(self.artifacts),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubtaskResult":
        """Reconstructs SubtaskResult from a dictionary representation with defaults."""
        return cls(
            subtask_id=str(data.get("subtask_id", "")),
            success=bool(data.get("success", False)),
            turns=int(data.get("turns", 1)),
            tool_calls_count=int(data.get("tool_calls_count", 0)),
            summary=str(data.get("summary", "")),
            error=data.get("error"),
            artifacts=list(data.get("artifacts", [])),
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serializes to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> "SubtaskResult":
        """Deserializes from a JSON string."""
        return cls.from_dict(json.loads(s))

    def __bool__(self) -> bool:
        """Allows direct truthiness evaluation based on success."""
        return self.success


@dataclass
class ParsedAction:
    """Parsed model action supporting attribute access, tuple unpacking, and indexing.

    Allows both:
        is_final, action, args = AutonomousWorker._parse_action(text)
    and:
        parsed = AutonomousWorker._parse_action(text)
        if parsed.is_final: ...
    """

    is_final: bool
    action: str
    args: Dict[str, Any] = field(default_factory=dict)
    thought: str = ""
    summary: str = ""
    error: Optional[str] = None

    def __iter__(self):
        """Supports 3-tuple unpacking: (is_final, action_or_summary, args)."""
        action_or_summary = self.summary if self.is_final else self.action
        return iter((self.is_final, action_or_summary, self.args))

    def __getitem__(self, index: int) -> Any:
        """Supports sequence indexing."""
        action_or_summary = self.summary if self.is_final else self.action
        return (self.is_final, action_or_summary, self.args)[index]

    def __len__(self) -> int:
        return 3


class AutonomousWorker:
    """Autonomous workforce agent executing SubTasks via a ReAct tool-calling loop.

    Uses single-concurrency serialization (threading.Lock) to prevent GPU VRAM
    exhaustion on the RTX 4060 during Ollama inference.
    """

    _inference_lock = threading.Lock()

    def __init__(
        self,
        model_name: Optional[str] = None,
        host: str = "http://localhost:11434",
        max_turns: Optional[int] = None,
        event_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
        client: Optional[Any] = None,
        execution_control: Optional[ExecutionControl] = None,
        recovery_policy: Optional[str] = None,
    ) -> None:
        self.model_name = (model_name or os.environ.get("OLLAMA_MODEL", "")).strip() or "gemma4:12b"
        self.host = host

        # Kept as a compatibility attribute only. Execution is bounded by
        # cancellation, inactivity watchdogs, and repeated no-progress signals.
        self.max_turns = max_turns
        self.event_callback = event_callback
        self.client = client
        self.recovery_policy = recovery_policy
        self.execution_control = execution_control or ExecutionControl()

    def _emit(self, event_type: str, subtask_id: str, payload: Dict[str, Any]) -> None:
        """Safely invokes event callback for live HUD rendering."""
        if self.event_callback:
            try:
                self.event_callback(event_type, subtask_id, payload)
            except Exception:
                pass

    @staticmethod
    def _repeated_action_cycle(history: List[str], fingerprint: str) -> Optional[int]:
        """Return the repeated cycle length when the latest actions make no progress."""
        history.append(fingerprint)
        del history[:-12]
        for period in range(1, 5):
            window = period * 3
            if len(history) >= window:
                recent = history[-window:]
                if recent[:period] * 3 == recent:
                    return period
        return None

    @classmethod
    def _parse_action(cls, text: str) -> ParsedAction:
        """Parses model output into structured action or completion verdict.

        Supports:
        - JSON function calls (e.g. {"thought": "...", "action": "...", "action_input": {...}})
        - Text ReAct format (Action: <tool> \n Action Input: <json>)
        - Final answer format (Final Answer: <summary> or Action: finish)
        """
        if not text or not text.strip():
            return ParsedAction(is_final=False, action="", error="Empty model response")

        raw = text.strip()

        # Check for JSON block (either raw or wrapped in markdown code fence)
        fence_match = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", raw, re.IGNORECASE)
        cleaned = fence_match.group(1).strip() if fence_match else raw

        candidate_json = cleaned if (cleaned.startswith("{") and cleaned.endswith("}")) else (
            raw if (raw.startswith("{") and raw.endswith("}")) else None
        )

        if candidate_json:
            try:
                obj = json.loads(candidate_json)
                if isinstance(obj, dict):
                    thought = str(obj.get("thought", ""))
                    action = str(obj.get("action", obj.get("name", obj.get("tool", "")))).strip()
                    action_input = obj.get("action_input", obj.get("arguments", obj.get("args", {})))
                    if isinstance(action_input, str):
                        try:
                            action_input = json.loads(action_input)
                        except Exception:
                            pass
                    if not isinstance(action_input, dict):
                        action_input = {"input": action_input} if action_input else {}

                    if (
                        action.lower() in ("finish", "final_answer", "complete")
                        or "final_answer" in obj
                        or "summary" in obj
                    ):
                        summary = str(
                            action_input.get("summary", obj.get("final_answer", obj.get("summary", "")))
                        )
                        return ParsedAction(
                            is_final=True,
                            action=action or "finish",
                            summary=summary,
                            thought=thought,
                            args=action_input,
                        )

                    return ParsedAction(
                        is_final=False,
                        action=action,
                        args=action_input,
                        thought=thought,
                    )
            except json.JSONDecodeError:
                pass

        # Extract Thought if present
        thought = ""
        thought_match = re.search(
            r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|$)", raw, re.DOTALL | re.IGNORECASE
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        # Check for Final Answer
        final_match = re.search(r"Final Answer:\s*(.+)", raw, re.DOTALL | re.IGNORECASE)
        if final_match:
            summary = final_match.group(1).strip()
            return ParsedAction(
                is_final=True,
                action="finish",
                summary=summary,
                thought=thought,
                args={"summary": summary},
            )

        # Check for Action: <tool>
        action_match = re.search(r"Action:\s*([a-zA-Z0-9_\-]+)", raw, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip()

            # Check for Action Input:
            input_match = re.search(r"Action Input:\s*(.+)$", raw, re.DOTALL | re.IGNORECASE)
            args: Dict[str, Any] = {}
            error: Optional[str] = None

            if input_match:
                input_raw = input_match.group(1).strip()
                if input_raw.startswith("```"):
                    sub_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", input_raw, re.IGNORECASE)
                    if sub_match:
                        input_raw = sub_match.group(1).strip()

                try:
                    parsed_args = json.loads(input_raw)
                    if isinstance(parsed_args, dict):
                        args = parsed_args
                    else:
                        args = {"value": parsed_args}
                except json.JSONDecodeError as exc:
                    error = f"Failed to parse Action Input JSON: {exc}"
            else:
                args = {}

            if action.lower() in ("finish", "complete", "final_answer"):
                summary = str(args.get("summary", ""))
                return ParsedAction(
                    is_final=True,
                    action=action,
                    summary=summary,
                    thought=thought,
                    args=args,
                    error=error,
                )

            return ParsedAction(
                is_final=False,
                action=action,
                args=args,
                thought=thought,
                error=error,
            )

        # Fallback if no specific action pattern matches
        return ParsedAction(
            is_final=True,
            action="finish",
            summary=raw,
            thought=thought,
            args={"summary": raw},
        )

    @staticmethod
    def _run_client(client: Any, prompt: str, system_prompt: Optional[str]) -> Optional[str]:
        """Runs a model client via generate(), generate_stream() or __call__.

        Returns the full accumulated text, or None if the client exposes no
        recognized inference method.
        """
        if hasattr(client, "generate"):
            return client.generate(prompt=prompt, system_prompt=system_prompt)
        if hasattr(client, "generate_stream"):
            accumulated = ""
            for _chunk, acc, _meta in client.generate_stream(
                prompt=prompt, system_prompt=system_prompt
            ):
                accumulated = acc
            return accumulated
        if hasattr(client, "chat"):
            return client.chat(prompt=prompt)
        if callable(client):
            return client(prompt)
        return None

    def _call_model(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Invokes model client with fallback handling."""
        if self.client is not None:
            result = self._run_client(self.client, prompt, system_prompt)
            if result is not None:
                return result

        # Default Ollama client invocation (WorkforceClient exposes generate_stream).
        try:
            from antigravity_hybrid_orchestrator.workforce.ollama_client import WorkforceClient

            client = WorkforceClient(base_url=self.host, model=self.model_name)
            result = self._run_client(client, prompt, system_prompt)
            if result is None:
                raise RuntimeError("Ollama client exposed no usable inference method")
            return result
        except Exception as exc:
            return f"Final Answer: Simulated completion due to offline workforce: {exc}"

    def _is_native_capable(self) -> bool:
        """True when the worker should use native Ollama tool-calling.

        Real backend (no injected client) supports `/api/chat` tools; an injected
        client opts in only if it exposes a `chat` method. Text-completion mock
        clients (used in tests) fall back to the legacy ReAct loop.
        """
        if self.client is None:
            return True
        return hasattr(self.client, "chat")

    def _get_chat_client(self) -> Any:
        """Returns a chat-capable client (injected one, or a real WorkforceClient)."""
        if self.client is not None and hasattr(self.client, "chat"):
            return self.client
        from antigravity_hybrid_orchestrator.workforce.ollama_client import WorkforceClient

        idle_timeout = self.execution_control.budget.model_idle_timeout_s
        return WorkforceClient(
            base_url=self.host,
            model=self.model_name,
            timeout=idle_timeout,
        )

    @contextmanager
    def _inference_slot(self, deadline=None):
        while True:
            stop = self.execution_control.check()
            reason = stop.reason if stop.stopped else ""
            if not reason and deadline is not None and time.monotonic() >= deadline:
                reason = "subtask_time_limit"
            if reason:
                yield reason
                return
            if self._inference_lock.acquire(timeout=.05):
                break
        try:
            yield ""
        finally:
            self._inference_lock.release()

    def execute_subtask(
        self,
        subtask: SubTask,
        tool_suite: WorkspaceToolSuite,
        feedback_context: Optional[str] = None,
        deadline: Optional[float] = None,
    ) -> SubtaskResult:
        """Executes a SubTask, preferring native tool-calling over legacy ReAct.

        ``deadline`` is a ``time.monotonic()`` instant after which the attempt stops
        or during a native HTTP call, so a blocked model cannot evade the limit.
        """
        if self._is_native_capable():
            return self._execute_native(subtask, tool_suite, feedback_context, deadline)
        return self._execute_react(subtask, tool_suite, feedback_context)

    def _execute_native(
        self,
        subtask: SubTask,
        tool_suite: WorkspaceToolSuite,
        feedback_context: Optional[str] = None,
        deadline: Optional[float] = None,
    ) -> SubtaskResult:
        """Executes a SubTask via Ollama native function-calling (`/api/chat` tools).

        Serialized across threads via class-level _inference_lock to preserve RTX 4060 VRAM.
        """
        with self._inference_slot(deadline) as stop_reason:
            if stop_reason:
                return SubtaskResult(
                    subtask_id=subtask.id, success=False, turns=0,
                    error=f"Execution paused: {stop_reason}",
                    metadata={"paused": True, "stop_reason": stop_reason},
                )
            self._emit(
                "subtask_start",
                subtask.id,
                {"subtask_id": subtask.id, "title": subtask.title},
            )

            tools = tool_suite.get_tool_definitions()
            system_prompt = (
                "You are an expert autonomous R&D coding workforce running locally on an "
                "RTX 4060 GPU. You accomplish the task by CALLING the provided tools "
                "(read_file, write_file, patch_file, list_dir, run_command, git_status, git_diff) — do not just "
                "describe what to do. Treat Input Files and Expected Outputs as authoritative. "
                "Never invent source paths. If a path is missing, inspect the supplied file "
                "list and report the gap instead of guessing. Do not install packages, use "
                "system package managers, or access the network; report unavailable runtime "
                "dependencies as limitations. Write real file contents with write_file. When a "
                "write_file call is required, always pass BOTH required JSON arguments with "
                "\"path\" FIRST (the exact relative path from Expected Outputs) followed by \"content\" "
                "(the full text), and emit at most ONE write_file call per turn so output is never truncated. "
                "Never call write_file with content alone. When a "
                "verification command is given, run it with run_command and fix failures. "
                "Only stop (reply with a short summary and NO tool call) once the work is "
                "done and verification passes."
            )
            user_msg = (
                f"SubTask ID: {subtask.id}\n"
                f"Title: {subtask.title}\n"
                f"Description: {subtask.description}\n"
                f"Input Files: {subtask.input_files}\n"
                f"Expected Outputs: {subtask.outputs}\n"
                f"Affected Files: {subtask.affected_files}\n"
                f"Success Criteria: {subtask.success_criteria}\n"
                f"Verification Command: {subtask.verification_command}\n"
            )
            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ]
            if feedback_context:
                messages.append({
                    "role": "user",
                    "content": (
                        f"The previous attempt failed:\n{feedback_context}\n"
                        "Analyze the failure, inspect files if needed, and correct it."
                    ),
                })
                self._emit(
                    "thought",
                    subtask.id,
                    {"thought": f"Addressing feedback: {feedback_context}", "turn": 0},
                )

            client = self._get_chat_client()
            tool_calls_count = 0
            in_tokens = 0
            out_tokens = 0
            cached_prompt_tokens = 0
            eval_duration_total_s = 0.0
            prompt_eval_duration_total_s = 0.0
            turns_with_thinking = 0
            thinking_chars_total = 0
            invalid_tool_arguments = 0
            context_overflows = 0
            truncated_generations = 0
            compacted_messages_total = 0
            successful_verifications = 0
            turn = 0
            action_history: List[str] = []
            turn_trajectory: List[Dict[str, Any]] = []
            files_read_list: List[str] = []
            edit_calls_count = 0
            test_command_calls_count = 0
            subtask_wall_start = time.monotonic()
            recovery_controller = RecoveryController(
                subtask_id=subtask.id,
                input_files=list(subtask.input_files or []),
                affected_files=list(subtask.affected_files or []),
                verification_command=str(subtask.verification_command or ""),
                policy=active_recovery_policy(self.recovery_policy),
            )

            model_digest = (
                client.get_model_digest()
                if hasattr(client, "get_model_digest")
                else "unknown"
            )
            think_cfg = (
                client._effective_think()
                if hasattr(client, "_effective_think")
                else ollama_think()
            )
            active_num_ctx = num_ctx()
            active_num_predict = effective_num_predict(has_tools=bool(tools), num_ctx_val=active_num_ctx)

            def _build_telemetry_meta(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                unique_reads = len(set(files_read_list))
                meta: Dict[str, Any] = {
                    "model": self.model_name,
                    "model_digest": model_digest,
                    "backend": "ollama_native_tools",
                    "think_enabled": bool(think_cfg),
                    "num_ctx": active_num_ctx,
                    "num_predict": active_num_predict,
                    "input_budget_tokens": max(1, active_num_ctx - active_num_predict),
                    "input_tokens": in_tokens,
                    "output_tokens": out_tokens,
                    "prompt_eval_cached_tokens": cached_prompt_tokens,
                    "eval_duration_s": round(eval_duration_total_s, 4),
                    "prompt_eval_duration_s": round(prompt_eval_duration_total_s, 4),
                    "wall_clock_s": round(time.monotonic() - subtask_wall_start, 4),
                    "turns_with_thinking": turns_with_thinking,
                    "thinking_chars": thinking_chars_total,
                    "tool_calls_count": tool_calls_count,
                    "read_file_calls": len(files_read_list),
                    "unique_files_read": unique_reads,
                    "repeated_file_reads": recovery_controller.repeated_reads_count,
                    "edit_calls": edit_calls_count,
                    "test_command_calls": test_command_calls_count,
                    "invalid_tool_arguments": invalid_tool_arguments,
                    "context_overflows": context_overflows,
                    "truncated_generations": truncated_generations,
                    "compacted_messages": compacted_messages_total,
                    "successful_verifications": successful_verifications,
                    **recovery_controller.telemetry_summary(),
                    "turn_trajectory": list(turn_trajectory),
                }
                if extra:
                    meta.update(extra)
                return meta

            try:
                while True:
                    turn += 1
                    if deadline is not None and time.monotonic() >= deadline:
                        return SubtaskResult(
                            subtask_id=subtask.id, success=False, turns=turn - 1,
                            tool_calls_count=tool_calls_count,
                            error="Attempt stopped: subtask time limit reached",
                            artifacts=list(subtask.affected_files),
                            metadata=_build_telemetry_meta({"limit_reached": True}),
                        )
                    stop = self.execution_control.check()
                    if stop.stopped:
                        return SubtaskResult(
                            subtask_id=subtask.id, success=False, turns=turn - 1,
                            tool_calls_count=tool_calls_count,
                            error=f"Execution paused: {stop.reason}",
                            artifacts=list(subtask.affected_files),
                            metadata=_build_telemetry_meta({"paused": True, "stop_reason": stop.reason}),
                        )
                    inference_started = time.monotonic()
                    compacted_messages_total += compact_messages(messages, reserved_tokens=active_num_predict)
                    from antigravity_hybrid_orchestrator.workforce.ollama_client import WorkforceClient
                    if isinstance(client, WorkforceClient):
                        resp = client.chat(messages, tools=tools, deadline=deadline,
                                           execution_control=self.execution_control)
                    else:
                        resp = client.chat(messages, tools=tools)
                    prompt_tokens = int(resp.get("input_tokens", 0))
                    eval_tokens = int(resp.get("output_tokens", 0))
                    in_tokens += prompt_tokens
                    out_tokens += eval_tokens
                    cached_prompt_tokens += int(resp.get("prompt_eval_cached_tokens", 0))
                    compacted_messages_total += int(resp.get("compacted_messages", 0))
                    invalid_tool_arguments += int(resp.get("malformed_tool_calls", 0))
                    if "num_ctx" in resp:
                        active_num_ctx = int(resp["num_ctx"])
                    if "num_predict" in resp:
                        active_num_predict = int(resp["num_predict"])
                    if "think_enabled" in resp:
                        think_cfg = bool(resp["think_enabled"])

                    inference_duration = float(
                        resp.get("eval_duration_s")
                        or (time.monotonic() - inference_started if eval_tokens else 0.0)
                    )
                    eval_duration_total_s += inference_duration
                    prompt_eval_duration_total_s += float(resp.get("prompt_eval_duration_s", 0.0) or 0.0)

                    thinking = str(resp.get("thinking", "") or "")
                    if thinking:
                        turns_with_thinking += 1
                        thinking_chars_total += len(thinking)

                    is_truncated = bool(resp.get("truncated", False)) or (
                        str(resp.get("done_reason", "") or "") == "length"
                    )
                    if is_truncated:
                        truncated_generations += 1

                    if prompt_tokens or eval_tokens:
                        self._emit("model_usage", subtask.id, {
                            "input_tokens": prompt_tokens,
                            "output_tokens": eval_tokens,
                            "duration_s": inference_duration,
                            "turn": turn,
                            "think_enabled": bool(think_cfg),
                            "has_thinking": bool(thinking),
                            "truncated": is_truncated,
                        })
                    content = resp.get("content", "") or ""
                    calls = resp.get("tool_calls", []) or []
                    turn_entry: Dict[str, Any] = {
                        "turn": turn,
                        "input_tokens": prompt_tokens,
                        "output_tokens": eval_tokens,
                        "prompt_eval_cached_tokens": int(resp.get("prompt_eval_cached_tokens", 0)),
                        "eval_duration_s": round(inference_duration, 4),
                        "thinking_chars": len(thinking),
                        "done_reason": str(resp.get("done_reason", "") or ""),
                        "truncated": is_truncated,
                        "tool_calls": [],
                    }
                    turn_trajectory.append(turn_entry)

                    # Thinking content is deliberately NOT emitted to HUD by default.
                    if content.strip():
                        self._emit("thought", subtask.id, {"thought": content, "turn": turn})

                    if not calls:
                        should_continue, recovery_prompt, terminal_err = (
                            recovery_controller.handle_non_tool_turn(
                                turn=turn,
                                content=content,
                                eval_tokens=eval_tokens,
                                is_truncated=is_truncated,
                                active_num_predict=active_num_predict,
                            )
                        )
                        if should_continue and recovery_prompt:
                            strip_completed_thinking(messages)
                            turn_entry["recovery_triggered"] = True
                            turn_entry["recovery_reason"] = (
                                recovery_controller.recovery_events[-1]["reason"]
                                if recovery_controller.recovery_events
                                else "recovery"
                            )
                            messages.append({
                                "role": "user",
                                "content": recovery_prompt,
                            })
                            continue
                        if terminal_err is not None:
                            self._emit(
                                "subtask_failed",
                                subtask.id,
                                {"subtask_id": subtask.id, "error": terminal_err},
                            )
                            return SubtaskResult(
                                subtask_id=subtask.id,
                                success=False,
                                turns=turn,
                                tool_calls_count=tool_calls_count,
                                error=terminal_err,
                                artifacts=list(subtask.affected_files),
                                metadata=_build_telemetry_meta({"truncated": is_truncated}),
                            )

                        # No tool call and not truncated -> the model considers the task complete.
                        strip_completed_thinking(messages)
                        summary = content.strip() or f"Subtask '{subtask.id}' completed."
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
                            metadata=_build_telemetry_meta(),
                        )

                    # Record the assistant turn that requested tools, preserving `thinking`
                    # during the active multi-step function-calling sequence.
                    assistant_entry: Dict[str, Any] = {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [
                            {"function": {"name": c["name"], "arguments": c["arguments"]}}
                            for c in calls
                        ],
                    }
                    if thinking:
                        assistant_entry["thinking"] = thinking
                    messages.append(assistant_entry)

                    for call in calls:
                        stop = self.execution_control.check()
                        if stop.stopped:
                            return SubtaskResult(
                                subtask_id=subtask.id, success=False, turns=turn,
                                tool_calls_count=tool_calls_count,
                                error=f"Execution paused: {stop.reason}",
                                artifacts=list(subtask.affected_files),
                                metadata=_build_telemetry_meta({"paused": True, "stop_reason": stop.reason}),
                            )
                        tool_calls_count += 1
                        name = call.get("name", "")
                        args = call.get("arguments", {}) or {}
                        self._emit(
                            "tool_call",
                            subtask.id,
                            {"tool": name, "args": args, "turn": turn},
                        )
                        try:
                            tool_result = tool_suite.execute(name, **args)
                        except Exception as exc:
                            tool_result = ToolResult(
                                success=False,
                                output="",
                                error=f"Tool execution exception: {exc}",
                                metadata={"exception": str(exc)},
                            )
                        if (
                            not tool_result.success
                            and (
                                (tool_result.error or "").startswith("Invalid arguments for tool")
                                or "Missing required argument" in (tool_result.error or "")
                                or "missing 1 required positional argument" in (tool_result.error or "")
                            )
                        ):
                            invalid_tool_arguments += 1
                        if name == "read_file" and isinstance(args.get("path"), str):
                            files_read_list.append(args["path"])
                        elif name in ("write_file", "patch_file"):
                            edit_calls_count += 1
                        elif name == "run_command":
                            test_command_calls_count += 1
                            cmd_str = str(args.get("command") or "")
                            if tool_result.success and recovery_controller._is_verification_command(cmd_str):
                                successful_verifications += 1
                        turn_entry["tool_calls"].append({
                            "name": name,
                            "target": str(args.get("path") or args.get("command") or ""),
                            "success": bool(tool_result.success),
                        })

                        self._emit(
                            "tool_result",
                            subtask.id,
                            {
                                "tool": name,
                                "success": tool_result.success,
                                "output": tool_result.output,
                                "error": tool_result.error,
                            },
                        )
                        obs = (
                            tool_result.output or "Success (no output)"
                            if tool_result.success
                            else f"Error: {tool_result.error}"
                        )
                        # For schema/argument failures, changing a large content
                        # string cannot count as progress. Normalize the bad
                        # payload so repeated missing-required-argument calls
                        # participate in cycle detection.
                        if (
                            not tool_result.success
                            and (tool_result.error or "").startswith("Invalid arguments for tool")
                        ):
                            fingerprint_facts = [name, "invalid_arguments", tool_result.error]
                        else:
                            fingerprint_facts = [name, args, tool_result.success, obs]
                        fingerprint_input = json.dumps(
                            fingerprint_facts,
                            sort_keys=True,
                            ensure_ascii=False,
                            default=str,
                        )
                        fingerprint = hashlib.sha256(fingerprint_input.encode("utf-8")).hexdigest()
                        cycle_length = self._repeated_action_cycle(action_history, fingerprint)
                        if cycle_length is not None:
                            err_msg = (
                                "Execution stalled: a tool-action cycle repeated without new "
                                f"evidence (cycle length {cycle_length})"
                            )
                            self._emit("subtask_stalled", subtask.id, {
                                "subtask_id": subtask.id,
                                "turn": turn,
                                "tool": name,
                                "fingerprint": fingerprint,
                                "cycle_length": cycle_length,
                            })
                            return SubtaskResult(
                                subtask_id=subtask.id,
                                success=False,
                                turns=turn,
                                tool_calls_count=tool_calls_count,
                                error=err_msg,
                                artifacts=list(subtask.affected_files),
                                metadata=_build_telemetry_meta({"stalled": True}),
                            )
                        messages.append({
                            "role": "tool",
                            "content": recovery_controller.annotate_observation(
                                prune_tool_output(str(obs)),
                                recovery_controller.record_tool_execution(
                                    turn=turn,
                                    tool_name=name,
                                    args=args,
                                    success=bool(tool_result.success),
                                    output=str(tool_result.output or ""),
                                    error=tool_result.error,
                                ),
                            ),
                            "tool_name": name,
                        })

            except Exception as exc:
                if "exceeds context budget" in str(exc):
                    context_overflows += 1
                stop = self.execution_control.check()
                if stop.stopped:
                    return SubtaskResult(
                        subtask_id=subtask.id, success=False, turns=turn,
                        tool_calls_count=tool_calls_count,
                        error=f"Execution paused: {stop.reason}",
                        metadata=_build_telemetry_meta({"paused": True, "stop_reason": stop.reason}),
                    )
                if deadline is not None and time.monotonic() >= deadline:
                    return SubtaskResult(
                        subtask_id=subtask.id, success=False, turns=turn,
                        tool_calls_count=tool_calls_count,
                        error="Attempt stopped: subtask time limit reached",
                        metadata=_build_telemetry_meta({"limit_reached": True}),
                    )
                err_msg = f"Workforce inference error: {exc}"
                self._emit(
                    "subtask_failed",
                    subtask.id,
                    {"subtask_id": subtask.id, "error": err_msg},
                )
                return SubtaskResult(
                    subtask_id=subtask.id,
                    success=False,
                    turns=0,
                    tool_calls_count=tool_calls_count,
                    error=err_msg,
                    artifacts=list(subtask.affected_files),
                    metadata=_build_telemetry_meta({"exception": str(exc)}),
                )

    def _execute_react(
        self,
        subtask: SubTask,
        tool_suite: WorkspaceToolSuite,
        feedback_context: Optional[str] = None,
    ) -> SubtaskResult:
        """Executes a SubTask through the legacy text ReAct loop using tool_suite.

        Serialized across threads via class-level _inference_lock to preserve RTX 4060 VRAM.
        """
        with self._inference_slot() as stop_reason:
            if stop_reason:
                return SubtaskResult(
                    subtask_id=subtask.id, success=False, turns=0,
                    error=f"Execution paused: {stop_reason}",
                    metadata={"paused": True, "stop_reason": stop_reason},
                )
            self._emit(
                "subtask_start",
                subtask.id,
                {"subtask_id": subtask.id, "title": subtask.title},
            )

            tool_definitions = tool_suite.get_tool_definitions()
            tool_names = [t["name"] for t in tool_definitions]

            system_prompt = (
                "You are an expert autonomous R&D coding workforce operating locally on an RTX 4060 GPU.\n"
                f"Available tools: {', '.join(tool_names)}\n"
                "Use the following ReAct format:\n"
                "Thought: Reason about the task and your next step\n"
                "Action: <tool_name>\n"
                "Action Input: <json_formatted_arguments>\n"
                "Observation: <tool_result>\n"
                "... (repeat Thought/Action/Action Input/Observation as needed)\n"
                "Thought: All requirements satisfied\n"
                "Final Answer: <concise summary of work completed>\n"
            )

            conversation_history = [
                f"SubTask ID: {subtask.id}\n"
                f"Title: {subtask.title}\n"
                f"Description: {subtask.description}\n"
                f"Input Files: {subtask.input_files}\n"
                f"Expected Outputs: {subtask.outputs}\n"
                f"Affected Files: {subtask.affected_files}\n"
                f"Success Criteria: {subtask.success_criteria}\n"
                f"Verification Command: {subtask.verification_command}\n"
            ]

            if feedback_context:
                conversation_history.append(
                    f"Previous attempt failed with error:\n{feedback_context}\n"
                    "Please analyze this feedback and implement corrections.\n"
                )
                self._emit(
                    "thought",
                    subtask.id,
                    {
                        "thought": f"Addressing feedback: {feedback_context}",
                        "turn": 0,
                        "feedback_context": feedback_context,
                    },
                )

            tool_calls_count = 0
            turn = 0
            action_history: List[str] = []

            while True:
                turn += 1
                stop = self.execution_control.check()
                if stop.stopped:
                    return SubtaskResult(
                        subtask_id=subtask.id, success=False, turns=turn - 1,
                        tool_calls_count=tool_calls_count,
                        error=f"Execution paused: {stop.reason}",
                        artifacts=list(subtask.affected_files),
                        metadata={"paused": True, "stop_reason": stop.reason},
                    )
                prompt = "\n".join(conversation_history)
                response_text = self._call_model(prompt=prompt, system_prompt=system_prompt)
                parsed = self._parse_action(response_text)

                if parsed.thought:
                    self._emit(
                        "thought",
                        subtask.id,
                        {"thought": parsed.thought, "turn": turn},
                    )

                if parsed.is_final:
                    self._emit(
                        "subtask_complete",
                        subtask.id,
                        {"subtask_id": subtask.id, "summary": parsed.summary},
                    )
                    return SubtaskResult(
                        subtask_id=subtask.id,
                        success=True,
                        turns=turn,
                        tool_calls_count=tool_calls_count,
                        summary=parsed.summary,
                        artifacts=list(subtask.affected_files),
                        metadata={"model": self.model_name},
                    )

                if parsed.error:
                    observation = f"Error: {parsed.error}. Please retry with valid JSON for Action Input."
                    conversation_history.append(f"Thought: {parsed.thought}")
                    conversation_history.append(f"Observation: {observation}")
                    continue

                # Tool execution step
                stop = self.execution_control.check()
                if stop.stopped:
                    return SubtaskResult(
                        subtask_id=subtask.id, success=False, turns=turn - 1,
                        tool_calls_count=tool_calls_count,
                        error=f"Execution paused: {stop.reason}",
                        artifacts=list(subtask.affected_files),
                        metadata={"paused": True, "stop_reason": stop.reason},
                    )
                tool_calls_count += 1
                self._emit(
                    "tool_call",
                    subtask.id,
                    {"tool": parsed.action, "args": parsed.args, "turn": turn},
                )

                try:
                    tool_result = tool_suite.execute(parsed.action, **parsed.args)
                except Exception as exc:
                    tool_result = ToolResult(
                        success=False,
                        output="",
                        error=f"Tool execution exception: {exc}",
                        metadata={"exception": str(exc)},
                    )

                self._emit(
                    "tool_result",
                    subtask.id,
                    {
                        "tool": parsed.action,
                        "success": tool_result.success,
                        "output": tool_result.output,
                        "error": tool_result.error,
                    },
                )

                if tool_result.success:
                    obs_str = tool_result.output or "Success (no output)"
                else:
                    obs_str = f"Error: {tool_result.error}"

                fingerprint_input = json.dumps(
                    [parsed.action, parsed.args, tool_result.success, obs_str],
                    sort_keys=True,
                    ensure_ascii=False,
                    default=str,
                )
                fingerprint = hashlib.sha256(fingerprint_input.encode("utf-8")).hexdigest()
                cycle_length = self._repeated_action_cycle(action_history, fingerprint)
                if cycle_length is not None:
                    err_msg = (
                        "Execution stalled: a tool-action cycle repeated without new "
                        f"evidence (cycle length {cycle_length})"
                    )
                    self._emit("subtask_stalled", subtask.id, {
                        "subtask_id": subtask.id,
                        "turn": turn,
                        "tool": parsed.action,
                        "fingerprint": fingerprint,
                        "cycle_length": cycle_length,
                    })
                    return SubtaskResult(
                        subtask_id=subtask.id,
                        success=False,
                        turns=turn,
                        tool_calls_count=tool_calls_count,
                        error=err_msg,
                        artifacts=list(subtask.affected_files),
                        metadata={"stalled": True},
                    )

                conversation_history.append(f"Thought: {parsed.thought}")
                conversation_history.append(f"Action: {parsed.action}")
                conversation_history.append(f"Action Input: {json.dumps(parsed.args)}")
                conversation_history.append(f"Observation: {obs_str}")


    # Legacy gauntlet compatibility methods
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> Generator[Tuple[str, str, Dict[str, Any]], None, None]:
        """Legacy generator yielding token chunks."""
        accumulated = ""
        mock_response = "```python\n# Autonomous worker output\npass\n```\n"
        words = mock_response.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            accumulated += chunk
            yield chunk, accumulated, {"done": False}

        yield "", accumulated, {
            "done": True,
            "eval_count": 420,
            "eval_duration_s": 0.5,
            "prompt_eval_count": 150,
            "total_duration_s": 0.6,
        }

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """Legacy synchronous generation."""
        for _, accumulated, meta in self.generate_stream(prompt, system_prompt, temperature):
            if meta.get("done"):
                return accumulated
        return "```python\n# Autonomous worker output\npass\n```\n"
