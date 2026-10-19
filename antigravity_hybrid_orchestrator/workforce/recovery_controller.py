"""Model-agnostic Recovery Controller for autonomous workforce tool-calling loops.

Implements GEMMA-ORION Phase 2B (`P0` baseline vs `P1` RecoveryController) and
provides a structured, declarative policy surface (`RecoveryPolicyConfig`) ready
for future Harness² (`OrchestratorSubstrate`) adaptation.

Safety contract:
- Never modifies workspace files or executes tools on its own.
- Never bypasses RIE, verification commands, or hidden tests.
- Only manages conversation state, tool-observation stage annotations, and
  bounded, evidence-preserving recovery prompts within explicit attempt limits.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


def _tail(text: str, limit: int) -> str:
    """Keep the end of tool output, where test runners print the assertion."""
    text = (text or "").strip()
    return text[-limit:] if limit > 0 else ""


def active_recovery_policy(override: Optional[str] = None) -> str:
    """Return canonical recovery policy ID: 'P0', 'P1_REACTIVE', 'P1_STAGE', or 'P1'."""
    raw = (override or os.environ.get("OLLAMA_RECOVERY_POLICY", "P0")).strip().upper()
    if raw in ("P1_REACTIVE", "REACTIVE"):
        return "P1_REACTIVE"
    if raw in ("P1_STAGE", "P1_HINTS", "STAGE", "HINTS"):
        return "P1_STAGE"
    if raw in ("P1", "P1_FULL", "RECOVERY", "ENHANCED", "TRUE", "1"):
        return "P1"
    return "P0"


@dataclass
class RecoveryPolicyConfig:
    """Declarative configuration for RecoveryController (Harness² c4-compatible)."""

    policy_id: str = "P1"
    max_recovery_attempts: int = 3
    max_repeated_file_reads: int = 1
    reactive_recovery: bool = True
    annotate_tool_observations: bool = True
    catch_premature_stop_before_verification: bool = True
    max_error_snippet_chars: int = 320

    @classmethod
    def for_policy(cls, policy: str) -> "RecoveryPolicyConfig":
        norm = active_recovery_policy(policy)
        if norm == "P0":
            return cls(
                policy_id="P0",
                reactive_recovery=False,
                annotate_tool_observations=False,
                catch_premature_stop_before_verification=False,
            )
        if norm == "P1_REACTIVE":
            return cls(
                policy_id="P1_REACTIVE",
                reactive_recovery=True,
                annotate_tool_observations=False,
                catch_premature_stop_before_verification=True,
            )
        if norm == "P1_STAGE":
            return cls(
                policy_id="P1_STAGE",
                reactive_recovery=False,
                annotate_tool_observations=True,
                catch_premature_stop_before_verification=False,
            )
        return cls(
            policy_id="P1",
            reactive_recovery=True,
            annotate_tool_observations=True,
            catch_premature_stop_before_verification=True,
        )


@dataclass
class RecoveryController:
    """Tracks tool-calling progress and synthesizes bounded recovery directives."""

    subtask_id: str
    input_files: List[str] = field(default_factory=list)
    affected_files: List[str] = field(default_factory=list)
    verification_command: str = ""
    policy: str = "P0"
    config: Optional[RecoveryPolicyConfig] = None
    # Controller-owned run of the verification command -> (passed, output). When set,
    # verification state never depends on how the model spelled its own command.
    verify_fn: Optional[Callable[[], Tuple[bool, str]]] = field(default=None, repr=False)

    # Runtime state tracked across turns
    all_read_files: List[str] = field(default_factory=list)
    clean_read_files: List[str] = field(default_factory=list)
    modified_since_read_files: List[str] = field(default_factory=list)
    repeated_reads_count: int = 0
    edits_applied: List[str] = field(default_factory=list)
    consecutive_failed_edits: int = 0
    last_edit_error: str = ""
    verification_runs: int = 0
    successful_verifications: int = 0
    verification_passed: bool = False
    last_verification_error: str = ""
    recovery_attempts: int = 0
    stage_hints_emitted: int = 0
    consecutive_non_tool_turns: int = 0
    recovery_events: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.policy = active_recovery_policy(self.policy)
        if self.config is None:
            self.config = RecoveryPolicyConfig.for_policy(self.policy)

    @property
    def is_reactive(self) -> bool:
        return bool(self.config and self.config.reactive_recovery)

    @property
    def is_stage_hints(self) -> bool:
        return bool(self.config and self.config.annotate_tool_observations)

    def _is_verification_command(self, cmd: str) -> bool:
        """Return True iff `cmd` strictly executes the subtask's verification command.

        When `self.verification_command` is set, checks exact normalized match first
        so legitimate compound verification commands (e.g. `cd . && python3 -m unittest -q`)
        are accepted while any appended `| tail`, `|| true`, partial test runs, or
        `python3 -c "assert True"` are rejected.
        """
        norm_cmd = " ".join(cmd.strip().split())
        if not norm_cmd:
            return False
        norm_verif = " ".join((self.verification_command or "").strip().split())
        if norm_verif:
            return norm_cmd == norm_verif
        # Fallback only when SubTask.verification_command was left empty:
        if any(op in norm_cmd for op in ("|", "&&")):
            return False
        return bool(re.search(r"\b(pytest|unittest|assert)\b", norm_cmd))

    def unread_input_files(self) -> List[str]:
        read_set = set(self.all_read_files)
        return [f for f in self.input_files if f not in read_set]

    def recommended_next_action(self) -> str:
        """Determine the concrete single next action from current evidence state."""
        unread = self.unread_input_files()
        if unread:
            return (
                f"Call `read_file` on `{unread[0]}` to inspect the remaining input file."
            )
        if not self.edits_applied:
            target = self.affected_files[0] if self.affected_files else (
                self.input_files[0] if self.input_files else "the target file"
            )
            return (
                f"All input files are already in context ({self.clean_read_files}). "
                f"Do NOT call `read_file` again. Call `patch_file` (or `write_file`) on `{target}` "
                "with ONE minimal fix."
            )
        if self.verification_command and not self.verification_passed:
            if self.last_verification_error:
                target = self.affected_files[0] if self.affected_files else "the affected file"
                return (
                    f"Last verification failed ({_tail(self.last_verification_error, 160)}). "
                    f"Apply a targeted fix on `{target}` via `patch_file` and re-run "
                    f"`run_command` with `{self.verification_command}`."
                )
            return (
                f"Edits applied to {self.edits_applied}. Now call `run_command` with "
                f"`{self.verification_command}` to verify."
            )
        if self.verification_passed:
            return (
                "Verification command PASSED. Do NOT call `read_file` or any other tool. "
                "Reply immediately with a concise 2-sentence final summary."
            )
        return "Emit ONE concise tool call or a final 2-sentence summary if complete."

    def record_tool_execution(
        self,
        *,
        turn: int,
        tool_name: str,
        args: Dict[str, Any],
        success: bool,
        output: str,
        error: Optional[str],
    ) -> Optional[str]:
        """Update controller state after a tool call and return an optional stage note."""
        self.consecutive_non_tool_turns = 0
        err_text = str(error or "").strip()

        if tool_name == "read_file":
            path = str(args.get("path") or "").strip()
            if path:
                is_repeated_clean_read = path in self.clean_read_files
                self.all_read_files.append(path)
                if success:
                    if path not in self.clean_read_files:
                        self.clean_read_files.append(path)
                    if path in self.modified_since_read_files:
                        self.modified_since_read_files.remove(path)
                if is_repeated_clean_read:
                    self.repeated_reads_count += 1
                    if self.is_reactive:
                        next_act = self.recommended_next_action()
                        self._log_event(
                            turn=turn,
                            reason="repeated_file_read",
                            detail=f"File '{path}' was re-read without intervening edits.",
                            action_recommended=next_act,
                        )
                        return (
                            f"[RECOVERY CONTROLLER: `{path}` was already read and unchanged. "
                            f"Use the file content already in context. Next step: {next_act}]"
                        )

        elif tool_name in ("patch_file", "write_file"):
            path = str(args.get("path") or "").strip()
            if success:
                self.consecutive_failed_edits = 0
                self.last_edit_error = ""
                if path:
                    if path not in self.edits_applied:
                        self.edits_applied.append(path)
                    # File was modified on disk; invalidate clean_read_files so a subsequent
                    # post-edit read_file is NOT misclassified as an unchanged repeated read.
                    if path in self.clean_read_files:
                        self.clean_read_files.remove(path)
                    if path not in self.modified_since_read_files:
                        self.modified_since_read_files.append(path)
                # Once edited, verification must be (re-)run
                self.verification_passed = False
            else:
                self.consecutive_failed_edits += 1
                self.last_edit_error = err_text[: self.config.max_error_snippet_chars]
                if self.is_reactive:
                    next_act = (
                        f"Check exact lines from `{path}` already in context and call "
                        "`patch_file` with exact `target_content` and `replacement_content` "
                        "(or `write_file` with `path` first, then `content`)."
                    )
                    self._log_event(
                        turn=turn,
                        reason="failed_edit_tool_call",
                        detail=self.last_edit_error,
                        action_recommended=next_act,
                    )
                    return f"[RECOVERY CONTROLLER: Edit failed ({self.last_edit_error}). {next_act}]"

        elif tool_name == "run_command":
            cmd = str(args.get("command") or "").strip()
            if self._is_verification_command(cmd):
                # The model ran the exact verification command; its result is authoritative.
                self._record_verification(turn, cmd, success, err_text or str(output or ""))
            elif self.verify_fn is not None:
                # Any other command (`-q test_x.py`, `| tail`, `assert True`) is only a
                # signal that the model is checking its work: run the real command instead.
                try:
                    passed, verify_output = self.verify_fn()
                except Exception as exc:
                    passed, verify_output = False, f"Trusted verification error: {exc}"
                self._record_verification(turn, self.verification_command, bool(passed), str(verify_output or ""))

        if self.is_stage_hints:
            self.stage_hints_emitted += 1
            return f"[STAGE STATE: {self.recommended_next_action()}]"
        return None

    def _record_verification(self, turn: int, cmd: str, passed: bool, output: str) -> None:
        self.verification_runs += 1
        if passed:
            self.successful_verifications += 1
            self.verification_passed = True
            self.last_verification_error = ""
            return
        self.verification_passed = False
        self.last_verification_error = _tail(output, self.config.max_error_snippet_chars)
        if self.is_reactive:
            self._log_event(
                turn=turn,
                reason="failed_verification_command",
                detail=f"Command '{cmd}' failed: {_tail(self.last_verification_error, 160)}",
                action_recommended=self.recommended_next_action(),
            )

    def annotate_observation(self, obs: str, stage_note: Optional[str]) -> str:
        """Append stage or recovery note to pruned tool observation when active."""
        if not stage_note:
            return obs
        return f"{obs}\n\n{stage_note}"

    def handle_non_tool_turn(
        self,
        *,
        turn: int,
        content: str,
        eval_tokens: int,
        is_truncated: bool,
        active_num_predict: int,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Handle a turn that produced no tool calls.

        Returns:
            (should_continue_loop, user_recovery_prompt, terminal_error)
        """
        has_Empty_or_truncated = is_truncated or (not content.strip() and eval_tokens > 0)

        if not self.is_reactive:
            # P0 / P1_STAGE non-tool-turn baseline behavior (exact match with Phase 1 / Phase 2A G0:
            # consecutive_non_tool_turns resets to 0 after any valid tool-call turn).
            if has_Empty_or_truncated:
                if self.consecutive_non_tool_turns < 2:
                    self.consecutive_non_tool_turns += 1
                    self.recovery_attempts += 1
                    nudge_reason = (
                        f"Your previous response hit the num_predict ({active_num_predict} tokens) "
                        "generation limit and was truncated before completing a tool call or final answer."
                        if is_truncated
                        else "Your previous response produced no text and no valid tool call "
                             "(it may have been truncated or malformed)."
                    )
                    prompt = (
                        f"{nudge_reason} Keep internal reasoning concise and emit ONE concise tool call "
                        "at a time (for write_file, always pass 'path' first before 'content'), "
                        "or reply with a short text summary if verification has already passed."
                    )
                    self._log_event(
                        turn=turn,
                        reason="p0_truncated_or_empty_nudge",
                        detail=nudge_reason,
                        action_recommended="generic_nudge",
                    )
                    return True, prompt, None
                err_msg = (
                    f"Execution failed: model output truncated at num_predict={active_num_predict} "
                    "without completing a valid tool call or final response"
                    if is_truncated
                    else "Execution failed: model repeatedly produced empty turns without tool calls"
                )
                return False, None, err_msg
            return False, None, None

        # P1 Enhanced RecoveryController behavior
        if has_Empty_or_truncated:
            self.consecutive_non_tool_turns += 1
            reason = "truncated_generation" if is_truncated else "empty_generation_no_tool_call"
            if self.recovery_attempts >= self.config.max_recovery_attempts:
                err_msg = (
                    f"Execution failed: RecoveryController limit ({self.config.max_recovery_attempts}) "
                    f"reached after {reason} at num_predict={active_num_predict}"
                )
                self._log_event(
                    turn=turn,
                    reason="recovery_limit_exceeded",
                    detail=err_msg,
                    action_recommended="abort_subtask",
                )
                return False, None, err_msg

            self.recovery_attempts += 1
            next_act = self.recommended_next_action()
            prompt = self._build_p1_recovery_prompt(
                reason=reason,
                active_num_predict=active_num_predict,
                next_action=next_act,
            )
            self._log_event(
                turn=turn,
                reason=reason,
                detail=f"eval_tokens={eval_tokens}, num_predict={active_num_predict}",
                action_recommended=next_act,
            )
            return True, prompt, None

        # Non-truncated text turn with no tool calls: check if verification was skipped
        needs_work_or_verify = bool(self.verification_command) and not self.verification_passed
        if (
            self.config.catch_premature_stop_before_verification
            and needs_work_or_verify
            and self.recovery_attempts < self.config.max_recovery_attempts
        ):
            self.recovery_attempts += 1
            self.consecutive_non_tool_turns += 1
            next_act = self.recommended_next_action()
            prompt = self._build_p1_recovery_prompt(
                reason="missing_tool_call_before_verification",
                active_num_predict=active_num_predict,
                next_action=next_act,
            )
            self._log_event(
                turn=turn,
                reason="missing_tool_call_before_verification",
                detail="Model emitted text without tool call before running verification command.",
                action_recommended=next_act,
            )
            return True, prompt, None

        return False, None, None

    def _build_p1_recovery_prompt(
        self,
        *,
        reason: str,
        active_num_predict: int,
        next_action: str,
    ) -> str:
        read_str = ", ".join(self.clean_read_files) if self.clean_read_files else "none"
        edits_str = ", ".join(self.edits_applied) if self.edits_applied else "none"
        verify_str = "PASSED" if self.verification_passed else (
            f"FAILED ({_tail(self.last_verification_error, 120)})"
            if self.last_verification_error
            else "NOT_RUN"
        )
        header = (
            f"[RECOVERY CONTROLLER ({self.recovery_attempts}/{self.config.max_recovery_attempts}) — {reason}]\n"
            f"- Files already in context (do NOT re-read): {read_str}\n"
            f"- Files edited so far: {edits_str}\n"
            f"- Verification status: {verify_str}\n"
            f"- Required next action: {next_action}\n"
            "Do NOT output prose analysis or repeat file contents. Emit ONE valid tool call now "
            "(or a 2-sentence final summary if Verification status is PASSED)."
        )
        return header

    def _log_event(
        self,
        *,
        turn: int,
        reason: str,
        detail: str,
        action_recommended: str,
    ) -> None:
        self.recovery_events.append({
            "turn": turn,
            "attempt": self.recovery_attempts,
            "reason": reason,
            "detail": detail,
            "action_recommended": action_recommended,
            "files_in_context": list(self.clean_read_files),
            "edits_applied": list(self.edits_applied),
            "verification_passed": self.verification_passed,
        })

    def telemetry_summary(self) -> Dict[str, Any]:
        return {
            "recovery_policy": self.policy,
            "reactive_recovery_enabled": self.is_reactive,
            "stage_hints_enabled": self.is_stage_hints,
            "recovery_attempts": self.recovery_attempts,
            "stage_hints_emitted": self.stage_hints_emitted,
            "repeated_unchanged_reads": self.repeated_reads_count,
            "recovery_events": list(self.recovery_events),
        }
