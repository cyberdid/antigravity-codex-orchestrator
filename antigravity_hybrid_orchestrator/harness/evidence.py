"""Rollout records and their rendering into proposer evidence.

The proposer sees what the agent itself could see: its own turns, tool calls and
errors, the visible test output and the final diff. Hidden holdout results are
kept on the record for the verifier gate and never rendered into evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping


@dataclass
class Rollout:
    label: str
    task_id: str
    harness_fingerprint: str
    visible_passed: bool
    holdout_passed: bool
    worker_success: bool
    wall_s: float
    tokens: int
    truncations: int
    bad_tool_calls: int
    context_overflows: int
    turns: int
    error: str = ""
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    diff: str = ""
    visible_output: str = ""

    @property
    def gate_accepted(self) -> bool:
        """Trusted verification: visible and hidden holdout tests pass, context intact."""
        return self.visible_passed and self.holdout_passed and self.context_overflows == 0

    @classmethod
    def from_benchmark(cls, label: str, harness_fingerprint: str, record: Mapping[str, Any]) -> "Rollout":
        meta = record.get("metadata", {}) or {}
        return cls(
            label=label,
            task_id=str(record.get("task_id", "")),
            harness_fingerprint=harness_fingerprint,
            visible_passed=bool(meta.get("visible_tests_passed")),
            holdout_passed=bool(meta.get("holdout_tests_passed")),
            worker_success=bool(record.get("worker_reported_success")),
            wall_s=float(record.get("wall_clock_s", 0.0)),
            tokens=int(meta.get("input_tokens", 0)) + int(meta.get("output_tokens", 0)),
            truncations=int(meta.get("truncated_generations", 0)),
            bad_tool_calls=int(meta.get("invalid_tool_arguments", 0)),
            context_overflows=int(meta.get("context_overflows", 0)),
            turns=int(record.get("turns", 0)),
            error=str(record.get("error") or ""),
            trajectory=list(meta.get("turn_trajectory", [])),
            diff=str(meta.get("workspace_diff", "")),
            visible_output=str(meta.get("visible_test_output", "")),
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "label": self.label, "task_id": self.task_id, "harness": self.harness_fingerprint[:12],
            "gate_accepted": self.gate_accepted, "visible_passed": self.visible_passed,
            "holdout_passed": self.holdout_passed, "wall_s": self.wall_s, "tokens": self.tokens,
            "truncations": self.truncations, "bad_tool_calls": self.bad_tool_calls,
            "turns": self.turns, "error": self.error,
        }


def _turn_line(entry: Mapping[str, Any]) -> str:
    head = f"[T{entry.get('turn')}]"
    if entry.get("attempt"):
        head = f"[A{entry['attempt']} T{entry.get('turn')}]"
    calls = entry.get("tool_calls") or []
    flags = []
    if entry.get("truncated"):
        flags.append(f"TRUNCATED at {entry.get('num_predict', '?')} tokens")
    if entry.get("discarded_tool_calls"):
        flags.append("discarded truncated call(s): " + ", ".join(entry["discarded_tool_calls"]))
    if entry.get("recovery_triggered"):
        flags.append(f"nudge: {entry.get('recovery_reason')}")
    if not calls:
        action = "no tool call (text)"
    else:
        rendered = []
        for call in calls:
            item = f"{call.get('name')} {str(call.get('target', ''))[:80]}"
            item += " ok" if call.get("success") else f" ERROR: {str(call.get('error', ''))[-160:]}"
            rendered.append(item)
        action = "; ".join(rendered)
    tokens = f"{entry.get('output_tokens', 0)} out tok, {entry.get('eval_duration_s', 0)} s"
    return f"{head} {action} ({tokens})" + (f" [{'; '.join(flags)}]" if flags else "")


def render_evidence(rollout: Rollout, *, max_chars: int = 6000) -> str:
    header = (
        f"## Rollout {rollout.label} (task {rollout.task_id})\n"
        f"outcome: worker {'finished' if rollout.worker_success else 'did not finish'}"
        f"{' — ' + rollout.error if rollout.error else ''}; visible tests "
        f"{'PASS' if rollout.visible_passed else 'FAIL'}; {rollout.turns} turns, "
        f"{rollout.wall_s:.0f} s, {rollout.tokens} tokens, {rollout.truncations} truncated turns\n"
    )
    turns = "\n".join(_turn_line(e) for e in rollout.trajectory)
    tests = rollout.visible_output.strip()[-1200:]
    diff = rollout.diff.strip()[:2000]
    text = (
        f"{header}\n### Turns\n{turns}\n\n### Visible test output (tail)\n{tests or '(none)'}\n\n"
        f"### Final workspace diff (head)\n{diff or '(no change)'}\n"
    )
    return text if len(text) <= max_chars else text[: max_chars - 20] + "\n[...truncated]\n"
