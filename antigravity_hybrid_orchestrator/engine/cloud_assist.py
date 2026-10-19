"""Owner-approved cloud guidance for a subtask the local worker keeps failing.

Built only by the coordinator when the owner passed ``--cloud-assist spec`` on the
current command line. Every outcome goes through ``audit``: requests and responses
verbatim (with the payload's SHA-256), blocked payloads, backend failures and
skipped calls. A failure here never fails the run; the ladder just continues.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set

from antigravity_hybrid_orchestrator.engine.control import ExecutionControl
from antigravity_hybrid_orchestrator.planner.dag import SubTask
from antigravity_hybrid_orchestrator.privacy.assist_payload import build_assist_payload, verify_assist_payload
from antigravity_hybrid_orchestrator.privacy.verifier import SecurityInvariantViolationError

AuditFn = Callable[[str, str, Dict[str, Any]], None]


class CloudAssist:
    def __init__(
        self,
        client: Any,
        workspace: Path,
        *,
        max_calls: int = 3,
        audit: AuditFn,
        execution_control: ExecutionControl,
    ) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be at least 1")
        self.client = client
        self.workspace = Path(workspace)
        self.max_calls = max_calls
        self.audit = audit
        self.execution_control = execution_control
        self.calls_made = 0
        self.tokens_used = 0
        self._assisted: Set[str] = set()

    def request(self, subtask: SubTask, failure_excerpt: str) -> Optional[str]:
        if subtask.id in self._assisted:
            return self._skip(subtask, "subtask_limit")
        if self.calls_made >= self.max_calls:
            return self._skip(subtask, "run_limit")
        if self.execution_control.check().stopped:
            return self._skip(subtask, "remote_budget")

        payload = build_assist_payload(self.workspace, subtask, failure_excerpt)
        try:
            verify_assist_payload(payload)
        except SecurityInvariantViolationError as exc:
            self.audit("cloud_assist_blocked", subtask.id, {"reason": str(exc)})
            return None

        self._assisted.add(subtask.id)
        self.calls_made += 1
        self.audit("cloud_assist_request", subtask.id, {
            "payload": payload,
            "payload_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "backend": getattr(self.client, "backend", "mock"),
        })
        try:
            response, input_tokens, output_tokens = self.client.assist_subtask(payload)
        except Exception as exc:  # backend CLI errors, timeouts, parse failures
            self.audit("cloud_assist_failed", subtask.id, {"error": str(exc)})
            return None
        tokens = int(input_tokens) + int(output_tokens)
        self.tokens_used += tokens
        self.execution_control.record_remote_tokens(tokens)
        self.audit("cloud_assist_response", subtask.id, {
            "response": response,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
        })
        return response or None

    def _skip(self, subtask: SubTask, reason: str) -> None:
        self.audit("cloud_assist_skipped", subtask.id, {"reason": reason})
        return None
