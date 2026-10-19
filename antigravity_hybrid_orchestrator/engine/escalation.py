"""Attempt-level escalation steps consulted by SelfCorrectionLoop.

The per-turn step (a larger generation limit for a truncated tool call) lives in
the worker. This module holds the steps that change between attempts: thinking on
retry and, only when the owner enabled it for the run, cloud guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    from antigravity_hybrid_orchestrator.engine.cloud_assist import CloudAssist


@dataclass
class EscalationLadder:
    think_on_retry: bool = False
    cloud_assist: Optional["CloudAssist"] = None

    def think_for_attempt(self, attempt: int) -> Optional[bool]:
        """``True`` for retries (0-based ``attempt`` >= 1) when enabled, else no override."""
        return True if self.think_on_retry and attempt >= 1 else None

    def assist_due(self, attempt: int) -> bool:
        """Cloud guidance comes once, right after the enabled local steps are exhausted."""
        return self.cloud_assist is not None and attempt == (2 if self.think_on_retry else 1)
