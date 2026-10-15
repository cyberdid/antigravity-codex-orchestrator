"""Cooperative cancellation and optional run-wide execution budgets."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable, Optional


@dataclass(frozen=True)
class ExecutionBudget:
    """Optional run limits and inactivity watchdog settings.

    ``None`` means no run-wide cap. A zero inactivity timeout disables that
    watchdog; it does not disable explicit cancellation or no-progress checks.
    """

    max_runtime_s: Optional[float] = None
    max_remote_tokens: Optional[int] = None
    model_idle_timeout_s: float = 900.0
    tool_idle_timeout_s: float = 300.0

    def __post_init__(self) -> None:
        if self.max_runtime_s is not None and self.max_runtime_s <= 0:
            raise ValueError("max_runtime_s must be positive or None")
        if self.max_remote_tokens is not None and self.max_remote_tokens <= 0:
            raise ValueError("max_remote_tokens must be positive or None")
        if self.model_idle_timeout_s < 0 or self.tool_idle_timeout_s < 0:
            raise ValueError("inactivity timeouts must be non-negative")


@dataclass(frozen=True)
class StopDecision:
    """A cooperative stop request and its machine-readable reason."""

    stopped: bool
    reason: str = ""


class ExecutionControl:
    """Thread-safe run control checked between model and tool operations."""

    def __init__(
        self,
        budget: Optional[ExecutionBudget] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.budget = budget or ExecutionBudget()
        self._clock = clock
        self._started_at = clock()
        self._remote_tokens = 0
        self._cancel_reason: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def elapsed_s(self) -> float:
        return max(0.0, self._clock() - self._started_at)

    @property
    def remote_tokens(self) -> int:
        with self._lock:
            return self._remote_tokens

    def record_remote_tokens(self, count: int) -> None:
        with self._lock:
            self._remote_tokens += max(0, int(count))

    def restore_usage(self, *, remote_tokens: int = 0, elapsed_s: float = 0.0) -> None:
        """Restore run-wide budget consumption from a paused run snapshot."""
        with self._lock:
            self._remote_tokens = max(0, int(remote_tokens))
            self._started_at = self._clock() - max(0.0, float(elapsed_s))

    def cancel(self, reason: str = "operator_cancelled") -> None:
        with self._lock:
            if self._cancel_reason is None:
                self._cancel_reason = reason or "operator_cancelled"

    def check(self) -> StopDecision:
        with self._lock:
            cancel_reason = self._cancel_reason
            remote_tokens = self._remote_tokens

        if cancel_reason:
            return StopDecision(True, cancel_reason)
        if (
            self.budget.max_runtime_s is not None
            and self.elapsed_s >= self.budget.max_runtime_s
        ):
            return StopDecision(True, "runtime_budget_exceeded")
        if (
            self.budget.max_remote_tokens is not None
            and remote_tokens >= self.budget.max_remote_tokens
        ):
            return StopDecision(True, "remote_token_budget_exceeded")
        return StopDecision(False)
