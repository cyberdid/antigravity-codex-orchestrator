"""Compatibility re-exports for MockAutonomousWorker and MockLocalWorkforceClient."""

from antigravity_hybrid_orchestrator.workforce.mock_worker import (
    MockAutonomousWorker,
    MockLocalWorkforceClient,
    MockWorkforceClient,
)

__all__ = [
    "MockAutonomousWorker",
    "MockLocalWorkforceClient",
    "MockWorkforceClient",
]
