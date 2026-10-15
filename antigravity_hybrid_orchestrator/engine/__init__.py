"""Autonomous Feedback and Self-Correction Engine for R&D Task Execution."""

from antigravity_hybrid_orchestrator.engine.verifier import (
    VerificationResult,
    VerificationRunner,
)
from antigravity_hybrid_orchestrator.engine.loop import (
    SelfCorrectionLoop,
    SubtaskExecutionState,
    DAGExecutionResult,
)

__all__ = [
    "VerificationResult",
    "VerificationRunner",
    "SelfCorrectionLoop",
    "SubtaskExecutionState",
    "DAGExecutionResult",
]
