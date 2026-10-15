"""Cloud Architect package for AST-level security strategy generation via Gemini."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class StrategyResult:
    """Remediation strategy and threat model produced by Cloud Architect."""
    threat_model: str = ""
    invariants: List[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    raw_strategy: str = ""


from antigravity_hybrid_orchestrator.architect.client import (
    CloudArchitectClient,
    CloudArchitectError,
)
from antigravity_hybrid_orchestrator.architect.mock_client import (
    MockCloudArchitectClient,
)

__all__ = [
    "StrategyResult",
    "CloudArchitectClient",
    "MockCloudArchitectClient",
    "CloudArchitectError",
]
