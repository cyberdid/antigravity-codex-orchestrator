"""Privacy engine for zero-knowledge AST-level code stripping and pre-flight verification."""

from antigravity_hybrid_orchestrator.privacy.ast_stripper import (
    ASTPrivacyStripper,
    strip_code_for_cloud,
)
from antigravity_hybrid_orchestrator.privacy.verifier import (
    SecurityInvariantViolationError,
    verify_cloud_payload,
)

__all__ = [
    "ASTPrivacyStripper",
    "strip_code_for_cloud",
    "SecurityInvariantViolationError",
    "verify_cloud_payload",
]
