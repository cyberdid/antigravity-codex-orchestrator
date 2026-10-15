"""On-Device Workforce package for adversarial code remediation and evaluation via Gemma 4."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PatchResult:
    """Remediation patch produced by an on-device builder agent."""
    builder_name: str
    patch_code: str
    eval_tokens: int = 0
    eval_duration_s: float = 0.0


class CriticResult:
    """Anonymized adversarial evaluation and winning patch selection from Blind Critic."""
    winner: str
    alpha_score: float
    beta_score: float
    critique: str
    winning_builder: str
    eval_tokens: int
    eval_duration_s: float

    def __init__(
        self,
        winner: str = "",
        *args,
        alpha_score: float = 0.0,
        beta_score: float = 0.0,
        critique: str = "",
        winning_builder: str = "",
        eval_tokens: int = 0,
        eval_duration_s: float = 0.0,
        **kwargs,
    ) -> None:
        self.winner = winner

        if len(args) == 3:
            # Called as (winner, alpha_score, beta_score, critique)
            self.alpha_score = float(args[0])
            self.beta_score = float(args[1])
            self.critique = str(args[2])
            self.winning_builder = winning_builder or kwargs.get("winning_builder", "")
            self.eval_tokens = int(eval_tokens or kwargs.get("eval_tokens", 0))
            self.eval_duration_s = float(eval_duration_s or kwargs.get("eval_duration_s", 0.0))
        elif len(args) >= 6:
            # Called as (winner, winning_builder, alpha_score, beta_score, critique, eval_tokens, eval_duration_s)
            self.winning_builder = str(args[0])
            self.alpha_score = float(args[1])
            self.beta_score = float(args[2])
            self.critique = str(args[3])
            self.eval_tokens = int(args[4])
            self.eval_duration_s = float(args[5])
        else:
            self.winning_builder = winning_builder or kwargs.get("winning_builder", "")
            self.alpha_score = float(alpha_score if alpha_score != 0.0 else kwargs.get("alpha_score", 0.0))
            self.beta_score = float(beta_score if beta_score != 0.0 else kwargs.get("beta_score", 0.0))
            self.critique = critique or kwargs.get("critique", "")
            self.eval_tokens = int(eval_tokens or kwargs.get("eval_tokens", 0))
            self.eval_duration_s = float(eval_duration_s or kwargs.get("eval_duration_s", 0.0))

    def __repr__(self) -> str:
        return (
            f"CriticResult(winner={self.winner!r}, "
            f"winning_builder={self.winning_builder!r}, "
            f"alpha_score={self.alpha_score}, "
            f"beta_score={self.beta_score}, "
            f"critique={self.critique!r}, "
            f"eval_tokens={self.eval_tokens}, "
            f"eval_duration_s={self.eval_duration_s})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CriticResult):
            return False
        return (
            self.winner == other.winner
            and self.winning_builder == other.winning_builder
            and self.alpha_score == other.alpha_score
            and self.beta_score == other.beta_score
            and self.critique == other.critique
            and self.eval_tokens == other.eval_tokens
            and self.eval_duration_s == other.eval_duration_s
        )


from antigravity_hybrid_orchestrator.workforce.ollama_client import (
    WorkforceClient,
    OllamaConnectionError,
)
from antigravity_hybrid_orchestrator.workforce.builder_a import BuilderAAgent
from antigravity_hybrid_orchestrator.workforce.builder_b import BuilderBAgent
from antigravity_hybrid_orchestrator.workforce.blind_critic import BlindCriticAgent
from antigravity_hybrid_orchestrator.workforce.mock_workforce import (
    MockWorkforceClient,
    MockBuilderAAgent,
    MockBuilderBAgent,
    MockBlindCriticAgent,
)
from antigravity_hybrid_orchestrator.workforce.tools import (
    ToolResult,
    WorkspaceToolSuite,
)
from antigravity_hybrid_orchestrator.workforce.worker import (
    AutonomousWorker,
    ParsedAction,
    SubtaskResult,
)
from antigravity_hybrid_orchestrator.workforce.mock_worker import (
    MockAutonomousWorker,
    MockLocalWorkforceClient,
)

__all__ = [
    "PatchResult",
    "CriticResult",
    "WorkforceClient",
    "OllamaConnectionError",
    "BuilderAAgent",
    "BuilderBAgent",
    "BlindCriticAgent",
    "MockWorkforceClient",
    "MockBuilderAAgent",
    "MockBuilderBAgent",
    "MockBlindCriticAgent",
    "ToolResult",
    "WorkspaceToolSuite",
    "AutonomousWorker",
    "ParsedAction",
    "SubtaskResult",
    "MockAutonomousWorker",
    "MockLocalWorkforceClient",
]

