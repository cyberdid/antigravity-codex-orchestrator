"""Blind Critic agent: Anonymized adversarial evaluator and patch selector."""

import random
import re
from typing import Callable, Optional, Tuple, Union

from antigravity_hybrid_orchestrator.architect import StrategyResult
from antigravity_hybrid_orchestrator.workforce import CriticResult, PatchResult
from antigravity_hybrid_orchestrator.workforce.ollama_client import WorkforceClient


def anonymize_code(code: str) -> str:
    """Strip builder self-identification comments and labels from candidate code."""
    cleaned = re.sub(r"#\s*Builder\s*[AB].*?\n", "\n", code, flags=re.IGNORECASE)
    cleaned = re.sub(r"#\s*Surgical\s+minimal\s+fix.*?\n", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"#\s*Defense-in-depth\s+fix.*?\n", "\n", cleaned, flags=re.IGNORECASE)
    return cleaned.strip() + "\n"


class BlindCriticAgent:
    """Impartial adversarial evaluator for candidate patches.

    Anonymizes and shuffles candidates into Candidate Alpha vs Candidate Beta,
    evaluating them across vulnerability eradication, contract preservation,
    robustness, and maintainability without builder identity bias.
    """

    def __init__(self, client: Optional[WorkforceClient] = None) -> None:
        self.client = client or WorkforceClient()

    def _build_prompt(
        self,
        candidate_alpha: str,
        candidate_beta: str,
        strategy: StrategyResult,
    ) -> str:
        invariants_bullet = "\n".join(f"- {inv}" for inv in strategy.invariants)
        return (
            "You are an impartial Chief Security Critic.\n"
            "You are evaluating two anonymized candidate security patches: Candidate Alpha and Candidate Beta.\n\n"
            f"Threat Model: {strategy.threat_model}\n"
            f"Security Invariants:\n{invariants_bullet}\n\n"
            "--- CANDIDATE ALPHA ---\n"
            "```python\n"
            f"{candidate_alpha.strip()}\n"
            "```\n\n"
            "--- CANDIDATE BETA ---\n"
            "```python\n"
            f"{candidate_beta.strip()}\n"
            "```\n\n"
            "Evaluate both candidates across these four dimensions (1 to 10 each):\n"
            "1. Vulnerability Eradication (Does it completely eradicate the security flaw?)\n"
            "2. Regression Risk & Contract Preservation (Does it preserve function signatures and behaviors?)\n"
            "3. Robustness & Edge Handling (Does it guard against invalid inputs, race conditions, or failures?)\n"
            "4. Code Clarity & Maintainability (Is the code clean, concise, and readable?)\n\n"
            "CRITICAL RULES:\n"
            "- Candidates MUST maintain functioning logic and passing contracts.\n"
            "- Heavily penalize (score < 4.0) and REJECT any candidate that replaces real logic with placeholder strings (e.g. 'placeholder', 'REPLACE_WITH_...'), mock stubs, or unsupported protocol rewrites.\n"
            "- Heavily penalize (score < 4.0) and REJECT any candidate that breaks functional queries by inventing arbitrary parameter whitelists (e.g. restricting roles to only admin/user and rejecting 'developer') or changing row_factory.\n"
            "- Heavily penalize (score < 4.0) and REJECT any candidate that validates secrets or raises ValueError at module import time instead of inside generate_token (the module must import cleanly when environment variables are unset).\n"
            "- Heavily penalize (score < 4.0) and REJECT any candidate that breaks the token format contract (e.g. prepending user_id or dots like user_id.signature; generate_token must return strictly the 64-char hex digest).\n"
            "- If a candidate hardcodes fallback secret strings or uses non-constant-time comparisons, penalize it.\n\n"
            "Format your response exactly as follows:\n\n"
            "---SCORES---\n"
            "Alpha Score: <float 1.0-10.0>\n"
            "Beta Score: <float 1.0-10.0>\n\n"
            "---WINNER---\n"
            "SELECTED_WINNER: Candidate Alpha OR Candidate Beta\n\n"
            "---CRITIQUE---\n"
            "<Detailed comparative analysis explaining strengths, weaknesses, and winning rationale>\n"
        )

    def _parse_critic_response(
        self,
        raw_text: str,
        alpha_builder: str,
        beta_builder: str,
        eval_tokens: int,
        eval_duration_s: float,
    ) -> CriticResult:
        """Parse structured scores, winner, and critique from Critic output."""
        alpha_score = 7.5
        beta_score = 7.5
        winner = "Candidate Alpha"

        alpha_match = re.search(r"Alpha\s*Score\s*:\s*([0-9]+(?:\.[0-9]+)?)", raw_text, re.IGNORECASE)
        if alpha_match:
            try:
                alpha_score = float(alpha_match.group(1))
            except ValueError:
                pass

        beta_match = re.search(r"Beta\s*Score\s*:\s*([0-9]+(?:\.[0-9]+)?)", raw_text, re.IGNORECASE)
        if beta_match:
            try:
                beta_score = float(beta_match.group(1))
            except ValueError:
                pass

        winner_match = re.search(
            r"SELECTED_WINNER\s*:\s*(Candidate\s+Alpha|Candidate\s+Beta)",
            raw_text,
            re.IGNORECASE,
        )
        if winner_match:
            raw_winner = winner_match.group(1).title()
            winner = "Candidate Alpha" if "Alpha" in raw_winner else "Candidate Beta"
        else:
            # Fallback based on score comparison
            winner = "Candidate Beta" if beta_score > alpha_score else "Candidate Alpha"

        critique_match = re.search(r"---CRITIQUE---\s*(.*)", raw_text, re.DOTALL)
        if critique_match:
            critique = critique_match.group(1).strip()
        else:
            critique = raw_text.strip()

        winning_builder = alpha_builder if winner == "Candidate Alpha" else beta_builder

        return CriticResult(
            winner=winner,
            winning_builder=winning_builder,
            alpha_score=alpha_score,
            beta_score=beta_score,
            critique=critique,
            eval_tokens=eval_tokens,
            eval_duration_s=eval_duration_s,
        )

    def evaluate(
        self,
        candidate_1: Union[PatchResult, str],
        candidate_2: Union[PatchResult, str],
        strategy: StrategyResult,
        on_chunk: Optional[Callable[[str, str], None]] = None,
        builder_1_name: str = "Builder A",
        builder_2_name: str = "Builder B",
        shuffle: bool = True,
    ) -> CriticResult:
        """Adversarially evaluate two candidate patches in an anonymized debate.

        Args:
            candidate_1: First candidate patch code or PatchResult.
            candidate_2: Second candidate patch code or PatchResult.
            strategy: Cloud Architect strategy and invariants.
            on_chunk: Optional callback receiving (token_chunk, accumulated_text).
            builder_1_name: Name of builder 1 (if candidate_1 is str).
            builder_2_name: Name of builder 2 (if candidate_2 is str).
            shuffle: Whether to randomly swap Candidate Alpha and Beta positions.

        Returns:
            CriticResult with scores, winning candidate, mapped winning builder, and critique.
        """
        code_1 = candidate_1.patch_code if isinstance(candidate_1, PatchResult) else candidate_1
        name_1 = candidate_1.builder_name if isinstance(candidate_1, PatchResult) else builder_1_name

        code_2 = candidate_2.patch_code if isinstance(candidate_2, PatchResult) else candidate_2
        name_2 = candidate_2.builder_name if isinstance(candidate_2, PatchResult) else builder_2_name

        # Anonymize source code
        clean_1 = anonymize_code(code_1)
        clean_2 = anonymize_code(code_2)

        # Shuffle positions
        if shuffle and random.random() < 0.5:
            alpha_code, alpha_builder = clean_2, name_2
            beta_code, beta_builder = clean_1, name_1
        else:
            alpha_code, alpha_builder = clean_1, name_1
            beta_code, beta_builder = clean_2, name_2

        prompt = self._build_prompt(alpha_code, beta_code, strategy)
        accumulated_text = ""
        eval_tokens = 0
        eval_duration_s = 0.0

        for token, current_accumulated, meta in self.client.generate_stream(prompt):
            accumulated_text = current_accumulated
            if on_chunk:
                on_chunk(token, accumulated_text)

            if meta.get("done", False):
                eval_tokens = meta.get("eval_count", 0)
                eval_duration_s = meta.get("eval_duration_s", 0.0)

        return self._parse_critic_response(
            accumulated_text,
            alpha_builder,
            beta_builder,
            eval_tokens,
            eval_duration_s,
        )
