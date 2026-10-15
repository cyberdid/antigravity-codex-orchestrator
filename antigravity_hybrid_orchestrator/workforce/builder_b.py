"""Builder B agent: Defensive / hardened specialist."""

from typing import Callable, Optional

from antigravity_hybrid_orchestrator.architect import StrategyResult
from antigravity_hybrid_orchestrator.workforce import PatchResult
from antigravity_hybrid_orchestrator.workforce.builder_a import (
    extract_python_code,
    get_module_remediation_requirements,
)
from antigravity_hybrid_orchestrator.workforce.ollama_client import WorkforceClient


class BuilderBAgent:
    """Builder B: Defense-in-depth and enterprise resilience specialist.

    Philosophy:
        Implement comprehensive hardening: strict input validation, fail-safe defaults,
        descriptive error handling, boundary checks, and robust state protection.
    """

    NAME = "Builder B"

    def __init__(self, client: Optional[WorkforceClient] = None) -> None:
        self.client = client or WorkforceClient()

    def _build_prompt(self, vulnerable_code: str, strategy: StrategyResult) -> str:
        invariants_bullet = "\n".join(f"- {inv}" for inv in strategy.invariants)
        module_reqs = get_module_remediation_requirements(vulnerable_code, is_builder_b=True)
        return (
            "You are Builder B, a defense-in-depth security engineer.\n"
            "Your philosophy is comprehensive defensive hardening and enterprise resilience.\n"
            "Fix the security vulnerabilities described in the strategy, add robust input validation,\n"
            "boundary checks, and fail-safe defaults. Guard against null pointers, malformed types,\n"
            "and concurrency edge cases.\n"
            "Preserve all public function signatures, parameter names, and return types.\n\n"
            "CRITICAL REMEDIATION REQUIREMENTS:\n"
            f"{module_reqs}\n"
            "- Do NOT replace working logic with placeholder stubs or change crypto protocols (e.g. do NOT replace HMAC with mock RSA). The module must be 100% functional and pass all tests.\n"
            "- CRITICAL: Output ONLY the remediated version of the target module. Do NOT append or define code from other modules.\n\n"
            f"Threat Model: {strategy.threat_model}\n"
            f"Security Invariants:\n{invariants_bullet}\n"
            f"Remediation Directive: {strategy.raw_strategy}\n\n"
            "Target Vulnerable Code:\n"
            "```python\n"
            f"{vulnerable_code.strip()}\n"
            "```\n\n"
            "Output ONLY the complete raw patched Python module code enclosed in a single ```python codeblock.\n"
        )

    def generate_patch(
        self,
        vulnerable_code: str,
        strategy: StrategyResult,
        on_chunk: Optional[Callable[[str, str], None]] = None,
    ) -> PatchResult:
        """Generate a defensively hardened patch using on-device Gemma 4.

        Args:
            vulnerable_code: The vulnerable source code to remediate.
            strategy: High-level architectural strategy and invariants from Gemini.
            on_chunk: Optional callback receiving (token_chunk, accumulated_text).

        Returns:
            PatchResult containing builder name, patched code, and token telemetry.
        """
        prompt = self._build_prompt(vulnerable_code, strategy)
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

        patch_code = extract_python_code(accumulated_text)
        return PatchResult(
            builder_name=self.NAME,
            patch_code=patch_code,
            eval_tokens=eval_tokens,
            eval_duration_s=eval_duration_s,
        )
