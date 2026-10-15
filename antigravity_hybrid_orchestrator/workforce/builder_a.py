"""Builder A agent: Minimal surgical fix specialist."""

import ast
import re
from typing import Callable, Optional

from antigravity_hybrid_orchestrator.architect import StrategyResult
from antigravity_hybrid_orchestrator.workforce import PatchResult
from antigravity_hybrid_orchestrator.workforce.ollama_client import WorkforceClient


def extract_python_code(text: str) -> str:
    """Extract python source code enclosed in markdown code fences or raw text.
    Handles closed fences, unclosed fences, thought tags, and raw python code."""
    if not text or not text.strip():
        return ""

    # Strip thinking/thought blocks if present
    cleaned_text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL)
    cleaned_text = cleaned_text.strip()

    # 1. Closed code fences ```python ... ``` or ``` ... ```
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
    if matches:
        for match in reversed(matches):
            candidate = match.strip() + "\n"
            try:
                ast.parse(candidate)
                return candidate
            except SyntaxError:
                continue
        return matches[-1].strip() + "\n"

    # 2. Unclosed code fence ```python ... (stream cut off before closing ```)
    unclosed_match = re.search(r"```(?:python)?\s*\n(.*)", cleaned_text, re.DOTALL | re.IGNORECASE)
    if unclosed_match:
        candidate = unclosed_match.group(1).strip() + "\n"
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            pass
        return candidate

    # 3. Direct raw code - check if it parses directly as Python
    candidate = cleaned_text + "\n"
    try:
        ast.parse(candidate)
        return candidate
    except SyntaxError:
        pass

    # 4. Fallback: find first 'import' or 'from' or 'def' or 'class' line and take from there
    lines = cleaned_text.splitlines()
    code_start_idx = None
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if (
            stripped_line.startswith(("import ", "from ", "def ", "class ", "#!/usr/bin/env"))
            or re.match(r"^[A-Z0-9_]+\s*=", stripped_line)
        ):
            code_start_idx = i
            break
    if code_start_idx is not None:
        candidate = "\n".join(lines[code_start_idx:]).strip() + "\n"
        return candidate

    return cleaned_text + "\n" if cleaned_text else ""


def get_module_remediation_requirements(vulnerable_code: str, is_builder_b: bool = False) -> str:
    code_lower = vulnerable_code.lower()
    if "authenticate_api_key" in code_lower or "jwt_secret" in code_lower:
        base = (
            "- Target Module: auth.py (Authentication & Token Security)\n"
            "- Hardcoded secrets remediation: Retrieve secrets dynamically from environment variables (os.environ.get) inside functions or helper functions (e.g. get_jwt_secret(), get_admin_api_key()).\n"
            "- In generate_token: Return strictly the standard 64-character SHA-256 hex digest `hmac.new(secret.encode(), user_id.encode(), hashlib.sha256).hexdigest()`. Do NOT prepend user_id or a dot (do NOT return `<user_id>.<signature>`), preserving the exact 64-character token contract.\n"
            "- In generate_token: raise ValueError if JWT_SECRET is unset or empty.\n"
            "- In verify_token: Compare token directly against `generate_token(user_id)` using `hmac.compare_digest(token, expected)`. Return False if token cannot be generated or verification fails.\n"
            "- In authenticate_api_key: return False if ADMIN_API_KEY is unset, empty, or invalid (never raise ValueError).\n"
            "- CRITICAL: NEVER validate secrets or raise ValueError at module import time — the module must import cleanly when environment variables are unset.\n"
            "- Constant-time comparisons: Use hmac.compare_digest for comparing passwords, tokens, hashes, and API keys.\n"
            "- Do NOT import sqlite3, threading, or include code from other modules. Remediate ONLY the auth module."
        )
        if is_builder_b:
            base += "\n- Defensive checks: Validate parameter types (ensure string inputs) and handle exceptions gracefully."
        return base
    elif "billingsystem" in code_lower or "withdraw" in code_lower:
        base = (
            "- Target Module: billing.py (Concurrent Billing System)\n"
            "- Concurrency remediation: Protect shared state modifications with threading.Lock() or threading.RLock().\n"
            "- In BillingSystem.__init__: initialize self.lock = threading.Lock() or threading.RLock().\n"
            "- In create_account, get_balance, and withdraw: wrap shared state access/mutation with `with self.lock:`.\n"
            "- In withdraw: access self.balances.get(account_id, 0.0) directly under the lock (or use threading.RLock) so checking balance does not cause self-deadlock.\n"
            "- Keep the simulated latency `time.sleep(0.005)` inside the withdraw lock to ensure atomic check-and-decrement.\n"
            "- Do NOT add code or imports from other modules. Remediate ONLY the billing module."
        )
        if is_builder_b:
            base += "\n- Defensive checks: Validate account_id is string and amount is a non-negative number, returning False on invalid inputs."
        return base
    elif "databasemanager" in code_lower or "sqlite3" in code_lower:
        base = (
            "- Target Module: database.py (SQLite Database Layer)\n"
            "- SQL injection remediation: Use parameterized queries ('?' placeholders with parameter tuples), NEVER string formatting or concatenation.\n"
            "- Keep original db_path parameter with default ':memory:'. Do NOT require environment variables for db_path.\n"
            "- Do NOT restrict allowed roles or usernames (any string role such as 'developer' must be allowed, and usernames with quotes or semicolons must be safely inserted via parameters).\n"
            "- Return standard sqlite3 rows/tuples without changing row_factory.\n"
            "- Do NOT add code or imports from other modules. Remediate ONLY the database module."
        )
        if is_builder_b:
            base += "\n- Defensive checks: Validate string types for inputs before executing queries."
        return base
    else:
        return (
            "- Remediate vulnerabilities identified in the threat model.\n"
            "- Preserve public API interfaces and type contracts.\n"
            "- Do NOT add code from other modules."
        )


class BuilderAAgent:
    """Builder A: Surgical precision and minimal blast radius specialist.

    Philosophy:
        Make the smallest possible edit that eradicates the vulnerability.
        Avoid extraneous refactorings, retain original variable names,
        and minimize regression surface area.
    """

    NAME = "Builder A"

    def __init__(self, client: Optional[WorkforceClient] = None) -> None:
        self.client = client or WorkforceClient()

    def _build_prompt(self, vulnerable_code: str, strategy: StrategyResult) -> str:
        invariants_bullet = "\n".join(f"- {inv}" for inv in strategy.invariants)
        module_reqs = get_module_remediation_requirements(vulnerable_code, is_builder_b=False)
        return (
            "You are Builder A, a surgical security remediation specialist.\n"
            "Your philosophy is minimal diff and minimal blast radius.\n"
            "Fix the security vulnerabilities described in the strategy with the smallest possible change.\n"
            "Preserve all function signatures, variable names, class layouts, and formatting.\n"
            "Do NOT add unnecessary boilerplate, comments, or unrelated refactoring.\n\n"
            "CRITICAL REMEDIATION REQUIREMENTS:\n"
            f"{module_reqs}\n"
            "- Do NOT use placeholders, mock functions, or stubs. The module must be 100% functional and pass all tests.\n"
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
        """Generate a minimal surgical patch using on-device Gemma 4.

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
