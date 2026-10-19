"""Cloud-written contract tests from a subtask's docstring specification (SPEC_TESTS).

Owner-approved, spec-level: the cloud sees signatures and docstrings of the
affected files (the same privacy level as cloud assist) and returns a unittest
module that checks every rule the docstrings state. The module is statically
validated before it is written next to the task's own tests, where the normal
``python3 -m unittest`` verification discovers it.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from antigravity_hybrid_orchestrator.planner.dag import SubTask
from antigravity_hybrid_orchestrator.privacy.assist_payload import describe_contract, verify_assist_payload
from antigravity_hybrid_orchestrator.privacy.verifier import SecurityInvariantViolationError

TEST_FILE = "test_spec_contract.py"
FORBIDDEN_CALLS = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "globals",
                   "locals", "vars", "getattr", "setattr", "delattr"}

PROMPT = """\
You write acceptance tests from a specification. Below are the signatures and docstrings of
Python functions; their implementation is hidden from you and may be wrong.
Write ONE Python unittest module that checks EVERY rule the docstrings state: correct results
for representative valid inputs (including edge values) and ValueError for every kind of
invalid input the docstrings describe. Only assert behaviour the docstrings state explicitly;
when a docstring is silent, do not test that case.
Rules: import only `unittest` and the module(s) under test ({modules}); no other imports, no
file, process or network access; one TestCase class; short test methods.
Answer directly in plain text. Do not use any tools: do not run commands, do not read or write
files and do not browse. Everything you can know is in this message.
Reply with the module in ONE fenced ```python block.

SPECIFICATION
{contract}
"""


def validate_contract_tests(source: str, modules: List[str]) -> List[str]:
    """Static safety and shape checks for a generated test module."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    allowed = {"unittest", *modules}
    problems: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in allowed:
                    problems.append(f"import of {alias.name!r} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in allowed:
                problems.append(f"import of {node.module!r} is not allowed")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            problems.append(f"call to {node.func.id!r} is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            problems.append(f"dunder attribute {node.attr!r} is not allowed")
    has_testcase = any(
        isinstance(node, ast.ClassDef) and any(
            (isinstance(b, ast.Attribute) and b.attr == "TestCase") or (isinstance(b, ast.Name) and b.id == "TestCase")
            for b in node.bases)
        for node in tree.body)
    if not has_testcase:
        problems.append("no unittest.TestCase subclass")
    return problems


def _extract_python(text: str) -> Optional[str]:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.S)
    return blocks[-1] if blocks else None


class SpecTestWriter:
    def __init__(self, complete: Callable[[str], Any],
                 audit: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> None:
        self.complete = complete
        self.audit = audit or (lambda event, payload: None)
        self.tokens_used = 0

    def generate(self, workspace: Path, subtask: SubTask, attempts: int = 2) -> Optional[str]:
        files = [str(f) for f in subtask.affected_files or []]
        modules = [Path(f).stem for f in files if f.endswith(".py")]
        contract = describe_contract(Path(workspace), files)
        if not contract.strip():
            self.audit("spec_tests_skipped", {"reason": "no signatures or docstrings"})
            return None
        try:
            verify_assist_payload(contract)
        except SecurityInvariantViolationError as exc:
            self.audit("spec_tests_blocked", {"reason": str(exc)})
            return None
        error = ""
        for _ in range(attempts):
            prompt = PROMPT.format(modules=", ".join(modules), contract=contract)
            if error:
                prompt += f"\nYOUR PREVIOUS REPLY WAS REJECTED: {error}\n"
            self.audit("spec_tests_request", {"prompt": prompt})
            try:
                text, t_in, t_out = self.complete(prompt)
            except Exception as exc:
                self.audit("spec_tests_failed", {"error": str(exc)})
                return None
            self.tokens_used += int(t_in) + int(t_out)
            self.audit("spec_tests_response", {"response": text, "input_tokens": int(t_in),
                                               "output_tokens": int(t_out)})
            source = _extract_python(text)
            if source is None:
                error = "no ```python block found"
                continue
            problems = validate_contract_tests(source, modules)
            if problems:
                error = "; ".join(problems)
                continue
            return source
        self.audit("spec_tests_rejected", {"error": error})
        return None

    @staticmethod
    def install(workspace: Path, subtask: SubTask, source: str) -> Path:
        path = Path(workspace) / TEST_FILE
        path.write_text(source, encoding="utf-8")
        if TEST_FILE not in subtask.input_files:
            subtask.input_files = [*subtask.input_files, TEST_FILE]
        return path
