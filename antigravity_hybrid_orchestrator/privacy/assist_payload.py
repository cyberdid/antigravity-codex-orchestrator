"""Spec-level payload for owner-approved cloud assist.

Beyond today's zero-knowledge outline this adds docstrings of the subtask's
affected functions and the failure excerpt of the trusted verification run. It
never includes function or test bodies, default values, file contents or diffs.
"""

from __future__ import annotations

import ast
import copy
import math
import re
from collections import Counter
from pathlib import Path
from typing import List

from antigravity_hybrid_orchestrator.planner.dag import SubTask
from antigravity_hybrid_orchestrator.privacy.verifier import SecurityInvariantViolationError

MAX_PAYLOAD_BYTES = 8192

_SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[bap]-[A-Za-z0-9-]{10,}"),
]
_TOKEN = re.compile(r"[A-Za-z0-9+/=_-]{32,}")
# Above the 4.0 bits/char ceiling of hex, so commit and file digests pass.
_ENTROPY_LIMIT = 4.2


def _entropy(token: str) -> float:
    counts = Counter(token)
    return -sum(n / len(token) * math.log2(n / len(token)) for n in counts.values())


def _signature(node: ast.AST) -> str:
    """``def name(args) -> ret`` with every default replaced by ``...``."""
    args = copy.deepcopy(node.args)
    args.defaults = [ast.Constant(value=Ellipsis) for _ in args.defaults]
    args.kw_defaults = [None if d is None else ast.Constant(value=Ellipsis) for d in args.kw_defaults]
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix} {node.name}({ast.unparse(args)}){returns}"


def _describe_file(workspace: Path, relative: str) -> List[str]:
    path = (workspace / relative).resolve()
    if path.suffix != ".py" or not path.is_relative_to(workspace) or not path.is_file():
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
        return []
    lines = [f"# {relative}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            lines.append(f"class {node.name}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lines.append(_signature(node))
        else:
            continue
        doc = ast.get_docstring(node)
        if doc:
            lines.extend("    " + line for line in doc.splitlines())
    return lines


def describe_contract(workspace: Path, files: List[str]) -> str:
    """Signatures and docstrings of the given files (the specification, never bodies)."""
    workspace = Path(workspace).resolve()
    lines: List[str] = []
    for relative in files or []:
        lines.extend(_describe_file(workspace, str(relative)))
    return "\n".join(lines)


def build_assist_payload(workspace: Path, subtask: SubTask, failure_excerpt: str) -> str:
    workspace = Path(workspace).resolve()
    sections = [
        "SUBTASK",
        f"id: {subtask.id}",
        f"title: {subtask.title}",
        f"description: {subtask.description}",
        f"success criteria: {subtask.success_criteria}",
        "",
        "SIGNATURES AND DOCSTRINGS",
    ]
    for relative in subtask.affected_files or []:
        sections.extend(_describe_file(workspace, str(relative)))
    sections += ["", "FAILURE", failure_excerpt.strip()]
    return "\n".join(sections)


def verify_assist_payload(payload: str, *, max_bytes: int = MAX_PAYLOAD_BYTES) -> None:
    size = len(payload.encode("utf-8"))
    if size > max_bytes:
        raise SecurityInvariantViolationError(f"Cloud assist payload is {size} bytes (limit {max_bytes})")
    for pattern in _SECRET_PATTERNS:
        if pattern.search(payload):
            raise SecurityInvariantViolationError("Cloud assist payload contains a credential-shaped value")
    for token in _TOKEN.findall(payload):
        if _entropy(token) > _ENTROPY_LIMIT:
            raise SecurityInvariantViolationError("Cloud assist payload contains a high-entropy string")
