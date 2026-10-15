"""Pre-flight verification engine for zero-knowledge cloud code offloading.

Validates that stripped code strictly enforces the 0-byte implementation leak invariant:
- Stripped code is parseable valid Python AST.
- Every function body contains solely an Ellipsis (`...`) or Pass statement.
- ZERO executable logic nodes (If, For, While, Try, With, Return, Yield, Call, BinOp, AugAssign, Lambda, Match, Comprehensions).
- ZERO leaked string/bytes literals (length >= 5) from original implementations.
"""

import ast
import re
from typing import List, Optional, Set, Union


class SecurityInvariantViolationError(Exception):
    """Raised when a code payload violates the zero-knowledge privacy invariant."""
    pass


FORBIDDEN_LOGIC_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.Return,
    ast.Yield,
    ast.YieldFrom,
    ast.Call,
    ast.BinOp,
    ast.AugAssign,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.Raise,
    ast.Delete,
    ast.Assert,
    ast.IfExp,
    ast.Compare,
    ast.NamedExpr,
    ast.BoolOp,
    ast.UnaryOp,
)

if hasattr(ast, "TryStar"):
    FORBIDDEN_LOGIC_NODES = FORBIDDEN_LOGIC_NODES + (ast.TryStar,)

if hasattr(ast, "Match"):
    FORBIDDEN_LOGIC_NODES = FORBIDDEN_LOGIC_NODES + (ast.Match,)


def verify_cloud_payload(stripped_code: str, original_code: Optional[str] = None) -> bool:
    """Verifies that stripped_code conforms strictly to the privacy invariant.

    Args:
        stripped_code: The candidate AST-stripped source code intended for cloud transmission.
        original_code: The original source code containing implementations and secrets.

    Returns:
        True if all privacy invariants are upheld.

    Raises:
        SecurityInvariantViolationError: If any implementation detail, forbidden node,
            or sensitive literal is detected.
    """
    trees: List[ast.AST] = []
    try:
        tree = ast.parse(stripped_code)
        trees.append(tree)
    except SyntaxError as e:
        # Support markdown code-block extraction
        blocks = re.findall(r"```(?:python|py)?\s*\n?(.*?)\n?```", stripped_code, flags=re.DOTALL)
        if not blocks:
            raise SecurityInvariantViolationError(f"Stripped code is not valid Python AST: {e}")
        for i, block in enumerate(blocks):
            if not block.strip():
                continue
            try:
                b_tree = ast.parse(block)
                trees.append(b_tree)
            except SyntaxError as block_err:
                raise SecurityInvariantViolationError(f"Code block #{i+1} is not valid Python AST: {block_err}")
        if not trees:
            trees.append(ast.Module(body=[], type_ignores=[]))

    for t in trees:
        # 1. Assert every function/method body is exactly 1 statement: Ellipsis (...) or pass
        for node in ast.walk(t):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if len(node.body) != 1:
                    raise SecurityInvariantViolationError(
                        f"Function '{node.name}' body length > 1 ({len(node.body)})"
                    )
                stmt = node.body[0]
                is_ellipsis = (
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)
                    and stmt.value.value is Ellipsis
                )
                is_pass = isinstance(stmt, ast.Pass)
                if not (is_ellipsis or is_pass):
                    raise SecurityInvariantViolationError(
                        f"Function '{node.name}' body is not Ellipsis or Pass statement"
                    )

            # 2. Check for forbidden executable logic nodes
            if isinstance(node, FORBIDDEN_LOGIC_NODES):
                raise SecurityInvariantViolationError(
                    f"Forbidden logic node {type(node).__name__} present in cloud payload"
                )

    # 3. Assert zero sensitive string/bytes literals (length >= 5) from the original code leak
    if original_code is not None:
        try:
            orig_tree = ast.parse(original_code)
        except Exception:
            orig_tree = None

        if orig_tree is not None:
            sensitive_str_literals: Set[str] = set()
            sensitive_bytes_literals: Set[bytes] = set()

            for node in ast.walk(orig_tree):
                if isinstance(node, ast.Constant):
                    if isinstance(node.value, str):
                        val_str = node.value.strip()
                        if len(val_str) >= 5:
                            sensitive_str_literals.add(val_str)
                    elif isinstance(node.value, bytes):
                        if len(node.value) >= 5:
                            sensitive_bytes_literals.add(node.value)

            # 3a. Direct check on AST Constant values in stripped payload
            for t in trees:
                for node in ast.walk(t):
                    if isinstance(node, ast.Constant):
                        if isinstance(node.value, str):
                            for s_lit in sensitive_str_literals:
                                if s_lit in node.value:
                                    raise SecurityInvariantViolationError(
                                        f"Original string/bytes literal {s_lit!r} leaked into cloud payload"
                                    )
                        elif isinstance(node.value, bytes):
                            for b_lit in sensitive_bytes_literals:
                                if b_lit in node.value:
                                    raise SecurityInvariantViolationError(
                                        f"Original string/bytes literal {b_lit!r} leaked into cloud payload"
                                    )

            # 3b. Regex quoted text check on non-import lines (preventing false-positive identifier collisions)
            non_import_lines = [
                line.strip()
                for line in stripped_code.splitlines()
                if not line.strip().startswith(("import ", "from "))
            ]
            non_import_text = "\n".join(non_import_lines)

            for s_lit in sensitive_str_literals:
                if re.search(r'["|\']' + re.escape(s_lit) + r'["|\']', non_import_text):
                    raise SecurityInvariantViolationError(
                        f"Original string/bytes literal {s_lit!r} leaked into cloud payload"
                    )

            for b_lit in sensitive_bytes_literals:
                b_repr = repr(b_lit)
                inner = (
                    b_repr[2:-1]
                    if b_repr.startswith(("b'", 'b"')) and b_repr.endswith(("'", '"'))
                    else str(b_lit)
                )
                if inner in non_import_text or b_repr in non_import_text:
                    raise SecurityInvariantViolationError(
                        f"Original string/bytes literal {b_lit!r} leaked into cloud payload"
                    )

    return True
