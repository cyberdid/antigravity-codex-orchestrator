"""Zero-knowledge workspace outline privacy validation and multi-file AST stripping."""

import ast
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Union

from antigravity_hybrid_orchestrator.privacy.ast_stripper import ASTPrivacyStripper
from antigravity_hybrid_orchestrator.privacy.verifier import (
    FORBIDDEN_LOGIC_NODES,
    SecurityInvariantViolationError,
    verify_cloud_payload,
)

SENSITIVE_FILENAME_PATTERNS: Set[str] = {
    ".env",
    "id_rsa",
    "id_dsa",
    "id_ed25519",
    "credentials.json",
    "secret",
    "private_key.pem",
}


class AnnotationTransformer(ast.NodeTransformer):
    """Sanitizes type annotations recursively.

    - Unpacks `Annotated[T, ...]` to its base type `T`.
    - Replaces string/bytes constants in `Literal[...]` slices with `...`.
    - Replaces any standalone string/bytes constants with `...`.
    - Eliminates any forbidden executable logic nodes inside annotations with `...`.
    """

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        is_annotated = (
            (isinstance(node.value, ast.Name) and node.value.id == "Annotated")
            or (isinstance(node.value, ast.Attribute) and node.value.attr == "Annotated")
        )
        if is_annotated:
            if isinstance(node.slice, ast.Tuple) and node.slice.elts:
                return self.visit(node.slice.elts[0])
            return self.visit(node.slice)

        is_literal = (
            (isinstance(node.value, ast.Name) and node.value.id == "Literal")
            or (isinstance(node.value, ast.Attribute) and node.value.attr == "Literal")
        )
        if is_literal:
            if isinstance(node.slice, ast.Tuple):
                new_elts = []
                for elt in node.slice.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, (str, bytes)):
                        new_elts.append(ast.Constant(value=Ellipsis))
                    else:
                        new_elts.append(self.visit(elt))
                if all(isinstance(e, ast.Constant) and e.value is Ellipsis for e in new_elts):
                    node.slice = ast.Constant(value=Ellipsis)
                else:
                    node.slice.elts = new_elts
            elif isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, (str, bytes)):
                node.slice = ast.Constant(value=Ellipsis)
            else:
                node.slice = self.visit(node.slice)
            return node

        self.generic_visit(node)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, (str, bytes)):
            return ast.Constant(value=Ellipsis)
        return node

    def generic_visit(self, node: ast.AST) -> ast.AST:
        if isinstance(node, FORBIDDEN_LOGIC_NODES):
            return ast.Constant(value=Ellipsis)
        return super().generic_visit(node)


class WorkspaceASTTransformer(ASTPrivacyStripper):
    """AST transformer for workspace modules that strictly eliminates implementation details.

    - Replaces function/async function bodies with single Ellipsis (`...`).
    - Strips default argument literals of length >= 5 or non-constant expressions.
    - Preserves short safe literal defaults (< 5 chars, bools, ints, floats, None, Ellipsis).
    - Recursively sanitizes subscripts (Literal, Annotated) and removes secret literals from annotations.
    - Deeply sanitizes chained decorators (Attribute, Subscript, Call) to prevent literal leaks.
    - Discards all docstrings (module, class, function).
    - Strips all top-level executable statements (If, For, While, Try, With, Match, etc.).
    - Preserves class hierarchies, function headers, type annotations, and imports.
    """

    @classmethod
    def _sanitize_annotation(cls, node: Optional[ast.AST]) -> Optional[ast.AST]:
        if node is None:
            return None
        if isinstance(node, FORBIDDEN_LOGIC_NODES):
            return ast.Constant(value=Ellipsis)
        sanitized = AnnotationTransformer().visit(node)
        for child in ast.walk(sanitized):
            if isinstance(child, FORBIDDEN_LOGIC_NODES):
                return ast.Constant(value=Ellipsis)
        return sanitized

    def _sanitize_default(self, expr: ast.AST) -> ast.AST:
        if isinstance(expr, ast.Constant):
            if expr.value is None or expr.value is Ellipsis:
                return expr
            if isinstance(expr.value, bool):
                return expr
            if isinstance(expr.value, (int, float)):
                return expr
            if isinstance(expr.value, str):
                if len(expr.value) < 5:
                    return expr
                return ast.Constant(value=Ellipsis)
            if isinstance(expr.value, bytes):
                if len(expr.value) < 5:
                    return expr
                return ast.Constant(value=Ellipsis)
            return ast.Constant(value=Ellipsis)
        return ast.Constant(value=Ellipsis)

    def _sanitize_decorator_expr(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.Call):
            return self._sanitize_decorator_expr(node.func)
        elif isinstance(node, ast.Name):
            return node
        elif isinstance(node, ast.Attribute):
            node.value = self._sanitize_decorator_expr(node.value)
            return node
        elif isinstance(node, ast.Subscript):
            node.value = self._sanitize_decorator_expr(node.value)
            node.slice = ast.Constant(value=Ellipsis)
            return node
        elif isinstance(node, ast.Constant):
            return ast.Constant(value=Ellipsis)
        else:
            return ast.Constant(value=Ellipsis)

    def _sanitize_decorators(self, decos: List[ast.AST]) -> List[ast.AST]:
        return [self._sanitize_decorator_expr(d) for d in decos]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.args.defaults = [self._sanitize_default(d) for d in node.args.defaults]
        node.args.kw_defaults = [
            self._sanitize_default(d) if d is not None else None
            for d in node.args.kw_defaults
        ]
        all_args = list(node.args.args) + list(node.args.posonlyargs) + list(node.args.kwonlyargs)
        if node.args.vararg:
            all_args.append(node.args.vararg)
        if node.args.kwarg:
            all_args.append(node.args.kwarg)
        for arg in all_args:
            arg.annotation = self._sanitize_annotation(arg.annotation)
        node.returns = self._sanitize_annotation(node.returns)
        node.decorator_list = self._sanitize_decorators(node.decorator_list)
        node.body = [ast.Expr(value=ast.Constant(value=Ellipsis))]
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        node.args.defaults = [self._sanitize_default(d) for d in node.args.defaults]
        node.args.kw_defaults = [
            self._sanitize_default(d) if d is not None else None
            for d in node.args.kw_defaults
        ]
        all_args = list(node.args.args) + list(node.args.posonlyargs) + list(node.args.kwonlyargs)
        if node.args.vararg:
            all_args.append(node.args.vararg)
        if node.args.kwarg:
            all_args.append(node.args.kwarg)
        for arg in all_args:
            arg.annotation = self._sanitize_annotation(arg.annotation)
        node.returns = self._sanitize_annotation(node.returns)
        node.decorator_list = self._sanitize_decorators(node.decorator_list)
        node.body = [ast.Expr(value=ast.Constant(value=Ellipsis))]
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        if node.value is not None:
            node.value = ast.Constant(value=Ellipsis)
        node.annotation = self._sanitize_annotation(node.annotation)
        return node

    def visit_TypeAlias(self, node: ast.AST) -> ast.AST:
        if hasattr(node, "value") and node.value is not None:
            node.value = self._sanitize_annotation(node.value)
        return node


class WorkspaceASTStripper:
    """Multi-file zero-knowledge AST stripper for Python workspaces."""

    def __init__(self, fallback_on_error: bool = True) -> None:
        self.fallback_on_error = fallback_on_error
        self.transformer = WorkspaceASTTransformer()

    def strip_source(self, source_code: str, filename: str = "<unknown>") -> str:
        """Strips a single Python source code string to its zero-knowledge interface signatures.

        Args:
            source_code: Raw Python code.
            filename: Optional path or module name for diagnostics.

        Returns:
            Stripped AST code containing only imports, classes, and function headers with `...` bodies.
        """
        if not source_code.strip():
            return ""

        try:
            tree = ast.parse(source_code, filename=filename)
        except SyntaxError as e:
            if not self.fallback_on_error:
                raise
            return f"# [SyntaxError in {filename}: {e}]"
        except Exception as e:
            if not self.fallback_on_error:
                raise
            return f"# [Error stripping AST in {filename}: {e}]"

        stripped_tree = self.transformer.visit(tree)
        ast.fix_missing_locations(stripped_tree)
        return ast.unparse(stripped_tree)

    def strip_module(self, source_code: str, file_path: str = "") -> str:
        """Alias for strip_source for backward/interface compatibility."""
        return self.strip_source(source_code, filename=file_path)

    def strip_file(self, file_path: Union[str, Path]) -> str:
        """Reads and strips a Python file from disk."""
        path = Path(file_path)
        content = path.read_text(encoding="utf-8", errors="replace")
        return self.strip_source(content, filename=path.as_posix())

    def strip_files(
        self,
        workspace_root: Union[str, Path],
        rel_paths: List[str],
    ) -> Dict[str, str]:
        """Batch strips multiple files, returning mapping of rel_path -> stripped signature."""
        root = Path(workspace_root)
        signatures: Dict[str, str] = {}
        for rel in sorted(rel_paths):
            full_path = root / rel
            if full_path.is_file() and full_path.suffix.lower() in (".py", ".pyi"):
                signatures[rel] = self.strip_file(full_path)
        return signatures

    def strip_workspace(
        self,
        workspace_root: Union[str, Path],
        python_files: List[str],
    ) -> Dict[str, str]:
        """Alias for strip_files."""
        return self.strip_files(workspace_root, python_files)


def verify_workspace_outline(
    outline: Any,
    original_sources: Optional[Union[Dict[str, str], str, Path]] = None,
    workspace_root: Optional[Union[str, Path]] = None,
) -> bool:
    """Verifies that a WorkspaceOutline satisfies all zero-knowledge privacy invariants.

    Checks:
    1. No sensitive file names exist in outline.file_tree.
    2. Every Python signature in outline.ast_signatures conforms to verify_cloud_payload:
       - Valid Python AST.
       - Every function body is strictly Ellipsis (...) or pass.
       - Zero forbidden executable logic nodes.
       - Zero leaked string literals (>= 5 chars) from original source code.
    3. The combined outline.to_prompt_string() is valid Python AST / code block and passes verify_cloud_payload.

    Args:
        outline: The WorkspaceOutline instance to verify.
        original_sources: Optional mapping of rel_path -> unstripped source code,
            or root path if passed as second positional argument.
        workspace_root: Optional canonical workspace root path.

    Returns:
        True if all privacy invariants are upheld.

    Raises:
        SecurityInvariantViolationError: If any invariant is violated.
    """
    # Normalize original_sources vs workspace_root arguments
    sources_dict: Optional[Dict[str, str]] = None
    root_path: Optional[Path] = None

    if isinstance(original_sources, (str, Path)):
        root_path = Path(original_sources)
    elif isinstance(original_sources, dict):
        sources_dict = original_sources

    if workspace_root is not None:
        root_path = Path(workspace_root)
    elif root_path is None and hasattr(outline, "workspace_root") and outline.workspace_root:
        root_path = Path(outline.workspace_root)

    # 1. Check file tree for sensitive filenames
    for file_path in getattr(outline, "file_tree", []):
        name_lower = Path(file_path).name.lower()
        for sensitive in SENSITIVE_FILENAME_PATTERNS:
            if sensitive in name_lower:
                raise SecurityInvariantViolationError(
                    f"Sensitive file path '{file_path}' detected in workspace file tree"
                )

    # 2. Check each module signature individually
    ast_signatures = getattr(outline, "ast_signatures", {})
    for rel_path, signature in ast_signatures.items():
        if not signature or not signature.strip():
            continue
        # Skip syntax error comments if recorded as error annotations
        if signature.strip().startswith("# [SyntaxError"):
            continue

        orig: Optional[str] = None
        if sources_dict and rel_path in sources_dict:
            orig = sources_dict[rel_path]
        elif root_path:
            full_path = root_path / rel_path
            if full_path.is_file():
                try:
                    orig = full_path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    orig = None

        try:
            verify_cloud_payload(signature, original_code=orig)
        except SecurityInvariantViolationError as e:
            raise SecurityInvariantViolationError(
                f"Privacy invariant violated in module '{rel_path}': {e}"
            ) from e

    # 3. Check full prompt string
    if hasattr(outline, "to_prompt_string"):
        prompt_str = outline.to_prompt_string()
        try:
            verify_cloud_payload(prompt_str, original_code=None)
        except SecurityInvariantViolationError as e:
            raise SecurityInvariantViolationError(
                f"Combined workspace prompt payload failed privacy verification: {e}"
            ) from e

    return True


# Backward-compatible alias used in test strategy
verify_workspace_outline_privacy = verify_workspace_outline
