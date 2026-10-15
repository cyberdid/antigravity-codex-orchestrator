"""AST Privacy Stripper for zero-knowledge cloud code offloading.

Transforms Python source code by stripping all function/method implementation bodies,
global assignment literal values, and docstrings, while strictly preserving module imports,
class structures, function headers, parameter signatures, and type annotations.
"""

import ast
from typing import Optional, List


class ASTPrivacyStripper(ast.NodeTransformer):
    """AST transformer that removes implementation details and sensitive literals.

    - Replaces function and async function bodies with a single Ellipsis expression (`...`).
    - Strips literal assignment values (`x = ...`, `y: int = ...`).
    - Strips default argument literal values (`api_key: str = ...`).
    - Sanitizes decorators to prevent literal leakage and remove logic calls.
    - Discards module, class, and function docstrings and standalone constant expressions.
    - Discards top-level executable control flow logic (If, For, While, Try, With, Match, AugAssign, Raise, Delete).
    - Preserves imports, class definitions, function signatures, and type annotations.
    """

    def _sanitize_default(self, expr: ast.AST) -> ast.AST:
        """Sanitizes default argument expressions to prevent literal and logic leaks."""
        if isinstance(expr, ast.Constant):
            if expr.value is None or expr.value is Ellipsis:
                return expr
            if isinstance(expr.value, bool):
                return expr
            if isinstance(expr.value, (int, float)):
                return expr
            return ast.Constant(value=Ellipsis)
        return ast.Constant(value=Ellipsis)

    def _sanitize_decorators(self, decos: List[ast.AST]) -> List[ast.AST]:
        """Sanitizes decorators, stripping arguments and complex literal expressions."""
        new_decos = []
        for d in decos:
            if isinstance(d, ast.Call):
                if isinstance(d.func, (ast.Name, ast.Attribute)):
                    new_decos.append(d.func)
                else:
                    new_decos.append(ast.Constant(value=Ellipsis))
            elif isinstance(d, ast.Constant) and isinstance(d.value, (str, bytes)):
                new_decos.append(ast.Constant(value=Ellipsis))
            elif isinstance(d, ast.Subscript):
                class SliceSanitizer(ast.NodeTransformer):
                    def visit_Constant(self, c: ast.Constant) -> ast.AST:
                        if isinstance(c.value, (str, bytes)):
                            return ast.Constant(value=Ellipsis)
                        return c
                new_decos.append(SliceSanitizer().visit(d))
            else:
                new_decos.append(d)
        return new_decos

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        # Sanitize parameter defaults
        node.args.defaults = [self._sanitize_default(d) for d in node.args.defaults]
        node.args.kw_defaults = [
            self._sanitize_default(d) if d is not None else None
            for d in node.args.kw_defaults
        ]
        # Sanitize literal string annotations
        for arg in (node.args.args + node.args.posonlyargs + node.args.kwonlyargs):
            if isinstance(arg.annotation, ast.Constant) and isinstance(arg.annotation.value, (str, bytes)):
                arg.annotation = ast.Constant(value=Ellipsis)
        if isinstance(node.returns, ast.Constant) and isinstance(node.returns.value, (str, bytes)):
            node.returns = ast.Constant(value=Ellipsis)
        # Sanitize decorators
        node.decorator_list = self._sanitize_decorators(node.decorator_list)
        # Replace entire function body with a single Ellipsis statement: ...
        node.body = [ast.Expr(value=ast.Constant(value=Ellipsis))]
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        # Sanitize parameter defaults
        node.args.defaults = [self._sanitize_default(d) for d in node.args.defaults]
        node.args.kw_defaults = [
            self._sanitize_default(d) if d is not None else None
            for d in node.args.kw_defaults
        ]
        # Sanitize literal string annotations
        for arg in (node.args.args + node.args.posonlyargs + node.args.kwonlyargs):
            if isinstance(arg.annotation, ast.Constant) and isinstance(arg.annotation.value, (str, bytes)):
                arg.annotation = ast.Constant(value=Ellipsis)
        if isinstance(node.returns, ast.Constant) and isinstance(node.returns.value, (str, bytes)):
            node.returns = ast.Constant(value=Ellipsis)
        # Sanitize decorators
        node.decorator_list = self._sanitize_decorators(node.decorator_list)
        # Replace entire function body with a single Ellipsis statement: ...
        node.body = [ast.Expr(value=ast.Constant(value=Ellipsis))]
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        node.decorator_list = self._sanitize_decorators(node.decorator_list)
        self.generic_visit(node)
        if not node.body:
            node.body = [ast.Expr(value=ast.Constant(value=Ellipsis))]
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        node.value = ast.Constant(value=Ellipsis)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        if node.value is not None:
            node.value = ast.Constant(value=Ellipsis)
        if isinstance(node.annotation, ast.Constant) and isinstance(node.annotation.value, (str, bytes)):
            node.annotation = ast.Constant(value=Ellipsis)
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> Optional[ast.AST]:
        # Discard top-level augmented assignments to prevent executable logic leaks
        return None

    def visit_Expr(self, node: ast.Expr) -> Optional[ast.AST]:
        if isinstance(node.value, ast.Constant):
            if node.value.value is Ellipsis:
                return node
            return None
        return None

    def visit_If(self, node: ast.If) -> Optional[ast.AST]:
        return None

    def visit_For(self, node: ast.For) -> Optional[ast.AST]:
        return None

    def visit_AsyncFor(self, node: ast.AsyncFor) -> Optional[ast.AST]:
        return None

    def visit_While(self, node: ast.While) -> Optional[ast.AST]:
        return None

    def visit_Try(self, node: ast.Try) -> Optional[ast.AST]:
        return None

    def visit_With(self, node: ast.With) -> Optional[ast.AST]:
        return None

    def visit_AsyncWith(self, node: ast.AsyncWith) -> Optional[ast.AST]:
        return None

    def visit_Assert(self, node: ast.Assert) -> Optional[ast.AST]:
        return None

    def visit_Match(self, node: ast.AST) -> Optional[ast.AST]:
        return None

    def visit_Raise(self, node: ast.Raise) -> Optional[ast.AST]:
        return None

    def visit_Delete(self, node: ast.Delete) -> Optional[ast.AST]:
        return None

    def visit_TypeAlias(self, node: ast.AST) -> ast.AST:
        if hasattr(node, "value"):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (str, bytes)):
                node.value = ast.Constant(value=Ellipsis)
            elif not isinstance(node.value, (ast.Name, ast.Attribute, ast.Subscript)):
                node.value = ast.Constant(value=Ellipsis)
        return node


def strip_code_for_cloud(source_code: str) -> str:
    """Parses source code, applies AST privacy stripping, and returns stripped signatures.

    Args:
        source_code: Raw Python module code string.

    Returns:
        Stripped AST source code containing solely interface signatures and imports.
    """
    tree = ast.parse(source_code)
    stripper = ASTPrivacyStripper()
    stripped_tree = stripper.visit(tree)
    ast.fix_missing_locations(stripped_tree)
    return ast.unparse(stripped_tree)
