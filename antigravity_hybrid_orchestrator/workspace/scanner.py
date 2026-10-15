"""Workspace scanner and outline extraction engine for zero-knowledge AST ingestion."""

from dataclasses import dataclass, field
import fnmatch
import os
from pathlib import Path
import re
import stat
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from antigravity_hybrid_orchestrator.privacy.workspace_privacy import (
    WorkspaceASTStripper,
    WorkspaceASTTransformer,
    verify_workspace_outline,
)


class WorkspaceNotADirectoryError(ValueError, NotADirectoryError):
    """Raised when workspace path exists but is not a directory."""
    pass


def _sanitize_comment_text(value: Any) -> str:
    """Sanitizes text to ensure it cannot break out of a single-line Python comment (#).

    Replaces all newlines, carriage returns, and line separators with a single space,
    preventing newline injection attacks in serialized prompt headers and comments.
    """
    if value is None:
        return ""
    text = str(value)
    return re.sub(r"[\r\n\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029]+", " ", text).strip()


@dataclass
class WorkspaceOutline:
    """Zero-knowledge structural outline of a target workspace.

    Attributes:
        workspace_root: Canonical path to the target workspace root.
        file_tree: Alphabetically sorted list of relative POSIX file paths.
        ast_signatures: Mapping of relative Python file path to stripped AST code.
        summary_metadata: Dictionary of repository metrics and language stats.
    """

    workspace_root: Path
    file_tree: List[str] = field(default_factory=list)
    ast_signatures: Dict[str, str] = field(default_factory=dict)
    summary_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_prompt_string(self) -> str:
        """Serializes the outline into a zero-knowledge prompt payload.

        All headers and metadata delimiters are formatted as Python comments (#)
        so that the entire multi-file payload is valid Python AST parseable by ast.parse().
        """
        lines: List[str] = []
        root_name = self.workspace_root.name if self.workspace_root else "workspace"
        clean_root = _sanitize_comment_text(root_name) or "workspace"
        lines.append("# ==============================================================================")
        lines.append(f"# ZERO-KNOWLEDGE WORKSPACE OUTLINE: {clean_root}")

        total_files = _sanitize_comment_text(
            self.summary_metadata.get("total_files", len(self.file_tree))
        )
        total_py = _sanitize_comment_text(
            self.summary_metadata.get("total_python_files", len(self.ast_signatures))
        )
        total_lines = _sanitize_comment_text(
            self.summary_metadata.get("total_lines", 0)
        )
        lines.append(
            f"# Total Files: {total_files} | "
            f"Python Files: {total_py} | "
            f"Total Lines: {total_lines}"
        )

        langs = self.summary_metadata.get("languages", {})
        if langs:
            lang_str = ", ".join(
                f"{_sanitize_comment_text(k)}: {_sanitize_comment_text(v)}"
                for k, v in sorted(langs.items())
            )
            lines.append(f"# Languages: {lang_str}")

        lines.append("# ==============================================================================")
        lines.append("# FILE TREE:")
        for path in self.file_tree:
            lines.append(f"# - {_sanitize_comment_text(path)}")
        lines.append("# ==============================================================================")
        lines.append("")

        for rel_path in sorted(self.ast_signatures.keys()):
            sig = self.ast_signatures[rel_path].strip()
            clean_rel = _sanitize_comment_text(rel_path)
            lines.append(f"# --- MODULE: {clean_rel} ---")
            if sig:
                lines.append(sig)
            else:
                lines.append("# (Empty module)")
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def to_markdown(self) -> str:
        """Renders outline as human-readable Markdown with code blocks."""
        md: List[str] = []
        root_name = self.workspace_root.name if self.workspace_root else "workspace"
        md.append(f"# Workspace Architecture Outline: `{root_name}`\n")
        md.append(f"- **Total Files**: {self.summary_metadata.get('total_files', len(self.file_tree))}")
        md.append(f"- **Python Modules**: {self.summary_metadata.get('total_python_files', len(self.ast_signatures))}")
        md.append(f"- **Total Source Lines**: {self.summary_metadata.get('total_lines', 0)}\n")

        md.append("## File Tree\n```text")
        for path in self.file_tree:
            md.append(f"{path}")
        md.append("```\n")

        md.append("## Python Module Interfaces (Zero-Knowledge AST Signatures)\n")
        for rel_path in sorted(self.ast_signatures.keys()):
            sig = self.ast_signatures[rel_path].strip()
            md.append(f"### `{rel_path}`\n```python")
            md.append(sig if sig else "# (Empty module)")
            md.append("```\n")

        return "\n".join(md)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the outline to a JSON-compatible dictionary."""
        return {
            "workspace_root": str(self.workspace_root),
            "file_tree": list(self.file_tree),
            "ast_signatures": dict(self.ast_signatures),
            "summary_metadata": dict(self.summary_metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkspaceOutline":
        """Constructs a WorkspaceOutline from a dictionary."""
        return cls(
            workspace_root=Path(data["workspace_root"]),
            file_tree=list(data.get("file_tree", [])),
            ast_signatures=dict(data.get("ast_signatures", {})),
            summary_metadata=dict(data.get("summary_metadata", {})),
        )

    def get_signature(self, rel_path: str) -> Optional[str]:
        """Retrieves AST signature for a specific relative file path."""
        norm_path = Path(rel_path).as_posix()
        return self.ast_signatures.get(norm_path)


class WorkspaceScanner:
    """Recursively scans and ingests workspace directory trees.

    Enforces:
    - Path traversal protection (prunes symlinks escaping workspace root).
    - Automatic exclusion of VCS directories, caches, virtualenvs, and build outputs.
    - Respects root .gitignore patterns.
    - Zero-knowledge multi-file AST signature extraction for Python modules.
    """

    DEFAULT_IGNORE_PATTERNS: Set[str] = {
        # Version Control
        ".git",
        ".svn",
        ".hg",
        # Virtual Environments
        ".venv",
        "venv",
        "env",
        ".env",
        ".tox",
        ".nox",
        # Caches & Bytecode
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".hypothesis",
        ".coverage",
        "htmlcov",
        # Build & Distribution
        "build",
        "dist",
        "*.egg-info",
        "*.egg",
        ".eggs",
        "wheels",
        # IDEs & System
        ".idea",
        ".vscode",
        ".cursor",
        "*.swp",
        "*~",
        ".DS_Store",
        # Dependencies
        "node_modules",
        "vendor",
        "bower_components",
        # Secrets and credentials
        ".env*",
        "*.pem",
        "*.key",
        "id_rsa*",
        "id_dsa*",
        "id_ed25519*",
        "credentials.json",
        # Compiled bytecode
        "*.pyc",
        "*.pyo",
        "*.pyd",
    }

    EXTENSION_LANGUAGE_MAP: Dict[str, str] = {
        ".py": "python",
        ".pyi": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".md": "markdown",
        ".rst": "restructuredtext",
        ".html": "html",
        ".css": "css",
        ".sh": "bash",
        ".bash": "bash",
        ".sql": "sql",
        ".dockerfile": "dockerfile",
    }

    def __init__(
        self,
        workspace_root: Union[str, Path],
        ignore_patterns: Optional[Union[List[str], Set[str]]] = None,
        custom_ignore_patterns: Optional[Union[List[str], Set[str]]] = None,
        max_file_size_bytes: int = 1_000_000,
        max_files: int = 10_000,
        respect_gitignore: bool = True,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        if not self.workspace_root.exists():
            raise FileNotFoundError(
                f"Workspace directory does not exist: {self.workspace_root}"
            )
        if self.workspace_root.is_file() or not self.workspace_root.is_dir():
            raise WorkspaceNotADirectoryError(
                f"Workspace path is not a directory: {self.workspace_root}"
            )

        self.ignore_patterns: Set[str] = set(self.DEFAULT_IGNORE_PATTERNS)
        if ignore_patterns:
            self.ignore_patterns.update(ignore_patterns)
        if custom_ignore_patterns:
            self.ignore_patterns.update(custom_ignore_patterns)

        self.max_file_size_bytes = max_file_size_bytes
        self.max_files = max_files
        self.respect_gitignore = respect_gitignore
        self._gitignore_patterns: List[str] = (
            self._load_gitignore() if respect_gitignore else []
        )
        self.stripper = WorkspaceASTStripper(fallback_on_error=True)

    def _load_gitignore(self) -> List[str]:
        """Loads and parses .gitignore rules from the workspace root if present."""
        gi_path = self.workspace_root / ".gitignore"
        patterns: List[str] = []
        if gi_path.is_file():
            try:
                for line in gi_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        patterns.append(line)
            except Exception:
                pass
        return patterns

    def should_ignore(self, rel_path: Union[str, Path], is_dir: bool = False) -> bool:
        """Determines if a relative path should be excluded based on ignore patterns."""
        p = Path(rel_path)
        name = p.name
        posix_str = p.as_posix()

        # Check path components
        for part in p.parts:
            for pattern in self.ignore_patterns:
                if fnmatch.fnmatch(part, pattern):
                    return True

        # Check full filename or relative path
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(posix_str, pattern):
                return True

        # Check .gitignore patterns
        for pattern in self._gitignore_patterns:
            clean_pat = pattern.rstrip("/")
            if fnmatch.fnmatch(name, clean_pat) or fnmatch.fnmatch(posix_str, clean_pat):
                return True
            if fnmatch.fnmatch(posix_str, f"*{clean_pat}*"):
                return True

        return False

    def is_ignored(self, name_or_path: Union[str, Path]) -> bool:
        """Alias for should_ignore to satisfy external caller contracts."""
        return self.should_ignore(name_or_path)

    def is_python_file(self, path: Union[str, Path]) -> bool:
        """Returns True if the path corresponds to a Python source file."""
        return Path(path).suffix.lower() in (".py", ".pyi")

    def crawl(self) -> Tuple[List[str], List[str]]:
        """Traverses workspace directory tree and returns (python_files, non_python_files)."""
        python_files: List[str] = []
        non_python_files: List[str] = []

        for root, dirs, files in os.walk(
            self.workspace_root, topdown=True, followlinks=False
        ):
            current_dir = Path(root)
            rel_dir = current_dir.relative_to(self.workspace_root)

            # In-place directory pruning to avoid descending into ignored trees
            dirs[:] = [
                d
                for d in dirs
                if not self.should_ignore(
                    (rel_dir / d) if rel_dir != Path(".") else Path(d), is_dir=True
                )
            ]

            for file_name in files:
                rel_file_path = (
                    (rel_dir / file_name) if rel_dir != Path(".") else Path(file_name)
                )

                if self.should_ignore(rel_file_path, is_dir=False):
                    continue

                abs_file_path = current_dir / file_name

                # Symlink containment check
                if abs_file_path.is_symlink():
                    try:
                        resolved = abs_file_path.resolve()
                        root_str = str(self.workspace_root)
                        resolved_str = str(resolved)
                        if not (
                            resolved_str == root_str
                            or resolved_str.startswith(root_str + os.sep)
                        ):
                            continue
                    except Exception:
                        continue

                # Regular file check (exclude FIFOs, sockets, device nodes) and file size check
                try:
                    stat_res = abs_file_path.stat()
                    if not stat.S_ISREG(stat_res.st_mode):
                        continue
                    if stat_res.st_size > self.max_file_size_bytes:
                        continue
                except OSError:
                    continue

                posix_rel = rel_file_path.as_posix()
                if self.is_python_file(file_name):
                    python_files.append(posix_rel)
                else:
                    non_python_files.append(posix_rel)

        return sorted(python_files), sorted(non_python_files)

    def scan(self) -> WorkspaceOutline:
        """Executes full workspace scan and generates a verified WorkspaceOutline."""
        file_tree: List[str] = []
        ast_signatures: Dict[str, str] = {}
        languages: Dict[str, int] = {}
        total_lines = 0
        total_python_files = 0
        excluded_files_count = 0
        parse_errors = 0

        for root, dirs, files in os.walk(
            self.workspace_root, topdown=True, followlinks=False
        ):
            current_dir = Path(root)
            rel_dir = current_dir.relative_to(self.workspace_root)

            # In-place directory pruning
            dirs[:] = [
                d
                for d in dirs
                if not self.should_ignore(
                    (rel_dir / d) if rel_dir != Path(".") else Path(d), is_dir=True
                )
            ]

            for file_name in files:
                rel_file_path = (
                    (rel_dir / file_name) if rel_dir != Path(".") else Path(file_name)
                )

                if self.should_ignore(rel_file_path, is_dir=False):
                    excluded_files_count += 1
                    continue

                abs_file_path = current_dir / file_name

                # Symlink containment check
                if abs_file_path.is_symlink():
                    try:
                        resolved = abs_file_path.resolve()
                        root_str = str(self.workspace_root)
                        resolved_str = str(resolved)
                        if not (
                            resolved_str == root_str
                            or resolved_str.startswith(root_str + os.sep)
                        ):
                            excluded_files_count += 1
                            continue
                    except Exception:
                        excluded_files_count += 1
                        continue

                # Regular file check (exclude FIFOs, sockets, device nodes) and file size check
                try:
                    stat_res = abs_file_path.stat()
                    if not stat.S_ISREG(stat_res.st_mode):
                        excluded_files_count += 1
                        continue
                    if stat_res.st_size > self.max_file_size_bytes:
                        excluded_files_count += 1
                        continue
                except OSError:
                    excluded_files_count += 1
                    continue

                posix_rel = rel_file_path.as_posix()
                file_tree.append(posix_rel)

                suffix = abs_file_path.suffix.lower()
                lang = self.EXTENSION_LANGUAGE_MAP.get(suffix, "other")
                languages[lang] = languages.get(lang, 0) + 1

                if self.is_python_file(file_name):
                    total_python_files += 1
                    try:
                        content = abs_file_path.read_text(
                            encoding="utf-8", errors="replace"
                        )
                        total_lines += len(content.splitlines())
                        stripped_sig = self.stripper.strip_module(
                            content, file_path=posix_rel
                        )
                        ast_signatures[posix_rel] = stripped_sig
                        if stripped_sig.startswith("# [SyntaxError"):
                            parse_errors += 1
                    except Exception as err:
                        parse_errors += 1
                        clean_err = _sanitize_comment_text(err)
                        ast_signatures[posix_rel] = f"# [SyntaxError in {posix_rel}: {clean_err}]"
                else:
                    try:
                        if stat_res.st_size < 200_000:
                            content = abs_file_path.read_text(
                                encoding="utf-8", errors="ignore"
                            )
                            total_lines += len(content.splitlines())
                    except Exception:
                        pass

                if len(file_tree) >= self.max_files:
                    break

            if len(file_tree) >= self.max_files:
                break

        file_tree.sort()

        summary_metadata: Dict[str, Any] = {
            "total_files": len(file_tree),
            "total_python_files": total_python_files,
            "total_lines": total_lines,
            "languages": languages,
            "scanned_timestamp": time.time(),
            "excluded_files_count": excluded_files_count,
            "parse_errors": parse_errors,
        }

        return WorkspaceOutline(
            workspace_root=self.workspace_root,
            file_tree=file_tree,
            ast_signatures=ast_signatures,
            summary_metadata=summary_metadata,
        )
