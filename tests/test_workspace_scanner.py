"""Comprehensive test suite for Milestone 1: Workspace Ingestion & Zero-Knowledge AST Scanner.

Covers:
- Feature F1: CLI Ingestion Flags (--goal, --workspace, --retry-budget, and backward compatibility)
- Feature F2: Workspace Directory Scanner (crawling, ignore filters, .gitignore, symlink protection)
- Feature F3: Multi-File Zero-Knowledge AST Stripper (headers, parameter defaults, docstring removal)
- Feature F4: Pre-Flight Privacy Verifier (verify_cloud_payload single/multi-arg, verify_workspace_outline)
"""

import ast
import os
from pathlib import Path
import signal
import time
from typing import Dict, List, Tuple
import pytest

from antigravity_hybrid_orchestrator.cli import build_parser, main
from antigravity_hybrid_orchestrator.privacy.verifier import (
    FORBIDDEN_LOGIC_NODES,
    SecurityInvariantViolationError,
    verify_cloud_payload,
)
from antigravity_hybrid_orchestrator.privacy.workspace_privacy import (
    WorkspaceASTStripper,
    WorkspaceASTTransformer,
    verify_workspace_outline,
    verify_workspace_outline_privacy,
)
from antigravity_hybrid_orchestrator.workspace.scanner import (
    WorkspaceNotADirectoryError,
    WorkspaceOutline,
    WorkspaceScanner,
)


# ==============================================================================
# TEST FIXTURES
# ==============================================================================


@pytest.fixture
def clean_workspace(tmp_path: Path) -> Path:
    """Creates a temporary realistic clean Python project workspace."""
    ws = tmp_path / "clean_repo"
    ws.mkdir()

    (ws / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    (ws / "README.md").write_text("# Demo Project\nDocumentation.\n", encoding="utf-8")

    src = ws / "src"
    src.mkdir()
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "main.py").write_text(
        "import os\nfrom typing import Optional\n\n"
        "class App:\n"
        "    def __init__(self, name: str = 'app'):\n"
        "        self.name = name\n\n"
        "    def run(self) -> int:\n"
        "        print('Running', self.name)\n"
        "        return 0\n",
        encoding="utf-8",
    )

    utils = src / "utils"
    utils.mkdir()
    (utils / "__init__.py").write_text("", encoding="utf-8")
    (utils / "math_helper.py").write_text(
        "def add(a: int, b: int = 10) -> int:\n"
        "    result = a + b\n"
        "    return result\n",
        encoding="utf-8",
    )

    tests = ws / "tests"
    tests.mkdir()
    (tests / "test_main.py").write_text(
        "from src.main import App\n\n"
        "def test_app():\n"
        "    app = App()\n"
        "    assert app.run() == 0\n",
        encoding="utf-8",
    )
    return ws


@pytest.fixture
def adversarial_workspace(tmp_path: Path) -> Tuple[Path, List[str]]:
    """Creates an adversarial workspace containing secrets, docstrings, loops, and control flow."""
    ws = tmp_path / "adversarial_repo"
    ws.mkdir()

    secrets = [
        "SUPER_CONFIDENTIAL_JWT_KEY_9999",
        "AKIAIOSFODNN7EXAMPLE_SECRET",
        "postgres://admin:topsecretpassword@db.internal:5432/prod",
        "PROPRIETARY_ALGORITHM_SALT_XYZ",
    ]

    (ws / "config.py").write_text(
        f'"""Proprietary module docstring with internal details."""\n\n'
        f'JWT_SECRET = "{secrets[0]}"\n'
        f'AWS_KEY = "{secrets[1]}"\n'
        f"DEBUG: bool = False\n",
        encoding="utf-8",
    )

    (ws / "service.py").write_text(
        f"import hmac\nimport hashlib\n\n"
        f'DATABASE_URL = "{secrets[2]}"\n\n'
        f'def authenticate(user: str, token: str, secret: str = "{secrets[3]}") -> bool:\n'
        f'    """Secret authentication function docstring."""\n'
        f"    digest = hmac.new(secret.encode(), user.encode(), hashlib.sha256).hexdigest()\n"
        f"    for _ in range(10):\n"
        f"        if digest == token:\n"
        f"            return True\n"
        f"    return False\n",
        encoding="utf-8",
    )

    return ws, secrets


@pytest.fixture
def workspace_with_ignored_and_binary(tmp_path: Path) -> Path:
    """Creates a workspace with ignored directories and binary files."""
    ws = tmp_path / "mixed_repo"
    ws.mkdir()

    # Clean files
    (ws / "app.py").write_text("def start():\n    pass\n", encoding="utf-8")

    # Ignored directories
    (ws / ".git").mkdir()
    (ws / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    (ws / ".venv").mkdir()
    (ws / ".venv" / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")

    (ws / "__pycache__").mkdir()
    (ws / "__pycache__" / "app.cpython-312.pyc").write_bytes(b"\x00\x01\x02\x03\x04")

    (ws / "node_modules").mkdir()
    (ws / "node_modules" / "package.json").write_text("{}", encoding="utf-8")

    # Binary file in project
    (ws / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

    # Invalid UTF-8 bytes
    (ws / "corrupt_data.bin").write_bytes(b"\xff\xfe\x00\x00\xaa\xbb")

    return ws


# ==============================================================================
# FEATURE F1: CLI INGESTION FLAGS
# ==============================================================================


class TestF1CLIIngestionFlags:
    """Tests command-line parsing extensions and backward compatibility."""

    def test_cli_f1_goal_parsing(self):
        """Tier 1: Verifies --goal parses string correctly."""
        parser = build_parser()
        args = parser.parse_args(["--goal", "Implement Fibonacci API"])
        assert args.goal == "Implement Fibonacci API"

    def test_cli_f1_workspace_parsing(self):
        """Tier 1: Verifies --workspace parses target path string."""
        parser = build_parser()
        args = parser.parse_args(["--workspace", "/home/user/my_project"])
        assert str(args.workspace) == "/home/user/my_project"

    def test_cli_f1_retry_budget_default(self):
        """Tier 1: Verifies default retry limit is 5 when omitted."""
        parser = build_parser()
        args = parser.parse_args([])
        assert args.retry_budget == 5

    def test_cli_f1_retry_budget_custom(self):
        """Tier 1: Verifies custom integer retry budget is parsed."""
        parser = build_parser()
        args = parser.parse_args(["--retry-budget", "10"])
        assert args.retry_budget == 10

    def test_cli_f1_combined_flags_and_backward_compat(self):
        """Tier 1: Verifies M1 flags coexist cleanly with legacy security gauntlet flags."""
        parser = build_parser()
        args = parser.parse_args([
            "--goal", "Refactor auth",
            "--workspace", "./src",
            "--mode", "mock",
            "--target", "billing",
            "--retry-budget", "3",
            "--no-hud",
            "--reset",
            "--refresh-rate", "12.0",
        ])
        assert args.goal == "Refactor auth"
        assert args.workspace == "./src"
        assert args.mode == "mock"
        assert args.target == "billing"
        assert args.retry_budget == 3
        assert args.no_hud is True
        assert args.reset is True
        assert args.refresh_rate == 12.0

    def test_cli_f1_empty_goal_string(self):
        """Tier 2: Empty goal string is preserved without error."""
        parser = build_parser()
        args = parser.parse_args(["--goal", ""])
        assert args.goal == ""

    def test_cli_f1_missing_goal_value(self):
        """Tier 2: Dangling --goal flag exits with error code 2."""
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--goal"])
        assert exc_info.value.code == 2

    def test_cli_f1_retry_budget_negative_or_zero(self):
        """Tier 2: Validates retry budget boundaries (0 allowed, negative rejected in main)."""
        parser = build_parser()
        args = parser.parse_args(["--retry-budget", "0"])
        assert args.retry_budget == 0

        # main() validates non-negative retry budget
        exit_code = main(["--retry-budget", "-1", "--mode", "mock"])
        assert exit_code == 2

    def test_cli_f1_retry_budget_invalid_type(self):
        """Tier 2: Non-integer retry budget causes SystemExit code 2."""
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--retry-budget", "five"])
        assert exc_info.value.code == 2

    def test_cli_f1_workspace_with_spaces_and_special_chars(self):
        """Tier 2: Path with spaces, parentheses, hashes, and symbols is preserved."""
        parser = build_parser()
        path_str = "/tmp/path with spaces/(v1.0)/#repo"
        args = parser.parse_args(["--workspace", path_str])
        assert str(args.workspace) == path_str


# ==============================================================================
# FEATURE F2: WORKSPACE DIRECTORY SCANNER
# ==============================================================================


class TestF2WorkspaceScanner:
    """Tests recursive crawling, directory pruning, and ignore patterns."""

    def test_scanner_f2_clean_project_traversal(self, clean_workspace: Path):
        """Tier 1: Crawls hierarchical repository structure cleanly."""
        scanner = WorkspaceScanner(clean_workspace)
        outline = scanner.scan()
        expected = [
            "README.md",
            "pyproject.toml",
            "src/__init__.py",
            "src/main.py",
            "src/utils/__init__.py",
            "src/utils/math_helper.py",
            "tests/test_main.py",
        ]
        assert outline.file_tree == sorted(expected)

    def test_scanner_f2_default_ignore_patterns(self, workspace_with_ignored_and_binary: Path):
        """Tier 1: Prunes .git, .venv, __pycache__, node_modules from crawl."""
        scanner = WorkspaceScanner(workspace_with_ignored_and_binary)
        outline = scanner.scan()
        for f in outline.file_tree:
            assert not f.startswith(".git")
            assert not f.startswith(".venv")
            assert not f.startswith("__pycache__")
            assert not f.startswith("node_modules")

    def test_scanner_f2_python_vs_non_python_separation(self, clean_workspace: Path):
        """Tier 1: Correctly classifies Python vs non-Python assets."""
        scanner = WorkspaceScanner(clean_workspace)
        outline = scanner.scan()
        assert outline.summary_metadata["total_files"] == 7
        assert outline.summary_metadata["total_python_files"] == 5
        assert len(outline.ast_signatures) == 5
        assert "README.md" not in outline.ast_signatures
        assert "pyproject.toml" not in outline.ast_signatures

    def test_scanner_f2_summary_metadata_counts(self, clean_workspace: Path):
        """Tier 1: Computes line counts and language breakdown accurately."""
        scanner = WorkspaceScanner(clean_workspace)
        outline = scanner.scan()
        assert outline.summary_metadata["total_lines"] > 0
        assert outline.summary_metadata["languages"]["python"] == 5
        assert outline.summary_metadata["languages"]["markdown"] == 1
        assert outline.summary_metadata["languages"]["toml"] == 1

    def test_scanner_f2_deterministic_ordering(self, tmp_path: Path):
        """Tier 1: Multiple repeated scans return identical alphabetically sorted file_tree."""
        ws = tmp_path / "order_ws"
        ws.mkdir()
        for f in ["z.py", "a.py", "m.py", "b.py"]:
            (ws / f).write_text("x = 1\n", encoding="utf-8")

        scanner = WorkspaceScanner(ws)
        out1 = scanner.scan()
        out2 = scanner.scan()
        assert out1.file_tree == ["a.py", "b.py", "m.py", "z.py"]
        assert out1.file_tree == out2.file_tree

    def test_scanner_f2_empty_directory(self, tmp_path: Path):
        """Tier 2: Empty directory yields empty file_tree and zero files."""
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        scanner = WorkspaceScanner(empty_dir)
        outline = scanner.scan()
        assert outline.file_tree == []
        assert outline.ast_signatures == {}
        assert outline.summary_metadata["total_files"] == 0

    def test_scanner_f2_nonexistent_directory(self, tmp_path: Path):
        """Tier 2: Non-existent directory raises FileNotFoundError."""
        missing = tmp_path / "missing_dir"
        with pytest.raises(FileNotFoundError):
            WorkspaceScanner(missing)

    def test_scanner_f2_workspace_is_file_not_dir(self, tmp_path: Path):
        """Tier 2: Regular file passed as workspace root raises error."""
        f = tmp_path / "file.py"
        f.write_text("x = 1\n", encoding="utf-8")
        with pytest.raises((ValueError, NotADirectoryError)):
            WorkspaceScanner(f)

    def test_scanner_f2_deeply_nested_subdirectories(self, tmp_path: Path):
        """Tier 2: Deeply nested directory (25 levels) is crawled without error."""
        nested = tmp_path / "nested"
        curr = nested
        for i in range(25):
            curr = curr / f"level_{i}"
        curr.mkdir(parents=True)
        (curr / "leaf.py").write_text("def leaf(): pass\n", encoding="utf-8")

        scanner = WorkspaceScanner(nested)
        outline = scanner.scan()
        assert len(outline.file_tree) == 1
        assert outline.file_tree[0].endswith("leaf.py")

    def test_scanner_f2_symlink_directory_loop(self, tmp_path: Path):
        """Tier 2: Circular symlink loop terminates without hang or infinite recursion."""
        ws = tmp_path / "loop_ws"
        ws.mkdir()
        sub = ws / "sub"
        sub.mkdir()
        (sub / "mod.py").write_text("x = 1\n", encoding="utf-8")

        try:
            os.symlink(str(sub), str(sub / "loop"))
        except OSError:
            pytest.skip("Symlinks not supported on this filesystem")

        scanner = WorkspaceScanner(ws)
        start_time = time.time()
        outline = scanner.scan()
        elapsed = time.time() - start_time

        assert elapsed < 1.0
        assert "sub/mod.py" in outline.file_tree

    def test_scanner_f2_symlink_pointing_outside_workspace(self, tmp_path: Path):
        """Tier 2: Symlink pointing outside workspace root is not traversed."""
        ext = tmp_path / "external_secret"
        ext.mkdir()
        (ext / "secret.py").write_text("SECRET = 123\n", encoding="utf-8")

        ws = tmp_path / "internal_ws"
        ws.mkdir()
        (ws / "main.py").write_text("x = 1\n", encoding="utf-8")

        try:
            os.symlink(str(ext), str(ws / "ext_link"))
        except OSError:
            pytest.skip("Symlinks not supported")

        scanner = WorkspaceScanner(ws)
        outline = scanner.scan()
        assert "ext_link/secret.py" not in outline.file_tree
        assert outline.file_tree == ["main.py"]

    def test_scanner_f2_binary_files_handling(self, workspace_with_ignored_and_binary: Path):
        """Tier 2: Binary and non-UTF8 files do not raise UnicodeDecodeError."""
        scanner = WorkspaceScanner(workspace_with_ignored_and_binary)
        outline = scanner.scan()
        assert "logo.png" in outline.file_tree
        assert "corrupt_data.bin" in outline.file_tree
        assert "logo.png" not in outline.ast_signatures
        assert "corrupt_data.bin" not in outline.ast_signatures

    def test_scanner_f2_only_ignored_files(self, tmp_path: Path):
        """Tier 2: Workspace containing only ignored directories yields 0 files."""
        ws = tmp_path / "ignored_only"
        ws.mkdir()
        git_dir = ws / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("git config\n", encoding="utf-8")
        pycache = ws / "__pycache__"
        pycache.mkdir()
        (pycache / "app.pyc").write_bytes(b"bytecode")

        scanner = WorkspaceScanner(ws)
        outline = scanner.scan()
        assert outline.file_tree == []
        assert outline.summary_metadata["total_files"] == 0

    def test_scanner_f2_respects_gitignore(self, tmp_path: Path):
        """Tier 1: .gitignore rules filter matched files and directories."""
        ws = tmp_path / "gi_ws"
        ws.mkdir()
        (ws / ".gitignore").write_text("*.log\nsecret_dir/\n# comment\n", encoding="utf-8")
        (ws / "app.py").write_text("x = 1\n", encoding="utf-8")
        (ws / "debug.log").write_text("log info\n", encoding="utf-8")
        sec_dir = ws / "secret_dir"
        sec_dir.mkdir()
        (sec_dir / "keys.json").write_text("{}", encoding="utf-8")

        scanner = WorkspaceScanner(ws, respect_gitignore=True)
        outline = scanner.scan()
        assert "app.py" in outline.file_tree
        assert "debug.log" not in outline.file_tree
        assert "secret_dir/keys.json" not in outline.file_tree

    def test_scanner_f2_crawl_method(self, clean_workspace: Path):
        """Tier 1: scanner.crawl() separates Python and non-Python relative files."""
        scanner = WorkspaceScanner(clean_workspace)
        py_files, non_py = scanner.crawl()
        assert len(py_files) == 5
        assert len(non_py) == 2

    def test_scanner_skips_fifos_and_special_files(self, tmp_path: Path):
        """Tier 2: Named pipes (FIFOs) and special non-regular files do not hang scanner."""
        if not hasattr(os, "mkfifo"):
            pytest.skip("Named pipes (mkfifo) not supported on this platform")

        ws = tmp_path / "fifo_ws"
        ws.mkdir()
        (ws / "main.py").write_text("def run(): pass\n", encoding="utf-8")
        fifo_path = ws / "pipe.py"

        try:
            os.mkfifo(fifo_path)
        except OSError:
            pytest.skip("mkfifo failed or not supported in environment")

        scanner = WorkspaceScanner(ws)

        # Protect against indefinite hang using signal.alarm with a BaseException
        # so that it cannot be swallowed by internal 'except Exception' handlers.
        class PipeHangTimeoutError(BaseException):
            pass

        def alarm_handler(signum, frame):
            raise PipeHangTimeoutError("WorkspaceScanner hung while attempting to read FIFO pipe")

        prev_handler = signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(2)
        try:
            outline = scanner.scan()
            assert "main.py" in outline.file_tree
            assert "main.py" in outline.ast_signatures
            assert "pipe.py" not in outline.ast_signatures
            assert "pipe.py" not in outline.file_tree

            py_files, non_py = scanner.crawl()
            assert "pipe.py" not in py_files
            assert "pipe.py" not in non_py
        except PipeHangTimeoutError:
            pytest.fail("WorkspaceScanner hung on FIFO named pipe (blocking read_text)!")
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev_handler)
            try:
                os.unlink(fifo_path)
            except OSError:
                pass


# ==============================================================================
# FEATURE F3: MULTI-FILE ZERO-KNOWLEDGE AST STRIPPER
# ==============================================================================


class TestF3WorkspaceASTStripper:
    """Tests multi-file AST transformation, signature extraction, and zero-knowledge stripping."""

    def test_ast_stripper_f3_multi_module_signatures(self, clean_workspace: Path):
        """Tier 1: Extracts signatures across multiple workspace Python modules."""
        scanner = WorkspaceScanner(clean_workspace)
        outline = scanner.scan()
        assert "src/main.py" in outline.ast_signatures
        assert "src/utils/math_helper.py" in outline.ast_signatures
        assert "tests/test_main.py" in outline.ast_signatures

    def test_ast_stripper_f3_function_body_replacement_ellipsis(self):
        """Tier 1: Replaces function bodies with single Ellipsis statement."""
        stripper = WorkspaceASTStripper()
        code = (
            "def compute_hash(val: str) -> str:\n"
            '    salt = "secret"\n'
            "    computed = val + salt\n"
            "    return hashlib.sha256(computed).hexdigest()\n"
        )
        stripped = stripper.strip_source(code, "test.py")
        assert "def compute_hash(val: str) -> str:" in stripped
        assert "..." in stripped
        assert "salt" not in stripped
        assert "hashlib" not in stripped
        assert "return" not in stripped

    def test_ast_stripper_f3_docstring_and_constant_removal(self):
        """Tier 1: Eliminates 100% of module, class, and method docstrings."""
        stripper = WorkspaceASTStripper()
        code = (
            '"""Module secret documentation."""\n\n'
            "class App:\n"
            '    """Class secret documentation."""\n'
            "    def run(self) -> None:\n"
            '        """Method secret documentation."""\n'
            "        pass\n"
        )
        stripped = stripper.strip_source(code, "test.py")
        assert "Module secret documentation" not in stripped
        assert "Class secret documentation" not in stripped
        assert "Method secret documentation" not in stripped

    def test_ast_stripper_f3_argument_default_sanitization(self):
        """Tier 1: Default string literals >= 5 chars sanitized to ...; short/primitives kept."""
        stripper = WorkspaceASTStripper()
        code = (
            'def connect(timeout: int = 30, api_key: str = "SUPER_SECRET_KEY_12345", '
            'debug: bool = True, mode: str = "r") -> None:\n'
            "    pass\n"
        )
        stripped = stripper.strip_source(code, "test.py")
        assert "SUPER_SECRET_KEY_12345" not in stripped
        assert "api_key: str = ..." in stripped or "api_key: str=..." in stripped
        assert "timeout: int = 30" in stripped or "timeout: int=30" in stripped
        assert "debug: bool = True" in stripped or "debug: bool=True" in stripped

    def test_ast_stripper_f3_outline_to_prompt_string(self, clean_workspace: Path):
        """Tier 1: to_prompt_string() returns valid Python AST formatted with comment headers."""
        scanner = WorkspaceScanner(clean_workspace)
        outline = scanner.scan()
        prompt_str = outline.to_prompt_string()
        assert isinstance(prompt_str, str)
        assert len(prompt_str) > 0

        # Entire prompt string is valid Python AST
        tree = ast.parse(prompt_str)
        assert tree is not None

    def test_ast_stripper_f3_syntax_error_in_workspace_file(self, tmp_path: Path):
        """Tier 2: SyntaxError in one file does not crash scanner; records error in metadata."""
        ws = tmp_path / "broken_ws"
        ws.mkdir()
        (ws / "good.py").write_text("def good(): pass\n", encoding="utf-8")
        (ws / "broken.py").write_text("def broken(\n", encoding="utf-8")

        scanner = WorkspaceScanner(ws)
        outline = scanner.scan()
        assert "good.py" in outline.ast_signatures
        assert "broken.py" in outline.ast_signatures
        assert outline.summary_metadata["parse_errors"] >= 1
        assert "# [SyntaxError" in outline.ast_signatures["broken.py"]

    def test_ast_stripper_f3_empty_python_file(self):
        """Tier 2: Empty Python file produces empty signature without error."""
        stripper = WorkspaceASTStripper()
        stripped = stripper.strip_source("", "__init__.py")
        assert stripped == ""

    def test_ast_stripper_f3_unicode_and_encoding_headers(self):
        """Tier 2: Handles coding cookies, unicode identifiers, and emojis."""
        stripper = WorkspaceASTStripper()
        code = (
            "# -*- coding: utf-8 -*-\n"
            'def café_latte(emoji: str = "☕") -> str:\n'
            '    return "delicious"\n'
        )
        stripped = stripper.strip_source(code, "unicode.py")
        assert "café_latte" in stripped
        assert "..." in stripped
        tree = ast.parse(stripped)
        assert tree is not None

    def test_ast_stripper_f3_modern_python312_syntax(self):
        """Tier 2: Modern syntax (type aliases, match-case) stripped cleanly."""
        stripper = WorkspaceASTStripper()
        code = (
            "type Vector[T] = list[T]\n\n"
            "def handle(val: int) -> str:\n"
            "    match val:\n"
            "        case 1:\n"
            '            return "one"\n'
            "        case _:\n"
            '            return "other"\n'
        )
        stripped = stripper.strip_source(code, "modern.py")
        assert "def handle(val: int) -> str:" in stripped
        assert "..." in stripped
        assert "match val:" not in stripped
        assert "case 1:" not in stripped

    def test_ast_stripper_f3_heavily_decorated_functions(self):
        """Tier 2: Decorator arguments with string literals sanitized."""
        stripper = WorkspaceASTStripper()
        code = (
            '@router.post("/api/v1/login", response_model=UserResponse)\n'
            '@rate_limit(secret="SECRET_VAL_12345")\n'
            "def login():\n"
            "    pass\n"
        )
        stripped = stripper.strip_source(code, "service.py")
        assert "SECRET_VAL_12345" not in stripped
        assert "/api/v1/login" not in stripped
        assert "@router.post" in stripped
        assert "@rate_limit" in stripped

    def test_ast_stripper_f3_forbidden_control_flow_stripping(self):
        """Tier 2: Module-level control flow (if, while, try) completely removed."""
        stripper = WorkspaceASTStripper()
        code = (
            "import os\n\n"
            'if __name__ == "__main__":\n'
            "    print('Starting')\n\n"
            "while True:\n"
            "    break\n\n"
            "try:\n"
            "    os.getcwd()\n"
            "except Exception:\n"
            "    pass\n"
        )
        stripped = stripper.strip_source(code, "main.py")
        assert "import os" in stripped
        assert "if __name__" not in stripped
        assert "while True" not in stripped
        assert "try:" not in stripped

    def test_ast_stripper_literal_subscript_sanitization(self):
        """Tier 2: Literal and Annotated subscripts with secret string literals are sanitized."""
        stripper = WorkspaceASTStripper()
        src = (
            "from typing import Literal, Annotated\n\n"
            "class Config:\n"
            '    DATABASE_URL: Annotated[str, "SECRET_CONN_STRING"] = "postgres://..."\n\n'
            'def login(token: Literal["SECRET_12345"]) -> Literal["ADMIN_SECRET_KEY"]:\n'
            '    return "granted"\n'
        )
        stripped = stripper.strip_source(src, "auth.py")

        # Invariant 1: Secrets must not appear anywhere in stripped AST output
        assert "SECRET_12345" not in stripped, "Secret literal leaked in parameter annotation!"
        assert "ADMIN_SECRET_KEY" not in stripped, "Secret literal leaked in return annotation!"
        assert "SECRET_CONN_STRING" not in stripped, "Secret literal leaked in Annotated metadata!"

        # Invariant 2: Resulting stripped code is valid Python AST
        tree = ast.parse(stripped)
        assert tree is not None

        # Invariant 3: Secrets must not appear in to_prompt_string()
        outline = WorkspaceOutline(
            workspace_root=Path("/repo"),
            file_tree=["auth.py"],
            ast_signatures={"auth.py": stripped},
            summary_metadata={},
        )
        prompt_str = outline.to_prompt_string()
        assert "SECRET_12345" not in prompt_str, "Secret leaked into prompt string!"
        assert "ADMIN_SECRET_KEY" not in prompt_str, "Secret leaked into prompt string!"
        assert "SECRET_CONN_STRING" not in prompt_str, "Secret leaked into prompt string!"

        # Invariant 4: Cloud verifier approves sanitized prompt payload
        assert verify_cloud_payload(prompt_str) is True

    def test_ast_stripper_chained_decorator_sanitization(self):
        """Tier 2: Chained decorators with subscripts sanitize inner literals and do not leak keys."""
        stripper = WorkspaceASTStripper()
        src = (
            '@registry["SECRET_KEY"].register()\n'
            "def handler():\n"
            "    pass\n"
        )
        stripped = stripper.strip_source(src, "api.py")

        # Invariant 1: Inner subscript literal in decorator chain must be sanitized
        assert "SECRET_KEY" not in stripped, "Secret key survived in chained decorator subscript!"

        tree = ast.parse(stripped)
        assert tree is not None

        outline = WorkspaceOutline(
            workspace_root=Path("/repo"),
            file_tree=["api.py"],
            ast_signatures={"api.py": stripped},
        )
        assert "SECRET_KEY" not in outline.to_prompt_string()
        assert verify_cloud_payload(stripped) is True

    def test_ast_stripper_binary_secret_keys(self):
        """Tier 2: Binary keys (>= 5 bytes) are stripped in default arguments based on byte length."""
        stripper = WorkspaceASTStripper()
        # 8-byte binary secret key
        src = 'def decrypt(data: bytes, key: bytes = b"\\xff\\xff\\xff\\xff\\xff\\xff\\xff\\xff"):\n    pass\n'
        stripped = stripper.strip_source(src, "crypto.py")

        # Invariant 1: Binary secret key must be stripped to ...
        assert r"\xff" not in stripped, "Binary key bytes leaked in default parameter!"
        assert "key: bytes = ..." in stripped or "key: bytes=..." in stripped

        # Invariant 2: Short binary default (< 5 bytes) is preserved
        short_src = 'def parse_bytes(b: bytes = b"\\x01\\x02"):\n    pass\n'
        short_stripped = stripper.strip_source(short_src, "crypto.py")
        assert 'b"\\x01\\x02"' in short_stripped or "b'\\x01\\x02'" in short_stripped

        # Invariant 3: Stripped code passes cloud verification against original
        assert verify_cloud_payload(stripped, original_code=src) is True

        # Invariant 4: Leaked binary key in candidate payload fails verification against original
        leaked_stripped = 'def decrypt(data: bytes, key: bytes = b"\\xff\\xff\\xff\\xff\\xff\\xff\\xff\\xff"): ...\n'
        with pytest.raises(SecurityInvariantViolationError):
            verify_cloud_payload(leaked_stripped, original_code=src)


# ==============================================================================
# FEATURE F4: PRE-FLIGHT PRIVACY VERIFIER
# ==============================================================================


class TestF4WorkspacePrivacyVerifier:
    """Tests zero-knowledge pre-flight verification and invariant certification."""

    def test_verifier_f4_clean_workspace_outline_passes(self, clean_workspace: Path):
        """Tier 1: Clean scanned workspace outline passes verify_workspace_outline_privacy."""
        scanner = WorkspaceScanner(clean_workspace)
        outline = scanner.scan()
        assert verify_workspace_outline_privacy(outline) is True

    def test_verifier_f4_zero_forbidden_logic_nodes(self, clean_workspace: Path):
        """Tier 1: Asserts mathematically zero FORBIDDEN_LOGIC_NODES in stripped signatures."""
        scanner = WorkspaceScanner(clean_workspace)
        outline = scanner.scan()
        for rel_path, sig in outline.ast_signatures.items():
            if not sig.strip():
                continue
            tree = ast.parse(sig)
            for node in ast.walk(tree):
                assert not isinstance(node, FORBIDDEN_LOGIC_NODES), (
                    f"Forbidden logic node {type(node).__name__} in {rel_path}"
                )

    def test_verifier_f4_body_length_is_one(self, clean_workspace: Path):
        """Tier 1: Every function/method has body length 1 containing Ellipsis or Pass."""
        scanner = WorkspaceScanner(clean_workspace)
        outline = scanner.scan()
        for rel_path, sig in outline.ast_signatures.items():
            if not sig.strip():
                continue
            tree = ast.parse(sig)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    assert len(node.body) == 1, f"Body length > 1 in {rel_path}:{node.name}"
                    stmt = node.body[0]
                    is_ellipsis = (
                        isinstance(stmt, ast.Expr)
                        and isinstance(stmt.value, ast.Constant)
                        and stmt.value.value is Ellipsis
                    )
                    is_pass = isinstance(stmt, ast.Pass)
                    assert is_ellipsis or is_pass

    def test_verifier_f4_zero_sensitive_literal_leakage(self, adversarial_workspace: Tuple[Path, List[str]]):
        """Tier 1: Secrets in original source files do not leak into stripped code."""
        ws, secrets = adversarial_workspace
        scanner = WorkspaceScanner(ws)
        outline = scanner.scan()

        # Outline verification against original disk files
        assert verify_workspace_outline_privacy(outline) is True

        prompt_str = outline.to_prompt_string()
        for secret in secrets:
            assert secret not in prompt_str

    def test_verifier_f4_prompt_string_privacy_check(self, clean_workspace: Path):
        """Tier 1: verify_cloud_payload accepts to_prompt_string() with single argument."""
        scanner = WorkspaceScanner(clean_workspace)
        outline = scanner.scan()
        prompt_str = outline.to_prompt_string()
        assert verify_cloud_payload(prompt_str) is True

    def test_verifier_f4_detects_leak_in_single_file_of_large_workspace(self, clean_workspace: Path):
        """Tier 2: Leak injected into one file causes whole outline verification to fail."""
        scanner = WorkspaceScanner(clean_workspace)
        outline = scanner.scan()

        # Inject leak in one signature
        outline.ast_signatures["src/main.py"] = (
            "class App:\n"
            "    def run(self) -> int:\n"
            "        return 42\n"
        )

        with pytest.raises(SecurityInvariantViolationError) as exc_info:
            verify_workspace_outline_privacy(outline)
        assert "src/main.py" in str(exc_info.value)

    def test_verifier_f4_detects_forbidden_node_in_decorator(self):
        """Tier 2: Unstripped executable lambda in decorator is rejected."""
        leaked_deco = (
            "@decorator(lambda x: x + 1)\n"
            "def calculate():\n"
            "    ...\n"
        )
        with pytest.raises(SecurityInvariantViolationError) as exc_info:
            verify_cloud_payload(leaked_deco)
        assert "Lambda" in str(exc_info.value) or "Forbidden logic node" in str(exc_info.value)

    def test_verifier_f4_detects_multiline_and_raw_string_secrets(self):
        """Tier 2: Multiline private key literal leak triggers violation."""
        orig_key = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEA0tG\n"
            "-----END RSA PRIVATE KEY-----"
        )
        orig_code = f'RSA_KEY = """{orig_key}"""\n'
        leaked_stripped = f'RSA_KEY = """{orig_key}"""\ndef f(): ...\n'

        with pytest.raises(SecurityInvariantViolationError) as exc_info:
            verify_cloud_payload(leaked_stripped, orig_code)
        assert "literal" in str(exc_info.value).lower()

    def test_verifier_f4_rejection_of_unparseable_stripped_payload(self):
        """Tier 2: Corrupted or unparseable stripped payload raises SecurityInvariantViolationError."""
        corrupted = "def broken(\n"
        with pytest.raises(SecurityInvariantViolationError):
            verify_cloud_payload(corrupted)

    def test_verifier_f4_short_string_collision_no_false_positive(self):
        """Tier 2: Common short identifiers (id, self, path) do not cause false positives."""
        orig = 'def get_path(self, path: str = "id") -> str:\n    return "path"\n'
        stripped = 'def get_path(self, path: str = ...) -> str:\n    ...\n'
        assert verify_cloud_payload(stripped, orig) is True

    def test_verifier_f4_mathematical_zero_leak_invariant(self, adversarial_workspace: Tuple[Path, List[str]]):
        """Tier 2: Formal proof of mathematical 0-byte leak across all adversarial secrets."""
        ws, secrets = adversarial_workspace
        scanner = WorkspaceScanner(ws)
        outline = scanner.scan()
        prompt_str = outline.to_prompt_string()

        leak_count = sum(1 for s in secrets if s in prompt_str)
        assert leak_count == 0, f"Leaked {leak_count} secrets!"
        assert verify_cloud_payload(prompt_str) is True

    def test_verifier_f4_detects_sensitive_file_in_file_tree(self):
        """Tier 2: Sensitive credential filenames in file_tree trigger violation."""
        outline = WorkspaceOutline(
            workspace_root=Path("/workspace"),
            file_tree=["src/main.py", ".env"],
            ast_signatures={"src/main.py": "def run(): ...\n"},
        )
        with pytest.raises(SecurityInvariantViolationError) as exc_info:
            verify_workspace_outline(outline)
        assert ".env" in str(exc_info.value)

    def test_verifier_f4_single_and_two_argument_calls(self):
        """Tier 1: verify_cloud_payload supports both 1-arg and 2-arg calls."""
        stripped = "def compute() -> int:\n    ...\n"
        original = "def compute() -> int:\n    return 42\n"

        # 1-argument call (new contract from PROJECT.md:53)
        assert verify_cloud_payload(stripped) is True

        # 2-argument call (legacy security gauntlet contract)
        assert verify_cloud_payload(stripped, original) is True

    def test_verifier_f4_outline_serialization_roundtrip(self, clean_workspace: Path):
        """Tier 3: WorkspaceOutline to_dict() and from_dict() roundtrip preservation."""
        scanner = WorkspaceScanner(clean_workspace)
        outline = scanner.scan()

        data = outline.to_dict()
        restored = WorkspaceOutline.from_dict(data)

        assert restored.workspace_root == outline.workspace_root
        assert restored.file_tree == outline.file_tree
        assert restored.ast_signatures == outline.ast_signatures
        assert restored.summary_metadata == outline.summary_metadata
        assert verify_workspace_outline_privacy(restored) is True

    def test_verifier_f4_markdown_code_block_extraction(self):
        """Tier 2: verify_cloud_payload extracts and certifies ```python code blocks in markdown."""
        markdown_payload = (
            "# Execution Plan Overview\n\n"
            "Here is the workspace signature:\n\n"
            "```python\n"
            "def calculate(x: int) -> int:\n"
            "    ...\n"
            "```\n\n"
            "Additional notes.\n"
        )
        assert verify_cloud_payload(markdown_payload) is True

        leaked_markdown = (
            "# Execution Plan Overview\n\n"
            "```python\n"
            "def calculate(x: int) -> int:\n"
            "    return x + 1\n"
            "```\n"
        )
        with pytest.raises(SecurityInvariantViolationError):
            verify_cloud_payload(leaked_markdown)

    def test_verifier_f4_scan_and_verify_real_orchestrator_project(self):
        """Tier 3: Runs scanner and privacy verifier on the actual orchestrator codebase."""
        project_dir = Path(__file__).parent.parent / "antigravity_hybrid_orchestrator"
        scanner = WorkspaceScanner(project_dir)
        outline = scanner.scan()

        assert outline.summary_metadata["total_files"] > 0
        assert outline.summary_metadata["total_python_files"] > 0
        assert verify_workspace_outline_privacy(outline) is True

    def test_verifier_forbidden_logic_nodes_extended(self):
        """Tier 2: IfExp, Compare, and NamedExpr in code/annotations raise SecurityInvariantViolationError."""
        # Invariant 1: FORBIDDEN_LOGIC_NODES tuple includes extended logic node types
        assert ast.IfExp in FORBIDDEN_LOGIC_NODES, "ast.IfExp must be in FORBIDDEN_LOGIC_NODES"
        assert ast.Compare in FORBIDDEN_LOGIC_NODES, "ast.Compare must be in FORBIDDEN_LOGIC_NODES"
        if hasattr(ast, "NamedExpr"):
            assert ast.NamedExpr in FORBIDDEN_LOGIC_NODES, "ast.NamedExpr must be in FORBIDDEN_LOGIC_NODES"

        # Invariant 2: ast.IfExp in annotation rejected
        ifexp_payload = "def check(a: (SECRET if COND else False)): ...\n"
        with pytest.raises(SecurityInvariantViolationError):
            verify_cloud_payload(ifexp_payload)

        # Invariant 3: ast.Compare in annotation rejected
        compare_payload = "def check(b: (X > Y)): ...\n"
        with pytest.raises(SecurityInvariantViolationError):
            verify_cloud_payload(compare_payload)

        # Invariant 4: ast.NamedExpr in annotation rejected
        if hasattr(ast, "NamedExpr"):
            walrus_payload = "def check(c: (val := True)): ...\n"
            with pytest.raises(SecurityInvariantViolationError):
                verify_cloud_payload(walrus_payload)

    def test_outline_prompt_newline_injection_sanitization(self):
        """Tier 2: Newlines in file paths or metadata cannot break out of # comments into AST statements."""
        outline = WorkspaceOutline(
            workspace_root=Path("/repo\nimport root_backdoor\n#"),
            file_tree=[
                "normal.py",
                "injected.py\nimport secret_backdoor\n#",
                "win_path.py\r\nimport crlf_backdoor\r\n#",
            ],
            ast_signatures={
                "normal.py": "def run(): ...\n",
                "sub/module.py\nimport key_injection\n#": "def compute(): ...\n",
            },
            summary_metadata={
                "languages": {"Python\nimport lang_injection": 5},
            },
        )
        prompt_str = outline.to_prompt_string()

        # Invariant 1: Resulting prompt string is valid Python AST
        tree = ast.parse(prompt_str)
        assert tree is not None

        # Invariant 2: None of the injected modules are parsed as AST Import/ImportFrom nodes
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)

        injected_names = {
            "root_backdoor",
            "secret_backdoor",
            "crlf_backdoor",
            "key_injection",
            "lang_injection",
        }
        leaked = imported_modules.intersection(injected_names)
        assert not leaked, f"Injected statements broke comment encapsulation and parsed as AST imports: {leaked}"

        # Invariant 3: All lines in the header/file tree section prior to signatures start with #
        header_section = prompt_str.split("# --- MODULE:")[0]
        for line in header_section.splitlines():
            line_str = line.strip()
            if line_str:
                assert line_str.startswith("#"), f"Uncommented line in outline header: {line}"
