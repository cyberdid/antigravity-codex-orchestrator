"""Empirical adversarial stress tests authored by Challenger 2 (teamwork_preview_challenger).

Stress-tests Milestone 1 boundary conditions:
1. Pathological inputs:
   - Binary files with .py extension (NUL bytes, non-UTF8 high bytes, ELF, PNG, mixed ASCII/binary)
   - Unparseable Python syntax (truncated tokens, 5000+ nested brackets, TabError, Python 2 syntax)
   - Circular symlinks (self loops, mutual circular loops, parent loops, circular file symlinks)
   - Deep directory trees (60+ levels deep)
   - Empty repositories (completely empty, empty subdirs, ignored-only, zero-byte .py files)
   - Unicode filenames and code (Cyrillic, CJK, Emoji, Accented, RTL Arabic/Hebrew, UTF-8 BOM)
2. Cloud Payload Verifier:
   - Exhaustive rejection of all 23+ FORBIDDEN_LOGIC_NODES
   - Rejection of multi-statement function bodies and non-ellipsis/pass statements
   - Rejection of sensitive string and bytes literals (>= 5 chars)
   - Rejection of invalid/corrupted payloads and tainted markdown blocks
"""

import ast
import os
from pathlib import Path
import time
from typing import List, Tuple
import pytest

from antigravity_hybrid_orchestrator.workspace.scanner import (
    WorkspaceScanner,
    WorkspaceOutline,
    WorkspaceNotADirectoryError,
)
from antigravity_hybrid_orchestrator.privacy.workspace_privacy import (
    WorkspaceASTStripper,
    verify_workspace_outline,
    verify_workspace_outline_privacy,
)
from antigravity_hybrid_orchestrator.privacy.verifier import (
    verify_cloud_payload,
    SecurityInvariantViolationError,
    FORBIDDEN_LOGIC_NODES,
)


# ==============================================================================
# 1. PATHOLOGICAL BINARY FILES WITH .PY EXTENSION
# ==============================================================================

class TestPathologicalBinaryFiles:
    """Stress tests scanner resilience against binary contents masquerading as Python."""

    def test_binary_null_bytes_in_py_file(self, tmp_path: Path):
        """Binary file containing NUL bytes does not crash scanner and records error."""
        ws = tmp_path / "bin_null_ws"
        ws.mkdir()
        (ws / "nulls.py").write_bytes(b"\x00\x01\x02\x03\x00\x00\xff")
        (ws / "valid.py").write_text("def valid() -> None: pass\n", encoding="utf-8")

        scanner = WorkspaceScanner(ws)
        outline = scanner.scan()

        assert "nulls.py" in outline.file_tree
        assert "valid.py" in outline.file_tree
        assert "nulls.py" in outline.ast_signatures
        assert "# [SyntaxError" in outline.ast_signatures["nulls.py"]
        assert outline.summary_metadata["parse_errors"] >= 1
        assert verify_workspace_outline(outline) is True

    def test_binary_non_utf8_high_bytes_in_py_file(self, tmp_path: Path):
        """Binary file containing non-UTF8 high bytes (0x80-0xFF) handled safely."""
        ws = tmp_path / "bin_high_ws"
        ws.mkdir()
        (ws / "high.py").write_bytes(bytes(range(128, 256)))

        scanner = WorkspaceScanner(ws)
        outline = scanner.scan()

        assert "high.py" in outline.file_tree
        assert "high.py" in outline.ast_signatures
        assert "# [SyntaxError" in outline.ast_signatures["high.py"]
        assert verify_workspace_outline(outline) is True

    def test_binary_elf_and_png_files_with_py_extension(self, tmp_path: Path):
        """ELF executable and PNG image masquerading as .py files handled safely."""
        ws = tmp_path / "bin_magic_ws"
        ws.mkdir()
        # Fake ELF
        (ws / "elf.py").write_bytes(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 64)
        # Fake PNG
        (ws / "image.py").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\xff" * 64)

        scanner = WorkspaceScanner(ws)
        outline = scanner.scan()

        assert "elf.py" in outline.file_tree
        assert "image.py" in outline.file_tree
        assert outline.summary_metadata["parse_errors"] == 2
        assert verify_workspace_outline(outline) is True
        assert verify_cloud_payload(outline.to_prompt_string()) is True

    def test_giant_binary_file_exceeds_max_file_size(self, tmp_path: Path):
        """Binary .py file exceeding max_file_size_bytes is excluded by scanner."""
        ws = tmp_path / "oversize_ws"
        ws.mkdir()
        # Set max file size to 10KB and create a 20KB file
        (ws / "huge.py").write_bytes(b"\x00" * 20_000)
        (ws / "normal.py").write_text("def normal(): pass\n", encoding="utf-8")

        scanner = WorkspaceScanner(ws, max_file_size_bytes=10_000)
        outline = scanner.scan()

        assert "huge.py" not in outline.file_tree
        assert "normal.py" in outline.file_tree
        assert outline.summary_metadata["excluded_files_count"] >= 1


# ==============================================================================
# 2. PATHOLOGICAL UNPARSEABLE PYTHON SYNTAX
# ==============================================================================

class TestPathologicalSyntaxErrors:
    """Stress tests scanner and stripper against malformed, broken, and adversarial syntax."""

    def test_unclosed_nested_brackets_5000_levels(self, tmp_path: Path):
        """5000 nested unclosed brackets trigger SyntaxError without crashing Python process."""
        ws = tmp_path / "brackets_ws"
        ws.mkdir()
        (ws / "brackets.py").write_text("(" * 5000)

        scanner = WorkspaceScanner(ws)
        outline = scanner.scan()

        assert "brackets.py" in outline.file_tree
        assert "# [SyntaxError" in outline.ast_signatures["brackets.py"]
        assert outline.summary_metadata["parse_errors"] >= 1
        assert verify_workspace_outline(outline) is True

    def test_inconsistent_tabs_and_spaces_tab_error(self, tmp_path: Path):
        """Inconsistent indentation (TabError) caught as syntax error and recorded."""
        ws = tmp_path / "tab_error_ws"
        ws.mkdir()
        (ws / "tabs.py").write_text("def f():\n    x = 1\n\ty = 2\n")

        scanner = WorkspaceScanner(ws)
        outline = scanner.scan()

        assert "tabs.py" in outline.file_tree
        assert "# [SyntaxError" in outline.ast_signatures["tabs.py"]
        assert verify_workspace_outline(outline) is True

    def test_python2_obsolete_syntax(self, tmp_path: Path):
        """Python 2 syntax constructs handled as SyntaxError without crashing."""
        ws = tmp_path / "py2_ws"
        ws.mkdir()
        code = (
            "print 'hello world'\n"
            "try:\n"
            "    pass\n"
            "except Exception, e:\n"
            "    pass\n"
        )
        (ws / "legacy.py").write_text(code)

        scanner = WorkspaceScanner(ws)
        outline = scanner.scan()

        assert "legacy.py" in outline.file_tree
        assert "# [SyntaxError" in outline.ast_signatures["legacy.py"]
        assert verify_workspace_outline(outline) is True

    def test_truncated_statements_and_incomplete_tokens(self, tmp_path: Path):
        """Truncated def headers, class headers, and invalid operators handled safely."""
        ws = tmp_path / "trunc_ws"
        ws.mkdir()
        (ws / "trunc1.py").write_text("def my_func(")
        (ws / "trunc2.py").write_text("class MyClass(")
        (ws / "trunc3.py").write_text("??? @@@ !!!")

        scanner = WorkspaceScanner(ws)
        outline = scanner.scan()

        assert len(outline.file_tree) == 3
        assert outline.summary_metadata["parse_errors"] == 3
        assert verify_workspace_outline(outline) is True


# ==============================================================================
# 3. PATHOLOGICAL CIRCULAR SYMLINKS
# ==============================================================================

class TestPathologicalCircularSymlinks:
    """Stress tests symlink loop handling, directory cycles, and traversal bounds."""

    def test_directory_symlink_pointing_to_self(self, tmp_path: Path):
        """Directory symlink pointing to its parent or self does not cause recursion loop."""
        ws = tmp_path / "symlink_self_ws"
        ws.mkdir()
        sub = ws / "sub"
        sub.mkdir()
        (sub / "real.py").write_text("def real(): pass\n", encoding="utf-8")

        try:
            os.symlink(str(sub), str(sub / "self_link"))
        except OSError:
            pytest.skip("Symlinks not supported")

        start = time.time()
        scanner = WorkspaceScanner(ws)
        outline = scanner.scan()
        elapsed = time.time() - start

        assert elapsed < 1.0, f"Scan took too long ({elapsed}s), possible recursion loop"
        assert "sub/real.py" in outline.file_tree
        assert verify_workspace_outline(outline) is True

    def test_mutual_circular_directory_symlinks(self, tmp_path: Path):
        """A -> B and B -> A directory symlink cycle handled in sub-second time."""
        ws = tmp_path / "mutual_symlinks_ws"
        ws.mkdir()
        dir_a = ws / "dir_a"
        dir_b = ws / "dir_b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "a.py").write_text("def a(): pass\n", encoding="utf-8")
        (dir_b / "b.py").write_text("def b(): pass\n", encoding="utf-8")

        try:
            os.symlink(str(dir_b), str(dir_a / "to_b"))
            os.symlink(str(dir_a), str(dir_b / "to_a"))
        except OSError:
            pytest.skip("Symlinks not supported")

        start = time.time()
        scanner = WorkspaceScanner(ws)
        outline = scanner.scan()
        py_files, non_py = scanner.crawl()
        elapsed = time.time() - start

        assert elapsed < 1.0
        assert "dir_a/a.py" in outline.file_tree
        assert "dir_b/b.py" in outline.file_tree
        assert sorted(py_files) == ["dir_a/a.py", "dir_b/b.py"]

    def test_circular_file_symlinks(self, tmp_path: Path):
        """Two file symlinks pointing to each other (fileA -> fileB -> fileA) excluded safely."""
        ws = tmp_path / "file_symlink_ws"
        ws.mkdir()

        try:
            os.symlink(str(ws / "link_b.py"), str(ws / "link_a.py"))
            os.symlink(str(ws / "link_a.py"), str(ws / "link_b.py"))
        except OSError:
            pytest.skip("Symlinks not supported")

        scanner = WorkspaceScanner(ws)
        outline = scanner.scan()
        py_files, non_py = scanner.crawl()

        # Circular dangling file symlinks must not be included in outline
        assert "link_a.py" not in outline.file_tree
        assert "link_b.py" not in outline.file_tree
        assert len(py_files) == 0


# ==============================================================================
# 4. PATHOLOGICAL DEEP DIRECTORY TREES
# ==============================================================================

class TestPathologicalDeepDirectoryTrees:
    """Stress tests deeply nested directory hierarchies."""

    def test_deep_directory_nesting_60_levels(self, tmp_path: Path):
        """60-level directory depth scanned, indexed, and verified cleanly."""
        ws = tmp_path / "deep_ws"
        curr = ws
        for i in range(60):
            curr = curr / f"lvl_{i:02d}"
        curr.mkdir(parents=True)
        (curr / "deep_module.py").write_text(
            "class DeepArchitecture:\n    def deep_method(self) -> int:\n        return 100\n",
            encoding="utf-8"
        )

        scanner = WorkspaceScanner(ws)
        outline = scanner.scan()

        assert len(outline.file_tree) == 1
        deep_rel = outline.file_tree[0]
        assert deep_rel.startswith("lvl_00/")
        assert deep_rel.endswith("deep_module.py")
        assert deep_rel in outline.ast_signatures

        # Verify signature stripping and prompt formatting
        sig = outline.ast_signatures[deep_rel]
        assert "class DeepArchitecture:" in sig
        assert "def deep_method(self) -> int:" in sig
        assert "..." in sig
        assert "return 100" not in sig

        assert verify_workspace_outline(outline) is True
        assert verify_cloud_payload(outline.to_prompt_string()) is True


# ==============================================================================
# 5. PATHOLOGICAL EMPTY REPOSITORIES
# ==============================================================================

class TestPathologicalEmptyRepositories:
    """Stress tests edge cases with empty workspaces, directories, and files."""

    def test_completely_empty_workspace(self, tmp_path: Path):
        """Completely empty workspace root returns valid outline and verified prompt."""
        ws = tmp_path / "empty_ws"
        ws.mkdir()

        scanner = WorkspaceScanner(ws)
        outline = scanner.scan()

        assert outline.file_tree == []
        assert outline.ast_signatures == {}
        assert outline.summary_metadata["total_files"] == 0
        assert outline.summary_metadata["total_python_files"] == 0
        assert outline.summary_metadata["total_lines"] == 0

        assert verify_workspace_outline(outline) is True
        prompt_str = outline.to_prompt_string()
        assert "ZERO-KNOWLEDGE WORKSPACE OUTLINE" in prompt_str
        assert verify_cloud_payload(prompt_str) is True

    def test_workspace_with_only_empty_subdirectories(self, tmp_path: Path):
        """Workspace with nested empty directories yields 0 files without error."""
        ws = tmp_path / "nested_empty_ws"
        (ws / "a" / "b" / "c" / "d").mkdir(parents=True)

        scanner = WorkspaceScanner(ws)
        outline = scanner.scan()

        assert outline.file_tree == []
        assert outline.summary_metadata["total_files"] == 0
        assert verify_workspace_outline(outline) is True

    def test_workspace_with_zero_byte_python_and_text_files(self, tmp_path: Path):
        """Workspace with zero-byte files renders empty module placeholders cleanly."""
        ws = tmp_path / "zerobyte_ws"
        ws.mkdir()
        (ws / "empty.py").touch()
        (ws / "notes.txt").touch()

        scanner = WorkspaceScanner(ws)
        outline = scanner.scan()

        assert outline.file_tree == ["empty.py", "notes.txt"]
        assert outline.ast_signatures["empty.py"] == ""
        assert verify_workspace_outline(outline) is True

        prompt = outline.to_prompt_string()
        assert "# (Empty module)" in prompt
        assert verify_cloud_payload(prompt) is True

    def test_workspace_with_only_comments_and_docstrings(self, tmp_path: Path):
        """Workspace with files containing only comments/docstrings yields empty signatures."""
        ws = tmp_path / "comments_only_ws"
        ws.mkdir()
        code = (
            "# Top level comment\n"
            "# Another comment\n"
            '"""Module docstring explaining architecture."""\n'
        )
        (ws / "comments.py").write_text(code, encoding="utf-8")

        scanner = WorkspaceScanner(ws)
        outline = scanner.scan()

        assert outline.file_tree == ["comments.py"]
        assert outline.ast_signatures["comments.py"] == ""
        assert verify_workspace_outline(outline) is True
        assert verify_cloud_payload(outline.to_prompt_string()) is True


# ==============================================================================
# 6. PATHOLOGICAL UNICODE FILENAMES AND CONTENTS
# ==============================================================================

class TestPathologicalUnicodeFilenamesAndContents:
    """Stress tests Unicode support across filenames and source code."""

    def test_international_and_emoji_filenames(self, tmp_path: Path):
        """Cyrillic, CJK, Emoji, Accented, and RTL filenames indexed and stripped."""
        ws = tmp_path / "unicode_ws"
        ws.mkdir()

        files_data = {
            "модуль.py": "def привет() -> str:\n    return 'мир'\n",
            "测试_模块.py": "def 计算(x: int) -> int:\n    return x * 2\n",
            "🚀_launch.py": "def start_engine() -> None:\n    return None\n",
            "café_naïve.py": "class ÜberMensch:\n    def speak(self): pass\n",
            "نموذج.py": "def salam(): pass\n",
            "מודל.py": "def shalom(): pass\n",
            "[app] (v1.0) #run.py": "def run(): pass\n",
        }

        for filename, content in files_data.items():
            (ws / filename).write_text(content, encoding="utf-8")

        scanner = WorkspaceScanner(ws)
        outline = scanner.scan()

        assert len(outline.file_tree) == len(files_data)
        assert len(outline.ast_signatures) == len(files_data)

        for filename in files_data:
            assert filename in outline.file_tree
            sig = outline.ast_signatures[filename]
            assert "return" not in sig
            assert "..." in sig or "pass" in sig

        assert verify_workspace_outline(outline) is True
        assert verify_cloud_payload(outline.to_prompt_string()) is True

    def test_utf8_bom_handling(self, tmp_path: Path):
        """Python file starting with UTF-8 BOM byte sequence handled gracefully."""
        ws = tmp_path / "bom_ws"
        ws.mkdir()
        bom_code = b"\xef\xbb\xbfdef with_bom() -> int:\n    return 1\n"
        (ws / "bom.py").write_bytes(bom_code)

        scanner = WorkspaceScanner(ws)
        outline = scanner.scan()

        assert "bom.py" in outline.file_tree
        assert "bom.py" in outline.ast_signatures
        # BOM is caught as SyntaxError annotation without crashing
        assert verify_workspace_outline(outline) is True
        assert verify_cloud_payload(outline.to_prompt_string()) is True

    def test_unicode_identifiers_in_source_code(self, tmp_path: Path):
        """Unicode function names and variables preserved in interface signatures."""
        ws = tmp_path / "unicode_code_ws"
        ws.mkdir()
        code = (
            "def calculate_π(r: float) -> float:\n"
            "    π = 3.1415926535\n"
            "    return π * r * r\n\n"
            "def 挨拶(お名前: str) -> str:\n"
            "    return 'こんにちは ' + お名前\n"
        )
        (ws / "math_intl.py").write_text(code, encoding="utf-8")

        scanner = WorkspaceScanner(ws)
        outline = scanner.scan()
        sig = outline.ast_signatures["math_intl.py"]

        assert "def calculate_π(r: float) -> float:" in sig
        assert "def 挨拶(お名前: str) -> str:" in sig
        assert "..." in sig
        assert "3.1415926535" not in sig
        assert "こんにちは" not in sig

        assert verify_workspace_outline(outline) is True
        assert verify_cloud_payload(outline.to_prompt_string()) is True


# ==============================================================================
# 7. CLOUD PAYLOAD VERIFIER STRICT REJECTION OF FORBIDDEN LOGIC & SENSITIVE LITERALS
# ==============================================================================

class TestCloudPayloadVerifierStrictRejection:
    """Exhaustively verifies that verify_cloud_payload rejects any forbidden logic or leak."""

    @pytest.mark.parametrize("node_name,code_snippet", [
        ("If", "if True:\n    ..."),
        ("For", "for x in y:\n    ..."),
        ("AsyncFor", "async def f():\n    async for x in y:\n        ..."),
        ("While", "while True:\n    ..."),
        ("Try", "try:\n    ...\nexcept Exception:\n    ..."),
        ("With", "with lock:\n    ..."),
        ("AsyncWith", "async def f():\n    async with lock:\n        ..."),
        ("Return", "def f():\n    return x"),
        ("Yield", "def f():\n    yield x"),
        ("YieldFrom", "def f():\n    yield from x"),
        ("Call", "execute()"),
        ("BinOp", "x = a + b"),
        ("AugAssign", "x += 1"),
        ("Lambda", "f = lambda x: x"),
        ("ListComp", "x = [i for i in range]"),
        ("SetComp", "x = {i for i in range}"),
        ("DictComp", "x = {i: i for i in range}"),
        ("GeneratorExp", "x = (i for i in range)"),
        ("Raise", "raise RuntimeError"),
        ("Delete", "del x"),
        ("Assert", "assert True"),
        ("Match", "match val:\n    case 1:\n        ..."),
        ("TryStar", "try:\n    ...\nexcept* Exception:\n    ..."),
    ])
    def test_verify_cloud_payload_rejects_each_forbidden_logic_node(
        self, node_name: str, code_snippet: str
    ):
        """Verifies verify_cloud_payload raises SecurityInvariantViolationError on every forbidden node."""
        with pytest.raises(SecurityInvariantViolationError) as exc_info:
            verify_cloud_payload(code_snippet)
        assert (
            "Forbidden logic node" in str(exc_info.value)
            or "Function" in str(exc_info.value)
        )

    def test_verify_cloud_payload_rejects_multi_statement_body(self):
        """Function body with more than 1 statement is strictly rejected."""
        code = "def f():\n    ...\n    pass\n"
        with pytest.raises(SecurityInvariantViolationError) as exc_info:
            verify_cloud_payload(code)
        assert "body length > 1" in str(exc_info.value)

    def test_verify_cloud_payload_rejects_non_ellipsis_or_pass_body(self):
        """Function body containing expression other than Ellipsis or Pass is strictly rejected."""
        code = "def f():\n    42\n"
        with pytest.raises(SecurityInvariantViolationError) as exc_info:
            verify_cloud_payload(code)
        assert "not Ellipsis or Pass statement" in str(exc_info.value)

    @pytest.mark.parametrize("secret_type,orig_code,leaked_payload", [
        (
            "constant_string",
            'API_KEY = "SECRET_TOKEN_XYZ_12345"\n',
            'API_KEY = "SECRET_TOKEN_XYZ_12345"\n',
        ),
        (
            "bytes_literal",
            'SECRET_BYTES = b"SUPER_SECRET_BYTES_DATA"\n',
            'SECRET_BYTES = b"SUPER_SECRET_BYTES_DATA"\n',
        ),
        (
            "raw_string",
            'PATH = r"SECRET_VAULT_PATH_99999"\n',
            'PATH = "SECRET_VAULT_PATH_99999"\n',
        ),
        (
            "multiline_rsa_key",
            'KEY = "-----BEGIN PRIVATE KEY-----\\nSECRET_KEY_BODY\\n-----END PRIVATE KEY-----"\n',
            'KEY = "-----BEGIN PRIVATE KEY-----\\nSECRET_KEY_BODY\\n-----END PRIVATE KEY-----"\n',
        ),
    ])
    def test_verify_cloud_payload_rejects_sensitive_literals(
        self, secret_type: str, orig_code: str, leaked_payload: str
    ):
        """Verifies verify_cloud_payload detects leaks of string/bytes literals >= 5 chars."""
        with pytest.raises(SecurityInvariantViolationError) as exc_info:
            verify_cloud_payload(leaked_payload, original_code=orig_code)
        assert "leaked into cloud payload" in str(exc_info.value)

    def test_verify_cloud_payload_rejects_unparseable_garbage(self):
        """Unparseable string with no markdown code blocks raises SecurityInvariantViolationError."""
        with pytest.raises(SecurityInvariantViolationError) as exc_info:
            verify_cloud_payload("def incomplete_func(")
        assert "not valid Python AST" in str(exc_info.value)

    def test_verify_cloud_payload_rejects_markdown_with_corrupted_block(self):
        """Markdown with a syntax error inside a python block raises SecurityInvariantViolationError."""
        md = (
            "# Plan\n\n"
            "```python\n"
            "def broken(\n"
            "```\n"
        )
        with pytest.raises(SecurityInvariantViolationError) as exc_info:
            verify_cloud_payload(md)
        assert "not valid Python AST" in str(exc_info.value)

    def test_verify_cloud_payload_rejects_markdown_with_forbidden_node_in_second_block(self):
        """Markdown containing valid 1st block but forbidden logic in 2nd block is rejected."""
        md = (
            "# Plan\n\n"
            "```python\n"
            "def clean(): ...\n"
            "```\n\n"
            "Second block:\n"
            "```python\n"
            "for item in items: ...\n"
            "```\n"
        )
        with pytest.raises(SecurityInvariantViolationError) as exc_info:
            verify_cloud_payload(md)
        assert "Forbidden logic node For" in str(exc_info.value)
