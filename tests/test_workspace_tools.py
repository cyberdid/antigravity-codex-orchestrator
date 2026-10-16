"""Unit and boundary test suite for WorkspaceToolSuite (Milestone 3, Feature F8).

Contains 28 tests (16 Tier 1 Feature Coverage + 12 Tier 2 Boundary/Corner Cases)
specified by spec_miner_m3_1_gen3/report.md.
"""

import os
from pathlib import Path
import subprocess
import sys
import time
import pytest

from antigravity_hybrid_orchestrator.workforce.tools import ToolResult, WorkspaceToolSuite


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceToolSuite:
    """Fixture providing a fresh WorkspaceToolSuite backed by tmp_path."""
    return WorkspaceToolSuite(workspace_root=tmp_path)


# ==============================================================================
# TIER 1: FEATURE COVERAGE (16 TESTS)
# ==============================================================================

def test_read_file_full_content(workspace: WorkspaceToolSuite, tmp_path: Path):
    """Test 1: Reads full content of multi-line text file."""
    content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n"
    (tmp_path / "notes.txt").write_text(content, encoding="utf-8")

    res = workspace.read_file("notes.txt")
    assert res.success is True
    assert res.output == content
    assert res.metadata["total_lines"] == 5
    assert res.metadata["lines_returned"] == 5


def test_read_file_line_slicing_middle_range(workspace: WorkspaceToolSuite, tmp_path: Path):
    """Test 2: Reads specific 1-indexed line range from middle of file."""
    lines = [f"Line {i}" for i in range(1, 11)]
    (tmp_path / "data.txt").write_text("\n".join(lines), encoding="utf-8")

    res = workspace.read_file("data.txt", start_line=3, end_line=7)
    assert res.success is True
    assert res.output == "Line 3\nLine 4\nLine 5\nLine 6\nLine 7"
    assert res.metadata["lines_returned"] == 5


def test_read_file_start_line_only(workspace: WorkspaceToolSuite, tmp_path: Path):
    """Test 3: Reads from start_line to end of file."""
    lines = [f"Line {i}" for i in range(1, 11)]
    (tmp_path / "data.txt").write_text("\n".join(lines), encoding="utf-8")

    res = workspace.read_file("data.txt", start_line=6, end_line=None)
    assert res.success is True
    assert res.output == "Line 6\nLine 7\nLine 8\nLine 9\nLine 10"
    assert res.metadata["lines_returned"] == 5


def test_read_file_end_line_only(workspace: WorkspaceToolSuite, tmp_path: Path):
    """Test 4: Reads from line 1 up to end_line."""
    lines = [f"Line {i}" for i in range(1, 11)]
    (tmp_path / "data.txt").write_text("\n".join(lines), encoding="utf-8")

    res = workspace.read_file("data.txt", start_line=None, end_line=4)
    assert res.success is True
    assert res.output == "Line 1\nLine 2\nLine 3\nLine 4"
    assert res.metadata["lines_returned"] == 4


def test_read_file_boundary_clamping(workspace: WorkspaceToolSuite, tmp_path: Path):
    """Test 5: Negative start_line clamped to 1, out-of-bounds end_line clamped to total_lines."""
    lines = [f"Line {i}" for i in range(1, 6)]
    (tmp_path / "data.txt").write_text("\n".join(lines), encoding="utf-8")

    res = workspace.read_file("data.txt", start_line=-10, end_line=100)
    assert res.success is True
    assert res.output == "\n".join(lines)
    assert res.metadata["start_line"] == 1
    assert res.metadata["end_line"] == 5
    assert res.metadata["lines_returned"] == 5


def test_read_file_empty_file(workspace: WorkspaceToolSuite, tmp_path: Path):
    """Test 6: Reads 0-byte file without error."""
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")

    res = workspace.read_file("empty.txt")
    assert res.success is True
    assert res.output == ""
    assert res.metadata["total_lines"] == 0


def test_write_file_root_level(workspace: WorkspaceToolSuite, tmp_path: Path):
    """Test 7: Writes file at workspace root."""
    content = "print('hello world')"
    res = workspace.write_file("hello.py", content)

    assert res.success is True
    assert (tmp_path / "hello.py").exists()
    assert (tmp_path / "hello.py").read_text(encoding="utf-8") == content
    assert res.metadata["bytes_written"] == len(content.encode("utf-8"))


def test_write_file_deep_nested_directory_creation(workspace: WorkspaceToolSuite, tmp_path: Path):
    """Test 8: Recursively creates parent directories when writing nested file."""
    content = "VALUE = 42\n"
    res = workspace.write_file("pkg/subpkg/module.py", content)

    assert res.success is True
    target = tmp_path / "pkg" / "subpkg" / "module.py"
    assert target.exists()
    assert target.read_text(encoding="utf-8") == content


def test_write_file_atomic_overwrite(workspace: WorkspaceToolSuite, tmp_path: Path):
    """Test 9: Atomically replaces existing file content and leaves no temp files."""
    target = tmp_path / "version.txt"
    target.write_text("v1.0", encoding="utf-8")

    res = workspace.write_file("version.txt", "v2.0")
    assert res.success is True
    assert target.read_text(encoding="utf-8") == "v2.0"

    # Confirm no leftover temp files
    leftover_tmp = [f.name for f in tmp_path.iterdir() if f.name.startswith(".version.txt.tmp_")]
    assert leftover_tmp == []


def test_run_command_exit_code_zero(workspace: WorkspaceToolSuite):
    """Test 10: Executes command successfully returning stdout and exit code 0."""
    res = workspace.run_command("echo 'test run success'")
    assert res.success is True
    assert res.output.strip() == "test run success"
    assert res.metadata["exit_code"] == 0
    assert res.error is None


def test_run_command_non_zero_exit_code(workspace: WorkspaceToolSuite):
    """Test 11: Captures non-zero exit code and sets success=False."""
    res = workspace.run_command(f"{sys.executable} -c 'import sys; sys.exit(42)'")
    assert res.success is False
    assert res.metadata["exit_code"] == 42
    assert "42" in (res.error or "")


def test_run_command_captures_stderr_output(workspace: WorkspaceToolSuite):
    """Test 12: Captures stderr output in result error and metadata."""
    cmd = f'{sys.executable} -c \'import sys; sys.stderr.write("fatal error\\n"); sys.exit(1)\''
    res = workspace.run_command(cmd)
    assert res.success is False
    assert "fatal error" in (res.error or "")
    assert "fatal error" in res.metadata.get("stderr", "")


def test_run_command_custom_working_directory(workspace: WorkspaceToolSuite, tmp_path: Path):
    """Test 13: Runs command inside custom relative subdirectory."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    res = workspace.run_command("pwd", cwd="subdir")
    assert res.success is True
    assert str(subdir.resolve()) in res.output.strip()


def test_git_status_in_clean_git_repository(workspace: WorkspaceToolSuite, tmp_path: Path):
    """Test 14: Clean git repository returns clean=True and empty changes."""
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(tmp_path), check=True)

    init_file = tmp_path / "init.txt"
    init_file.write_text("initial", encoding="utf-8")
    subprocess.run(["git", "add", "init.txt"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=str(tmp_path), check=True)

    res = workspace.git_status()
    assert res.success is True
    assert res.metadata["is_git_repo"] is True
    assert res.metadata["clean"] is True
    assert res.metadata["changes"] == []


def test_git_status_detects_untracked_and_modified_files(workspace: WorkspaceToolSuite, tmp_path: Path):
    """Test 15: Detects modified tracked files and newly added untracked files."""
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(tmp_path), check=True)

    (tmp_path / "existing.py").write_text("original", encoding="utf-8")
    subprocess.run(["git", "add", "existing.py"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-m", "add existing"], cwd=str(tmp_path), check=True)

    # Modify existing and create new untracked
    (tmp_path / "existing.py").write_text("modified", encoding="utf-8")
    (tmp_path / "new.py").write_text("new content", encoding="utf-8")

    res = workspace.git_status()
    assert res.success is True
    changes = res.metadata["changes"]
    assert "existing.py" in changes
    assert "new.py" in changes


def test_git_diff_unstaged_and_staged(workspace: WorkspaceToolSuite, tmp_path: Path):
    """Test 16: Differentiates between unstaged and staged diffs."""
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(tmp_path), check=True)

    (tmp_path / "a.py").write_text("a_v1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("b_v1\n", encoding="utf-8")
    subprocess.run(["git", "add", "a.py", "b.py"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), check=True)

    # Modify a.py (unstaged) and b.py (staged)
    (tmp_path / "a.py").write_text("a_v2_modified\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("b_v2_staged\n", encoding="utf-8")
    subprocess.run(["git", "add", "b.py"], cwd=str(tmp_path), check=True)

    unstaged_diff = workspace.git_diff(staged=False)
    assert unstaged_diff.success is True
    assert "a_v2_modified" in unstaged_diff.output
    assert "b_v2_staged" not in unstaged_diff.output

    staged_diff = workspace.git_diff(staged=True)
    assert staged_diff.success is True
    assert "b_v2_staged" in staged_diff.output
    assert "a_v2_modified" not in staged_diff.output


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES (12 TESTS)
# ==============================================================================

def test_read_file_path_traversal_dot_dot_escape(workspace: WorkspaceToolSuite):
    """Test 17: Prevents relative upward path traversal ('..') escaping workspace."""
    res = workspace.read_file("../../etc/passwd")
    assert res.success is False
    assert "Permission denied" in (res.error or "")


def test_read_file_absolute_path_escape(workspace: WorkspaceToolSuite):
    """Test 18: Blocks absolute paths escaping workspace root."""
    res = workspace.read_file("/etc/shadow")
    assert res.success is False
    assert "Permission denied" in (res.error or "")


def test_write_file_path_traversal_attack(workspace: WorkspaceToolSuite):
    """Test 19: Blocks writing outside workspace via directory traversal."""
    res = workspace.write_file("../../../tmp/malicious.sh", "evil")
    assert res.success is False
    assert "Permission denied" in (res.error or "")


def test_symlink_directory_breakout_prevention(workspace: WorkspaceToolSuite, tmp_path: Path):
    """Test 20: Prevents escaping workspace via symlinks pointing to external targets."""
    outside_dir = tmp_path.parent / f"outside_{tmp_path.name}"
    outside_dir.mkdir(exist_ok=True)
    (outside_dir / "secret.txt").write_text("super_secret", encoding="utf-8")

    link = tmp_path / "outside_link"
    link.symlink_to(outside_dir)

    read_res = workspace.read_file("outside_link/secret.txt")
    assert read_res.success is False
    assert "Permission denied" in (read_res.error or "")

    write_res = workspace.write_file("outside_link/hack.txt", "exploit")
    assert write_res.success is False
    assert "Permission denied" in (write_res.error or "")


def test_git_directory_modification_protection(workspace: WorkspaceToolSuite, tmp_path: Path):
    """Test 21: Strictly rejects writing to or tampering with .git directory."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    res = workspace.write_file(".git/config", "corrupted")
    assert res.success is False
    assert "Writing to .git directory is prohibited" in (res.error or "")

    res_hook = workspace.write_file(".git/hooks/pre-commit", "#!/bin/sh\nexit 1")
    assert res_hook.success is False
    assert "Writing to .git directory is prohibited" in (res_hook.error or "")


def test_process_tree_cleanup_on_timeout(workspace: WorkspaceToolSuite):
    """Test 22: Enforces command timeout and kills process group leader without orphaned process."""
    start = time.perf_counter()
    res = workspace.run_command("sleep 100", timeout=1)
    duration = time.perf_counter() - start

    assert duration < 3.0
    assert res.success is False
    assert res.metadata["timed_out"] is True
    assert res.metadata["exit_code"] == -1


def test_configured_default_command_timeout_is_used(tmp_path: Path):
    workspace = WorkspaceToolSuite(workspace_root=tmp_path, default_command_timeout=0.1)
    result = workspace.run_command("sleep 2")

    assert result.success is False
    assert result.metadata["timed_out"] is True
    assert "0.1 seconds" in (result.error or "")


def test_idle_timeout_resets_when_command_emits_progress(tmp_path: Path):
    workspace = WorkspaceToolSuite(workspace_root=tmp_path, default_idle_timeout=0.15)
    command = "python3 -c 'import time; [ (print(i, flush=True), time.sleep(0.08)) for i in range(5) ]'"

    result = workspace.run_command(command)

    assert result.success is True
    assert "4" in result.output


def test_idle_timeout_stops_silent_hung_command(tmp_path: Path):
    workspace = WorkspaceToolSuite(workspace_root=tmp_path, default_idle_timeout=0.1)

    result = workspace.run_command("sleep 2")

    assert result.success is False
    assert result.metadata["timed_out"] is True
    assert "no output" in (result.error or "")


def test_idle_timeout_still_monitors_process_after_pipes_close(tmp_path: Path):
    workspace = WorkspaceToolSuite(workspace_root=tmp_path, default_idle_timeout=0.1)
    command = "python3 -c 'import os,time; os.close(1); os.close(2); time.sleep(2)'"

    result = workspace.run_command(command)

    assert result.success is False
    assert result.metadata["timed_out"] is True
    assert "no output" in (result.error or "")


def test_run_budget_cancels_active_command_process_group(tmp_path: Path):
    import threading
    from antigravity_hybrid_orchestrator.engine.control import ExecutionBudget, ExecutionControl

    control = ExecutionControl(ExecutionBudget())
    workspace = WorkspaceToolSuite(
        workspace_root=tmp_path, default_idle_timeout=None, execution_control=control
    )
    timer = threading.Timer(0.1, lambda: control.cancel("operator_cancelled"))
    timer.start()
    try:
        result = workspace.run_command("sleep 5")
    finally:
        timer.cancel()

    assert result.success is False
    assert "operator_cancelled" in (result.error or "")


def test_nested_process_tree_killed_on_timeout(workspace: WorkspaceToolSuite):
    """Test 23: Enforces termination of grandchild processes via process group signal."""
    cmd = (
        f'{sys.executable} -c "'
        'import subprocess, time; '
        'subprocess.Popen([\'sleep\', \'100\']); '
        'time.sleep(100)"'
    )
    start = time.perf_counter()
    res = workspace.run_command(cmd, timeout=1)
    duration = time.perf_counter() - start

    assert duration < 3.0
    assert res.success is False
    assert res.metadata["timed_out"] is True


def test_command_not_found_handling(workspace: WorkspaceToolSuite):
    """Test 24: Missing executable produces exit code 127 without unhandled crash."""
    res = workspace.run_command("nonexistent_binary_xyz_99")
    assert res.success is False
    assert res.metadata["exit_code"] == 127


@pytest.mark.parametrize("command", [
    "python -m pip install torchaudio",
    "pip3 install torch",
    "sudo apt-get install ffmpeg",
    "uv pip install transformers",
])
def test_autonomous_command_runner_blocks_host_package_installers(workspace, command):
    result = workspace.run_command(command)

    assert result.success is False
    assert result.metadata.get("blocked_by_policy") is True
    assert "package installation is disabled" in result.error.lower()


def test_binary_file_detection_and_rejection(workspace: WorkspaceToolSuite, tmp_path: Path):
    """Test 25: Binary files containing null bytes are safely detected and rejected."""
    bin_file = tmp_path / "model.bin"
    bin_file.write_bytes(b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00")

    res = workspace.read_file("model.bin")
    assert res.success is False
    assert "Binary file cannot be displayed as text" in (res.error or "")
    assert res.metadata.get("is_binary") is True


def test_non_git_workspace_handling_git_status_and_diff(workspace: WorkspaceToolSuite):
    """Test 26: Clean handling of non-git workspace for status and diff."""
    status_res = workspace.git_status()
    assert status_res.success is True
    assert status_res.output == "Workspace is not a Git repository."
    assert status_res.metadata["is_git_repo"] is False

    diff_res = workspace.git_diff()
    assert diff_res.success is True
    assert diff_res.output == "Workspace is not a Git repository."
    assert diff_res.metadata["is_git_repo"] is False


def test_large_output_streaming_no_deadlock(workspace: WorkspaceToolSuite):
    """Test 27: Processes producing large output (>2MB) complete without pipe deadlock."""
    cmd = f"{sys.executable} -c 'print(\"A\" * 2000000)'"
    start = time.perf_counter()
    res = workspace.run_command(cmd, timeout=10)
    duration = time.perf_counter() - start

    assert duration < 5.0
    assert res.success is True
    assert len(res.output) >= 2000000


def test_path_sanitizer_null_bytes_and_empty_paths(workspace: WorkspaceToolSuite):
    """Test 28: Path resolution strictly rejects empty strings and null bytes."""
    with pytest.raises(ValueError):
        workspace._resolve_path("")

    with pytest.raises(ValueError):
        workspace._resolve_path("   ")

    with pytest.raises(ValueError):
        workspace._resolve_path("file\0.txt")

    # Public method returns failed ToolResult
    read_empty = workspace.read_file("")
    assert read_empty.success is False
    assert "Invalid path" in (read_empty.error or "")

    read_null = workspace.read_file("file\0.txt")
    assert read_null.success is False
    assert "Invalid path" in (read_null.error or "")


# ==============================================================================
# TIER 3: LIST_DIR AND PATCH_FILE TOOLS
# ==============================================================================

def test_list_dir_lists_files_and_directories(workspace: WorkspaceToolSuite, tmp_path: Path):
    """Lists directory entries with trailing slash for subdirectories."""
    (tmp_path / "alpha.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.txt").write_text("hello", encoding="utf-8")

    res = workspace.list_dir(".")
    assert res.success is True
    assert "alpha.py" in res.metadata["entries"]
    assert "subdir/" in res.metadata["entries"]
    assert "alpha.py" in res.output
    assert "subdir/" in res.output

    sub_res = workspace.list_dir("subdir")
    assert sub_res.success is True
    assert sub_res.metadata["entries"] == ["nested.txt"]


def test_list_dir_rejects_escape_and_non_directory(workspace: WorkspaceToolSuite, tmp_path: Path):
    """Blocks path traversal and reports error when path is a file or missing."""
    (tmp_path / "file.txt").write_text("hi", encoding="utf-8")

    escape_res = workspace.list_dir("../..")
    assert escape_res.success is False
    assert "Permission denied" in (escape_res.error or "")

    file_res = workspace.list_dir("file.txt")
    assert file_res.success is False
    assert "not a directory" in (file_res.error or "").lower()

    missing_res = workspace.list_dir("no_such_dir")
    assert missing_res.success is False
    assert "not found" in (missing_res.error or "").lower()


def test_patch_file_replaces_unique_target(workspace: WorkspaceToolSuite, tmp_path: Path):
    """Atomically patches an exact substring in an existing file."""
    target = tmp_path / "app.py"
    target.write_text("def greet():\n    return 'hello'\n", encoding="utf-8")

    res = workspace.patch_file(
        "app.py",
        target_content="return 'hello'",
        replacement_content="return 'world'",
    )
    assert res.success is True
    assert target.read_text(encoding="utf-8") == "def greet():\n    return 'world'\n"
    assert res.metadata["replacements"] == 1


def test_patch_file_rejects_missing_or_ambiguous_target_and_git_dir(
    workspace: WorkspaceToolSuite, tmp_path: Path
):
    """Rejects missing target, ambiguous multi-match when allow_multiple=False, and .git edits."""
    target = tmp_path / "dup.py"
    target.write_text("x = 1\nx = 1\n", encoding="utf-8")

    missing = workspace.patch_file("dup.py", target_content="x = 2", replacement_content="x = 3")
    assert missing.success is False
    assert "not found" in (missing.error or "").lower()

    ambiguous = workspace.patch_file("dup.py", target_content="x = 1", replacement_content="x = 9")
    assert ambiguous.success is False
    assert "2 times" in (ambiguous.error or "")

    multi_ok = workspace.patch_file(
        "dup.py", target_content="x = 1", replacement_content="x = 9", allow_multiple=True
    )
    assert multi_ok.success is True
    assert target.read_text(encoding="utf-8") == "x = 9\nx = 9\n"

    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("core", encoding="utf-8")
    git_res = workspace.patch_file(".git/config", target_content="core", replacement_content="hack")
    assert git_res.success is False
    assert "Writing to .git directory is prohibited" in (git_res.error or "")


def test_tool_definitions_and_execute_expose_list_dir_and_patch_file(
    workspace: WorkspaceToolSuite, tmp_path: Path
):
    """Ensures get_tool_definitions and execute() include list_dir and patch_file."""
    defs = workspace.get_tool_definitions()
    names = {d["name"] for d in defs}
    assert "list_dir" in names
    assert "patch_file" in names

    (tmp_path / "mod.py").write_text("VAL = 10\n", encoding="utf-8")
    patch_res = workspace.execute(
        "patch_file",
        path="mod.py",
        target_content="VAL = 10",
        replacement_content="VAL = 20",
    )
    assert patch_res.success is True
    list_res = workspace.execute("list_dir", path=".")
    assert list_res.success is True
    assert "mod.py" in list_res.output

