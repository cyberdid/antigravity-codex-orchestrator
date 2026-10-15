"""Adversarial stress tests authored by Challenger 2 (teamwork_preview_challenger).

Empirically tests:
1. CLI argument parsing robustness, choices enforcement, and graceful error exits.
2. Atomic patch file writing safety under concurrent access and simulated agent/disk failures.
3. Privacy assertion exception handling across all forbidden AST nodes, leaks, and lifecycle cleanup.
"""

import ast
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from antigravity_hybrid_orchestrator.cli import build_parser, main
from antigravity_hybrid_orchestrator.coordinator import HybridCoordinator
from antigravity_hybrid_orchestrator.gauntlet.baselines import (
    AUTH_VULNERABLE,
    BILLING_VULNERABLE,
    DATABASE_VULNERABLE,
    reset_gauntlet,
    restore_secure,
)
from antigravity_hybrid_orchestrator.privacy import (
    SecurityInvariantViolationError,
    verify_cloud_payload,
)
from antigravity_hybrid_orchestrator.privacy.verifier import FORBIDDEN_LOGIC_NODES


@pytest.fixture(autouse=True)
def cleanup_gauntlet():
    """Ensure gauntlet baselines are clean and secure before and after tests."""
    restore_secure()
    for mod in ["auth", "billing", "database"]:
        sys.modules.pop(f"antigravity_hybrid_orchestrator.gauntlet.{mod}", None)
    yield
    restore_secure()
    for mod in ["auth", "billing", "database"]:
        sys.modules.pop(f"antigravity_hybrid_orchestrator.gauntlet.{mod}", None)


# ==============================================================================
# 1. CLI ARGUMENT PARSING & GRACEFUL EXIT ADVERSARIAL TESTS
# ==============================================================================

def test_cli_invalid_mode_exit_code_2():
    """Adversarial test: invalid --mode must trigger exit code 2 with helpful choice error."""
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--mode", "invalid_mode_xyz"])
    assert exc_info.value.code == 2

    # Verify subprocess invocation matches standard unix behavior
    cmd = [sys.executable, "-m", "antigravity_hybrid_orchestrator", "--mode", "invalid_mode_xyz"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 2
    assert "invalid choice" in res.stderr.lower()


def test_cli_invalid_target_exit_code_2():
    """Adversarial test: invalid --target must trigger exit code 2."""
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--target", "nonexistent_target.py"])
    assert exc_info.value.code == 2

    cmd = [sys.executable, "-m", "antigravity_hybrid_orchestrator", "--target", "unknown_module"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 2
    assert "invalid choice" in res.stderr.lower()


def test_cli_bad_refresh_rate_string_exit_code_2():
    """Adversarial test: non-float --refresh-rate must trigger exit code 2."""
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--refresh-rate", "not_a_number"])
    assert exc_info.value.code == 2

    cmd = [sys.executable, "-m", "antigravity_hybrid_orchestrator", "--refresh-rate", "nan_string"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 2
    assert "invalid float" in res.stderr.lower()


@pytest.mark.parametrize("bad_rate", [0.0, -1.0, -10.5])
def test_cli_non_positive_refresh_rate_graceful_fallback(bad_rate):
    """Adversarial test: non-positive refresh rate must gracefully degrade to headless stream mode."""
    exit_code = main(["--mode", "mock", "--target", "auth", "--no-hud", "--refresh-rate", str(bad_rate)])
    assert exit_code == 0


def test_coordinator_invalid_target_value_error():
    """Adversarial test: programmatic coordinator invocation with unknown target raises ValueError."""
    coordinator = HybridCoordinator(mode="mock", target="ghost_module.py", no_hud=True)
    with pytest.raises(ValueError) as exc_info:
        coordinator.run()
    assert "Unknown gauntlet target" in str(exc_info.value)


# ==============================================================================
# 2. ATOMIC PATCH FILE SAFETY ADVERSARIAL TESTS
# ==============================================================================

def test_atomic_write_concurrency_stress():
    """Stress test: 10 threads continuously reading while atomic writes swap 50KB payloads.
    Verifies 0 partial reads, 0 0-byte reads, and 0 corrupted chunks.
    """
    temp_dir = Path(tempfile.mkdtemp())
    try:
        target_file = temp_dir / "stress_auth.py"
        content_v1 = "# Payload Alpha\n" + ("x = 100\n" * 5000)
        content_v2 = "# Payload Beta\n" + ("y = 200\n" * 5000)
        target_file.write_text(content_v1, encoding="utf-8")

        coordinator = HybridCoordinator(mode="mock", target="auth", no_hud=True)

        stop_event = threading.Event()
        read_anomalies = []

        def reader():
            while not stop_event.is_set():
                try:
                    data = target_file.read_text(encoding="utf-8")
                    if len(data) == 0:
                        read_anomalies.append("Observed 0-byte truncated file")
                    elif data != content_v1 and data != content_v2:
                        read_anomalies.append(f"Observed partial/corrupted content (len={len(data)})")
                except Exception as e:
                    read_anomalies.append(f"Read error: {e}")

        readers = [threading.Thread(target=reader) for _ in range(10)]
        for r in readers:
            r.start()

        # Perform 100 rapid atomic swaps
        for i in range(100):
            payload = content_v2 if i % 2 == 0 else content_v1
            coordinator._atomic_write(target_file, payload)

        stop_event.set()
        for r in readers:
            r.join()

        assert len(read_anomalies) == 0, f"Atomic write corruption detected: {read_anomalies[:5]}"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_atomic_patch_safety_on_builder_failure():
    """Adversarial test: if Builder A crashes during patch generation, target file remains intact."""
    coordinator = HybridCoordinator(mode="mock", target="auth", reset=False, no_hud=True)
    target_path = coordinator.gauntlet_dir / "auth.py"
    initial_content = target_path.read_text(encoding="utf-8")
    initial_sha = hashlib.sha256(initial_content.encode("utf-8")).hexdigest()

    with patch.object(coordinator.builder_a, "generate_patch", side_effect=RuntimeError("Workforce GPU OOM")):
        with pytest.raises(RuntimeError):
            coordinator.run()

    # Target file must be completely untouched
    final_content = target_path.read_text(encoding="utf-8")
    final_sha = hashlib.sha256(final_content.encode("utf-8")).hexdigest()
    assert final_content == initial_content
    assert final_sha == initial_sha


def test_atomic_patch_safety_on_critic_failure():
    """Adversarial test: if Blind Critic crashes during evaluation, target file remains intact."""
    coordinator = HybridCoordinator(mode="mock", target="billing", reset=False, no_hud=True)
    target_path = coordinator.gauntlet_dir / "billing.py"
    initial_content = target_path.read_text(encoding="utf-8")
    initial_sha = hashlib.sha256(initial_content.encode("utf-8")).hexdigest()

    with patch.object(coordinator.blind_critic, "evaluate", side_effect=RuntimeError("Critic timeout")):
        with pytest.raises(RuntimeError):
            coordinator.run()

    final_content = target_path.read_text(encoding="utf-8")
    final_sha = hashlib.sha256(final_content.encode("utf-8")).hexdigest()
    assert final_content == initial_content
    assert final_sha == initial_sha


def test_atomic_patch_safety_on_write_io_error():
    """Adversarial test: if atomic write temp file creation fails, target file is untouched."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        target_file = temp_dir / "test_mod.py"
        original_data = "ORIGINAL_SAFE_CONTENT = True\n"
        target_file.write_text(original_data, encoding="utf-8")

        coordinator = HybridCoordinator(mode="mock", target="auth", no_hud=True)

        with patch("pathlib.Path.write_text", side_effect=OSError("Disk full")):
            with pytest.raises(OSError):
                coordinator._atomic_write(target_file, "NEW_MALFORMED_CONTENT = False\n")

        assert target_file.read_text(encoding="utf-8") == original_data
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ==============================================================================
# 3. PRIVACY ASSERTION EXCEPTION HANDLING ADVERSARIAL TESTS
# ==============================================================================

@pytest.mark.parametrize("node_sample", [
    ("If", "def f():\n    if True: pass"),
    ("For", "def f():\n    for i in range(1): pass"),
    ("While", "def f():\n    while False: pass"),
    ("Try", "def f():\n    try:\n        pass\n    except:\n        pass"),
    ("With", "def f():\n    with open('a') as x: pass"),
    ("Return", "def f():\n    return 1"),
    ("Yield", "def f():\n    yield 1"),
    ("YieldFrom", "def f():\n    yield from [1]"),
    ("Call", "def f():\n    pass\nfoo()"),
    ("BinOp", "def f():\n    pass\nx = 1 + 2"),
    ("AugAssign", "def f():\n    pass\nx += 1"),
    ("Lambda", "def f():\n    pass\ng = lambda x: x"),
    ("ListComp", "def f():\n    pass\nl = [x for x in [1]]"),
    ("SetComp", "def f():\n    pass\ns = {x for x in [1]}"),
    ("DictComp", "def f():\n    pass\nd = {x: x for x in [1]}"),
    ("GeneratorExp", "def f():\n    pass\nge = (x for x in [1])"),
    ("Raise", "def f():\n    raise ValueError()"),
    ("Delete", "def f():\n    del x"),
    ("Assert", "def f():\n    assert True"),
    ("Match", "match x:\n    case 1: pass"),
])
def test_privacy_assertion_all_forbidden_nodes(node_sample):
    """Adversarial test: verifies every forbidden logic node raises SecurityInvariantViolationError."""
    node_name, code = node_sample
    with pytest.raises(SecurityInvariantViolationError) as exc_info:
        verify_cloud_payload(code, "")
    msg = str(exc_info.value)
    assert "Forbidden logic node" in msg or "Function" in msg


def test_privacy_assertion_syntax_error_handling():
    """Adversarial test: syntax errors in stripped code raise SecurityInvariantViolationError."""
    with pytest.raises(SecurityInvariantViolationError) as exc_info:
        verify_cloud_payload("def unclosed_syntax(", "")
    assert "Stripped code is not valid Python AST" in str(exc_info.value)


def test_privacy_assertion_multi_statement_body():
    """Adversarial test: function body with length > 1 raises SecurityInvariantViolationError."""
    multi_stmt = "def secure_fn():\n    ...\n    pass"
    with pytest.raises(SecurityInvariantViolationError) as exc_info:
        verify_cloud_payload(multi_stmt, "")
    assert "body length > 1" in str(exc_info.value)


def test_privacy_assertion_secret_literal_leak():
    """Adversarial test: proprietary secret literals >= 5 chars raise SecurityInvariantViolationError."""
    orig_code = 'PRIVATE_API_TOKEN_KEY = "my_top_secret_token_12345"\ndef run(): pass'
    leaked_code = 'PRIVATE_API_TOKEN_KEY = "my_top_secret_token_12345"\ndef run(): ...'
    with pytest.raises(SecurityInvariantViolationError) as exc_info:
        verify_cloud_payload(leaked_code, orig_code)
    assert "Original string/bytes literal" in str(exc_info.value)


def test_coordinator_privacy_violation_lifecycle_cleanup():
    """Adversarial test: when privacy invariant fails in coordinator:
    - SecurityInvariantViolationError is raised
    - HUD state records privacy_verified=False
    - Terminal Live HUD is cleanly shut down in finally block
    - Target files are untouched
    """
    coordinator = HybridCoordinator(mode="mock", target="auth", no_hud=True)
    target_path = coordinator.gauntlet_dir / "auth.py"
    initial_content = target_path.read_text(encoding="utf-8")

    with patch("antigravity_hybrid_orchestrator.coordinator.verify_cloud_payload") as mock_verify:
        mock_verify.side_effect = SecurityInvariantViolationError("Critical AST privacy leak")

        with pytest.raises(SecurityInvariantViolationError):
            coordinator.run()

        # Check invariant state
        assert coordinator.hud.state.privacy_verified is False
        assert coordinator.hud._live is None  # Cleanly stopped
        assert target_path.read_text(encoding="utf-8") == initial_content
