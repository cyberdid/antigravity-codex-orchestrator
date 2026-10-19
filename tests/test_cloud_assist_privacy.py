"""Spec-level cloud assist payload: what may leave the machine, and what blocks it."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from antigravity_hybrid_orchestrator.planner.dag import SubTask
from antigravity_hybrid_orchestrator.privacy.verifier import SecurityInvariantViolationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from benchmark_gemma_orion_phase2b import setup_h6_workspace  # noqa: E402

EXCERPT = "FAIL: test_hours_and_minutes (test_durations.TestParseDuration) | AssertionError: 90 != 5400"


def _payload(tmp_path: Path, subtask: SubTask | None = None) -> str:
    from antigravity_hybrid_orchestrator.privacy.assist_payload import build_assist_payload

    subtask = subtask or setup_h6_workspace(tmp_path)
    return build_assist_payload(tmp_path, subtask, EXCERPT)


def test_payload_contains_signature_docstring_and_failure(tmp_path):
    payload = _payload(tmp_path)
    assert "def parse_duration(text: str) -> int" in payload
    assert "Units are h (3600 s)" in payload
    assert EXCERPT in payload


def test_payload_excludes_function_and_test_bodies(tmp_path):
    payload = _payload(tmp_path)
    for leaked in ("re.findall", "UNITS = {", "total +=", "assertEqual", "class TestParseDuration"):
        assert leaked not in payload


def test_default_values_are_not_sent(tmp_path):
    (tmp_path / "fdr.py").write_text(
        'def bh(p_values, alpha: float = 0.05123):\n    """Step-up procedure."""\n    return 1\n'
    )
    payload = _payload(tmp_path, SubTask(id="s", title="t", affected_files=["fdr.py"]))
    assert "def bh(p_values, alpha: float=...)" in payload
    assert "0.05123" not in payload


def test_missing_or_invalid_affected_file_is_skipped(tmp_path):
    (tmp_path / "broken.py").write_text("def f(:\n")
    payload = _payload(tmp_path, SubTask(id="s", title="t", affected_files=["nope.py", "broken.py"]))
    assert "SUBTASK" in payload and EXCERPT in payload


def test_file_outside_workspace_is_skipped(tmp_path):
    outside = tmp_path.parent / "outside_secret.py"
    outside.write_text('def leak():\n    """OUTSIDE_DOCSTRING"""\n')
    workspace = tmp_path / "ws"
    workspace.mkdir()
    from antigravity_hybrid_orchestrator.privacy.assist_payload import build_assist_payload

    payload = build_assist_payload(
        workspace, SubTask(id="s", title="t", affected_files=["../outside_secret.py"]), EXCERPT
    )
    assert "OUTSIDE_DOCSTRING" not in payload


def test_planted_secret_blocks_payload(tmp_path):
    from antigravity_hybrid_orchestrator.privacy.assist_payload import verify_assist_payload

    (tmp_path / "m.py").write_text(
        'def f():\n    """Token ghp_' + "a1B2c3D4e5" * 3 + 'xyzWVU"""\n'
    )
    payload = _payload(tmp_path, SubTask(id="s", title="t", affected_files=["m.py"]))
    with pytest.raises(SecurityInvariantViolationError):
        verify_assist_payload(payload)


def test_high_entropy_string_blocks_payload_but_hex_digest_does_not():
    from antigravity_hybrid_orchestrator.privacy.assist_payload import verify_assist_payload

    verify_assist_payload("commit 9ae29e2622858a2f044b5635120fd91016fec9f4 failed")
    with pytest.raises(SecurityInvariantViolationError):
        verify_assist_payload("key=Zq8+Lr2/Wm5xKp9Tn3Hs7Vb1Jd6Fg4Yc0QeRuIoP")


def test_cap_is_measured_in_utf8_bytes():
    from antigravity_hybrid_orchestrator.privacy.assist_payload import verify_assist_payload

    verify_assist_payload("ж" * 3000)
    with pytest.raises(SecurityInvariantViolationError):
        verify_assist_payload("ж" * 5000)
