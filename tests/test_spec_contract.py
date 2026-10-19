"""Docstring-contract checks: local self-check (SPECCHECK) and cloud contract tests (SPEC_TESTS)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from antigravity_hybrid_orchestrator.workforce.recovery_controller import RecoveryController

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from benchmark_gemma_orion_phase2b import setup_h6_workspace  # noqa: E402

CONTRACT = "def parse_duration(text: str) -> int\n    Units are h (3600 s), m (60 s) and s (1 s)."


def _passed_controller(spec_contract: str = CONTRACT) -> RecoveryController:
    rc = RecoveryController(subtask_id="s", input_files=["durations.py"], affected_files=["durations.py"],
                            verification_command="python3 -m unittest -q", policy="P1_STAGE_SPEC",
                            spec_contract=spec_contract, verify_fn=lambda: (True, "OK"))
    rc.record_tool_execution(turn=1, tool_name="read_file", args={"path": "durations.py"},
                             success=True, output="...", error=None)
    rc.record_tool_execution(turn=2, tool_name="patch_file", args={"path": "durations.py"},
                             success=True, output="Patched", error=None)
    return rc


def test_spec_policy_is_stage_hints_plus_self_check():
    rc = _passed_controller()
    assert rc.is_stage_hints and not rc.is_reactive and rc.config.spec_self_check


def test_first_pass_asks_for_contract_self_check_with_docstring():
    rc = _passed_controller()
    note = rc.record_tool_execution(turn=3, tool_name="run_command", args={"command": "python3 -m unittest -q"},
                                    success=True, output="OK", error=None)
    assert "Units are h (3600 s)" in note and "python3 -c" in note
    assert rc.spec_check_requested and not rc.spec_check_done


def test_contract_text_is_sent_once_then_a_short_reminder():
    rc = _passed_controller()
    rc.record_tool_execution(turn=3, tool_name="run_command", args={"command": "python3 -m unittest -q"},
                             success=True, output="OK", error=None)
    again = rc.record_tool_execution(turn=4, tool_name="read_file", args={"path": "durations.py"},
                                     success=True, output="...", error=None)
    assert "Units are h (3600 s)" not in again and "contract" in again.lower()


def test_self_check_command_completes_the_check():
    rc = _passed_controller()
    rc.record_tool_execution(turn=3, tool_name="run_command", args={"command": "python3 -m unittest -q"},
                             success=True, output="OK", error=None)
    note = rc.record_tool_execution(turn=4, tool_name="run_command",
                                    args={"command": "python3 -c \"from durations import parse_duration as p; assert p('1h')==3600\""},
                                    success=True, output="", error=None)
    assert rc.spec_check_done
    assert "PASSED" in note and "Reply" in note


def test_premature_stop_before_self_check_gets_one_nudge():
    rc = _passed_controller()
    rc.record_tool_execution(turn=3, tool_name="run_command", args={"command": "python3 -m unittest -q"},
                             success=True, output="OK", error=None)
    cont, prompt, err = rc.handle_non_tool_turn(turn=4, content="Done.", eval_tokens=10,
                                                is_truncated=False, active_num_predict=2048)
    assert cont and "contract" in prompt.lower() and err is None
    cont2, _, _ = rc.handle_non_tool_turn(turn=5, content="Done.", eval_tokens=10,
                                          is_truncated=False, active_num_predict=2048)
    assert cont2 is False


def test_plain_stage_policy_is_unchanged():
    rc = RecoveryController(subtask_id="s", affected_files=["d.py"], verification_command="python3 -m unittest -q",
                            policy="P1_STAGE", spec_contract=CONTRACT, verify_fn=lambda: (True, "OK"))
    rc.record_tool_execution(turn=1, tool_name="patch_file", args={"path": "d.py"}, success=True, output="", error=None)
    note = rc.record_tool_execution(turn=2, tool_name="run_command", args={"command": "python3 -m unittest -q"},
                                    success=True, output="OK", error=None)
    assert "Units are h" not in note and "PASSED" in note


def test_describe_contract_reads_signature_and_docstring(tmp_path):
    from antigravity_hybrid_orchestrator.privacy.assist_payload import describe_contract

    setup_h6_workspace(tmp_path)
    text = describe_contract(tmp_path, ["durations.py"])
    assert "def parse_duration(text: str) -> int" in text and "Units are h (3600 s)" in text
    assert "re.findall" not in text


# --- SPEC_TESTS: cloud-written contract tests -----------------------------------

GOOD_TESTS = '''import unittest
from durations import parse_duration


class TestContract(unittest.TestCase):
    def test_hours(self):
        self.assertEqual(parse_duration("1h"), 3600)

    def test_rejects_garbage(self):
        with self.assertRaises(ValueError):
            parse_duration("abc")
'''


def test_validate_contract_tests_accepts_plain_unittest():
    from antigravity_hybrid_orchestrator.engine.spec_tests import validate_contract_tests

    assert validate_contract_tests(GOOD_TESTS, ["durations"]) == []


@pytest.mark.parametrize("snippet, fragment", [
    ("import os\n", "import of 'os'"),
    ("from subprocess import run\n", "import of 'subprocess'"),
    ("x = open('f')\n", "call to 'open'"),
    ("eval('1')\n", "call to 'eval'"),
    ("__import__('os')\n", "call to '__import__'"),
    ("y = (1).__class__\n", "dunder attribute"),
])
def test_validate_contract_tests_rejects_unsafe_code(snippet, fragment):
    from antigravity_hybrid_orchestrator.engine.spec_tests import validate_contract_tests

    problems = validate_contract_tests(GOOD_TESTS + snippet, ["durations"])
    assert any(fragment in p for p in problems), problems


def test_validate_contract_tests_requires_a_testcase():
    from antigravity_hybrid_orchestrator.engine.spec_tests import validate_contract_tests

    assert any("TestCase" in p for p in validate_contract_tests("import unittest\n", ["durations"]))
    assert any("syntax" in p for p in validate_contract_tests("def (:\n", ["durations"]))


def test_writer_writes_validated_file_and_audits(tmp_path):
    from antigravity_hybrid_orchestrator.engine.spec_tests import SpecTestWriter

    subtask = setup_h6_workspace(tmp_path)
    prompts, events = [], []

    def complete(prompt):
        prompts.append(prompt)
        return "Here:\n```python\n" + GOOD_TESTS + "```", 100, 50

    writer = SpecTestWriter(complete, audit=lambda e, p: events.append((e, p)))
    source = writer.generate(tmp_path, subtask)
    assert source == GOOD_TESTS
    path = writer.install(tmp_path, subtask, source)
    assert path.name == "test_spec_contract.py" and path.read_text() == GOOD_TESTS
    assert "test_spec_contract.py" in subtask.input_files
    assert "Units are h (3600 s)" in prompts[0] and "re.findall" not in prompts[0]
    assert "Do not use any tools" in prompts[0]
    assert [e for e, _ in events] == ["spec_tests_request", "spec_tests_response"]


def test_writer_rejects_unsafe_reply_after_one_retry(tmp_path):
    from antigravity_hybrid_orchestrator.engine.spec_tests import SpecTestWriter

    subtask = setup_h6_workspace(tmp_path)
    replies = ["```python\nimport os\n" + GOOD_TESTS + "```", "```python\nimport os\n" + GOOD_TESTS + "```"]
    writer = SpecTestWriter(lambda p: (replies.pop(0), 1, 1))
    assert writer.generate(tmp_path, subtask) is None
