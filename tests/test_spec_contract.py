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


def test_contract_includes_dataclass_fields_without_defaults(tmp_path):
    from antigravity_hybrid_orchestrator.privacy.assist_payload import describe_contract

    (tmp_path / "types_mod.py").write_text(
        "from dataclasses import dataclass\nfrom typing import List\n\n"
        "@dataclass\nclass Decision:\n    \"\"\"Outcome.\"\"\"\n    rejected_mask: List[bool]\n"
        "    critical_threshold: float = 0.05123\n"
    )
    text = describe_contract(tmp_path, ["types_mod.py"])
    assert "class Decision" in text
    assert "    rejected_mask: List[bool]" in text and "    critical_threshold: float" in text
    assert "0.05123" not in text


def test_writer_contract_covers_input_type_modules(tmp_path):
    from antigravity_hybrid_orchestrator.engine.spec_tests import SpecTestWriter
    from benchmark_gemma_orion_phase2b import setup_h1_workspace

    subtask = setup_h1_workspace(tmp_path)
    prompts = []
    source = ("import unittest\nfrom fdr_control import benjamini_hochberg\nfrom fdr_types import FdrDecision\n\n"
              "class T(unittest.TestCase):\n    def test_empty(self):\n"
              "        self.assertEqual(benjamini_hochberg([]).num_rejected, 0)\n")
    writer = SpecTestWriter(lambda p: (prompts.append(p) or ("```python\n" + source + "```", 1, 1)))
    assert writer.generate(tmp_path, subtask) == source
    assert "rejected_mask: List[bool]" in prompts[0]
    assert "test_fdr_control" not in prompts[0]


# --- benchmark independent checks ------------------------------------------------

def test_independent_checks_ignore_generated_tests_and_survive_hangs(tmp_path):
    from benchmark_gemma_orion_phase2b import _independent_checks

    (tmp_path / "mod.py").write_text("def f(x):\n    if x == 'hang':\n        while True:\n            pass\n    return x\n")
    (tmp_path / "test_mod.py").write_text(
        "import unittest\nfrom mod import f\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n"
        "        self.assertEqual(f(1), 1)\n")
    (tmp_path / "test_spec_contract.py").write_text(
        "import unittest\nfrom mod import f\n\nclass C(unittest.TestCase):\n    def test_hang(self):\n"
        "        f('hang')\n")

    def holdout(ws):
        import subprocess, sys
        return subprocess.run([sys.executable, "-c", "from mod import f; f('hang')"], cwd=ws, timeout=1).returncode == 0

    visible, held, output = _independent_checks(tmp_path, ["test_mod"], holdout, timeout=5.0)
    assert visible is True
    assert held is False and "timed out" in output


def test_resume_prefills_completed_runs_and_contract_tests(tmp_path):
    import json as _json
    import benchmark_gemma_orion_phase2b as bench

    partial = tmp_path / "x.partial.json"
    partial.write_text(_json.dumps({"complete": False, "git_commit": "abc", "runs": [
        {"policy_id": "P1_STAGE_ADAPTIVE", "seed": 303, "task_id": "H1_fdr_step_up_fix", "metadata": {}},
        {"policy_id": "P1_STAGE_ADAPTIVE_SPECTESTS", "seed": 303, "task_id": "H2_pareto_judge_fix",
         "metadata": {"spec_tests_source": "SRC", "spec_tests_tokens": 7, "spec_tests_events": []}},
        {"policy_id": "OTHER", "seed": 303, "task_id": "H1_fdr_step_up_fix", "metadata": {}},
    ]}))
    bench.SPEC_TEST_CACHE.clear()
    done, prefill = bench._load_resume(partial, ["P1_STAGE_ADAPTIVE", "P1_STAGE_ADAPTIVE_SPECTESTS"], [303, 404])
    assert done == {("P1_STAGE_ADAPTIVE", 303, "H1_fdr_step_up_fix"),
                    ("P1_STAGE_ADAPTIVE_SPECTESTS", 303, "H2_pareto_judge_fix")}
    assert len(prefill["P1_STAGE_ADAPTIVE"][303]) == 1 and prefill["P1_STAGE_ADAPTIVE"][404] == []
    assert bench.SPEC_TEST_CACHE["H2_pareto_judge_fix", "full"]["source"] == "SRC"


# --- disclosure ablation and checklist control --------------------------------------

TYPES_MODULE = (
    "from dataclasses import dataclass\n\n"
    "@dataclass\nclass Decision:\n    \"\"\"Outcome of a test.\"\"\"\n    rejected: bool\n\n"
    "def decide(p: float) -> Decision:\n    \"\"\"Reject when p < 0.05.\"\"\"\n    return Decision(p < 0.05)\n"
)


@pytest.mark.parametrize("level, docs, fields", [("signatures", False, False), ("docstrings", True, False), ("full", True, True)])
def test_describe_contract_disclosure_levels(tmp_path, level, docs, fields):
    from antigravity_hybrid_orchestrator.privacy.assist_payload import describe_contract

    (tmp_path / "mod.py").write_text(TYPES_MODULE)
    text = describe_contract(tmp_path, ["mod.py"], level=level)
    assert "class Decision" in text and "def decide(p: float) -> Decision" in text
    assert ("Reject when p < 0.05." in text) is docs
    assert ("    rejected: bool" in text) is fields
    assert "return Decision" not in text


def test_describe_contract_rejects_unknown_level(tmp_path):
    from antigravity_hybrid_orchestrator.privacy.assist_payload import describe_contract

    with pytest.raises(ValueError):
        describe_contract(tmp_path, [], level="bodies")


def test_writer_uses_requested_disclosure(tmp_path):
    from antigravity_hybrid_orchestrator.engine.spec_tests import SpecTestWriter

    subtask = setup_h6_workspace(tmp_path)
    prompts = []
    writer = SpecTestWriter(lambda p: (prompts.append(p) or ("```python\n" + GOOD_TESTS + "```", 1, 1)),
                            disclosure="signatures")
    assert writer.generate(tmp_path, subtask) == GOOD_TESTS
    assert "def parse_duration" in prompts[0] and "Units are h (3600 s)" not in prompts[0]


CHECKLIST_REPLY = "Rules:\n1. \"1h\" returns 3600.\n2. Units must appear in the order h, m, s.\n3) An empty string raises ValueError.\n"


def test_checklist_writer_returns_numbered_rules_and_installs_them(tmp_path):
    from antigravity_hybrid_orchestrator.engine.spec_tests import SpecChecklistWriter

    subtask = setup_h6_workspace(tmp_path)
    prompts, events = [], []
    writer = SpecChecklistWriter(lambda p: (prompts.append(p) or (CHECKLIST_REPLY, 10, 5)),
                                 audit=lambda e, p: events.append(e))
    checklist = writer.generate(tmp_path, subtask)
    assert checklist.splitlines() == ['1. "1h" returns 3600.', "2. Units must appear in the order h, m, s.",
                                      "3) An empty string raises ValueError."]
    assert writer.tokens_used == 15
    assert "Units are h (3600 s)" in prompts[0] and "Do not use any tools" in prompts[0]
    assert "re.findall" not in prompts[0]
    assert events == ["spec_checklist_request", "spec_checklist_response"]
    before = subtask.description
    SpecChecklistWriter.install(subtask, checklist)
    assert subtask.description.startswith(before) and checklist in subtask.description


def test_checklist_writer_rejects_code_and_empty_replies(tmp_path):
    from antigravity_hybrid_orchestrator.engine.spec_tests import SpecChecklistWriter

    subtask = setup_h6_workspace(tmp_path)
    replies = ["```python\nprint(1)\n```\n1. x", "no numbered rules here"]
    assert SpecChecklistWriter(lambda p: (replies.pop(0), 1, 1)).generate(tmp_path, subtask) is None


def test_benchmark_registers_control_and_ablation_arms():
    import benchmark_gemma_orion_phase2b as bench

    assert bench.POLICIES["P1_STAGE_ADAPTIVE_CHECKLIST"].get("spec_checklist") is True
    assert bench.POLICIES["P1_STAGE_ADAPTIVE_SPECTESTS_SIG"]["disclosure"] == "signatures"
    assert bench.POLICIES["P1_STAGE_ADAPTIVE_SPECTESTS_DOC"]["disclosure"] == "docstrings"
    assert bench.POLICIES["P1_STAGE_ADAPTIVE_SPECTESTS"].get("disclosure", "full") == "full"


def test_install_spec_tests_caches_per_disclosure_level(tmp_path):
    import benchmark_gemma_orion_phase2b as bench

    calls = []

    class Cloud:
        def complete(self, prompt):
            calls.append(prompt)
            return "```python\n" + GOOD_TESTS + "```", 1, 1

    bench.SPEC_TEST_CACHE.clear()
    for level in ("full", "signatures", "full"):
        ws = tmp_path / f"ws{len(calls)}_{level}"
        ws.mkdir()
        meta = bench._install_spec_tests("H6_x", setup_h6_workspace, ws, setup_h6_workspace(ws), Cloud(), level)
        assert meta["spec_tests_disclosure"] == level and meta["spec_tests_wall_s"] is not None
    assert len(calls) == 2
