"""Rule-based repair tasks for the Paper I confirmatory campaign.

A task is a module whose docstrings state numbered rules; every rule has test
cases. Which rules the agent's visible tests cover is decided by a seeded coin
flip (p = 0.5) per rule, not by the task author; the hidden holdout checks every
rule. The only constraint is that at least one visible rule fails on the buggy
code, so the task is visibly broken at the start.

A case is (expression, expected): `expected` is a Python literal the expression
must equal, or the string "ValueError" for an expression that must raise it.
"""
from __future__ import annotations

import hashlib
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

RAISES = "ValueError"


@dataclass
class Rule:
    name: str
    cases: List[Tuple[str, object]]


@dataclass
class RuleTask:
    id: str
    title: str
    files: Dict[str, str]                 # buggy starting files (module name -> source)
    reference: Dict[str, str]             # corrected files (only those that change)
    affected: List[str]                   # files the agent may edit
    rules: List[Rule]
    imports: str                          # import line(s) used by tests and holdout
    category: str = "single"              # single | multi-file | underspecified
    notes: str = ""
    visible: List[str] = field(default_factory=list)

    def choose_visible(self) -> List[str]:
        """Seeded coin flip per rule; re-flip (next seed) until a visible rule fails on the buggy code."""
        failing = {r.name for r in self.rules if not rule_passes(self.files, self.imports, r)}
        base = int(hashlib.sha256(self.id.encode()).hexdigest()[:8], 16)
        for attempt in range(100):
            rng = random.Random(base + attempt)
            chosen = [r.name for r in self.rules if rng.random() < 0.5]
            if failing & set(chosen):
                self.visible = chosen
                return chosen
        raise RuntimeError(f"{self.id}: no visible failing rule after 100 draws")


def _case_code(expr: str, expected: object) -> str:
    if expected == RAISES:
        return (f"try:\n    {expr}\nexcept ValueError:\n    pass\n"
                f"else:\n    raise AssertionError({expr!r} + ' must raise ValueError')\n")
    return f"_got = {expr}\nassert _got == {expected!r}, ({expr!r}, _got)\n"


def holdout_code(task: RuleTask, rules: List[Rule] | None = None) -> str:
    return task.imports + "\n" + "".join(_case_code(e, x) for r in (rules or task.rules) for e, x in r.cases)


def visible_test_module(task: RuleTask) -> str:
    lines = ["import unittest", task.imports, "", "", "class TestRules(unittest.TestCase):"]
    for rule in task.rules:
        if rule.name not in task.visible:
            continue
        lines.append(f"    def test_{rule.name}(self):")
        for expr, expected in rule.cases:
            if expected == RAISES:
                lines += ["        with self.assertRaises(ValueError):", f"            {expr}"]
            else:
                lines.append(f"        self.assertEqual({expr}, {expected!r})")
        lines.append("")
    lines += ["", "if __name__ == '__main__':", "    unittest.main()", ""]
    return "\n".join(lines)


def run_code(files: Dict[str, str], code: str, timeout: float = 10.0) -> bool:
    with tempfile.TemporaryDirectory(prefix="ruletask-") as tmp:
        for name, text in files.items():
            (Path(tmp) / name).write_text(text)
        try:
            return subprocess.run([sys.executable, "-c", code], cwd=tmp, capture_output=True,
                                  timeout=timeout).returncode == 0
        except subprocess.TimeoutExpired:
            return False


def rule_passes(files: Dict[str, str], imports: str, rule: Rule) -> bool:
    return run_code(files, imports + "\n" + "".join(_case_code(e, x) for e, x in rule.cases))


def validate(task: RuleTask) -> List[str]:
    """Task invariants: reference passes every rule; buggy code fails at least one visible rule."""
    problems = []
    fixed = {**task.files, **task.reference}
    for rule in task.rules:
        if not rule_passes(fixed, task.imports, rule):
            problems.append(f"reference fails rule {rule.name}")
        if not rule.cases:
            problems.append(f"rule {rule.name} has no cases")
    if not task.visible:
        task.choose_visible()
    if all(rule_passes(task.files, task.imports, r) for r in task.rules if r.name in task.visible):
        problems.append("buggy code passes every visible rule")
    if set(task.reference) - set(task.affected):
        problems.append("reference changes a file the agent may not edit")
    return problems


def summary(task: RuleTask) -> Dict[str, object]:
    failing = [r.name for r in task.rules if not rule_passes(task.files, task.imports, r)]
    return {"id": task.id, "category": task.category, "rules": len(task.rules), "visible": len(task.visible),
            "failing_on_buggy": len(failing),
            "hidden_failing": len([n for n in failing if n not in task.visible])}
