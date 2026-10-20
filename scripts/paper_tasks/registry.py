"""Paper I task set: freeze (manifest) and adapter to the benchmark's task tuples.

Freeze:  python3 scripts/paper_tasks/registry.py   -> validates every task, writes manifest.json
The benchmark reads the visible-rule choice from manifest.json and refuses a task whose
content changed after freezing.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import framework as F  # noqa: E402
import tasks_multi  # noqa: E402
import tasks_single  # noqa: E402
import tasks_under  # noqa: E402

ALL = tasks_single.TASKS + tasks_multi.TASKS + tasks_under.TASKS
MANIFEST = HERE / "manifest.json"


def digest(task: F.RuleTask) -> str:
    content = {"files": task.files, "reference": task.reference, "affected": task.affected, "imports": task.imports,
               "rules": [[r.name, [list(map(repr, c)) for c in r.cases]] for r in task.rules]}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def freeze() -> dict:
    manifest = {}
    for task in ALL:
        problems = F.validate(task)
        if problems:
            raise SystemExit(f"{task.id}: {problems}")
        manifest[task.id] = dict(F.summary(task), visible_rules=task.visible, sha256=digest(task))
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def benchmark_tasks(run_holdout, subtask_cls) -> list:
    """(task_id, setup_fn, holdout_fn) tuples in the benchmark's format, from the frozen manifest."""
    manifest = json.loads(MANIFEST.read_text())
    for task in ALL:
        entry = manifest[task.id]
        if entry["sha256"] != digest(task):
            raise RuntimeError(f"{task.id} changed after freezing; re-run registry.py and re-register")
        task.visible = entry["visible_rules"]
    return adapt(ALL, run_holdout, subtask_cls)


def adapt(tasks, run_holdout, subtask_cls) -> list:
    """Benchmark tuples for RuleTasks whose visible rules are already fixed."""
    out = []
    for task in tasks:
        test_file = f"test_{Path(task.affected[0]).stem}.py"

        def setup(ws: Path, task=task, test_file=test_file):
            for name, text in task.files.items():
                (Path(ws) / name).write_text(text, encoding="utf-8")
            (Path(ws) / test_file).write_text(F.visible_test_module(task), encoding="utf-8")
            listed = ", ".join(f"`{name}`" for name in [*task.files, test_file])
            return subtask_cls(
                id=task.id, title=task.title,
                description=(f"Inspect {listed}. Fix {', '.join(f'`{a}`' for a in task.affected)} so the code "
                             "satisfies its docstring specification and passes `python3 -m unittest -q`."),
                input_files=[*task.files, test_file], affected_files=list(task.affected),
                verification_command="python3 -m unittest -q")

        def holdout(ws: Path, task=task) -> bool:
            return run_holdout(ws, F.holdout_code(task))

        out.append((task.id, setup, holdout))
    return out


if __name__ == "__main__":
    for task_id, entry in freeze().items():
        print(task_id, {k: entry[k] for k in ("category", "rules", "visible", "failing_on_buggy", "hidden_failing")})
