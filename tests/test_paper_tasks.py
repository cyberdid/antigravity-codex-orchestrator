"""Paper I task set: frozen manifest and task invariants through the benchmark's own setup."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "paper_tasks"))

import benchmark_gemma_orion_phase2b as bench  # noqa: E402
import registry  # noqa: E402

PAPER = {tid: (setup, hold) for tid, setup, hold in bench.TASKS if tid.startswith("P")}
BY_ID = {task.id: task for task in registry.ALL}


def test_manifest_matches_task_content():
    import json

    manifest = json.loads(registry.MANIFEST.read_text())
    assert set(manifest) == set(BY_ID) and len(manifest) == 25
    for task in registry.ALL:
        assert manifest[task.id]["sha256"] == registry.digest(task)


@pytest.mark.parametrize("task_id", sorted(PAPER))
def test_reference_passes_visible_and_holdout_buggy_fails(tmp_path, task_id):
    setup, hold = PAPER[task_id]
    buggy = tmp_path / "buggy"
    fixed = tmp_path / "fixed"
    buggy.mkdir()
    fixed.mkdir()
    setup(buggy)
    setup(fixed)
    for name, text in BY_ID[task_id].reference.items():
        (fixed / name).write_text(text)
    run = lambda ws: subprocess.run([sys.executable, "-m", "unittest", "-q"], cwd=ws, capture_output=True).returncode == 0
    assert run(fixed) and hold(fixed)
    assert not run(buggy) and not hold(buggy)


def test_natural_skeleton_keeps_interface_and_drops_bodies():
    import natural

    task = BY_ID["P9_order_total"]
    target = task.affected[0]
    skel = natural.skeleton({**task.files, **task.reference}[target])
    assert "def order_total(items: list, discount_percent: int=0) -> int:" in skel
    assert "rounded down to whole cents" in skel and "subtotal * discount_percent" not in skel
    prompt = natural.build_prompt(task, target)
    assert "class Item" in prompt and "Read-only module `models.py`" in prompt


def test_natural_task_uses_generated_file_and_reference_fix(tmp_path):
    import framework as F
    import natural
    import registry as reg

    spec = BY_ID["P6_chunk_indices"]
    buggy = "def chunk_indices(n, size):\n    return [(s, s + size) for s in range(0, n, size)]\n"
    ntask = natural.natural_task(spec, 6, buggy, "chunks.py")
    ntask.choose_visible()
    assert ntask.id == "N6_chunk_indices" and ntask.files["chunks.py"] == buggy
    assert not F.validate(ntask)
    (tid, setup, hold), = reg.adapt([ntask], bench._run_holdout, bench.SubTask)
    setup(tmp_path)
    assert (tmp_path / "chunks.py").read_text() == buggy and not hold(tmp_path)


def test_restore_spec_puts_back_docstrings_and_rejects_incomplete_code():
    import ast
    import natural

    original = '"""Mod."""\n\n\ndef f(x):\n    """Return x doubled."""\n    return 2 * x\n\n\ndef g():\n    """G."""\n    return 1\n'
    generated = 'def f(x):\n    """changed"""\n    return x + x + 1\n\n\ndef g():\n    return 1\n'
    restored = natural.restore_spec(generated, original)
    tree = ast.parse(restored)
    assert ast.get_docstring(tree) == "Mod."
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert ast.get_docstring(funcs["f"]) == "Return x doubled." and "x + x + 1" in restored
    assert ast.get_docstring(funcs["g"]) == "G."
    assert natural.restore_spec("def f(x):\n    return x\n", original) is None
    assert natural.restore_spec("def (:", original) is None
